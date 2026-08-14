"""Teach a model to ASK the expert and then act on the answer.

    python tools/make_consult_dataset.py runs/simppoeval-*.jsonl \
        --expert ../raifuwars-rl/runs/ppo-sim/best.pt --out data/consult

WHY THIS EXISTS. Fine-tuning on the built-in AI's decisions is exhausted. Two attempts, the second
with six defects fixed, seven maps and 280x the tier-up examples:

    4B v1   74.4% held-out agreement   ->   5.0% win rate
    4B v2   91.5% held-out agreement   ->   2.6% win rate

Agreement rose 17 points and wins did not move. Per-decision imitation of a 28% heuristic does not
compose into winning a 90-turn match, and more of it will not.

Meanwhile a 57,730-parameter RL policy wins 55.1%. So the useful thing to teach a language model is
not how to play -- it is how to USE something that already can. This dataset teaches exactly one
behaviour: call `consult_expert`, read what comes back, then act.

WHAT THE TARGET IS, AND WHAT THAT COSTS. Every row's answer is the expert's top choice, so a model
trained on this becomes a WRAPPER: it will follow ~always and should land near the expert's own
55%, up from 2.6%. That is the point of a first step and it is also its ceiling. Knowing when to
OVERRIDE -- on cards the expert never saw, or because a team-mate asked for something -- cannot be
taught from these rows, because there is no example of a good override in them. That is a separate
dataset and it needs a source of good overrides, which does not exist yet.

THE SOURCE IS THE EXPERT'S OWN TOURNAMENT. `simppoeval-*.jsonl` is the 55.1% policy playing 160
real matches. It served greedily, so its recorded `chosen_action` IS the expert's argmax -- the
target and the tool result cannot disagree, by construction, without either being fabricated.

PROMPTS COME FROM THE LIVE RENDERER. `_render_state` and `_render_actions` are imported from the
sidecar rather than reimplemented, so a row cannot drift from what the model is served at
inference. Same rule as make_dataset.py, for the same reason.
"""

import argparse
import collections
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "sidecar"))
from expert import Expert                                          # noqa: E402
from policy import SYSTEM_PROMPT, LLMPolicy                        # noqa: E402


def rows_from(paths, limit=0):
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # ONLY the warrior's own decisions. The file also carries every built-in AI
                # decision in the same matches, and training on those would be teaching the
                # 28% heuristic again -- the exact thing this dataset exists to stop doing.
                if d.get("source") != "warrior":
                    continue
                yield d
                if limit:
                    limit -= 1
                    if limit <= 0:
                        return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--expert", required=True)
    ap.add_argument("--rl-path", default=None)
    ap.add_argument("--out", default="data/consult")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    paths = []
    for p in args.traces:
        paths.extend(sorted(glob.glob(p)) or [p])

    expert = Expert(args.expert, rl_path=args.rl_path)
    policy = LLMPolicy(expert=expert)

    out_rows = []
    seen = set()
    kinds = collections.Counter()
    disagreed = 0
    matches = collections.Counter()

    for d in rows_from(paths, args.limit):
        state = d.get("state") or {}
        actions = d.get("available_actions") or []
        chosen = str(d.get("chosen_action", ""))
        if len(actions) < 2 or not chosen:
            continue
        if chosen not in {str(a.get("action_id")) for a in actions}:
            continue

        ranked = expert.rank(state, actions, k=args.top_k)
        # The trace was produced by this same policy served greedily, so its choice should be the
        # expert's argmax. When it is not, the checkpoint has moved since the tournament or the
        # row is from a different policy -- either way the tool result would be showing one thing
        # and the target doing another, which is exactly the incoherence this teaches against.
        if ranked[0]["action_id"] != chosen:
            disagreed += 1
            continue

        user = policy._render_state(state) + "\n\n" + policy._render_actions(actions)
        user += "\n\nTake one action now."

        key = (user, chosen)
        if key in seen:
            continue
        seen.add(key)

        act = next(a for a in actions if str(a.get("action_id")) == chosen)
        kinds[act.get("type") or "?"] += 1
        matches[(d.get("map"), d.get("match_id"))] += 1

        out_rows.append({
            "key": (d.get("map"), d.get("match_id")),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                # THE CONSULT. A fixed id per row: it only has to be unique within the
                # conversation, and a stable one keeps the rows diffable.
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_consult", "type": "function",
                    "function": {"name": "consult_expert", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call_consult", "name": "consult_expert",
                 "content": expert.render(ranked)},
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_act", "type": "function",
                    "function": {"name": "take_action",
                                 "arguments": json.dumps({"action_id": chosen})}}]},
            ],
            "tools": policy._tools(actions),
        })

    if not out_rows:
        print("no usable rows")
        return 1

    # SPLIT BY MATCH, NEVER BY ROW. Decisions inside one match share a board, a roster and a map,
    # so a random row split puts near-identical positions on both sides and reports memorisation
    # as validation.
    keys = sorted({r["key"] for r in out_rows})
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    val_keys = set(keys[:max(1, int(len(keys) * args.val_frac))])

    os.makedirs(args.out, exist_ok=True)
    n_train = n_val = 0
    with open(os.path.join(args.out, "train.jsonl"), "w", encoding="utf-8", newline="\n") as tr, \
            open(os.path.join(args.out, "val.jsonl"), "w", encoding="utf-8", newline="\n") as va:
        for r in out_rows:
            key = r.pop("key")
            fh = va if key in val_keys else tr
            fh.write(json.dumps(r) + "\n")
            if key in val_keys:
                n_val += 1
            else:
                n_train += 1

    total = n_train + n_val
    print("=== consult dataset ===")
    print("  rows           : %d  (%d train / %d val)" % (total, n_train, n_val))
    print("  matches        : %d  (%d val -- SPLIT BY MATCH)" % (len(keys), len(val_keys)))
    print("  dropped, dupes : %d" % (len(seen) - total if len(seen) > total else 0))
    print("  dropped, expert disagreed with the trace : %d" % disagreed)
    print("\n  action mix in the target:")
    for k, n in kinds.most_common():
        print("    %-14s %6d  %5.1f%%" % (k, n, 100.0 * n / total))
    print("\n  every row is: consult_expert -> tool result -> take_action(expert's top choice).")
    print("  a model trained on this becomes a WRAPPER. That is the intent and the ceiling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
