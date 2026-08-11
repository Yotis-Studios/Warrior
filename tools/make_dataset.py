"""Turn game traces into chat-template training rows.

    python tools/make_dataset.py <traces.jsonl>... --out data/sft

Produces train.jsonl / val.jsonl in OpenAI chat format: a system turn, a user turn holding the
rendered board and legal actions, and an assistant turn that is a `take_action` TOOL CALL. That is
the shape the sidecar actually elicits at inference, so a model trained on it is being trained on
its own job rather than on a paraphrase of it.

WHY THIS IS WORTH DOING AT ALL. Measured over 80 matches: told in plain English what the built-in
AI would do, the model complied 13% of the time on `fight` and 0 times out of 1258 on `rush`. The
hint is a suggestion and it can be declined. A trained target cannot.

FIVE DECISIONS, each of which changes what the model learns.

1. THE PROMPT IS BUILT BY THE SIDECAR, NOT BY THIS FILE. It imports LLMPolicy and calls the same
   _render_state / _render_actions the live path calls. Reimplementing the rendering here would
   create a second copy that drifts, and the failure is silent and total: the model trains on one
   distribution and is served another, and nothing reports it. This codebase has the same rule for
   the rulebook, for lang_fonts(), for packet ids -- one implementation, referenced not copied.

2. HINTS ARE STRIPPED. Traces from a hint-level-1 run carry "the built-in AI would fight" in the
   payload. Training on that teaches the model to read a field that will not be there, and worse,
   to depend on it. The point of the fine-tune is to put the built-in AI's judgement INSIDE the
   weights, not to teach the model to look it up.

3. `last_action` IS OMITTED, and this is a known and stated mismatch. The live path appends "your
   previous action resolved: ..." mid-turn; the trace does not record it. Rather than invent one,
   it is left out and written down here. Inventing plausible context is how a dataset acquires
   facts the game never produced.

4. DEDUPLICATED BY CONTENT. The hint-level-0 and hint-level-1 tournaments ran the SAME SEEDS, so
   their CPU decisions overlap heavily -- the boards only diverge once the warrior seat perturbs
   them. Concatenating the two files without this would silently weight early-turn positions twice
   and call it more data.

5. SPLIT BY MATCH, NEVER BY ROW. Decisions within one match share a board, a roster and a map, so
   a random row split puts near-identical positions on both sides and reports a validation score
   that is really a memorisation score. The split is on match_id.
"""

import argparse
import collections
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sidecar.policy import SYSTEM_PROMPT, LLMPolicy          # noqa: E402


def rows_from(paths, sources):
    pol = LLMPolicy()          # never called over the network; only its renderers are used
    seen = set()
    dupes = 0
    skipped_trivial = 0
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
                if d.get("source") not in sources:
                    continue
                actions = d.get("available_actions") or []
                chosen = str(d.get("chosen_action", "")).strip()
                if not chosen or len(actions) < 2:
                    # Nothing was decided, so there is nothing to imitate.
                    skipped_trivial += 1
                    continue
                ids = {a.get("action_id") for a in actions}
                if chosen not in ids:
                    # The enumerator and the policy disagreed. check-action-coverage.py exists to
                    # find these; a dataset is not the place to paper over one.
                    skipped_trivial += 1
                    continue

                user = pol._render_state(d.get("state") or {})
                user += "\n\n" + pol._render_actions(actions)
                user += "\n\nTake one action now."

                key = hashlib.sha1((user + "\x00" + chosen).encode("utf-8")).hexdigest()
                if key in seen:
                    dupes += 1
                    continue
                seen.add(key)

                args = {"action_id": chosen}
                why = str(d.get("why", "") or "").strip()
                if why:
                    args["why"] = why

                yield d.get("match_id", "?"), {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": None, "tool_calls": [{
                            "type": "function",
                            "function": {"name": "take_action",
                                         "arguments": json.dumps(args)},
                        }]},
                    ],
                    "tools": pol._tools(actions),
                }, chosen
    rows_from.dupes = dupes
    rows_from.skipped = skipped_trivial


def family(a):
    for p, n in (("move_", "move"), ("atk_s", "attack"), ("card_use_", "play card"),
                 ("card_drop_", "discard"), ("pick_", "select tile")):
        if a.startswith(p):
            return n
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--source", default="cpu",
                    help="comma-separated: cpu, warrior, human (default cpu)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    sources = set(s.strip() for s in args.source.split(",") if s.strip())
    collected = list(rows_from(args.traces, sources))
    if not collected:
        print("no usable rows. Check --source and that the traces contain decisions.")
        return 1

    by_match = collections.defaultdict(list)
    fams = collections.Counter()
    for match_id, row, chosen in collected:
        by_match[match_id].append(row)
        fams[family(chosen)] += 1

    matches = sorted(by_match)
    random.Random(args.seed).shuffle(matches)
    n_val = max(1, int(len(matches) * args.val_frac))
    val_matches = set(matches[:n_val])

    os.makedirs(args.out, exist_ok=True)
    paths = {"train": os.path.join(args.out, "train.jsonl"),
             "val": os.path.join(args.out, "val.jsonl")}
    counts = {"train": 0, "val": 0}
    with open(paths["train"], "w", encoding="utf-8") as ftr, \
            open(paths["val"], "w", encoding="utf-8") as fva:
        for m in matches:
            fh = fva if m in val_matches else ftr
            k = "val" if m in val_matches else "train"
            for row in by_match[m]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[k] += 1

    total = counts["train"] + counts["val"]
    print("=== dataset ===")
    print("  sources kept   : %s" % ", ".join(sorted(sources)))
    print("  rows           : %d  (%d train / %d val)" % (total, counts["train"], counts["val"]))
    print("  matches        : %d  (%d train / %d val -- SPLIT BY MATCH, not by row)"
          % (len(matches), len(matches) - len(val_matches), len(val_matches)))
    print("  dropped, dupes : %d  (same rendered prompt AND same action)"
          % getattr(rows_from, "dupes", 0))
    print("  dropped, other : %d  (fewer than 2 legal actions, or a choice not in the list)"
          % getattr(rows_from, "skipped", 0))
    print("\n  action mix in the training target:")
    for f, c in fams.most_common():
        print("    %-14s %6d  %5.1f%%" % (f, c, 100.0 * c / total))
    print("\n  hints are stripped and last_action is omitted -- see the module docstring.")
    print("  wrote %s and %s" % (paths["train"], paths["val"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
