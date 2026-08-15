"""Warrior-SFT: teach a model to read the protocol, use the advice, and say why.

    python tools/make_warrior_sft.py ../raifuwars-rl/runs/speval-*.jsonl \
        --expert ../raifuwars-rl/runs/ppo-selfplay/last.pt --out data/warrior-sft

ONE ROW IS ONE DECISION: system prompt, the rendered position with the RL policy's probability
mass inline on every action, and a single `take_action` call carrying the action and a
justification. No consult round trip -- the advice is already in the prompt.

WHAT IT TEACHES, AND WHAT IT CANNOT. Every target is the expert's own choice, so a model trained
on this is a WRAPPER: it should approach the expert and will not exceed it. That is the point of a
first dataset and it is also its ceiling. Deciding when the expert is WRONG cannot be learned here,
because there is no example of a good override in it -- measured, not assumed: across 2,872
decisions, in all 22 positions where the rules prove what the right answer is (a tier-up that ends
the match), the expert already picks it.

THE JUSTIFICATIONS ARE DERIVED, NEVER INVENTED. They are built from the same structured fields the
features are built from -- does this destination sit on a point, is this the best hit chance
offered, does this tier-up end the match -- and not from a language model asked to explain a move
it did not make. A generated rationalisation is a plausible sentence with no causal link to the
choice, and training on those teaches a model to produce confident reasoning that does not track
what it is doing. The cost is that they read like templates, because they are.

PROMPTS COME FROM THE LIVE RENDERER. `_render_state` and `_render_actions` are imported from the
sidecar rather than reimplemented here, so a row cannot drift from what the model is served at
inference. A model trained on a layout it is never served is a model trained on nothing.
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
from policy import SYSTEM_PROMPT, LLMPolicy                        # noqa: E402


def cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def point_at(state, xy):
    """The capture point covering a tile, or None. A point is a SQUARE of half-width `radius`."""
    for p in ((state.get("board") or {}).get("points") or []):
        t = p.get("tile") or [0, 0]
        r = p.get("radius", p.get("stars_per_turn", 1)) or 1
        if cheb([float(t[0]), float(t[1])], xy) <= float(r):
            return p
    return None


def parse_xy(action_id, prefix):
    try:
        xs, ys = action_id[len(prefix):].split("_", 1)
        return [float(xs), float(ys)]
    except Exception:                                               # noqa: BLE001
        return None


def justify(state, actions, chosen, share):
    """One line on why this action, from structured fields only."""
    me = state.get("self") or {}
    act = next((a for a in actions if str(a.get("action_id")) == chosen), {})
    kind = act.get("type") or ""
    conf = "the recommendation puts %d%% here" % round(100 * share) if share >= 0.005 else None

    if kind == "tier_up":
        tier = float(me.get("tier") or 0)
        return ("tiering here takes me to 4 and ends the match" if tier >= 3
                else "I am in my own base and can tier, so I take the grade")

    if kind == "tier_choice":
        return ("taking the KO track" if chosen.endswith("kills")
                else "taking the stars track")

    if kind == "attack":
        hc = float(act.get("hit_chance") or 0)
        best = max((float(a.get("hit_chance") or 0) for a in actions
                    if a.get("type") == "attack"), default=0)
        by_seat = {int(float(p.get("seat", -1))): p for p in (state.get("players") or [])}
        tgt = by_seat.get(int(float(act.get("target_seat", -1)))) or {}
        hp = tgt.get("health")
        lead = ("the best shot on offer at %d%%" % round(100 * hc) if hc >= best
                else "a %d%% shot" % round(100 * hc))
        return lead + (" on a target at %s health" % LLMPolicy._num(hp) if hp is not None else "")

    if kind == "move":
        xy = parse_xy(chosen, "move_")
        if xy:
            p = point_at(state, xy)
            if p is not None and not p.get("is_base"):
                rel = p.get("relationship")
                if rel == "unclaimed":
                    return "this takes the unclaimed point at (%d,%d)" % (xy[0], xy[1])
                if rel == "enemy":
                    return "this contests the enemy point at (%d,%d)" % (xy[0], xy[1])
                return "this holds my point at (%d,%d)" % (xy[0], xy[1])
            if p is not None and p.get("is_base"):
                return "moving onto my base, where I can tier"
            # No point involved: say what the move does to the distance that matters, using the
            # game's own precomputed note rather than recomputing geometry here.
            note = (act.get("note") or "").strip()
            if note:
                return note[:110]
        return conf or "repositioning"

    if kind == "rush":
        return "spending the combat action on a second move instead of shooting"
    if kind == "reload":
        ammo, mx = float(me.get("ammo") or 0), float(me.get("max_ammo") or 5)
        return ("at full ammo, so this is armour against the next hit" if ammo >= mx
                else "topping the magazine back up")
    if kind == "play_card":
        return "playing %s" % (act.get("label") or "a card").replace("Play ", "")
    if kind == "discard_card":
        return "discarding %s -- nothing to spend it on" % \
               (act.get("label") or "a card").replace("Discard ", "")
    if kind == "end_turn":
        return conf or "nothing better on offer this turn"
    return conf or "taking the recommended action"


def _is_end(row):
    """Does this row target an end_turn? Read from the emitted call, not a side table."""
    try:
        args = json.loads(row["messages"][-1]["tool_calls"][0]["function"]["arguments"])
        return str(args.get("action_id")) in ("end", "end_turn")
    except Exception:                                           # noqa: BLE001
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--expert", required=True)
    ap.add_argument("--out", default="data/warrior-sft")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--end-turn-frac", type=float, default=0.15,
                    help="cap end_turn at this share of rows; 0 to keep all")
    args = ap.parse_args()

    from expert import Expert                                       # noqa: E402
    expert = Expert(args.expert)
    policy = LLMPolicy.__new__(LLMPolicy)
    policy.expert = expert

    files = [f for pat in args.traces for f in glob.glob(pat)]
    rows, kinds, followed, seen = [], collections.Counter(), 0, set()

    for f in sorted(files):
        for line in open(f, encoding="utf-8", errors="replace"):
            try:
                d = json.loads(line, strict=False)
            except Exception:                                       # noqa: BLE001
                continue
            if d.get("source") != "warrior":
                continue
            actions = d.get("available_actions") or []
            chosen = str(d.get("chosen_action") or "")
            if len(actions) < 2 or not chosen:
                continue
            if not any(str(a.get("action_id")) == chosen for a in actions):
                continue                                            # rejected, never applied

            # DEDUPED ON THE POSITION, not the row. The same board recurs across seeds and the
            # duplicates are exact, so leaving them in would weight whatever happens to repeat.
            state = d.get("state") or {}
            key = (d.get("map"), d.get("match_id"), d.get("turn"),
                   chosen, len(actions))
            if key in seen:
                continue
            seen.add(key)

            ranked = expert.rank(state, actions, k=len(actions))
            if not ranked:
                continue
            advice = {r["action_id"]: r["share"] for r in ranked}
            if ranked[0]["action_id"] == chosen:
                followed += 1

            user = policy._render_state(state) + "\n\n" + policy._render_actions(actions, advice)
            rows.append({
                "key": (d.get("map"), d.get("match_id")),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "call_act", "type": "function",
                        "function": {"name": "take_action", "arguments": json.dumps({
                            "action_id": chosen,
                            "justification": justify(state, actions, chosen,
                                                     advice.get(chosen, 0.0)),
                        })}}]},
                ],
                "tools": policy._tools(actions),
            })
            kinds[next((a.get("type") for a in actions
                        if str(a.get("action_id")) == chosen), "?")] += 1
            if args.max_rows and len(rows) >= args.max_rows:
                break

    if not rows:
        print("no usable rows")
        return 1

    # END_TURN IS CAPPED, NOT DROPPED. It is a third of the raw rows and the least informative
    # third: 542 of 869 in the first pass justified themselves with nothing but "the recommendation
    # puts 100% here", because ending a turn genuinely has no structured reason behind it. Left at
    # natural frequency, the single most common thing this dataset teaches is "end the turn and
    # cite the advice".
    #
    # Capped rather than removed because knowing WHEN a turn is over is a real skill -- a model
    # that never ends its turn stalls the match, which is worse than one that ends it early. The
    # cap keeps the decision represented without letting it dominate.
    if args.end_turn_frac > 0:
        ends = [i for i, r in enumerate(rows) if _is_end(r)]
        budget = int(len(rows) * args.end_turn_frac)
        if len(ends) > budget:
            rng0 = random.Random(args.seed)
            drop = set(rng0.sample(ends, len(ends) - budget))
            kept = [r for i, r in enumerate(rows) if i not in drop]
            print("end_turn capped: %d -> %d rows (%.0f%% of %d)"
                  % (len(ends), budget, 100 * args.end_turn_frac, len(kept)))
            rows = kept

    # SPLIT BY MATCH, NEVER BY ROW. Decisions inside one match share a board, a roster and a map,
    # so a random row split puts near-identical positions on both sides and reports memorisation
    # as validation.
    keys = sorted({r["key"] for r in rows})
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    val_keys = set(keys[:max(1, int(len(keys) * args.val_frac))])

    os.makedirs(args.out, exist_ok=True)
    n_tr = n_va = 0
    with open(os.path.join(args.out, "train.jsonl"), "w", encoding="utf-8", newline="\n") as tr, \
            open(os.path.join(args.out, "val.jsonl"), "w", encoding="utf-8", newline="\n") as va:
        for r in rows:
            k = r.pop("key")
            fh = va if k in val_keys else tr
            fh.write(json.dumps(r) + "\n")
            n_va, n_tr = (n_va + 1, n_tr) if k in val_keys else (n_va, n_tr + 1)

    print("rows: %d train, %d val, from %d matches" % (n_tr, n_va, len(keys)))
    print("action mix: %s" % dict(kinds.most_common()))
    print("target == the recommendation's top choice: %.1f%%" % (100 * followed / len(rows)))
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
