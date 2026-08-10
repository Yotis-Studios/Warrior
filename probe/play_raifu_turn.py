"""Put a REAL Raifu Wars turn in front of a sidecar and show what comes back.

The payloads come out of a live match: run the game with RW_WARRIOR_DUMP=1 and every turn logs
the exact body of a POST /v1/act. This replays one at a sidecar, so what the model does here is
what it would have done in the match.

    python probe/play_raifu_turn.py acts.jsonl --turn 40
    python probe/play_raifu_turn.py acts.jsonl --all --limit 20

The point of `--all` is the number at the end: how many of a real match's turns produced a legal
action. That is the one measurement worth having before wiring the seat up, because an enumerator
that is subtly incomplete looks exactly like a model that plays badly.
"""

import argparse
import json
import sys
import time
import urllib.request

# Card titles are real game strings and some are not cp1252-encodable, which is what a Windows
# console defaults to -- printing one raises UnicodeEncodeError and takes the whole run with it.
# Replace rather than fail: a mangled character in a card name is cosmetic, losing the run is not.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def post(url, payload, timeout=180):
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/act", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def show_turn(payload, resp, dt, verbose=True):
    st = payload["state"]
    me = st["self"]
    acts = payload["available_actions"]
    legal = {a["action_id"] for a in acts}
    chosen = resp.get("action_id")
    ok = chosen in legal

    if verbose:
        print("=" * 72)
        print(f"TURN {st['turn']}  seat {payload['seat']}  {me['name']}")
        print("=" * 72)
        print(st["board"]["ascii"])
        tp = me["tier_progress"]
        print(f"  tile {me['tile']}  health {me['health']}/{me['max_health']}  "
              f"ammo {me['ammo']}/{me['max_ammo']}  stars {me['stars']}  tier {me['tier']}")
        print(f"  tier progress: {tp['have']}/{tp['need']} {tp['condition']} -> tier {tp['next_tier']}")
        print(f"  this turn: " + ", ".join(f"{k}={v}" for k, v in me["this_turn"].items()))
        if me["hand"]:
            print("  hand: " + ", ".join(f"{c['name']} ({c['cost_stars']}*)" for c in me["hand"]))
        print(f"  in friendly territory: {me['in_friendly_territory']}")

        # The redaction, shown rather than asserted -- opponents are a count, never a list.
        print("  opponents:")
        for p in st["players"]:
            if p["is_self"]:
                continue
            print(f"    seat {p['seat']} {p['name']:<22} hp {p['health']} tier {p['tier']} "
                  f"stars {p['stars']:<5} {p['cards_in_hand']} cards in hand  [{p['status']}]")

        by_type = {}
        for a in acts:
            by_type.setdefault(a["type"], []).append(a["action_id"])
        print(f"  {len(acts)} legal actions: "
              + ", ".join(f"{t} x{len(v)}" for t, v in sorted(by_type.items())))
        for a in acts:
            if a["type"] == "attack":
                print(f"    {a['action_id']}: {a['label']}  "
                      f"hit {int(a.get('hit_chance', 0) * 100)}%")

        mark = "LEGAL" if ok else "*** ILLEGAL ***"
        print(f"\n  -> MODEL CHOSE: {chosen!r}   [{mark}]  ({dt:.1f}s)")
        label = next((a["label"] for a in acts if a["action_id"] == chosen), None)
        if label:
            print(f"     {label}")
        if resp.get("args"):
            print(f"     args: {resp['args']}")
        print()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acts", help="jsonl of POST /v1/act bodies from RW_WARRIOR_DUMP=1")
    ap.add_argument("--url", default="http://127.0.0.1:8879")
    ap.add_argument("--turn", type=int, default=None, help="index into the file")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    payloads = []
    with open(args.acts, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                payloads.append(json.loads(line))
    print(f"{len(payloads)} turns in {args.acts}\n")

    if not args.all:
        idx = args.turn if args.turn is not None else len(payloads) // 2
        p = payloads[idx]
        t0 = time.time()
        r = post(args.url, p)
        show_turn(p, r, time.time() - t0)
        return 0

    legal = 0
    tried = 0
    lat = []
    for p in payloads[:args.limit]:
        t0 = time.time()
        try:
            r = post(args.url, p)
        except Exception as exc:  # noqa: BLE001
            print(f"  turn {p['state']['turn']} seat {p['seat']}: TRANSPORT {exc}")
            tried += 1
            continue
        dt = time.time() - t0
        lat.append(dt)
        tried += 1
        ok = show_turn(p, r, dt, verbose=False)
        legal += 1 if ok else 0
        acts = {a["action_id"] for a in p["available_actions"]}
        mark = "ok " if ok else "BAD"
        print(f"  [{mark}] turn {p['state']['turn']:>3} seat {p['seat']}  "
              f"{len(acts):>3} actions  ->  {r.get('action_id'):<18} {dt:5.1f}s")

    print(f"\n{legal}/{tried} turns produced a LEGAL action"
          + (f", {sum(lat) / len(lat):.1f}s average" if lat else ""))
    return 0 if legal == tried else 1


if __name__ == "__main__":
    sys.exit(main())
