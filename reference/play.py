"""Run a full Skirmish match and report. The protocol's end-to-end proof.

    # No model needed -- four scripted seats. This is the CI gate.
    python reference/play.py --seats 4 --scripted random

    # One warrior against three scripted opponents.
    python reference/play.py --warrior 0=http://127.0.0.1:8879

    # Warriors on every seat.
    python reference/play.py --warrior 0=http://127.0.0.1:8879 --warrior 1=http://127.0.0.1:8880

Exit code is 0 for a decided match, 1 for anything that stopped it finishing. That makes it usable
as a regression gate: "do four seats of legal actions still complete a match" catches the worst
failure a turn-based engine has, which is a state with no legal way out of it.
"""

import argparse
import sys
import time

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import ScriptedSeat, WarriorSeat  # noqa: E402
from game import Skirmish  # noqa: E402

# A turn is several actions, and a runaway seat must not spin forever. This is a SAFETY NET, not a
# rule: a seat that hits it is reported, because silently truncating turns would make the match
# look healthy while it was not.
MAX_ACTIONS_PER_TURN = 12
MAX_TURNS = 60


def run_match(game, seats, verbose=True, log=print):
    for s in seats.values():
        s.notify("/v1/match/start", {
            "protocol_version": "0.1", "match_id": game.match_id, "seat": s.seat,
            "seats": len(seats), "rules": {"max_turns": MAX_TURNS},
        })

    started = time.time()
    turn_index = 0
    capped_turns = 0

    while game.winner is None and game.turn < MAX_TURNS:
        seat = game.current
        agent = seats[seat]
        unit = game.unit(seat)
        if not unit.alive:
            if game.end_turn():
                break
            continue

        game.begin_turn()
        if verbose:
            log(f"\n-- turn {game.turn}, seat {seat} ({unit.name}, {unit.hp}hp, "
                f"{unit.ammo}ammo, roll {game.move_roll}) [{agent.name}]")

        last_action = None
        reason = "turn_start"
        for step in range(MAX_ACTIONS_PER_TURN):
            turn_index += 1
            action_id, args = agent.choose(
                game, f"{game.match_id}-t{game.turn}-s{seat}-a{step}", reason, last_action)

            # None means the seat was never asked, because nothing was legal. Ending the turn is
            # the only thing left, and it is the CLIENT ending it rather than the agent failing to.
            if action_id is None:
                if verbose:
                    log("     (no legal action -- turn ends)")
                break

            result, over = game.apply(action_id, args)
            if verbose:
                extra = f"  \"{args['message']}\"" if args.get("message") else ""
                log(f"     {action_id:<22} {result}{extra}")

            if over:
                break
            last_action = {"action_id": action_id, "ok": True, "result": result}
            reason = "action_result"

            if not game.living() or len(game.living()) <= 1:
                break
        else:
            capped_turns += 1
            if verbose:
                log(f"     ! seat {seat} hit the {MAX_ACTIONS_PER_TURN}-action cap")

        if game.end_turn():
            break

    elapsed = time.time() - started
    for s in seats.values():
        s.notify("/v1/match/end", {
            "protocol_version": "0.1", "match_id": game.match_id, "seat": s.seat,
            "result": {"winner": game.winner},
        })

    return {
        "winner": game.winner,
        "turns": game.turn,
        "actions": game.action_count,
        "elapsed_s": round(elapsed, 1),
        "capped_turns": capped_turns,
        "decided": game.winner is not None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seats", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--warrior", action="append", default=[],
                    metavar="SEAT=URL", help="seat a warrior, e.g. 0=http://127.0.0.1:8879")
    ap.add_argument("--scripted", default="random", choices=["random", "aggressive"],
                    help="policy for seats with no warrior")
    ap.add_argument("--deadline-ms", type=int, default=60000)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    game = Skirmish(seats=args.seats, seed=args.seed)

    warriors = {}
    for spec in args.warrior:
        if "=" not in spec:
            raise SystemExit(f"--warrior wants SEAT=URL, got {spec!r}")
        seat_s, url = spec.split("=", 1)
        warriors[int(seat_s)] = url

    seats = {}
    for s in range(args.seats):
        if s in warriors:
            seats[s] = WarriorSeat(s, warriors[s], deadline_ms=args.deadline_ms)
        else:
            seats[s] = ScriptedSeat(s, policy=args.scripted)

    print(f"=== Skirmish {game.match_id}: {args.seats} seats ===")
    for s in sorted(seats):
        try:
            seats[s].handshake()
        except ConnectionError as exc:
            print(f"  ! {exc}")
            return 1
        print(f"  seat {s}: {seats[s].name}")

    result = run_match(game, seats, verbose=not args.quiet)

    print("\n=== result ===")
    if result["winner"] is None:
        print(f"  UNDECIDED after {result['turns']} turns "
              f"-- no seat won inside the turn cap")
    else:
        print(f"  winner: seat {result['winner']} ({game.unit(result['winner']).name}) "
              f"[{seats[result['winner']].name}]")
    print(f"  turns {result['turns']}, actions {result['actions']}, {result['elapsed_s']}s"
          + (f", {result['capped_turns']} turns hit the action cap"
             if result["capped_turns"] else ""))

    print("\n=== seats ===")
    unhealthy = []
    for s in sorted(seats):
        st = seats[s].stats
        d = st.as_dict()
        flag = "" if st.healthy() else "   <-- NOT PLAYING CLEANLY"
        if not st.healthy():
            unhealthy.append(s)
        print(f"  seat {s} {seats[s].name:<28} actions {d['actions']:>3}  "
              f"rejected {d['rejected']:>3}  retries {d['retries']:>3}  "
              f"forfeits {d['forfeits']:>2}  {d['avg_latency_s']:>5.2f}s{flag}")

    if unhealthy:
        # Loud, because a warrior whose ids are rejected a quarter of the time is not a warrior
        # that played badly -- it is one that barely played, and a win column will not say so.
        print(f"\n  seats {unhealthy} had a high rejection or forfeit rate. Their result is not"
              f"\n  a measurement of how well they play.")

    return 0 if result["decided"] else 1


if __name__ == "__main__":
    sys.exit(main())
