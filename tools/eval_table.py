"""Read the tournament logs a checkpoint sweep left behind and print one comparable table.

    python tools/eval_table.py ppo-selfplay2 ppo-bignet ppo-cover

WHY IT READS LOGS RATHER THAN BEING PART OF THE RUNNER. The runs take an hour and get interrupted;
the numbers have to be recoverable from what is on disk afterwards, including for an arm that only
half finished. A summary that exists only in the runner's memory is one Ctrl-C from being gone.

IT REPORTS THE INTERVAL, ALWAYS. 16 matches is +/-21 points at 95%, which is wide enough that two
checkpoints a dozen points apart are the same checkpoint as far as this data is concerned. Every
earlier round of this project produced a ranking off single-digit match counts and then argued
about the order; the interval is printed next to the number so that argument cannot start.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

MATCH = re.compile(r"match (\d+): warrior on seat (\d+), won=(\d), turns=(\d+)")
# The summary's own warrior row: seats, wins, win%, stars, kills, deaths.
WARRIOR = re.compile(r"^\s*warrior\s+(\d+)\s+(\d+)\s+(\d+)%\s+(\d+)\s+([\d.]+)\s+([\d.]+)", re.M)
CLASSIC = re.compile(r"^\s*classic\s+(\d+)\s+(\d+)\s+(\d+)%\s+(\d+)\s+([\d.]+)\s+([\d.]+)", re.M)


def read(tag, mp):
    """One arm: wins, matches, and the per-seat averages, or None if it never ran."""
    path = os.path.join(DATA, "%s-%s.log" % (tag, mp))
    if not os.path.exists(path):
        return None
    text = open(path, encoding="utf-8", errors="replace").read()
    rows = MATCH.findall(text)
    out = {"matches": len(rows), "wins": sum(1 for r in rows if r[2] == "1"),
           "turns": (sum(int(r[3]) for r in rows) / len(rows)) if rows else 0.0}
    w = WARRIOR.search(text)
    c = CLASSIC.search(text)
    if w:
        out.update(stars=int(w.group(4)), kills=float(w.group(5)), deaths=float(w.group(6)))
    if c:
        # Per-SEAT stars for the built-in AI: its row aggregates three seats a match, so the raw
        # number is three times the warrior's by construction and comparing them directly would
        # read as a rout in the wrong direction.
        out.update(cpu_stars=int(c.group(4)) / max(1, int(c.group(1))) * max(1, len(rows)))
    return out


def ci(wins, n):
    """95% normal-approximation half-width, in points. Crude, and that is the point: it is here to
    stop a 12-point gap on 16 matches from being read as an ordering."""
    if not n:
        return 0.0
    p = wins / n
    return 196.0 * (p * (1 - p) / n) ** 0.5


def main(names, maps=("Arboretum", "Islands", "Crossroads")):
    print()
    print("  %-16s %s" % ("", "  ".join("%-14s" % m for m in maps)))
    print("  %-16s %s" % ("checkpoint", "  ".join("%-14s" % "win% (n)" for _ in maps)))
    print("  " + "-" * (16 + 16 * len(maps)))
    for name in names:
        tag = name if name.startswith("ck-") else "ck-" + name
        cells, total_w, total_n = [], 0, 0
        for mp in maps:
            r = read(tag, mp)
            if not r or not r["matches"]:
                cells.append("%-14s" % "--")
                continue
            total_w += r["wins"]
            total_n += r["matches"]
            cells.append("%-14s" % ("%d%% (%d/%d)" % (
                round(100 * r["wins"] / r["matches"]), r["wins"], r["matches"])))
        overall = ""
        if total_n:
            overall = "   all %d%% +/-%d" % (round(100 * total_w / total_n),
                                             round(ci(total_w, total_n)))
        print("  %-16s %s%s" % (name.replace("ck-", ""), "  ".join(cells), overall))

    print()
    print("  a fair share for one seat of four is 25%")
    print()
    print("  %-16s %-11s %-7s %-7s %-7s %-7s" % ("", "map", "stars", "cpu*", "kills", "deaths"))
    for name in names:
        tag = name if name.startswith("ck-") else "ck-" + name
        for mp in maps:
            r = read(tag, mp)
            if not r or "stars" not in r:
                continue
            print("  %-16s %-11s %-7d %-7d %-7.1f %-7.1f"
                  % (name.replace("ck-", ""), mp, r["stars"], r.get("cpu_stars", 0),
                     r.get("kills", 0), r.get("deaths", 0)))
    print()
    print("  cpu* is the built-in AI's stars per seat, not per match -- its row in the tournament")
    print("  summary aggregates three seats and reads as a rout if compared raw.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]] or ["ppo-selfplay2", "ppo-bignet", "ppo-cover"]
    main(args)
