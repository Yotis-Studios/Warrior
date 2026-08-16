#!/usr/bin/env bash
# Play every named checkpoint on every map, one at a time, and print one table.
#
#   tools/eval-checkpoints.sh ppo-selfplay2 ppo-bignet ppo-cover
#   MATCHES=16 tools/eval-checkpoints.sh ppo-cover
#
# SEQUENTIAL ON PURPOSE. preflight refuses to launch while another tournament driver is alive,
# because a driver between maps holds no lock, passes the check, and then takes the next map's
# lock out from under the run you just started. Two of three arms exited within a second of each
# other, twice, that way. So arms do not overlap, and each gets a fresh sidecar of its own -- which
# is also required rather than tidy: the feature encoder's width is frozen at first import, so a
# checkpoint trained with terrain features on cannot share a process with one trained without.
set -uo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
RUNS=${RW_RUNS:-$W/../raifuwars-rl/runs}
MATCHES=${MATCHES:-16}
MAPS=${MAPS:-"Arboretum Islands Crossroads"}
CKPT=${CKPT:-last.pt}
PORT=${PORT:-8996}

[ $# -gt 0 ] || { echo "usage: $0 <run-name> [run-name ...]"; exit 2; }

for name in "$@"; do
  path="$RUNS/$name/$CKPT"
  [ -f "$path" ] || { echo "!! no checkpoint at $path -- skipping $name"; continue; }
  echo
  echo "######## $name  ($(date +%H:%M:%S))"
  TAG="ck-$name" EXPERT="$path" MATCHES="$MATCHES" MAPS="$MAPS" \
    POLICY=hybrid PORT="$PORT" bash "$W/tools/run-seat.sh"
done

echo
echo "######## table"
python "$W/tools/eval_table.py" "$@"
