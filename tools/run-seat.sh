#!/usr/bin/env bash
# Play matches with one policy, on one or more maps. The reproducible entry point for an eval.
#
#   tools/run-seat.sh                                   # hybrid seat, RL plays, no chat
#   MODEL=google/gemini-3.5-flash-lite tools/run-seat.sh # ...and it talks
#   POLICY=llm MODEL=... MAPS="Arboretum" tools/run-seat.sh
#
# THIS LIVES IN THE REPO ON PURPOSE. Every eval in this project so far was launched from an
# ad-hoc script in a scratch directory, and the consequences were not small: two of three map arms
# silently refused by leftover locks while the third ran on and read as a full table, a run
# measured against a game binary that predated the change under test, and three overlapping runs
# writing the same result files. A runner nobody can re-run is a result nobody can check.
#
# It gates on tools/preflight.sh, which refuses to start on a stale build, a stale lock, or another
# driver still alive.
set -uo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
RW=${RW_REPO:-$W/../RaifuWars}

POLICY=${POLICY:-hybrid}
EXPERT=${EXPERT:-$W/../raifuwars-rl/runs/ppo-selfplay/last.pt}
MODEL=${MODEL:-}
MATCHES=${MATCHES:-8}
MAPS=${MAPS:-"Arboretum Islands Crossroads"}
SKILL=${SKILL:-1.0}
TAG=${TAG:-seat}
PORT=${PORT:-8996}

export RW_AI_MATCH_FRAMES=${RW_AI_MATCH_FRAMES:-6000000}
export RW_WARRIOR_DEADLINE_MS=${RW_WARRIOR_DEADLINE_MS:-60000}
RUNNER=$(cygpath -w "${RW_RUNNER:-/c/ProgramData/GameMakerStudio2-LTS2026/Cache/runtimes/runtime-2026.0.0.23/windows/x64/Runner.exe}")
GAME=$(cygpath -w "${RW_GAME_WIN:-/c/Users/Nicholas/Documents/GameMakerStudio2/raifuwars-rl/runs/game/RaifuWars.win}")
export RW_LAUNCH="$RUNNER -game $GAME"

bash "$RW/tools/preflight.sh" || { echo "preflight failed -- not launching"; exit 1; }

cd "$W"
ARGS=(--policy "$POLICY" --port "$PORT")
# hybrid: the net plays and the model only talks, so it wants a small token budget and a
# warm temperature -- the opposite of a seat that has to emit a legal action_id.
[ "$POLICY" = "hybrid" ] && ARGS+=(--expert "$EXPERT" --skill "$SKILL" --max-tokens "${MAX_TOKENS:-120}" --temperature "${TEMPERATURE:-0.9}")
# MAX_TOKENS IS PER MODEL, not a constant. 4000 is right for a hosted frontier model and wrong for
# a small local one: with prompts at 1,600-2,700 tokens, asking a 4B for 4,000 more overruns its
# context, and instead of an error you get a 30-second call that returns no tool call at all and
# falls back. Measured: 16 fallbacks in the first minutes of a run, and 1.0s median once it was 200.
if [ "$POLICY" = "llm" ]; then
  ARGS+=(--max-tokens "${MAX_TOKENS:-4000}" --temperature "${TEMPERATURE:-0.7}")
  [ -n "${EXPERT:-}" ] && ARGS+=(--expert "$EXPERT")
fi
# A LOCAL ENDPOINT, not OpenRouter. URL= points the llm policy at an OpenAI-compatible server on
# this machine -- a fine-tune under llama.cpp, say -- with no API key and no per-call cost.
if [ -n "${URL:-}" ]; then
  ARGS+=(--url "$URL")
fi
if [ -n "$MODEL" ]; then
  export OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-$(tr -d ' \r\n' < /c/Users/Nicholas/Projects/openrouter.txt)}
  ARGS+=(--openrouter "$MODEL")
fi

# THE PORT MUST BE EMPTY BEFORE WE START, and this is not paranoia -- it silently destroyed an
# arm of a sweep. HTTPServer sets SO_REUSEADDR, so on Windows a second sidecar binds a port the
# first still owns, prints a normal startup banner, and then receives nothing: the OLD process goes
# on answering with the OLD checkpoint. The sweep's second arm came back byte-identical to its
# first -- same wins, same stars, same kills, match for match -- and nothing anywhere errored.
for _ in $(seq 1 30); do
  curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1 || break
  echo "  waiting for port $PORT to be released..."
  sleep 2
done
if curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1; then
  echo "!! something is still serving port $PORT. Refusing to start -- a new sidecar would bind"
  echo "   it anyway and the OLD one would answer. Kill it first:"
  curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/health"; echo
  exit 1
fi

python sidecar/server.py "${ARGS[@]}" > "$W/data/$TAG-sc.log" 2>&1 &
SC=$!
# The sidecar loads torch and a checkpoint before it binds. Poll rather than sleep a guessed
# number: a fixed sleep is either wasted time or a race, and this one used to be both.
for _ in $(seq 1 60); do
  curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1 && break
  sleep 1
done

# AND THE THING ANSWERING MUST BE SERVING OUR CHECKPOINT. Every run writes `last.pt`, so the
# sidecar's own log line ("expert loaded: last.pt") named nothing; health now reports the run
# directory too. Checked rather than assumed, because the failure mode this catches produces a
# complete, plausible, entirely wrong results table.
if [ "$POLICY" = "hybrid" ]; then
  WANT="$(basename "$(dirname "$EXPERT")")/$(basename "$EXPERT")"
  GOT=$(curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/health" \
        | sed -n 's/.*"expert": *"\([^"]*\)".*/\1/p')
  if [ "$GOT" != "$WANT" ]; then
    echo "!! sidecar on $PORT is serving '$GOT', not '$WANT'. Aborting."
    tail -5 "$W/data/$TAG-sc.log"
    kill $SC 2>/dev/null
    exit 1
  fi
  echo "  sidecar serving $GOT"
fi

cd "$RW"
SCRATCH=${LOCALAPPDATA:-$HOME}/Temp/raifuwars-compile
for MAP in $MAPS; do
  # RE-CHECK PER MAP, not once at launch. preflight cannot see a driver that takes a lock AFTER
  # this run starts, and the failure it misses is the expensive one: the arm exits inside a second
  # and its empty log reads as a map that simply went badly. Islands lost three runs this way.
  #
  # Only STALE locks are fatal -- one held by a live pid means a legitimate concurrent run, which
  # preflight already refuses to start alongside.
  for f in "$SCRATCH"/warrior-tourney-*.lock; do
    [ -e "$f" ] || continue
    pid=$(tr -dc '0-9' < "$f" 2>/dev/null)
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
      echo "!! stale lock $(basename "$f") (pid ${pid:-?} gone) -- $MAP would exit instantly. Stopping."
      kill $SC 2>/dev/null
      exit 1
    fi
  done
  echo "===== $MAP $(date +%H:%M:%S) ====="
  RW_TOURNEY_ID="$TAG-$MAP" RW_MAP="$MAP" RW_TRACE_OUT="$W/data/$TAG-tr-$MAP.jsonl" \
    tools/warrior-tournament.sh "$MATCHES" "http://127.0.0.1:$PORT" 2>&1 \
    | tee "$W/data/$TAG-$MAP.log" | grep -a --line-buffered "match \|chance for"
done

# STOP IT, AND WAIT UNTIL IT HAS ACTUALLY STOPPED ANSWERING. `kill` returns immediately and the
# sidecar keeps a thread per open connection, so returning here while it still holds the port is
# what let the next arm bind alongside it. Escalate to the Windows pid, because $! is an MSYS pid
# and taskkill does not speak that; the sidecar prints its own os.getpid() to health.
WPID=$(curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/health" \
       | sed -n 's/.*"pid": *\([0-9]*\).*/\1/p')
kill $SC 2>/dev/null
for _ in $(seq 1 15); do
  curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1 || break
  sleep 1
done
if curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1; then
  echo "  sidecar ignored kill; taskkill //PID $WPID"
  [ -n "$WPID" ] && taskkill //F //PID "$WPID" >/dev/null 2>&1
  for _ in $(seq 1 10); do
    curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1 || break
    sleep 1
  done
fi
if curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1; then
  echo "!! port $PORT is STILL served after kill and taskkill. Later arms would measure this one."
  exit 1
fi
echo "SEAT RUN DONE $(date +%H:%M:%S)"
