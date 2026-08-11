#!/usr/bin/env bash
# Screen several models against the held-out split, cheaply, and print one table.
#
#   OPENROUTER_API_KEY=... tools/screen_models.sh <val.jsonl> [n] [model ...]
#
# WHY A SCREEN AND NOT A TOURNAMENT. A 40-match tournament is ~2,000 model calls; this is `n` per
# model, default 20. Nothing here decides anything -- it answers "is this model worth a tournament",
# and the two columns that answer it are not the score.
#
# READ THE FAILURE COLUMNS FIRST. Measured while writing this: one provider rejects
# tool_choice=required outright (every request 400s), and one model given the weaker `auto`
# constraint answered in prose on 76% of decisions. Both look like a score of zero and neither is
# a statement about how well the model plays. A model that cannot reliably emit a tool call cannot
# play through this protocol at all, whatever its agreement number would have been.
#
# The score itself is only meaningful against the always-move baseline printed beside it, which
# varies with the sample -- so compare a model to ITS OWN baseline line, never to another run's.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VAL="${1:?usage: screen_models.sh <val.jsonl> [n] [model ...]}"
N="${2:-20}"
shift 2 2>/dev/null || shift 1
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(openai/gpt-oss-120b google/gemini-2.5-flash-lite mistralai/mistral-small-3.2-24b-instruct z-ai/glm-4.7-flash)
fi

: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"

printf '%-46s %8s %9s %8s %8s\n' "model" "top-1" "baseline" "no-call" "failed"
for m in "${MODELS[@]}"; do
  out=$(timeout 900 python tools/eval_sft.py "$VAL" --url https://openrouter.ai/api \
        --model "$m" --n "$N" 2>/dev/null)
  top=$(echo "$out" | grep -a 'top-1 agreement' | grep -oE '[0-9]+\.[0-9]%' | head -1)
  base=$(echo "$out" | grep -a 'always-' | grep -oE '[0-9]+\.[0-9]%' | head -1)
  nc=$(echo "$out" | grep -a 'no tool call' | grep -oE '[0-9]+\.[0-9]%' | head -1)
  fail=$(echo "$out" | grep -a 'request FAILED' | grep -oE '[0-9]+\.[0-9]%' | head -1)
  printf '%-46s %8s %9s %8s %8s\n' "$m" "${top:-?}" "${base:-?}" "${nc:-?}" "${fail:-?}"
done
