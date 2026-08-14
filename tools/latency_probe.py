"""How fast is each model at ONE Warrior decision, and does it answer correctly at all?

    python tools/latency_probe.py --n 5
    python tools/latency_probe.py --n 5 --consult    # two calls, as an advisor arm does

WHY THIS EXISTS. Latency was discovered from inside tournaments, three times, each time looking
like something else:

  - a model slower than the game's 30s protocol deadline reports `status -1 http 200` -- a reply
    that arrived too late -- so the turn is ended and the match finishes in seconds with nobody
    having played. It presents as a broken sidecar.
  - the AI trial's match timeout is a FRAME budget, so a slow model has its matches truncated --
    and the ones that survive are the short decisive ones, which biases every win rate computed
    from them.
  - a model taking 12s per decision turns a 20-match arm into a seven-hour job, discovered after
    committing the box to it.

All three are answerable in a minute per model, before spending hours. This sends a REAL rendered
payload -- the same prompt the sidecar builds, from a real recorded position -- and reports the
distribution, the tool-call rate, and what a full match and a 20-match arm would cost.

WHAT IT CHECKS BEYOND SPEED. A model that answers in 2s and cannot emit a tool call is worse than
one that takes 20s and can. Three of ten models screened earlier could not emit a tool call at all,
which no latency number would have revealed.
"""

import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "sidecar"))
from policy import SYSTEM_PROMPT, LLMPolicy                        # noqa: E402

URL = "https://openrouter.ai/api/v1/chat/completions"

# A spread across vendors and tiers, all verified tool-capable against /api/v1/models -- guessing
# IDs produces a 400 that reads like a model failure rather than a typo. Cheap flash models and
# heavyweight reasoners answer this task very differently, and the reasoners break the deadline.
DEFAULT_MODELS = [
    "google/gemini-3.7-flash",
    "google/gemini-3.6-flash",
    "google/gemini-3.5-flash-lite",
    "google/gemini-3.1-pro-preview",
    "openai/gpt-5.6-luna",
    "openai/gpt-5.6-terra",
    "openai/gpt-5.6-sol",
    "openai/gpt-5.4-mini",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-5",
    "x-ai/grok-4.6",
    "x-ai/grok-4.3",
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-pro",
    "qwen/qwen3.7-flash",
    "qwen/qwen3.7-plus",
    "qwen/qwen3.8-max",
    "moonshotai/kimi-k3",
    "moonshotai/kimi-k2.6",
    "z-ai/glm-5.2",
    "z-ai/glm-4.7-flash",
    "minimax/minimax-m3",
    "nvidia/nemotron-3.5-lightning",
]


def build_prompt(trace_path):
    """The exact user message the sidecar would send, from a real recorded position."""
    policy = LLMPolicy()
    with open(trace_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            d = json.loads(line, strict=False)
            acts = d.get("available_actions") or []
            if len(acts) >= 12:                      # a position with a real choice in it
                user = policy._render_state(d["state"]) + "\n\n" + policy._render_actions(acts)
                return user + "\n\nTake one action now.", acts, policy


def call(model, body, key, timeout):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {
        "Content-Type": "application/json", "Authorization": "Bearer " + key,
        "HTTP-Referer": "https://github.com/yotisstudios/Warrior", "X-Title": "Warrior latency probe",
    })
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read().decode("utf-8", "replace"))
    # OpenRouter can return 200 with an error object in the body -- an upstream refusal that is not
    # an HTTPError and would otherwise be counted as a successful fast reply.
    if payload.get("error"):
        raise RuntimeError(str(payload["error"])[:120])
    return time.time() - t0, payload


def probe(model, user, tools, legal, key, args):
    """Sequential samples for one model, so its own latency is never measured under self-contention."""
    lats, toks, costs, tool_ok, legal_ok, err = [], [], [], 0, 0, ""
    for _ in range(args.n):
        body = {"model": model, "usage": {"include": True},
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": user}],
                "tools": tools, "tool_choice": args.tool_choice,
                "temperature": 0.7, "max_tokens": args.max_tokens}
        try:
            dt, payload = call(model, body, key, args.timeout)
            if args.consult:
                # A second round trip with the tool result appended -- what an advisor arm actually
                # pays per decision.
                msg = (payload.get("choices") or [{}])[0].get("message") or {}
                body["messages"].append({"role": "assistant", "content": msg.get("content"),
                                         "tool_calls": msg.get("tool_calls")})
                tc = (msg.get("tool_calls") or [{}])[0].get("id") or "c1"
                body["messages"].append({"role": "tool", "tool_call_id": tc,
                                         "name": "consult_expert",
                                         "content": "The expert ranks: move_9_1 66%, "
                                                    "move_9_2 22%, move_9_3 12%."})
                dt2, payload = call(model, body, key, args.timeout)
                dt += dt2
            lats.append(dt)
            u = payload.get("usage") or {}
            toks.append(u.get("completion_tokens") or 0)
            costs.append(u.get("cost") or 0.0)
            calls = ((payload.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []
            if calls:
                tool_ok += 1
                a = json.loads(calls[0]["function"].get("arguments") or "{}")
                if str(a.get("action_id")) in legal:
                    legal_ok += 1
        except urllib.error.HTTPError as e:
            try:
                err = "HTTP %s: %s" % (e.code, json.loads(e.read())["error"]["message"][:70])
            except Exception:                                        # noqa: BLE001
                err = "HTTP %s" % e.code
        except Exception as e:                                       # noqa: BLE001
            err = ("%s: %s" % (type(e).__name__, e))[:90]
    return model, lats, toks, costs, tool_ok, legal_ok, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--consult", action="store_true",
                    help="two calls per decision, as an arm holding an advisor does")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--workers", type=int, default=8)
    # A reasoning model spends its output budget thinking BEFORE it emits the call, so too low a cap
    # truncates it mid-thought and scores as "cannot call tools" -- 2500 does exactly that to
    # minimax-m3 (needs ~6k) and nemotron (~16k). This is the sidecar's default too.
    ap.add_argument("--max-tokens", type=int, default=2500)
    # Alibaba's provider rejects tool_choice=required with a 400. The sidecar downgrades to "auto"
    # on that error automatically (policy.py); this flag mirrors it.
    ap.add_argument("--tool-choice", default="required", choices=["required", "auto"])
    ap.add_argument("--decisions-per-match", type=int, default=60)
    ap.add_argument("--traces", default="../RaifuWars/data/cpu-traces-v3-ffa.jsonl")
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        print("OPENROUTER_API_KEY not set")
        return 2

    user, acts, policy = build_prompt(args.traces)
    legal = {str(a.get("action_id")) for a in acts}
    tools = policy._tools(acts)
    print("prompt: %d chars, %d legal actions%s\n"
          % (len(user), len(acts), "  (consult: 2 calls per decision)" if args.consult else ""))

    # Models run concurrently (different upstreams, nothing local is the bottleneck) but each
    # model's own samples are sequential, so a model is never timed against itself.
    results, failed = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(probe, m, user, tools, legal, key, args) for m in args.models]
        for i, f in enumerate(concurrent.futures.as_completed(futs), 1):
            r = f.result()
            (results if r[1] else failed).append(r)
            print("  [%2d/%d] %-34s %s" % (i, len(futs), r[0],
                  "%.1fs" % statistics.median(r[1]) if r[1] else r[6]), file=sys.stderr)

    print("\n%-30s %7s %7s %7s %6s %7s  %5s %5s %8s %8s" %
          ("model", "median", "min", "max", "out_tk", "$/dec", "tool", "legal", "match", "20-arm"))
    print("-" * 108)
    for model, lats, toks, costs, tool_ok, legal_ok, _ in sorted(
            results, key=lambda r: statistics.median(r[1])):
        med = statistics.median(lats)
        match_s = med * args.decisions_per_match
        flag = "  <-- OVER 30s DEADLINE" if med > 30 else ""
        print("%-30s %6.1fs %6.1fs %6.1fs %6.0f %7.4f  %2d/%-2d %2d/%-2d %6.0fm %7.1fh%s"
              % (model, med, min(lats), max(lats), statistics.mean(toks), statistics.mean(costs),
                 tool_ok, len(lats), legal_ok, len(lats), match_s / 60, match_s * 20 / 3600, flag))
    for model, _, _, _, _, _, err in failed:
        print("%-30s %7s  %s" % (model, "FAIL", err or "no response"))

    n_dec = args.decisions_per_match * 20
    print("\n  match = median x %d decisions; 20-arm = one tournament arm (wall clock, unparallelised)."
          % args.decisions_per_match)
    print("  $/dec x %d = arm cost. out_tk is completion tokens: a reasoner burning thousands per" % n_dec)
    print("  decision is what breaks both the deadline and the budget.")
    print("  the game's protocol deadline is RW_WARRIOR_DEADLINE_MS (default 30s). A median above it")
    print("  ends turns instead of playing them and the match finishes with nobody having played.")
    print("  tool/legal are out of %d attempts -- a fast model that cannot emit a legal tool call is"
          % args.n)
    print("  useless here, whatever its latency.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
