"""Did the fine-tune learn the built-in AI's decisions? Ask the held-out split, not a tournament.

    python tools/eval_sft.py data/sft/val.jsonl --url http://127.0.0.1:8080

A tournament is the real test and it costs hours. This is the cheap one that comes first: replay
every held-out decision, ask the model for one action, and compare it to what the built-in AI
actually did on that exact board. Minutes, not hours -- so a model that learned nothing is found
out before anybody spends an evening on match data.

It is deliberately NOT a claim about playing strength. Top-1 agreement with the teacher is what
supervised training optimises, so a high score here is the training working, not the model being
good -- the teacher is a hand-written heuristic, and copying it perfectly caps you at its level.
A LOW score, though, is decisive: if the model cannot reproduce the teacher on held-out boards
from the same map, no tournament result is going to be a pleasant surprise.

THREE NUMBERS, and the second and third are the ones that matter most early.

    top-1 agreement   how often it picks the teacher's exact action_id
    illegal rate      how often it answers with an id that was not offered. Legality is the
                      protocol's floor and a fine-tune can destroy it -- the base model scored
                      25/25 legal before training, so any regression here is caused by us
    no-call rate      how often no tool call comes back at all. A model that stops emitting tool
                      calls looks like a scoring problem and is a formatting one

BASELINES ARE PRINTED BESIDE THE SCORE, because a bare percentage is unreadable. Always picking
the most common action in the split is a real policy and it is what a degenerate fine-tune
collapses to -- a model that scores 35% has learned nothing if `move` is 35% of the answers.
"""

import argparse
import collections
import json
import os
import sys
import urllib.error
import urllib.request


def family(a):
    for p, n in (("move_", "move"), ("atk_s", "attack"), ("card_use_", "play card"),
                 ("card_drop_", "discard"), ("pick_", "select tile")):
        if a.startswith(p):
            return n
    return a


def ask(url, model, messages, tools, timeout, api_key=None):
    body = {
        "messages": messages,
        "tools": tools,
        "tool_choice": "required",
        "temperature": 0.0,
        "max_tokens": 200,
        "stream": False,
    }
    if model:
        body["model"] = model
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
        headers["HTTP-Referer"] = "https://github.com/yotisstudios/Warrior"
        headers["X-Title"] = "Warrior protocol"
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            d = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        # NOT EVERY PROVIDER SUPPORTS tool_choice="required". Measured on OpenRouter:
        # qwen3.7-flash's provider answers 400 `The tool_choice parameter does not support
        # "required"` to every request -- which, without a fallback, reads as a model scoring zero
        # rather than an API that refused to be asked.
        #
        # "auto" is weaker: the model MAY answer in prose instead of calling the tool. But a weaker
        # constraint that runs beats a stronger one that 400s, and the caller already reports a
        # missing tool call as its own number.
        #
        # Retried once, on that specific complaint only. A blanket downgrade would silently weaken
        # the constraint for every provider that does honour it.
        if exc.code == 400 and "tool_choice" in detail:
            body["tool_choice"] = "auto"
            req2 = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                          data=json.dumps(body).encode("utf-8"),
                                          headers=headers)
            try:
                with urllib.request.urlopen(req2, timeout=timeout) as resp:
                    d = json.loads(resp.read().decode("utf-8"))
            except Exception as exc2:                           # noqa: BLE001
                return None, "http %s then auto failed: %s" % (exc.code, exc2)
        else:
            return None, "http %s: %s" % (exc.code, detail[:200])
    except Exception as exc:                                    # noqa: BLE001
        return None, "transport: %s" % exc

    msg = (d.get("choices") or [{}])[0].get("message") or {}
    calls = msg.get("tool_calls") or []
    if not calls:
        return None, "no tool call"
    try:
        args = json.loads(calls[0]["function"]["arguments"])
    except Exception:                                           # noqa: BLE001
        return None, "unparseable arguments"
    aid = args.get("action_id")
    return (str(aid) if aid is not None else None), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("val")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=0, help="cap rows (0 = all)")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--api-key", default=None,
                    help="bearer token; defaults to $OPENROUTER_API_KEY")
    args = ap.parse_args()
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")

    rows = []
    with open(args.val, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.n:
        rows = rows[:args.n]
    if not rows:
        print("no rows in %s" % args.val)
        return 1

    hit = illegal = nocall = 0
    failed = 0                 # request never reached a model: HTTP or transport
    fail_examples = []
    per_fam = collections.Counter()
    per_fam_hit = collections.Counter()
    target_fams = collections.Counter()
    predicted_fams = collections.Counter()

    for i, row in enumerate(rows):
        msgs = row["messages"]
        target = json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"])["action_id"]
        legal = set(row["tools"][0]["function"]["parameters"]["properties"]["action_id"]["enum"])
        # Only the system and user turns -- the assistant turn IS the answer.
        pred, err = ask(args.url, args.model, msgs[:2], row["tools"], args.timeout, api_key)

        tf = family(target)
        target_fams[tf] += 1
        per_fam[tf] += 1
        if pred is None:
            # SPLIT, because they have opposite causes. "no tool call" is a fact about the model;
            # an HTTP or transport error means the question never arrived, and averaging the two
            # into one column is how a wrong URL gets reported as a model limitation.
            if err and not err.startswith("no tool call"):
                failed += 1
                if len(fail_examples) < 3:
                    fail_examples.append(err)
            else:
                nocall += 1
        else:
            predicted_fams[family(pred)] += 1
            if pred not in legal:
                illegal += 1
            if pred == target:
                hit += 1
                per_fam_hit[tf] += 1
        if (i + 1) % 50 == 0:
            print("  ... %d/%d" % (i + 1, len(rows)), file=sys.stderr)

    n = len(rows)
    # THE BASELINE. A fine-tune that collapses to one action scores exactly this, and without it
    # printed next to the result a collapse reads as partial success.
    most_common_fam, mc = target_fams.most_common(1)[0]
    print("\n=== held-out agreement with the built-in AI: %d decisions ===" % n)
    print("  top-1 agreement : %5.1f%%  (%d/%d)" % (100.0 * hit / n, hit, n))
    print("  always-%-9s: %5.1f%%   <-- baseline: the single most common action family"
          % (most_common_fam, 100.0 * mc / n))
    print("  illegal answers : %5.1f%%  (%d)  -- action_id not in the offered set"
          % (100.0 * illegal / n, illegal))
    print("  no tool call    : %5.1f%%  (%d)  -- model answered without calling the tool"
          % (100.0 * nocall / n, nocall))
    print("  request FAILED  : %5.1f%%  (%d)  -- never reached a model (HTTP/transport)"
          % (100.0 * failed / n, failed))
    for e in fail_examples:
        print("      %s" % e)
    if failed:
        print("\n  REQUESTS FAILED. This is not a result about the model -- check the URL is a raw")
        print("  OpenAI-compatible endpoint (NOT the sidecar, which serves /v1/act), the model id")
        print("  exists, and the key is set.")

    print("\n  %-14s %8s %8s %7s" % ("target family", "count", "hit", "rate"))
    for f, c in target_fams.most_common():
        h = per_fam_hit[f]
        print("  %-14s %8d %8d %6.1f%%" % (f, c, h, 100.0 * h / c if c else 0))

    print("\n  predicted mix vs target mix (collapse shows up here first):")
    for f in sorted(set(target_fams) | set(predicted_fams), key=lambda x: -target_fams.get(x, 0)):
        t = 100.0 * target_fams.get(f, 0) / n
        p = 100.0 * predicted_fams.get(f, 0) / n
        print("    %-14s target %5.1f%%   predicted %5.1f%%" % (f, t, p))

    print("\n  Top-1 agreement is what training optimises, so a high score means the training")
    print("  worked -- not that the model plays well. It is capped by the teacher. A LOW score")
    print("  is the decisive one: run a tournament only if this clears the baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
