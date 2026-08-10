"""Does a real /v1/act still work with a screenshot attached, and does the image change the answer?

    python probe/probe_vision_act.py <screenshot.png> --trace <traces.jsonl> [--url http://127.0.0.1:8881]

probe_vision.py established that the model can READ a Raifu Wars screenshot -- exact HUD digits,
figure counts, UI language, all against a blind control. This asks the next question, which is a
different one: with the protocol's own payload wrapped around it, does the image survive the trip
and does it change what the model decides?

Three failure modes it is looking for, none of which probe_vision.py could see:

    the image is dropped      the sidecar advertises vision but the payload never reaches the model,
                              so a vision run is silently a text run and every number from it is
                              mislabelled
    the image breaks the call the tool call fails or comes back illegal once an image is present --
                              a real risk, because tool_choice=required and image tokens compete
    the image changes nothing identical action with and without. Not a bug: a NULL, and the whole
                              point of measuring before building an annotated renderer

REAL PAYLOADS, not synthetic ones. The states come from a trace file the game actually produced,
so the action list, the board and the rules text are exactly what a warrior was handed. A
hand-written payload would test the sidecar against a game that does not exist.

The image is deliberately the SAME for every state, because a viewport screenshot from a different
turn is wrong for this board -- and that is fine here. The question is whether an image changes the
decision at all, not whether a correct image improves it. A difference caused by an irrelevant
image is still evidence the image is being attended to; no difference is evidence it is not.
"""

import argparse
import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request


def post(url, body, timeout=180.0):
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/act",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {"error": "HTTP %s: %s" % (exc.code, exc.read().decode("utf-8", "replace")[:200])}
    except Exception as exc:                                    # noqa: BLE001
        return {"error": str(exc)}


def load_states(path, limit):
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            acts = d.get("available_actions") or []
            # Only decisions with a real choice. A one-action state cannot show a difference, so
            # including it would dilute the agreement rate with rows that agree by construction.
            if len(acts) < 3:
                continue
            out.append(d)
            if len(out) >= limit:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8881")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    for p in (args.image, args.trace):
        if not os.path.isfile(p):
            print("no such file: %s" % p)
            return 2

    with open(args.image, "rb") as fh:
        raw = fh.read()
    b64 = base64.b64encode(raw).decode("ascii")
    mime = mimetypes.guess_type(args.image)[0] or "image/png"

    health = post(args.url, {}) and None
    try:
        with urllib.request.urlopen(args.url.rstrip("/") + "/v1/health", timeout=10) as r:
            h = json.loads(r.read().decode("utf-8"))
        vision = bool((h.get("capabilities") or {}).get("vision"))
        model = h.get("model")
    except Exception as exc:                                    # noqa: BLE001
        print("cannot reach sidecar at %s: %s" % (args.url, exc))
        return 1

    print("=== vision /v1/act probe ===")
    print("  sidecar : %s" % args.url)
    print("  model   : %s" % model)
    print("  vision  : %s" % vision)
    print("  image   : %s (%d KB -> %d base64 chars)"
          % (os.path.basename(args.image), len(raw) // 1024, len(b64)))
    if not vision:
        print("\n  REFUSING: this sidecar advertises vision:false. PROTOCOL.md s3 says a client")
        print("  MUST NOT send a screenshot to it. Start one with --vision.")
        return 1

    states = load_states(args.trace, args.n)
    if not states:
        print("no usable states in %s" % args.trace)
        return 1
    print("  states  : %d real decisions from %s\n" % (len(states), os.path.basename(args.trace)))

    same = 0
    errors = 0
    for i, st in enumerate(states):
        ids = [a.get("action_id") for a in st["available_actions"]]
        base_req = {
            "protocol_version": "0.1",
            "match_id": "probe-vision",
            "seat": st.get("seat", 0),
            "reason": "turn_start",
            "deadline_ms": 60000,
            "state": st["state"],
            "available_actions": st["available_actions"],
        }
        # DISTINCT request_ids. The sidecar caches by request_id for idempotency -- reusing one
        # would return the first answer for the second call and manufacture a perfect agreement.
        blind = dict(base_req, request_id="probe-%d-blind" % i)
        seen = dict(base_req, request_id="probe-%d-seen" % i,
                    screenshot={"b64": b64, "mime": mime})

        rb, rs = post(args.url, blind), post(args.url, seen)
        ab = rb.get("action_id", "ERR:" + str(rb.get("error"))[:40])
        as_ = rs.get("action_id", "ERR:" + str(rs.get("error"))[:40])
        if str(ab).startswith("ERR") or str(as_).startswith("ERR"):
            errors += 1
        legal = "" if as_ in ids else "  <-- ILLEGAL, not in available_actions"
        if ab == as_:
            same += 1
        print("  %2d. blind=%-16s sighted=%-16s %s%s"
              % (i + 1, ab, as_, "same" if ab == as_ else "DIFFER", legal))

    n = len(states)
    print("\n  identical with and without the image: %d / %d (%.0f%%)"
          % (same, n, 100.0 * same / n))
    print("  errors: %d" % errors)
    if same == n:
        print("\n  The image changed nothing. Either it is not reaching the model, or it is being")
        print("  ignored -- and a vision tournament would be a text tournament wearing a label.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
