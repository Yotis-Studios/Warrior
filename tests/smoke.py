"""End-to-end smoke test: start a real sidecar, speak the real protocol at it, check the replies.

Deliberately not a unit test of the policy. What breaks in practice is the seam -- a handler that
404s because a path gained a trailing slash, a response missing the field the client reads, a
retry that is not idempotent. Those only fail over a socket.

    python tests/smoke.py                 # random policy, no model needed
    python tests/smoke.py --policy llm    # exercises the model path too

Exits non-zero on the first failure, so it works as a CI gate.
"""

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "probe"))

from http.server import ThreadingHTTPServer  # noqa: E402

from sidecar.policy import build_policy  # noqa: E402
from sidecar.server import Handler, Sidecar  # noqa: E402
from state_fixture import AVAILABLE_ACTIONS, MAP_ASCII, MAP_LEGEND, PLAYERS, POINTS, SELF, VALID_IDS  # noqa: E402

FAILURES = []


def check(label, condition, detail=""):
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f"  -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)
    return condition


def post(base, path, payload, timeout=180):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(base, path, timeout=10):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read())


def act_request(reason="turn_start", request_id="m1-t23-a1", last_action=None, actions=None):
    return {
        "protocol_version": "0.1",
        "request_id": request_id,
        "match_id": "m1",
        "seat": 1,
        "reason": reason,
        "deadline_ms": 30000,
        "state": {
            "turn": 23,
            "self": SELF,
            "players": PLAYERS,
            "board": {"ascii": MAP_ASCII, "legend": MAP_LEGEND, "points": POINTS},
            "events_since_last_turn": ["Arisaka shot you for 1 damage from (10,4)."],
            "chat": [{"seat": 3, "text": "truce on the left?"}],
        },
        "available_actions": AVAILABLE_ACTIONS if actions is None else actions,
        "last_action": last_action,
        "screenshot": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="random", choices=["random", "first-legal", "llm"])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default=None)
    ap.add_argument("--port", type=int, default=8891)
    args = ap.parse_args()

    policy = build_policy(args.policy, url=args.url, model=args.model)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    httpd.sidecar = Sidecar(policy)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{args.port}"
    time.sleep(0.2)

    print(f"\n=== warrior sidecar smoke test (policy={args.policy}) ===\n")

    print("handshake")
    h = get(base, "/v1/health")
    check("health responds", isinstance(h, dict))
    check("advertises protocol_version", h.get("protocol_version") == "0.1", str(h))
    check("advertises capabilities", isinstance(h.get("capabilities"), dict))

    print("\nmatch lifecycle")
    check("match/start accepted",
          post(base, "/v1/match/start", {"match_id": "m1", "seat": 1}).get("ok") is True)
    check("event accepted",
          post(base, "/v1/event", {"match_id": "m1", "event": "chat",
                                   "seat": 3, "text": "hi"}).get("ok") is True)

    print("\ncore act call")
    t0 = time.time()
    r = post(base, "/v1/act", act_request())
    dt = time.time() - t0
    check("returns an action_id", bool(r.get("action_id")), str(r))
    check("action_id is one that was offered", r.get("action_id") in VALID_IDS,
          f"got {r.get('action_id')!r}")
    check("args is an object", isinstance(r.get("args"), dict))
    print(f"         chose {r.get('action_id')!r} in {dt:.1f}s"
          + (f" (reported {r['took_ms']}ms)" if "took_ms" in r else ""))

    print("\nidempotency (PROTOCOL.md s7)")
    again = post(base, "/v1/act", act_request())
    check("same request_id returns the same action",
          again.get("action_id") == r.get("action_id"),
          f"{r.get('action_id')!r} then {again.get('action_id')!r}")

    print("\nretry path (PROTOCOL.md s6)")
    retry = post(base, "/v1/act", act_request(
        reason="retry", request_id="m1-t23-a1r",
        last_action={"action_id": "atk_seat9", "ok": False,
                     "error": "atk_seat9 was not in available_actions"}))
    check("retry returns a legal action", retry.get("action_id") in VALID_IDS,
          f"got {retry.get('action_id')!r}")

    print("\ndegenerate inputs")
    only_end = [a for a in AVAILABLE_ACTIONS if a["type"] == "end_turn"]
    r_end = post(base, "/v1/act", act_request(request_id="m1-only-end", actions=only_end))
    check("single legal action is chosen", r_end.get("action_id") == "end_turn",
          str(r_end.get("action_id")))

    r_none = post(base, "/v1/act", act_request(request_id="m1-none", actions=[]))
    check("empty action list does not crash", bool(r_none.get("action_id")), str(r_none))

    # The fallback must come from what was OFFERED. A sidecar that answers "end_turn" to a list
    # that never contained it has committed the exact illegal-action error the protocol exists
    # to prevent -- so offer a list with no end_turn and check we stay inside it.
    no_end = [a for a in AVAILABLE_ACTIONS if a["type"] != "end_turn"]
    r_noend = post(base, "/v1/act", act_request(request_id="m1-noend", actions=no_end))
    offered = {a["action_id"] for a in no_end}
    check("never invents end_turn when it was not offered",
          r_noend.get("action_id") in offered, f"got {r_noend.get('action_id')!r}")

    print("\n404 handling")
    try:
        post(base, "/v1/nonsense", {})
        check("unknown path 404s", False, "no error raised")
    except urllib.error.HTTPError as exc:
        check("unknown path 404s", exc.code == 404, f"got {exc.code}")

    httpd.shutdown()
    print()
    if FAILURES:
        print(f"=== {len(FAILURES)} FAILED: {', '.join(FAILURES)} ===")
        return 1
    print("=== all checks passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
