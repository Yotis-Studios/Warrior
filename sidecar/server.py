"""The Warrior sidecar: HTTP server implementing PROTOCOL.md, stdlib only.

Run it, point a game client at it, and one seat is played by whatever policy you chose.

    python -m sidecar.server --policy random                 # CI stub, no model needed
    python -m sidecar.server --policy llm --url http://127.0.0.1:8080

Stdlib on purpose. This is the reference implementation of a protocol; anything a reader has to
`pip install` before they can see how the protocol works is in the way. ThreadingHTTPServer is
enough -- a sidecar answers one seat, and even four seats on one process is a handful of requests
per second at model latency.
"""

import argparse
import json
import logging
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from .policy import PolicyError, build_policy
except ImportError:  # run as a plain script rather than a module
    from policy import PolicyError, build_policy

PROTOCOL_VERSION = "0.1"

log = logging.getLogger("warrior")


class Sidecar:
    """Policy plus the bookkeeping the protocol requires of a sidecar.

    Holds two things beyond the policy: an idempotency cache keyed by request_id, and per-match
    counters. The cache matters because clients retry -- PROTOCOL.md section 7 -- and a policy
    asked the same question twice will happily give two different answers, which reads to the
    game as a warrior changing its mind mid-action.
    """

    def __init__(self, policy):
        self.policy = policy
        self.lock = threading.Lock()
        self.seen = {}          # request_id -> response
        self.matches = {}       # match_id -> counters
        self.total_acts = 0
        self.total_errors = 0

    def _counters(self, match_id):
        return self.matches.setdefault(
            match_id or "-", {"acts": 0, "retries": 0, "errors": 0, "started": time.time()})

    def health(self):
        caps = dict(self.policy.capabilities())
        caps.setdefault("max_deadline_ms", 30000)
        return {
            "protocol_version": PROTOCOL_VERSION,
            "name": "warrior-reference",
            "policy": self.policy.name,
            # ASKED, not remembered. The harness stamps this into every result line and trace, so
            # a stale name here silently misattributes a whole run. See LLMPolicy.loaded_model.
            "model": (self.policy.loaded_model() if hasattr(self.policy, "loaded_model")
                      else getattr(self.policy, "model", None)),
            "capabilities": caps,
            "stats": {"acts": self.total_acts, "errors": self.total_errors,
                      "matches": len(self.matches)},
        }

    def act(self, req):
        rid = req.get("request_id")
        if rid:
            with self.lock:
                cached = self.seen.get(rid)
            if cached is not None:
                log.info("request_id %s served from cache (client retried)", rid)
                return cached

        counters = self._counters(req.get("match_id"))
        counters["acts"] += 1
        self.total_acts += 1
        if req.get("reason") == "retry":
            counters["retries"] += 1

        actions = req.get("available_actions") or []
        legal = {a["action_id"] for a in actions if "action_id" in a}

        t0 = time.time()
        try:
            action_id, args, commentary = self.policy.act(req)
        except PolicyError as exc:
            # The policy failed to decide. Fall back to something the client definitely offered
            # rather than returning an error the client has to interpret: a warrior that cannot
            # think must still not stall the match. PROTOCOL.md section 6.3.
            counters["errors"] += 1
            self.total_errors += 1
            log.warning("policy failed (%s) -- falling back", exc)
            action_id, args, commentary = self._fallback(actions), {}, None
        except Exception as exc:  # noqa: BLE001 -- a crashed sidecar must not hang a game
            counters["errors"] += 1
            self.total_errors += 1
            log.exception("policy raised")
            action_id, args, commentary = self._fallback(actions), {}, f"sidecar error: {exc}"

        # VALIDATE OUR OWN OUTPUT. The game validates too and is the authority -- but a bad id
        # costs a whole retry round trip, and we already know the schema enum does not bind the
        # model (PROTOCOL.md section 0). Catching it here turns a wasted turn into a log line.
        if legal and action_id not in legal:
            counters["errors"] += 1
            self.total_errors += 1
            log.warning("policy chose %r which was not offered -- falling back", action_id)
            action_id = self._fallback(actions)
            args = {}

        resp = {"action_id": action_id, "args": args or {}}
        if commentary:
            resp["commentary"] = commentary
        resp["took_ms"] = int((time.time() - t0) * 1000)

        if rid:
            with self.lock:
                self.seen[rid] = resp
                # Bounded. A long match is thousands of requests and this process may outlive
                # several of them.
                if len(self.seen) > 4096:
                    for k in list(self.seen)[:2048]:
                        del self.seen[k]
        return resp

    @staticmethod
    def _fallback(actions):
        """The safest offered action. Ending the turn if we may, otherwise whatever exists.

        Never a hardcoded "end_turn" string: if the client did not offer it, sending it is exactly
        the illegal-action problem this file exists to prevent.
        """
        for a in actions:
            if a.get("type") == "end_turn":
                return a["action_id"]
        return actions[0]["action_id"] if actions else "end_turn"


class Handler(BaseHTTPRequestHandler):
    server_version = "WarriorSidecar/" + PROTOCOL_VERSION
    sidecar = None  # set on the server instance

    def log_message(self, fmt, *a):
        log.debug("%s - %s", self.address_string(), fmt % a)

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length))

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/health", "/health"):
            self._send(200, self.server.sidecar.health())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.rstrip("/")
        try:
            req = self._read_json()
        except json.JSONDecodeError as exc:
            self._send(400, {"error": f"bad JSON: {exc}"})
            return

        sc = self.server.sidecar
        try:
            if path == "/v1/act":
                self._send(200, sc.act(req))
            elif path == "/v1/match/start":
                sc.policy.match_start(req)
                log.info("match %s start, seat %s", req.get("match_id"), req.get("seat"))
                self._send(200, {"ok": True})
            elif path == "/v1/match/end":
                sc.policy.match_end(req)
                counters = sc.matches.get(req.get("match_id") or "-", {})
                log.info("match %s end: %s | %s", req.get("match_id"),
                         req.get("result"), counters)
                self._send(200, {"ok": True})
            elif path == "/v1/event":
                sc.policy.event(req)
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            log.exception("handler failed")
            self._send(500, {"error": str(exc)})


def main(argv=None):
    ap = argparse.ArgumentParser(description="Warrior sidecar")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8879)
    ap.add_argument("--policy", default="random",
                    choices=["random", "first-legal", "llm"])
    ap.add_argument("--url", default="http://127.0.0.1:8080",
                    help="OpenAI-compatible endpoint, for --policy llm")
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--thinking", action="store_true",
                    help="let the model emit reasoning first: ~20x slower, no measured "
                         "accuracy gain on the probe fixture")
    ap.add_argument("--vision", action="store_true",
                    help="accept screenshots and forward them to the model")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    policy = build_policy(
        args.policy, url=args.url, model=args.model, temperature=args.temperature,
        max_tokens=args.max_tokens, thinking=args.thinking, vision=args.vision, seed=args.seed)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.sidecar = Sidecar(policy)
    log.info("warrior sidecar v%s on http://%s:%d  policy=%s%s",
             PROTOCOL_VERSION, args.host, args.port, policy.name,
             f" -> {args.url}" if args.policy == "llm" else "")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
