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
import os
import sys
import threading
import time
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

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

    @staticmethod
    def _key(req):
        """Idempotency key: the MATCH and the request id, never the request id alone.

        The game mints ids as rw-t<turn>-s<seat>-n<seq> and resets both the turn number and the
        sequence counter at every match boundary, so match 2 opens with the same "rw-t0-s0-n1" that
        match 1 did. Keyed on request_id alone -- and never cleared between matches, only FIFO
        evicted at 4096 -- the second match's opening decisions are answered with the FIRST match's
        actions, chosen for a different board. It presents as a warrior that starts each match by
        making a couple of nonsense moves and then recovers, which reads like a bad model rather
        than a stale cache.
        """
        return (req.get("match_id") or "-", req.get("request_id"))

    def act(self, req):
        rid = req.get("request_id")
        if rid:
            key = self._key(req)
            with self.lock:
                cached = self.seen.get(key)
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
                self.seen[self._key(req)] = resp
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


def dispatch(sc, path, req):
    """One request -> (status, body). The ONLY place a route is decided.

    Both transports call this. When HTTP owned the routing, adding TCP would have meant a second
    copy of the same if-chain, and the two would drift the first time a route changed -- which is
    how a transport ends up quietly supporting a different protocol version than the one beside it.
    """
    path = (path or "").rstrip("/")
    try:
        if path == "/v1/act":
            return 200, sc.act(req)
        if path == "/v1/match/start":
            sc.policy.match_start(req)
            log.info("match %s start, seat %s", req.get("match_id"), req.get("seat"))
            return 200, {"ok": True}
        if path == "/v1/match/end":
            sc.policy.match_end(req)
            counters = sc.matches.get(req.get("match_id") or "-", {})
            log.info("match %s end: %s | %s", req.get("match_id"), req.get("result"), counters)
            return 200, {"ok": True}
        if path == "/v1/event":
            sc.policy.event(req)
            return 200, {"ok": True}
        if path in ("/v1/health", "/health"):
            return 200, sc.health()
        return 404, {"error": "not found"}
    except Exception as exc:                                        # noqa: BLE001
        log.exception("handler failed")
        return 500, {"error": str(exc)}


def serve_tcp(sidecar, host, port, stop):
    """A PERSISTENT framed connection, as an alternative to one HTTP request per decision.

    WHY. Not latency -- that was measured and it is not the problem. A localhost HTTP round trip
    costs single-digit milliseconds against 600-4000ms of model call, so the transport is under 0.3%
    of a decision and rewriting it buys nothing measurable. What HTTP/1.0 costs is a NEW CONNECTION
    per decision: across one screening the sidecar logged 38 resets and the client re-sent 2,357 of
    5,999 requests. None of that broke a match -- the idempotency cache absorbed it -- but it is
    thousands of connections and thousands of duplicate deliveries to be absorbed, and every one is
    a chance for the absorbing to be wrong.

    FRAMING. 4-byte big-endian length, then that many bytes of UTF-8 JSON:

        {"path": "/v1/act", "body": {...}}   ->   {"status": 200, "body": {...}}

    A length prefix rather than a delimiter because the payload is JSON containing arbitrary text --
    chat messages included -- so any sentinel byte has to be escaped, and an unescaped one is a
    parser that stops mid-message. The length is read first and exactly that many bytes are
    consumed, which is also what makes TCP reassembly correct: recv() returns what it has, not what
    was sent, and a reader that assumes one recv is one message works until a message spans packets.

    ONE THREAD PER CONNECTION, AND NOT ONE PER REQUEST. The distinction is the whole bug this
    docstring used to describe wrongly.

    A thread per REQUEST is what put 28-35 GB and several thousand threads into this process once
    already, and it buys nothing: the game asks for one decision and blocks, so there is never a
    second request in flight on a connection.

    But serving one CONNECTION at a time -- accept, run to EOF, accept the next -- means any client
    that holds its socket open blocks every later one. A Runner that is still alive from a previous
    match (this project reliably leaves a couple stuck behind modal dialogs, immune to taskkill)
    keeps its connection, so the next match's frames are never read at all. The game waits out its
    60s deadline, abandons the request, and the reply that finally arrives finds nothing pending.
    Measured as 28% of decisions retried against 0.8% over HTTP, and it did not show up under a
    policy that answers instantly, because nothing was ever queued behind anything.

    Connections are bounded and long-lived -- one per game process -- so a thread each is a handful,
    not thousands.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(8)
    srv.settimeout(0.5)                     # so `stop` is noticed without a signal
    log.info("warrior sidecar tcp on %s:%d (framed json)", host, port)

    def read_exactly(conn, n):
        """n bytes or None at clean EOF. Short reads are normal, not an error."""
        buf = bytearray()
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)

    def serve_conn(conn, addr):
        log.info("tcp client connected from %s", addr)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while not stop.is_set():
                head = read_exactly(conn, 4)
                if head is None:
                    break
                size = int.from_bytes(head, "big")
                # A frame larger than this is a bug or a stray non-client, not a decision: the
                # biggest real payload measured is ~35 KB. Refusing beats allocating it.
                if size <= 0 or size > (8 << 20):
                    log.warning("tcp frame size %d refused", size)
                    break
                raw = read_exactly(conn, size)
                if raw is None:
                    break
                mid = None
                try:
                    msg = json.loads(raw.decode("utf-8", "replace"))
                    mid = msg.get("id")
                    status, body = dispatch(sidecar, msg.get("path"), msg.get("body") or {})
                except json.JSONDecodeError as exc:
                    status, body = 400, {"error": "bad JSON: %s" % exc}
                # ECHO THE FRAME ID. The client pairs replies by it, and must not pair them by
                # arrival order: the game can have more than one request outstanding -- its
                # watchdog re-asks while an earlier one is still in flight -- and matching the
                # oldest pending request to the next reply then binds an action chosen for one
                # board to a decision about another.
                out = json.dumps({"id": mid, "status": status, "body": body}).encode("utf-8")
                conn.sendall(len(out).to_bytes(4, "big") + out)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError) as exc:
            # The game exiting mid-match closes its socket. Expected, not an error.
            log.info("tcp client went away (%s)", type(exc).__name__)
        except OSError as exc:
            log.info("tcp connection ended (%s)", type(exc).__name__)
        except Exception:                                           # noqa: BLE001
            log.exception("tcp connection failed")
        finally:
            try:
                conn.close()
            except OSError:
                pass

    while not stop.is_set():
        try:
            conn, addr = srv.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        # Daemon: a stuck client must never keep the process alive at shutdown.
        threading.Thread(target=serve_conn, args=(conn, addr),
                         name="warrior-tcp-conn", daemon=True).start()
    srv.close()


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
        status, body = dispatch(self.server.sidecar, self.path, {})
        self._send(status, body)

    def do_POST(self):
        try:
            req = self._read_json()
        except json.JSONDecodeError as exc:
            self._send(400, {"error": f"bad JSON: {exc}"})
            return
        status, body = dispatch(self.server.sidecar, self.path, req)
        self._send(status, body)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Warrior sidecar")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8879)
    # ON BY DEFAULT, at http port + 1. TCP is the transport; the port is derived rather than
    # configured so parallel arms stay on distinct ports without a second variable to keep in step
    # with the first. 0 disables it.
    ap.add_argument("--tcp-port", type=int, default=-1,
                    help="framed-JSON TCP port (default: --port + 1; 0 to disable)")
    ap.add_argument("--skill", type=float, default=1.0,
                    help="hybrid only: 1.0 plays the net's argmax; lower samples its own "
                         "distribution, for an opponent that is good rather than perfect")
    ap.add_argument("--policy", default="random",
                    choices=["random", "first-legal", "llm", "hybrid"])
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
    ap.add_argument("--api-key", default=None,
                    help="bearer token for a hosted endpoint. Defaults to $OPENROUTER_API_KEY; "
                         "prefer the environment so the key stays out of your shell history")
    ap.add_argument("--openrouter", metavar="MODEL",
                    help="shorthand for --url https://openrouter.ai/api --model MODEL, e.g. "
                         "--openrouter anthropic/claude-3.5-sonnet")
    # THE EXPERT IS A TOOL, NOT A ROUTER. The obvious composition is a switch -- let the RL policy
    # take the actions and the LLM do the talking -- and it throws away the interesting half. The
    # scorer beats the built-in AI 55% to 28% and cannot say a word about why; the model can hold a
    # plan and explain itself and plays badly. Offering the scorer as something the model may ASK
    # keeps the model in charge and gives it a strong opinion to accept or overrule.
    ap.add_argument("--expert", metavar="CHECKPOINT",
                    help="an RL checkpoint the model may consult as a tool")
    ap.add_argument("--rl-path", default=None,
                    help="path to the raifuwars-rl repo (default: sibling of this one, or RW_RL_PATH)")
    ap.add_argument("--persona", default="",
                    help="how the seat carries itself, e.g. 'a cheerful anime girl who takes the "
                         "sport extremely seriously'. Presentation only -- it never licenses an "
                         "illegal action.")
    ap.add_argument("--chat-opener", action="store_true",
                    help="invite a chat message at the start of each of your own turns")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    if args.openrouter:
        args.url = "https://openrouter.ai/api"
        args.model = args.openrouter
        args.policy = "llm"

    # FAIL HERE, NOT ON THE FIRST TURN OF A TOURNAMENT. A hosted endpoint without a key answers
    # 401 on every request, and the game reads that as the seat failing to decide -- forty matches
    # of forfeits that look like a model being bad at the game.
    key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if args.policy == "llm" and args.url.startswith("https://") and not key:
        ap.error("a hosted endpoint needs a key: pass --api-key or set OPENROUTER_API_KEY")
    if args.policy == "llm" and args.url.startswith("https://") and not args.model:
        ap.error("a hosted endpoint needs an explicit --model (or --openrouter MODEL)")

    expert = None
    if args.expert:
        from expert import Expert, ExpertUnavailable
        try:
            expert = Expert(args.expert, rl_path=args.rl_path)
            print("[warrior] expert loaded: %s -- the model may consult it" % expert.name, flush=True)
        except ExpertUnavailable as exc:
            # LOUD, NOT SILENT. A sidecar that quietly runs without the expert it was asked for
            # produces a run whose numbers mean something other than what was intended, and
            # nothing in the output would say so.
            raise SystemExit("--expert given but unusable: %s" % exc)

    policy = build_policy(
        args.policy, url=args.url, model=args.model, temperature=args.temperature,
        max_tokens=args.max_tokens, skill=args.skill, thinking=args.thinking, vision=args.vision, seed=args.seed,
        api_key=args.api_key, expert=expert, persona=args.persona,
        chat_opener=args.chat_opener)

    # SINGLE-THREADED. ThreadingHTTPServer's handler threads never exit here -- measured at ~7
    # surviving threads and 3.8 MB per request, which reached 28-35 GB across one tournament,
    # starved the machine, and killed the game with "Memory allocation failed". The same fix was
    # already applied to the RL sidecar and NOT here, and this server then died mid-screening
    # after ~900 retried requests, taking an arm's results with it.
    #
    # Threads bought nothing: the game is a SEQUENTIAL client. It asks for one decision and blocks
    # until it gets it, so there is never a second request in flight for this server to overlap.
    # Parallel arms are separate processes on separate ports.
    httpd = HTTPServer((args.host, args.port), Handler)
    sidecar = Sidecar(policy)
    httpd.sidecar = sidecar
    log.info("warrior sidecar v%s on http://%s:%d  policy=%s%s",
             PROTOCOL_VERSION, args.host, args.port, policy.name,
             f" -> {args.url}" if args.policy == "llm" else "")

    # TCP IS THE TRANSPORT; HTTP STAYS UP BESIDE IT. Not as an A/B: the game holds one connection
    # for its whole life and speaks frames, while /v1/health over HTTP is what every script,
    # readiness probe and curl in this repo already uses, and both go through the same dispatch()
    # so neither can drift from the other.
    stop = threading.Event()
    tcp = None
    tcp_port = args.port + 1 if args.tcp_port < 0 else args.tcp_port
    if tcp_port:
        tcp = threading.Thread(target=serve_tcp, name="warrior-tcp",
                               args=(sidecar, args.host, tcp_port, stop), daemon=True)
        tcp.start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        stop.set()
        if tcp is not None:
            tcp.join(timeout=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
