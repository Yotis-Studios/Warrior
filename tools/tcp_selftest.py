"""Does the framed TCP transport behave exactly like HTTP, including when the network is unkind?

    python tools/tcp_selftest.py

Starts a sidecar with both transports, then checks the things that actually break:

  1. every route answers over TCP, and answers the SAME as over HTTP
  2. many decisions over ONE connection -- the entire point of the change
  3. a frame split across two sends is reassembled (recv returns what arrived, not what was sent)
  4. two frames arriving in ONE packet are not merged into one
  5. the idempotency cache is scoped to the match, not to request_id alone

5 is not about TCP. It is here because the game resets its request-id counter every match, so
"rw-t0-s0-n1" recurs, and a cache keyed on the id alone answers match 2 with match 1's move.
"""
import json
import socket
import subprocess
import sys
import time
import urllib.request

HOST = "127.0.0.1"
HTTP, TCP = 8987, 8988


def frame(sock, path, body):
    msg = json.dumps({"path": path, "body": body}).encode()
    sock.sendall(len(msg).to_bytes(4, "big") + msg)
    return read_frame(sock)


def read_frame(sock):
    head = b""
    while len(head) < 4:
        head += sock.recv(4 - len(head))
    size = int.from_bytes(head, "big")
    buf = b""
    while len(buf) < size:
        buf += sock.recv(size - len(buf))
    return json.loads(buf.decode())


def http(path, body):
    r = urllib.request.Request("http://%s:%d%s" % (HOST, HTTP, path),
                               json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=10) as x:
        return json.loads(x.read())


def act(match, rid, acts):
    return {"protocol_version": "0.1", "request_id": rid, "match_id": match, "seat": 0,
            "reason": "turn_start", "deadline_ms": 30000, "available_actions": acts,
            "state": {"self": {"tile": [1, 1], "seat": 0, "health": 6, "max_health": 6,
                               "ammo": 5, "max_ammo": 5, "tier": 0, "stars": 0, "kills": 0,
                               "deaths": 0, "range": 5, "this_turn": {}, "hand": [],
                               "tier_progress": {"have": 0, "need": 25, "condition": "stars"}},
                      "players": [{"seat": 0, "is_self": True, "team": 1, "tile": [1, 1],
                                   "health": 6, "status": "alive"}],
                      "board": {"width": 21, "height": 21, "ascii": "", "legend": {},
                                "points": []}, "turn": 0}}


A = [{"action_id": "move_5_5", "type": "move", "label": "Move to (5,5)"},
     {"action_id": "end", "type": "end_turn", "label": "End"}]
B = [{"action_id": "move_9_9", "type": "move", "label": "Move to (9,9)"},
     {"action_id": "end", "type": "end_turn", "label": "End"}]


def main():
    proc = subprocess.Popen(
        [sys.executable, "sidecar/server.py", "--policy", "first-legal",
         "--port", str(HTTP), "--tcp-port", str(TCP)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    fails = []

    def check(name, ok, detail=""):
        print("  %-52s %s%s" % (name, "PASS" if ok else "FAIL", "  " + detail if detail else ""))
        if not ok:
            fails.append(name)

    try:
        for _ in range(40):
            try:
                socket.create_connection((HOST, TCP), timeout=0.5).close()
                break
            except OSError:
                time.sleep(0.25)

        s = socket.create_connection((HOST, TCP), timeout=10)

        r = frame(s, "/v1/health", {})
        check("health over TCP", r["status"] == 200 and "protocol_version" in r["body"])

        r = frame(s, "/v1/match/start", {"match_id": "M1", "seat": 0})
        check("match/start over TCP", r["status"] == 200 and r["body"].get("ok"))

        r = frame(s, "/v1/act", act("M1", "rw-t0-s0-n1", A))
        check("act over TCP returns a legal action", r["body"]["action_id"] in ("move_5_5", "end"),
              r["body"]["action_id"])

        # 2. many decisions, one connection -- the reason this transport exists
        ids = [frame(s, "/v1/act", act("M1", "rw-t1-s0-n%d" % i, A))["body"]["action_id"]
               for i in range(2, 42)]
        check("40 decisions on ONE connection", len(ids) == 40 and all(i in ("move_5_5", "end") for i in ids))

        # 3. a frame split across two sends, with a pause between
        msg = json.dumps({"path": "/v1/act", "body": act("M1", "split-1", A)}).encode()
        s.sendall(len(msg).to_bytes(4, "big") + msg[:10])
        time.sleep(0.2)
        s.sendall(msg[10:])
        check("frame split across two packets is reassembled",
              read_frame(s)["body"]["action_id"] in ("move_5_5", "end"))

        # 4. two frames in one packet must not be merged
        m1 = json.dumps({"path": "/v1/act", "body": act("M1", "batch-1", A)}).encode()
        m2 = json.dumps({"path": "/v1/act", "body": act("M1", "batch-2", B)}).encode()
        s.sendall(len(m1).to_bytes(4, "big") + m1 + len(m2).to_bytes(4, "big") + m2)
        r1, r2 = read_frame(s), read_frame(s)
        check("two frames in one packet stay two messages",
              r1["body"]["action_id"] == "move_5_5" and r2["body"]["action_id"] == "move_9_9",
              "%s / %s" % (r1["body"]["action_id"], r2["body"]["action_id"]))

        # 1b. same request over both transports must agree
        t = frame(s, "/v1/act", act("M9", "same-1", B))["body"]["action_id"]
        h = http("/v1/act", act("M9", "same-1", B))["action_id"]
        check("TCP and HTTP agree on the same request", t == h, "%s vs %s" % (t, h))

        # 5. the cache must be scoped to the match
        first = frame(s, "/v1/act", act("MATCH-A", "rw-t0-s0-n1", A))["body"]["action_id"]
        second = frame(s, "/v1/act", act("MATCH-B", "rw-t0-s0-n1", B))["body"]["action_id"]
        check("new match, recycled request_id -> not the old answer",
              second in ("move_9_9", "end") and second != "move_5_5",
              "match A gave %s, match B gave %s" % (first, second))

        r = frame(s, "/v1/match/end", {"match_id": "M1", "result": {"won": True}})
        check("match/end over TCP", r["status"] == 200)

        r = frame(s, "/v1/nope", {})
        check("unknown route returns 404, connection survives", r["status"] == 404)
        check("connection still usable after a 404",
              frame(s, "/v1/health", {})["status"] == 200)
        s.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    print("\n%s" % ("ALL PASS" if not fails else "FAILED: " + ", ".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
