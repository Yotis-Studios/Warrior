"""The CLIENT half of the Warrior protocol -- the part a game engine implements.

This is the piece worth copying. Skirmish is a throwaway game; `WarriorSeat` is the reference for
what any engine must do to seat a warrior safely, and porting it (to GML, to Ganbatte, to
anything) is a matter of translating these four responsibilities:

  1. ENUMERATE      hand over the complete legal set, and nothing outside it
  2. REDACT         build the seat's view by whitelist
  3. VALIDATE       treat the reply as untrusted input -- the enum does not bind the model
  4. NEVER STALL    a warrior that cannot answer forfeits its turn; the match continues

(3) is the one that looks paranoid and is not. Measured on a live model: offered a schema enum of
exactly ["end_turn"] with an attack advertised in the prompt, it returned the attack 6 times out
of 6. A returned action_id is a suggestion from a program you did not write.

(4) is the one that gets skipped and should not. Every failure mode here -- a sidecar that is
down, that times out, that returns nonsense three times running, that crashes mid-match -- ends
in the same place: take the safe action, log it, play on. A game that hangs because an agent
misbehaved is a worse outcome than a game that agent played badly.
"""

import json
import time
import urllib.error
import urllib.request

PROTOCOL_VERSION = "0.1"


class SeatStats:
    """What a seat did, and how much of it was the protocol failing rather than the model losing.

    Separated on purpose. "Lost the match" and "never produced a legal action" look identical in a
    win column and mean completely different things -- one is a policy to improve, the other is a
    bug in the harness or the sidecar. A ladder that cannot tell them apart will rank a broken
    warrior as merely a bad one.
    """

    def __init__(self):
        self.actions = 0
        self.rejected = 0        # illegal action_id returned
        self.retries = 0
        self.forfeits = 0        # retry budget exhausted -> forced end_turn
        self.transport_errors = 0
        self.total_latency = 0.0

    def as_dict(self):
        return {
            "actions": self.actions, "rejected": self.rejected, "retries": self.retries,
            "forfeits": self.forfeits, "transport_errors": self.transport_errors,
            "avg_latency_s": round(self.total_latency / self.actions, 2) if self.actions else 0.0,
        }

    def healthy(self):
        """Was this seat actually playing, or merely present?"""
        if not self.actions:
            return False
        return (self.rejected / self.actions) < 0.25 and self.forfeits == 0


class WarriorSeat:
    """One seat, driven by one sidecar."""

    def __init__(self, seat, url, max_retries=3, deadline_ms=30000, timeout=180.0, log=print):
        self.seat = seat
        self.url = url.rstrip("/")
        self.max_retries = max_retries
        self.deadline_ms = deadline_ms
        self.timeout = timeout
        self.log = log
        self.stats = SeatStats()
        self.capabilities = {}
        self.name = url

    # -- transport -----------------------------------------------------------

    def _post(self, path, payload):
        req = urllib.request.Request(
            self.url + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def handshake(self):
        try:
            with urllib.request.urlopen(self.url + "/v1/health", timeout=10) as r:
                h = json.loads(r.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise ConnectionError(f"seat {self.seat}: no sidecar at {self.url} ({exc})") from exc

        their = h.get("protocol_version")
        if their != PROTOCOL_VERSION:
            # Reported, not refused. A minor-version difference is usually survivable and finding
            # out mid-match is worse than a loud line at startup -- but silently proceeding when
            # the shapes genuinely differ is how a bug gets blamed on the model.
            self.log(f"  ! seat {self.seat} sidecar speaks {their!r}, we speak "
                     f"{PROTOCOL_VERSION!r} -- proceeding, but mismatches are on you")
        self.capabilities = h.get("capabilities") or {}
        self.name = f"{h.get('policy', '?')}/{h.get('model') or 'n/a'}"
        return h

    def notify(self, path, payload):
        """Best-effort. A sidecar that ignores lifecycle calls is still a valid sidecar."""
        try:
            self._post(path, payload)
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            self.log(f"  ! seat {self.seat} {path} failed: {exc}")

    # -- the core loop -------------------------------------------------------

    def choose(self, game, request_id, reason, last_action):
        """Ask for one action and return an id that is GUARANTEED to be in the legal set.

        The guarantee is the whole contract. Everything downstream -- game.apply() in particular --
        is allowed to assume the id is legal precisely because this function will not return
        otherwise, and centralising that here is what keeps the check from being re-implemented
        (and forgotten) at each call site.
        """
        actions = game.available_actions()
        legal = {a["action_id"] for a in actions}
        state = game.redacted_state(self.seat)

        attempt = 0
        while attempt <= self.max_retries:
            payload = {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": f"{request_id}-r{attempt}" if attempt else request_id,
                "match_id": game.match_id,
                "seat": self.seat,
                "reason": reason if attempt == 0 else "retry",
                "deadline_ms": self.deadline_ms,
                "state": state,
                "available_actions": actions,
                "last_action": last_action,
                "screenshot": None,
            }

            t0 = time.time()
            try:
                resp = self._post("/v1/act", payload)
            except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
                self.stats.transport_errors += 1
                self.log(f"  ! seat {self.seat} transport error: {exc}")
                return self._forfeit(actions), {}
            dt = time.time() - t0
            self.stats.total_latency += dt

            if dt * 1000 > self.deadline_ms:
                # Enforced here, not merely advertised. A late action is not a slow action: the
                # board it was chosen against may already be gone.
                self.log(f"  ! seat {self.seat} missed its {self.deadline_ms}ms deadline "
                         f"({dt * 1000:.0f}ms)")
                return self._forfeit(actions), {}

            action_id = resp.get("action_id")
            args = resp.get("args") or {}

            if action_id in legal:
                self.stats.actions += 1
                return action_id, args

            # ILLEGAL. Not fuzzy-matched, not repaired, not guessed at. Reconstructing intent from
            # a malformed id is exactly how a warrior ends up taking an action no human could.
            self.stats.rejected += 1
            self.log(f"  ! seat {self.seat} returned {action_id!r}, which was not offered")
            last_action = {
                "action_id": action_id, "ok": False,
                "error": f"{action_id!r} is not in available_actions. "
                         f"Choose one of the action_id values exactly as written.",
            }
            attempt += 1
            if attempt <= self.max_retries:
                self.stats.retries += 1

        return self._forfeit(actions), {}

    def _forfeit(self, actions):
        """Out of retries, or unreachable. Take the safest thing that was actually offered."""
        self.stats.forfeits += 1
        self.stats.actions += 1
        for a in actions:
            if a["type"] == "end_turn":
                return a["action_id"]
        # A list with no end_turn is possible in principle; returning a hardcoded "end_turn"
        # would be the same illegal-action bug this class exists to prevent.
        return actions[0]["action_id"] if actions else "end_turn"


class ScriptedSeat:
    """A seat played by local code rather than a sidecar -- the control, and the CI default.

    Same interface as WarriorSeat, so a match can mix warriors and scripted opponents without the
    runner knowing which is which. That mixing is the point: a benchmark needs a fixed opponent,
    and a regression test needs to run with no model at all.
    """

    def __init__(self, seat, policy="random", rng=None, log=print):
        import random
        self.seat = seat
        self.policy = policy
        self.rng = rng or random.Random(seat)
        self.stats = SeatStats()
        self.capabilities = {}
        self.name = f"scripted/{policy}"
        self.log = log

    def handshake(self):
        return {"protocol_version": PROTOCOL_VERSION, "policy": self.policy}

    def notify(self, path, payload):
        pass

    def choose(self, game, request_id, reason, last_action):
        actions = game.available_actions()
        self.stats.actions += 1
        if self.policy == "aggressive":
            for a in actions:
                if a["type"] == "attack":
                    return a["action_id"], {}
            for a in actions:
                if a["type"] == "move":
                    return a["action_id"], {}
            return "end_turn", {}
        # random, with end_turn weighted down so matches actually play out
        weights = [0.12 if a["type"] == "end_turn" else
                   (0.3 if a["type"] == "chat" else 1.0) for a in actions]
        chosen = self.rng.choices(actions, weights=weights, k=1)[0]
        args = {"message": self.rng.choice(["gg", "hm", "watch out"])} \
            if chosen["type"] == "chat" else {}
        return chosen["action_id"], args
