"""How a seat decides. One interface, three implementations.

A policy receives the parsed `/v1/act` request and returns `(action_id, args, commentary)`. It is
handed the legal set and is not trusted with it -- `server.py` validates whatever comes back,
because the game validates it too and a sidecar that lets a bad id through just wastes a retry.

RandomPolicy is not a toy. "Do four seats of random legal actions finish a match" is the cheapest
regression gate a game engine can have, it needs no model and no GPU, and it catches the failure
that matters most: a match that stalls forever because some state has no legal action out of it.
"""

import json
import random
import time
import urllib.error
import urllib.request


class Policy:
    name = "policy"

    def capabilities(self):
        return {"vision": False, "chat": False, "commentary": False}

    def act(self, req):
        """-> (action_id, args_dict, commentary_or_None)"""
        raise NotImplementedError

    def match_start(self, req):
        pass

    def match_end(self, req):
        pass

    def event(self, req):
        pass


class RandomPolicy(Policy):
    """Uniform over the legal set, with one thumb on the scale.

    Pure uniform ends turns almost immediately -- `end_turn` is legal at nearly every decision
    point, so a uniform walk takes it about as often as anything else and the resulting "match"
    exercises almost no game logic. Weighting it down means matches actually play out while
    remaining trivially legal by construction.
    """

    name = "random"

    def __init__(self, seed=None, end_turn_weight=0.15):
        self.rng = random.Random(seed)
        self.end_turn_weight = end_turn_weight

    def act(self, req):
        actions = req.get("available_actions") or []
        if not actions:
            return "end_turn", {}, None
        weights = [self.end_turn_weight if a.get("type") == "end_turn" else 1.0 for a in actions]
        chosen = self.rng.choices(actions, weights=weights, k=1)[0]
        args = {}
        if chosen.get("type") == "chat":
            args["message"] = self.rng.choice(
                ["gg", "nice shot", "hm.", "watch the left", "truce?"])
        return chosen["action_id"], args, None


class FirstLegalPolicy(Policy):
    """The most boring possible opponent: take the first offered action that is not ending.

    Deterministic, so it is the control a stochastic policy is measured against.
    """

    name = "first-legal"

    def act(self, req):
        actions = req.get("available_actions") or []
        for a in actions:
            if a.get("type") != "end_turn":
                return a["action_id"], {}, None
        return "end_turn", {}, None


# DELIBERATELY GAME-AGNOSTIC. This describes the PROTOCOL -- how a turn is asked and answered --
# and says nothing about any particular game. Whatever the model needs to know about the game it
# is playing arrives in `state.briefing`, written by the game itself.
#
# The split matters. Rules in a sidecar's system prompt would be a second copy that nobody updates
# when the game changes, and a sidecar is meant to be pointed at any game that speaks the
# protocol. It is also the same principle the protocol is built on: the rulebook lives in exactly
# one implementation, and that implementation is the engine.
SYSTEM_PROMPT = (
    "You are playing one seat in a turn-based game, against other players.\n"
    "\n"
    "Each time it is your turn you are given: a briefing on the game's rules, the current state "
    "of the board as your seat can see it, and the COMPLETE list of actions you are allowed to "
    "take right now.\n"
    "\n"
    "How to act:\n"
    "- Call the take_action tool exactly once.\n"
    "- action_id must be copied EXACTLY from the legal action list. Do not invent one, do not "
    "adjust one, do not combine two.\n"
    "- If what you want to do is not on the list, you may not do it this turn. The list is "
    "complete; anything missing from it is forbidden rather than forgotten.\n"
    "- Numbers you are given -- hit chances, distances, costs -- are computed by the game. Trust "
    "them and do not recalculate them.\n"
    "- You take ONE action at a time. After it resolves you will be asked again with an updated "
    "board, so plan for the next action rather than the whole turn.\n"
    "\n"
    "You can only see what your seat is entitled to see. Other players' hidden information is "
    "withheld deliberately -- reason about what they are likely to hold, and do not assume you "
    "know it.\n"
    "\n"
    "You have a memory. take_action accepts an optional `why` -- one line on what you are "
    "doing and why -- which you will see again on your next few actions, and an optional "
    "`notes` field that replaces private notes carried for the rest of the match. Nothing "
    "else remembers your intentions between actions, so a plan that takes more than one "
    "action only survives if you write it down.\n"
    "\n"
    "Play to win."
)


class LLMPolicy(Policy):
    """An OpenAI-compatible chat-completions endpoint, driven with a single tool.

    The single-tool encoding is not a style preference. Probing a 27B with one tool per action
    type -- attack/rush/reload/play_card, with typed arguments -- produced illegal actions in
    roughly one call in eight: a target seat that did not exist, and on another run the model's
    own seat. Collapsing every rules-affecting choice into an enumerated action_id scored 16/16.
    See PROTOCOL.md section 0 and probe/.
    """

    name = "llm"

    def __init__(self, url="http://127.0.0.1:8080", model=None, temperature=0.7,
                 max_tokens=2500, thinking=False, timeout=120.0, vision=False):
        self.url = url.rstrip("/")
        self.model = model
        self.temperature = temperature
        # GENEROUS ON PURPOSE. At 900 this model returned no tool call at all in half of samples
        # -- it was still reasoning when the cap hit, and a truncated reply is indistinguishable
        # from a refusal once parsed. At 2500, zero truncations.
        self.max_tokens = max_tokens
        # OFF BY DEFAULT: measured at 18.8s and 861 completion tokens with reasoning on, versus
        # 0.9s and 37 without -- and legality did not suffer. Opt in per seat when a harder
        # decision is worth twenty times the wall clock.
        self.thinking = thinking
        self.timeout = timeout
        self.vision = vision

    def capabilities(self):
        return {"vision": self.vision, "chat": True, "commentary": True}

    def loaded_model(self):
        """What the inference server ACTUALLY has loaded, asked rather than remembered.

        `--model` is a routing hint, and a hint is a copy: swap the model behind llama-server and
        the flag keeps naming the old one, while inference quietly runs on the new. That is not a
        cosmetic mislabel. The harness stamps this string into every result line and every trace,
        so a run would carry the wrong provenance -- and provenance is not recoverable afterwards,
        which is the entire reason it is recorded.

        Caught in exactly that state: the sidecar reported Wichtel-Q4_K_M.gguf for a full session
        after a Qwen3-VL-4B had replaced it behind the same port.

        Falls back to the flag when the server will not say. Unreachable is not the same as
        unknown, so the two are distinguishable rather than both becoming None.
        """
        try:
            req = urllib.request.Request(self.url + "/v1/models")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                d = json.loads(resp.read().decode("utf-8"))
            entries = d.get("data") or d.get("models") or []
            if entries:
                name = entries[0].get("id") or entries[0].get("model") or entries[0].get("name")
                if name:
                    # llama-server reports the path it was launched with; the basename is what a
                    # human recognises and what the old flag used to carry.
                    return str(name).replace("\\", "/").rsplit("/", 1)[-1]
        except Exception:                                       # noqa: BLE001
            pass
        return self.model or "<unknown>"

    # -- prompt construction -------------------------------------------------

    def _render_actions(self, actions):
        lines = ["LEGAL ACTIONS -- you may ONLY choose one of these action_id values:"]
        for a in actions:
            bits = []
            if a.get("hit_chance") is not None:
                bits.append(f"hit chance {round(a['hit_chance'] * 100)}%")
            if a.get("expected_damage") is not None:
                bits.append(f"~{a['expected_damage']} damage")
            if a.get("note"):
                bits.append(a["note"])
            suffix = f"  [{'; '.join(bits)}]" if bits else ""
            label = a.get("label") or a.get("type") or a["action_id"]
            lines.append(f"  {a['action_id']}: {label}{suffix}")
        return "\n".join(lines)

    @staticmethod
    def _num(v):
        """5.0 -> 5, 0.45 -> 0.45.

        GML's json_stringify does not preserve the integer/real distinction, so EVERY number
        arrives as a float: "TURN 5.0", "seat 1.0", "health 5.0", "need 25.0". That is noise on
        every number in the payload, on every request, and it makes a board of small integers read
        like sensor telemetry. Cosmetic, and cosmetics are most of what a prompt is.
        """
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)

    @classmethod
    def _tile(cls, t):
        if isinstance(t, (list, tuple)) and len(t) >= 2:
            return "(%s,%s)" % (cls._num(t[0]), cls._num(t[1]))
        return str(t)

    def _render_state(self, state):
        """Text, in a FIXED ORDER, with no raw JSON.

        Two things this gets right that dumping the structs did not.

        ORDER IS FIXED HERE, because it is not fixed in the payload. GML struct key order is not
        stable through json_stringify, so a dumped object put `is_fortified` first one turn and
        `tile` first the next -- the model saw a differently-shaped description of the same
        situation every turn, which is precisely the opposite of what helps it pattern-match.

        AND IT IS PROSE. The board section was already laid out and reads well; the rest was
        json.dumps of the same structs, which is denser, uglier and costs more tokens for less.
        A model asked to weigh "should I close on that point" should not first have to parse
        {"kills": 0.0, "stars": 2.0, "name": ...}.

        Anything the client did not send simply does not appear. Deliberately never defaulted: a
        fabricated zero reads to the model as a fact about the game.
        """
        n = self._num
        out = []

        if state.get("briefing"):
            out.append(str(state["briefing"]))
            out.append("")
        if state.get("turn") is not None:
            out.append("=== TURN %s ===" % n(state["turn"]))

        board = state.get("board") or {}
        if board.get("ascii"):
            out.append("\nBOARD")
            out.append(board["ascii"])
        if board.get("legend"):
            out.append("LEGEND: " + "; ".join("%s = %s" % (k, v)
                                              for k, v in board["legend"].items()))

        notes = (state.get("your_notes") or "").strip()
        if notes:
            out.append("\nYOUR NOTES (written by you, only you can see them)")
            out.append("  " + notes.replace("\n", "\n  "))
        hist = state.get("your_recent_actions") or []
        if hist:
            out.append("\nYOUR LAST FEW ACTIONS")
            for h in hist:
                out.append("  - %s" % h)

        me = state.get("self") or {}
        if me:
            out.append("\nYOU: %s at %s" % (me.get("name", "?"), self._tile(me.get("tile"))))
            out.append("  health %s/%s   ammo %s/%s   range %s%s"
                       % (n(me.get("health")), n(me.get("max_health")),
                          n(me.get("ammo")), n(me.get("max_ammo")), n(me.get("range")),
                          "   FORTIFIED" if me.get("is_fortified") else ""))
            out.append("  %s stars, tier %s, %s kills, %s deaths"
                       % (n(me.get("stars")), n(me.get("tier")),
                          n(me.get("kills")), n(me.get("deaths"))))
            tp = me.get("tier_progress") or {}
            if tp:
                out.append("  TIER PROGRESS: %s of %s %s needed to reach tier %s"
                           % (n(tp.get("have")), n(tp.get("need")),
                              tp.get("condition"), n(tp.get("next_tier"))))
            if me.get("in_friendly_territory") is not None:
                out.append("  standing in friendly territory: %s"
                           % n(me.get("in_friendly_territory")))
            tt = me.get("this_turn") or {}
            if tt:
                # Named in the order a turn actually happens, not whatever order the struct
                # arrived in -- moved, then the combat action, then the once-per-turn extras.
                order = ["moved", "attacked", "rushed", "played_card", "tiered"]
                spent = [k for k in order if tt.get(k)]
                out.append("  ALREADY DONE THIS TURN: %s"
                           % (", ".join(spent) if spent else "nothing yet"))
            hand = me.get("hand") or []
            if hand:
                out.append("  YOUR HAND:")
                for c in hand:
                    out.append("    %s (%s stars, needs tier %s) -- %s"
                               % (c.get("name"), n(c.get("cost_stars")),
                                  n(c.get("tier_required")), c.get("text", "")))
            else:
                out.append("  YOUR HAND: empty")

        others = [p for p in (state.get("players") or []) if not p.get("is_self")]
        if others:
            out.append("\nOTHER PLAYERS")
            for p in others:
                out.append("  %s at %s -- %s health, tier %s, %s stars, %s kills, "
                           "%s cards in hand%s%s"
                           % (p.get("name"), self._tile(p.get("tile")), n(p.get("health")),
                              n(p.get("tier")), n(p.get("stars")), n(p.get("kills")),
                              n(p.get("cards_in_hand")),
                              ", FORTIFIED" if p.get("is_fortified") else "",
                              "" if p.get("status") == "alive" else ", %s" % p.get("status")))
            out.append("  (you cannot see which cards they hold, only how many)")

        points = board.get("points") or []
        if points:
            out.append("\nCAPTURE POINTS")
            for pt in points:
                who = pt.get("relationship", "?")
                held = pt.get("held_by_seat")
                if held is not None and n(held) != "-1":
                    who += " (seat %s)" % n(held)
                out.append("  %s -- %s%s" % (self._tile(pt.get("tile")), who,
                                             "  [a start base]" if pt.get("is_base") else ""))

        ev = state.get("events_since_last_turn") or []
        if ev:
            out.append("\nSINCE YOUR LAST TURN")
            for e in ev:
                out.append("  - %s" % e)

        chat = state.get("chat") or []
        if chat:
            out.append("\nTABLE CHAT")
            for c in chat:
                out.append("  seat %s: %s" % (n(c.get("seat")), c.get("text")))

        return "\n".join(out)

    def _tools(self, actions):
        ids = sorted({a["action_id"] for a in actions})
        return [{"type": "function", "function": {
            "name": "take_action",
            "description": ("Take exactly one of the legal actions offered this turn. "
                            "action_id must be copied exactly from the legal action list."),
            "parameters": {
                "type": "object",
                "properties": {
                    # The enum is DEFENCE IN DEPTH, not a guarantee. Offered an enum of exactly
                    # ["end_turn"] with an attack advertised in the prompt, this server returned
                    # the attack 6 times out of 6 -- llama.cpp did not compile it into a sampling
                    # grammar. server.py validates regardless, and so does the game.
                    "action_id": {"type": "string", "enum": ids,
                                  "description": "the action_id you choose"},
                    "message": {"type": "string",
                                "description": "what to say -- only for a chat action"},
                    # WORKING MEMORY, and it rides on the action rather than needing a tool of
                    # its own. A separate update_notes tool would cost a whole extra round trip
                    # per thought, at which point remembering something is more expensive than
                    # doing something -- and the note is most accurate at the moment of decision
                    # anyway.
                    "why": {"type": "string",
                            "description": "one short line on why you chose this, so you can "
                                           "remember your intention on your next action"},
                    "notes": {"type": "string",
                              "description": "replace your private notes. They persist for the "
                                             "rest of the match and only you can see them. Use "
                                             "them for plans that take several actions, or "
                                             "anything you want to remember about opponents. "
                                             "Omit this to leave your notes unchanged."},
                },
                "required": ["action_id"],
            },
        }}]

    # -- the call ------------------------------------------------------------

    def act(self, req):
        actions = req.get("available_actions") or []
        if not actions:
            return "end_turn", {}, None

        user = self._render_state(req.get("state") or {})
        user += "\n\n" + self._render_actions(actions)

        last = req.get("last_action")
        if last and last.get("error"):
            # A retry says WHY, or the model has no reason to choose differently and will
            # cheerfully repeat itself until the retry budget is gone.
            user += (f"\n\nYour previous choice {last.get('action_id')!r} was REJECTED: "
                     f"{last['error']}. Choose an action_id from the list above, exactly as written.")
        elif last and last.get("result"):
            user += f"\n\nYour previous action resolved: {last['result']}"

        # A HINT, WHEN THE GAME SENT ONE. Placed last, right before the instruction to act, and
        # attributed to the built-in AI rather than stated as fact -- the model should be able to
        # disagree with it. Read from the request, not the state: a hint is about the decision,
        # not about the board.
        #
        # The game records hint_level in its own stats line and in every dumped trace, because a
        # hint contaminates the measurement it is mixed into and a run whose hint level is unknown
        # cannot be compared to anything.
        hint = (req.get("hint") or "").strip()
        if hint:
            user += "\n\nHINT: " + hint

        content = user + "\n\nTake one action now."
        shot = req.get("screenshot")
        if self.vision and shot and shot.get("b64"):
            message_content = [
                {"type": "text", "text": content},
                {"type": "image_url",
                 "image_url": {"url": f"data:{shot.get('mime', 'image/png')};base64,{shot['b64']}"}},
            ]
        else:
            message_content = content

        body = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message_content},
            ],
            "tools": self._tools(actions),
            "tool_choice": "required",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.model:
            body["model"] = self.model
        if not self.thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}

        payload = self._post(body)
        return self._parse(payload, actions)

    def _post(self, body):
        req = urllib.request.Request(
            self.url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _parse(self, payload, actions):
        choice = (payload.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        calls = msg.get("tool_calls") or []
        commentary = msg.get("reasoning_content") or None

        if not calls:
            # Say which it was. `length` means we truncated it and the fix is ours; `stop` means
            # the model declined and the fix is the prompt. Collapsing the two costs an afternoon.
            raise PolicyError(
                f"no tool call (finish_reason={choice.get('finish_reason')!r}); "
                f"{len(commentary or '')} chars of reasoning")

        fn = calls[0].get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError as exc:
            raise PolicyError(f"tool arguments were not JSON: {exc}") from exc

        action_id = args.get("action_id")
        if not action_id:
            raise PolicyError("tool call carried no action_id")

        out = {}
        if args.get("message"):
            out["message"] = args["message"]
        # Passed through untouched. server.py forwards them to the game, which stores them
        # verbatim -- the moment anything here edits them they stop being the model's memory.
        for k in ("why", "notes"):
            if args.get(k):
                out[k] = args[k]
        return action_id, out, commentary


class PolicyError(RuntimeError):
    """The policy could not produce an action. Distinct from producing an illegal one."""


def build_policy(kind, **kw):
    if kind == "random":
        return RandomPolicy(seed=kw.get("seed"))
    if kind == "first-legal":
        return FirstLegalPolicy()
    if kind == "llm":
        return LLMPolicy(
            url=kw.get("url", "http://127.0.0.1:8080"),
            model=kw.get("model"),
            temperature=kw.get("temperature", 0.7),
            max_tokens=kw.get("max_tokens", 2500),
            thinking=kw.get("thinking", False),
            vision=kw.get("vision", False),
        )
    raise ValueError(f"unknown policy {kind!r}")
