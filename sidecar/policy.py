"""How a seat decides. One interface, three implementations.

A policy receives the parsed `/v1/act` request and returns `(action_id, args, commentary)`. It is
handed the legal set and is not trusted with it -- `server.py` validates whatever comes back,
because the game validates it too and a sidecar that lets a bad id through just wastes a retry.

RandomPolicy is not a toy. "Do four seats of random legal actions finish a match" is the cheapest
regression gate a game engine can have, it needs no model and no GPU, and it catches the failure
that matters most: a match that stalls forever because some state has no legal action out of it.
"""

import json
import os
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
    # NO STRATEGY SECTION HERE, AND THAT IS A RESULT RATHER THAN AN OVERSIGHT.
    #
    # One lived here: which tier track you are on, that points pay per point per turn so parking on
    # one you already hold wastes the turn, that cover decides fights, that a bad shot is worse than
    # a move. Every line was drawn from a measured failure, and it was measured after:
    #
    #     map           with it      without
    #     Arboretum     0/8   0%     0/8   0%
    #     Islands       1/8  12%     0/8   0%
    #     Crossroads    2/8  25%     8/16 50%      <-- halved
    #     TOTAL         3/24 12%     8/32 25%
    #
    # gpt-5.6-luna went 0/9 on Arboretum under the same text. Nothing improved and the one board
    # the model was good at got worse: told to contest territory, it stopped kill-rushing, which
    # was the only thing that had been working. The knowledge was never the bottleneck -- these
    # models can already read cover and points off the board -- so telling them what matters does
    # not make them able to act on it. The 57k-parameter RL net scores 50% on Arboretum from 33
    # floats and no language at all.
    #
    # If you are about to add strategy advice here, measure it on Crossroads AND a cover board
    # before keeping it. This is the second time an intuition about this prompt has been wrong.
    "When `consult_expert` is offered, it asks a policy trained by reinforcement learning what it "
    "would do in this exact position. Measured against the game's built-in AI on three boards it "
    "wins 69%, 50% and 31% where a fair share is 25% -- so it is strong. What it is strongest at "
    "is territory: on the hardest board it wins while averaging 0.1 kills a match. It is also mute: "
    "it cannot tell you why, hold a plan "
    "across turns, read the chat, or notice anything the board does not say. Its answer is advice, "
    "not an order -- but when it disagrees with you about where to move, it is usually right and "
    "you should have a reason. Consulting is free and does not cost your turn, but you must still "
    "call take_action afterwards.\n"
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
                 max_tokens=2500, thinking=False, timeout=120.0, vision=False,
                 api_key=None, expert=None, persona="", chat_opener=False):
        self.url = url.rstrip("/")
        self.model = model
        # From the environment by default, so a key never has to appear in a command line, a shell
        # history or a process list. warrior-tournament.sh echoes its own invocation.
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or ""
        # Set once a provider has rejected tool_choice="required", so the downgrade is paid for
        # exactly one round trip rather than on every turn of a forty-match tournament.
        self._tool_choice_downgraded = False
        # Held so the consult loop can score the SAME position the model was shown. Re-deriving
        # it from the messages would mean parsing the prompt back into a state, which is a second
        # copy of the renderer waiting to drift.
        self.expert = expert
        self._last_state = {}
        self._last_actions = []
        self._last_expert = None
        self._decisions = []
        # A PERSONA IS PRESENTATION, NEVER PERMISSION. Appended after the rules so it cannot argue
        # with them: a model told to be in character still may not invent an action_id.
        self.persona = (persona or "").strip()
        self.chat_opener = chat_opener
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
        # AN EXPLICIT --model WINS. Asking the endpoint is the right move for a server hosting one
        # model, where the flag can silently go stale. It is the wrong move for a hosted gateway:
        # OpenRouter's /v1/models lists hundreds, and picking the first would report a model the
        # run never touched -- a worse failure than the staleness this was written to prevent,
        # because it looks authoritative.
        if self.model:
            return self.model
        try:
            req = urllib.request.Request(self.url + "/v1/models", headers=self._headers())
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
            # WHAT MAKES THIS RAIFU DIFFERENT. The numbers below say range 20; only this says that
            # 20 is Mosin's +2 and outshoots everyone else on the board. A character is drawn per
            # match and its modifiers are the largest single difference between two seats, so an
            # agent that cannot name its own advantage cannot play to it. Empty for Krag, who has
            # none -- and being told you have no special advantage is also information.
            abilities = me.get("abilities") or []
            if abilities:
                out.append("  your character's abilities: " + "; ".join(str(a) for a in abilities))
            # The doctrine is FLAVOUR, and it is here because it is the only thing in the payload
            # that says what this athlete is *like* rather than what she can do. It costs one line
            # and it is what a persona has to work with.
            if me.get("school") or me.get("doctrine"):
                out.append("  your school: %s -- %s"
                           % (me.get("school") or "?", me.get("doctrine") or ""))
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
        tools = [{"type": "function", "function": {
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

        # A SECOND TOOL, OFFERED ONLY WHEN THERE IS AN EXPERT TO ASK. Advertising a tool that
        # cannot answer teaches a model to call something that fails, and a failed call still
        # costs a round trip and still has to be handled -- worse than not having it.
        if self.expert is not None:
            tools.append({"type": "function", "function": {
                "name": "consult_expert",
                "description": (
                    "Ask the reinforcement-learning expert what it would do in this exact "
                    "position. It returns its top choices and how much of its confidence it puts "
                    "on each. Free, does not cost your turn, and its answer is advice you may "
                    "take or overrule. You must still call take_action afterwards."),
                "parameters": {"type": "object", "properties": {}, "required": []},
            }})
        return tools

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

        # Cached for the consult loop, which has to score the position the model was SHOWN.
        # Re-deriving it would mean parsing the prompt back into a state -- a second copy of the
        # renderer, waiting to drift from the first.
        self._last_state = req.get("state") or {}
        self._last_actions = actions
        self._last_expert = None

        # AN INVITATION, NOT AN INSTRUCTION, and only at the top of your own turn -- nudging on
        # every decision turns a match into a monologue. Gated on `chat` actually being offered,
        # so it can never suggest something the game would reject, and on nothing having happened
        # this turn yet, which is what "the start of your turn" means in a protocol that asks once
        # per action rather than once per turn.
        if self.chat_opener and any(a.get("type") == "chat" for a in actions):
            me = (req.get("state") or {}).get("self") or {}
            if not any((me.get("this_turn") or {}).values()):
                user += ("\n\nIt is the start of your turn and the table can hear you. If you have "
                         "something to say -- a greeting, a threat, a read on somebody, a complaint "
                         "about the dice -- say it now with the chat action. It costs you nothing "
                         "and you will be asked to act again straight after. If you have nothing "
                         "worth saying, just act.")

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
                {"role": "system", "content": self._system()},
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

        # THE CONSULT LOOP. The protocol is one action per request, so a consult cannot be folded
        # into the answer -- the model asks, reads, then decides. Bounded at two: a model that
        # keeps consulting instead of acting is stuck, and the game is holding its turn open the
        # whole time. Cheap to allow because Raifu Wars has no per-action clock, which is the same
        # property that lets a model take thirty seconds to think in the first place.
        for _ in range(2):
            payload = self._post(body)
            consult = self._expert_call(payload)
            if consult is None:
                break
            body["messages"].append(consult["assistant"])
            body["messages"].append(consult["tool_result"])
            # `auto`, not `required`, on the follow-up: the model has what it asked for and must
            # now be free to answer rather than being pushed into another tool call.
            body["tool_choice"] = "auto" if self._tool_choice_downgraded else "required"

        return self._parse(payload, actions)

    def _expert_call(self, payload):
        """If the reply is a consult, run the expert and build the two messages that answer it."""
        if self.expert is None:
            return None
        choice = (payload.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        for call in (msg.get("tool_calls") or []):
            if (call.get("function") or {}).get("name") != "consult_expert":
                continue
            ranked = self.expert.rank(self._last_state, self._last_actions, k=5)
            self._last_expert = ranked
            return {
                "assistant": {"role": "assistant", "content": msg.get("content"),
                              "tool_calls": msg.get("tool_calls")},
                "tool_result": {"role": "tool", "tool_call_id": call.get("id"),
                                "name": "consult_expert",
                                "content": self.expert.render(ranked)},
            }
        return None

    def _system(self):
        if not self.persona:
            return SYSTEM_PROMPT
        return (SYSTEM_PROMPT + "\n\nHOW YOU CARRY YOURSELF. " + self.persona
                + " This is how you TALK, not how you play: it changes your chat messages and "
                  "your `why` notes, and it changes nothing about which action is correct. You "
                  "still may not invent an action_id, and you still play to win.")

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = "Bearer " + self.api_key
            # OpenRouter attributes requests by these and shows them on the account's activity
            # page. Harmless anywhere else -- a local llama-server ignores unknown headers.
            h["HTTP-Referer"] = "https://github.com/yotisstudios/Warrior"
            h["X-Title"] = "Warrior protocol"
        return h

    def _post(self, body):
        # RETRIED ON RATE LIMIT AND ON A GATEWAY HICCUP, which a local server never needed. A
        # hosted endpoint answers 429 under load and 502/503 when a provider behind it drops, and
        # the game's deadline is 30 seconds -- so a single unlucky request would otherwise be
        # scored as the model failing to answer. Bounded and short: three tries, and the backoff
        # has to fit inside the deadline or retrying is just a slower way to miss it.
        #
        # Nothing else is retried. A 400 is a malformed request and will be malformed next time.
        last = None
        for attempt in range(3):
            req = urllib.request.Request(
                self.url + "/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers=self._headers(),
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                if exc.code == 400 and not self._tool_choice_downgraded:
                    peek = exc.read().decode("utf-8", "replace")
                    if "tool_choice" in peek:
                        # See eval_sft.py: some providers reject "required" outright. Downgrade
                        # once, remember it for the rest of the process so every later turn does
                        # not pay a wasted round trip, and say so -- a run served under a weaker
                        # constraint than the one requested is a fact about the run.
                        self._tool_choice_downgraded = True
                        body["tool_choice"] = "auto"
                        print("[warrior] provider refused tool_choice=required; using auto",
                              flush=True)
                        continue
                    raise RuntimeError("HTTP 400 from %s: %s" % (self.url, peek[:300]))
                if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                    # The body carries the provider's actual complaint -- "no endpoints found for
                    # model", "tool use not supported" -- and losing it turns a five-second fix
                    # into a debugging session.
                    detail = exc.read().decode("utf-8", "replace")[:300]
                    raise RuntimeError("HTTP %s from %s: %s" % (exc.code, self.url, detail))
                last = exc
                time.sleep(0.6 * (attempt + 1))
        raise RuntimeError("giving up after retries: %s" % last)

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

        # DID IT LISTEN? Recorded per decision, because "the LLM consults the expert" is worth
        # nothing on its own -- the interesting quantities are whether it asks at all, whether it
        # follows when it does, and whether the matches it OVERRODE in went better or worse than
        # the ones it deferred in.
        #
        # The control is the expert playing alone, which wins ~55%. An LLM on top of it has to
        # beat that or it is overhead, and a model that overrides good advice will score BELOW the
        # expert it is holding. That is the measurement this line exists for.
        if self.expert is not None and self._last_expert:
            top = self._last_expert[0]
            followed = str(action_id) == top["action_id"]
            self.expert.overruled += 0 if followed else 1
            self._decisions.append({
                "followed": followed,
                "expert_top": top["action_id"],
                "expert_share": top["share"],
                "chose": str(action_id),
                # Confidence matters more than agreement: overriding a 95% call is a different act
                # from overriding a 30% one, and averaging them together hides both.
                "confident": top["share"] >= 0.8,
            })

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
            api_key=kw.get("api_key"),
            expert=kw.get("expert"),
            persona=kw.get("persona", ""),
            chat_opener=kw.get("chat_opener", False),
        )
    raise ValueError(f"unknown policy {kind!r}")
