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


# THE TRAINED SYSTEM PROMPT, BYTE FOR BYTE, pasted from messages[0] of
# yotisstudios/Warrior-SFT-v2. Not a tidied version of the same words -- an earlier upload had the
# paragraph breaks collapsed (1,932 chars against 2,166) and serving the readable text would have
# differed from training on every single request. selftest asserts the hash below, so this file
# cannot drift from the data without the suite going red.
#
# The `justification` sentence was removed from BOTH sides together: the dataset carries no
# justifications in any of its 5,423 rows, and an earlier revision of this prompt still demanded
# one -- which trained the model, on every example, to violate a requirement stated in its own
# system prompt. take_action still ACCEPTS a justification; nothing now claims it is required.
#
# To change it: change the DATASET first, retrain, then paste the new string here.
SYSTEM_PROMPT = (
    "You are playing one seat in a turn-based game, against other players.\n\nEach time it is "
    "your turn you are given: a briefing on the game's rules, the current state of the board "
    "as your seat can see it, and the COMPLETE list of actions you are allowed to take right "
    "now.\n\nHow to act:\n- Call the take_action tool exactly once.\n- action_id must be copied "
    "EXACTLY from the legal action list. Do not invent one, do not adjust one, do not combine "
    "two.\n- If what you want to do is not on the list, you may not do it this turn. The list "
    "is complete; anything missing from it is forbidden rather than forgotten.\n- Numbers you "
    "are given -- hit chances, distances, costs -- are computed by the game. Trust them and "
    "do not recalculate them.\n- You take ONE action at a time. After it resolves you will be "
    "asked again with an updated board, so plan for the next action rather than the whole "
    "turn.\n\nYou can only see what your seat is entitled to see. Other players' hidden "
    "information is withheld deliberately -- reason about what they are likely to hold, and "
    "do not assume you know it.\n\nYou may TALK. When a `chat` action is offered, taking it "
    "with a `message` says that line to every player at the table, and it does NOT cost your "
    "turn -- you will be asked to act again immediately. One line, under 100 "
    "characters.\n\nAnswer anyone who speaks to you, react to what actually happened, and do "
    "not narrate your own move -- the table can see the board.\n\nLegal actions are provided "
    "for you as tool calls below. Make decisions based on your own judgement of the current "
    "state of gameplay which is provided in this context. Another AI model provides you with "
    "recommended actions; generally, you should follow the highest probability one for "
    "optimal success but again you are free to use your own discretion.\n\ntake_action also "
    "accepts an optional `notes` field that replaces a private scratchpad carried for the "
    "rest of the match; only you can see it.\n\nPlay to win."
)
SYSTEM_PROMPT_SHA16 = "3260db8dac9c3f79"


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
        # A PERSONA IS PRESENTATION, NEVER PERMISSION. Appended after the rules so it cannot argue
        # with them: a model told to be in character still may not invent an action_id.
        self.persona = (persona or "").strip()
        self.chat_opener = chat_opener
        # Bounded context. Both grow without limit over a 150-turn match otherwise.
        self.chat_window = 10
        self.history_window = 8
        self._reasons = []          # (action_id, justification) this match
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

    def _render_actions(self, actions, advice=None):
        """The legal set, optionally annotated with the RL policy's own probability mass.

        INLINE, NOT A TOOL CALL. The advice used to arrive through a `consult_expert` tool the
        model could choose to invoke, which cost a SECOND round trip per decision -- 6s instead of
        3s, and 14 minutes a match -- and only helped on the turns the model remembered to ask.
        Rendered here it is unconditional and free.

        This is also the format a fine-tune will be trained on, so it has to be produced by this
        function and not a copy in the dataset generator. A model trained on a layout it is never
        served is a model trained on nothing.
        """
        lines = ["LEGAL ACTIONS -- you may ONLY choose one of these action_id values:"]
        for a in actions:
            bits = []
            if a.get("hit_chance") is not None:
                bits.append(f"hit chance {round(a['hit_chance'] * 100)}%")
            if a.get("expected_damage") is not None:
                bits.append(f"~{a['expected_damage']} damage")
            if a.get("note"):
                bits.append(a["note"])
            if advice is not None:
                # FIRST in the bracket, not last. It is the single most useful number on the line
                # and the notes can run long. Shown for every action the expert gives any mass to;
                # below half a percent it is rounding noise dressed as a recommendation.
                share = advice.get(a["action_id"])
                if share is not None and share >= 0.005:
                    bits.insert(0, f"expert {round(100 * share)}%")
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
        # NO DECISION HISTORY. It rendered the seat's own past choices with the justification it
        # gave for each, which is how a plan survived more than one action -- and it is disabled
        # because the v2 training data contains no justifications at all. Serving a section the
        # model never saw in training is the mismatch this file keeps paying for, and with the
        # reasons gone the section degrades to a bare list of ids the state already implies.
        #
        # Turn it back on in the same commit that puts justifications back in the DATASET, not
        # before.
        if False:                                                   # noqa: SIM108 -- see above
            pass

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

        # PRUNED TO THE LAST FEW. A long match accumulates chat without bound, and it is the one
        # section that grows with nothing to cap it -- the board is fixed size and the action list
        # is bounded by the rules. Old table talk is also the least useful thing in the prompt: a
        # line from forty turns ago is not what anybody is answering.
        chat = state.get("chat") or []
        if chat:
            out.append("\nTABLE CHAT")
            shown = chat[-getattr(self, "chat_window", 10):]
            if len(chat) > len(shown):
                out.append("  (%d earlier lines not shown)" % (len(chat) - len(shown)))
            for c in shown:
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
                    # OPTIONAL, and it was required until a fine-tune was measured against it.
                    # Requiring it doubles what the model must emit correctly on every decision,
                    # and a 4B trained on this format failed to produce a tool call at all on 36%
                    # of decisions -- emitting the learned justification phrasing as prose instead.
                    # It is still worth asking for: it feeds the seat's own decision history back
                    # next turn, it is what a human reads when the seat does something surprising,
                    # and it is what a fine-tune learns to produce. Asked for, not demanded.
                    "justification": {"type": "string",
                                      "description": "one short line on why you chose this "
                                                     "action. You will see it again as your own "
                                                     "decision history on later turns."},
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

        # ONE TOOL. There used to be a second, `consult_expert`, which the model could call to ask
        # the RL policy what it would do. It is gone because the answer is now rendered inline on
        # every action: the tool cost a second round trip per decision (6s instead of 3s, and 14
        # minutes a match) and only helped on the turns the model chose to ask.
        return tools

    # -- the call ------------------------------------------------------------

    def act(self, req):
        actions = req.get("available_actions") or []
        if not actions:
            return "end_turn", {}, None

        state = req.get("state") or {}

        # THE EXPERT'S WHOLE DISTRIBUTION, INLINE. Not a top-k and not a tool call: the model is
        # choosing among exactly these actions, so it sees what the policy thinks of each.
        #
        # CHAT IS EXCLUDED. The net has never seen a chat action -- the game only began offering
        # one offline recently -- so any mass it puts there is an artefact of features that do not
        # describe talking, and presenting it would be inventing an opinion the net does not hold.
        # In the hybrid seat the same leak hung a match outright: 58,317 chat actions, because chat
        # does not consume the turn, so once the net rated it top it rated it top forever.
        advice = None
        if self.expert is not None:
            try:
                scorable = [a for a in actions if a.get("type") != "chat"]
                if scorable:
                    ranked = self.expert.rank(state, scorable, k=len(scorable))
                    advice = {r["action_id"]: r["share"] for r in ranked}
            except Exception as exc:                                # noqa: BLE001
                # A turn without advice is still a playable turn. Saying so beats a silent
                # fallback that looks like the expert having no opinion.
                print("[warrior] expert unavailable this turn: %s" % exc, flush=True)

        user = self._render_state(state)
        user += "\n\n" + self._render_actions(actions, advice)

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

        # ONE ROUND TRIP PER DECISION. There used to be a consult loop here: the model called
        # `consult_expert`, read the answer, then decided -- two calls, so 6s a decision instead of
        # 3s and 14 minutes a match, on a protocol that already asks ~90 decisions per match. The
        # expert's whole distribution is rendered inline on the action list now, so the model has
        # the same information before it answers and there is nothing left to ask for.
        payload = self._post(body)
        return self._parse(payload, actions)

    def match_start(self, req):
        """A new match is a new memory. Decisions from the last one are about a different board."""
        self._reasons = []

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

    def complete(self, system, user, max_tokens=120, temperature=None):
        """A plain completion -- no tools, no action list. Used by the hybrid seat for chat.

        Shares _post, so retries, the tool_choice downgrade and auth behave identically to a real
        decision. A separate HTTP path here would be a second client to keep in step with the
        first, which is the shape of bug this codebase already has several of.
        """
        body = {
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if self.model:
            body["model"] = self.model
        # THINKING OFF, same as the decision path. Omitting this is not a small difference for a
        # one-line reply: the model spends the whole budget reasoning and returns EMPTY content,
        # which reads downstream as "the model had nothing to say" rather than as a broken call.
        # A hybrid seat went a full match without speaking for exactly this reason, with no error
        # anywhere -- the compose path treats failure as silence by design.
        if not getattr(self, "thinking", False):
            body["chat_template_kwargs"] = {"enable_thinking": False}
        data = self._post(body)
        if not data:
            return ""
        choice = (data.get("choices") or [{}])[0]
        return (choice.get("message") or {}).get("content") or ""

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
        # The game stores this as `why`; the tool calls it `justification`, which is what a
        # model understands without being told. Mapped here rather than renaming the wire
        # field, which every recorded trace and every existing dataset row already uses.
        if args.get("justification"):
            out["why"] = args["justification"]
        # KEPT HERE, because the game does not send it back. `your_recent_actions` carries ids
        # only, so without this the seat's own stated reasoning is lost the moment it is given.
        if not hasattr(self, "_reasons"):
            self._reasons = []
        self._reasons.append((action_id, args.get("justification") or ""))
        if args.get("notes"):
            out["notes"] = args["notes"]
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
    if kind == "hybrid":
        # THE RL NET TAKES THE SEAT; the language model is optional and only writes chat. With no
        # endpoint configured the seat plays and stays quiet, which is useful in its own right --
        # it is the strongest opponent this project has, at no API cost.
        try:
            from .hybrid import HybridPolicy
        except ImportError:
            from hybrid import HybridPolicy
        expert = kw.get("expert")
        if expert is None:
            raise ValueError("--policy hybrid needs --expert CHECKPOINT")
        # A TALKER IS BUILT FOR A LOCAL URL TOO, not only for a hosted model name. Gating on
        # --openrouter/--model meant pointing the seat at a local endpoint produced a silent
        # warrior with no error anywhere -- it played perfectly and never said a word.
        talker = None
        if kw.get("model") or kw.get("openrouter") or kw.get("url"):
            talker = LLMPolicy(
                url=kw.get("url", "http://127.0.0.1:8080"),
                model=kw.get("model"),
                temperature=kw.get("temperature", 0.9),
                max_tokens=200,
                api_key=kw.get("api_key"),
            )
        return HybridPolicy(expert=expert, llm=talker,
                            skill=kw.get("skill", 1.0), seed=kw.get("seed"))
    raise ValueError(f"unknown policy {kind!r}")
