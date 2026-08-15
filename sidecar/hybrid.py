"""The hybrid seat: the RL policy plays, a language model does the talking.

    python sidecar/server.py --policy hybrid --expert best.pt --openrouter <model>

WHY THIS SHAPE, MEASURED RATHER THAN ASSUMED.

The RL net wins 69% / 50% / 31% across Crossroads, Arboretum and Islands, where a fair share for
one seat of four is 25%. Frontier language models win 71% on Crossroads -- the one board in the
pool with NO cover on it, where parking and kill-rushing works -- and 8% and 0% on the two that
have any. A prompt that explained the win condition made that WORSE, not better (3/24 against
8/32), because the knowledge was never the bottleneck. So the net plays.

IT ALSO FIXES THE THING THAT MADE AN LLM SEAT UNUSABLE AGAINST A HUMAN. The protocol asks one
decision per ACTION, so a match is ~90 round trips. At 3s a call that is 5-9 minutes of a person
watching a spinner, and it was 14 minutes a match with an advisor consult in the loop. The net
answers in about a millisecond; the model is asked only when there is something to SAY, which is a
handful of times a match.

WHAT IT IS FOR. Not maximising the win rate. This project's own AI notes already record that an
EV-maximising opponent can be strictly better at winning and worse to play against, because risking
a shot and missing is a large part of the fun. Hence `skill`: the seat plays from the net's own
probability mass at a temperature, so it can be a good opponent rather than a perfect one.
"""

import random

try:
    from .policy import Policy
except ImportError:                                                 # run as a plain script
    from policy import Policy


# NO INVENTED WORLD DETAIL. An earlier version described the sport, gave each character a school
# and a doctrine, and told the model to speak in that voice -- all of it taken from a lore PROPOSAL
# document. Nothing in that repo is canon, so this was a fiction being asserted to every model on
# every request as though the game had settled it. What a seat is actually entitled to know about
# itself comes from the payload, written by the game.
CHAT_SYSTEM = (
    "You are playing one seat in a turn-based game against other players. You are writing ONE "
    "short line of table chat.\n"
    "\n"
    "Tone: competitive but good-natured. Warm and a bit playful with your opponents, never "
    "vicious. PG-13.\n"
    "\n"
    "Rules for the line:\n"
    "- ONE line, under 100 characters. No surrounding quotes, no name prefix, no emoji spam.\n"
    "- React to what actually just happened, or to what somebody actually said. Never generic.\n"
    "- If someone spoke to you, answer THEM.\n"
    "- Do not narrate your own move; the table can see the board.\n"
    "\n"
    "Reply with the line and nothing else."
)


class HybridPolicy(Policy):
    name = "hybrid"

    def __init__(self, expert, llm=None, skill=1.0, chat_cooldown=3, idle_turns=12,
                 seed=None):
        self.expert = expert
        self.llm = llm                      # an LLMPolicy, used ONLY to compose chat
        self.skill = float(skill)
        self.chat_cooldown = int(chat_cooldown)
        self.idle_turns = int(idle_turns)
        self.rng = random.Random(seed)
        self._reset_match()

    def _reset_match(self):
        self.turns_since_spoke = 99         # so an opener can fire immediately
        self.seen_chat = 0                  # table lines already read
        self.spoke_this_match = 0
        self.prev = None                    # last turn's readings, for delta events

    def capabilities(self):
        return {"vision": False, "chat": True, "commentary": True}

    def match_start(self, req):
        self._reset_match()

    def act(self, req):
        actions = req.get("available_actions") or []
        if not actions:
            return "end_turn", {}, None
        state = req.get("state") or {}

        chat_action = next((a for a in actions if a.get("type") == "chat"), None)
        if chat_action is not None and self.llm is not None:
            why = self._reason_to_speak(state)
            if why:
                line = self._compose(state, why)
                if line:
                    self.turns_since_spoke = 0
                    self.spoke_this_match += 1
                    self.seen_chat = len(state.get("chat") or [])
                    return chat_action["action_id"], {"message": line}, why

        self.turns_since_spoke += 1
        return self._play(state, actions)

    def _play(self, state, actions):
        """The net's choice. At skill 1.0 its argmax; below that, its own distribution, flattened.

        Sampling from the policy's OWN mass rather than mixing in random legal actions: a weaker
        opponent should still be recognisably playing, taking moves it rates second-best, not
        occasionally walking into a wall. A bot that blunders at random is not easier, it is broken.
        """
        # CHAT IS NEVER THE NET'S TO CHOOSE, and this is not tidiness -- it hung a match.
        #
        # The net scores every action it is offered, and it has never seen `chat`: the game only
        # started offering it offline today, long after the policy was trained, so its score for it
        # is whatever the features happen to produce for an action type with no destination, no
        # target and no cost. Chat also does NOT consume the turn -- the seat is asked again
        # immediately -- so once the net rates it top, it rates it top again, forever. Measured:
        # 58,317 chat actions in one match, which reached turn 4 in ten minutes.
        #
        # Talking is the language model's job in this policy. The net plays.
        playable = [a for a in actions if a.get("type") != "chat"]
        if not playable:
            return actions[0]["action_id"], {}, None
        ranked = self.expert.rank(state, playable, k=8)
        if not ranked:
            return playable[0]["action_id"], {}, None
        if self.skill >= 0.999 or len(ranked) == 1:
            return ranked[0]["action_id"], {}, None
        t = max(0.05, self.skill)
        weights = [max(r.get("share", 0.0), 1e-6) ** (1.0 / t) for r in ranked]
        pick = self.rng.choices(ranked, weights=weights, k=1)[0]
        return pick["action_id"], {}, None

    # WHAT COUNTS AS WORTH SPEAKING ABOUT, in priority order. The first match wins.
    #
    # Written as a table because the first version was a chain of ifs that fired on three things
    # and produced ONE line per match, all of them openers -- 3 lines across 3 matches, on 1.3% of
    # the chances it was offered. A seat that only ever says "good luck" is not company.
    #
    # Everything here is a DELTA against the last turn, because the payload describes a position
    # and not what happened to reach it. `events_since_last_turn` carries some of it, but not
    # whether YOUR health fell, whether YOU took the knockout, or whether the board tipped -- and
    # those are exactly the moments a person would say something.
    def _events(self, state):
        me = state.get("self") or {}
        prev = self.prev
        f = lambda k, d=0.0: float(me.get(k) or d)
        pts = ((state.get("board") or {}).get("points")) or []
        mine = sum(1 for p in pts if p.get("relationship") == "friendly" and not p.get("is_base"))
        enemies = [p for p in (state.get("players") or []) if not p.get("is_self")]
        etier = max((float(p.get("tier") or 0) for p in enemies), default=0.0)
        now = {"kills": f("kills"), "deaths": f("deaths"), "health": f("health", 6),
               "tier": f("tier"), "points": mine, "enemy_tier": etier}
        out = []
        if prev:
            if now["kills"] > prev["kills"]:
                out.append((90, "you just knocked somebody out"))
            if now["deaths"] > prev["deaths"]:
                out.append((85, "you were just knocked out"))
            if now["tier"] > prev["tier"]:
                out.append((80, "you just went up a grade, to tier %d" % int(now["tier"])))
            if now["enemy_tier"] > prev["enemy_tier"] and now["enemy_tier"] >= 3:
                out.append((75, "an opponent is one grade away from winning"))
            if now["points"] > prev["points"]:
                out.append((60, "you just took a capture point"))
            if now["points"] < prev["points"]:
                out.append((55, "you just lost a capture point"))
            if now["health"] < prev["health"]:
                out.append((50, "you just took a hit -- %d health left" % int(now["health"])))
        self.prev = now
        return out

    def _reason_to_speak(self, state):
        """Why speak RIGHT NOW, in words, or "" to stay quiet.

        Words rather than a bool, because the reason is also what the model is told to react to. A
        line with no reason behind it is the generic filler that makes a bot tiresome to sit with.
        """
        events = self._events(state)
        chat = state.get("chat") or []

        # SOMEBODY TALKED. Always answered, ahead of everything and exempt from the cooldown -- a
        # table where the bot ignores you is worse than one that never speaks at all.
        fresh = [c for c in chat[self.seen_chat:] if float(c.get("seat", -1)) >= 0]
        if fresh:
            return "someone at the table just said: " + "; ".join(
                str(c.get("text", ""))[:120] for c in fresh[-2:])

        if self.spoke_this_match == 0 and float(state.get("turn") or 0) <= 2:
            return "the match is beginning -- greet the table"

        # Big moments jump a short cooldown; small ones wait out the full one. Without the split,
        # a knockout and a one-point health scratch are equally likely to be what you hear about.
        if events:
            pri, why = max(events)
            # A major event is worth interrupting for, so its cooldown is zero rather
            # than one: the seat had just spoken when it took a knockout, and a
            # cooldown of 1 meant the knockout went unmentioned while the line it
            # displaced was "good luck everyone".
            cooldown = 0 if pri >= 75 else self.chat_cooldown
            if self.turns_since_spoke >= cooldown:
                return why

        # THE IDLE POLL. Long stretches of a match are just walking, and a seat that only speaks on
        # incident goes quiet for twenty turns. Rarer than an event so it cannot drown one out.
        if self.turns_since_spoke >= self.idle_turns:
            me = state.get("self") or {}
            if float(me.get("tier") or 0) >= 3:
                return "you are one grade from taking the match"
            if float(me.get("health") or 6) <= 2:
                return "you are badly hurt and still standing"
            return "nothing has happened for a while -- say something to the table"
        return ""

    def _compose(self, state, why):
        """One line, in character. Failure is silence -- never a stall and never a fallback line.

        A canned line on failure would be indistinguishable from a working model saying something
        bland, which is exactly the kind of quiet breakage this project keeps finding.
        """
        me = state.get("self") or {}
        name = str(me.get("name") or "an athlete").split(" (")[0]
        chat = state.get("chat") or []
        recent = "\n".join("  seat %s: %s" % (c.get("seat"), str(c.get("text"))[:120])
                           for c in chat[-4:]) or "  (nothing said yet)"
        user = (
            "You are %s.\n"
            "Your listed abilities: %s\n\n"
            "The table so far:\n%s\n\n"
            "Turn %s. Your tier is %s of 4, health %s.\n\n"
            "Why you are speaking now: %s\n\n"
            "Write the line."
            % (name,
               ", ".join(me.get("abilities") or []) or "(no listed deviations)",
               recent, state.get("turn"), me.get("tier"), me.get("health"), why))
        try:
            line = self.llm.complete(CHAT_SYSTEM, user, max_tokens=120)
        except Exception as exc:                                    # noqa: BLE001
            print("[warrior] chat compose failed: %s" % exc, flush=True)
            return ""
        return (line or "").strip().strip('"').replace("\n", " ")[:140]
