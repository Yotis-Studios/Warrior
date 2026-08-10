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


SYSTEM_PROMPT = (
    "You are a competitive player taking your turn in a turn-based tactics game. "
    "You are given the board, your situation, and the complete list of actions you are allowed "
    "to take. Call the take_action tool exactly once, with an action_id copied exactly from that "
    "list. You may not do anything that is not on the list."
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

    def _render_state(self, state):
        """Text, not raw JSON.

        A model reads prose and a labelled board far better than it reads a nested object, and
        the token cost is lower. Anything the client did not send simply does not appear -- this
        deliberately never invents a default, because a fabricated zero reads to the model as a
        fact about the game.
        """
        out = []
        if state.get("turn") is not None:
            out.append(f"TURN {state['turn']}")
        board = state.get("board") or {}
        if board.get("ascii"):
            out.append("\nBOARD:\n" + board["ascii"])
        if board.get("legend"):
            out.append("LEGEND: " + "; ".join(f"{k} = {v}" for k, v in board["legend"].items()))
        if state.get("self"):
            out.append("\nYOU: " + json.dumps(state["self"], separators=(", ", ": ")))
        if state.get("players"):
            out.append("\nPLAYERS:")
            for p in state["players"]:
                out.append("  " + json.dumps(p, separators=(", ", ": ")))
        if board.get("points"):
            out.append("\nCAPTURE POINTS:")
            for pt in board["points"]:
                out.append("  " + json.dumps(pt, separators=(", ", ": ")))
        if state.get("events_since_last_turn"):
            out.append("\nSINCE YOUR LAST TURN:")
            for e in state["events_since_last_turn"]:
                out.append(f"  - {e}")
        if state.get("chat"):
            out.append("\nTABLE CHAT:")
            for c in state["chat"]:
                out.append(f"  seat {c.get('seat')}: {c.get('text')}")
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
