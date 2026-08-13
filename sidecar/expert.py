"""The RL policy, exposed to the LLM as something it can ASK.

    expert = Expert("runs/ppo-sim/best.pt")
    expert.rank(state, actions, k=5)   -> [{action_id, type, label, score, share}, ...]

WHY A TOOL AND NOT A ROUTER. The obvious composition is a switch: let the RL policy take the
actions and the LLM do the talking. That throws away the interesting half. The RL policy is a
236 KB action-scorer that beats the built-in AI 55% to 28% and cannot say a word about why; the
LLM can hold a plan, explain itself, and talk to the table, and plays this game badly. Making the
scorer a TOOL keeps the LLM in charge and gives it a strong opinion to accept, override, or argue
with -- which is a different thing from being overruled by a router it cannot see.

It also makes disagreement measurable. Every consult records what the expert wanted and what the
model did, so "how often does the LLM overrule the expert, and does it win more or less when it
does" is a question with an answer rather than a vibe.

WHAT IT RETURNS IS A DISTRIBUTION, NOT AN ORDER. The scorer softmaxes over exactly the offered
actions, so `share` is how much probability mass it puts on each one. A 0.95 top choice and a 0.30
top choice are different kinds of advice -- the first is "obviously this", the second is "I am
guessing between four things" -- and an LLM handed only a ranking cannot tell those apart.
"""

import os
import sys


class ExpertUnavailable(RuntimeError):
    pass


class Expert:
    def __init__(self, checkpoint, rl_path=None, device="cpu"):
        rl_path = rl_path or os.environ.get(
            "RW_RL_PATH",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "..", "raifuwars-rl"))
        rl_path = os.path.abspath(rl_path)
        if not os.path.isdir(rl_path):
            raise ExpertUnavailable("raifuwars-rl not found at %s -- set RW_RL_PATH" % rl_path)
        if rl_path not in sys.path:
            sys.path.insert(0, rl_path)

        try:
            import torch
            from raifuwars_rl.features import encode_actions, encode_state
            from raifuwars_rl.policy import ActionScorer
        except ImportError as exc:                                   # noqa: BLE001
            raise ExpertUnavailable("could not import the RL policy: %s" % exc) from exc

        self._torch = torch
        self._encode_state = encode_state
        self._encode_actions = encode_actions

        blob = torch.load(checkpoint, map_location=device, weights_only=False)
        self.net = ActionScorer().to(device)
        self.net.load_state_dict(blob["model"] if "model" in blob else blob)
        self.net.eval()
        self.device = torch.device(device)
        self.name = os.path.basename(checkpoint)
        self.consults = 0
        self.overruled = 0

    def rank(self, state, actions, k=5):
        """The expert's top k, with the probability mass it assigns each."""
        torch = self._torch
        with torch.no_grad():
            s = torch.tensor(self._encode_state(state), device=self.device)
            a = torch.tensor(self._encode_actions(state, actions), device=self.device)
            logits = self.net(s, a)
            if not torch.all(torch.isfinite(logits)):
                # Uniform rather than a crash: bad advice is recoverable, a dead sidecar stalls
                # the match, and the game has no clock to time it out.
                logits = torch.zeros_like(logits)
            probs = torch.softmax(logits, dim=0)

            # THE SECOND HEAD, AND IT ANSWERS A DIFFERENT QUESTION. The ranking says what to do;
            # this says how you are DOING -- PPO's estimate of the return from this position,
            # independent of which action you pick. The board tells the model where everyone
            # stands, not whether it is winning, and that is exactly the multi-turn judgement the
            # expert cannot supply for itself: it scores each position from scratch and has no
            # plan to notice is failing.
            value = float(self.net.value_of(s))

        order = torch.argsort(probs, descending=True)[:max(1, k)]
        self.last_value = value
        out = []
        for idx in order.tolist():
            act = actions[idx]
            out.append({
                "action_id": str(act.get("action_id")),
                "type": act.get("type"),
                "label": act.get("label"),
                "share": round(float(probs[idx]), 4),
            })
        self.consults += 1
        return out

    def render(self, ranked):
        """The advice as the model will read it. Prose, not JSON: it goes into a tool result that
        the model reads as text, and a table of numbers invites it to pattern-match on the numbers
        rather than weigh them."""
        if not ranked:
            return "The expert had no opinion."
        top = ranked[0]
        lines = ["The expert (%s, wins ~55%% against the built-in AI's ~28%%) ranks the offered "
                 "actions:" % self.name]
        for r in ranked:
            lines.append("  %-14s %-24s %.0f%% of its confidence"
                         % (r["action_id"], r["label"] or r["type"] or "", 100 * r["share"]))
        # The scale is the reward's, so it is only meaningful by comparison -- quoted against what
        # a won match actually scores rather than left as a bare number a model would have no way
        # to interpret. ~17 is a winning position; ~4 is where a random policy sits.
        v = getattr(self, "last_value", None)
        if v is not None:
            lines.append("It rates your POSITION at %.1f (a winning position scores about 17, "
                         "a losing one about 4). This is how you are doing, not what to do." % v)
        if top["share"] >= 0.8:
            lines.append("It is confident. Disagreeing is a real decision, not a coin flip.")
        elif top["share"] <= 0.4:
            lines.append("It is NOT confident -- it is spreading its bet. Your judgement is worth "
                         "as much as its ranking here.")
        return "\n".join(lines)
