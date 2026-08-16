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
        except ImportError as exc:                                   # noqa: BLE001
            raise ExpertUnavailable("could not import torch: %s" % exc) from exc

        # READ THE CHECKPOINT BEFORE IMPORTING THE ENCODER, because the encoder's width is fixed
        # at import time by RW_FEAT_COVER and cannot be changed afterwards.
        #
        # The runs are not one architecture. `ppo-bignet` is 256/128 rather than 128/64, and
        # `ppo-cover` was trained with the terrain features switched on, which widens the input
        # from 33/27 to 35/28. The old code called `ActionScorer()` with its defaults and would
        # have thrown a shape error on both -- the good case. The bad case is the one that
        # motivates reading the file first: run `ppo-cover` with RW_FEAT_COVER already set for
        # some other reason and every checkpoint of the OTHER shape loads a 35-wide encoder into
        # a 33-wide net. So the file decides, and nothing about the environment gets a vote.
        blob = torch.load(checkpoint, map_location=device, weights_only=False)
        sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        try:
            d_state = sd["state_tower.0.weight"].shape[1]
            d_action = sd["action_tower.0.weight"].shape[1]
            hidden = sd["state_tower.0.weight"].shape[0]
            embed = sd["state_tower.2.weight"].shape[0]
        except (KeyError, AttributeError, IndexError) as exc:        # noqa: BLE001
            raise ExpertUnavailable(
                "%s does not look like an ActionScorer checkpoint: %s" % (checkpoint, exc)) from exc

        wants_cover = d_state > 33
        os.environ["RW_FEAT_COVER"] = "1" if wants_cover else "0"

        try:
            from raifuwars_rl.features import D_ACTION, D_STATE, encode_actions, encode_state
            from raifuwars_rl.policy import ActionScorer
        except ImportError as exc:                                   # noqa: BLE001
            raise ExpertUnavailable("could not import the RL policy: %s" % exc) from exc

        # THE ENCODER IS ALREADY IMPORTED SOMEWHERE ELSE case. Setting the env var above is only
        # effective if `features` had not been imported yet; if it had, the dims are already
        # frozen and may not be the ones this checkpoint wants. Fail loudly rather than serve a
        # policy reading garbage -- a mis-scaled feature vector produces confident nonsense, and
        # a seat that plays badly for an invisible reason is the exact failure this project keeps
        # paying for.
        if (D_STATE, D_ACTION) != (d_state, d_action):
            raise ExpertUnavailable(
                "%s wants %d/%d features but the encoder is built for %d/%d. It was trained with "
                "terrain features %s; set RW_FEAT_COVER=%s before starting the sidecar, and do "
                "not load two checkpoints of different widths in one process."
                % (checkpoint, d_state, d_action, D_STATE, D_ACTION,
                   "ON" if wants_cover else "OFF", "1" if wants_cover else "0"))

        self._torch = torch
        self._encode_state = encode_state
        self._encode_actions = encode_actions

        self.net = ActionScorer(d_state=d_state, d_action=d_action,
                                hidden=hidden, embed=embed).to(device)
        self.net.load_state_dict(sd)                                 # strict: no silent partials
        self.net.eval()
        self.arch = {"d_state": d_state, "d_action": d_action, "hidden": hidden, "embed": embed,
                     "cover": wants_cover,
                     "params": int(sum(v.numel() for v in sd.values()))}
        self.device = torch.device(device)
        # RUN DIRECTORY PLUS FILENAME. Every run writes `last.pt`, so the basename alone names
        # nothing -- the sidecar log said "expert loaded: last.pt" for all ten of them.
        self.name = "/".join(os.path.abspath(checkpoint).replace("\\", "/").split("/")[-2:])
        # A short content hash, so a caller can tell two checkpoints apart without trusting a path.
        import hashlib
        h = hashlib.sha256()
        for k in sorted(sd):
            h.update(k.encode())
            h.update(sd[k].detach().cpu().numpy().tobytes())
        self.sha = h.hexdigest()[:12]
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
