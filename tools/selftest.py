"""Hard assertions over the seams that have actually broken. One command, no dependencies.

    python tools/selftest.py            # everything
    python tools/selftest.py --quick    # skip the ones that start a server

EVERY CHECK HERE CORRESPONDS TO A BUG THAT SHIPPED. That is the selection rule -- not "what could
be tested", which is unbounded and produces a suite nobody trusts. If a check here has never caught
anything and could not have caught anything, it is decoration and should be deleted.

The bugs, and what now covers them:

  the idempotency cache was keyed on request_id alone, while the game resets its
  request counter every match -- so match 2 opened with match 1's actions           -> cache_*
  a policy returning an id that was not offered wasted a whole retry round trip     -> fallback_*
  a retry that does not tell the model WHY loops until the retry budget is gone     -> retry_*
  chat rendered into the prompt but the game never offered the action offline       -> render_*
  the expert checkpoint is a blob with the weights under "model", not a state_dict  -> expert_*
  a feature that is constant across every offered action is a bias term             -> feature_*
  framing, reassembly, reply pairing and connection concurrency                     -> tcp_selftest.py
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "sidecar"))

FAILS = []


def check(name, ok, detail=""):
    print("  %-52s %s%s" % (name, "PASS" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


def act_req(match="M1", rid="r1", actions=None, reason="turn_start", last=None):
    req = {"protocol_version": "0.1", "request_id": rid, "match_id": match, "seat": 0,
           "reason": reason, "deadline_ms": 30000,
           "available_actions": actions if actions is not None else [
               {"action_id": "move_5_5", "type": "move", "label": "Move to (5,5)"},
               {"action_id": "end", "type": "end_turn", "label": "End your turn"}],
           "state": {"self": {"tile": [1, 1], "seat": 0, "health": 6, "max_health": 6, "ammo": 5,
                              "max_ammo": 5, "tier": 0, "stars": 0, "kills": 0, "deaths": 0,
                              "range": 5, "this_turn": {}, "hand": [],
                              "tier_progress": {"have": 0, "need": 25, "condition": "stars"}},
                     "players": [{"seat": 0, "is_self": True, "team": 1, "tile": [1, 1],
                                  "health": 6, "status": "alive"}],
                     "chat": [{"seat": 2, "text": "hello table"}],
                     "board": {"width": 21, "height": 21, "ascii": "", "legend": {},
                               "points": [{"tile": [4, 4], "radius": 1, "relationship": "friendly",
                                           "is_base": True, "stars_per_turn": 1}]},
                     "turn": 3}}
    if last:
        req["last_action"] = last
    return req


def suite_sidecar():
    from server import Sidecar, dispatch                            # noqa: E402

    class Fixed:
        """A policy that returns whatever it is told to, including something illegal."""
        name = "fixed"

        def __init__(self, out):
            self.out = out

        def act(self, req):
            return self.out, {}, None

        def capabilities(self):
            return {}

        def match_start(self, req):
            pass

        def match_end(self, req):
            pass

        def event(self, req):
            pass

    # THE CACHE KEY. The game resets both its turn number and its sequence counter at every match,
    # so "rw-t0-s0-n1" recurs. Keyed on the id alone, match 2 is answered with match 1's action --
    # chosen for a different board, and legal there by coincidence if at all.
    sc = Sidecar(Fixed("move_5_5"))
    a = sc.act(act_req(match="A", rid="rw-t0-s0-n1"))
    sc.policy = Fixed("move_9_9")
    b = sc.act(act_req(match="B", rid="rw-t0-s0-n1",
                       actions=[{"action_id": "move_9_9", "type": "move", "label": "m"},
                                {"action_id": "end", "type": "end_turn", "label": "e"}]))
    check("cache_scoped_to_match", b["action_id"] == "move_9_9",
          "match A gave %s, match B gave %s" % (a["action_id"], b["action_id"]))

    # ...but a genuine retry of the SAME request must still be idempotent, or a policy asked twice
    # contradicts itself mid-action.
    sc2 = Sidecar(Fixed("move_5_5"))
    first = sc2.act(act_req(match="A", rid="same-id"))
    sc2.policy = Fixed("end")
    again = sc2.act(act_req(match="A", rid="same-id"))
    check("cache_idempotent_within_match", again["action_id"] == first["action_id"],
          "%s then %s" % (first["action_id"], again["action_id"]))

    # AN ILLEGAL CHOICE MUST NOT REACH THE GAME. It costs a full retry round trip, and the schema
    # enum demonstrably does not bind a model.
    sc3 = Sidecar(Fixed("move_99_99"))
    r = sc3.act(act_req(rid="illegal-1"))
    check("fallback_on_unoffered_id", r["action_id"] in ("move_5_5", "end"), r["action_id"])
    check("fallback_counts_an_error", sc3.total_errors == 1, str(sc3.total_errors))

    # The fallback must come from the OFFERED list -- never a hardcoded "end_turn", which is the
    # illegal-action bug wearing a helpful hat.
    only = [{"action_id": "move_1_1", "type": "move", "label": "m"}]
    check("fallback_never_invents_end_turn",
          Sidecar._fallback(only) == "move_1_1", Sidecar._fallback(only))

    # ROUTING PARITY. Both transports go through dispatch(); a route that answers on one and not
    # the other is how a transport ends up supporting a different protocol version.
    sc4 = Sidecar(Fixed("end"))
    for path in ("/v1/act", "/v1/match/start", "/v1/match/end", "/v1/event", "/v1/health"):
        st, _ = dispatch(sc4, path, act_req(rid="d-" + path))
        check("dispatch_200_%s" % path.replace("/", "_"), st == 200, str(st))
    st, _ = dispatch(sc4, "/v1/nope", {})
    check("dispatch_404_unknown_route", st == 404, str(st))


def suite_policy():
    import policy as P                                              # noqa: E402

    pol = P.LLMPolicy.__new__(P.LLMPolicy)                          # no network, no constructor
    pol.hint_style = getattr(pol, "hint_style", "plain")

    state = act_req()["state"]
    rendered = P.LLMPolicy._render_state(pol, state)

    # Chat has to reach the prompt, or the model cannot answer what the table said.
    check("render_includes_table_chat", "hello table" in rendered)
    # Points are under board, not state -- getting this wrong silently reports "0% on a point" for
    # every policy including the built-in AI, which is how it was caught.
    check("render_includes_points", "4" in rendered and "point" in rendered.lower())

    acts = act_req()["available_actions"]
    ra = P.LLMPolicy._render_actions(pol, acts)
    check("render_lists_every_offered_id", all(a["action_id"] in ra for a in acts))

    # NO ASSERTION THAT THE PROMPT TEACHES STRATEGY, and its removal is the point.
    #
    # Three checks lived here demanding the system prompt explain the win condition -- which track
    # you are on, that points pay per turn, that cover decides fights. They passed, and the thing
    # they were protecting was then measured and found to be a REGRESSION: 3/24 with that text
    # against 8/32 without, and the one board the model was good at halved. A test that pins a
    # hypothesis in place is worse than no test, because it makes reverting look like a break.
    #
    # What is asserted instead is the part that is true regardless of strategy: the prompt must
    # tell the model to copy an id exactly from the offered list, which is the one instruction
    # every measured failure to act legally traces back to.
    # THE PROMPT MUST STILL BE THE ONE THE MODEL WAS TRAINED ON, byte for byte. It is pasted from
    # the dataset rather than written here, and whitespace counts: the conversion collapses the
    # paragraph breaks, so a re-tidied version differs on every request from what training saw.
    import hashlib
    check("system_prompt_matches_the_trained_bytes",
          hashlib.sha256(P.SYSTEM_PROMPT.encode()).hexdigest()[:16] == P.SYSTEM_PROMPT_SHA16,
          P.SYSTEM_PROMPT_SHA16)

    sp = P.SYSTEM_PROMPT.lower()
    check("system_prompt_demands_exact_action_id",
          "exactly" in sp and "action_id" in sp)
    check("system_prompt_is_game_agnostic",
          "raifu" not in sp and "tier" not in sp,
          "rules belong in state.briefing, written by the game")


def _ckpt_width(torch, path):
    """The state-feature width a checkpoint was trained at, read from its first layer."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    try:
        return int(sd["state_tower.0.weight"].shape[1])
    except Exception:                                                # noqa: BLE001
        return -1


def suite_expert():
    import glob
    ck = sorted(glob.glob(os.path.join(ROOT, "..", "raifuwars-rl", "runs", "*", "last.pt")))
    if not ck:
        print("  (no checkpoint found -- expert checks skipped)")
        return
    try:
        import torch
        sys.path.insert(0, os.path.join(ROOT, "..", "raifuwars-rl"))
        from raifuwars_rl.features import D_ACTION, D_STATE, encode_actions, encode_state
        from raifuwars_rl.policy import ActionScorer
    except Exception as exc:                                        # noqa: BLE001
        print("  (torch/raifuwars_rl unavailable: %s -- expert checks skipped)" % exc)
        return

    # The checkpoint is a blob with the weights under "model" plus a weight_mode, NOT a bare
    # state_dict. Loading it as one raises on every key.
    blob = torch.load(ck[0], map_location="cpu")
    check("expert_checkpoint_has_model_key", isinstance(blob, dict) and "model" in blob,
          str(list(blob)[:3]) if isinstance(blob, dict) else type(blob).__name__)

    # EVERY CHECKPOINT ON DISK MUST LOAD THROUGH THE REAL LOADER, one subprocess each.
    #
    # This check used to build an `ActionScorer()` at its defaults and load one arbitrary
    # checkpoint into it -- a path the sidecar does not use, asserting a fact that was not true.
    # It went green for as long as every run had the same shape, then crashed the whole suite the
    # day a 256-wide run appeared. Neither behaviour is a test.
    #
    # A subprocess per checkpoint because the feature width is frozen at the first import of
    # `raifuwars_rl.features`: a run trained with terrain features on is 35/28 and cannot be
    # verified in a process that has already imported the 33/27 encoder. One process, one
    # checkpoint, which is also how the sidecar runs.
    import json
    import subprocess
    probe = (
        "import json,sys;sys.path.insert(0,%r);from expert import Expert;"
        "print(json.dumps(Expert(sys.argv[1]).arch))" % os.path.join(ROOT, "sidecar"))
    for path in ck:
        run = os.path.basename(os.path.dirname(path))
        p = subprocess.run([sys.executable, "-c", probe, path],
                           capture_output=True, text=True, timeout=180)
        tail = (p.stdout.strip().splitlines() or [""])[-1]
        try:
            arch = json.loads(tail)
        except ValueError:
            arch = None
        check("expert_loads_%s" % run, arch is not None,
              "%dw %d/%d params=%d%s" % (arch["hidden"], arch["d_state"], arch["d_action"],
                                         arch["params"], " cover" if arch["cover"] else "")
              if arch else (p.stderr.strip().splitlines() or ["no output"])[-1][:120])

    # A CHECKPOINT WHOSE WIDTH THE ENCODER CANNOT SERVE MUST SAY SO. This process has already
    # imported the encoder at D_STATE, so a checkpoint of any other width is unservable here --
    # and the required behaviour is a named refusal, not confident nonsense from a net fed a
    # feature vector of the wrong shape.
    other = next((p for p in ck if _ckpt_width(torch, p) != D_STATE), None)
    if other is None:
        check("expert_refuses_mismatched_feature_width", True, "(no checkpoint of another width)")
    else:
        try:
            from expert import Expert, ExpertUnavailable
            Expert(other)
            check("expert_refuses_mismatched_feature_width", False, "loaded a %d-wide net anyway"
                  % _ckpt_width(torch, other))
        except ExpertUnavailable as exc:
            check("expert_refuses_mismatched_feature_width", "RW_FEAT_COVER" in str(exc),
                  os.path.basename(os.path.dirname(other)))
        except Exception as exc:                                     # noqa: BLE001
            check("expert_refuses_mismatched_feature_width", False,
                  "raised %s, not ExpertUnavailable" % type(exc).__name__)

    # NO CONSTANT FEATURES. A feature identical across every offered action carries no information
    # and is a bias term that costs a weight -- exactly what dest_exposure turned out to be.
    req = act_req(actions=[{"action_id": "move_%d_%d" % (x, 3), "type": "move", "label": "m"}
                           for x in range(2, 9)])
    mat = encode_actions(req["state"], req["available_actions"])
    vec = encode_state(req["state"])
    check("features_are_finite", bool(mat.size and vec.size) and
          bool((mat == mat).all() and (vec == vec).all()))
    varying = sum(1 for c in range(mat.shape[1]) if mat[:, c].min() != mat[:, c].max())
    check("some_action_features_vary_across_moves", varying >= 2, "%d columns vary" % varying)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip checks that start a server")
    args = ap.parse_args()

    print("sidecar")
    suite_sidecar()
    print("policy")
    suite_policy()
    print("expert / features")
    suite_expert()

    if not args.quick:
        print("transport (tools/tcp_selftest.py)")
        r = subprocess.run([sys.executable, os.path.join(HERE, "tcp_selftest.py")],
                           cwd=ROOT, capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if line.strip().startswith(("PASS", "FAIL")) or "  PASS" in line or "  FAIL" in line:
                print("  " + line.strip())
        if r.returncode != 0:
            FAILS.append("tcp_selftest")
        print("  %-52s %s" % ("tcp_selftest overall", "PASS" if r.returncode == 0 else "FAIL"))

    print()
    if FAILS:
        print("FAILED (%d): %s" % (len(FAILS), ", ".join(FAILS)))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
