"""Package an RL checkpoint into a self-contained Hugging Face model repo.

    python tools/hf/publish.py ppo-cover RaifuWars-RL-ActionScorer-Cover --dry-run
    python tools/hf/publish.py ppo-cover RaifuWars-RL-ActionScorer-Cover

Each repo ships the weights, the exact feature and policy modules they were trained with, and a
`serve.py` that speaks the Warrior protocol -- so a download runs without this repository.

THE MODULES ARE COPIED, NOT REFERENCED. `features.py` defines the input encoding, and a checkpoint
is meaningless against a different one: two of these runs differ from the others in exactly that
file's output width. Pinning the encoder next to the weights is the only way a download stays
runnable after the training repo moves on.

IT VERIFIES BEFORE IT UPLOADS. The staged directory is exercised in a subprocess -- load the
checkpoint through the shipped serve.py, at the RW_FEAT_COVER the card tells you to set -- and the
upload does not happen if that fails. Publishing a checkpoint nobody can load is the failure this
guards, and it is not hypothetical: the serve.py already on the Hub builds the network at fixed
dimensions and cannot load two of these three.
"""

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WARRIOR = os.path.dirname(os.path.dirname(HERE))
RL = os.path.join(WARRIOR, "..", "raifuwars-rl")
ORG = "yotisstudios"


def stage(run, repo, ckpt="last.pt", outdir=None):
    src = os.path.join(RL, "runs", run, ckpt)
    if not os.path.isfile(src):
        sys.exit("no checkpoint at %s" % src)
    card = os.path.join(HERE, "cards", "%s.md" % run)
    if not os.path.isfile(card):
        sys.exit("no model card at %s" % card)

    out = outdir or os.path.join(WARRIOR, "data", "hf", repo)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "raifuwars_rl"))

    weights = "raifuwars-actionscorer-%s.pt" % run.replace("ppo-", "")
    shutil.copy(src, os.path.join(out, weights))

    # SPLICE THE SHARED TAIL, with this checkpoint's real dimensions substituted in. The
    # architecture block is written once because three cards describing three different shapes are
    # three chances to publish the wrong numbers next to the right weights -- and a reader has no
    # way to check them against the .pt.
    import torch
    blob = torch.load(src, map_location="cpu", weights_only=False)
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    dims = {
        "D_STATE": str(sd["state_tower.0.weight"].shape[1]),
        "D_ACTION": str(sd["action_tower.0.weight"].shape[1]),
        "HIDDEN": str(sd["state_tower.0.weight"].shape[0]),
        "EMBED": str(sd["state_tower.2.weight"].shape[0]),
        "COVERFLAG": "1" if sd["state_tower.0.weight"].shape[1] > 33 else "0",
        "WEIGHTS": weights,
    }
    common = open(os.path.join(HERE, "cards", "_common.md"), encoding="utf-8").read()
    common = common.split("-->", 1)[1].strip() if "-->" in common else common
    for k, v in dims.items():
        common = common.replace(k, v)
    body = open(card, encoding="utf-8").read()
    if "COMMON" not in body:
        sys.exit("%s has no COMMON marker -- the shared sections would be dropped" % card)
    body = body.replace("COMMON", common)
    open(os.path.join(out, "README.md"), "w", encoding="utf-8", newline="\n").write(body)
    shutil.copy(os.path.join(HERE, "serve.py"), os.path.join(out, "serve.py"))
    for f in ("features.py", "policy.py"):
        shutil.copy(os.path.join(RL, "raifuwars_rl", f), os.path.join(out, "raifuwars_rl", f))
    open(os.path.join(out, "raifuwars_rl", "__init__.py"), "w").close()
    return out, weights


def verify(staged, weights):
    """Load the shipped weights through the shipped serve.py, in a fresh process.

    A fresh process because the feature width freezes at the first import of `features`; this one
    has to be the checkpoint's own. The cover flag is derived from the checkpoint rather than
    passed in, which also checks that the instruction the card gives is the correct one.
    """
    import torch
    blob = torch.load(os.path.join(staged, weights), map_location="cpu", weights_only=False)
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    cover = "1" if sd["state_tower.0.weight"].shape[1] > 33 else "0"

    env = dict(os.environ, RW_FEAT_COVER=cover)
    code = ("import runpy,sys;sys.argv=['serve.py',%r,'--port','8987'];"
            "import threading,os;threading.Timer(6, lambda: os._exit(0)).start();"
            "runpy.run_path('serve.py', run_name='__main__')" % weights)
    p = subprocess.run([sys.executable, "-c", code], cwd=staged, env=env,
                       capture_output=True, text=True, timeout=180)
    banner = next((l for l in (p.stdout + p.stderr).splitlines() if l.startswith("[serve]")), "")
    if not banner:
        print((p.stdout + p.stderr)[-1500:])
        sys.exit("VERIFY FAILED: %s did not start under RW_FEAT_COVER=%s" % (weights, cover))
    return cover, banner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run directory under raifuwars-rl/runs")
    ap.add_argument("repo", help="repo name under the org")
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    staged, weights = stage(args.run, args.repo, args.ckpt)
    cover, banner = verify(staged, weights)
    print("staged  %s" % staged)
    print("verify  %s" % banner)
    print("        (RW_FEAT_COVER=%s)" % cover)
    for f in sorted(os.listdir(staged)):
        print("        %s" % f)

    if args.dry_run:
        print("dry run -- not uploading")
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    rid = "%s/%s" % (ORG, args.repo)
    api.create_repo(rid, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=staged, repo_id=rid, repo_type="model",
                      commit_message="add %s (%s)" % (args.run, args.ckpt))
    print("uploaded https://huggingface.co/%s" % rid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
