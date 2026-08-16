---
license: gpl-3.0
library_name: pytorch
tags:
- reinforcement-learning
- ppo
- negative-results
- ablation
- sim-to-real
- games
- raifu-wars
- warrior-protocol
pipeline_tag: reinforcement-learning
---

# Raifu Wars — RL Action Scorer (Cover-Features Arm)

A **58,114-parameter** policy that plays a seat in [Raifu Wars](https://raifuwars.com), a turn-based
strategy game, through the [Warrior protocol](https://github.com/Yotis-Studios/Warrior).

**This is a published negative result, and the negative is about the experiment rather than the
hypothesis.** This run was built to test whether adding terrain features would fix the policy's
collapse on cover-heavy boards. It did not test that. The three terrain features were **constant
zero for the entire eight-hour run**, because the simulator it trained in does not put the map into
the state it sends the policy. The hypothesis remains open; this checkpoint is the evidence that
the experiment missed it.

The sharp version, because it is easy to state this too loosely: the sim *does* have trees. It
scatters vegetation and that vegetation *does* change hit rolls. Cover is mechanically present and
perceptually absent — the policy was being shot at through trees it had no channel to see.

It is published because the finding it carries — *the sim cannot see the thing we were trying to
train on* — invalidates a class of results, and the weights are the proof.

## What it was supposed to test

The predecessor `ppo-sim` won 68% on Crossroads and **13%** on Arboretum. The two boards differ in
one obvious way:

| board | cover tiles | `ppo-sim` |
|---|---|---|
| Crossroads | **0 (0.0%)** | 68% |
| Arboretum | 172 (37.2%) | **13%** |

Nothing in the 33 state and 27 action features describes terrain. The policy can read `hit_chance`
for a shot *it* takes but has no way to represent "this destination leaves me exposed" or "that one
puts me behind a tree" — only nearer and further. The hypothesis was that Arboretum is a
**perception** failure, not a training one.

So this arm turns on three features — `cover_density` and `cover_here` on the state, `dest_cover`
on each action — widening the input from 33/27 to **35/28**, and copies `ppo-sim`'s configuration
exactly otherwise: same boards, steps, batch, learning rate, hours and greedy opponent, so the
feature change is the only variable.

## Why it tested nothing

The simulator's `board` payload contains `width`, `height` and `points`. There is no map in it. The
feature code derives cover from the board's ASCII rendering, which the real game sends and the sim
does not, so all three new features evaluate to zero on every sim state — while the trees they were
meant to describe go on affecting every shot taken.

Three independent confirmations:

1. **The payload.** Sim `board` keys are `width, height, points`; the real game's are
   `ascii, legend, width, height, points`.
2. **The features.** `cover_density` reads **0.000 across 6,001 sim states** (max 0.000). On the
   real game's Arboretum it reads **0.376**, and `cover_here` averages 0.447 — agents are behind
   cover almost half the time.
3. **The weights in this repo.** An input that is always exactly zero contributes exactly zero
   gradient to its weights. Comparing `best.pt` (update 5,835) with `last.pt` (update 9,127):

   | first-layer column | mean absolute change over 3,292 updates |
   |---|---|
   | the 33 base state features | 2.0e-02 |
   | `cover_density`, `cover_here` | **0.0, 0.0** |
   | the 27 base action features | 7.5e-03 |
   | `dest_cover` | **0.0** |

   The terrain columns are byte-identical between the two checkpoints while every other column
   moved. They never received a single update and sit at their initialisation.

Because a zero input contributes nothing to the pre-activation, this network is **functionally
identical to a 33/27 network** — it is `ppo-sim` re-run under a different seed, carrying 384 frozen
parameters. Read its scores as a seed replicate, not as a cover-aware policy.

## Results

Real game, against the built-in AI, 16 matches per board, chance 25%:

| board | this model | `ppo-sim` (its true sibling) | `ppo-selfplay` |
|---|---|---|---|
| Arboretum | 2/16 — 12% | 13% *(40 matches)* | **8/16 — 50%** |
| Islands | 0/16 — 0% | not evaluated | 5/16 — 31% |
| Crossroads | 9/16 — 56% | 68% *(40 matches)* | 69% |
| **overall** | **11/48 — 23%** | — | 24/48 — 50% |

23% against a chance rate of 25% — **not distinguishable from picking at random** (p=0.68). And on
Arboretum, the board this arm exists to fix, it scores **12% against `ppo-sim`'s 13%**: exactly
where a policy with three dead features and a different random seed should land. That prediction was
made before the run and it held, which is the strongest evidence that the diagnosis above is right.

The sim numbers make the same point from the other side. Against three greedy bots, **on Arboretum
specifically**, 400 matches each:

| policy | sim, Arboretum | real game, Arboretum |
|---|---|---|
| this model | 79.0% | 12% |
| `ppo-selfplay` | 78.0% | 50% |
| `ppo-selfplay2` | 78.5% | 12% |
| `ppo-bignet` | 75.8% | 12% |
| `ppo-sim` | 73.8% | 13% |

The sim compresses a 38-point real-game spread into five points, and rates a board with 172 cover
tiles about the same as one with none. **Sim results cannot rank policies on cover-heavy boards**,
because in the sim those boards are not cover-heavy — they are empty rooms with the same
dimensions.

## Training

- **8 hours**, **9,127 updates**, **10,514,304 agent decisions**, ~421 steps/sec, from scratch —
  changing the input width makes an existing checkpoint unusable as an initialiser.
- PPO, lr 5e-5, 6 envs × 192 steps, batch 128, against a scripted greedy opponent, in the
  [Hemlock](https://github.com/Yotis-Studios/raifusim) reimplementation.
- Boards: `Arboretum, Crossroads, Dustbowl, Glacier, Cornfield, Trench Warfare, Twin Rivers`.
- Final mean return 16.24, peak 18.09 at update 5,835. Return against greedy saturates near 17 well
  before skill does, so these figures rank nothing.

## What would actually test the hypothesis

Make the simulator emit `board.ascii`, verify with `RW_FEAT_COVER=1` that `cover_density` is
non-zero on sim states **before** launching, and re-run. The check is one line and it would have
saved eight hours of GPU time and a wrong conclusion. As it stands the original question — *is
Arboretum a perception failure?* — has never been put to a fair test.

The companion capacity arm,
[RaifuWars-RL-ActionScorer-BigNet](https://huggingface.co/yotisstudios/RaifuWars-RL-ActionScorer-BigNet),
ran alongside this one and *was* valid, since it changed no features.

COMMON

## Limitations, stated plainly

- **The three terrain features are dead weights.** They are shipped so the checkpoint loads at the
  width it was trained at; they carry no information. Do not cite this model as evidence that
  terrain features help or do not help.
- **It requires `RW_FEAT_COVER=1`**, and `serve.py` refuses at startup and says so otherwise. Note
  what the check buys: the weights themselves load fine under the wrong flag, and the mismatch only
  appears on the first decision as a matrix shape error. A host that catches policy errors and
  falls back to a legal action will play the whole match on fallbacks and still hand you a results
  table, so the failure is cheap to miss without the startup check.
- Trained against a **scripted greedy opponent only**, so like its sibling `ppo-sim` it is expected
  to lose badly to self-play policies while scoring well against scripted ones. In head-to-head it
  is untested; it cannot share an arena process with the 33/27 runs.
- 16 matches per board in the real game, ±21 points at 95%.
