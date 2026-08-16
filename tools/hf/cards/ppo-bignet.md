---
license: gpl-3.0
library_name: pytorch
tags:
- reinforcement-learning
- ppo
- ablation
- negative-results
- sim-to-real
- games
- raifu-wars
- warrior-protocol
pipeline_tag: reinforcement-learning
---

# Raifu Wars — RL Action Scorer (Capacity Arm)

A **213,762-parameter** policy that plays a seat in [Raifu Wars](https://raifuwars.com), a turn-based
strategy game, through the [Warrior protocol](https://github.com/Yotis-Studios/Warrior).

This is a **control arm, published as a negative result.** It is `ppo-sim` at 3.7× the parameters —
256/128 wide instead of 128/64 — with the features, boards, opponent, learning rate, batch size and
wall-clock budget held identical. It exists to test one claim made in this project's own notes:
*the network is not the bottleneck.*

It is not. The extra capacity produced a **worse** policy, and the way it is worse is the useful
part.

## The result

| | this model (214K) | `ppo-sim` (58K) | `ppo-selfplay` (58K) | `ppo-selfplay2` (58K) |
|---|---|---|---|---|
| sim, head-to-head, 744 matches | **3.8%** | 13.3% | 42.3% | **48.1%** |
| sim vs three greedy bots, 400 | **71.2%** | **84.5%** | 82.0% | 77.8% |
| sim vs greedy, Crossroads only | **96.0%** | 83.8% | 89.0% | 90.8% |
| sim vs greedy, Arboretum only | 75.8% | 73.8% | 78.0% | 78.5% |
| training mean return | **16.90** | 16.02 | — | 10.92 |

And in the real game, against the built-in AI, 16 matches per board where chance for one seat of
four is 25%:

| board | this model | `ppo-selfplay` (58K) | `ppo-selfplay2` (58K) |
|---|---|---|---|
| Arboretum | 2/16 — 12% | **8/16 — 50%** | 2/16 — 12% |
| Islands | 1/16 — 6% | 5/16 — 31% | **13/16 — 81%** |
| Crossroads | 8/16 — 50% | 69% | **75%** |
| **overall** | **11/48 — 23%** | 24/48 — 50% | 27/48 — 56% |

23% against a chance rate of 25%. This model is **not distinguishable from a policy picking at
random among the offered actions** (p=0.68), on a board pool where its 58K-parameter twin scores
50–56%. Its one respectable board is Crossroads, the one with no cover — the same board it
dominates in the sim.


Read the first two rows together. This model has the **highest training return of any
greedy-trained arm** and the **lowest head-to-head win rate of any policy measured** — 3.8%, where
chance is 25% and its 58K-parameter twin manages 13.3%. It is also the single best policy in the
project at Crossroads-against-greedy, at 96.0%.

That is what overfitting to a fixed opponent looks like with room to do it in. The extra capacity
did not buy a better model of the game; it bought a sharper model of a scripted bot, most of it
spent on the one board where the scripted bot is easiest to beat. Against an opponent that adapts,
what it learned is worth less than what the smaller network learned.

**Mean return did not report any of this.** 16.90 was the best number on the board while the policy
underneath it was the worst. Return against greedy saturates near 17 long before skill does, and
this run is the cleanest demonstration in the project that it cannot be used to rank anything.

## Why the arm was run

Its sibling [RaifuWars-RL-ActionScorer-Cover](https://huggingface.co/yotisstudios/RaifuWars-RL-ActionScorer-Cover)
tested whether the policy's collapse on cover-heavy boards was a *perception* failure — nothing in
the feature set describes terrain. This arm was the falsifier: if simply making the network bigger
moved the same boards, the perception story was wrong and the model had merely been undertrained.

Both arms launched together, on identical configurations, differing in exactly one variable each.
The capacity arm answered its question — capacity was not the limit, and taking more of it hurt.
The terrain arm did not answer its question, for a reason worth reading its card for: the
simulator emits no terrain, so its new features were constant zero for the entire run. **This arm
is the valid half of that pair**, since it changed no features.

## Training

- **8 hours**, **8,364 updates**, **9,635,328 agent decisions**, ~455 steps/sec, from scratch — a
  width change makes an existing checkpoint unusable as an initialiser.
- PPO, lr 5e-5, 6 envs × 192 steps, batch 128, against a scripted greedy opponent, in the
  [Hemlock](https://github.com/Yotis-Studios/raifusim) reimplementation.
- Boards: `Arboretum, Crossroads, Dustbowl, Glacier, Cornfield, Trench Warfare, Twin Rivers`.
- Final mean return 16.90, peak 17.99 at update 5,829.
- It ran ~8% fewer steps than the 58K arm in the same wall-clock, which is the entire cost of the
  extra capacity at this scale — the bottleneck is the simulator, not the forward pass.

COMMON

## Limitations, stated plainly

- **Do not use this as a playing policy.** Both 58K self-play checkpoints beat it by an order of
  magnitude head-to-head. It is published so the ablation is reproducible and so the
  return-does-not-rank-policies claim has a checkpoint attached to it.
- Trained against a **scripted greedy opponent only**. Its collapse against adaptive opponents is
  expected for that recipe; `ppo-sim` shows the same pattern less severely.
- The comparison is a **single seed per arm**. Two arms, one run each, so "bigger is worse" here is
  one observation and not a curve. What is solidly established is the weaker claim that motivated
  the arm: more capacity did not fix the boards the small net was failing on.
- 16 matches per board in the real game, ±21 points at 95%.
