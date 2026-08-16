---
license: gpl-3.0
library_name: pytorch
tags:
- reinforcement-learning
- ppo
- self-play
- sim-to-real
- games
- raifu-wars
- warrior-protocol
pipeline_tag: reinforcement-learning
---

# Raifu Wars — RL Action Scorer (Self-Play 2)

A **57,730-parameter** policy that plays a seat in [Raifu Wars](https://raifuwars.com), a turn-based
strategy game, through the [Warrior protocol](https://github.com/Yotis-Studios/Warrior).

Continues [RaifuWars-RL-ActionScorer-SelfPlay](https://huggingface.co/yotisstudios/RaifuWars-RL-ActionScorer-SelfPlay)
for a further **12 hours and 29.6M agent decisions** of self-play. Same architecture, same
features, same seven boards. The only change is more of the same training.

It is a **better policy overall and a differently-shaped one**, and the shape is the interesting
part: it did not improve uniformly, it traded one board for another.

## Real game, vs the built-in AI

16 matches per board, one seat of four, over the Warrior protocol. Chance is 25%.

| board | cover | predecessor | **this model** | |
|---|---|---|---|---|
| Islands | 321 (33.4%) | 5/16 — 31% | **13/16 — 81%** | improved, p=0.011 |
| Crossroads | **0 (0.0%)** | 11/16 — 69% | **12/16 — 75%** | unchanged, p=1.0 |
| Arboretum | 172 (37.2%) | **8/16 — 50%** | 2/16 — 12% | regressed, p=0.054 |
| **overall** | | 24/48 — 50% | **27/48 — 56%** | |

Islands is the largest single-board gain any change has produced in this project, and Arboretum is
the largest loss. Neither the Islands gain nor the overall figure is in doubt (p=0.011 and
p=0.000004 against chance). **The Arboretum regression is not established at p<0.05** — 2/16 is
also not significantly *below* chance (p=0.20) — and 16 matches cannot resolve it further. Treat it
as a strong signal that wants a longer run, not a settled number.

### What changed in how it plays

| board | stars | kills | deaths | |
|---|---|---|---|---|
| Islands | 2095 | **0.0** | **0.0** | never fires a shot, wins 81% |
| Crossroads | 1960 | 6.9 | 1.2 | kill-rush, up from 6.1 |
| Arboretum | 1462 | 1.4 | 1.6 | dies more than it kills |

On Islands it records **zero kills and zero deaths across sixteen matches** and wins four out of
five. It is not fighting at all; it is taking and holding capture points while three scripted
opponents shoot each other. Its predecessor was already drifting that way (0.1 kills) — this run
completed the move.

That is the whole story of this checkpoint. Self-play pushed it toward two pure strategies —
territorial where cover permits it, lethal on the open board — and Arboretum is the board that
rewards neither cleanly. The predecessor's 50% there was a middling policy doing a bit of both.

## Sim results, and why they disagree

| | **this model** | predecessor | `ppo-sim` | `bignet` |
|---|---|---|---|---|
| head-to-head, 744 matches | **48.1%** | 42.3% | 13.3% | 3.8% |
| vs three greedy bots, 400 matches | 77.8% | **82.0%** | **84.5%** | 71.2% |

The two columns rank the field almost in reverse. `ppo-sim` is the best policy in the game against
scripted opponents and close to the worst against real ones; this model is the reverse. Both facts
are real and they are measuring different things — beating a fixed weak opponent is not the same
skill as beating a policy that adapts.

**Neither sim column predicted the real-game result.** In-sim head-to-head ranks this model first;
in the real game it lost a board outright. The sim's state payload contains no terrain at all (see
Limitations), so a self-play run tuned inside it optimises against boards that are all effectively
open — and Arboretum, the most cover-dense board evaluated, is where it paid.

## Training

- Initialised from the predecessor's `last.pt`, then **12 hours**, **9,647 updates**,
  **29,635,584 agent decisions**, 126,728 matches, ~1,374 steps/sec.
- PPO, lr 5e-5, 12 envs × 256 steps, batch 256, in the
  [Hemlock](https://github.com/Yotis-Studios/raifusim) reimplementation of the game.
- Boards: `Arboretum, Crossroads, Dustbowl, Glacier, Cornfield, Trench Warfare, Twin Rivers`.
- Seat win rates at the end of training: `{0: 0.336, 1: 0.245, 2: 0.180, 3: 0.309}` — seat 2 is
  the hard seat and stayed the hard seat.

**Use `last.pt`, which is what this repo ships.** Checkpoint selection by mean return is
meaningless under self-play: four copies of one policy always produce exactly one winner, so the
terminal term is pinned at 10/4 = 2.5 and the rest is noise. This run's `best.pt` was written at
**update 28 of 9,647** — 0.3% of the way in — and was never beaten. Mean return over the whole run
moved 11.55 → 10.92, i.e. downward, while the policy got measurably stronger.

COMMON

## Limitations, stated plainly

- **It lost Arboretum.** 12% against the predecessor's 50%. That is the single reason not to treat
  this as a drop-in replacement, and on a three-board average it is still ahead (56% vs 50%). Which
  checkpoint you want depends on the board.
- **The sim it trained in has no terrain.** Its `board` payload carries `width`, `height` and
  `points` and no map at all, so cover is invisible to anything trained there. Measured: the
  terrain features read **0.000 across 6,001 sim states** against **0.376** on the real game's
  Arboretum. In-sim, every policy tested scores 74–79% on Arboretum while the real game spreads
  them 12–50%. Sim results cannot rank policies on cover-heavy boards, and this run is the clearest
  demonstration of that so far.
- **16 matches per board** is ±21 points at 95%. It separates 81% from 31%; it cannot rank two
  checkpoints a few points apart.
- **Pure self-play against the current policy only**, no opponent pool. The regression on one board
  while two improved is consistent with narrowing, which is the risk this design carries.
- Dustbowl, Glacier, Cornfield, Trench Warfare and Twin Rivers were trained on and **not evaluated
  in the real game**.
