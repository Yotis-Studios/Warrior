<!-- Shared tail sections. Not published on its own; pasted into each card so every repo stands
     alone on the Hub. Kept here so the architecture block cannot drift between the three. -->

## Architecture

Two towers and an interaction term. The state is embedded once, each candidate action is embedded,
and the score is their elementwise product — so nothing in the network knows how many actions there
are, which is the requirement: the legal set runs from 2 to ~670 between decisions and varies with
board size, dice roll and hand.

```
state  D_STATE  -> HIDDEN -> EMBED
action D_ACTION -> HIDDEN -> EMBED
head   EMBED*3  -> HIDDEN -> 1        softmax over exactly the N offered
value  EMBED    -> HIDDEN -> 1
```

Scoring rather than classifying makes an illegal action *unrepresentable* rather than merely
penalised, and lets the same weights run on a 17×21 board and a 27×27 one. A classifier over "all
possible actions" would need an output per tile per action type, would mask nearly all of them
every step, and would learn nothing transferable between boards.

## Usage

```bash
RW_FEAT_COVER=COVERFLAG python serve.py WEIGHTS --port 8901
# then point the game at http://127.0.0.1:8901 via the Warrior protocol
```

`serve.py` reads the architecture out of the checkpoint's own first layer rather than assuming
one, so it loads any of the published ActionScorer variants. The one thing it cannot infer is the
**feature width**, which is fixed at import time by `RW_FEAT_COVER`: set it wrong and the network
loads cleanly and reads the wrong numbers. It checks, and exits naming the flag to set.

## Reading the numbers on this page

Three different measurements appear in these cards and they do not agree with each other. That is
the most useful thing they have to say, so they are labelled rather than averaged:

- **Real game, vs the built-in AI.** The shipped game, over the Warrior protocol, one seat of four
  against three scripted opponents. 16 matches per board — a 95% interval of roughly ±21 points,
  enough to separate 50% from 13% and not enough to rank two policies a few points apart.
- **Sim head-to-head.** All four seats drawn from the policies under test, in the
  [Hemlock](https://github.com/Yotis-Studios/raifusim) reimplementation. This ranks policies
  against each other and says nothing about the real game.
- **Sim vs greedy.** One learner against three scripted bots. This is the number training optimises
  and it saturates: policies 30 points apart in head-to-head sit within 10 points of each other
  here.

Mean return is reported for completeness and should not be used to rank anything. Under self-play
it is pinned by construction — four copies of one policy produce exactly one winner — and against
greedy it saturates near 17 well before skill does.
