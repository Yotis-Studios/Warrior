# Warrior

**A protocol for letting a language model take a human seat in a game.**

A *warrior* is an agent playing one seat of a normal multiplayer match. It connects the way a
person does, sees what the person at that seat would see, and may only do what that person could
do. It is not a bot with engine access, and the protocol is built to make that enforceable rather
than aspirational.

Built for [Raifu Wars](https://raifuwars.com) first, but nothing here is Raifu Wars specific — the
game supplies state and a list of legal actions, the sidecar picks one.

**Status: 0.1 draft.** The protocol is specified and the reference sidecar works end to end against
a local model. The game-side client is not written yet.

- **[PROTOCOL.md](PROTOCOL.md)** — the wire contract
- **`sidecar/`** — reference implementation, stdlib only
- **`probe/`** — the experiments the design is based on
- **`tests/`** — end-to-end smoke test

---

## How it fits together

The game is the HTTP **client** and the sidecar is the **server** — the game calls out when it
wants an action. [PROTOCOL.md §2](PROTOCOL.md) explains why that direction rather than the obvious
one.

```
┌────────────┐   POST /v1/act        ┌───────────┐   /v1/chat/completions   ┌───────┐
│ game client│ ────────────────────► │  sidecar  │ ───────────────────────► │  LLM  │
│  (one seat)│ ◄──────────────────── │           │ ◄─────────────────────── │       │
└────────────┘   { action_id }       └───────────┘   tool_calls             └───────┘
  authority on rules,                 picks one of
  legality, and what                  the offered
  the seat may see                    actions
```

The game decides everything that matters. It computes what is legal, it redacts what the seat is
not entitled to see, and it validates whatever comes back. The sidecar chooses — nothing else.

## Quick start

No dependencies. Python 3.8+.

```bash
# A sidecar that plays random legal actions -- no model required.
python -m sidecar.server --policy random

# A sidecar backed by any OpenAI-compatible endpoint (llama.cpp, vLLM, ollama, an API).
python -m sidecar.server --policy llm --url http://127.0.0.1:8080

# Prove it works, over a real socket, with the real protocol.
python tests/smoke.py --policy random
python tests/smoke.py --policy llm
```

The random policy is not a toy. *"Do four seats of random legal actions finish a match"* is the
cheapest regression gate a game engine can have — it needs no GPU, and it catches the failure that
matters most: a state with no legal action out of it, which stalls a match forever.

## What the probes found

The design is not a matter of taste. `probe/` runs a realistic mid-match fixture against a local
model and the results decided three things:

**One tool, not one tool per action.** Given a conventional `attack(target_seat: int)` tool, a 27B
scored 88% legal — and the failures were `atk_seat5`, a seat that does not exist in a four-player
match, and on another run **its own seat**. An earlier probe with no target list invented
`target_id: "enemy_at_5_4"` from nothing. Collapsing every rules-affecting choice into an
enumerated `action_id` scored 16/16. Models confabulate into whatever gap a schema leaves, so the
protocol leaves none: the only free-form field in the whole thing is chat text, which is the one
value that cannot be illegal.

**A JSON-Schema `enum` does not bind the model.** Offered an enum containing only `end_turn` while
the prompt advertised an attack, the model returned the attack **6 times out of 6**. llama.cpp did
not compile it into a sampling grammar. The enum is worth sending as defence in depth, but the
client validates every returned id, always. This one is why the probe exists — it was about to
become a guarantee in the spec on the strength of eight agreeing samples.

**Thinking off, by default.** Reasoning on: 18.8s and 861 completion tokens per action. Off: 0.9s
and 37 tokens, with legality slightly *better*. That is the difference between a tournament that
runs over lunch and one that runs overnight. It stays available per seat, because a harder
decision on a stronger model may be worth twenty times the wall clock — but it is not the default.

A fourth result was pure self-inflicted: at `max_tokens: 900`, half the samples returned no tool
call at all. The model was still reasoning when the cap hit, and a truncated reply is
indistinguishable from a refusal once parsed. Set a generous cap and read `finish_reason`.

```bash
python probe/probe_action_format.py --trials 8    # which encoding survives contact
python probe/probe_constraints.py --trials 6      # is the enum enforced? what does thinking cost?
```

Both take `--url` and `--model`, so they are the first thing to run against a model you are
thinking of seating.

## Design principles

1. **You see what the player sees.** The client holds far more than the seat is entitled to — in
   Raifu Wars every card draw is broadcast to every client, so opponents' hands sit in memory on
   every machine. Withholding them is the *game's* job, and a deliberate feature rather than an
   accident. Redaction is a whitelist, never a copy-and-delete.
2. **The engine decides legality; the model only chooses.** The sidecar never pathfinds, never
   computes hit chance, never checks affordability.
3. **A warrior may only do what a human at that seat could do.**
4. **The rules live in exactly one implementation.** The sidecar does not reimplement the rulebook
   in a second language, and is never asked to check the engine's work.
5. **Latency is normal.** Nothing resolves synchronously.

## Repository layout

```
PROTOCOL.md          the wire contract -- read this first
sidecar/
  server.py          HTTP server implementing the protocol (stdlib only)
  policy.py          how a seat decides: random, first-legal, llm
probe/
  state_fixture.py   one realistic mid-match turn, shared by every experiment
  probe_action_format.py   named tools vs single-enum vs single-free
  probe_constraints.py     enum enforcement; cost of reasoning
tests/
  smoke.py           end-to-end over a real socket
```

## Prior version

This repository previously held a stub for a reinforcement-learning approach to the same goal.
That is preserved on the **`rl-archive`** branch. It was never implemented — every method body was
a comment — and the LLM design here needs different things from the client (an enumerated legal
set and a readable board, rather than tensor observations and millions of cheap steps).

Worth noting the two are not actually in conflict at the protocol level: Gym's
`step(action) -> (obs, reward, done, info)` and this turn loop are the same shape. As long as
action ids are canonically ordered and stable, an RL adapter remains possible later without the
protocol committing to reward shaping now. Stable ids also make replays diffable, which is worth
having regardless.

## License

GPL-3.0. See [LICENSE](LICENSE).
