# Warrior Protocol

**Version 0.1 — draft.** The contract between a game client and a *sidecar* that plays one seat.

A **warrior** is an agent occupying a human seat. It is not a bot with engine access: it connects
as an ordinary client, sees what the player at that seat would see, and may only do what that
player could do. The protocol exists to make both of those enforceable rather than aspirational.

The game is the HTTP **client**; the sidecar is the HTTP **server**. That direction is deliberate
and §2 explains it.

---

## 0. Findings that shaped this

Every rule below with teeth came out of probing a local 27B (`Wichtel-Q4_K_M`, llama.cpp) against
a realistic mid-match fixture. `probe/` reproduces all of it. The results, because a protocol
written without them tends to assume the model is more careful than it is:

| Finding | Measured | Consequence |
|---|---|---|
| Free-form arguments get confabulated | One tool per action type with a `target_seat` integer: **88% legal over 8 trials**. The failures were `atk_seat5` (a seat that does not exist — the match has four) and, in an earlier run, `atk_seat1` (**its own seat**). A first probe with no target list at all invented `target_id: "enemy_at_5_4"` outright. | §4: **one** tool, and every rules-affecting value is an enumerated id. Never a free integer, never a free coordinate. |
| A schema `enum` does not bind | Offered an `enum` of exactly `["end_turn"]` while the prompt advertised an attack, **6/6 samples returned `atk_seat0`**. llama.cpp did not compile the enum into a sampling grammar. | §6: the enum is defence in depth. **The client validates every returned id, always.** An unvalidated id is a cheat vector, not a formatting slip. |
| Truncation is indistinguishable from refusal | At `max_tokens: 900`, 3 of 6 samples returned no tool call. At 2500, zero did. The encoding was never the problem. | §7: sidecars must set a generous cap and report `finish_reason`. A missing action is not evidence about the model until truncation is ruled out. |
| Reasoning is nearly all of the latency | Thinking on: 18.8s, 861 completion tokens. Thinking off: **0.9s, 37 tokens**, and legality went 5/6 → 6/6. | §7: thinking is **off by default**, opt-in per seat. It is the difference between a tournament that runs over lunch and one that runs overnight. |

The single-tool encoding scored **16/16 legal** across both runs. The conventional named-tools
encoding scored 7/8. That gap is the protocol's whole reason for looking the way it does.

> One honest caveat, kept because it is easy to lose: "stayed inside the enum" would have been
> evidence, not proof — a model that would have complied anyway is indistinguishable from a
> grammar that stopped it. That test only became conclusive because it *failed*.

---

## 1. Principles

1. **You see what the player sees.** The client holds far more than the seat is entitled to — in
   Raifu Wars every card draw is broadcast to every client, so opponents' hands are in memory on
   every machine. Hiding them is the *game's* job and it is a feature, not an accident of
   architecture. See §5.
2. **The engine decides legality; the model only chooses.** The sidecar never pathfinds, never
   computes hit chance, never checks affordability. It receives a complete list of what a human at
   this seat could do and picks one.
3. **A warrior may only do what a human at that seat could do.** This is the same invariant the
   game already enforces on its CPU players. It is not a courtesy to opponents; it is what makes a
   result meaningful.
4. **Rules live in exactly one implementation.** The sidecar does not reimplement the rulebook in
   another language. It cannot check the engine's work and is not asked to.
5. **Latency is normal.** Nothing resolves synchronously. Dice may be minted by a server, an
   action may take a round trip, and the sidecar may think for seconds.

---

## 2. Why the game calls out

The obvious design — the client hosts a REST API and agents connect to it — is the wrong one here,
for four reasons:

- **The client is a game engine.** GameMaker offers raw TCP (`network_create_server`) and no HTTP
  server. Hosting one means hand-rolling HTTP/1.1 framing and connection lifetime inside the game
  loop. Being an HTTP *client* is already a solved, in-use path (`http_request` + the async HTTP
  event).
- **Async falls out for free.** `http_request` returns immediately and the reply arrives in an
  event later — exactly the shape a turn loop with a thinking opponent needs.
- **The client owns the pixels.** Screenshots for a multimodal warrior are a field in a request the
  client is already sending, not a second endpoint with its own synchronisation problem.
- **It matches the mental model**: *"hey, your turn, here's what you can do."*

"The game exposes a standard API" still holds at the system level — the sidecar ships alongside the
client and is the stable, documented surface. Third parties implement one endpoint in any language.

```
   ┌────────────┐   POST /v1/act        ┌───────────┐   /v1/chat/completions   ┌───────┐
   │ game client│ ────────────────────► │  sidecar  │ ───────────────────────► │  LLM  │
   │  (one seat)│ ◄──────────────────── │           │ ◄─────────────────────── │       │
   └────────────┘   { action_id }       └───────────┘   tool_calls             └───────┘
     authority on          §3–§7                              §8
     rules + legality
```

---

## 2b. Transport: one long-lived TCP connection

**The game opens a single TCP connection and holds it for the life of the process.** Every
decision travels over that one connection. This replaced one HTTP request per decision.

Why: not speed. A localhost HTTP round trip is single-digit milliseconds against 600–4000 ms of
model call — under 0.3% of a decision, so the transport was never the cost. What one connection
per decision *did* cost was churn. Across one screening the sidecar logged 38 connection resets
and answered **2,357 of 5,999 requests from its retry cache**. No match failed — the idempotency
cache absorbed all of it — but that is thousands of connections and thousands of duplicate
deliveries being absorbed, and each one is a chance for the absorbing to be wrong. A run that used
to open 138 connections now opens one.

### Framing

4-byte **big-endian** length, then exactly that many bytes of UTF-8 JSON:

```
[len:u32be][{"path":"/v1/act","body":{ ...the same body as the HTTP POST... }}]
[len:u32be][{"status":200,"body":{ ...the same reply as the HTTP response... }}]
```

`path` and `body` are the endpoint and payload from §3 unchanged, so a sidecar has **one**
implementation of every route and the transport is genuinely just a transport.

A length prefix rather than a delimiter, because the payload is JSON carrying arbitrary text —
chat messages included — so any sentinel byte needs escaping, and an unescaped one is a parser
that stops mid-message.

### Rules for both ends

- **Read the length first, then exactly that many bytes.** `recv` returns what has arrived, not
  what was sent: one frame may span several reads and several frames may share one. A reader that
  assumes one read is one message works until a payload crosses a packet boundary — and the
  payloads here reach ~35 KB.
- **Every frame carries an `id`, and the reply echoes it.** Replies are paired BY THAT ID, never
  by arrival order. "The game asks one question and waits" is not true in the failure cases: its
  watchdog re-asks while an earlier request is still in flight, so order-pairing binds an action
  chosen for one board to a decision about another. The client rejects it as not offered and
  retries, which puts a second request in flight, which makes it worse. Measured at 51% of
  decisions rejected against 0.8% over HTTP.
- **Refuse an implausible length** rather than allocating it. The reference implementation caps at
  8 MB.
- **A closed connection is not an error.** The game exits between matches; the sidecar outlives it
  and accepts the next one.

### Port, and what stays on HTTP

The TCP port defaults to the **HTTP port + 1** on both ends, so parallel arms stay distinct
without a second setting to keep in step. `--tcp-port 0` disables it; `RW_WARRIOR_TCP=0` makes the
game use HTTP.

**HTTP stays up beside it** and serves the same routes through the same dispatch. `GET /v1/health`
over HTTP is what readiness probes and `curl` use, and a sidecar that cannot be reached by TCP is
still a sidecar the game will talk to — the client falls back rather than forfeiting its turn.

## 3. Endpoints

All JSON, all `POST` except health. Paths are versioned; a sidecar rejecting an unknown
`protocol_version` should say so in `/v1/health` rather than failing mid-match.

| Endpoint | When | Returns |
|---|---|---|
| `GET /v1/health` | before a match; handshake and capabilities | `{protocol_version, name, model, capabilities}` |
| `POST /v1/match/start` | seat assigned, match beginning | `{ok: true}` |
| `POST /v1/act` | **the core call** — an action is wanted | `{action_id, args?}` |
| `POST /v1/event` | something happened that needs no action | `{ok: true}` |
| `POST /v1/match/end` | match over | `{ok: true}` |

`/v1/event` exists so a warrior has continuity between its turns — chat arriving, a KO, a point
changing hands — without the client having to pretend it is that warrior's turn. Sidecars may
ignore it; clients must not depend on a reply beyond the ack.

### 3.1 Capabilities

```json
{
  "protocol_version": "0.1",
  "name": "warrior-reference",
  "model": "Wichtel-Q4_K_M.gguf",
  "capabilities": {
    "vision": false,
    "chat": true,
    "commentary": true,
    "max_deadline_ms": 30000
  }
}
```

A client MUST NOT send a `screenshot` to a sidecar advertising `vision: false`. It is wasted
bandwidth and, on a text-only model, wasted context.

---

## 4. `POST /v1/act`

```json
{
  "protocol_version": "0.1",
  "request_id": "m41-t23-a2",
  "match_id": "m41",
  "seat": 1,
  "reason": "turn_start",
  "deadline_ms": 30000,
  "state": { "...": "see §5" },
  "available_actions": [
    {
      "action_id": "atk_seat0",
      "type": "attack",
      "label": "Attack Arisaka at (10,4)",
      "hit_chance": 0.45,
      "expected_damage": 1,
      "note": "range 4, distance 3.6, target in the open"
    },
    { "action_id": "rush_1", "type": "rush", "label": "Rush — spend your shot for a second move" },
    { "action_id": "end_turn", "type": "end_turn", "label": "End your turn" }
  ],
  "last_action": null,
  "screenshot": null
}
```

`reason` is one of:

- `turn_start` — the turn has just opened
- `action_result` — the previous action resolved; `last_action` carries what happened
- `retry` — the previous reply was rejected; `last_action.error` says why (§6)

Response:

```json
{ "action_id": "atk_seat0", "args": {}, "commentary": "optional, for spectators" }
```

### 4.1 The enumeration rule

**Every value that affects the rules is an enumerated `action_id`. The only free-form field in the
entire protocol is chat text.**

This is the direct consequence of §0. A free integer became a nonexistent seat and then the
warrior's own seat; a free string became an invented target. So:

- attack targets → one `action_id` per legal target
- cards → one per playable card, and separately per discardable card
- **movement destinations → one per reachable tile.** This sounds explosive and is not: movement
  is dice-limited, so the reachable set is bounded and small. Enumerating it is cheaper than
  letting a model emit a coordinate.
- placement/throw targets → one per legal tile, same reasoning
- chat → `action_id: "chat_1"` with `args.message` as free text

Chat is safe to leave free precisely because it is the one thing that cannot be illegal. Table
talk is the game; a badly-formed coordinate is a cheat.

`args` therefore carries **only** `message`, and only for chat actions. If a future action needs a
parameter that is not enumerable, that is a signal the action was modelled wrong.

---

## 5. State, and what is deliberately missing

`state` is whatever the seat's own player can see. The protocol does not enumerate every field —
that is per game — but it fixes the shape:

```json
{
  "turn": 23,
  "self":    { "seat": 1, "tile": [7,2], "health": 3, "ammo": 1, "hand": [ "..." ], "this_turn": {} },
  "players": [ { "seat": 0, "tile": [10,4], "health": 4, "cards_in_hand": 2 } ],
  "board":   { "ascii": "...", "legend": {}, "points": [] },
  "events_since_last_turn": [ "Arisaka shot you for 1 damage from (10,4)." ],
  "chat": [ { "seat": 3, "text": "truce on the left?" } ]
}
```

### 5.1 Say what the rules use, not what the HUD shows

Three fields were added after a referee (`check-rules.py`) compared offers against the state they
were made in and found the payload naming a quantity the rules do not use:

| field | why it exists |
|---|---|
| `self.move_budget` | `move_roll` is the **die**. The budget is the roll plus a character modifier, forced to 1 by one card and raised by another. On 25% of decisions they differ, so an agent computing its own reachable set from the roll computed the wrong one. |
| `board.points[].radius` | A point covers a **square**, not a tile. The same number was already being sent as `stars_per_turn`, so one field carried two unrelated meanings and the payload named only the one that decides nothing. Tiering is legal anywhere inside your base's square — a fact no agent could read. |
| `self.abilities` | The numbers say `range: 20`. Only this says that 20 is *+2 over everyone else*. A character is drawn per match and its modifiers are the largest single difference between two seats. |

The rule they share: **if the engine derives a quantity before applying a rule, send the derived
one.** Sending the base and expecting the agent to re-derive it means shipping a second copy of the
rule in every client that talks to you, and that copy is never updated when yours changes.

`self.school` and `self.doctrine` are flavour, not mechanics — they are what the seat is *like*
rather than what it can do, and they exist so a persona has something to work with.

### 5.2 `chat` is two-way, and the spec had only ever been half-built

This document specified `state.chat` from the beginning and the game never populated it. The
sidecar had a renderer for it, so a warrior could **talk and could never listen** — for the whole
life of the protocol, with nothing failing. A field-variation checker even reported `chat` as never
varying and it was dismissed as "only present online". It was absent online too.

Both halves are now real:

- **Reading.** `state.chat` carries the last few lines as `{seat, text}`. `seat` is `-1` for system
  lines (the kill feed, the card feed), so it joins onto the roster the agent already has. Only the
  last few: an unbounded log grows without limit across a long match and pushes the board out of
  the model's attention, and what matters for a reply is what was just said.
- **Writing.** A `chat` action with `args.message`. It does **not** consume the turn — the agent is
  asked to act again immediately — which is the property that makes talking affordable at all.

If a game localises its strings, whatever the agent reads should be generated in ONE language
regardless of the client's locale. Otherwise the machine's locale silently decides what language a
corpus is collected in, and any dataset built from it inherits that permanently.

Two rules carry the weight:

**Redaction is a whitelist.** The client builds `state` by naming what goes in, never by copying
its world and removing what should not be there. Everything is in memory — hands, deck order, other
seats' internals — and a blacklist leaks the moment anyone adds a field. Note the example above:
opponents have `cards_in_hand: 2`, a count, never the cards. Getting this wrong does not crash
anything; it silently makes a hidden-information card worthless and every result meaningless.

**`events_since_last_turn` is load-bearing, not a convenience.** It is how a warrior re-enters
cheaply instead of replaying history. On the probe fixture the whole payload was ~1400 prompt
tokens; a naive full-history transcript would exceed a small model's context inside twenty turns.

`board.ascii` is for the model's benefit and is **not authoritative** — coordinates in `players`,
`self` and `available_actions` are. Where they disagree, the structured fields win.

---

## 6. Validation — non-negotiable

> The `enum` in the tool schema **does not bind the model**. This was measured, not assumed: 6/6
> samples returned an action that the enum excluded. Treat a returned `action_id` as untrusted
> input from a program you did not write.

### 6.0 Never ask with an empty legal set

**A client MUST NOT send `/v1/act` when `available_actions` would be empty.** There is no legal
answer, so every possible reply is illegal by construction and the only reachable outcome is a
forfeit — recorded against a seat that did nothing wrong.

This is not hypothetical. Of the first twenty-five real Raifu Wars turns replayed at a live model,
six "failed" and **all six were this**: knocked-out seats, where the respawn roll is on a timer and
a human could not have pressed anything either. The other nineteen were nineteen for nineteen.
Sending them was the client's bug, and reading them as the model's failure would have been the
wrong lesson entirely.

If a seat has nothing to do, the client takes the turn forward itself.

### 6.1 Validate every reply

On receiving a response the client MUST:

1. Reject any `action_id` not in the `available_actions` it just sent. Not "fuzzy match", not
   "nearest tile" — reject. Re-deriving intent from a malformed id is how a warrior takes an
   action no human could.
2. Re-issue `/v1/act` with `reason: "retry"` and `last_action.error` describing the rejection.
3. After `max_retries` (default **3**), take the safe default action — normally `end_turn` — and
   log it. A warrior that cannot produce a legal action forfeits its turn; it never stalls a match.
4. Enforce `deadline_ms` independently. A sidecar that does not answer is a sidecar that has
   crashed, and the turn timer is already the game's answer to that.

Rejections are **game feedback, not faults**. They do not consume the turn and should not raise
errors at the operator. But every one is counted, and a warrior whose rejection rate is
non-trivial is a warrior that is not really playing.

---

## 7. Sidecar obligations

- **Set a generous generation cap.** 2500 tokens was enough where 900 truncated 50% of samples.
  A truncated reply looks exactly like a refusal and will be misdiagnosed as one.
- **Thinking off by default.** Measured at 20× the latency for no accuracy gain on this fixture.
  Expose it per seat, because a stronger model on a harder decision may well be worth the wait —
  but the default must be the fast one.
- **Be idempotent on `request_id`.** Clients retry. A sidecar that treats a retry as a new decision
  will contradict itself mid-turn.
- **Never assume the previous request arrived.** `/v1/act` carries everything needed to act.
  A sidecar may keep history keyed by `match_id`, but must be correct without it.
- **Answer within `deadline_ms` or not at all.** A late answer is worse than none: the turn has
  moved on and the action is no longer legal.

---

## 8. Sidecar → model

Not part of the wire contract — a sidecar may be a language model, a scripted policy, or a random
stub for CI — but the reference implementation works like this, and the shape is what §0 measured:

- One tool, `take_action(action_id, message?, why?, notes?)`, `tool_choice: "required"`.
- `action_id` carries an `enum` of the legal set. Belt and braces only; §6 still applies.
- The board, the legal actions and their annotations go in the user message as text.
- `reasoning_content`, when a model emits it, is returned as `commentary` for spectator overlays.
  It is never parsed for meaning.

### 8.1 A second tool: consulting an expert

**Not protocol.** The game knows nothing about this; it is entirely between a sidecar and its
model. It is documented here because it changes what the sidecar is for.

A sidecar may hold a second policy — in the reference implementation an RL action-scorer that wins
~55% of its matches against the game's built-in AI, where that AI wins ~28% of its own — and offer
it to the model as `consult_expert()`. The model asks, reads, and then still calls `take_action`.

The obvious composition is a **router**: let the strong policy take the actions and the model do
the talking. That throws away the interesting half. What each side has is genuinely different:

| the expert has | the model has |
|---|---|
| a calibrated distribution over the legal set | a plan that survives across turns |
| a value estimate for the position | the ability to read card text |
| ~55% against the built-in AI | the table — what other players just said |
| no memory, no plan, and no words | judgement about when the above matters |

So it returns a **distribution and a position value**, not an order:

```
move_4_2   Move to (4,2)   99% of its confidence
It rates your POSITION at 14.3 (a winning position scores about 17, a losing one about 4).
It is confident. Disagreeing is a real decision, not a coin flip.
```

Confidence is the part that makes it useful. A 70/30 split is information the model never had —
previously it chose among forty ids with no sense of which two were even in contention — and a 30%
action is a defensible choice when the model knows something the expert cannot see. A teammate
saying "cover the left" is exactly that: the expert has no channel for it, no notion of an ally,
and no memory of the request. Overriding for that reason is not second-guessing a stronger player.

Two things follow for anyone building this:

- **The consult must be free.** It costs a round trip and no turn, which is only affordable because
  the game has no per-action clock (§2). Bound it — two consults per decision — because a model
  that keeps asking instead of acting is stuck and the game is holding its turn open meanwhile.
- **Record whether it listened.** "The model consults an expert" is worth nothing on its own. The
  measurable quantities are how often it asks, how often it follows *when the expert is confident*,
  and whether the matches it overrode in went better or worse. The control is the expert playing
  alone: a model on top of a 55% policy must beat 55%, and one that overrides good advice will
  score below the expert it is holding.

A persona, if a sidecar offers one, belongs after the rules and scoped to presentation. A model
told to be in character still may not invent an `action_id`.

---

## 8b. Who should actually hold the seat

The protocol does not care what answers `/v1/act`. This section records what was measured, because
the obvious answer turned out to be wrong twice and finding that out cost real time.

**A language model alone plays one board and not the others.** Three frontier models, 8 matches per
board, against the game's built-in AI, where a fair share for one seat of four is 25%:

| | Crossroads | Arboretum | Islands |
|---|---|---|---|
| cover on the board | **0 tiles** | 172 (37%) | 321 (33%) |
| gpt-5.6-luna / deepseek-v4-flash / gemini-3.5-flash-lite | **71%** | 8% | **0%** |
| a 57,730-parameter RL policy | 69% | **50%** | **31%** |

Crossroads is the only board in the pool with no cover at all, so every shot is clean
line-of-sight and four knockouts wins outright. The models were not playing it well; they were
kill-rushing a board that permits it. Given cover, they stop scoring entirely.

**Telling the model how to win made it worse.** A prompt section explaining the win condition --
which tier track you are on, that points pay per point per turn, that cover decides fights -- was
written from the failures above and then measured: 3/24 with it against 8/32 without, and the one
board the model was good at halved. The knowledge was never the bottleneck.

**Fine-tuning a 4B on the RL policy's own decisions did not transfer its play.** 5,423 rows, the
policy's probability mass inline on every action, 91.2% of targets its top choice. In live play the
fine-tune agreed with that advice **34-60%** of the time, and the agreement tracked the win rate
almost exactly:

| | follows the advice | wins |
|---|---|---|
| Crossroads | 60% | 62% |
| Arboretum | 45% | 0% |
| Islands | 34% | 0% |

It was trained on positions the expert reached, then asked to play from positions its own weaker
choices produce. One divergence lands it somewhere the expert never was, and it compounds. That is
the standard behaviour-cloning failure and it is not something more supervised data fixes.

### So: the policy plays, the model talks

`--policy hybrid` gives the seat to the RL policy and calls a language model only when there is
something to SAY.

    --policy hybrid --expert CKPT [--url URL | --openrouter MODEL] [--skill 1.0]

This is also the only arrangement usable against a human. The protocol asks one decision per
ACTION, so a match is ~90 round trips; at 3s a call that is five to fourteen minutes of a person
watching a spinner. The policy answers in **2ms**, and the model is asked a handful of times a
match.

Two things it does deliberately:

- **`--skill` below 1.0 samples the policy's own distribution instead of its argmax.** An
  EV-maximising opponent can be strictly better at winning and worse to play against -- risking a
  shot and missing is a large part of the fun. It samples from the policy's own mass rather than
  mixing in random legal actions, so a weaker opponent is still recognisably playing rather than
  occasionally walking into a wall.
- **Chat fires on a reason, never a timer** -- somebody spoke, the match opened, an event happened,
  low health, one tier from winning -- and that reason is what the model is told to react to. A
  line with no reason behind it is the filler that makes a bot tiresome to sit with. Failure to
  compose is silence, never a canned line: a fallback is indistinguishable from a working model
  saying something bland.

**The policy is never offered the chat action.** It has never seen one, so whatever score it
assigns is an artefact of features that do not describe talking. Chat is also the only action that
does not consume the turn, so an agent that rates it top rates it top again, forever -- 58,317 chat
actions in a single match, which reached turn 4 in ten minutes. The game now offers chat once per
turn (section 3), and the hybrid seat filters it out of the policy's ranking as well.


## 9. Open questions

1. **Multi-character seats.** Split control (one seat, several characters) needs `character_id` on
   actions, and an ordering rule: fixed or freely interleaved.
2. **Team chat.** Public-only makes deception harder and funnier; private channels enable real
   coordination. It changes what a ladder measures. Public chat is implemented (§5.2); private is
   not, and that is a design choice rather than a technical gap.
3. **Disconnect and reconnect.** Grace period, hand off to the built-in AI, or forfeit.
4. **Replay format.** A seed plus an action log should reproduce a match exactly — worth confirming
   before relying on it, given dice may come from a server.
5. **Identity for ranked play.** A locally-attached warrior runs on hardware its operator controls,
   so its redaction cannot be trusted by anyone else. An honest ladder is hosted-only; a local one
   is self-declared and should be labelled as such. This is a stronger constraint than verifying
   *which model* is playing.
6. **True headless.** Warriors connecting straight to a game server, with no client, would need the
   server to know the rules — which is exactly what §1.4 rules out. Deferred deliberately.

7. **Annotated screenshots.** The `screenshot` field exists and is unused. The interesting version
   is not a raw frame but one the game has **labelled** — capture points, your base, enemies, the
   objects that block a shot — drawn on by the client, which already knows where all of them are.

   The motivation is measured rather than speculative. A warrior given only coordinates moved on
   **24%** of its decisions against the built-in AI's **46.7%**, and rushed on 0.3% against 9%.
   Both are what "cannot tell which direction is forward" predicts: it is not choosing badly
   between destinations, it is barely choosing destinations at all. Spatial relationships are
   native to an image and have to be reconstructed from a coordinate list, and reconstruction is
   the step that appears to be failing.

   Three things would decide whether it works:

   - **Labels must be the same identifiers the action list uses.** If the image says "Point A" and
     the actions say `move_6_6`, the model has to bridge two namespaces and will do it wrong. Label
     with coordinates, or carry the letter into the action note — but pick one vocabulary.
   - **It shows more than a human sees**, which is a decision rather than an oversight. A player
     does not have "3 tiles to Point A" floating over the board. The same call was already made for
     `point.value`, and the honest framing is that a warrior with no vision is being compensated,
     not privileged — but it should be deliberate and recorded, like `hint_level`.
   - **Test whether the ASCII board is used at all first.** It is already in every prompt. If a
     model ignores a grid it can read, an image it must interpret is unlikely to land — and that
     experiment costs nothing, where rendering an annotated frame is real client work.

   Out of scope for 0.1. The field is reserved so that adding it later is not a protocol change.
