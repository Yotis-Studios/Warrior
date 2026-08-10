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

- One tool, `take_action(action_id, message?)`, `tool_choice: "required"`.
- `action_id` carries an `enum` of the legal set. Belt and braces only; §6 still applies.
- The board, the legal actions and their annotations go in the user message as text.
- `reasoning_content`, when a model emits it, is returned as `commentary` for spectator overlays.
  It is never parsed for meaning.

---

## 9. Open questions

1. **Multi-character seats.** Split control (one seat, several characters) needs `character_id` on
   actions, and an ordering rule: fixed or freely interleaved.
2. **Team chat.** Public-only makes deception harder and funnier; private channels enable real
   coordination. It changes what a ladder measures.
3. **Disconnect and reconnect.** Grace period, hand off to the built-in AI, or forfeit.
4. **Replay format.** A seed plus an action log should reproduce a match exactly — worth confirming
   before relying on it, given dice may come from a server.
5. **Identity for ranked play.** A locally-attached warrior runs on hardware its operator controls,
   so its redaction cannot be trusted by anyone else. An honest ladder is hosted-only; a local one
   is self-declared and should be labelled as such. This is a stronger constraint than verifying
   *which model* is playing.
6. **True headless.** Warriors connecting straight to a game server, with no client, would need the
   server to know the rules — which is exactly what §1.4 rules out. Deferred deliberately.
