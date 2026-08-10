"""Skirmish: a small, complete game that implements the CLIENT half of the Warrior protocol.

WHY THIS EXISTS

A protocol document describes a contract; it cannot show that the contract is implementable, and
it certainly cannot show that a model can play through it. Writing the client half against a real
game engine first (Raifu Wars, in GML) would have meant discovering the protocol's mistakes in the
least convenient language available, entangled with an existing UI.

So Skirmish is the protocol's own reference implementation of the awkward side. It is deliberately
NOT Raifu Wars -- if the protocol only fits the game it was extracted from, it is not a protocol.
But it keeps the features that make Raifu Wars hard to serve:

  - hidden information that is trivially available in memory (every seat's hand is right here)
  - a legal action set that changes shape every action, not just every turn
  - dice, so a plan cannot be computed once at turn start
  - actions with targets, which is where a model most wants to invent an identifier

Everything here is engine-side truth. The seat's view is built by `redacted_state`, and that
function is the only thing allowed to decide what a warrior may see.

The rules, in full: two to four seats, one unit each, on a grid with walls. On your turn you may
move once (a dice roll caps the distance), attack once if a target is in range and you have ammo,
reload instead of attacking, play a card, and chat freely. Last seat standing wins.
"""

import random

# Action types are protocol-level vocabulary, not game-level. A sidecar written against Skirmish
# should transfer to any game that uses the same types -- which is the point of naming them here
# rather than inventing per-game strings.
MOVE, ATTACK, RELOAD, PLAY_CARD, CHAT, END_TURN = (
    "move", "attack", "reload", "play_card", "chat", "end_turn")

CARDS = {
    "c_patch":   {"name": "Patch",    "text": "Heal 2 HP.",                       "cost": 0},
    "c_scope":   {"name": "Scope",    "text": "+2 range until end of turn.",      "cost": 0},
    "c_dash":    {"name": "Dash",     "text": "Move again this turn.",            "cost": 0},
    "c_ambush":  {"name": "Ambush",   "text": "Your next attack deals +1 damage.", "cost": 0},
}

DEFAULT_MAP = [
    "..........",
    "..##...#..",
    "..........",
    "...#..##..",
    "..#....#..",
    "..........",
    "..#...##..",
    "..........",
]


class Unit:
    def __init__(self, seat, name, x, y):
        self.seat = seat
        self.name = name
        self.x, self.y = x, y
        self.hp = 5
        self.max_hp = 5
        self.ammo = 2
        self.max_ammo = 2
        self.base_range = 4
        self.range_bonus = 0
        self.damage_bonus = 0
        self.hand = []
        self.kills = 0
        self.alive = True
        # per-turn
        self.moved = False
        self.attacked = False
        self.played_card = False
        self.extra_move = False

    @property
    def range(self):
        return self.base_range + self.range_bonus

    def start_turn(self):
        self.moved = self.attacked = self.played_card = False
        self.extra_move = False
        self.range_bonus = 0


class Skirmish:
    def __init__(self, seats=4, seed=0, grid=None):
        self.rng = random.Random(seed)
        self.grid = list(grid or DEFAULT_MAP)
        self.h = len(self.grid)
        self.w = len(self.grid[0])
        self.turn = 0
        self.match_id = f"skirmish-{seed}"
        self.events = []            # since each seat's last turn: seat -> list
        self.chat_log = []
        self.winner = None
        self.action_count = 0

        spawns = [(0, 0), (self.w - 1, self.h - 1), (self.w - 1, 0), (0, self.h - 1)]
        names = ["Red", "Blue", "Green", "Gold"]
        self.units = []
        for s in range(seats):
            x, y = spawns[s % len(spawns)]
            u = Unit(s, names[s % len(names)], x, y)
            u.hand = [self.rng.choice(list(CARDS)) for _ in range(2)]
            self.units.append(u)
        self.pending = {u.seat: [] for u in self.units}
        self.current = 0
        self.move_roll = 0

    # -- board ---------------------------------------------------------------

    def wall(self, x, y):
        return not (0 <= x < self.w and 0 <= y < self.h) or self.grid[y][x] == "#"

    def occupied(self, x, y):
        return any(u.alive and u.x == x and u.y == y for u in self.units)

    def unit(self, seat):
        return self.units[seat]

    def living(self):
        return [u for u in self.units if u.alive]

    def reachable(self, unit, steps):
        """Breadth-first, walls block, other units block. Returns list of (x, y).

        Enumerated rather than described because PROTOCOL.md forbids handing a model a free
        coordinate: a destination it invents is a destination the rules never approved.
        """
        seen = {(unit.x, unit.y): 0}
        frontier = [(unit.x, unit.y)]
        out = []
        for _ in range(steps):
            nxt = []
            for (x, y) in frontier:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in seen or self.wall(nx, ny) or self.occupied(nx, ny):
                        continue
                    seen[(nx, ny)] = 1
                    nxt.append((nx, ny))
                    out.append((nx, ny))
            frontier = nxt
        return out

    def line_blocked(self, ax, ay, bx, by):
        """Crude supercover line: sample along the segment and reject if it crosses a wall."""
        steps = max(abs(bx - ax), abs(by - ay)) * 3 or 1
        for i in range(1, steps):
            t = i / steps
            x = round(ax + (bx - ax) * t)
            y = round(ay + (by - ay) * t)
            if self.wall(x, y):
                return True
        return False

    def targets(self, unit):
        out = []
        for other in self.living():
            if other.seat == unit.seat:
                continue
            d = abs(other.x - unit.x) + abs(other.y - unit.y)
            if d <= unit.range and not self.line_blocked(unit.x, unit.y, other.x, other.y):
                # Nearer is likelier. A number the model is given rather than one it derives --
                # principle 2: the engine computes, the model chooses.
                chance = max(0.2, min(0.95, 1.0 - 0.12 * (d - 1)))
                out.append((other, round(chance, 2), d))
        return out

    # -- the legal set -------------------------------------------------------

    def available_actions(self):
        """Everything the seat could do right now, and nothing else.

        This is the protocol's load-bearing function. Two properties matter more than the
        contents: it is PURE (asking must not change the game -- a predicate with a side effect
        turns "what may I do" into "I did something"), and it is COMPLETE (anything absent is
        thereby forbidden, so an omission is a rules change).
        """
        u = self.unit(self.current)
        acts = []

        if u.alive and (not u.moved or u.extra_move):
            for (x, y) in self.reachable(u, self.move_roll):
                acts.append({
                    "action_id": f"move_{x}_{y}",
                    "type": MOVE,
                    "label": f"Move to ({x},{y})",
                    "note": f"{abs(x - u.x) + abs(y - u.y)} tiles away",
                })

        if u.alive and not u.attacked and u.ammo > 0:
            for target, chance, dist in self.targets(u):
                acts.append({
                    "action_id": f"attack_seat{target.seat}",
                    "type": ATTACK,
                    "label": f"Attack {target.name} at ({target.x},{target.y})",
                    "hit_chance": chance,
                    "expected_damage": 1 + u.damage_bonus,
                    "note": f"{dist} tiles away, target has {target.hp} HP",
                })

        if u.alive and not u.attacked and u.ammo < u.max_ammo:
            acts.append({"action_id": "reload", "type": RELOAD,
                         "label": f"Reload to {u.max_ammo} ammo",
                         "note": "you may not attack this turn if you reload"})

        if u.alive and not u.played_card:
            for card_id in u.hand:
                c = CARDS[card_id]
                acts.append({"action_id": f"play_{card_id}", "type": PLAY_CARD,
                             "label": f"Play {c['name']}", "note": c["text"]})

        acts.append({"action_id": "chat", "type": CHAT, "label": "Say something to the table",
                     "note": "free text in args.message"})
        acts.append({"action_id": "end_turn", "type": END_TURN, "label": "End your turn"})
        return acts

    # -- redaction -----------------------------------------------------------

    def redacted_state(self, seat):
        """What THIS seat may see. A whitelist, and the only place that decides.

        Built by naming each field rather than copying the game and deleting the secrets. The
        difference matters: with a blacklist, every field added later is exposed by default and
        nothing fails visibly when it happens. Note `cards_in_hand` on opponents -- a count, never
        the contents. Getting this wrong does not crash anything, it silently makes every hidden
        information card worthless.
        """
        me = self.unit(seat)
        return {
            "turn": self.turn,
            "self": {
                "seat": me.seat, "name": me.name, "tile": [me.x, me.y],
                "hp": me.hp, "max_hp": me.max_hp,
                "ammo": me.ammo, "max_ammo": me.max_ammo,
                "range": me.range, "kills": me.kills,
                "move_roll": self.move_roll,
                "this_turn": {"moved": me.moved, "attacked": me.attacked,
                              "played_card": me.played_card},
                "hand": [{"card_id": c, "name": CARDS[c]["name"], "text": CARDS[c]["text"]}
                         for c in me.hand],
            },
            "players": [
                {"seat": u.seat, "name": u.name, "tile": [u.x, u.y], "hp": u.hp,
                 "kills": u.kills, "status": "alive" if u.alive else "dead",
                 # THE REDACTION. u.hand is right there and is never sent.
                 "cards_in_hand": len(u.hand),
                 **({"is_self": True} if u.seat == seat else {})}
                for u in self.units
            ],
            "board": {
                "ascii": self.render_ascii(seat),
                "legend": {".": "open", "#": "wall", "@": "you", "0-3": "a unit's seat number"},
                "width": self.w, "height": self.h,
            },
            "events_since_last_turn": list(self.pending.get(seat, [])),
            "chat": self.chat_log[-6:],
        }

    def render_ascii(self, seat):
        rows = []
        header = "   " + " ".join(str(x % 10) for x in range(self.w))
        rows.append(header)
        for y in range(self.h):
            cells = []
            for x in range(self.w):
                who = next((u for u in self.living() if u.x == x and u.y == y), None)
                if who is not None:
                    cells.append("@" if who.seat == seat else str(who.seat))
                else:
                    cells.append(self.grid[y][x])
            rows.append(f"{y:2d} " + " ".join(cells))
        return "\n".join(rows)

    # -- applying an action --------------------------------------------------

    def apply(self, action_id, args):
        """Apply a VALIDATED action. Returns (human_result, turn_over).

        Callers must have checked the id against available_actions first. This does not re-check:
        one authority, checked once, at the boundary -- see protocol_client.take_turn.
        """
        u = self.unit(self.current)
        self.action_count += 1

        if action_id.startswith("move_"):
            _, sx, sy = action_id.split("_")
            u.x, u.y = int(sx), int(sy)
            if u.extra_move:
                u.extra_move = False
            else:
                u.moved = True
            return f"You moved to ({u.x},{u.y}).", False

        if action_id.startswith("attack_seat"):
            target = self.unit(int(action_id.replace("attack_seat", "")))
            u.attacked = True
            u.ammo -= 1
            chance = dict((t.seat, c) for t, c, _ in self.targets(u)).get(target.seat, 0.0)
            if self.rng.random() <= chance:
                dmg = 1 + u.damage_bonus
                u.damage_bonus = 0
                target.hp -= dmg
                msg = f"You hit {target.name} for {dmg}."
                self.broadcast(f"{u.name} hit {target.name} for {dmg}.", exclude=u.seat)
                if target.hp <= 0:
                    target.alive = False
                    u.kills += 1
                    msg += f" {target.name} is down."
                    self.broadcast(f"{target.name} was knocked out by {u.name}.", exclude=u.seat)
                return msg, False
            self.broadcast(f"{u.name} shot at {target.name} and missed.", exclude=u.seat)
            return f"You missed {target.name}.", False

        if action_id == "reload":
            u.ammo = u.max_ammo
            u.attacked = True   # reloading spends the shot
            return f"You reloaded to {u.ammo}.", False

        if action_id.startswith("play_"):
            card_id = action_id[len("play_"):]
            if card_id in u.hand:
                u.hand.remove(card_id)
            u.played_card = True
            if card_id == "c_patch":
                u.hp = min(u.max_hp, u.hp + 2)
                res = f"You healed to {u.hp} HP."
            elif card_id == "c_scope":
                u.range_bonus += 2
                res = f"Your range is now {u.range}."
            elif card_id == "c_dash":
                u.extra_move = True
                self.move_roll = self.roll()
                res = f"You may move again, up to {self.move_roll}."
            else:
                u.damage_bonus += 1
                res = "Your next hit deals +1."
            self.broadcast(f"{u.name} played {CARDS[card_id]['name']}.", exclude=u.seat)
            return res, False

        if action_id == "chat":
            text = (args or {}).get("message") or "..."
            self.chat_log.append({"seat": u.seat, "name": u.name, "text": text})
            self.broadcast(f"{u.name} said: {text}", exclude=u.seat)
            return f"You said: {text}", False

        if action_id == "end_turn":
            return "You ended your turn.", True

        raise ValueError(f"apply() got an id it does not implement: {action_id!r}")

    def broadcast(self, text, exclude=None):
        for u in self.units:
            if u.seat != exclude:
                self.pending.setdefault(u.seat, []).append(text)

    def roll(self):
        return self.rng.randint(2, 6)

    # -- turn structure ------------------------------------------------------

    def begin_turn(self):
        u = self.unit(self.current)
        u.start_turn()
        self.move_roll = self.roll()
        self.pending[u.seat] = []

    def end_turn(self):
        alive = self.living()
        if len(alive) <= 1:
            self.winner = alive[0].seat if alive else None
            return True
        for _ in range(len(self.units)):
            self.current = (self.current + 1) % len(self.units)
            if self.unit(self.current).alive:
                break
        if self.current == 0:
            self.turn += 1
        return False
