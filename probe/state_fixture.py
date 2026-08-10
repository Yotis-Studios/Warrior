"""A realistic mid-match Raifu Wars turn, used as the fixture every probe runs against.

Hand-built rather than captured from the game, because the point of the probe is to answer
"what shape of payload can a model actually play from" BEFORE the client can emit one. The
numbers are plausible rather than authoritative -- what matters is the SHAPE: four seats, a
board with cover and capture points, a hand of cards with star costs, and a legal-action set
that a human at this seat could actually reach.

Kept in one place so every variant is compared on identical input. If a variant wins because
it was handed a different board, the comparison is worthless.
"""

# 12x9. Legend is sent to the model with the map; a bare grid is unreadable without it.
MAP_ASCII = """\
   0 1 2 3 4 5 6 7 8 9 10 11
0  . . . ^ ^ . . . . . .  .
1  . B . ^ ^ . . # # . .  .
2  . . . . . . . A . . .  .
3  ~ ~ ~ . . C . . . . .  .
4  ~ ~ ~ . . . . . . . D  .
5  . . . . # # . . . . .  .
6  . . . . # # . . . C .  .
7  . . M . . . . . . . B  .
8  . . . . . . . . . . .  .\
"""

MAP_LEGEND = {
    ".": "open ground, passable",
    "#": "wall, blocks movement and line of fire",
    "^": "forest, passable, gives cover to whoever stands in it",
    "~": "water, impassable without a pontoon",
    "B": "capture point (base) -- hold at end of turn to earn stars",
    "C": "capture point (contested)",
    "A": "your raifu",
    "D": "an enemy raifu",
    "M": "a minefield",
}

# The self view. Everything here is something the seat's own player can see on their screen.
SELF = {
    "seat": 1,
    "name": "Puska",
    "team": "A",
    "tile": [7, 2],
    "health": 3,
    "max_health": 4,
    "ammo": 1,
    "max_ammo": 2,
    "stars": 14,
    "tier": 2,
    "range": 4,
    "kills": 2,
    "deaths": 1,
    "tier_progress": {"next_tier": 3, "condition": "stars", "have": 14, "need": 20},
    "this_turn": {
        "moved": True,
        "attacked": False,
        "rushed": False,
        "played_card": False,
    },
    "hand": [
        {"card_id": "c_barricade", "name": "Barricade",
         "cost_stars": 4, "text": "Place a wall on an adjacent empty tile."},
        {"card_id": "c_hastyshot", "name": "Hasty Shot",
         "cost_stars": 6, "text": "You may attack a second time this turn."},
        {"card_id": "c_luckycharm", "name": "Lucky Charm",
         "cost_stars": 8, "text": "Your next movement roll is a 6."},
    ],
}

# Public info about every seat. NOTE what is absent: no `hand` on anyone but self. That absence
# IS the protocol -- see the redaction section of PROTOCOL.md.
PLAYERS = [
    {"seat": 0, "name": "Arisaka", "team": "B", "tile": [10, 4], "health": 4,
     "tier": 2, "stars": 11, "kills": 1, "cards_in_hand": 2, "status": "alive"},
    {"seat": 1, "name": "Puska", "team": "A", "tile": [7, 2], "health": 3,
     "tier": 2, "stars": 14, "kills": 2, "cards_in_hand": 3, "status": "alive", "is_self": True},
    {"seat": 2, "name": "Mosin", "team": "B", "tile": [2, 7], "health": 2,
     "tier": 1, "stars": 6, "kills": 0, "cards_in_hand": 1, "status": "alive"},
    {"seat": 3, "name": "Garand", "team": "A", "tile": [1, 1], "health": 4,
     "tier": 3, "stars": 19, "kills": 3, "cards_in_hand": 4, "status": "alive"},
]

POINTS = [
    {"point_id": "pt_b1", "tile": [1, 1], "held_by_seat": 3, "stars_per_turn": 2},
    {"point_id": "pt_c3", "tile": [5, 3], "held_by_seat": None, "stars_per_turn": 3},
    {"point_id": "pt_c6", "tile": [9, 6], "held_by_seat": 0, "stars_per_turn": 3},
    {"point_id": "pt_b7", "tile": [10, 7], "held_by_seat": 0, "stars_per_turn": 2},
]

EVENTS = [
    "Garand captured pt_b1 and is 1 star from tier 4.",
    "Arisaka shot you for 1 damage from (10,4).",
    "Mosin played Barricade at (4,5).",
]

# THE LEGAL SET. Every entry is something a human at this seat could do right now, and nothing
# else is offered. Note the absences and why they are correct:
#   - no `rush`: rush needs moved && !attacked && !rushed -- moved is true, so rush IS legal.
#     (it is present below)
#   - no `move`: already moved this turn
#   - no c_hastyshot / c_luckycharm play: 6 and 8 stars, and 14 available -- both affordable,
#     so both ARE offered. c_barricade needs an adjacent empty tile, which exists.
#   - `reload` is offered as reload (ammo 1 of 2, not full) rather than fortify.
AVAILABLE_ACTIONS = [
    {"action_id": "atk_seat0", "type": "attack", "label": "Attack Arisaka at (10,4)",
     "target_seat": 0, "hit_chance": 0.45, "expected_damage": 1,
     "note": "range 4, distance 3.6, target in the open"},
    {"action_id": "rush_1", "type": "rush",
     "label": "Rush -- spend your shot for a second movement",
     "note": "forfeits attacking this turn"},
    {"action_id": "reload_1", "type": "reload", "label": "Reload to 2 ammo",
     "note": "cannot attack this turn if you reload"},
    {"action_id": "play_c_barricade", "type": "play_card", "card_id": "c_barricade",
     "label": "Play Barricade (4 stars)", "requires_target": "adjacent_empty_tile"},
    {"action_id": "play_c_hastyshot", "type": "play_card", "card_id": "c_hastyshot",
     "label": "Play Hasty Shot (6 stars)", "requires_target": None},
    {"action_id": "play_c_luckycharm", "type": "play_card", "card_id": "c_luckycharm",
     "label": "Play Lucky Charm (8 stars)", "requires_target": None},
    {"action_id": "discard_c_luckycharm", "type": "discard_card", "card_id": "c_luckycharm",
     "label": "Discard Lucky Charm"},
    {"action_id": "chat_1", "type": "chat", "label": "Say something in table chat",
     "requires_target": "message_text"},
    {"action_id": "end_turn", "type": "end_turn", "label": "End your turn"},
]

VALID_IDS = {a["action_id"] for a in AVAILABLE_ACTIONS}


def render_state_text():
    """The whole turn as the text a model sees. One function so every variant shares it."""
    lines = []
    lines.append("=== RAIFU WARS -- YOUR TURN (turn 23) ===")
    lines.append("")
    lines.append("You are Puska, seat 1, team A.")
    lines.append("Win by reaching tier 4. You tier up by earning stars or KOs.")
    lines.append("Capture points earn stars at the end of each of your turns.")
    lines.append("")
    lines.append("BOARD:")
    lines.append(MAP_ASCII)
    lines.append("")
    lines.append("LEGEND: " + "; ".join(f"{k} = {v}" for k, v in MAP_LEGEND.items()))
    lines.append("")
    lines.append(f"YOU: tile {SELF['tile']}, health {SELF['health']}/{SELF['max_health']}, "
                 f"ammo {SELF['ammo']}/{SELF['max_ammo']}, {SELF['stars']} stars, tier {SELF['tier']}, "
                 f"range {SELF['range']}")
    tp = SELF["tier_progress"]
    lines.append(f"TIER PROGRESS: {tp['have']}/{tp['need']} {tp['condition']} to reach tier {tp['next_tier']}")
    tt = SELF["this_turn"]
    lines.append("THIS TURN SO FAR: " + ", ".join(
        f"{k}={'yes' if v else 'no'}" for k, v in tt.items()))
    lines.append("")
    lines.append("YOUR HAND:")
    for c in SELF["hand"]:
        lines.append(f"  {c['card_id']}: {c['name']} ({c['cost_stars']} stars) -- {c['text']}")
    lines.append("")
    lines.append("PLAYERS:")
    for p in PLAYERS:
        me = " (you)" if p.get("is_self") else ""
        lines.append(f"  seat {p['seat']} {p['name']}{me}, team {p['team']}, tile {p['tile']}, "
                     f"health {p['health']}, tier {p['tier']}, {p['stars']} stars, "
                     f"{p['cards_in_hand']} cards in hand")
    lines.append("")
    lines.append("CAPTURE POINTS:")
    for pt in POINTS:
        holder = "unheld" if pt["held_by_seat"] is None else f"held by seat {pt['held_by_seat']}"
        lines.append(f"  {pt['point_id']} at {pt['tile']}, {holder}, "
                     f"{pt['stars_per_turn']} stars/turn")
    lines.append("")
    lines.append("SINCE YOUR LAST TURN:")
    for e in EVENTS:
        lines.append(f"  - {e}")
    return "\n".join(lines)


def render_actions_text():
    """The legal set as text. Used by variants that put actions in the prompt body."""
    lines = ["LEGAL ACTIONS (you may ONLY choose one of these action_id values):"]
    for a in AVAILABLE_ACTIONS:
        extra = []
        if "hit_chance" in a:
            extra.append(f"hit chance {int(a['hit_chance'] * 100)}%")
        if a.get("requires_target"):
            extra.append(f"requires {a['requires_target']}")
        if a.get("note"):
            extra.append(a["note"])
        suffix = (" [" + "; ".join(extra) + "]") if extra else ""
        lines.append(f"  {a['action_id']}: {a['label']}{suffix}")
    return "\n".join(lines)
