"""Which action-encoding can a small local model actually play from without inventing things?

THE QUESTION THIS ANSWERS. A first hand-probe of this server handed it an `attack` tool with a
free-string `target_id` and no list of targets. It called the tool correctly and passed
`target_id: "enemy_at_5_4"` -- an identifier it made up, because the schema left a gap and a
language model fills gaps. Any encoding we pick has to be judged on whether it makes that
impossible, not on whether it reads nicely.

THREE ENCODINGS, same fixture, same seat, same board:

  named    one tool per action type (attack / rush / reload / play_card / ...), typed arguments.
           Closest to what instruction-tuned models have seen, so it should reason best -- but
           every free-form argument is somewhere to confabulate.

  enum     ONE tool, `take_action(action_id)`, where action_id carries a JSON-Schema `enum` of
           exactly the legal ids. If llama.cpp compiles that enum into its sampling grammar then
           an illegal action is not unlikely, it is UNREPRESENTABLE -- which is the strongest
           possible form of "never offered an action a human could not take".

  free     ONE tool, `take_action(action_id)`, action_id a free string, ids listed in the prompt
           only. The CONTROL: it isolates how much of enum's win comes from grammar enforcement
           versus from simply having the ids written down somewhere.

Reported per variant: how often the chosen id was legal, how often the call was well-formed at
all, latency, and token cost -- plus every distinct id returned, so an invalid one can be read
rather than merely counted.
"""

import argparse
import collections
import json
import time
import urllib.error
import urllib.request

from state_fixture import (
    AVAILABLE_ACTIONS,
    VALID_IDS,
    render_actions_text,
    render_state_text,
)

SYSTEM = (
    "You are a competitive Raifu Wars player taking your turn. "
    "Think briefly about the board, then call exactly one tool to take one action. "
    "You may only take actions that are offered to you."
)


def build_named_tools():
    """One tool per action type, with typed arguments -- the conventional encoding."""
    return [
        {"type": "function", "function": {
            "name": "attack",
            "description": "Fire at an enemy raifu. Costs 1 ammo and your shot for this turn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_seat": {"type": "integer",
                                    "description": "seat number of the enemy to shoot"},
                },
                "required": ["target_seat"],
            },
        }},
        {"type": "function", "function": {
            "name": "rush",
            "description": "Give up your shot this turn in exchange for a second movement.",
            "parameters": {"type": "object", "properties": {}},
        }},
        {"type": "function", "function": {
            "name": "reload",
            "description": "Refill ammo. You cannot attack this turn if you reload.",
            "parameters": {"type": "object", "properties": {}},
        }},
        {"type": "function", "function": {
            "name": "play_card",
            "description": "Play a card from your hand, paying its star cost.",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_id": {"type": "string", "description": "card_id from your hand"},
                },
                "required": ["card_id"],
            },
        }},
        {"type": "function", "function": {
            "name": "discard_card",
            "description": "Discard a card from your hand without playing it.",
            "parameters": {
                "type": "object",
                "properties": {"card_id": {"type": "string"}},
                "required": ["card_id"],
            },
        }},
        {"type": "function", "function": {
            "name": "chat",
            "description": "Say something to the other players at the table.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        }},
        {"type": "function", "function": {
            "name": "end_turn",
            "description": "End your turn.",
            "parameters": {"type": "object", "properties": {}},
        }},
    ]


def build_single_tool(with_enum):
    action_id = {"type": "string", "description": "the action_id of the action you choose"}
    if with_enum:
        action_id["enum"] = sorted(VALID_IDS)
    return [{"type": "function", "function": {
        "name": "take_action",
        "description": ("Take exactly one of the legal actions offered to you this turn. "
                        "action_id must be copied exactly from the legal action list."),
        "parameters": {
            "type": "object",
            "properties": {
                "action_id": action_id,
                "message": {"type": "string",
                            "description": "only for chat_1: what to say"},
            },
            "required": ["action_id"],
        },
    }}]


# How a returned tool call is mapped back to an action_id, per variant. Kept beside the tool
# definitions because the two have to agree, and a mismatch here would look like a model failure.
def resolve_named(call):
    name = call["function"]["name"]
    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return None, "unparseable arguments"
    if name == "attack":
        seat = args.get("target_seat")
        return (f"atk_seat{seat}", None) if seat is not None else (None, "attack with no seat")
    if name == "rush":
        return "rush_1", None
    if name == "reload":
        return "reload_1", None
    if name == "play_card":
        return f"play_{args.get('card_id')}", None
    if name == "discard_card":
        return f"discard_{args.get('card_id')}", None
    if name == "chat":
        return "chat_1", None
    if name == "end_turn":
        return "end_turn", None
    return None, f"unknown tool {name}"


def resolve_single(call):
    if call["function"]["name"] != "take_action":
        return None, f"unknown tool {call['function']['name']}"
    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return None, "unparseable arguments"
    aid = args.get("action_id")
    return (aid, None) if aid else (None, "no action_id")


VARIANTS = {
    "named": {"tools": build_named_tools(), "resolve": resolve_named, "list_in_prompt": False},
    "enum":  {"tools": build_single_tool(True),  "resolve": resolve_single, "list_in_prompt": True},
    "free":  {"tools": build_single_tool(False), "resolve": resolve_single, "list_in_prompt": True},
}


def call_model(url, model, tools, user_content, temperature, timeout, max_tokens):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "tools": tools,
        "tool_choice": "required",
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    return payload, time.time() - t0


def run_variant(name, cfg, args):
    user = render_state_text()
    if cfg["list_in_prompt"]:
        user += "\n\n" + render_actions_text()
    user += "\n\nTake one action now."

    legal = malformed = truncated = 0
    ids = collections.Counter()
    problems = []
    latencies, prompt_toks, completion_toks = [], [], []

    for i in range(args.trials):
        try:
            payload, dt = call_model(args.url, args.model, cfg["tools"], user,
                                     args.temperature, args.timeout, args.max_tokens)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            malformed += 1
            problems.append(f"transport: {exc}")
            continue

        latencies.append(dt)
        usage = payload.get("usage", {})
        prompt_toks.append(usage.get("prompt_tokens", 0))
        completion_toks.append(usage.get("completion_tokens", 0))

        choice = payload["choices"][0]
        msg = choice.get("message", {})
        calls = msg.get("tool_calls") or []
        if not calls:
            malformed += 1
            # TRUNCATION AND REFUSAL LOOK IDENTICAL in the parsed result -- both are "no tool
            # call" -- so record finish_reason. A `length` here means the token cap cut the model
            # off mid-reasoning, which is a setting we chose and not a property of the encoding.
            fr = choice.get("finish_reason")
            if fr == "length":
                truncated += 1
            reasoning = (msg.get("reasoning_content") or "")
            problems.append(f"no tool_call (finish={fr}, reasoning {len(reasoning)} chars): "
                            + (msg.get("content") or reasoning or "")[-100:].replace("\n", " "))
            continue

        aid, err = cfg["resolve"](calls[0])
        if err or aid is None:
            malformed += 1
            problems.append(err or "no action id")
            continue

        ids[aid] += 1
        if aid in VALID_IDS:
            legal += 1
        else:
            problems.append(f"ILLEGAL action_id: {aid!r}")

    n = args.trials
    return {
        "variant": name,
        "trials": n,
        "legal": legal,
        "malformed": malformed,
        "truncated": truncated,
        "legal_pct": 100.0 * legal / n if n else 0.0,
        "chosen": ids.most_common(),
        "problems": problems[:12],
        "avg_latency_s": sum(latencies) / len(latencies) if latencies else None,
        "avg_prompt_tokens": sum(prompt_toks) / len(prompt_toks) if prompt_toks else None,
        "avg_completion_tokens": (sum(completion_toks) / len(completion_toks)
                                  if completion_toks else None),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="Wichtel-Q4_K_M.gguf")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--max-tokens", type=int, default=2500,
                    help="generation cap. Must be generous: this model emits long "
                         "reasoning_content before the call, and a cap that lands mid-reasoning "
                         "produces a missing tool call that looks like an encoding failure.")
    ap.add_argument("--variants", default="named,enum,free")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    print(f"fixture prompt is {len(render_state_text())} chars "
          f"({len(render_state_text().splitlines())} lines), "
          f"{len(AVAILABLE_ACTIONS)} legal actions\n")

    results = []
    for name in args.variants.split(","):
        name = name.strip()
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant {name}")
        print(f"--- {name} ---", flush=True)
        res = run_variant(name, VARIANTS[name], args)
        results.append(res)
        print(f"  legal      {res['legal']}/{res['trials']}  ({res['legal_pct']:.0f}%)")
        print(f"  malformed  {res['malformed']}"
              + (f"  (of which {res['truncated']} hit the token cap)" if res["truncated"] else ""))
        if res["avg_latency_s"]:
            print(f"  latency    {res['avg_latency_s']:.1f}s avg")
            print(f"  tokens     {res['avg_prompt_tokens']:.0f} prompt / "
                  f"{res['avg_completion_tokens']:.0f} completion")
        print(f"  chose      {res['chosen']}")
        for p in res["problems"]:
            print(f"  ! {p}")
        print(flush=True)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
