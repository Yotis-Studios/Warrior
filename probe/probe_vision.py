"""Can the model actually SEE a Raifu Wars screenshot, or is it guessing?

    python probe/probe_vision.py <screenshot.png> [--url http://127.0.0.1:8080]

WHY THIS NEEDS A CONTROL. Asking a model "what do you see" and getting a plausible answer proves
nothing. A language model handed a question about a screenshot it cannot see will answer anyway,
confidently, from the question's own wording and from whatever it knows about games -- and a
Raifu Wars HUD is guessable enough that several of the answers below could come out right by
inference alone. Every wrong conclusion about vision starts here.

So every question is asked TWICE: once with the image attached, once with nothing attached. The
blind run is not a formality, it is the measurement. What matters is the DIFFERENCE:

    blind wrong, sighted right   the image is being read
    both right                   the question was guessable -- it proves nothing, discard it
    both wrong                   the image is not reaching the model, or it cannot use it
    blind right, sighted wrong   the image is actively hurting

Questions are chosen to be unguessable and checkable: exact digits off the HUD, a count of
figures on the terrain, the language of the UI. Nothing that could be inferred from "this is a
turn-based strategy game".

THE ANSWERS ARE NOT AUTO-GRADED. Expected values are printed beside each reply for a human to
compare, because a grader doing fuzzy string matching on free text is one more thing that can be
wrong in the direction that flatters the result.
"""

import argparse
import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request

# Ground truth read off ui-captures/1280x720-game/uicapture_7_in_game_rot0.png by eye. If you point
# this at a different screenshot these are wrong -- pass --no-expect and read the replies yourself.
QUESTIONS = [
    ("How many character figures are standing on the battlefield terrain? Do not count the "
     "portrait icons in the bar along the top of the screen. Answer with a single number.",
     "1"),
    ("What text is displayed in the bottom-LEFT corner of the screen, next to the heart icon? "
     "Answer with exactly what is written.",
     "06/06"),
    ("What text is displayed in the bottom-RIGHT corner of the screen, next to the ammunition "
     "icon? Answer with exactly what is written.",
     "05/05"),
    ("What language is the text on the buttons written in? Answer with the language name only.",
     "Japanese"),
    ("Is there any water visible anywhere on the map? Answer yes or no.",
     "no"),
    ("What is the dominant terrain covering the battlefield? Answer in three words or fewer.",
     "forest / pine trees"),
]


def ask(url, model, question, b64=None, mime="image/png", timeout=180.0):
    if b64:
        content = [
            {"type": "text", "text": question},
            {"type": "image_url",
             "image_url": {"url": "data:%s;base64,%s" % (mime, b64)}},
        ]
    else:
        content = question

    body = {
        "messages": [{"role": "user", "content": content}],
        # Deterministic, so the blind and sighted runs differ because of the IMAGE rather than
        # because of sampling. Without this a difference proves nothing at all.
        "temperature": 0.0,
        "max_tokens": 200,
        "stream": False,
    }
    if model:
        body["model"] = model

    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            d = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return "<HTTP %s: %s>" % (exc.code, exc.read().decode("utf-8", "replace")[:200])
    except Exception as exc:                                    # noqa: BLE001
        return "<error: %s>" % exc

    try:
        return (d["choices"][0]["message"]["content"] or "").strip().replace("\n", " ")[:220]
    except (KeyError, IndexError):
        return "<no content: %s>" % json.dumps(d)[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-expect", action="store_true",
                    help="the image is not the one EXPECTED describes; hide the expectations")
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        print("no such file: %s" % args.image)
        return 2

    with open(args.image, "rb") as fh:
        raw = fh.read()
    b64 = base64.b64encode(raw).decode("ascii")
    mime = mimetypes.guess_type(args.image)[0] or "image/png"

    print("=== vision probe ===")
    print("  image  : %s (%d KB, %s)" % (os.path.basename(args.image), len(raw) // 1024, mime))
    print("  server : %s" % args.url)
    print("  base64 : %d chars" % len(b64))
    print()

    agree = 0
    for i, (q, expected) in enumerate(QUESTIONS):
        print("Q%d. %s" % (i + 1, q))
        if not args.no_expect:
            print("    EXPECTED : %s" % expected)
        blind = ask(args.url, args.model, q, None)
        sighted = ask(args.url, args.model, q, b64, mime)
        print("    blind    : %s" % blind)
        print("    SIGHTED  : %s" % sighted)
        if blind.strip().lower() == sighted.strip().lower():
            agree += 1
            print("    ^^ identical to the blind answer -- this question discriminates nothing")
        print()

    print("=== %d of %d questions gave the SAME answer with and without the image ==="
          % (agree, len(QUESTIONS)))
    if agree == len(QUESTIONS):
        print("  The image changed nothing. Either it is not reaching the model, or it is being")
        print("  ignored. Check that llama-server was started with an mmproj / vision projector.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
