"""Battlesnake HTTP server for the Sandworm-backed bot.

Implements the four endpoints the Battlesnake game engine calls:
  GET  /        -> snake appearance + metadata
  POST /start   -> a game has started
  POST /move    -> return our next move for this turn
  POST /end     -> a game has ended
"""

import json
import logging
import os
from pathlib import Path
import subprocess
from typing import Any, Dict

from flask import Flask, request

from logic import choose_move, get_info

app = Flask("battlesnake")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("battlesnake")

ROOT_DIR = Path(__file__).resolve().parent
SANDWORM_MOVE_PATH = Path(os.environ.get("SANDWORM_MOVE_PATH", ROOT_DIR / "sandworm" / "bin" / "move"))
SANDWORM_TIMEOUT_SECONDS = 0.34
VALID_MOVES = {"up", "down", "left", "right"}


def choose_move_sandworm(game_state: Dict[str, Any]) -> str:
    """Run the Sandworm C move engine.

    Args:
        game_state: Battlesnake request payload.

    Returns:
        Move returned by Sandworm.
    """
    payload = json.dumps(game_state, separators=(",", ":")).encode("utf-8")
    result = subprocess.run(
        [str(SANDWORM_MOVE_PATH)],
        input=payload,
        capture_output=True,
        timeout=SANDWORM_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[-1000:]
        raise RuntimeError(f"Sandworm failed with code {result.returncode}: {stderr}")

    stdout = result.stdout.decode("utf-8", errors="replace")
    json_start = stdout.rfind("{")
    if json_start == -1:
        raise ValueError("Sandworm response had no JSON object")
    response = json.loads(stdout[json_start:])
    move = response.get("move")
    if move not in VALID_MOVES:
        raise ValueError(f"Sandworm returned invalid move: {move!r}")
    return move


@app.get("/")
def on_info():
    return get_info()


@app.post("/start")
def on_start():
    game_state = request.get_json()
    log.info("GAME START %s", game_state["game"]["id"])
    return "ok"


@app.post("/move")
def on_move():
    game_state = request.get_json()
    try:
        move = choose_move_sandworm(game_state)
        engine = "sandworm"
    except Exception as exc:  # noqa: BLE001 - fallback must keep gameplay alive
        log.warning("SANDWORM FALLBACK turn=%s error=%s", game_state["turn"], exc)
        move = choose_move(game_state)
        engine = "python"
    log.info("MOVE turn=%s -> %s", game_state["turn"], move)
    return {"move": move, "shout": engine}


@app.post("/end")
def on_end():
    game_state = request.get_json()
    log.info("GAME END %s", game_state["game"]["id"])
    return "ok"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    log.info("Starting Battlesnake server on port %s", port)
    app.run(host="0.0.0.0", port=port)
