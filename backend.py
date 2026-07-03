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
SANDWORM_TIMEOUT_SECONDS = 0.26
VALID_MOVES = {"up", "down", "left", "right"}
MOVE_DELTAS = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}


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


def safe_move_or_fallback(game_state: Dict[str, Any], move: str) -> str:
    """Return move if immediately safe; otherwise choose any safe move.

    Args:
        game_state: Battlesnake request payload.
        move: Candidate move.

    Returns:
        Safe move when one exists.
    """
    safe_moves = [candidate for candidate in VALID_MOVES if is_immediately_safe_move(game_state, candidate)]
    if not safe_moves:
        return move if move in VALID_MOVES else "up"
    if move in safe_moves and not is_edge_chasing_move(game_state, move, safe_moves):
        return move
    return max(safe_moves, key=lambda candidate: move_safety_score(game_state, candidate))


def is_edge_chasing_move(game_state: Dict[str, Any], move: str, safe_moves: list[str]) -> bool:
    """Return True when move hugs a wall while safer interior moves exist.

    Args:
        game_state: Battlesnake request payload.
        move: Candidate move.
        safe_moves: Moves that pass immediate safety.

    Returns:
        True if candidate should be replaced by a more central safe move.
    """
    candidate_score = move_safety_score(game_state, move)
    best_score = max(move_safety_score(game_state, candidate) for candidate in safe_moves)
    return best_score >= candidate_score + 2


def move_safety_score(game_state: Dict[str, Any], move: str) -> int:
    """Score next cell by wall distance and local exits.

    Args:
        game_state: Battlesnake request payload.
        move: Candidate move.

    Returns:
        Higher score means less likely to run straight into an edge.
    """
    board = game_state["board"]
    you = game_state["you"]
    head = (you["head"]["x"], you["head"]["y"])
    dx, dy = MOVE_DELTAS[move]
    nxt = (head[0] + dx, head[1] + dy)
    wall_distance = min(nxt[0], board["width"] - 1 - nxt[0], nxt[1], board["height"] - 1 - nxt[1])
    exits = 0
    for ddx, ddy in MOVE_DELTAS.values():
        neighbor = (nxt[0] + ddx, nxt[1] + ddy)
        if 0 <= neighbor[0] < board["width"] and 0 <= neighbor[1] < board["height"]:
            exits += 1
    return wall_distance * 4 + exits


def is_immediately_safe_move(game_state: Dict[str, Any], move: str) -> bool:
    """Check walls and occupied cells for the next move.

    Args:
        game_state: Battlesnake request payload.
        move: Candidate move.

    Returns:
        True if the move does not immediately hit a wall or body.
    """
    if move not in MOVE_DELTAS:
        return False

    board = game_state["board"]
    you = game_state["you"]
    head = (you["head"]["x"], you["head"]["y"])
    dx, dy = MOVE_DELTAS[move]
    nxt = (head[0] + dx, head[1] + dy)
    if not (0 <= nxt[0] < board["width"] and 0 <= nxt[1] < board["height"]):
        return False

    blocked = set()
    for snake in board["snakes"]:
        body = snake["body"]
        for segment in body:
            blocked.add((segment["x"], segment["y"]))
        if len(body) >= 2:
            tail = (body[-1]["x"], body[-1]["y"])
            before_tail = (body[-2]["x"], body[-2]["y"])
            if tail != before_tail:
                blocked.discard(tail)
    return nxt not in blocked


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
    safe_move = safe_move_or_fallback(game_state, move)
    if safe_move != move:
        log.warning("UNSAFE MOVE REPLACED turn=%s %s -> %s", game_state["turn"], move, safe_move)
        move = safe_move
        engine = f"{engine}-safe"
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
