"""Battlesnake HTTP server for the model inference bot.

Implements the four endpoints the Battlesnake game engine calls:
  GET  /        -> snake appearance + metadata
  POST /start   -> a game has started
  POST /move    -> return our next move for this turn
  POST /end     -> a game has ended
"""

import logging
import os

from flask import Flask, request

from logic import choose_move, get_info

app = Flask("battlesnake")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("battlesnake")


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
    move = choose_move(game_state)
    log.info("MOVE turn=%s -> %s", game_state["turn"], move)
    return {"move": move}


@app.post("/end")
def on_end():
    game_state = request.get_json()
    log.info("GAME END %s", game_state["game"]["id"])
    return "ok"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    log.info("Starting Battlesnake server on port %s", port)
    app.run(host="0.0.0.0", port=port)
