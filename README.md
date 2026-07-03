# Battlesnake ML Inference Bot

A [Battlesnake](https://play.battlesnake.com) written in Python and Flask. This
version uses a pretrained model checkpoint to choose moves.

## What It Does

Each turn, `logic.py`:

- Gets legal moves for the current board.
- Applies a safety filter to avoid preventable head-to-head losses and
  one-turn traps.
- Calculates per-move features.
- Scores each safe move with a pure-Python linear model.
- Returns the highest-scoring move.

The model is used for ranking, not for basic survival guarantees. If every move
is risky, the bot keeps the least-bad legal candidates instead of failing.

## Files

- `backend.py` — Battlesnake HTTP server with `/`, `/start`, `/move`, and `/end`.
- `logic.py` — embedded checkpoint, feature extraction, move scoring, safety
  filters, and fallback logic.
- `test_logic.py` — lightweight unit tests for basic move safety.
- `requirements.txt` — runtime dependencies.
- `render.yaml` — Render deployment config.

## Run Locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python backend.py
```

Test your battlesnake with the Battlesnake CLI:

```bash
battlesnake play -W 11 -H 11 \
  -n ml -u http://localhost:8000 \
  -g solo \
  -v -c -d 300
```

## Run Tests

```bash
python3 -m unittest test_logic.py
```

## Deploy to Render

1. Push this repo to GitHub.
2. In the [Render dashboard](https://dashboard.render.com): **New -> Blueprint**,
   connect the repo. Render reads `render.yaml` and provisions a free web
   service running `gunicorn backend:app`.
   - Or **New -> Web Service** manually with build command
     `pip install -r requirements.txt` and start command
     `gunicorn backend:app --bind 0.0.0.0:$PORT`.
3. Wait for the deploy to go live. Note the public URL, e.g.
   `https://battlesnake-xxxx.onrender.com`.
4. Visit that URL in a browser — you should see the appearance JSON.

## Register on Battlesnake

1. Create an account at [play.battlesnake.com](https://play.battlesnake.com).
2. **Create Battlesnake** -> paste your Render URL as the server URL.
3. Now you can use it in a game!
