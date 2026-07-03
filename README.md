# Battlesnake Sandworm Bot

A [Battlesnake](https://play.battlesnake.com) written in Python and Flask. This
version uses the vendored [Sandworm](sandworm/README.md) C move engine. Python
keeps the HTTP API and fallback logic.

## What It Does

Each turn, `backend.py`:

- Receives a Battlesnake move request through `backend.py`.
- Sends the raw request JSON to the compiled Sandworm binary.
- Falls back to Python logic if Sandworm fails or times out.


## Files

- `backend.py` — Battlesnake HTTP server with `/`, `/start`, `/move`, and `/end`.
- `logic.py` — Python fallback logic and snake appearance.
- `sandworm/` — vendored Sandworm C move engine.
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
