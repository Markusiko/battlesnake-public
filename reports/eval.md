# Arena Evaluation Notes

Use this file to record champion-vs-candidate Battlesnake results.

## Current Matchup

| Role | Port | Path | Notes |
|---|---:|---|---|
| Champion | 8000 | `battlesnake-champion/` | current stable bot |
| Candidate | 8001 | `battlesnake-candidate/` | challenger bot |

## Commands

Start champion:

```bash
cd battlesnake-champion
PORT=8000 python3 backend.py
```

Start candidate:

```bash
cd battlesnake-candidate
PORT=8001 python3 backend.py
```

Run 1v1:

```bash
battlesnake play -W 11 -H 11 \
  -n champion -u http://localhost:8000 \
  -n candidate -u http://localhost:8001 \
  -v -c -d 0 -s
```

Run 4-snake:

```bash
battlesnake play -W 11 -H 11 \
  -n champion1 -u http://localhost:8000 \
  -n champion2 -u http://localhost:8000 \
  -n candidate1 -u http://localhost:8001 \
  -n candidate2 -u http://localhost:8001 \
  -v -c -d 0 -s
```

## Results

| Time | Candidate | Setup | Games | Champion Wins | Candidate Wins | Draws | Avg Turns | Decision |
|---|---|---|---:|---:|---:|---:|---:|---|
| 2026-07-03 | candidate-next | offline smoke only | 0 | 0 | 0 | 0 | N/A | Superseded by server/fallback arena |
| 2026-07-03 | candidate-next | fallback 1v1 direct-logic arena | 100 | 3 | 27 | 70 | 216.85 | Promote recommended; official CLI absent |

## Death Patterns

- Shared `backend.py` and `requirements.txt` were copied into both bot directories with user approval.
- Both servers started: champion on port 8000, candidate on port 8001.
- `/` responded for both bots.
- `/move` responded for both bots on the critical-food smoke: champion chose `left`, candidate chose `right`.
- HTTP smoke latency was about 1.4 ms for both bots.
- Official `battlesnake` CLI was not found on PATH, so the official CLI arena was not run.
- Fallback local arena used deterministic 11x11 1v1 direct-logic games with core wall/body/head-to-head/food/starvation rules.
- Fallback deaths: champion starvation 28, candidate starvation 4.
- Fallback max direct `choose_move` latency: 3.139 ms; average direct latency was about 0.79 ms per bot.

## Promotion Recommendation

Promote recommended based on server smoke checks and fallback arena evidence. Do not overwrite `battlesnake-champion/logic.py` unless explicitly approved.
