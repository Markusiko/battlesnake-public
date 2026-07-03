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
| TBD | candidate-next | 1v1 | TBD | TBD | TBD | TBD | TBD | TBD |

## Death Patterns

- TBD

## Promotion Recommendation

TBD
