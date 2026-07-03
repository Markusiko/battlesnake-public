# Battlesnake Champion-Candidate Agent Instructions

## Goal

Build the strongest reliable Battlesnake bot through safe challenger iterations.

The current stable bot is the champion. Every new idea must be implemented in a candidate copy, tested against the champion in local Battlesnake arena games, and promoted only if it is clearly better or at least equally strong with fewer obvious failure modes.

## Hard Rules

- Never overwrite the champion directly.
- Treat `battlesnake-champion/` as read-only during candidate work.
- Make algorithm changes only in `battlesnake-candidate/`.
- Promote a candidate only after arena evaluation.
- If results are noisy or unclear, keep the current champion.
- Do not implement RL during the first hackathon cycle.
- Do not add heavy dependencies.
- Do not refactor `backend.py` unless the HTTP server is broken.
- `choose_move(game_state)` must never crash.
- The bot must return a move within 500 ms.

## Repository Layout

Recommended structure:

```text
sberhack/
├── battlesnake-champion/
│   ├── logic.py
│   ├── backend.py
│   └── ...
├── battlesnake-candidate/
│   ├── logic.py
│   ├── backend.py
│   └── ...
├── AGENTS.md
├── plan.md
├── agent_prompts.md
└── reports/
    ├── eval.md
    └── research.md
```

If docs live inside one repo instead, keep them in the same directory as `logic.py` and update paths in `agent_prompts.md`.

## Roles

### Coordinator / Champion Keeper

Owns integration and promotion decisions.

Responsibilities:

- Keep champion safe.
- Create or refresh the candidate copy from champion.
- Dispatch focused agents.
- Review candidate diff.
- Review arena results.
- Promote or reject candidate.
- Update `plan.md`.

### Candidate Coder

Works only in `battlesnake-candidate/`.

Responsibilities:

- Make one small heuristic improvement at a time.
- Edit only candidate files.
- Prefer scoring changes over broad rewrites.
- Preserve fallback behavior.
- Run syntax checks.

### Arena Evaluator

Runs local games between champion and candidate.

Responsibilities:

- Start champion on port 8000.
- Start candidate on port 8001.
- Run `battlesnake play`.
- Record wins, turns, deaths, and visible failure patterns.
- Write results to `reports/eval.md`.

### Strategy Researcher

Analyzes behavior and suggests next safe improvements.

Responsibilities:

- Read current logic and eval notes.
- Propose small tactical improvements.
- Avoid speculative or large rewrites.
- Write notes to `reports/research.md`.

## Candidate Development Strategy

Focus on small, measurable improvements:

- health-critical food behavior;
- head-to-head avoidance against equal/longer snakes;
- avoiding pockets smaller than our length;
- better own-tail reachability;
- less greedy food chasing;
- better escape-route scoring;
- safer territory control.

Avoid:

- full RL/self-play;
- large architecture changes;
- random tournament behavior;
- expensive search that risks the 500 ms limit;
- changing deployment files during algorithm tuning.

## Arena Commands

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

Run 1v1 arena:

```bash
battlesnake play -W 11 -H 11 \
  -n champion -u http://localhost:8000 \
  -n candidate -u http://localhost:8001 \
  -v -c -d 0 -s
```

Run 4-snake arena:

```bash
battlesnake play -W 11 -H 11 \
  -n champion1 -u http://localhost:8000 \
  -n champion2 -u http://localhost:8000 \
  -n candidate1 -u http://localhost:8001 \
  -n candidate2 -u http://localhost:8001 \
  -v -c -d 0 -s
```

## Promotion Criteria

Promote candidate only if:

- syntax checks pass;
- server starts;
- `/` returns appearance JSON;
- `/move` returns legal-looking moves;
- per-move latency is safely under 500 ms;
- arena results are better than champion or clearly not worse;
- no new obvious failure mode appears.

When in doubt, reject the candidate and keep champion.
