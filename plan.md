# Battlesnake Champion-Candidate Plan

## Goal

Improve the Battlesnake bot safely by keeping a stable champion and testing every candidate against it.

## Current Champion

| Version | Source | Status | Notes |
|---|---|---|---|
| champion-v1 | current stable `logic.py` | active | strong heuristic baseline with model and old heuristic fallback |

## Current Candidate

| Candidate | Status | Change | Decision |
|---|---|---|---|
| candidate-next | not started | TBD | TBD |

## Main Checklist

- [ ] Confirm champion directory exists.
- [ ] Confirm candidate directory exists.
- [ ] Confirm champion is not edited directly.
- [ ] Start champion on port 8000.
- [ ] Start candidate on port 8001.
- [ ] Run syntax checks on candidate.
- [ ] Run smoke checks on candidate.
- [ ] Run 1v1 arena games.
- [ ] Run multi-snake arena games if time allows.
- [ ] Record arena results in `reports/eval.md`.
- [ ] Review research notes in `reports/research.md`.
- [ ] Promote or reject candidate.
- [ ] If promoted, tag/update current champion.
- [ ] Freeze final champion before deployment.

## Agent Board

| Agent | Scope | Status | Output |
|---|---|---|---|
| Coordinator / Champion Keeper | integration and promotion | active | `plan.md` |
| Candidate Coder | `battlesnake-candidate/logic.py` | pending | candidate patch |
| Arena Evaluator | local games | pending | `reports/eval.md` |
| Strategy Researcher | tactical ideas | pending | `reports/research.md` |

## Candidate Experiments

| Time | Candidate | Change | Games vs Champion | Result | Decision | Notes |
|---|---|---|---:|---|---|---|
| TBD | candidate-next | TBD | TBD | TBD | TBD | TBD |

## Promotion Rules

Promote candidate only if:

- it passes `python3 -m compileall logic.py backend.py`;
- it does not crash in smoke checks;
- it keeps per-move latency under 500 ms;
- it beats champion in arena games, or looks clearly safer with similar win rate;
- it does not introduce obvious new failure modes.

If results are ambiguous, keep champion.

## Known Risks

- Arena results can be noisy with few games.
- Self-play against the same bot can hide weaknesses.
- Hand-tuned weights can overfit to a small number of games.
- Treating enemy tails as occupied is safe but can miss opportunities.
- Relaxing safety rules can improve aggression but cause sudden early deaths.

## Next Candidate Ideas

Priority order:

1. Add hard health-critical food behavior.
2. Tune head-to-head penalties after arena deaths.
3. Improve trapped-pocket detection.
4. Adjust food greediness when healthy.
5. Improve territory pressure only when longer and safe.
