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
| candidate-next | evaluated | health-critical safe-food gate | promote recommended; awaiting explicit champion update approval |

## Main Checklist

- [x] Confirm champion directory exists.
- [x] Confirm candidate directory exists.
- [x] Confirm champion is not edited directly.
- [x] Start champion on port 8000.
- [x] Start candidate on port 8001.
- [x] Run syntax checks on candidate.
- [x] Run smoke checks on candidate.
- [x] Run 1v1 arena games.
- [ ] Run multi-snake arena games if time allows.
- [x] Record arena results in `reports/eval.md`.
- [x] Review research notes in `reports/research.md`.
- [x] Promote or reject candidate.
- [ ] If promoted, tag/update current champion.
- [ ] Freeze final champion before deployment.

## Agent Board

| Agent | Scope | Status | Output |
|---|---|---|---|
| Coordinator / Champion Keeper | integration and promotion | active | `plan.md` |
| Candidate Coder | `battlesnake-candidate/logic.py` | complete | critical food gate patch |
| Arena Evaluator | local games | complete | `reports/eval.md` |
| Strategy Researcher | tactical ideas | complete | `reports/research.md` |

## Candidate Experiments

| Time | Candidate | Change | Games vs Champion | Result | Decision | Notes |
|---|---|---|---:|---|---|---|
| 2026-07-03 | candidate-next | health-critical safe-food gate | 100 fallback | candidate 27, champion 3, draws 70 | promote recommended | official CLI unavailable; fallback arena shows fewer candidate starvation losses |

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
