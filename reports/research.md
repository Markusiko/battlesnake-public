# Strategy Research Notes

Use this file to propose the next small candidate patch.

## Arena Interpretation

Champion and candidate servers now start independently after copying the shared `backend.py` and `requirements.txt` into both bot directories. The official Battlesnake CLI is still unavailable on PATH, so evaluation used a fallback local 1v1 arena.

The candidate patch targets one observed scoring gap: at health 8, the baseline can choose open space over adjacent safe food on the board edge. In fallback arena games this translated into fewer starvation losses for the candidate.

## High-Impact Candidate Ideas

| Idea | Why it helps | Cost | Risk | Priority |
|---|---|---|---|---|
| Health-critical food gate | Prevents starving when health is near zero | Low | Can chase dangerous food | High |
| Tune head-to-head danger | Reduces equal/longer head collisions | Low | Can become too passive | High |
| Stronger pocket penalty | Avoids traps with too little space | Medium | Can overvalue open space | High |
| Safer food when healthy | Reduces greedy deaths | Low | Can miss growth chances | Medium |
| Aggression only when longer | Can kill shorter snakes safely | Medium | Risky if over-weighted | Medium |

## Best Next Candidate Patch

Promote the current health-critical food candidate after explicit approval to update the champion copy. The next smallest candidate patch should be head-to-head tuning, but only after running the promoted bot against fresh arena scenarios and observing equal-or-longer head collision deaths.

## Ideas To Avoid For Now

- Full RL/self-play.
- Large minimax/search rewrite.
- Heavy dependencies.
- Random exploration in tournament mode.
- Deployment changes during algorithm tuning.
