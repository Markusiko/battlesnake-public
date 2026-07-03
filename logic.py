"""Risk-aware Battlesnake move selection.

Public interface expected by the server:
    get_info() -> dict
    choose_move(game_state) -> one of: up/down/left/right

The agent is designed for standard 11x11 multiplayer games and a 500 ms
request timeout.  It uses:
  * exact simultaneous one-turn simulation of the standard rules;
  * full opponent-response enumeration at the root;
  * selective depth-2 search (and depth 3 in small endgames);
  * CVaR-style risk aggregation instead of pure minimax;
  * dynamic Voronoi territory with body release times;
  * a deadline and a cheap legal fallback.

No third-party packages are required.
"""

from __future__ import annotations

import itertools
import math
import os
import time
from collections import Counter, deque
from typing import Dict, FrozenSet, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}
MOVE_NAMES: Tuple[str, ...] = tuple(DIRECTIONS)

# Search settings. The environment variable is convenient for deployment tuning.
DEFAULT_COMPUTE_MS = 315
NETWORK_RESERVE_MS = 145
MIN_COMPUTE_MS = 35
MAX_COMPUTE_MS = int(os.getenv("BATTLESNAKE_SEARCH_MS", str(DEFAULT_COMPUTE_MS)))
DEADLINE_CHECK_MASK = 15  # check the clock once per 16 visited nodes

WIN_SCORE = 1_000_000.0
DEATH_SCORE = -1_000_000.0
HUNGRY_HEALTH = 45
CRITICAL_HEALTH = 22
CVaR_FRACTION = 0.20


class SnakeState(NamedTuple):
    sid: str
    health: int
    body: Tuple[Point, ...]

    @property
    def head(self) -> Point:
        return self.body[0]

    @property
    def length(self) -> int:
        return len(self.body)


class SearchState(NamedTuple):
    width: int
    height: int
    food: FrozenSet[Point]
    hazards: Tuple[Point, ...]  # duplicates intentionally represent stacked hazards
    snakes: Tuple[SnakeState, ...]
    you_id: str
    hazard_damage: int
    turn: int


class SearchTimeout(RuntimeError):
    """Raised internally when the move deadline has been reached."""


class SearchContext:
    __slots__ = (
        "deadline",
        "nodes",
        "eval_cache",
        "fast_eval_cache",
        "value_cache",
        "move_cache",
        "solid_cache",
    )

    def __init__(self, deadline: float) -> None:
        self.deadline = deadline
        self.nodes = 0
        self.eval_cache: Dict[SearchState, float] = {}
        self.fast_eval_cache: Dict[SearchState, float] = {}
        self.value_cache: Dict[Tuple[SearchState, int], float] = {}
        self.move_cache: Dict[Tuple[SearchState, str, int], Tuple[str, ...]] = {}
        self.solid_cache: Dict[SearchState, FrozenSet[Point]] = {}

    def tick(self) -> None:
        self.nodes += 1
        if (self.nodes & DEADLINE_CHECK_MASK) == 0 and time.perf_counter() >= self.deadline:
            raise SearchTimeout


# ---------------------------------------------------------------------------
# Public API


def get_info() -> Dict[str, str]:
    """Appearance and metadata returned from GET /."""
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "1.0.0",
    }


def choose_move(game_state: Dict) -> str:
    """Choose a move while always returning before the Battlesnake timeout."""
    try:
        state = _parse_state(game_state)
    except Exception:  # malformed requests must not crash the server
        return "up"

    you = _snake_by_id(state, state.you_id)
    if you is None or not you.body:
        return "up"

    fallback = _fallback_move(state, you)

    timeout_ms = int(game_state.get("game", {}).get("timeout", 500) or 500)
    available_ms = max(MIN_COMPUTE_MS, timeout_ms - NETWORK_RESERVE_MS)
    compute_ms = min(MAX_COMPUTE_MS, available_ms)
    deadline = time.perf_counter() + compute_ms / 1000.0
    ctx = SearchContext(deadline)

    best_completed = fallback
    snake_count = len(state.snakes)
    max_depth = 3 if snake_count <= 2 else 2

    try:
        # Iterative deepening guarantees that a completed shallower answer is
        # available if the next level runs out of time.
        for depth in range(1, max_depth + 1):
            depth_started = time.perf_counter()
            move, _rank = _search_root(state, depth, ctx)
            best_completed = move
            now = time.perf_counter()
            depth_seconds = now - depth_started
            remaining = deadline - now
            # Do not start a deeper iteration when the completed level already
            # indicates that it is unlikely to finish. This avoids burning the
            # entire budget on an unusable partial result in wide openings.
            if remaining <= max(0.035, depth_seconds * 1.20):
                break
    except SearchTimeout:
        pass
    except Exception:
        # Search errors should never turn into an invalid /move response.
        pass

    return best_completed


# Kept for compatibility with projects that imported the old helper directly.
def choose_move_heuristic(game_state: Dict) -> str:
    try:
        state = _parse_state(game_state)
        you = _snake_by_id(state, state.you_id)
        return _fallback_move(state, you) if you is not None else "up"
    except Exception:
        return "up"


# ---------------------------------------------------------------------------
# Parsing and exact turn simulation


def _parse_state(game_state: Mapping) -> SearchState:
    board = game_state["board"]
    game = game_state.get("game", {})
    ruleset = game.get("ruleset", {})
    settings = ruleset.get("settings", {}) or {}

    snakes: List[SnakeState] = []
    for raw in board.get("snakes", []):
        body = tuple((int(p["x"]), int(p["y"])) for p in raw.get("body", []))
        if body:
            snakes.append(SnakeState(str(raw["id"]), int(raw.get("health", 100)), body))

    food = frozenset((int(p["x"]), int(p["y"])) for p in board.get("food", []))
    hazards = tuple((int(p["x"]), int(p["y"])) for p in board.get("hazards", []))

    return SearchState(
        width=int(board["width"]),
        height=int(board["height"]),
        food=food,
        hazards=hazards,
        snakes=tuple(snakes),
        you_id=str(game_state["you"]["id"]),
        hazard_damage=int(settings.get("hazardDamagePerTurn", 0) or 0),
        turn=int(game_state.get("turn", 0)),
    )


def _simulate_turn(state: SearchState, moves: Mapping[str, str]) -> SearchState:
    """Apply one simultaneous standard-rules turn.

    The order mirrors the official rules implementation:
      movement -> health -1 -> hazard damage -> food/growth -> elimination.
    Random future food spawns and future Royale shrinking are intentionally not
    predicted because they are unknowable from a /move request.
    """
    moved: List[SnakeState] = []

    # 1. Movement: prepend a new head and always pop the old tail.
    for snake in state.snakes:
        move = moves.get(snake.sid, "up")
        dx, dy = DIRECTIONS.get(move, DIRECTIONS["up"])
        hx, hy = snake.head
        new_head = (hx + dx, hy + dy)
        new_body = (new_head,) + snake.body[:-1]
        moved.append(SnakeState(snake.sid, snake.health, new_body))

    # 2-3. Starvation and hazard damage. Food protects from hazard damage on
    # that square in the standard rules implementation.
    hazard_counts = Counter(state.hazards)
    damaged: List[SnakeState] = []
    for snake in moved:
        health = snake.health - 1
        if snake.head not in state.food:
            health -= state.hazard_damage * hazard_counts.get(snake.head, 0)
        damaged.append(SnakeState(snake.sid, max(0, health), snake.body))

    # 4. Feeding: all snakes on food are fed before collisions are resolved.
    eaten: Set[Point] = set()
    fed: List[SnakeState] = []
    for snake in damaged:
        if snake.head in state.food:
            eaten.add(snake.head)
            # Growth is represented by duplicating the current tail. This is
            # why a freshly fed tail does not vacate on the next turn.
            grown_body = snake.body + (snake.body[-1],)
            fed.append(SnakeState(snake.sid, 100, grown_body))
        else:
            fed.append(snake)

    # 5a. Remove snakes that are already dead from health or bounds. Their
    # bodies do not participate in collision checks in the official engine.
    stage_alive = [
        snake
        for snake in fed
        if snake.health > 0 and _in_bounds(snake.head, state.width, state.height)
    ]

    # 5b. Body collisions are based on the simultaneously updated bodies.
    dead_ids: Set[str] = set()
    for snake in stage_alive:
        head = snake.head
        if head in snake.body[1:]:
            dead_ids.add(snake.sid)
            continue
        for other in stage_alive:
            if other.sid != snake.sid and head in other.body[1:]:
                dead_ids.add(snake.sid)
                break

    # 5c. Head-to-head: a unique longest snake survives; equal longest heads
    # all die. Food growth has already changed lengths at this stage.
    heads: Dict[Point, List[SnakeState]] = {}
    for snake in stage_alive:
        heads.setdefault(snake.head, []).append(snake)
    for group in heads.values():
        if len(group) < 2:
            continue
        longest = max(s.length for s in group)
        winners = [s for s in group if s.length == longest]
        if len(winners) == 1:
            winner_id = winners[0].sid
            dead_ids.update(s.sid for s in group if s.sid != winner_id)
        else:
            dead_ids.update(s.sid for s in group)

    survivors = tuple(s for s in stage_alive if s.sid not in dead_ids)
    return SearchState(
        state.width,
        state.height,
        frozenset(p for p in state.food if p not in eaten),
        state.hazards,
        survivors,
        state.you_id,
        state.hazard_damage,
        state.turn + 1,
    )


# ---------------------------------------------------------------------------
# Search


def _search_root(
    state: SearchState,
    depth: int,
    ctx: SearchContext,
) -> Tuple[str, Tuple[float, float, float]]:
    you = _snake_by_id(state, state.you_id)
    if you is None:
        return "up", (-1.0, DEATH_SCORE, DEATH_SCORE)

    moves = list(_candidate_moves(state, you, width=4, ctx=ctx))
    moves.sort(key=lambda m: _quick_move_score(state, you, m), reverse=True)

    best_move = moves[0] if moves else "up"
    best_rank = (-math.inf, -math.inf, -math.inf)

    for move in moves:
        ctx.tick()
        values = _action_outcomes(
            state,
            you,
            move,
            depth,
            ctx,
            full_enemy_width=True,
            dynamic_leaf=(depth == 1),
        )
        rank = _root_rank(values)
        if depth > 1:
            # Keep the first component exact even though deeper continuations
            # use a selective subset of opponent combinations.
            death_fraction = _immediate_death_fraction(state, you, move, ctx)
            rank = (-death_fraction, rank[1], rank[2])
        if rank > best_rank:
            best_rank = rank
            best_move = move

    return best_move, best_rank


def _search_value(state: SearchState, depth: int, ctx: SearchContext) -> float:
    ctx.tick()
    terminal = _terminal_score(state)
    if terminal is not None:
        return terminal
    if depth <= 0:
        return _evaluate_state(state, ctx)

    cache_key = (state, depth)
    cached = ctx.value_cache.get(cache_key)
    if cached is not None:
        return cached

    you = _snake_by_id(state, state.you_id)
    if you is None:
        return DEATH_SCORE

    moves = list(_candidate_moves(state, you, width=3, ctx=ctx))
    moves.sort(key=lambda m: _quick_move_score(state, you, m), reverse=True)

    best = DEATH_SCORE
    for move in moves:
        outcomes = _action_outcomes(
            state,
            you,
            move,
            depth,
            ctx,
            full_enemy_width=False,
            dynamic_leaf=False,
        )
        value = _robust_value(outcomes)
        if value > best:
            best = value

    if len(ctx.value_cache) < 12_000:
        ctx.value_cache[cache_key] = best
    return best


def _action_outcomes(
    state: SearchState,
    you: SnakeState,
    our_move: str,
    depth: int,
    ctx: SearchContext,
    full_enemy_width: bool,
    dynamic_leaf: bool,
) -> List[float]:
    enemies = [s for s in state.snakes if s.sid != you.sid]
    if not enemies:
        child = _simulate_turn(state, {you.sid: our_move})
        return [_search_value(child, depth - 1, ctx)]

    per_enemy: List[Tuple[str, ...]] = []
    for enemy in enemies:
        width = 4 if full_enemy_width else 2
        enemy_moves = _candidate_moves(state, enemy, width=width, ctx=ctx)
        per_enemy.append(enemy_moves)

    combinations = list(itertools.product(*per_enemy))
    # Depth 1 checks every plausible opponent response. Deeper search keeps the
    # most realistic and tactically dangerous joint responses so that a full
    # two-turn search fits comfortably inside the request deadline.
    if not (depth <= 1 and full_enemy_width and dynamic_leaf):
        limit = 8 if full_enemy_width else 4
        combinations = _select_enemy_combinations(
            state, you, our_move, enemies, combinations, limit
        )

    values: List[float] = []
    for combo in combinations:
        ctx.tick()
        joint = {you.sid: our_move}
        joint.update((enemy.sid, move) for enemy, move in zip(enemies, combo))
        child = _simulate_turn(state, joint)
        if _snake_by_id(child, state.you_id) is None:
            values.append(DEATH_SCORE)
        elif depth <= 1:
            values.append(_evaluate_state(child, ctx) if dynamic_leaf else _evaluate_state_fast(child, ctx))
        else:
            values.append(_search_value(child, depth - 1, ctx))

    return values or [DEATH_SCORE]




def _immediate_death_fraction(
    state: SearchState,
    you: SnakeState,
    our_move: str,
    ctx: SearchContext,
) -> float:
    enemies = [snake for snake in state.snakes if snake.sid != you.sid]
    if not enemies:
        child = _simulate_turn(state, {you.sid: our_move})
        return 1.0 if _snake_by_id(child, you.sid) is None else 0.0

    move_lists = [
        _candidate_moves(state, enemy, width=4, ctx=ctx) for enemy in enemies
    ]
    deaths = 0
    total = 0
    for combo in itertools.product(*move_lists):
        ctx.tick()
        joint = {you.sid: our_move}
        joint.update((enemy.sid, move) for enemy, move in zip(enemies, combo))
        child = _simulate_turn(state, joint)
        total += 1
        if _snake_by_id(child, you.sid) is None:
            deaths += 1
    return deaths / max(1, total)

def _select_enemy_combinations(
    state: SearchState,
    you: SnakeState,
    our_move: str,
    enemies: Sequence[SnakeState],
    combinations: Sequence[Tuple[str, ...]],
    limit: int,
) -> List[Tuple[str, ...]]:
    if len(combinations) <= limit:
        return list(combinations)

    dx, dy = DIRECTIONS[our_move]
    our_target = (you.head[0] + dx, you.head[1] + dy)
    solid = _nominal_solid_no_cache(state)
    ranked: List[Tuple[float, Tuple[str, ...]]] = []

    for combo in combinations:
        score = 0.0
        for enemy, move in zip(enemies, combo):
            score += _quick_move_score(state, enemy, move, solid)
            edx, edy = DIRECTIONS[move]
            target = (enemy.head[0] + edx, enemy.head[1] + edy)
            # Always surface plausible head-to-head attacks near the top.
            if target == our_target:
                score += 100_000.0 if enemy.length >= you.length else 8_000.0
            if target in state.food:
                score += 800.0
        ranked.append((score, combo))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [combo for _score, combo in ranked[:limit]]

def _root_rank(values: Sequence[float]) -> Tuple[float, float, float]:
    deaths = sum(1 for v in values if v <= DEATH_SCORE / 2)
    death_fraction = deaths / max(1, len(values))
    return (-death_fraction, _cvar(values), sum(values) / len(values))


def _robust_value(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    cvar = _cvar(values)
    deaths = sum(1 for v in values if v <= DEATH_SCORE / 2) / len(values)
    # The explicit death term makes the search strongly prefer actions that
    # survive a larger share of plausible simultaneous responses.
    return 0.72 * cvar + 0.28 * mean - 120_000.0 * deaths


def _cvar(values: Sequence[float]) -> float:
    ordered = sorted(values)
    count = max(1, math.ceil(len(ordered) * CVaR_FRACTION))
    return sum(ordered[:count]) / count


def _terminal_score(state: SearchState) -> Optional[float]:
    you = _snake_by_id(state, state.you_id)
    if you is None:
        return DEATH_SCORE
    if len(state.snakes) == 1:
        return WIN_SCORE
    return None


# ---------------------------------------------------------------------------
# Move generation and fallback


def _candidate_moves(
    state: SearchState,
    snake: SnakeState,
    width: int,
    ctx: SearchContext,
) -> Tuple[str, ...]:
    cache_key = (state, snake.sid, width)
    cached = ctx.move_cache.get(cache_key)
    if cached is not None:
        return cached

    solid = _nominal_solid(state, ctx)
    hazard_counts = Counter(state.hazards)
    plausible: List[str] = []
    in_bounds: List[str] = []

    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (snake.head[0] + dx, snake.head[1] + dy)
        if not _in_bounds(nxt, state.width, state.height):
            continue
        in_bounds.append(move)
        if nxt in solid:
            continue
        future_health = snake.health - 1
        if nxt not in state.food:
            future_health -= state.hazard_damage * hazard_counts.get(nxt, 0)
        if future_health <= 0:
            continue
        plausible.append(move)

    # A trapped snake still has to return a direction. Including all four lets
    # exact simulation distinguish the least bad outcome.
    candidates = plausible or in_bounds or list(MOVE_NAMES)
    candidates.sort(key=lambda m: _quick_move_score(state, snake, m, solid), reverse=True)
    result = tuple(candidates[:width])
    ctx.move_cache[cache_key] = result
    return result


def _nominal_solid(state: SearchState, ctx: Optional[SearchContext] = None) -> FrozenSet[Point]:
    if ctx is not None:
        cached = ctx.solid_cache.get(state)
        if cached is not None:
            return cached

    counts: Counter[Point] = Counter()
    for snake in state.snakes:
        counts.update(snake.body)

    # A non-duplicated tail normally vacates during the next movement stage.
    # We remove only one occurrence so coiled/growing tails remain solid.
    for snake in state.snakes:
        if snake.body and (len(snake.body) == 1 or snake.body[-1] != snake.body[-2]):
            tail = snake.body[-1]
            counts[tail] -= 1
            if counts[tail] <= 0:
                del counts[tail]

    result = frozenset(counts)
    if ctx is not None:
        ctx.solid_cache[state] = result
    return result


def _quick_move_score(
    state: SearchState,
    snake: SnakeState,
    move: str,
    solid: Optional[FrozenSet[Point]] = None,
) -> float:
    dx, dy = DIRECTIONS[move]
    nxt = (snake.head[0] + dx, snake.head[1] + dy)
    if not _in_bounds(nxt, state.width, state.height):
        return -100_000.0

    score = 0.0
    if solid is None:
        solid = _nominal_solid_no_cache(state)
    if nxt in solid:
        score -= 50_000.0

    degree = 0
    for ddx, ddy in DIRECTIONS.values():
        nb = (nxt[0] + ddx, nxt[1] + ddy)
        if _in_bounds(nb, state.width, state.height) and nb not in solid:
            degree += 1
    score += degree * 120.0

    # Prefer cells away from walls when other signals are equal.
    wall_distance = min(nxt[0], state.width - 1 - nxt[0], nxt[1], state.height - 1 - nxt[1])
    score += wall_distance * 10.0

    # Avoid possible head-to-heads against equal/longer snakes; seek shorter
    # heads only mildly, because forcing kills is useful but not worth a trap.
    for other in state.snakes:
        if other.sid == snake.sid:
            continue
        if _manhattan(nxt, other.head) == 1:
            if other.length >= snake.length:
                score -= 2_500.0
            else:
                score += 120.0

    if state.food:
        old_dist = min(_manhattan(snake.head, food) for food in state.food)
        new_dist = min(_manhattan(nxt, food) for food in state.food)
        urgency = max(0.0, (HUNGRY_HEALTH - snake.health) / HUNGRY_HEALTH)
        score += (old_dist - new_dist) * (30.0 + 220.0 * urgency)
        if nxt in state.food:
            score += 300.0 + 1_000.0 * urgency

    if nxt in state.hazards and nxt not in state.food:
        score -= state.hazard_damage * 35.0
    return score


def _nominal_solid_no_cache(state: SearchState) -> FrozenSet[Point]:
    counts: Counter[Point] = Counter()
    for snake in state.snakes:
        counts.update(snake.body)
    for snake in state.snakes:
        if snake.body and (len(snake.body) == 1 or snake.body[-1] != snake.body[-2]):
            counts[snake.body[-1]] -= 1
            if counts[snake.body[-1]] <= 0:
                del counts[snake.body[-1]]
    return frozenset(counts)


def _fallback_move(state: SearchState, you: SnakeState) -> str:
    """Cheap one-ply fallback computed before the timed search begins."""
    ctx = SearchContext(time.perf_counter() + 10.0)
    moves = _candidate_moves(state, you, width=4, ctx=ctx)
    solid = set(_nominal_solid(state, ctx))

    best_move = moves[0] if moves else "up"
    best_score = -math.inf
    enemy_replies: Dict[str, Set[Point]] = {}
    for enemy in state.snakes:
        if enemy.sid == you.sid:
            continue
        cells: Set[Point] = set()
        for move in _candidate_moves(state, enemy, width=4, ctx=ctx):
            dx, dy = DIRECTIONS[move]
            cells.add((enemy.head[0] + dx, enemy.head[1] + dy))
        enemy_replies[enemy.sid] = cells

    for move in moves:
        dx, dy = DIRECTIONS[move]
        nxt = (you.head[0] + dx, you.head[1] + dy)
        if nxt in solid:
            space = 0
        else:
            space = _static_flood(nxt, solid, state.width, state.height)
        score = space * 100.0 + _quick_move_score(state, you, move)

        for enemy in state.snakes:
            if enemy.sid == you.sid:
                continue
            if nxt in enemy_replies.get(enemy.sid, set()):
                score += 600.0 if you.length > enemy.length else -20_000.0

        if space < you.length:
            score -= (you.length - space + 1) * 8_000.0
        if score > best_score:
            best_score = score
            best_move = move

    return best_move


# ---------------------------------------------------------------------------
# Position evaluation


def _evaluate_state(state: SearchState, ctx: SearchContext) -> float:
    cached = ctx.eval_cache.get(state)
    if cached is not None:
        return cached

    terminal = _terminal_score(state)
    if terminal is not None:
        return terminal

    you = _snake_by_id(state, state.you_id)
    if you is None:
        return DEATH_SCORE

    release = _body_release_times(state)
    max_length = max((s.length for s in state.snakes), default=1)
    horizon = min(34, state.width + state.height + max_length // 2)

    distances: Dict[str, Dict[Point, int]] = {}
    for snake in state.snakes:
        distances[snake.sid] = _dynamic_distances(
            snake.head,
            release,
            state.width,
            state.height,
            horizon,
        )

    territory: Dict[str, int] = {snake.sid: 0 for snake in state.snakes}
    contested = 0
    for x in range(state.width):
        for y in range(state.height):
            cell = (x, y)
            arrivals: List[Tuple[int, int, str]] = []
            for snake in state.snakes:
                dist = distances[snake.sid].get(cell)
                if dist is not None:
                    arrivals.append((dist, snake.length, snake.sid))
            if not arrivals:
                continue
            best_time = min(a[0] for a in arrivals)
            tied = [a for a in arrivals if a[0] == best_time]
            best_length = max(a[1] for a in tied)
            winners = [a for a in tied if a[1] == best_length]
            if len(winners) == 1:
                territory[winners[0][2]] += 1
            else:
                contested += 1

    my_territory = territory.get(you.sid, 0)
    enemy_territories = [v for sid, v in territory.items() if sid != you.sid]
    strongest_enemy_territory = max(enemy_territories, default=0)

    solid = set(_nominal_solid(state, ctx))
    # The current head is a valid flood-fill origin even though it is occupied.
    solid.discard(you.head)
    static_space = _static_flood(you.head, solid, state.width, state.height)

    candidate_moves = _candidate_moves(state, you, width=4, ctx=ctx)
    mobility = len(candidate_moves)
    safe_mobility = _safe_mobility(state, you, candidate_moves, ctx)

    my_dist = distances[you.sid]
    tail_distance = my_dist.get(you.body[-1])
    reaches_tail = 1.0 if tail_distance is not None else 0.0

    enemies = [s for s in state.snakes if s.sid != you.sid]
    max_enemy_length = max((s.length for s in enemies), default=you.length)
    length_advantage = you.length - max_enemy_length

    score = 0.0
    score += (my_territory - strongest_enemy_territory) * 145.0
    score += my_territory * 18.0
    score -= contested * 2.0
    score += min(static_space, state.width * state.height) * 52.0
    score += safe_mobility * 1_100.0 + mobility * 180.0
    score += reaches_tail * 650.0
    score += length_advantage * 260.0

    # Strong anti-trap term: available chamber size should exceed body length.
    required_space = you.length + 2
    if static_space < required_space:
        score -= (required_space - static_space) * 12_000.0
    elif static_space < int(1.5 * you.length):
        score -= (int(1.5 * you.length) - static_space) * 1_400.0

    # Food is a constraint when health is low, not a permanent objective.
    food_distances = [my_dist[f] for f in state.food if f in my_dist]
    nearest_food = min(food_distances, default=None)
    if nearest_food is not None:
        if you.health <= CRITICAL_HEALTH:
            score += max(-25_000.0, (you.health - nearest_food - 2) * 2_200.0)
        elif you.health < HUNGRY_HEALTH:
            score += (HUNGRY_HEALTH - nearest_food) * 260.0
        elif you.health > 75:
            # Growing without need can reduce mobility, so the reward is tiny.
            score += max(0.0, 8.0 - nearest_food) * 15.0
    elif you.health < HUNGRY_HEALTH:
        score -= (HUNGRY_HEALTH - you.health) * 1_100.0

    # Health reserve and current hazard exposure.
    score += min(you.health, 60) * 10.0
    if you.head in state.hazards and you.head not in state.food:
        score -= state.hazard_damage * 180.0

    # Mild centre preference only breaks otherwise similar positions.
    centre = ((state.width - 1) / 2.0, (state.height - 1) / 2.0)
    score -= (abs(you.head[0] - centre[0]) + abs(you.head[1] - centre[1])) * 8.0

    if len(ctx.eval_cache) < 15_000:
        ctx.eval_cache[state] = score
    return score



def _evaluate_state_fast(state: SearchState, ctx: SearchContext) -> float:
    """Cheaper leaf evaluation used only below the first searched turn.

    It keeps the same strategic signals but uses static shortest paths. The
    root one-ply search always uses the full dynamic release-time evaluation.
    """
    cached = ctx.fast_eval_cache.get(state)
    if cached is not None:
        return cached

    terminal = _terminal_score(state)
    if terminal is not None:
        return terminal
    you = _snake_by_id(state, state.you_id)
    if you is None:
        return DEATH_SCORE

    solid = set(_nominal_solid(state, ctx))
    distances: Dict[str, Dict[Point, int]] = {}
    for snake in state.snakes:
        blocked = solid - {snake.head}
        distances[snake.sid] = _static_distances(
            snake.head, blocked, state.width, state.height
        )

    territory: Dict[str, int] = {snake.sid: 0 for snake in state.snakes}
    for x in range(state.width):
        for y in range(state.height):
            cell = (x, y)
            arrivals: List[Tuple[int, int, str]] = []
            for snake in state.snakes:
                distance = distances[snake.sid].get(cell)
                if distance is not None:
                    arrivals.append((distance, snake.length, snake.sid))
            if not arrivals:
                continue
            best_time = min(item[0] for item in arrivals)
            tied = [item for item in arrivals if item[0] == best_time]
            best_length = max(item[1] for item in tied)
            winners = [item for item in tied if item[1] == best_length]
            if len(winners) == 1:
                territory[winners[0][2]] += 1

    my_dist = distances[you.sid]
    my_space = len(my_dist)
    my_territory = territory.get(you.sid, 0)
    strongest_enemy = max(
        (value for sid, value in territory.items() if sid != you.sid),
        default=0,
    )
    enemies = [snake for snake in state.snakes if snake.sid != you.sid]
    max_enemy_length = max((snake.length for snake in enemies), default=you.length)
    moves = _candidate_moves(state, you, width=4, ctx=ctx)
    safe_moves = _safe_mobility(state, you, moves, ctx)

    score = 0.0
    score += (my_territory - strongest_enemy) * 140.0
    score += my_territory * 15.0
    score += my_space * 50.0
    score += safe_moves * 1_050.0 + len(moves) * 160.0
    score += (you.length - max_enemy_length) * 250.0

    if my_space < you.length + 2:
        score -= (you.length + 2 - my_space) * 12_000.0

    nearest_food = min((my_dist[f] for f in state.food if f in my_dist), default=None)
    if nearest_food is not None:
        if you.health <= CRITICAL_HEALTH:
            score += max(-25_000.0, (you.health - nearest_food - 2) * 2_100.0)
        elif you.health < HUNGRY_HEALTH:
            score += (HUNGRY_HEALTH - nearest_food) * 250.0
    elif you.health < HUNGRY_HEALTH:
        score -= (HUNGRY_HEALTH - you.health) * 1_100.0

    score += min(you.health, 60) * 10.0
    if you.body[-1] in my_dist:
        score += 500.0

    if len(ctx.fast_eval_cache) < 20_000:
        ctx.fast_eval_cache[state] = score
    return score


def _static_distances(
    start: Point,
    blocked: Set[Point],
    width: int,
    height: int,
) -> Dict[Point, int]:
    distances: Dict[Point, int] = {start: 0}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        next_distance = distances[(x, y)] + 1
        for dx, dy in DIRECTIONS.values():
            nxt = (x + dx, y + dy)
            if (
                0 <= nxt[0] < width
                and 0 <= nxt[1] < height
                and nxt not in blocked
                and nxt not in distances
            ):
                distances[nxt] = next_distance
                queue.append(nxt)
    return distances

def _safe_mobility(
    state: SearchState,
    you: SnakeState,
    moves: Sequence[str],
    ctx: SearchContext,
) -> int:
    dangerous: Set[Point] = set()
    for enemy in state.snakes:
        if enemy.sid == you.sid or enemy.length < you.length:
            continue
        for move in _candidate_moves(state, enemy, width=4, ctx=ctx):
            dx, dy = DIRECTIONS[move]
            dangerous.add((enemy.head[0] + dx, enemy.head[1] + dy))

    count = 0
    for move in moves:
        dx, dy = DIRECTIONS[move]
        if (you.head[0] + dx, you.head[1] + dy) not in dangerous:
            count += 1
    return count


def _body_release_times(state: SearchState) -> Dict[Point, int]:
    """Earliest turn on which each currently occupied cell is nominally free."""
    release: Dict[Point, int] = {}
    for snake in state.snakes:
        length = snake.length
        for index, cell in enumerate(snake.body):
            turns_until_free = length - index
            # A coordinate appearing multiple times remains occupied until its
            # last stacked segment leaves, hence max rather than min.
            release[cell] = max(release.get(cell, 0), turns_until_free)
    return release


def _dynamic_distances(
    start: Point,
    release: Mapping[Point, int],
    width: int,
    height: int,
    horizon: int,
) -> Dict[Point, int]:
    """Optimistic time-aware shortest paths through moving bodies.

    Arrival times are small integers, so a bucket queue is faster than a heap
    on an 11x11 board. Entering a body cell is delayed until the segment should
    have vacated. The delay approximates spending those turns in a loop.
    """
    infinity = horizon + 1
    dist: Dict[Point, int] = {start: 0}
    buckets = [deque() for _ in range(horizon + 1)]
    buckets[0].append(start)

    for current_time in range(horizon + 1):
        bucket = buckets[current_time]
        while bucket:
            cell = bucket.popleft()
            if dist.get(cell) != current_time:
                continue
            x, y = cell
            for dx, dy in DIRECTIONS.values():
                nxt = (x + dx, y + dy)
                if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                    continue
                arrival = current_time + 1
                release_time = release.get(nxt, 0)
                if release_time > arrival:
                    arrival = release_time
                if arrival > horizon or arrival >= dist.get(nxt, infinity):
                    continue
                dist[nxt] = arrival
                buckets[arrival].append(nxt)
    return dist


# ---------------------------------------------------------------------------
# Small utilities


def _snake_by_id(state: SearchState, sid: str) -> Optional[SnakeState]:
    for snake in state.snakes:
        if snake.sid == sid:
            return snake
    return None


def _static_flood(start: Point, blocked: Set[Point], width: int, height: int) -> int:
    if not _in_bounds(start, width, height):
        return 0
    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS.values():
            nxt = (x + dx, y + dy)
            if nxt in seen or nxt in blocked or not _in_bounds(nxt, width, height):
                continue
            seen.add(nxt)
            queue.append(nxt)
    return len(seen)


def _in_bounds(point: Point, width: int, height: int) -> bool:
    return 0 <= point[0] < width and 0 <= point[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def choose_move_model(game_state: Dict) -> Optional[str]:
    """Compatibility wrapper for the previous model-backed interface."""
    return choose_move(game_state)


__all__ = [
    "get_info",
    "choose_move",
    "choose_move_heuristic",
    "choose_move_model",
]
