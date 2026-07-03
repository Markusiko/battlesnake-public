"""Deadline-aware hybrid Battlesnake policy for 11x11 four-player games.

The decision pipeline is deliberately defensive:

1. Produce an immediate legal fallback move.
2. Rank moves with tail-aware flood fill and head-to-head checks.
3. Simulate one simultaneous turn against plausible moves of every opponent.
4. Evaluate the resulting states with time-aware space, Voronoi territory,
   mobility, tail reachability, food races, and health slack.
5. Refine the worst root scenarios with a reduced depth-2 adversarial search.
6. Stop on a hard deadline and return the best result obtained so far.

Only the Python standard library is used.  The public entry points are
``get_info()``, ``choose_move()``, ``choose_move_model()`` and
``choose_move_heuristic()`` so this module can replace the original policy
without changing the HTTP layer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

Point = Tuple[int, int]

# Dict order is also the deterministic final tie-break order.
DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}
_DIRECTION_NAMES: Tuple[str, ...] = tuple(DIRECTIONS)
_NEIGHBORS: Tuple[Point, ...] = tuple(DIRECTIONS.values())

# Search configuration.  The request timeout includes network overhead, so the
# agent intentionally leaves a large reserve instead of using all 500 ms.
MAX_SEARCH_BUDGET_MS = 240.0
MIN_SEARCH_BUDGET_MS = 10.0
BASE_NETWORK_RESERVE_MS = 165.0

# At most three plausible moves are retained per enemy.  With three enemies
# this gives at most 27 joint scenarios per candidate move.
MAX_ENEMY_MOVES = 3
NEAR_ENEMY_DISTANCE = 6
MAX_SCENARIOS_PER_MOVE = 27
MAX_DEPTH2_SCENARIOS = 8
MAX_REFINED_ROOT_MOVES = 2
MAX_REFINED_STATES_PER_MOVE = 4

# Large sentinels enforce lexicographic priorities: survival first, then all
# positional considerations.
DEAD_SCORE = -1_000_000.0
WIN_SCORE = 1_000_000.0
_ILLEGAL_SCORE = -10_000_000.0
_BIG_DISTANCE = 10_000


@dataclass(frozen=True, slots=True)
class Snake:
    id: str
    body: Tuple[Point, ...]
    health: int

    @property
    def head(self) -> Point:
        return self.body[0]

    @property
    def tail(self) -> Point:
        return self.body[-1]

    @property
    def length(self) -> int:
        return len(self.body)


@dataclass(frozen=True, slots=True)
class State:
    width: int
    height: int
    food: FrozenSet[Point]
    hazards: Tuple[Point, ...]
    hazard_damage: int
    snakes: Tuple[Snake, ...]
    you_id: str


@dataclass
class ScenarioResult:
    score: float
    state: State


@dataclass
class MoveStats:
    fast_score: float
    scenarios: List[ScenarioResult]


# ---------------------------------------------------------------------------
# Public API


def get_info() -> Dict[str, str]:
    """Appearance and metadata returned from ``GET /``."""
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "0.3.0-hybrid",
    }


def choose_move(game_state: Dict) -> str:
    """Return a move before a conservative request-local deadline.

    Any malformed state or unexpected search failure falls back to a tiny,
    exception-safe heuristic.  Gameplay should never fail because analysis did.
    """
    try:
        move = choose_move_model(game_state)
        if move is not None:
            return move
    except Exception:  # noqa: BLE001 - a move response is more important than logs
        pass

    try:
        return choose_move_heuristic(game_state)
    except Exception:  # noqa: BLE001
        return "up"


def choose_move_model(game_state: Dict) -> Optional[str]:
    """Choose a move using deadline-aware simultaneous-move search.

    The name is retained for compatibility with the original module.  This is
    now a search policy rather than a fragile linear ranking model.
    """
    request_start = perf_counter()
    state = _parse_state(game_state)
    me = _find_snake(state, state.you_id)
    if me is None:
        return None

    candidates = _potential_moves(state, me)
    if not candidates:
        return _least_bad_direction(state, me)

    deadline = request_start + _search_budget_seconds(game_state)

    # This is computed before any expensive work and is always a valid
    # best-effort fallback if the search budget is exhausted.
    fast_scores = {move: _fast_move_score(state, me, move) for move in candidates}
    fallback = max(candidates, key=lambda move: (fast_scores[move], -_direction_rank(move)))

    if perf_counter() >= deadline:
        return fallback

    searched = _joint_search(state, me, candidates, fast_scores, deadline)
    return searched or fallback


def choose_move_heuristic(game_state: Dict) -> str:
    """Cheap tail-aware fallback used when full search cannot run."""
    state = _parse_state(game_state)
    me = _find_snake(state, state.you_id)
    if me is None:
        return "up"

    moves = _potential_moves(state, me)
    if not moves:
        return _least_bad_direction(state, me)
    return max(moves, key=lambda move: (_fast_move_score(state, me, move), -_direction_rank(move)))


# ---------------------------------------------------------------------------
# Parsing and timing


def _parse_state(game_state: Mapping) -> State:
    board = game_state["board"]
    you_id = str(game_state["you"]["id"])

    snakes: List[Snake] = []
    for raw in board.get("snakes", []):
        body = tuple((int(p["x"]), int(p["y"])) for p in raw.get("body", []))
        if not body:
            continue
        snakes.append(
            Snake(
                id=str(raw["id"]),
                body=body,
                health=int(raw.get("health", 100)),
            )
        )

    food = frozenset((int(p["x"]), int(p["y"])) for p in board.get("food", []))
    # Keep duplicates: some maps use stacked hazards.
    hazards = tuple((int(p["x"]), int(p["y"])) for p in board.get("hazards", []))

    settings = game_state.get("game", {}).get("ruleset", {}).get("settings", {})
    hazard_damage = int(settings.get("hazardDamagePerTurn", 0) or 0)

    return State(
        width=int(board["width"]),
        height=int(board["height"]),
        food=food,
        hazards=hazards,
        hazard_damage=hazard_damage,
        snakes=tuple(snakes),
        you_id=you_id,
    )


def _search_budget_seconds(game_state: Mapping) -> float:
    timeout_ms = float(game_state.get("game", {}).get("timeout", 500) or 500)

    # ``latency`` is the previous response latency when supplied by the engine.
    # It is often a string in API payloads, so parsing must be defensive.
    observed_latency = 0.0
    try:
        observed_latency = float(game_state.get("you", {}).get("latency", 0) or 0)
    except (TypeError, ValueError):
        observed_latency = 0.0

    reserve_ms = max(BASE_NETWORK_RESERVE_MS, observed_latency + 80.0)
    budget_ms = min(MAX_SEARCH_BUDGET_MS, timeout_ms - reserve_ms)
    budget_ms = max(MIN_SEARCH_BUDGET_MS, budget_ms)
    return budget_ms / 1000.0


# ---------------------------------------------------------------------------
# Search


def _joint_search(
    state: State,
    me: Snake,
    candidates: Sequence[str],
    fast_scores: Mapping[str, float],
    deadline: float,
) -> Optional[str]:
    """Evaluate root moves fairly, then deepen the worst plausible branches.

    Phase 1 performs an exact simultaneous one-turn simulation for plausible
    joint enemy moves. Scenarios are processed round-robin so a deadline never
    favors the first direction in dictionary order. Phase 2 spends remaining
    time on a reduced adversarial second ply for the two best root moves.
    """
    enemies = [snake for snake in state.snakes if snake.id != me.id]
    plans: Dict[str, List[Tuple[str, ...]]] = {}

    for our_move in candidates:
        target = _step(me.head, our_move)
        option_lists = [
            _plausible_enemy_moves(state, enemy, me, target)
            for enemy in enemies
        ]
        combos = list(product(*option_lists)) if option_lists else [tuple()]
        combos.sort(
            key=lambda combo: _scenario_threat_key(state, enemies, combo, target),
            reverse=True,
        )
        plans[our_move] = combos[:MAX_SCENARIOS_PER_MOVE]

    stats = {
        move: MoveStats(fast_score=fast_scores[move], scenarios=[])
        for move in candidates
    }
    indices = {move: 0 for move in candidates}
    eval_cache: Dict[Tuple, float] = {}

    while True:
        progressed = False
        for our_move in candidates:
            if perf_counter() >= deadline:
                return _best_aggregated_move(candidates, stats)

            index = indices[our_move]
            scenarios = plans[our_move]
            if index >= len(scenarios):
                continue

            enemy_combo = scenarios[index]
            indices[our_move] = index + 1
            progressed = True

            joint_moves = {me.id: our_move}
            for enemy, move in zip(enemies, enemy_combo):
                joint_moves[enemy.id] = move

            next_state = _simulate_joint_turn(state, joint_moves)
            score = _cached_evaluation(next_state, state.you_id, eval_cache)
            stats[our_move].scenarios.append(ScenarioResult(score, next_state))

        if not progressed:
            break

    # Import the useful idea from the paranoid minimax method, but apply it only
    # where it pays: the worst states of the best root moves. This keeps depth 2
    # practical under a 500 ms external timeout on a small Render instance.
    ranked = sorted(
        candidates,
        key=lambda move: _aggregate_move(move, stats),
        reverse=True,
    )
    _refine_depth_two(
        ranked[:MAX_REFINED_ROOT_MOVES],
        stats,
        state.you_id,
        deadline,
        eval_cache,
    )
    return _best_aggregated_move(candidates, stats)


def _cached_evaluation(state: State, you_id: str, cache: Dict[Tuple, float]) -> float:
    key = _state_key(state)
    score = cache.get(key)
    if score is None:
        score = _evaluate_state(state, you_id)
        cache[key] = score
    return score


def _refine_depth_two(
    root_moves: Sequence[str],
    stats: Mapping[str, MoveStats],
    you_id: str,
    deadline: float,
    eval_cache: Dict[Tuple, float],
) -> None:
    """Run a reduced second adversarial ply on each move's lower tail."""
    queues: Dict[str, List[ScenarioResult]] = {}
    for move in root_moves:
        queues[move] = sorted(
            stats[move].scenarios,
            key=lambda result: result.score,
        )[:MAX_REFINED_STATES_PER_MOVE]

    for index in range(MAX_REFINED_STATES_PER_MOVE):
        for move in root_moves:
            if perf_counter() >= deadline:
                return
            items = queues.get(move, [])
            if index >= len(items):
                continue
            result = items[index]
            deeper = _reduced_adversarial_value(
                result.state,
                you_id,
                deadline,
                eval_cache,
            )
            if deeper is not None:
                # Keep some immediate positional value, but make the exact second
                # ply dominant. Terminal scores naturally remain dominant.
                result.score = 0.35 * result.score + 0.65 * deeper


def _reduced_adversarial_value(
    state: State,
    you_id: str,
    deadline: float,
    eval_cache: Dict[Tuple, float],
) -> Optional[float]:
    me = _find_snake(state, you_id)
    if me is None:
        return DEAD_SCORE
    enemies = [snake for snake in state.snakes if snake.id != you_id]
    if not enemies:
        return WIN_SCORE

    candidates = _potential_moves(state, me)
    if not candidates:
        candidates = [_least_bad_direction(state, me)]

    best: Optional[float] = None
    for our_move in candidates:
        if perf_counter() >= deadline:
            break

        target = _step(me.head, our_move)
        option_lists: List[List[str]] = []
        for enemy in enemies:
            options = _plausible_enemy_moves(state, enemy, me, target)
            if len(options) > 2:
                direct = next(
                    (move for move in options if _step(enemy.head, move) == target),
                    None,
                )
                options = options[:2]
                if direct is not None and direct not in options:
                    options[-1] = direct
            option_lists.append(options)

        combos = list(product(*option_lists)) if option_lists else [tuple()]
        combos.sort(
            key=lambda combo: _scenario_threat_key(state, enemies, combo, target),
            reverse=True,
        )

        scores: List[float] = []
        for combo in combos[:MAX_DEPTH2_SCENARIOS]:
            if perf_counter() >= deadline:
                break
            joint_moves = {me.id: our_move}
            for enemy, move in zip(enemies, combo):
                joint_moves[enemy.id] = move
            next_state = _simulate_joint_turn(state, joint_moves)
            scores.append(_cached_evaluation(next_state, you_id, eval_cache))

        if not scores:
            continue
        value = _risk_sensitive_value(scores)
        if best is None or value > best:
            best = value

    return best


def _risk_sensitive_value(scores: Sequence[float]) -> float:
    ordered = sorted(scores)
    average = sum(ordered) / len(ordered)
    lower_index = min(len(ordered) - 1, int(0.20 * len(ordered)))
    lower_quantile = ordered[lower_index]
    worst = ordered[0]
    return 0.50 * average + 0.32 * lower_quantile + 0.18 * worst


def _aggregate_move(move: str, stats: Mapping[str, MoveStats]) -> Tuple[float, float, int]:
    item = stats[move]
    scores = [result.score for result in item.scenarios]
    if not scores:
        return item.fast_score, item.fast_score, -_direction_rank(move)

    combined = _risk_sensitive_value(scores) + 0.02 * item.fast_score
    return combined, item.fast_score, -_direction_rank(move)


def _best_aggregated_move(candidates: Sequence[str], stats: Mapping[str, MoveStats]) -> str:
    return max(candidates, key=lambda move: _aggregate_move(move, stats))


def _plausible_enemy_moves(state: State, enemy: Snake, me: Snake, target: Point) -> List[str]:
    moves = _potential_moves(state, enemy)
    if not moves:
        return [_least_bad_direction(state, enemy)]

    ranked = sorted(
        moves,
        key=lambda move: _enemy_move_priority(state, enemy, me, target, move),
        reverse=True,
    )

    limit = MAX_ENEMY_MOVES
    if _manhattan(enemy.head, me.head) > NEAR_ENEMY_DISTANCE:
        limit = min(2, MAX_ENEMY_MOVES)

    selected = ranked[:limit]

    # Never prune a direct contest of our destination: it is the most important
    # tactical reply even if the enemy's general heuristic dislikes it.
    direct = next((move for move in moves if _step(enemy.head, move) == target), None)
    if direct is not None and direct not in selected:
        if len(selected) >= limit:
            selected[-1] = direct
        else:
            selected.append(direct)
    return selected


def _enemy_move_priority(
    state: State,
    enemy: Snake,
    me: Snake,
    target: Point,
    move: str,
) -> float:
    destination = _step(enemy.head, move)
    score = 0.0

    if destination == target:
        # Strong snakes are more credible head-to-head threats.
        score += 20_000.0 if enemy.length >= me.length else 2_000.0

    score += 12.0 * _quick_space(state, enemy, move, cap=18)
    score += 25.0 * _free_neighbor_count_after_move(state, enemy, move)

    if destination in state.food:
        score += 700.0 if enemy.health < 55 else 100.0
    if destination in state.hazards:
        score -= 80.0 * max(1, state.hazard_damage) * _hazard_stack_count(state, destination)

    # A mild centre preference is only a tie-breaker, never a main objective.
    score -= 0.5 * _distance_to_center(destination, state.width, state.height)
    return score


def _scenario_threat_key(
    state: State,
    enemies: Sequence[Snake],
    combo: Sequence[str],
    our_target: Point,
) -> Tuple[int, int]:
    direct_contests = 0
    nearby_heads = 0
    for enemy, move in zip(enemies, combo):
        destination = _step(enemy.head, move)
        if destination == our_target:
            direct_contests += 1
        if _manhattan(destination, our_target) <= 1:
            nearby_heads += 1
    return direct_contests, nearby_heads


# ---------------------------------------------------------------------------
# Exact one-turn simulation


def _simulate_joint_turn(state: State, moves: Mapping[str, str]) -> State:
    """Apply the official standard-rules turn order for the known board state.

    Food spawning is intentionally omitted because its position is unknown. The
    deterministic parts are exact: move/pop tail, starvation, hazard damage
    (food protects from hazard damage), feeding/growth, then eliminations.
    """
    provisional: List[Snake] = []
    hazard_dead: Set[str] = set()
    eaten_food: Set[Point] = set()

    # Movement and regular health loss. Every snake moves, even if the selected
    # direction is fatal; this matters for simultaneous collisions.
    for snake in state.snakes:
        move = moves.get(snake.id)
        if move not in DIRECTIONS:
            move = _least_bad_direction(state, snake)

        new_head = _step(snake.head, move)
        moved_body = (new_head,) + snake.body[:-1]
        health = snake.health - 1

        # In the production rules, a food cell suppresses hazard damage for that
        # turn, then feeding restores health to 100. Stacked hazards multiply.
        if new_head in state.hazards and new_head not in state.food:
            health -= state.hazard_damage * _hazard_stack_count(state, new_head)
            if health <= 0:
                hazard_dead.add(snake.id)

        # Feeding happens after movement. Growth duplicates the *new* tail, not
        # the old pre-move tail. This is a subtle but important rules detail.
        if new_head in state.food and snake.id not in hazard_dead:
            if moved_body:
                moved_body = moved_body + (moved_body[-1],)
            health = 100
            eaten_food.add(new_head)

        provisional.append(Snake(id=snake.id, body=moved_body, health=health))

    # Health/out-of-bounds eliminations happen before collision checks. Bodies of
    # those snakes therefore cannot kill another snake during this turn.
    dead: Set[str] = set(hazard_dead)
    for snake in provisional:
        if snake.health <= 0 or not _in_bounds(snake.head, state.width, state.height):
            dead.add(snake.id)

    active = [snake for snake in provisional if snake.id not in dead]

    # Collision eliminations are collected before being applied. A snake already
    # scheduled to die from a body collision can still participate as the other
    # head in a head-to-head check, matching the engine's batched resolution.
    collision_dead: Set[str] = set()
    for snake in active:
        if snake.head in snake.body[1:]:
            collision_dead.add(snake.id)
            continue

        body_hit = False
        for other in active:
            if other.id == snake.id:
                continue
            if snake.head in other.body[1:]:
                collision_dead.add(snake.id)
                body_hit = True
                break
        if body_hit:
            continue

        for other in active:
            if other.id == snake.id:
                continue
            if snake.head == other.head and snake.length <= other.length:
                collision_dead.add(snake.id)
                break

    dead.update(collision_dead)
    survivors = tuple(snake for snake in provisional if snake.id not in dead)
    return State(
        width=state.width,
        height=state.height,
        food=frozenset(state.food.difference(eaten_food)),
        hazards=state.hazards,
        hazard_damage=state.hazard_damage,
        snakes=survivors,
        you_id=state.you_id,
    )


def _hazard_stack_count(state: State, point: Point) -> int:
    count = state.hazards.count(point)
    return max(1, count)


# ---------------------------------------------------------------------------
# State evaluation


def _evaluate_state(state: State, you_id: str) -> float:
    me = _find_snake(state, you_id)
    if me is None:
        return DEAD_SCORE

    enemies = [snake for snake in state.snakes if snake.id != you_id]
    if not enemies:
        return WIN_SCORE

    release_times = _release_times(state, you_id)
    arrival = _time_aware_distances(
        me.head,
        release_times,
        state.width,
        state.height,
    )
    reachable_area = len(arrival)
    space_margin = reachable_area - me.length
    reaches_tail = me.tail in arrival

    safe_moves = _safe_moves(state, me)
    mobility = len(safe_moves)

    territory, contested, distance_maps = _voronoi_territory(state, you_id)
    my_distances = distance_maps.get(you_id, {})

    nearest_food = min((my_distances.get(food, _BIG_DISTANCE) for food in state.food), default=_BIG_DISTANCE)
    food_control = _food_control_count(state, me, distance_maps)

    max_enemy_length = max((enemy.length for enemy in enemies), default=0)
    length_advantage = me.length - max_enemy_length

    score = 0.0

    # Fewer opponents is a large strategic gain.
    score -= 1_400.0 * len(enemies)

    # Space and territory.  Being unable to fit our body is treated as a severe
    # near-term failure, not merely a small positional disadvantage.
    score += 7.0 * reachable_area
    score += 11.0 * territory
    score += 1.5 * contested
    score += 28.0 * min(space_margin, 20)
    if space_margin < 0:
        score += 500.0 * space_margin
    elif space_margin <= 2:
        score -= 350.0

    # Local mobility catches corridors and one-way traps earlier than area alone.
    score += 260.0 * mobility
    if mobility == 0:
        score -= 50_000.0
    elif mobility == 1:
        score -= 500.0

    score += 150.0 if reaches_tail else -120.0
    score += 22.0 * length_advantage
    score += 90.0 * food_control

    # Health-aware food pressure uses actual path distance rather than Manhattan
    # distance.  Food is aggressively valued only when health slack is poor.
    if nearest_food < _BIG_DISTANCE:
        safety_buffer = 5
        health_slack = me.health - nearest_food - safety_buffer
        if health_slack < 0:
            score += 260.0 * health_slack
        urgency = max(0, 55 - me.health)
        score -= 2.2 * urgency * nearest_food
        if me.health > 75:
            score -= 1.0 * nearest_food
    elif me.health < 45:
        score -= 12_000.0

    if me.head in state.hazards:
        score -= 120.0 * max(1, state.hazard_damage) * _hazard_stack_count(state, me.head)

    return score


def _release_times(state: State, you_id: str) -> Dict[Point, int]:
    """Approximate the turn on which each occupied cell becomes traversable.

    Own-body timings are exact under no future growth.  Enemy bodies get a
    one-turn safety pad because their food choices are unknown beyond the exact
    simulated turn.
    """
    release: Dict[Point, int] = {}
    for snake in state.snakes:
        padding = 0 if snake.id == you_id else 1
        length = snake.length
        for index, cell in enumerate(snake.body):
            turns = length - index + padding
            if turns > release.get(cell, 0):
                release[cell] = turns
    return release


def _time_aware_distances(
    start: Point,
    release_times: Mapping[Point, int],
    width: int,
    height: int,
) -> Dict[Point, int]:
    distances: Dict[Point, int] = {start: 0}
    queue = deque([start])

    while queue:
        point = queue.popleft()
        next_time = distances[point] + 1
        for neighbor in _neighbors(point):
            if not _in_bounds(neighbor, width, height):
                continue
            if neighbor in distances:
                continue
            if next_time < release_times.get(neighbor, 0):
                continue
            distances[neighbor] = next_time
            queue.append(neighbor)
    return distances


def _voronoi_territory(
    state: State,
    you_id: str,
) -> Tuple[int, int, Dict[str, Dict[Point, int]]]:
    blocked = _occupied_cells(state.snakes)

    # Unstacked tails are likely to vacate and should not behave like permanent
    # walls in a territory estimate.  Exact one-turn safety is handled by search.
    for snake in state.snakes:
        if _tail_is_unstacked(snake):
            blocked.discard(snake.tail)

    distances: Dict[str, Dict[Point, int]] = {}
    for snake in state.snakes:
        distances[snake.id] = _bfs_distances(
            [snake.head],
            blocked,
            state.width,
            state.height,
        )

    me = _find_snake(state, you_id)
    if me is None:
        return 0, 0, distances

    owned = 0
    contested = 0
    for x in range(state.width):
        for y in range(state.height):
            cell = (x, y)
            arrivals = [
                (dist_map[cell], snake)
                for snake in state.snakes
                if cell in (dist_map := distances[snake.id])
            ]
            if not arrivals:
                continue

            best_distance = min(distance for distance, _ in arrivals)
            tied = [snake for distance, snake in arrivals if distance == best_distance]
            if not any(snake.id == you_id for snake in tied):
                continue

            if len(tied) == 1:
                owned += 1
                continue

            strongest_enemy = max(
                (snake.length for snake in tied if snake.id != you_id),
                default=-1,
            )
            if me.length > strongest_enemy:
                owned += 1
            else:
                contested += 1

    return owned, contested, distances


def _food_control_count(
    state: State,
    me: Snake,
    distances: Mapping[str, Mapping[Point, int]],
) -> int:
    count = 0
    my_dist = distances.get(me.id, {})
    for food in state.food:
        my_arrival = my_dist.get(food, _BIG_DISTANCE)
        if my_arrival == _BIG_DISTANCE:
            continue

        enemy_arrivals = [
            (distances.get(enemy.id, {}).get(food, _BIG_DISTANCE), enemy.length)
            for enemy in state.snakes
            if enemy.id != me.id
        ]
        best_enemy = min((distance for distance, _ in enemy_arrivals), default=_BIG_DISTANCE)
        if my_arrival < best_enemy:
            count += 1
        elif my_arrival == best_enemy:
            tied_enemy_length = max(
                (length for distance, length in enemy_arrivals if distance == my_arrival),
                default=-1,
            )
            if me.length > tied_enemy_length:
                count += 1
    return count


# ---------------------------------------------------------------------------
# Move generation and fast fallback scoring


def _potential_moves(state: State, snake: Snake) -> List[str]:
    """Moves that are not immediately blocked by a non-vacating body cell.

    Every unstacked tail is treated as potentially vacating.  If its owner eats,
    exact joint simulation keeps that tail and correctly eliminates a colliding
    snake.  This avoids both false bans and unsafe assumptions.
    """
    solid = _occupied_cells(state.snakes)
    for occupant in state.snakes:
        if _tail_is_unstacked(occupant):
            solid.discard(occupant.tail)

    moves: List[str] = []
    for move in _DIRECTION_NAMES:
        destination = _step(snake.head, move)
        if _in_bounds(destination, state.width, state.height) and destination not in solid:
            moves.append(move)
    return moves


def _safe_moves(state: State, snake: Snake) -> List[str]:
    safe: List[str] = []
    for move in _potential_moves(state, snake):
        destination = _step(snake.head, move)
        resulting_length = snake.length + (1 if destination in state.food else 0)
        if not _has_losing_head_to_head_threat(state, snake.id, resulting_length, destination):
            safe.append(move)
    return safe


def _fast_move_score(state: State, snake: Snake, move: str) -> float:
    destination = _step(snake.head, move)
    if not _in_bounds(destination, state.width, state.height):
        return _ILLEGAL_SCORE

    potential = _potential_moves(state, snake)
    if move not in potential:
        return _ILLEGAL_SCORE

    ate = destination in state.food
    resulting_length = snake.length + (1 if ate else 0)

    blocked = _occupied_cells(state.snakes)
    if not ate and _tail_is_unstacked(snake):
        blocked.discard(snake.tail)
    for other in state.snakes:
        if other.id != snake.id and _guaranteed_tail_vacates(state, other):
            blocked.discard(other.tail)
    blocked.discard(destination)

    space = _flood_fill_count(
        destination,
        blocked,
        state.width,
        state.height,
        limit=state.width * state.height,
    )
    margin = space - resulting_length
    escapes = _free_neighbor_count(destination, blocked, state.width, state.height)

    score = 9.0 * space + 45.0 * min(space, resulting_length + 3)
    score += 120.0 * escapes
    score += 35.0 * min(margin, 12)
    if margin < 0:
        score += 700.0 * margin
    if escapes == 0:
        score -= 30_000.0
    elif escapes == 1:
        score -= 350.0

    if _has_losing_head_to_head_threat(state, snake.id, resulting_length, destination):
        score -= 80_000.0
    else:
        score += 350.0 * _shorter_head_targets(state, snake.id, resulting_length, destination)

    # Entering an enemy tail is only conditionally safe: if that enemy eats on
    # the same turn, the tail does not move.  Exact search resolves this, while
    # the fallback receives a strong conservative penalty.
    for other in state.snakes:
        if other.id == snake.id or other.tail != destination:
            continue
        if not _guaranteed_tail_vacates(state, other):
            score -= 25_000.0

    if state.food:
        food_distance = min(_manhattan(destination, food) for food in state.food)
        urgency = max(0, 55 - snake.health)
        score -= 2.5 * urgency * food_distance
        if ate:
            score += 1_200.0 if snake.health < 55 else 120.0

    if destination in state.hazards:
        score -= 100.0 * max(1, state.hazard_damage) * _hazard_stack_count(state, destination)

    score -= 0.8 * _distance_to_center(destination, state.width, state.height)
    return score


def _quick_space(state: State, snake: Snake, move: str, cap: int) -> int:
    destination = _step(snake.head, move)
    blocked = _occupied_cells(state.snakes)
    if destination not in state.food and _tail_is_unstacked(snake):
        blocked.discard(snake.tail)
    blocked.discard(destination)
    return _flood_fill_count(destination, blocked, state.width, state.height, cap)


def _free_neighbor_count_after_move(state: State, snake: Snake, move: str) -> int:
    destination = _step(snake.head, move)
    blocked = _occupied_cells(state.snakes)
    if destination not in state.food and _tail_is_unstacked(snake):
        blocked.discard(snake.tail)
    blocked.discard(destination)
    return _free_neighbor_count(destination, blocked, state.width, state.height)


def _has_losing_head_to_head_threat(
    state: State,
    snake_id: str,
    resulting_length: int,
    destination: Point,
) -> bool:
    for enemy in state.snakes:
        if enemy.id == snake_id:
            continue
        enemy_resulting_length = enemy.length + (1 if destination in state.food else 0)
        if enemy_resulting_length < resulting_length:
            continue
        if _manhattan(enemy.head, destination) != 1:
            continue
        if any(_step(enemy.head, move) == destination for move in _potential_moves(state, enemy)):
            return True
    return False


def _shorter_head_targets(
    state: State,
    snake_id: str,
    resulting_length: int,
    destination: Point,
) -> int:
    count = 0
    for enemy in state.snakes:
        enemy_resulting_length = enemy.length + (1 if destination in state.food else 0)
        if enemy.id == snake_id or enemy_resulting_length >= resulting_length:
            continue
        if _manhattan(enemy.head, destination) != 1:
            continue
        if any(_step(enemy.head, move) == destination for move in _potential_moves(state, enemy)):
            count += 1
    return count


def _guaranteed_tail_vacates(state: State, snake: Snake) -> bool:
    if not _tail_is_unstacked(snake):
        return False
    # If any plausible head move reaches food, its tail might remain this turn.
    return not any(
        _step(snake.head, move) in state.food
        for move in _potential_moves_without_tail_assumption(state, snake)
    )


def _potential_moves_without_tail_assumption(state: State, snake: Snake) -> List[str]:
    solid = _occupied_cells(state.snakes)
    if _tail_is_unstacked(snake):
        solid.discard(snake.tail)
    return [
        move
        for move in _DIRECTION_NAMES
        if _in_bounds((destination := _step(snake.head, move)), state.width, state.height)
        and destination not in solid
    ]


def _least_bad_direction(state: State, snake: Snake) -> str:
    in_bounds = [
        move
        for move in _DIRECTION_NAMES
        if _in_bounds(_step(snake.head, move), state.width, state.height)
    ]
    if not in_bounds:
        return "up"

    occupied = _occupied_cells(state.snakes)
    return max(
        in_bounds,
        key=lambda move: (
            _step(snake.head, move) not in occupied,
            _free_neighbor_count(
                _step(snake.head, move),
                occupied,
                state.width,
                state.height,
            ),
            -_direction_rank(move),
        ),
    )


# Compatibility helper retained from the original module.
def _legal_moves(game_state: Dict) -> List[str]:
    state = _parse_state(game_state)
    me = _find_snake(state, state.you_id)
    return _potential_moves(state, me) if me is not None else []


# ---------------------------------------------------------------------------
# Generic graph helpers


def _occupied_cells(snakes: Iterable[Snake] | Iterable[Mapping]) -> Set[Point]:
    occupied: Set[Point] = set()
    for snake in snakes:
        if isinstance(snake, Snake):
            occupied.update(snake.body)
        else:
            occupied.update((int(p["x"]), int(p["y"])) for p in snake["body"])
    return occupied


def _bfs_distances(
    sources: Iterable[Point],
    blocked: Set[Point],
    width: int,
    height: int,
) -> Dict[Point, int]:
    distances: Dict[Point, int] = {}
    queue = deque()
    for source in sources:
        if source not in distances:
            distances[source] = 0
            queue.append(source)

    while queue:
        point = queue.popleft()
        next_distance = distances[point] + 1
        for neighbor in _neighbors(point):
            if not _in_bounds(neighbor, width, height):
                continue
            if neighbor in blocked or neighbor in distances:
                continue
            distances[neighbor] = next_distance
            queue.append(neighbor)
    return distances


def _flood_fill_count(
    start: Point,
    blocked: Set[Point],
    width: int,
    height: int,
    limit: int,
) -> int:
    if not _in_bounds(start, width, height):
        return 0

    seen = {start}
    stack = [start]
    while stack and len(seen) < limit:
        point = stack.pop()
        for neighbor in _neighbors(point):
            if not _in_bounds(neighbor, width, height):
                continue
            if neighbor in blocked or neighbor in seen:
                continue
            seen.add(neighbor)
            stack.append(neighbor)
            if len(seen) >= limit:
                break
    return len(seen)


# Original helper signature retained for drop-in compatibility.
def _flood_fill(start: Point, occupied: Set[Point], width: int, height: int, limit: int) -> int:
    return _flood_fill_count(start, occupied, width, height, limit)


def _free_neighbor_count(point: Point, blocked: Set[Point], width: int, height: int) -> int:
    return sum(
        1
        for neighbor in _neighbors(point)
        if _in_bounds(neighbor, width, height) and neighbor not in blocked
    )


def _tail_is_unstacked(snake: Snake) -> bool:
    return snake.length == 1 or snake.body[-1] != snake.body[-2]


def _find_snake(state: State, snake_id: str) -> Optional[Snake]:
    return next((snake for snake in state.snakes if snake.id == snake_id), None)


def _state_key(state: State) -> Tuple:
    return (
        state.width,
        state.height,
        state.food,
        state.hazards,
        state.hazard_damage,
        tuple((snake.id, snake.health, snake.body) for snake in state.snakes),
    )


def _neighbors(point: Point) -> Iterable[Point]:
    x, y = point
    for dx, dy in _NEIGHBORS:
        yield x + dx, y + dy


def _step(point: Point, move: str) -> Point:
    dx, dy = DIRECTIONS[move]
    return point[0] + dx, point[1] + dy


def _in_bounds(point: Point, width: int, height: int) -> bool:
    return 0 <= point[0] < width and 0 <= point[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _distance_to_center(point: Point, width: int, height: int) -> float:
    return abs(point[0] - (width - 1) / 2.0) + abs(point[1] - (height - 1) / 2.0)


def _direction_rank(move: str) -> int:
    try:
        return _DIRECTION_NAMES.index(move)
    except ValueError:
        return len(_DIRECTION_NAMES)