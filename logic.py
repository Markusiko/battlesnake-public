"""Competitive Battlesnake move selection.

Drop-in replacement for the project's logic.py.  It uses only the Python
standard library and combines:

* exact one-turn Battlesnake simulation (standard / royale / wrapped basics),
* tail-aware legal move generation,
* future-body release-time flood fill,
* length-aware Voronoi territory,
* contested-food and starvation scoring,
* head-to-head threat / kill handling,
* iterative-deepening paranoid search with a hard deadline,
* deterministic heuristic fallback.

The code is intentionally deterministic: identical states produce identical
moves, which makes replay debugging and A/B testing much easier.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import product
import math
import time
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

Point = Tuple[int, int]
GameState = Dict

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

# Stable tie-break order. It is deliberately fixed for reproducible replays.
MOVE_ORDER: Tuple[str, ...] = ("up", "left", "right", "down")
MOVE_RANK = {move: index for index, move in enumerate(MOVE_ORDER)}

WIN_SCORE = 1_000_000_000.0
LOSE_SCORE = -1_000_000_000.0
BIG = 10_000

# Search uses at most this much CPU time even if the API timeout is larger.
MAX_SEARCH_SECONDS = 0.27
# Keep room for framework overhead, JSON serialization, network jitter, etc.
TIMEOUT_RESERVE_MS = 150


class SearchTimeout(RuntimeError):
    """Raised internally when the move deadline has been reached."""


@dataclass
class SearchContext:
    you_id: str
    root_enemy_count: int
    deadline: float
    transposition: MutableMapping[Tuple, float] = field(default_factory=dict)
    nodes: int = 0

    def check_time(self) -> None:
        self.nodes += 1
        # Checking every node is cheap on an 11x11 board and prevents a single
        # wide joint-move layer from running past the API deadline.
        if time.perf_counter() >= self.deadline:
            raise SearchTimeout


def get_info() -> Dict[str, str]:
    """Metadata returned by the Battlesnake GET / endpoint."""
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#1d4ed8",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "1.0.0-hybrid-search",
    }


def choose_move(game_state: GameState) -> str:
    """Choose the strongest move completed within the request deadline.

    The last fully completed iterative-deepening result is always retained. If
    search cannot finish even depth 1, the deterministic positional ranking is
    returned instead.
    """
    started = time.perf_counter()
    you_id = game_state.get("you", {}).get("id", "")
    timeout_ms = int(game_state.get("game", {}).get("timeout", 500) or 500)
    compute_ms = max(20, timeout_ms - TIMEOUT_RESERVE_MS)
    budget = min(MAX_SEARCH_SECONDS, compute_ms / 1000.0)
    deadline = started + budget

    try:
        moves = _survival_moves(game_state, you_id)
        if not moves:
            # If every move dies from health/hazard, keep geometrically valid
            # moves so the fallback can at least choose the least bad outcome.
            moves = _geometric_moves(game_state, you_id)
        if not moves:
            return _default_move_for_snake(game_state.get("you", {}), game_state)
        if len(moves) == 1:
            return moves[0]

        ordered = sorted(
            moves,
            key=lambda move: (
                _quick_move_score(game_state, you_id, move),
                -MOVE_RANK[move],
            ),
            reverse=True,
        )
        fallback = ordered[0]

        snakes = game_state.get("board", {}).get("snakes", [])
        if len(snakes) <= 1 or time.perf_counter() >= deadline:
            return fallback

        context = SearchContext(
            you_id=you_id,
            root_enemy_count=max(0, len(snakes) - 1),
            deadline=deadline,
        )
        searched = _iterative_deepening(game_state, ordered, context)
        return searched or fallback
    except Exception:
        # A Battlesnake should never throw from /move. Keep this fallback tiny
        # and dependency-free so malformed edge states still receive a move.
        return _emergency_move(game_state)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def _iterative_deepening(
    state: GameState,
    ordered_root_moves: Sequence[str],
    context: SearchContext,
) -> Optional[str]:
    snake_count = len(state["board"]["snakes"])
    if snake_count == 2:
        max_depth = 8
    elif snake_count == 3:
        max_depth = 4
    else:
        max_depth = 3

    best_completed: Optional[str] = None
    principal_variation = list(ordered_root_moves)

    for depth in range(1, max_depth + 1):
        try:
            move, _score = _search_root(state, depth, principal_variation, context)
        except SearchTimeout:
            break
        best_completed = move
        # Principal-variation ordering makes the next depth much cheaper.
        principal_variation = [move] + [m for m in principal_variation if m != move]

    return best_completed


def _search_root(
    state: GameState,
    depth: int,
    ordered_moves: Sequence[str],
    context: SearchContext,
) -> Tuple[str, float]:
    context.check_time()
    alpha = -math.inf
    beta = math.inf
    best_move = ordered_moves[0]
    best_score = -math.inf

    for move in ordered_moves:
        context.check_time()
        score = _enemy_reply_value(state, move, depth, alpha, beta, context)
        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, best_score)

    return best_move, best_score


def _max_value(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    context: SearchContext,
) -> float:
    context.check_time()

    terminal = _terminal_value(state, context)
    if terminal is not None:
        return terminal
    if depth <= 0:
        return _evaluate_state(state, context)

    cache_key = ("max", depth, _state_key(state, context.you_id))
    cached = context.transposition.get(cache_key)
    if cached is not None:
        return cached

    moves = _survival_moves(state, context.you_id)
    if not moves:
        moves = _geometric_moves(state, context.you_id)
    if not moves:
        return LOSE_SCORE + state.get("turn", 0)

    moves.sort(
        key=lambda move: (
            _quick_move_score(state, context.you_id, move),
            -MOVE_RANK[move],
        ),
        reverse=True,
    )

    value = -math.inf
    fully_searched = True
    for move in moves:
        context.check_time()
        child = _enemy_reply_value(state, move, depth, alpha, beta, context)
        value = max(value, child)
        alpha = max(alpha, value)
        if alpha >= beta:
            fully_searched = False
            break

    if fully_searched:
        context.transposition[cache_key] = value
    return value


def _enemy_reply_value(
    state: GameState,
    my_move: str,
    depth: int,
    alpha: float,
    beta: float,
    context: SearchContext,
) -> float:
    """Return the worst plausible simultaneous enemy reply.

    Immediate lethal replies are never pruned from an enemy's candidate set.
    In large multiplayer positions the less relevant enemy options are capped
    to keep the joint branching factor bounded.
    """
    context.check_time()
    enemies = [
        snake
        for snake in state["board"]["snakes"]
        if snake["id"] != context.you_id
    ]

    if not enemies:
        next_state = _simulate_turn(state, {context.you_id: my_move})
        return _max_value(next_state, depth - 1, alpha, beta, context)

    my_snake = _snake_by_id(state, context.you_id)
    my_destination = None
    if my_snake is not None:
        my_destination = _step(_head(my_snake), my_move, state)

    option_lists: List[List[str]] = []
    enemy_count = len(enemies)
    per_enemy_cap = 4 if enemy_count <= 2 else (3 if enemy_count == 3 else 2)

    for enemy in enemies:
        options = _survival_moves(state, enemy["id"])
        if not options:
            options = _geometric_moves(state, enemy["id"])
        if not options:
            options = [_default_move_for_snake(enemy, state)]

        options.sort(
            key=lambda move: _enemy_move_order_score(
                state, enemy, move, my_destination, my_snake
            ),
            reverse=True,
        )
        option_lists.append(options[:per_enemy_cap])

    _cap_joint_options(option_lists, max_joint=96)

    # Replies likely to hit our destination are ordered first, improving
    # alpha-beta cutoffs around tactical head-to-head positions.
    replies = list(product(*option_lists))
    replies.sort(
        key=lambda joint: _joint_reply_order_score(
            state, enemies, joint, my_destination, my_snake
        ),
        reverse=True,
    )

    worst = math.inf
    fully_searched = True
    for joint in replies:
        context.check_time()
        moves_by_id = {context.you_id: my_move}
        for enemy, move in zip(enemies, joint):
            moves_by_id[enemy["id"]] = move

        next_state = _simulate_turn(state, moves_by_id)
        value = _max_value(next_state, depth - 1, alpha, beta, context)
        worst = min(worst, value)
        beta = min(beta, worst)
        if beta <= alpha:
            fully_searched = False
            break

    # If an unexpected empty product ever appears, preserve a legal fallback.
    if math.isinf(worst) and worst > 0:
        next_state = _simulate_turn(state, {context.you_id: my_move})
        worst = _max_value(next_state, depth - 1, alpha, beta, context)

    return worst


def _cap_joint_options(option_lists: List[List[str]], max_joint: int) -> None:
    """Trim the widest option lists until their cartesian product is bounded."""
    while option_lists:
        joint_count = math.prod(max(1, len(options)) for options in option_lists)
        if joint_count <= max_joint:
            return
        widest = max(range(len(option_lists)), key=lambda i: len(option_lists[i]))
        if len(option_lists[widest]) <= 1:
            return
        option_lists[widest].pop()


def _enemy_move_order_score(
    state: GameState,
    enemy: Dict,
    move: str,
    my_destination: Optional[Point],
    my_snake: Optional[Dict],
) -> float:
    score = _quick_move_score(state, enemy["id"], move)
    destination = _step(_head(enemy), move, state)
    if destination == my_destination and my_snake is not None:
        if enemy["length"] >= my_snake["length"]:
            score += 1_000_000.0
        else:
            score -= 1_000_000.0
    return score


def _joint_reply_order_score(
    state: GameState,
    enemies: Sequence[Dict],
    joint: Sequence[str],
    my_destination: Optional[Point],
    my_snake: Optional[Dict],
) -> float:
    score = 0.0
    for enemy, move in zip(enemies, joint):
        destination = _step(_head(enemy), move, state)
        if destination == my_destination and my_snake is not None:
            score += 10_000.0 if enemy["length"] >= my_snake["length"] else -10_000.0
        score += 0.001 * _quick_move_score(state, enemy["id"], move)
    return score


# ---------------------------------------------------------------------------
# Position evaluation
# ---------------------------------------------------------------------------


def _terminal_value(state: GameState, context: SearchContext) -> Optional[float]:
    you = _snake_by_id(state, context.you_id)
    if you is None:
        return LOSE_SCORE + state.get("turn", 0)
    enemies = [s for s in state["board"]["snakes"] if s["id"] != context.you_id]
    if not enemies:
        return WIN_SCORE + you["length"] * 100 + you["health"]
    return None


def _evaluate_state(state: GameState, context: SearchContext) -> float:
    you = _snake_by_id(state, context.you_id)
    if you is None:
        return LOSE_SCORE + state.get("turn", 0)

    board = state["board"]
    enemies = [s for s in board["snakes"] if s["id"] != context.you_id]
    if not enemies:
        return WIN_SCORE + you["length"] * 100 + you["health"]

    width, height = board["width"], board["height"]
    board_area = width * height
    head = _head(you)
    blocked = _blocked_for_pathing(state)
    blocked.discard(head)

    static_space = _flood_fill_count(state, head, blocked, limit=board_area)
    temporal_space, temporal_depth = _temporal_reachability(
        state, context.you_id, max_time=min(board_area, you["length"] + 24)
    )
    territory, contested = _voronoi_territory(state, context.you_id)
    mobility = len(_survival_moves(state, context.you_id))
    tail_reachable = _tail_reachable(state, context.you_id)
    food_score = _food_state_score(state, context.you_id)

    max_enemy_length = max((s["length"] for s in enemies), default=you["length"])
    length_advantage = you["length"] - max_enemy_length
    killed = max(0, context.root_enemy_count - len(enemies))

    # A region only slightly larger than the body is often a delayed death.
    required_space = min(board_area, you["length"] + max(2, you["length"] // 4))
    space_deficit = max(0, required_space - static_space)
    temporal_deficit = max(0, you["length"] + 2 - temporal_space)

    score = 0.0
    score += killed * 15_000.0
    score -= len(enemies) * 300.0
    score += territory * 28.0
    score += contested * 4.0
    score += static_space * 5.0
    score += temporal_space * 8.0
    score += temporal_depth * 12.0
    score += mobility * 95.0
    score += 230.0 if tail_reachable else -170.0
    score += length_advantage * 95.0
    score += you["health"] * 2.2
    score += food_score
    score -= space_deficit * 650.0
    score -= temporal_deficit * 900.0

    if mobility == 0:
        score -= 25_000.0
    elif mobility == 1:
        score -= 750.0

    # Penalize standing in hazards, especially with low health. Stacked hazards
    # appear multiple times in board.hazards.
    hazard_stacks = _hazard_counter(state)
    stack = hazard_stacks.get(head, 0)
    if stack:
        damage = _hazard_damage(state) * stack
        score -= damage * (12.0 if you["health"] < 45 else 5.0)

    # Length-aware tactical pressure. A long snake should convert trapped short
    # opponents, but should not hover near a longer head without compensation.
    for enemy in enemies:
        distance = _board_distance(state, head, _head(enemy))
        enemy_mobility = len(_survival_moves(state, enemy["id"]))
        if enemy["length"] < you["length"]:
            if distance <= 2:
                score += 80.0
            if enemy_mobility <= 1:
                score += 220.0
        elif distance <= 2:
            score -= 180.0

    return score


def _quick_move_score(state: GameState, snake_id: str, move: str) -> float:
    """Cheap root/move-order score; tactical search remains authoritative."""
    snake = _snake_by_id(state, snake_id)
    if snake is None:
        return LOSE_SCORE

    board = state["board"]
    destination = _step(_head(snake), move, state)
    if not _point_is_on_board(destination, state):
        return LOSE_SCORE

    blocked = _blocked_after_tails_move(state)
    if destination in blocked:
        return LOSE_SCORE

    health_after = _health_after_entering(state, snake, destination)
    score = 0.0
    if health_after <= 0:
        score -= 5_000_000.0

    blocked.discard(destination)
    space = _flood_fill_count(
        state,
        destination,
        blocked,
        limit=board["width"] * board["height"],
    )
    exits = _open_degree(state, destination, blocked)
    wall_distance = _wall_distance(destination, state)

    score += space * 16.0
    score += exits * 55.0
    score += wall_distance * 4.0

    # Hard warning for losing/tied head-to-head destinations.
    for enemy in board["snakes"]:
        if enemy["id"] == snake_id:
            continue
        enemy_destinations = {
            _step(_head(enemy), candidate, state)
            for candidate in _survival_moves(state, enemy["id"])
        }
        if destination in enemy_destinations:
            if enemy["length"] >= snake["length"]:
                score -= 500_000.0
            else:
                score += 6_000.0

    foods = {_point(item) for item in board.get("food", [])}
    if destination in foods:
        urgency = max(0, 75 - snake["health"])
        score += 120.0 + urgency * 18.0
    elif foods and snake["health"] < 60:
        nearest = min(_board_distance(state, destination, food) for food in foods)
        score += (board["width"] + board["height"] - nearest) * (8.0 if snake["health"] < 30 else 3.0)

    if destination in _hazard_counter(state) and destination not in foods:
        score -= _hazard_damage(state) * _hazard_counter(state)[destination] * 20.0

    return score


def _food_state_score(state: GameState, snake_id: str) -> float:
    snake = _snake_by_id(state, snake_id)
    if snake is None:
        return -10_000.0
    foods = [_point(food) for food in state["board"].get("food", [])]
    if not foods or _is_constrictor(state):
        return 0.0

    blocked = _blocked_for_pathing(state)
    blocked.discard(_head(snake))
    my_dist = _distance_map(state, [_head(snake)], blocked)

    enemy_maps: List[Tuple[Dict, Dict[Point, int]]] = []
    for enemy in state["board"]["snakes"]:
        if enemy["id"] == snake_id:
            continue
        enemy_blocked = set(blocked)
        enemy_blocked.discard(_head(enemy))
        enemy_maps.append((enemy, _distance_map(state, [_head(enemy)], enemy_blocked)))

    health = snake["health"]
    best = -math.inf
    reachable_costs: List[int] = []
    health_costs = _food_health_costs(state, snake_id)

    for food in foods:
        distance = my_dist.get(food, BIG)
        if distance >= BIG:
            continue
        reachable_costs.append(health_costs.get(food, distance))

        enemy_arrivals = [
            (distances.get(food, BIG), enemy["length"])
            for enemy, distances in enemy_maps
        ]
        enemy_distance = min((d for d, _length in enemy_arrivals), default=BIG)
        enemy_equal_or_longer = any(
            d <= distance and length >= snake["length"]
            for d, length in enemy_arrivals
        )

        if health < 25:
            urgency = (25 - health) * 30.0
            distance_cost = distance * 28.0
        elif health < 55:
            urgency = (55 - health) * 9.0
            distance_cost = distance * 12.0
        else:
            urgency = max(0, 75 - health) * 2.0
            distance_cost = distance * 4.0

        contest = 0.0
        if distance < enemy_distance:
            contest += 180.0
        elif enemy_equal_or_longer:
            contest -= 420.0
        else:
            contest -= 80.0

        value = urgency - distance_cost + contest
        best = max(best, value)

    if not reachable_costs:
        return -3_500.0 if health < 50 else -500.0

    cheapest_health_cost = min(reachable_costs)
    starvation_margin = health - cheapest_health_cost
    starvation_penalty = 0.0
    if starvation_margin <= 0:
        starvation_penalty = 6_000.0
    elif starvation_margin < 8:
        starvation_penalty = (8 - starvation_margin) * 350.0

    return (best if best > -math.inf else 0.0) - starvation_penalty


def _food_health_costs(state: GameState, snake_id: str) -> Dict[Point, int]:
    """Dijkstra cost to food where hazards consume additional health."""
    snake = _snake_by_id(state, snake_id)
    if snake is None:
        return {}
    foods = {_point(food) for food in state["board"].get("food", [])}
    blocked = _blocked_for_pathing(state)
    start = _head(snake)
    blocked.discard(start)
    hazards = _hazard_counter(state)
    hazard_damage = _hazard_damage(state)

    distances: Dict[Point, int] = {start: 0}
    heap: List[Tuple[int, Point]] = [(0, start)]
    found: Dict[Point, int] = {}

    while heap and len(found) < len(foods):
        cost, point = heappop(heap)
        if cost != distances.get(point):
            continue
        if point in foods:
            found[point] = cost
        for neighbor in _neighbors(point, state):
            if neighbor in blocked:
                continue
            step_cost = 1
            if neighbor not in foods:
                step_cost += hazards.get(neighbor, 0) * hazard_damage
            new_cost = cost + step_cost
            if new_cost < distances.get(neighbor, BIG):
                distances[neighbor] = new_cost
                heappush(heap, (new_cost, neighbor))

    return found


def _voronoi_territory(state: GameState, you_id: str) -> Tuple[float, float]:
    board = state["board"]
    snakes = board["snakes"]
    you = _snake_by_id(state, you_id)
    if you is None:
        return 0.0, 0.0

    blocked = _blocked_for_pathing(state)
    for snake in snakes:
        blocked.discard(_head(snake))

    maps: Dict[str, Dict[Point, int]] = {
        snake["id"]: _distance_map(state, [_head(snake)], blocked)
        for snake in snakes
    }
    lengths = {snake["id"]: snake["length"] for snake in snakes}

    owned = 0.0
    contested = 0.0
    for x in range(board["width"]):
        for y in range(board["height"]):
            point = (x, y)
            if point in blocked:
                continue
            arrivals = [(maps[snake["id"]].get(point, BIG), snake["id"]) for snake in snakes]
            best_distance = min(distance for distance, _snake_id in arrivals)
            if best_distance >= BIG:
                continue
            first = [snake_id for distance, snake_id in arrivals if distance == best_distance]
            if first == [you_id]:
                owned += 1.0
            elif you_id in first:
                strongest = max(lengths[snake_id] for snake_id in first)
                if lengths[you_id] > max(
                    (lengths[snake_id] for snake_id in first if snake_id != you_id),
                    default=-1,
                ):
                    owned += 0.75
                elif lengths[you_id] == strongest:
                    contested += 1.0

    return owned, contested


def _temporal_reachability(
    state: GameState,
    snake_id: str,
    max_time: int,
) -> Tuple[int, int]:
    """Reachable cells while current bodies disappear on their tail schedule."""
    snake = _snake_by_id(state, snake_id)
    if snake is None:
        return 0, 0

    release = _body_release_times(state)
    start = _head(snake)
    earliest: Dict[Point, int] = {start: 0}
    queue = deque([start])
    max_depth = 0

    while queue:
        point = queue.popleft()
        current_time = earliest[point]
        max_depth = max(max_depth, current_time)
        if current_time >= max_time:
            continue
        arrival = current_time + 1
        for neighbor in _neighbors(point, state):
            if arrival < release.get(neighbor, 0):
                continue
            if arrival >= earliest.get(neighbor, BIG):
                continue
            earliest[neighbor] = arrival
            queue.append(neighbor)

    return len(earliest), max_depth


def _body_release_times(state: GameState) -> Dict[Point, int]:
    release: Dict[Point, int] = {}
    for snake in state["board"]["snakes"]:
        body = [_point(segment) for segment in snake["body"]]
        length = len(body)
        for index, point in enumerate(body):
            turns = length - index
            release[point] = max(release.get(point, 0), turns)
    return release


def _tail_reachable(state: GameState, snake_id: str) -> bool:
    snake = _snake_by_id(state, snake_id)
    if snake is None or not snake.get("body"):
        return False
    head = _head(snake)
    tail = _point(snake["body"][-1])
    blocked = _blocked_for_pathing(state)
    blocked.discard(head)
    blocked.discard(tail)
    return tail in _distance_map(state, [head], blocked)


# ---------------------------------------------------------------------------
# Move generation and board helpers
# ---------------------------------------------------------------------------


def _geometric_moves(state: GameState, snake_id: str) -> List[str]:
    snake = _snake_by_id(state, snake_id)
    if snake is None:
        return []
    blocked = _blocked_after_tails_move(state)
    moves: List[str] = []
    for move in MOVE_ORDER:
        destination = _step(_head(snake), move, state)
        if _point_is_on_board(destination, state) and destination not in blocked:
            moves.append(move)
    return moves


def _survival_moves(state: GameState, snake_id: str) -> List[str]:
    snake = _snake_by_id(state, snake_id)
    if snake is None:
        return []
    moves = []
    for move in _geometric_moves(state, snake_id):
        destination = _step(_head(snake), move, state)
        if _health_after_entering(state, snake, destination) > 0:
            moves.append(move)
    return moves


def _health_after_entering(state: GameState, snake: Dict, destination: Point) -> int:
    if _is_constrictor(state):
        return snake.get("health", 100)
    foods = {_point(food) for food in state["board"].get("food", [])}
    if destination in foods:
        return 100
    return (
        snake.get("health", 100)
        - 1
        - _hazard_counter(state).get(destination, 0) * _hazard_damage(state)
    )


def _blocked_after_tails_move(state: GameState) -> Set[Point]:
    occupancy: Counter = Counter()
    for snake in state["board"]["snakes"]:
        body = [_point(segment) for segment in snake["body"]]
        occupancy.update(body)
        # Movement always pops one tail segment. A duplicated tail remains
        # occupied because only one of the stacked segments is removed.
        if len(body) >= 2 and body[-1] != body[-2]:
            occupancy[body[-1]] -= 1
    return {point for point, count in occupancy.items() if count > 0}


def _blocked_for_pathing(state: GameState) -> Set[Point]:
    """Current bodies with heads and definitely-vacating tails opened."""
    blocked = _blocked_after_tails_move(state)
    for snake in state["board"]["snakes"]:
        blocked.discard(_head(snake))
    return blocked


def _open_degree(state: GameState, point: Point, blocked: Set[Point]) -> int:
    return sum(1 for neighbor in _neighbors(point, state) if neighbor not in blocked)


def _flood_fill_count(
    state: GameState,
    start: Point,
    blocked: Set[Point],
    limit: int,
) -> int:
    if start in blocked or not _point_is_on_board(start, state):
        return 0
    seen = {start}
    stack = [start]
    while stack and len(seen) < limit:
        point = stack.pop()
        for neighbor in _neighbors(point, state):
            if neighbor in blocked or neighbor in seen:
                continue
            seen.add(neighbor)
            stack.append(neighbor)
            if len(seen) >= limit:
                break
    return len(seen)


def _distance_map(
    state: GameState,
    sources: Iterable[Point],
    blocked: Set[Point],
) -> Dict[Point, int]:
    distances: Dict[Point, int] = {}
    queue = deque()
    for source in sources:
        if _point_is_on_board(source, state) and source not in distances:
            distances[source] = 0
            queue.append(source)

    while queue:
        point = queue.popleft()
        distance = distances[point]
        for neighbor in _neighbors(point, state):
            if neighbor in blocked or neighbor in distances:
                continue
            distances[neighbor] = distance + 1
            queue.append(neighbor)
    return distances


def _neighbors(point: Point, state: GameState) -> Iterable[Point]:
    for move in MOVE_ORDER:
        neighbor = _step(point, move, state)
        if _point_is_on_board(neighbor, state):
            yield neighbor


def _step(point: Point, move: str, state: GameState) -> Point:
    dx, dy = DIRECTIONS[move]
    x, y = point[0] + dx, point[1] + dy
    if _is_wrapped(state):
        width = state["board"]["width"]
        height = state["board"]["height"]
        return x % width, y % height
    return x, y


def _point_is_on_board(point: Point, state: GameState) -> bool:
    if _is_wrapped(state):
        return True
    return 0 <= point[0] < state["board"]["width"] and 0 <= point[1] < state["board"]["height"]


def _wall_distance(point: Point, state: GameState) -> int:
    if _is_wrapped(state):
        return min(state["board"]["width"], state["board"]["height"]) // 2
    width, height = state["board"]["width"], state["board"]["height"]
    return min(point[0], width - 1 - point[0], point[1], height - 1 - point[1])


def _board_distance(state: GameState, a: Point, b: Point) -> int:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    if _is_wrapped(state):
        dx = min(dx, state["board"]["width"] - dx)
        dy = min(dy, state["board"]["height"] - dy)
    return dx + dy


def _snake_by_id(state: GameState, snake_id: str) -> Optional[Dict]:
    for snake in state.get("board", {}).get("snakes", []):
        if snake.get("id") == snake_id:
            return snake
    return None


def _head(snake: Mapping) -> Point:
    head = snake.get("head") or snake.get("body", [{}])[0]
    return int(head["x"]), int(head["y"])


def _point(item: Mapping) -> Point:
    return int(item["x"]), int(item["y"])


def _hazard_counter(state: GameState) -> Counter:
    return Counter(_point(hazard) for hazard in state["board"].get("hazards", []))


def _hazard_damage(state: GameState) -> int:
    settings = state.get("game", {}).get("ruleset", {}).get("settings", {})
    return int(settings.get("hazardDamagePerTurn", 14) or 0)


def _ruleset_name(state: GameState) -> str:
    return str(state.get("game", {}).get("ruleset", {}).get("name", "standard")).lower()


def _map_name(state: GameState) -> str:
    return str(state.get("game", {}).get("map", "standard")).lower()


def _is_wrapped(state: GameState) -> bool:
    return "wrapped" in _ruleset_name(state) or "wrapped" in _map_name(state)


def _is_constrictor(state: GameState) -> bool:
    return "constrictor" in _ruleset_name(state) or "constrictor" in _map_name(state)


def _default_move_for_snake(snake: Mapping, state: GameState) -> str:
    body = snake.get("body", [])
    if len(body) >= 2:
        head = _point(body[0])
        neck = _point(body[1])
        for move in MOVE_ORDER:
            if _step(neck, move, state) == head:
                return move
    return "up"


def _emergency_move(state: GameState) -> str:
    you_id = state.get("you", {}).get("id", "")
    moves = _geometric_moves(state, you_id)
    if moves:
        return moves[0]
    return _default_move_for_snake(state.get("you", {}), state)


# ---------------------------------------------------------------------------
# Rules-accurate turn simulation (core standard rules + wrapped/constrictor)
# ---------------------------------------------------------------------------


def _simulate_turn(state: GameState, moves_by_id: Mapping[str, str]) -> GameState:
    board = state["board"]
    width, height = board["width"], board["height"]
    foods = {_point(food) for food in board.get("food", [])}
    hazards = _hazard_counter(state)
    hazard_damage = _hazard_damage(state)
    wrapped = _is_wrapped(state)
    constrictor = _is_constrictor(state)

    moved: List[Dict] = []
    pre_eliminated: Set[str] = set()

    # Official standard order: movement (append head / pop tail), starvation,
    # hazard damage, feeding/growth, then elimination/collisions.
    for original in board["snakes"]:
        snake_id = original["id"]
        move = moves_by_id.get(snake_id)
        if move not in DIRECTIONS:
            move = _default_move_for_snake(original, state)

        old_body = [_point(segment) for segment in original["body"]]
        old_head = old_body[0]
        new_head = _step(old_head, move, state)
        new_body = [new_head] + old_body[:-1]

        health = int(original.get("health", 100))
        if not constrictor:
            health -= 1

        out_of_bounds = not wrapped and not (0 <= new_head[0] < width and 0 <= new_head[1] < height)
        eating = new_head in foods

        # In the official standard pipeline, food protects against hazard
        # damage on the same square and then restores health to 100.
        if not constrictor and not eating:
            health -= hazards.get(new_head, 0) * hazard_damage
        health = max(0, min(100, health))

        if eating:
            if new_body:
                new_body.append(new_body[-1])
            health = 100
        elif constrictor and new_body:
            new_body.append(new_body[-1])

        simulated = {
            "id": snake_id,
            "name": original.get("name", snake_id),
            "health": health,
            "body": [{"x": x, "y": y} for x, y in new_body],
            "head": {"x": new_head[0], "y": new_head[1]},
            "length": len(new_body),
            "latency": original.get("latency", "0"),
            "shout": original.get("shout", ""),
            "squad": original.get("squad", ""),
            "_eating": eating,
        }
        moved.append(simulated)
        if out_of_bounds or health <= 0:
            pre_eliminated.add(snake_id)

    # Collision checks use only snakes still alive after health/bounds checks.
    alive_for_collisions = [s for s in moved if s["id"] not in pre_eliminated]
    collision_eliminated: Set[str] = set()

    # Self/body collisions are checked before head-to-head, matching the
    # production rules pipeline.
    for snake in alive_for_collisions:
        head = _head(snake)
        own_body = [_point(segment) for segment in snake["body"]][1:]
        if head in own_body:
            collision_eliminated.add(snake["id"])
            continue

        for other in alive_for_collisions:
            if other["id"] == snake["id"]:
                continue
            other_body = [_point(segment) for segment in other["body"]][1:]
            if head in other_body:
                collision_eliminated.add(snake["id"])
                break

    # Head-to-head is checked after body-collision detection, but collision
    # eliminations are only applied after every snake has been inspected. This
    # means a snake already marked for a body collision can still be the
    # opponent that eliminates another snake head-to-head, matching the
    # production rules implementation.
    for snake in alive_for_collisions:
        if snake["id"] in collision_eliminated:
            continue
        snake_head = _head(snake)
        for other in alive_for_collisions:
            if other["id"] == snake["id"]:
                continue
            if _head(other) == snake_head and snake["length"] <= other["length"]:
                collision_eliminated.add(snake["id"])
                break

    # Food is consumed before collision elimination. Therefore a snake that
    # eats and then dies in a body/head collision still removes that food.
    eaten_food: Set[Point] = {
        _head(snake)
        for snake in moved
        if snake["id"] not in pre_eliminated and snake.get("_eating", False)
    }

    eliminated = pre_eliminated | collision_eliminated
    live_snakes: List[Dict] = []
    for snake in moved:
        snake.pop("_eating", None)
        if snake["id"] in eliminated:
            continue
        live_snakes.append(snake)

    next_food = [
        {"x": x, "y": y}
        for x, y in sorted(foods - eaten_food)
    ]

    you_id = state.get("you", {}).get("id", "")
    live_you = next((snake for snake in live_snakes if snake["id"] == you_id), None)
    if live_you is None:
        old_you = state.get("you", {})
        next_you = {
            **old_you,
            "health": 0,
            "body": [],
            "length": 0,
        }
    else:
        next_you = live_you

    return {
        "game": state.get("game", {}),
        "turn": int(state.get("turn", 0)) + 1,
        "board": {
            "height": height,
            "width": width,
            "food": next_food,
            "hazards": list(board.get("hazards", [])),
            "snakes": live_snakes,
        },
        "you": next_you,
    }


# ---------------------------------------------------------------------------
# Transposition key
# ---------------------------------------------------------------------------


def _state_key(state: GameState, you_id: str) -> Tuple:
    snakes = []
    for snake in sorted(state["board"]["snakes"], key=lambda item: item["id"]):
        body = tuple(_point(segment) for segment in snake["body"])
        snakes.append((snake["id"], snake["health"], body))
    food = tuple(sorted(_point(item) for item in state["board"].get("food", [])))
    hazards = tuple(sorted(_point(item) for item in state["board"].get("hazards", [])))
    return (
        state.get("turn", 0),
        state["board"]["width"],
        state["board"]["height"],
        _ruleset_name(state),
        _map_name(state),
        you_id,
        tuple(snakes),
        food,
        hazards,
    )


# Compatibility aliases for older tests/imports in the project.
choose_move_safety_first = choose_move
choose_move_heuristic = choose_move
choose_move_minimax = choose_move
