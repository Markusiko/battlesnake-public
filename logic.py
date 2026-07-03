"""Fast search policy for Battlesnake Standard (11x11, four snakes).

Drop-in interface compatible with the provided baseline:
    get_info() -> dict
    choose_move(game_state: dict) -> str

The implementation is dependency-free and uses:
- exact simultaneous one-turn simulation;
- iterative-deepening multi-agent search;
- probability-weighted robust aggregation of opponent responses;
- Voronoi/space/tail/food/head-to-head evaluation;
- a strict deadline derived from game.timeout.

Coordinates follow the Battlesnake API: (0, 0) is bottom-left.
"""

from __future__ import annotations

import itertools
import math
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

# Stable tie-breaking makes local testing reproducible.
MOVE_ORDER: Tuple[str, ...] = ("up", "left", "right", "down")

WIN_SCORE = 1_000_000_000.0
LOSS_SCORE = -WIN_SCORE
INF = 10**9

# Leave room for JSON parsing, framework overhead, networking, and jitter.
DEFAULT_COMPUTE_BUDGET_S = 0.240
MIN_COMPUTE_BUDGET_S = 0.055
NETWORK_MARGIN_S = 0.200

# Search branching controls. The first layer is deliberately broader because
# tactical errors at the root are the most expensive.
ROOT_FULL_JOINT_LIMIT = 64
ROOT_DEEP_JOINT_LIMIT = 8
DEEP_JOINT_LIMIT = 2
MAX_SEARCH_DEPTH = 4


@dataclass(frozen=True)
class Snake:
    id: str
    health: int
    body: Tuple[Point, ...]

    @property
    def head(self) -> Point:
        return self.body[0]

    @property
    def tail(self) -> Point:
        return self.body[-1]

    @property
    def length(self) -> int:
        return len(self.body)


@dataclass(frozen=True)
class State:
    width: int
    height: int
    food: FrozenSet[Point]
    hazards: Tuple[Point, ...]
    snakes: Tuple[Snake, ...]
    you_id: str
    hazard_damage: int

    def snake(self, snake_id: str) -> Optional[Snake]:
        for snake in self.snakes:
            if snake.id == snake_id:
                return snake
        return None


class SearchTimeout(RuntimeError):
    pass


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from GET /."""
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "1.0.0-search",
    }


def choose_move(game_state: Dict) -> str:
    """Return a legal move before the Battlesnake deadline.

    This function never raises: on malformed input or an interrupted search it
    falls back to a cheap safety policy.
    """
    try:
        state = _parse_state(game_state)
        timeout_ms = int(game_state.get("game", {}).get("timeout", 500))
        available_s = max(MIN_COMPUTE_BUDGET_S, timeout_ms / 1000.0 - NETWORK_MARGIN_S)
        budget_s = min(DEFAULT_COMPUTE_BUDGET_S, available_s)
        deadline = time.perf_counter() + budget_s
        return SearchEngine(state, deadline).choose()
    except Exception:  # noqa: BLE001 - gameplay must always return a move
        try:
            return _fallback_move_from_api(game_state)
        except Exception:  # noqa: BLE001
            return "up"


class SearchEngine:
    def __init__(self, root: State, deadline: float) -> None:
        self.root = root
        self.deadline = deadline
        self.nodes = 0
        self.tt: Dict[Tuple[State, int], float] = {}
        self.eval_cache: Dict[State, float] = {}
        self.move_cache: Dict[Tuple[State, str], Tuple[str, ...]] = {}
        self.prior_cache: Dict[Tuple[State, str], Tuple[Tuple[str, float], ...]] = {}

    def choose(self) -> str:
        legal = list(self._legal_moves(self.root, self.root.you_id))
        if not legal:
            return _fallback_move_from_state(self.root)
        if len(legal) == 1:
            return legal[0]

        # Always establish a complete cheap answer first.
        legal.sort(key=lambda move: self._root_static_score(move), reverse=True)
        best_move = legal[0]
        best_value = LOSS_SCORE

        alive = len(self.root.snakes)
        depth_cap = 3 if alive >= 3 else MAX_SEARCH_DEPTH

        for depth in range(1, depth_cap + 1):
            iteration_best = best_move
            iteration_value = LOSS_SCORE
            try:
                # Principal-variation ordering from the previous completed depth.
                ordered = [best_move] + [m for m in legal if m != best_move]
                for move in ordered:
                    self._check_deadline()
                    value = self._value_of_our_move(self.root, move, depth, ply=0)
                    if value > iteration_value:
                        iteration_value = value
                        iteration_best = move
                best_move, best_value = iteration_best, iteration_value
            except SearchTimeout:
                break

            # A forced win does not need deeper search.
            if best_value > WIN_SCORE * 0.9:
                break

        return best_move

    def _search(self, state: State, depth: int, ply: int) -> float:
        self._check_deadline()
        terminal = self._terminal_value(state, ply)
        if terminal is not None:
            return terminal
        if depth <= 0:
            return self._evaluate(state)

        key = (state, depth)
        cached = self.tt.get(key)
        if cached is not None:
            return cached

        moves = list(self._legal_moves(state, state.you_id))
        if not moves:
            return LOSS_SCORE + ply

        moves.sort(key=lambda move: self._quick_move_score(state, state.you_id, move), reverse=True)
        best = LOSS_SCORE
        for move in moves:
            self._check_deadline()
            value = self._value_of_our_move(state, move, depth, ply)
            if value > best:
                best = value

        self.tt[key] = best
        return best

    def _value_of_our_move(self, state: State, our_move: str, depth: int, ply: int) -> float:
        opponents = [snake for snake in state.snakes if snake.id != state.you_id]
        if not opponents:
            next_state = _simulate_turn(state, {state.you_id: our_move})
            return self._search(next_state, depth - 1, ply + 1)

        distributions: List[Tuple[str, Tuple[Tuple[str, float], ...]]] = []
        for snake in opponents:
            priors = self._opponent_priors(state, snake.id)
            if not priors:
                # The engine still needs a direction; an impossible move will
                # be resolved as death in the simulator.
                priors = (("up", 1.0),)
            distributions.append((snake.id, priors))

        scenarios = []
        for choices in itertools.product(*(dist for _, dist in distributions)):
            moves = {state.you_id: our_move}
            probability = 1.0
            for (snake_id, _), (move, prob) in zip(distributions, choices):
                moves[snake_id] = move
                probability *= prob
            danger = self._scenario_danger(state, our_move, moves)
            scenarios.append((danger, probability, moves))

        # Prioritize likely and tactically dangerous responses. At the root all
        # usual 3^3 combinations fit; deeper nodes use a beam.
        scenarios.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if ply == 0 and depth <= 1:
            limit = ROOT_FULL_JOINT_LIMIT
        elif ply == 0:
            limit = ROOT_DEEP_JOINT_LIMIT
        else:
            limit = DEEP_JOINT_LIMIT
        if len(scenarios) > limit:
            scenarios = scenarios[:limit]
            total_prob = sum(prob for _, prob, _ in scenarios) or 1.0
            scenarios = [(d, p / total_prob, m) for d, p, m in scenarios]

        outcomes: List[Tuple[float, float]] = []
        for _, probability, moves in scenarios:
            self._check_deadline()
            child = _simulate_turn(state, moves)
            if depth <= 1:
                value = self._terminal_value(child, ply + 1)
                if value is None:
                    value = self._evaluate(child)
            else:
                value = self._search(child, depth - 1, ply + 1)
            outcomes.append((value, probability))

        return _robust_aggregate(outcomes)

    def _opponent_priors(self, state: State, snake_id: str) -> Tuple[Tuple[str, float], ...]:
        key = (state, snake_id)
        cached = self.prior_cache.get(key)
        if cached is not None:
            return cached

        moves = self._legal_moves(state, snake_id)
        if not moves:
            result: Tuple[Tuple[str, float], ...] = ()
            self.prior_cache[key] = result
            return result

        scored = [(move, self._quick_move_score(state, snake_id, move)) for move in moves]
        scored.sort(key=lambda item: item[1], reverse=True)

        # Keep every normal move at the root branching scale. A reverse move is
        # usually illegal, so this is typically at most three.
        max_score = scored[0][1]
        temperature = 18.0
        raw = [(move, math.exp(max(-40.0, min(0.0, (score - max_score) / temperature)))) for move, score in scored]
        total = sum(weight for _, weight in raw) or 1.0
        result = tuple((move, weight / total) for move, weight in raw)
        self.prior_cache[key] = result
        return result

    def _legal_moves(self, state: State, snake_id: str) -> Tuple[str, ...]:
        key = (state, snake_id)
        cached = self.move_cache.get(key)
        if cached is not None:
            return cached

        snake = state.snake(snake_id)
        if snake is None:
            return ()

        blocked = _certainly_blocked_cells(state)
        result: List[str] = []
        for move in MOVE_ORDER:
            dx, dy = DIRECTIONS[move]
            nxt = (snake.head[0] + dx, snake.head[1] + dy)
            if not _in_bounds(nxt, state.width, state.height):
                continue
            if nxt in blocked:
                continue
            result.append(move)

        answer = tuple(result)
        self.move_cache[key] = answer
        return answer

    def _quick_move_score(self, state: State, snake_id: str, move: str) -> float:
        snake = state.snake(snake_id)
        if snake is None:
            return LOSS_SCORE
        dx, dy = DIRECTIONS[move]
        nxt = (snake.head[0] + dx, snake.head[1] + dy)

        blocked = _certainly_blocked_cells(state)
        blocked.discard(nxt)
        area = _flood_count(nxt, blocked, state.width, state.height, cap=state.width * state.height)
        exits = _free_neighbor_count(nxt, blocked, state.width, state.height)

        score = area * 3.0 + exits * 14.0
        score += _wall_distance(nxt, state.width, state.height) * 1.5

        # Head-to-head geometry: avoid equal/longer snakes, pressure shorter ones.
        for other in state.snakes:
            if other.id == snake_id:
                continue
            if _manhattan(nxt, other.head) == 1:
                if other.length >= snake.length:
                    score -= 110.0
                else:
                    score += 35.0

        if nxt in state.food:
            urgency = max(0.0, 55.0 - snake.health)
            score += 25.0 + urgency * 1.8
        elif state.food:
            nearest = min(_manhattan(nxt, food) for food in state.food)
            if snake.health < 45:
                score -= nearest * (2.5 + (45 - snake.health) * 0.08)

        hazard_stacks = Counter(state.hazards)
        if nxt in hazard_stacks:
            damage = state.hazard_damage * hazard_stacks[nxt]
            if nxt not in state.food:
                score -= damage * 5.0

        return score

    def _root_static_score(self, move: str) -> float:
        state = _simulate_turn_with_default_opponents(self.root, move)
        terminal = self._terminal_value(state, 1)
        if terminal is not None:
            return terminal
        return self._evaluate(state)

    def _scenario_danger(self, state: State, our_move: str, moves: Mapping[str, str]) -> float:
        me = state.snake(state.you_id)
        if me is None:
            return 0.0
        dx, dy = DIRECTIONS[our_move]
        our_dest = (me.head[0] + dx, me.head[1] + dy)
        danger = 0.0
        for snake in state.snakes:
            if snake.id == state.you_id:
                continue
            move = moves.get(snake.id, "up")
            odx, ody = DIRECTIONS[move]
            dest = (snake.head[0] + odx, snake.head[1] + ody)
            if dest == our_dest:
                danger += 100.0 if snake.length >= me.length else 25.0
            elif _manhattan(dest, our_dest) == 1:
                danger += 5.0
        return danger

    def _terminal_value(self, state: State, ply: int) -> Optional[float]:
        me = state.snake(state.you_id)
        if me is None:
            return LOSS_SCORE + ply * 1000.0
        if len(state.snakes) == 1:
            return WIN_SCORE - ply * 1000.0
        return None

    def _evaluate(self, state: State) -> float:
        cached = self.eval_cache.get(state)
        if cached is not None:
            return cached

        self._check_deadline()
        me = state.snake(state.you_id)
        if me is None:
            return LOSS_SCORE

        blocked = _certainly_blocked_cells(state)
        my_area = _flood_count(me.head, blocked - {me.head}, state.width, state.height, cap=state.width * state.height)
        my_exits = _free_neighbor_count(me.head, blocked - {me.head}, state.width, state.height)

        voronoi, contested = _voronoi_control(state, state.you_id)
        tail_reachable = _tail_reachable(state, me)
        nearest_food = _nearest_food_distance(state, me.head, blocked - {me.head})

        opponents = [snake for snake in state.snakes if snake.id != state.you_id]
        largest_enemy = max((snake.length for snake in opponents), default=0)
        enemy_area_max = 0
        enemy_exits_total = 0
        for snake in opponents:
            area = _flood_count(snake.head, blocked - {snake.head}, state.width, state.height, cap=state.width * state.height)
            enemy_area_max = max(enemy_area_max, area)
            enemy_exits_total += _free_neighbor_count(snake.head, blocked - {snake.head}, state.width, state.height)

        score = 0.0
        score += my_area * 38.0
        score += voronoi * 52.0
        score += contested * 7.0
        score += my_exits * 115.0
        score += 260.0 if tail_reachable else -340.0
        score += (me.length - largest_enemy) * 42.0
        score -= enemy_area_max * 7.0
        score -= enemy_exits_total * 8.0

        # Space smaller than our body is an immediate strategic emergency.
        if my_area < me.length:
            score -= (me.length - my_area + 1) * 700.0
        elif my_area < me.length + 3:
            score -= 260.0

        # Food is valued according to shortest-path feasibility and health.
        if nearest_food is not None:
            starvation_margin = me.health - nearest_food
            if me.health <= 45:
                score += (48 - me.health) * 30.0
                score -= nearest_food * (34.0 + max(0, 32 - me.health) * 1.5)
            elif me.health >= 80 and me.length > largest_enemy + 2:
                score -= max(0, 5 - nearest_food) * 16.0
            if starvation_margin <= 3:
                score -= (4 - starvation_margin) * 450.0
        elif me.health < 55:
            score -= (55 - me.health) * 80.0

        hazard_stacks = Counter(state.hazards)
        if me.head in hazard_stacks and me.head not in state.food:
            score -= state.hazard_damage * hazard_stacks[me.head] * 38.0

        # Mild center preference only as a tie-breaker; space dominates it.
        center_x = (state.width - 1) / 2.0
        center_y = (state.height - 1) / 2.0
        score -= (abs(me.head[0] - center_x) + abs(me.head[1] - center_y)) * 2.0

        self.eval_cache[state] = score
        return score

    def _check_deadline(self) -> None:
        self.nodes += 1
        # Checking every node is cheap at this board size and avoids timeout
        # spikes on low-tier shared CPUs.
        if time.perf_counter() >= self.deadline:
            raise SearchTimeout


def _parse_state(game_state: Dict) -> State:
    board = game_state["board"]
    settings = game_state.get("game", {}).get("ruleset", {}).get("settings", {})
    snakes = []
    for raw in board.get("snakes", []):
        body = tuple((int(p["x"]), int(p["y"])) for p in raw["body"])
        if body:
            snakes.append(Snake(id=str(raw["id"]), health=int(raw["health"]), body=body))

    hazards = tuple((int(p["x"]), int(p["y"])) for p in board.get("hazards", []))
    food = frozenset((int(p["x"]), int(p["y"])) for p in board.get("food", []))
    return State(
        width=int(board["width"]),
        height=int(board["height"]),
        food=food,
        hazards=hazards,
        snakes=tuple(snakes),
        you_id=str(game_state["you"]["id"]),
        hazard_damage=int(settings.get("hazardDamagePerTurn", 14)),
    )


def _simulate_turn(state: State, moves: Mapping[str, str]) -> State:
    """Apply one simultaneous turn closely following the standard rules."""
    hazard_stacks = Counter(state.hazards)
    moved: List[Snake] = []
    eaten: set[Point] = set()

    for snake in state.snakes:
        move = moves.get(snake.id, "up")
        dx, dy = DIRECTIONS.get(move, DIRECTIONS["up"])
        new_head = (snake.head[0] + dx, snake.head[1] + dy)
        ate = new_head in state.food

        # Food restores full health and protects against hazard damage on the
        # same turn. This also handles the standard map (no hazards).
        if ate:
            health = 100
        else:
            health = snake.health - 1
            health -= state.hazard_damage * hazard_stacks.get(new_head, 0)

        moved_body = (new_head,) + snake.body[:-1]
        if ate:
            # The engine first moves (popping the old tail), then duplicates
            # the new tail cell to grow. This is not the same as retaining the
            # old tail coordinate.
            body = moved_body + (moved_body[-1],)
            eaten.add(new_head)
        else:
            body = moved_body
        moved.append(Snake(snake.id, health, body))

    # The production rules eliminate out-of-health/out-of-bounds snakes before
    # collision checks. Collision eliminations themselves are collected and
    # applied together, so colliding snakes still block one another this turn.
    pre_dead: set[str] = set()
    for snake in moved:
        if not _in_bounds(snake.head, state.width, state.height) or snake.health <= 0:
            pre_dead.add(snake.id)

    eligible = [snake for snake in moved if snake.id not in pre_dead]
    collision_dead: set[str] = set()
    body_cells_by_id = {snake.id: set(snake.body[1:]) for snake in eligible}

    for snake in eligible:
        # Self-collision has precedence in the official attribution, though for
        # search only the survivor set matters.
        if snake.head in body_cells_by_id[snake.id]:
            collision_dead.add(snake.id)
            continue

        if any(
            snake.head in body_cells_by_id[other.id]
            for other in eligible
            if other.id != snake.id
        ):
            collision_dead.add(snake.id)
            continue

        # A snake loses a shared-head collision to every equal-or-longer snake.
        if any(
            snake.head == other.head and snake.length <= other.length
            for other in eligible
            if other.id != snake.id
        ):
            collision_dead.add(snake.id)

    dead = pre_dead | collision_dead

    survivors = tuple(snake for snake in moved if snake.id not in dead)
    return State(
        width=state.width,
        height=state.height,
        food=frozenset(food for food in state.food if food not in eaten),
        hazards=state.hazards,
        snakes=survivors,
        you_id=state.you_id,
        hazard_damage=state.hazard_damage,
    )


def _simulate_turn_with_default_opponents(state: State, our_move: str) -> State:
    """Cheap deterministic rollout used only to seed root move ordering."""
    moves = {state.you_id: our_move}
    engine = _TinyMoveChooser(state)
    for snake in state.snakes:
        if snake.id != state.you_id:
            moves[snake.id] = engine.best(snake.id)
    return _simulate_turn(state, moves)


class _TinyMoveChooser:
    def __init__(self, state: State) -> None:
        self.state = state

    def best(self, snake_id: str) -> str:
        snake = self.state.snake(snake_id)
        if snake is None:
            return "up"
        blocked = _certainly_blocked_cells(self.state)
        best_move = "up"
        best_score = -INF
        for move in MOVE_ORDER:
            dx, dy = DIRECTIONS[move]
            nxt = (snake.head[0] + dx, snake.head[1] + dy)
            if not _in_bounds(nxt, self.state.width, self.state.height) or nxt in blocked:
                continue
            area = _flood_count(nxt, blocked - {nxt}, self.state.width, self.state.height, cap=40)
            score = area * 4 + _free_neighbor_count(nxt, blocked - {nxt}, self.state.width, self.state.height) * 10
            if nxt in self.state.food and snake.health < 60:
                score += 40
            if score > best_score:
                best_score = score
                best_move = move
        return best_move


def _robust_aggregate(outcomes: Sequence[Tuple[float, float]]) -> float:
    """Blend expected value with lower-tail risk.

    Pure maximin is too pessimistic in four-player games because it assumes all
    opponents coordinate and willingly sacrifice themselves. Pure expectation
    is too optimistic around forced head-to-heads. This CVaR-like blend retains
    tactical caution without freezing the snake into passive play.
    """
    if not outcomes:
        return LOSS_SCORE

    total_weight = sum(max(0.0, weight) for _, weight in outcomes) or 1.0
    normalized = [(value, max(0.0, weight) / total_weight) for value, weight in outcomes]
    expected = sum(value * weight for value, weight in normalized)

    ordered = sorted(normalized, key=lambda item: item[0])
    tail_mass = 0.25
    consumed = 0.0
    tail_sum = 0.0
    for value, weight in ordered:
        take = min(weight, tail_mass - consumed)
        if take > 0:
            tail_sum += value * take
            consumed += take
        if consumed >= tail_mass - 1e-12:
            break
    cvar = tail_sum / consumed if consumed > 0 else ordered[0][0]
    worst = ordered[0][0]

    return expected * 0.42 + cvar * 0.46 + worst * 0.12


def _certainly_blocked_cells(state: State) -> set[Point]:
    """Cells guaranteed to remain occupied after the next simultaneous move.

    A non-stacked tail may vacate and is therefore left traversable during move
    generation. Exact turn simulation later rejects the move if that tail eats
    and does not vacate.
    """
    blocked: set[Point] = set()
    for snake in state.snakes:
        tail_vacates = len(snake.body) < 2 or snake.body[-1] != snake.body[-2]
        body = snake.body[:-1] if tail_vacates else snake.body
        blocked.update(body)
    return blocked


def _voronoi_control(state: State, you_id: str) -> Tuple[int, int]:
    """Return cells controlled by us and contested cells.

    Distances are computed around bodies with potentially moving tails opened.
    Equal arrival times are awarded to a uniquely longer head; otherwise they
    are contested.
    """
    blocked = _certainly_blocked_cells(state)
    snakes = list(state.snakes)
    distances: Dict[str, Dict[Point, int]] = {}
    for snake in snakes:
        distances[snake.id] = _bfs_distances(snake.head, blocked - {snake.head}, state.width, state.height)

    controlled = 0
    contested = 0
    snake_by_id = {snake.id: snake for snake in snakes}
    for x in range(state.width):
        for y in range(state.height):
            cell = (x, y)
            arrivals = [(dist.get(cell, INF), snake_id) for snake_id, dist in distances.items()]
            best_dist = min(distance for distance, _ in arrivals)
            if best_dist >= INF:
                continue
            leaders = [snake_id for distance, snake_id in arrivals if distance == best_dist]
            if len(leaders) == 1:
                if leaders[0] == you_id:
                    controlled += 1
                continue

            max_len = max(snake_by_id[snake_id].length for snake_id in leaders)
            longest = [snake_id for snake_id in leaders if snake_by_id[snake_id].length == max_len]
            if len(longest) == 1:
                if longest[0] == you_id:
                    controlled += 1
            else:
                contested += 1
    return controlled, contested


def _tail_reachable(state: State, snake: Snake) -> bool:
    blocked = _certainly_blocked_cells(state)
    blocked.discard(snake.head)
    blocked.discard(snake.tail)
    return _path_distance(snake.head, snake.tail, blocked, state.width, state.height) is not None


def _nearest_food_distance(state: State, start: Point, blocked: set[Point]) -> Optional[int]:
    if not state.food:
        return None
    distances = _bfs_distances(start, blocked, state.width, state.height)
    values = [distances[food] for food in state.food if food in distances]
    return min(values) if values else None


def _bfs_distances(start: Point, blocked: set[Point], width: int, height: int) -> Dict[Point, int]:
    if not _in_bounds(start, width, height):
        return {}
    dist = {start: 0}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        next_distance = dist[cell] + 1
        for neighbor in _neighbors(cell):
            if neighbor in dist or neighbor in blocked or not _in_bounds(neighbor, width, height):
                continue
            dist[neighbor] = next_distance
            queue.append(neighbor)
    return dist


def _path_distance(start: Point, goal: Point, blocked: set[Point], width: int, height: int) -> Optional[int]:
    if start == goal:
        return 0
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        cell, distance = queue.popleft()
        for neighbor in _neighbors(cell):
            if neighbor == goal:
                return distance + 1
            if neighbor in seen or neighbor in blocked or not _in_bounds(neighbor, width, height):
                continue
            seen.add(neighbor)
            queue.append((neighbor, distance + 1))
    return None


def _flood_count(start: Point, blocked: set[Point], width: int, height: int, cap: int) -> int:
    if not _in_bounds(start, width, height) or start in blocked:
        return 0
    stack = [start]
    seen = {start}
    count = 0
    while stack and count < cap:
        cell = stack.pop()
        count += 1
        for neighbor in _neighbors(cell):
            if neighbor in seen or neighbor in blocked or not _in_bounds(neighbor, width, height):
                continue
            seen.add(neighbor)
            stack.append(neighbor)
    return count


def _free_neighbor_count(cell: Point, blocked: set[Point], width: int, height: int) -> int:
    return sum(1 for p in _neighbors(cell) if _in_bounds(p, width, height) and p not in blocked)


def _neighbors(cell: Point) -> Iterable[Point]:
    x, y = cell
    for dx, dy in DIRECTIONS.values():
        yield x + dx, y + dy


def _wall_distance(cell: Point, width: int, height: int) -> int:
    x, y = cell
    return min(x, width - 1 - x, y, height - 1 - y)


def _in_bounds(point: Point, width: int, height: int) -> bool:
    return 0 <= point[0] < width and 0 <= point[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _fallback_move_from_state(state: State) -> str:
    me = state.snake(state.you_id)
    if me is None:
        return "up"
    blocked = _certainly_blocked_cells(state)
    best_move = "up"
    best_score = -INF
    for move in MOVE_ORDER:
        dx, dy = DIRECTIONS[move]
        nxt = (me.head[0] + dx, me.head[1] + dy)
        if not _in_bounds(nxt, state.width, state.height) or nxt in blocked:
            continue
        score = _flood_count(nxt, blocked - {nxt}, state.width, state.height, cap=state.width * state.height)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


def _fallback_move_from_api(game_state: Dict) -> str:
    state = _parse_state(game_state)
    return _fallback_move_from_state(state)


__all__ = ["get_info", "choose_move"]
