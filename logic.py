"""Model-backed move-selection logic for the Battlesnake.

The served policy uses a linear ranking model, scores each legal move,
and returns the highest-scoring direction. A compact heuristic remains as a
fallback so gameplay still returns a legal move if model scoring fails.

Board coordinates: ``(0, 0)`` is the bottom-left corner.
  up    -> y + 1
  down  -> y - 1
  left  -> x - 1
  right -> x + 1

Game-state schema reference: https://docs.battlesnake.com/api
"""

from collections import deque
from itertools import product
import time
from typing import Dict, List, Optional, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

# Penalty applied to a move that could lose a head-to-head collision.
HEAD_TO_HEAD_PENALTY = 10_000
# Below this health we start actively steering toward food.
HUNGRY_THRESHOLD = 50
SEARCH_TIME_SECONDS = 0.35
MINIMAX_DEPTH = 2
LOSE_SCORE = -1_000_000.0
WIN_SCORE = 1_000_000.0


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from ``GET /``."""
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#1d4ed8",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "0.3.2",
    }


def choose_move(game_state: Dict) -> str:
    """Return the next move using minimax, with model and heuristic fallbacks."""
    try:
        move = choose_move_minimax(game_state)
    except Exception:  # noqa: BLE001 - gameplay must never fail the request
        move = None
    if move is not None:
        return move

    try:
        move = choose_move_model(game_state)
    except Exception:  # noqa: BLE001 - a model issue must never break gameplay
        move = None
    if move is not None:
        return move
    return choose_move_heuristic(game_state)


def choose_move_heuristic(game_state: Dict) -> str:
    """Return the next move for the current turn."""
    board = game_state["board"]
    you = game_state["you"]
    width: int = board["width"]
    height: int = board["height"]

    head: Point = (you["head"]["x"], you["head"]["y"])
    my_length: int = you["length"]
    health: int = you["health"]

    occupied = _occupied_cells(board["snakes"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]

    best_move = None
    best_score = float("-inf")

    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)

        if not _in_bounds(nxt, width, height):
            continue
        if nxt in occupied:
            continue

        # Reachable open space from this cell. If we can't fit our own body in
        # the space we'd be moving into, we're about to trap ourselves.
        space = _flood_fill(nxt, occupied, width, height, limit=my_length + 1)
        score = float(space)

        if nxt in danger:
            score -= HEAD_TO_HEAD_PENALTY

        # When hungry, nudge toward the closest food.
        if foods and health < HUNGRY_THRESHOLD:
            nearest = min(_manhattan(nxt, f) for f in foods)
            score += (width + height - nearest) * 2

        if score > best_score:
            best_score = score
            best_move = move

    # No safe move found -> we're cornered. Move up and hope for the best.
    return best_move or "up"


def _occupied_cells(snakes: List[Dict]) -> Set[Point]:
    """All cells currently filled by any snake's body.

    We keep tails occupied too; they only free up *next* turn and treating them
    as solid is the conservative, safe choice for a base bot.
    """
    occupied: Set[Point] = set()
    for snake in snakes:
        for seg in snake["body"]:
            occupied.add((seg["x"], seg["y"]))
    return occupied


def _head_to_head_cells(snakes: List[Dict], my_id: str, my_length: int) -> Set[Point]:
    """Cells adjacent to enemy heads that are >= our length.

    Moving onto one of these risks a head-to-head collision we would lose or
    tie, so they are heavily penalized (but not forbidden — sometimes it's the
    only move).
    """
    danger: Set[Point] = set()
    for snake in snakes:
        if snake["id"] == my_id:
            continue
        if snake["length"] < my_length:
            continue
        ehead = (snake["head"]["x"], snake["head"]["y"])
        for dx, dy in DIRECTIONS.values():
            danger.add((ehead[0] + dx, ehead[1] + dy))
    return danger


def _flood_fill(start: Point, occupied: Set[Point], width: int, height: int, limit: int) -> int:
    """Count open cells reachable from ``start`` (capped at ``limit``).

    Used to avoid moves that would seal us into a small pocket.
    """
    seen: Set[Point] = {start}
    stack: List[Point] = [start]
    count = 0
    while stack:
        x, y = stack.pop()
        count += 1
        if count >= limit:
            break
        for dx, dy in DIRECTIONS.values():
            nbr = (x + dx, y + dy)
            if nbr in seen:
                continue
            if not _in_bounds(nbr, width, height):
                continue
            if nbr in occupied:
                continue
            seen.add(nbr)
            stack.append(nbr)
    return count


def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# --- Paranoid minimax over a Voronoi heuristic ------------------------------


def choose_move_minimax(game_state: Dict) -> Optional[str]:
    """Pick a move with shallow paranoid minimax.

    Args:
        game_state: Battlesnake request payload.

    Returns:
        Best move, or None if no move was found before timeout.
    """
    start_time = time.perf_counter()
    cutoff = start_time + SEARCH_TIME_SECONDS
    legal_moves = _legal_moves_tail_aware(game_state, game_state["you"]["id"])
    if not legal_moves:
        return None

    depth = MINIMAX_DEPTH
    if len(game_state["board"]["snakes"]) >= 4:
        depth = 1

    best_move = None
    best_score = float("-inf")
    ordered_moves = sorted(
        legal_moves,
        key=lambda move: _static_move_score(game_state, move),
        reverse=True,
    )

    for move in ordered_moves:
        if time.perf_counter() > cutoff and best_move is not None:
            break
        try:
            score = _worst_enemy_reply(game_state, move, depth, cutoff)
        except TimeoutError:
            break
        if score > best_score:
            best_score = score
            best_move = move

    return best_move


def _worst_enemy_reply(game_state: Dict, my_move: str, depth: int, cutoff: float) -> float:
    """Score our move under coordinated worst-case enemy replies.

    Args:
        game_state: Current game state.
        my_move: Candidate move for our snake.
        depth: Remaining full-turn search depth.
        cutoff: Monotonic time cutoff.

    Returns:
        Worst score reachable after enemy moves.
    """
    if time.perf_counter() > cutoff:
        raise TimeoutError

    you_id = game_state["you"]["id"]
    enemies = [snake for snake in game_state["board"]["snakes"] if snake["id"] != you_id]
    if not enemies:
        next_state = _simulate_turn(game_state, {you_id: my_move})
        return _minimax_value(next_state, depth - 1, cutoff)

    enemy_move_lists = []
    for enemy in enemies:
        moves = _legal_moves_tail_aware(game_state, enemy["id"])
        if not moves:
            moves = list(DIRECTIONS)
        enemy_move_lists.append(moves)

    worst_score = float("inf")
    for enemy_moves in product(*enemy_move_lists):
        moves_by_id = {you_id: my_move}
        for enemy, move in zip(enemies, enemy_moves):
            moves_by_id[enemy["id"]] = move
        next_state = _simulate_turn(game_state, moves_by_id)
        score = _minimax_value(next_state, depth - 1, cutoff)
        if score < worst_score:
            worst_score = score
    return worst_score


def _minimax_value(game_state: Dict, depth: int, cutoff: float) -> float:
    """Return max value for us from this state.

    Args:
        game_state: Simulated Battlesnake state.
        depth: Remaining full-turn search depth.
        cutoff: Monotonic time cutoff.

    Returns:
        Evaluation score from our perspective.
    """
    if time.perf_counter() > cutoff:
        raise TimeoutError

    you_id = game_state["you"]["id"]
    if not _snake_by_id(game_state, you_id):
        return LOSE_SCORE
    if len(game_state["board"]["snakes"]) == 1:
        return WIN_SCORE + game_state["you"]["length"]
    if depth <= 0:
        return _evaluate_state(game_state)

    legal_moves = _legal_moves_tail_aware(game_state, you_id)
    if not legal_moves:
        return LOSE_SCORE

    best_score = float("-inf")
    for move in sorted(legal_moves, key=lambda m: _static_move_score(game_state, m), reverse=True):
        score = _worst_enemy_reply(game_state, move, depth, cutoff)
        if score > best_score:
            best_score = score
    return best_score


def _evaluate_state(game_state: Dict) -> float:
    """Evaluate board position from our perspective.

    Args:
        game_state: Simulated Battlesnake state.

    Returns:
        Higher score for safer, roomier, stronger positions.
    """
    board = game_state["board"]
    you = _snake_by_id(game_state, game_state["you"]["id"])
    if you is None:
        return LOSE_SCORE

    enemies = [snake for snake in board["snakes"] if snake["id"] != you["id"]]
    if not enemies:
        return WIN_SCORE + you["length"]

    owned, lost = _voronoi_counts(game_state)
    head = (you["head"]["x"], you["head"]["y"])
    foods = [(food["x"], food["y"]) for food in board["food"]]
    enemy_lengths = sum(snake["length"] for snake in enemies)
    food_bonus = 0.0
    if foods:
        nearest_food = min(_manhattan(head, food) for food in foods)
        if you["health"] < HUNGRY_THRESHOLD:
            food_bonus += (board["width"] + board["height"] - nearest_food) * 8
        food_bonus += sum(12 for food in foods if _owned_food(game_state, food))

    danger = 0.0
    for enemy in enemies:
        enemy_head = (enemy["head"]["x"], enemy["head"]["y"])
        if enemy["length"] >= you["length"] and _manhattan(head, enemy_head) <= 2:
            danger += 80
        if enemy["length"] < you["length"] and _manhattan(head, enemy_head) <= 2:
            danger -= 25

    tail_bonus = 30 if _can_reach_tail(game_state, you["id"]) else -40

    return (
        owned * 12
        - lost * 4
        + you["length"] * 15
        - enemy_lengths * 6
        + you["health"] * 0.4
        + food_bonus
        + tail_bonus
        - danger
    )


def _voronoi_counts(game_state: Dict) -> Tuple[int, int]:
    """Count cells we reach before enemies and cells enemies reach first.

    Args:
        game_state: Battlesnake state.

    Returns:
        Tuple of owned and lost cell counts.
    """
    board = game_state["board"]
    you = _snake_by_id(game_state, game_state["you"]["id"])
    if you is None:
        return 0, board["width"] * board["height"]

    blocked = _body_cells_without_heads(board["snakes"])
    my_head = (you["head"]["x"], you["head"]["y"])
    enemy_heads = [
        (snake["head"]["x"], snake["head"]["y"])
        for snake in board["snakes"]
        if snake["id"] != you["id"]
    ]
    my_dist = _bfs_dist([my_head], blocked, board["width"], board["height"])
    enemy_dist = _bfs_dist(enemy_heads, blocked, board["width"], board["height"]) if enemy_heads else {}

    owned = 0
    lost = 0
    for x in range(board["width"]):
        for y in range(board["height"]):
            point = (x, y)
            if point in blocked:
                continue
            md = my_dist.get(point, _BIG)
            ed = enemy_dist.get(point, _BIG)
            if md < ed:
                owned += 1
            elif ed < md:
                lost += 1
    return owned, lost


def _owned_food(game_state: Dict, food: Point) -> bool:
    """Return whether we reach a food before all enemies.

    Args:
        game_state: Battlesnake state.
        food: Food coordinate.

    Returns:
        True when our distance to food is strictly lowest.
    """
    board = game_state["board"]
    you = _snake_by_id(game_state, game_state["you"]["id"])
    if you is None:
        return False

    blocked = _body_cells_without_heads(board["snakes"])
    my_head = (you["head"]["x"], you["head"]["y"])
    my_dist = _bfs_dist([my_head], blocked, board["width"], board["height"]).get(food, _BIG)
    for enemy in board["snakes"]:
        if enemy["id"] == you["id"]:
            continue
        enemy_head = (enemy["head"]["x"], enemy["head"]["y"])
        enemy_dist = _bfs_dist([enemy_head], blocked, board["width"], board["height"]).get(food, _BIG)
        if enemy_dist <= my_dist:
            return False
    return my_dist < _BIG


def _can_reach_tail(game_state: Dict, snake_id: str) -> bool:
    """Check whether snake head can reach its own tail.

    Args:
        game_state: Battlesnake state.
        snake_id: Snake id to inspect.

    Returns:
        True when the tail is reachable through currently open cells.
    """
    board = game_state["board"]
    snake = _snake_by_id(game_state, snake_id)
    if snake is None or not snake["body"]:
        return False

    head = (snake["head"]["x"], snake["head"]["y"])
    tail = (snake["body"][-1]["x"], snake["body"][-1]["y"])
    blocked = _occupied_cells(board["snakes"]) - {tail}
    return tail in _bfs_dist([head], blocked, board["width"], board["height"])


def _static_move_score(game_state: Dict, move: str) -> float:
    """Cheap move score used only for move ordering.

    Args:
        game_state: Current state.
        move: Candidate move.

    Returns:
        Heuristic score for ordering search.
    """
    board = game_state["board"]
    you = game_state["you"]
    head = (you["head"]["x"], you["head"]["y"])
    dx, dy = DIRECTIONS[move]
    nxt = (head[0] + dx, head[1] + dy)
    occupied = _occupied_cells(board["snakes"]) - _moving_tail_cells(board["snakes"])
    if not _in_bounds(nxt, board["width"], board["height"]) or nxt in occupied:
        return LOSE_SCORE
    space = _flood_fill(nxt, occupied, board["width"], board["height"], limit=board["width"] * board["height"])
    foods = [(food["x"], food["y"]) for food in board["food"]]
    food_score = 0
    if foods and you["health"] < HUNGRY_THRESHOLD:
        food_score = board["width"] + board["height"] - min(_manhattan(nxt, food) for food in foods)
    return float(space + food_score * 3)


def _legal_moves_tail_aware(game_state: Dict, snake_id: str) -> List[str]:
    """Return moves not immediately blocked by walls or bodies.

    Args:
        game_state: Battlesnake state.
        snake_id: Snake id.

    Returns:
        List of legal move strings.
    """
    board = game_state["board"]
    snake = _snake_by_id(game_state, snake_id)
    if snake is None:
        return []

    head = (snake["head"]["x"], snake["head"]["y"])
    blocked = _occupied_cells(board["snakes"]) - _moving_tail_cells(board["snakes"])
    moves = []
    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)
        if _in_bounds(nxt, board["width"], board["height"]) and nxt not in blocked:
            moves.append(move)
    return moves


def _moving_tail_cells(snakes: List[Dict]) -> Set[Point]:
    """Return tail cells that probably vacate this turn.

    Args:
        snakes: Snakes from board state.

    Returns:
        Set of tail coordinates safe to treat as open.
    """
    tails = set()
    for snake in snakes:
        body = snake["body"]
        if len(body) < 2:
            continue
        tail = (body[-1]["x"], body[-1]["y"])
        before_tail = (body[-2]["x"], body[-2]["y"])
        if tail != before_tail:
            tails.add(tail)
    return tails


def _body_cells_without_heads(snakes: List[Dict]) -> Set[Point]:
    """Return occupied body cells excluding heads.

    Args:
        snakes: Snakes from board state.

    Returns:
        Set of body coordinates excluding head cells.
    """
    cells = set()
    for snake in snakes:
        for segment in snake["body"][1:]:
            cells.add((segment["x"], segment["y"]))
    return cells


def _snake_by_id(game_state: Dict, snake_id: str) -> Optional[Dict]:
    """Find a live snake by id.

    Args:
        game_state: Battlesnake state.
        snake_id: Snake id.

    Returns:
        Snake dict or None.
    """
    for snake in game_state["board"]["snakes"]:
        if snake["id"] == snake_id:
            return snake
    return None


def _simulate_turn(game_state: Dict, moves_by_id: Dict[str, str]) -> Dict:
    """Simulate one simultaneous Battlesnake turn.

    Args:
        game_state: Current state.
        moves_by_id: Mapping from snake id to move.

    Returns:
        New simulated game state.
    """
    board = game_state["board"]
    width = board["width"]
    height = board["height"]
    food = {(item["x"], item["y"]) for item in board["food"]}
    moved = []

    for snake in board["snakes"]:
        move = moves_by_id.get(snake["id"])
        if move not in DIRECTIONS:
            continue
        dx, dy = DIRECTIONS[move]
        old_head = (snake["head"]["x"], snake["head"]["y"])
        new_head = (old_head[0] + dx, old_head[1] + dy)
        eating = new_head in food
        body_points = [(new_head[0], new_head[1])] + [(p["x"], p["y"]) for p in snake["body"]]
        if not eating:
            body_points = body_points[:-1]
        moved.append(
            {
                "id": snake["id"],
                "name": snake.get("name", snake["id"]),
                "health": 100 if eating else snake["health"] - 1,
                "body": [{"x": x, "y": y} for x, y in body_points],
                "head": {"x": new_head[0], "y": new_head[1]},
                "length": len(body_points),
                "_new_head": new_head,
                "_eating": eating,
                "_dead": not _in_bounds(new_head, width, height),
            }
        )

    body_cells = set()
    for snake in moved:
        for segment in snake["body"][1:]:
            body_cells.add((segment["x"], segment["y"]))

    for snake in moved:
        if snake["_new_head"] in body_cells or snake["health"] <= 0:
            snake["_dead"] = True

    heads: Dict[Point, List[Dict]] = {}
    for snake in moved:
        if not snake["_dead"]:
            heads.setdefault(snake["_new_head"], []).append(snake)
    for snakes_at_head in heads.values():
        if len(snakes_at_head) < 2:
            continue
        max_length = max(snake["length"] for snake in snakes_at_head)
        winners = [snake for snake in snakes_at_head if snake["length"] == max_length]
        for snake in snakes_at_head:
            if len(winners) > 1 or snake["length"] < max_length:
                snake["_dead"] = True

    live_snakes = []
    eaten_food = set()
    for snake in moved:
        if snake["_dead"]:
            continue
        if snake["_eating"]:
            eaten_food.add(snake["_new_head"])
        live_snakes.append(
            {
                "id": snake["id"],
                "name": snake["name"],
                "health": snake["health"],
                "body": snake["body"],
                "head": snake["head"],
                "length": snake["length"],
            }
        )

    next_food = [{"x": x, "y": y} for x, y in sorted(food - eaten_food)]
    you = next((snake for snake in live_snakes if snake["id"] == game_state["you"]["id"]), game_state["you"])
    return {
        "game": game_state.get("game", {}),
        "turn": game_state.get("turn", 0) + 1,
        "board": {
            "height": height,
            "width": width,
            "food": next_food,
            "hazards": board.get("hazards", []),
            "snakes": live_snakes,
        },
        "you": you,
    }


# --- Embedded model features -------------------------------------------------

_BIG = 10_000
_NEIGHBORS = ((0, 1), (0, -1), (-1, 0), (1, 0))


def _bfs_dist(sources, blocked, width, height):
    """Shortest free-cell distances from seed cells."""
    dist = {}
    dq = deque()
    for source in sources:
        if source not in dist:
            dist[source] = 0
            dq.append(source)
    while dq:
        x, y = dq.popleft()
        d = dist[(x, y)]
        for dx, dy in _NEIGHBORS:
            nb = (x + dx, y + dy)
            if 0 <= nb[0] < width and 0 <= nb[1] < height and nb not in blocked and nb not in dist:
                dist[nb] = d + 1
                dq.append(nb)
    return dist


def _candidate_features(state: Dict, move: str) -> Dict[str, float]:
    """Feature vector for playing ``move`` from ``state``. Assumes ``move`` is legal."""
    board = state["board"]
    you = state["you"]
    width, height = board["width"], board["height"]
    head = (you["head"]["x"], you["head"]["y"])
    my_length = you["length"]
    health = you["health"]

    dx, dy = DIRECTIONS[move]
    nxt = (head[0] + dx, head[1] + dy)

    occupied = _occupied_cells(board["snakes"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]
    enemies = [s for s in board["snakes"] if s["id"] != you["id"]]
    enemy_heads = [(s["head"]["x"], s["head"]["y"]) for s in enemies]
    bigger_heads = [(s["head"]["x"], s["head"]["y"]) for s in enemies if s["length"] >= my_length]

    # Voronoi control: cells we reach strictly before any enemy.
    my_dist = _bfs_dist([nxt], occupied, width, height)
    enemy_dist = _bfs_dist(enemy_heads, occupied, width, height) if enemy_heads else {}
    voronoi = sum(1 for cell, md in my_dist.items() if md < enemy_dist.get(cell, _BIG))

    # Tail reachability is a useful anti-self-trap signal.
    my_tail = (you["body"][-1]["x"], you["body"][-1]["y"])
    reach = _bfs_dist([nxt], occupied - {my_tail}, width, height)
    reaches_tail = 1.0 if my_tail in reach else 0.0

    escape = sum(
        1
        for ddx, ddy in _NEIGHBORS
        if _in_bounds((nxt[0] + ddx, nxt[1] + ddy), width, height)
        and (nxt[0] + ddx, nxt[1] + ddy) not in occupied
    )

    nearest_now = min((_manhattan(head, f) for f in foods), default=_BIG)
    nearest_next = min((_manhattan(nxt, f) for f in foods), default=_BIG)
    hungry = health < HUNGRY_THRESHOLD

    return {
        "space_capped": float(_flood_fill(nxt, occupied, width, height, limit=my_length + 1)),
        "open_space": float(_flood_fill(nxt, occupied, width, height, limit=width * height)),
        "voronoi": float(voronoi),
        "reaches_tail": reaches_tail,
        "escape": float(escape),
        "h2h_danger": 1.0 if nxt in danger else 0.0,
        "near_bigger_head": float(min((_manhattan(nxt, h) for h in bigger_heads), default=width + height)),
        "near_enemy_head": float(min((_manhattan(nxt, h) for h in enemy_heads), default=width + height)),
        "wall_dist": float(min(nxt[0], width - 1 - nxt[0], nxt[1], height - 1 - nxt[1])),
        "food_score": float((width + height - nearest_next) * 2) if hungry and foods else 0.0,
        "food_delta": float(nearest_now - nearest_next) if foods else 0.0,
        "is_food": 1.0 if nxt in foods else 0.0,
        "dist_to_center": abs(nxt[0] - (width - 1) / 2) + abs(nxt[1] - (height - 1) / 2),
    }


# --- Model -----------------------------------------------------
# Embedded standardized linear model.

_MODEL: Dict = {
    "feature_names": [
        "space_capped",
        "open_space",
        "voronoi",
        "reaches_tail",
        "escape",
        "h2h_danger",
        "near_bigger_head",
        "near_enemy_head",
        "wall_dist",
        "food_score",
        "food_delta",
        "is_food",
        "dist_to_center",
    ],
    "mean": [
        7.357954545454546,
        100.9034090909091,
        48.26988636363637,
        0.9943181818181818,
        2.4431818181818183,
        0.04261363636363636,
        9.673295454545455,
        4.676136363636363,
        1.625,
        0.8920454545454546,
        0.14772727272727273,
        0.036931818181818184,
        5.056818181818182,
    ],
    "std": [
        3.5995966185276513,
        22.80542174802676,
        31.41119158524981,
        0.07516338951888041,
        0.6235520417417705,
        0.20198444088469822,
        7.9675173248507924,
        2.2532045017839604,
        1.3552297691803878,
        5.861056404757769,
        0.9449599886584031,
        0.18859442989548575,
        2.34451950177747,
    ],
    "coef": [
        0.00010539398521136327,
        -1.6778512168946185,
        80.89420182766183,
        9.793855564450467,
        0.7884630868036275,
        -11.025170822665032,
        -0.7981723553489,
        0.5410534990053248,
        1.5629078731518526,
        7.582325762611304,
        0.12463070008097832,
        0.21036618806863483,
        1.836259515524985,
    ],
    "intercept": 0.0,
    "top1_accuracy": 0.9928571428571429,
}


def choose_move_model(game_state: Dict) -> Optional[str]:
    """Score each legal move with the trained model; return the best.

    Returns ``None`` (so the caller falls back to the heuristic) if the model
    isn't available or the snake is trapped with no legal move.
    """
    legal = _legal_moves(game_state)
    if not legal:
        return None

    names = _MODEL["feature_names"]
    mean = _MODEL["mean"]
    std = _MODEL["std"]
    coef = _MODEL["coef"]
    intercept = _MODEL["intercept"]

    best_move, best_score = None, float("-inf")
    for move in legal:
        feats = _candidate_features(game_state, move)
        score = intercept
        for i, name in enumerate(names):
            z = (feats.get(name, 0.0) - mean[i]) / std[i] if std[i] else 0.0
            score += coef[i] * z
        if score > best_score:
            best_score, best_move = score, move
    return best_move


def _legal_moves(game_state: Dict) -> List[str]:
    board = game_state["board"]
    width, height = board["width"], board["height"]
    head = (game_state["you"]["head"]["x"], game_state["you"]["head"]["y"])
    occupied = _occupied_cells(board["snakes"])
    return [
        move
        for move, (dx, dy) in DIRECTIONS.items()
        if _in_bounds((head[0] + dx, head[1] + dy), width, height)
        and (head[0] + dx, head[1] + dy) not in occupied
    ]
