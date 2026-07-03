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
# Keep one-turn adversarial search bounded for games with many snakes.
MAX_ENEMY_SCENARIOS = 64
ENEMY_BRANCHING = 2
# Fallback heuristic search depth, measured in our future moves.
HEURISTIC_MINIMAX_DEPTH = 2


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from ``GET /``."""
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "0.1.0",
    }


def choose_move(game_state: Dict) -> str:
    """Return the next move using the model, with a heuristic fallback."""
    try:
        move = choose_move_model(game_state)
    except Exception:  # noqa: BLE001 - a model issue must never break gameplay
        move = None
    if move is not None:
        return move
    return choose_move_heuristic(game_state)


def choose_move_heuristic(game_state: Dict) -> str:
    """Return the next move with a paranoid minimax-style heuristic.

    Battlesnake turns are simultaneous, so the heuristic uses the "paranoid"
    adaptation from minimax: after each candidate move, assume the other snakes
    can choose from their strongest replies and score our worst likely outcome.
    Terminal boards are evaluated by space control, Voronoi territory, tail
    reachability, health, food pressure, and head-to-head risk.
    """
    candidates = _safe_moves(game_state)
    if not candidates:
        return "up"

    best_move = candidates[0]
    best_score = float("-inf")

    for move in candidates:
        score = _paranoid_minimax_after_move(
            game_state,
            move,
            depth=HEURISTIC_MINIMAX_DEPTH,
            alpha=best_score,
            beta=float("inf"),
        )

        if score > best_score:
            best_score = score
            best_move = move

    return best_move


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
    """Score safe moves with the model plus one-turn enemy simulations.

    Returns ``None`` (so the caller falls back to the heuristic) if the model
    isn't available or the snake is trapped with no legal move.
    """
    candidates = _safe_moves(game_state)
    if not candidates:
        return None

    best_move, best_score = None, float("-inf")
    for move in candidates:
        score = _linear_model_score(game_state, move)
        score += 0.18 * _adversarial_score(game_state, move)
        score += _immediate_tactical_score(game_state, move)
        if score > best_score:
            best_score, best_move = score, move
    return best_move


def _linear_model_score(game_state: Dict, move: str) -> float:
    names = _MODEL["feature_names"]
    mean = _MODEL["mean"]
    std = _MODEL["std"]
    coef = _MODEL["coef"]
    score = _MODEL["intercept"]
    feats = _candidate_features(game_state, move)
    for i, name in enumerate(names):
        z = (feats.get(name, 0.0) - mean[i]) / std[i] if std[i] else 0.0
        score += coef[i] * z
    return score


def _safe_moves(game_state: Dict) -> List[str]:
    """Prefer moves that survive common multi-snake traps.

    The filter is staged so a cornered snake still returns *some* legal move
    instead of becoming overly strict and falling back to a hardcoded direction.
    """
    legal = _legal_moves(game_state)
    if len(legal) <= 1:
        return legal

    board = game_state["board"]
    you = game_state["you"]
    width, height = board["width"], board["height"]
    my_length = you["length"]
    head = _snake_head(you)
    occupied = _occupied_cells(board["snakes"])
    if you["body"]:
        occupied.discard(_snake_tail(you))
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)

    scored = []
    for move in legal:
        nxt = _step(head, move)
        space = _flood_fill(nxt, occupied, width, height, limit=width * height)
        escape = _escape_count(nxt, occupied, width, height)
        in_h2h_danger = nxt in danger
        score = space + escape * 3
        if space < my_length:
            score -= (my_length - space) * 4
        if in_h2h_danger:
            score -= HEAD_TO_HEAD_PENALTY
        scored.append((move, score, space, escape, in_h2h_danger))

    no_head_to_head = [move for move, _, _, _, danger_hit in scored if not danger_hit]
    if no_head_to_head:
        scored = [item for item in scored if item[0] in no_head_to_head]

    enough_space = [move for move, _, space, _, _ in scored if space >= max(3, min(my_length, 10))]
    if enough_space:
        scored = [item for item in scored if item[0] in enough_space]

    with_escape = [move for move, _, _, escape, _ in scored if escape > 0]
    if with_escape:
        scored = [item for item in scored if item[0] in with_escape]

    scored.sort(key=lambda item: item[1], reverse=True)
    return [move for move, *_ in scored]


def _immediate_tactical_score(game_state: Dict, move: str) -> float:
    board = game_state["board"]
    you = game_state["you"]
    width, height = board["width"], board["height"]
    head = _snake_head(you)
    nxt = _step(head, move)
    foods = {(f["x"], f["y"]) for f in board["food"]}
    enemies = [s for s in board["snakes"] if s["id"] != you["id"]]
    occupied = _occupied_cells(board["snakes"])
    if you["body"]:
        occupied.discard(_snake_tail(you))

    score = 0.0
    score += _escape_count(nxt, occupied, width, height) * 8.0
    score += min(nxt[0], width - 1 - nxt[0], nxt[1], height - 1 - nxt[1]) * 3.0

    if foods:
        nearest_food = min(_manhattan(nxt, food) for food in foods)
        if you["health"] < HUNGRY_THRESHOLD:
            score += (width + height - nearest_food) * 4.0
        elif nxt in foods and you["length"] <= max((e["length"] for e in enemies), default=0) + 1:
            score += 12.0

    for enemy in enemies:
        dist = _manhattan(nxt, _snake_head(enemy))
        if enemy["length"] >= you["length"] and dist <= 2:
            score -= (3 - dist) * 35.0
        elif you["length"] > enemy["length"] and dist <= 2:
            score += (3 - dist) * 18.0
    return score


def _adversarial_score(game_state: Dict, move: str) -> float:
    scenarios = _enemy_move_scenarios(game_state)
    if not scenarios:
        simulated = _simulate_turn(game_state, move, {})
        return _scenario_score(game_state, simulated)

    scores = []
    for enemy_moves in scenarios:
        simulated = _simulate_turn(game_state, move, enemy_moves)
        scores.append(_scenario_score(game_state, simulated))

    worst = min(scores)
    avg = sum(scores) / len(scores)
    return worst * 0.70 + avg * 0.30


def _paranoid_minimax_after_move(
    game_state: Dict,
    move: str,
    depth: int,
    alpha: float,
    beta: float,
) -> float:
    """Evaluate ``move`` by letting enemy scenarios minimize our future score."""
    scenarios = _enemy_move_scenarios(game_state)
    if not scenarios:
        scenarios = [{}]

    worst_score = float("inf")
    for enemy_moves in scenarios:
        simulated = _simulate_turn(game_state, move, enemy_moves)
        if simulated["you"] is None or depth <= 1:
            score = _scenario_score(game_state, simulated)
        else:
            score = _paranoid_minimax(simulated, depth - 1, alpha, beta)

        worst_score = min(worst_score, score)
        beta = min(beta, worst_score)
        if beta <= alpha:
            break

    return worst_score


def _paranoid_minimax(game_state: Dict, depth: int, alpha: float, beta: float) -> float:
    """Maximize our score while treating all other snakes as one adversary."""
    if game_state["you"] is None:
        return -1_000_000.0
    if depth <= 0:
        return _scenario_score(game_state, game_state)

    candidates = _safe_moves(game_state)
    if not candidates:
        return -1_000_000.0

    best_score = float("-inf")
    for move in candidates:
        score = _paranoid_minimax_after_move(game_state, move, depth, alpha, beta)
        best_score = max(best_score, score)
        alpha = max(alpha, best_score)
        if alpha >= beta:
            break

    return best_score


def _enemy_move_scenarios(game_state: Dict) -> List[Dict[str, str]]:
    enemies = [s for s in game_state["board"]["snakes"] if s["id"] != game_state["you"]["id"]]
    scenarios: List[Tuple[Dict[str, str], float]] = [({}, 0.0)]

    for enemy in enemies:
        ranked_moves = _rank_enemy_moves(game_state, enemy)[:ENEMY_BRANCHING]
        if not ranked_moves:
            continue
        next_scenarios: List[Tuple[Dict[str, str], float]] = []
        for partial, partial_score in scenarios:
            for enemy_move, enemy_score in ranked_moves:
                merged = dict(partial)
                merged[enemy["id"]] = enemy_move
                next_scenarios.append((merged, partial_score + enemy_score))
        next_scenarios.sort(key=lambda item: item[1], reverse=True)
        scenarios = next_scenarios[:MAX_ENEMY_SCENARIOS]

    return [moves for moves, _ in scenarios]


def _rank_enemy_moves(game_state: Dict, enemy: Dict) -> List[Tuple[str, float]]:
    board = game_state["board"]
    width, height = board["width"], board["height"]
    moves = _legal_moves_for_snake(game_state, enemy)
    if not moves:
        return []

    occupied = _occupied_cells(board["snakes"])
    if enemy["body"]:
        occupied.discard(_snake_tail(enemy))
    foods = [(f["x"], f["y"]) for f in board["food"]]
    danger = _head_to_head_cells(board["snakes"], enemy["id"], enemy["length"])
    head = _snake_head(enemy)

    ranked = []
    for move in moves:
        nxt = _step(head, move)
        space = _flood_fill(nxt, occupied, width, height, limit=width * height)
        escape = _escape_count(nxt, occupied, width, height)
        score = float(space) + escape * 5.0
        score += min(nxt[0], width - 1 - nxt[0], nxt[1], height - 1 - nxt[1]) * 2.0
        if foods and enemy["health"] < HUNGRY_THRESHOLD:
            nearest = min(_manhattan(nxt, food) for food in foods)
            score += (width + height - nearest) * 3.0
        if nxt in danger:
            score -= HEAD_TO_HEAD_PENALTY
        ranked.append((move, score))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _simulate_turn(game_state: Dict, my_move: str, enemy_moves: Dict[str, str]) -> Dict:
    board = game_state["board"]
    foods = {(f["x"], f["y"]) for f in board["food"]}
    moves = {game_state["you"]["id"]: my_move, **enemy_moves}

    moved = []
    for snake in board["snakes"]:
        move = moves.get(snake["id"])
        if move is None:
            continue
        new_head = _step(_snake_head(snake), move)
        ate = new_head in foods
        body_points = [new_head] + [(seg["x"], seg["y"]) for seg in snake["body"]]
        if not ate:
            body_points = body_points[:-1]
        health = 100 if ate else snake["health"] - 1
        if health <= 0:
            continue
        moved.append(
            {
                **snake,
                "health": health,
                "body": [{"x": x, "y": y} for x, y in body_points],
                "head": {"x": new_head[0], "y": new_head[1]},
                "length": len(body_points),
                "_ate": ate,
            }
        )

    eliminated: Set[str] = set()
    heads: Dict[Point, List[Dict]] = {}
    for snake in moved:
        heads.setdefault(_snake_head(snake), []).append(snake)

    for same_cell in heads.values():
        if len(same_cell) <= 1:
            continue
        max_length = max(s["length"] for s in same_cell)
        winners = [s for s in same_cell if s["length"] == max_length]
        if len(winners) != 1:
            eliminated.update(s["id"] for s in same_cell)
        else:
            eliminated.update(s["id"] for s in same_cell if s["id"] != winners[0]["id"])

    body_cells: Set[Point] = set()
    for snake in moved:
        for seg in snake["body"][1:]:
            body_cells.add((seg["x"], seg["y"]))

    for snake in moved:
        if snake["id"] in eliminated:
            continue
        if _snake_head(snake) in body_cells:
            eliminated.add(snake["id"])

    survivors = []
    consumed_food: Set[Point] = set()
    for snake in moved:
        if snake["id"] in eliminated:
            continue
        if snake.pop("_ate", False):
            consumed_food.add(_snake_head(snake))
        survivors.append(snake)

    next_food = [food for food in foods if food not in consumed_food]
    my_next = next((s for s in survivors if s["id"] == game_state["you"]["id"]), None)
    return {
        **game_state,
        "turn": game_state.get("turn", 0) + 1,
        "board": {
            **board,
            "snakes": survivors,
            "food": [{"x": x, "y": y} for x, y in next_food],
        },
        "you": my_next,
    }


def _scenario_score(previous_state: Dict, simulated_state: Dict) -> float:
    you = simulated_state["you"]
    if you is None:
        return -1_000_000.0

    board = simulated_state["board"]
    width, height = board["width"], board["height"]
    enemies = [s for s in board["snakes"] if s["id"] != you["id"]]
    previous_enemy_count = len(previous_state["board"]["snakes"]) - 1
    killed_enemies = previous_enemy_count - len(enemies)

    head = _snake_head(you)
    occupied = _occupied_cells(board["snakes"])
    if you["body"]:
        occupied.discard(_snake_tail(you))
    open_space = _flood_fill(head, occupied, width, height, limit=width * height)
    escape = _escape_count(head, occupied, width, height)

    enemy_heads = [_snake_head(enemy) for enemy in enemies]
    my_dist = _bfs_dist([head], occupied, width, height)
    enemy_dist = _bfs_dist(enemy_heads, occupied, width, height) if enemy_heads else {}
    territory = sum(1 for cell, md in my_dist.items() if md < enemy_dist.get(cell, _BIG))

    my_tail = _snake_tail(you)
    reaches_tail = 1.0 if my_tail in _bfs_dist([head], occupied - {my_tail}, width, height) else 0.0
    length_lead = you["length"] - max((enemy["length"] for enemy in enemies), default=0)

    score = 0.0
    score += open_space * 3.5
    score += territory * 6.0
    score += escape * 25.0
    score += reaches_tail * 80.0
    score += length_lead * 18.0
    score += you["health"] * 0.6
    score += killed_enemies * 180.0

    foods = [(f["x"], f["y"]) for f in board["food"]]
    if foods:
        nearest_food = min(_manhattan(head, food) for food in foods)
        if you["health"] < HUNGRY_THRESHOLD:
            score += (width + height - nearest_food) * 8.0
        else:
            score -= nearest_food * 0.8

    danger = _head_to_head_cells(board["snakes"], you["id"], you["length"])
    if head in danger:
        score -= HEAD_TO_HEAD_PENALTY
    return score


def _snake_head(snake: Dict) -> Point:
    return snake["head"]["x"], snake["head"]["y"]


def _snake_tail(snake: Dict) -> Point:
    tail = snake["body"][-1]
    return tail["x"], tail["y"]


def _step(point: Point, move: str) -> Point:
    dx, dy = DIRECTIONS[move]
    return point[0] + dx, point[1] + dy


def _escape_count(point: Point, occupied: Set[Point], width: int, height: int) -> int:
    return sum(
        1
        for dx, dy in DIRECTIONS.values()
        if _in_bounds((point[0] + dx, point[1] + dy), width, height)
        and (point[0] + dx, point[1] + dy) not in occupied
    )


def _legal_moves(game_state: Dict) -> List[str]:
    return _legal_moves_for_snake(game_state, game_state["you"])


def _legal_moves_for_snake(game_state: Dict, snake: Dict) -> List[str]:
    board = game_state["board"]
    width, height = board["width"], board["height"]
    head = (snake["head"]["x"], snake["head"]["y"])
    occupied = _occupied_cells(board["snakes"])
    if snake["body"]:
        tail = snake["body"][-1]
        occupied.discard((tail["x"], tail["y"]))
    return [
        move
        for move, (dx, dy) in DIRECTIONS.items()
        if _in_bounds((head[0] + dx, head[1] + dy), width, height)
        and (head[0] + dx, head[1] + dy) not in occupied
    ]
