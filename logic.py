"""Battlesnake move-selection logic."""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

Point = Tuple[int, int]
DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}
HEAD_TO_HEAD_PENALTY = 10_000
HUNGRY_THRESHOLD = 50
_BIG = 10_000
_NEIGHBORS = ((0, 1), (0, -1), (-1, 0), (1, 0))


def get_info() -> Dict[str, str]:
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#F8F8F2",
        "head": "smile",
        "tail": "curled",
        "version": "0.2.2",
    }


def choose_move(game_state: Dict) -> str:
    try:
        move = choose_move_model(game_state)
    except Exception:
        move = None
    return move or choose_move_heuristic(game_state)


def choose_move_heuristic(game_state: Dict) -> str:
    board = game_state["board"]
    you = game_state["you"]
    width, height = board["width"], board["height"]
    head = _point(you["head"])
    foods = [_point(food) for food in board["food"]]
    danger = _head_to_head_cells(board["snakes"], you["id"], you["length"])

    candidates = _safe_moves(game_state) or _legal_moves(game_state)
    if not candidates:
        return "up"

    best_move, best_score = None, float("-inf")
    for move in candidates:
        nxt = _next_point(head, move)
        occupied = _blocking_cells_for_move(game_state, move)
        score = float(_flood_fill(nxt, occupied, width, height, limit=you["length"] + 1))
        score += _followup_count(game_state, move) * 3

        if nxt in danger:
            score -= HEAD_TO_HEAD_PENALTY

        if foods and you["health"] < HUNGRY_THRESHOLD:
            nearest = min(_manhattan(nxt, food) for food in foods)
            score += (width + height - nearest) * 2
            if nxt in foods:
                score += 10

        if score > best_score:
            best_move, best_score = move, score

    return best_move or "up"


def choose_move_model(game_state: Dict) -> Optional[str]:
    candidates = _safe_moves(game_state) or _legal_moves(game_state)
    if not candidates:
        return None

    names = _MODEL["feature_names"]
    mean = _MODEL["mean"]
    std = _MODEL["std"]
    coef = _MODEL["coef"]
    intercept = _MODEL["intercept"]

    best_move, best_score = None, float("-inf")
    for move in candidates:
        feats = _candidate_features(game_state, move)
        score = intercept
        for i, name in enumerate(names):
            z = (feats.get(name, 0.0) - mean[i]) / std[i] if std[i] else 0.0
            score += coef[i] * z
        if score > best_score:
            best_move, best_score = move, score
    return best_move


def _point(cell: Dict) -> Point:
    return cell["x"], cell["y"]


def _next_point(head: Point, move: str) -> Point:
    dx, dy = DIRECTIONS[move]
    return head[0] + dx, head[1] + dy


def _food_cells(board: Dict) -> Set[Point]:
    return {_point(food) for food in board["food"]}


def _occupied_cells(snakes: List[Dict]) -> Set[Point]:
    return {_point(seg) for snake in snakes for seg in snake["body"]}


def _blocking_cells_for_move(game_state: Dict, move: str) -> Set[Point]:
    board = game_state["board"]
    you = game_state["you"]
    occupied = _occupied_cells(board["snakes"])

    if you["body"]:
        nxt = _next_point(_point(you["head"]), move)
        my_tail = _point(you["body"][-1])
        if nxt not in _food_cells(board):
            occupied.discard(my_tail)

    return occupied


def _head_to_head_cells(snakes: List[Dict], my_id: str, my_length: int) -> Set[Point]:
    danger: Set[Point] = set()
    for snake in snakes:
        if snake["id"] == my_id or snake["length"] < my_length:
            continue
        ehead = _point(snake["head"])
        for dx, dy in DIRECTIONS.values():
            danger.add((ehead[0] + dx, ehead[1] + dy))
    return danger


def _legal_moves(game_state: Dict) -> List[str]:
    board = game_state["board"]
    width, height = board["width"], board["height"]
    head = _point(game_state["you"]["head"])
    legal: List[str] = []

    for move in DIRECTIONS:
        nxt = _next_point(head, move)
        if _in_bounds(nxt, width, height) and nxt not in _blocking_cells_for_move(game_state, move):
            legal.append(move)

    return legal


def _safe_moves(game_state: Dict) -> List[str]:
    legal = _legal_moves(game_state)
    if not legal:
        return []

    board = game_state["board"]
    you = game_state["you"]
    head = _point(you["head"])
    danger = _head_to_head_cells(board["snakes"], you["id"], you["length"])

    non_h2h = [move for move in legal if _next_point(head, move) not in danger]
    candidates = non_h2h or legal
    with_followup = [move for move in candidates if _followup_count(game_state, move) > 0]
    return with_followup or candidates


def _followup_count(game_state: Dict, move: str) -> int:
    return len(_legal_moves(_simulate_you_after_move(game_state, move)))


def _simulate_you_after_move(game_state: Dict, move: str) -> Dict:
    board = game_state["board"]
    you = game_state["you"]
    nxt = _next_point(_point(you["head"]), move)
    ate_food = nxt in _food_cells(board)

    old_body = [_point(seg) for seg in you["body"]]
    new_body_points = [nxt] + old_body
    if not ate_food:
        new_body_points = new_body_points[:-1]

    new_body = [{"x": x, "y": y} for x, y in new_body_points]
    new_you = {
        **you,
        "head": {"x": nxt[0], "y": nxt[1]},
        "body": new_body,
        "length": len(new_body),
        "health": 100 if ate_food else max(you["health"] - 1, 0),
    }

    return {
        **game_state,
        "you": new_you,
        "board": {
            **board,
            "snakes": [
                new_you if snake["id"] == you["id"] else snake
                for snake in board["snakes"]
            ],
            "food": [
                food for food in board["food"]
                if (food["x"], food["y"]) != nxt
            ],
        },
    }


def _flood_fill(start: Point, occupied: Set[Point], width: int, height: int, limit: int) -> int:
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
            if nbr not in seen and _in_bounds(nbr, width, height) and nbr not in occupied:
                seen.add(nbr)
                stack.append(nbr)

    return count


def _bfs_dist(sources, blocked, width, height):
    dist = {}
    dq = deque()
    for source in sources:
        if source not in dist:
            dist[source] = 0
            dq.append(source)

    while dq:
        x, y = dq.popleft()
        for dx, dy in _NEIGHBORS:
            nb = (x + dx, y + dy)
            if 0 <= nb[0] < width and 0 <= nb[1] < height and nb not in blocked and nb not in dist:
                dist[nb] = dist[(x, y)] + 1
                dq.append(nb)

    return dist


def _candidate_features(state: Dict, move: str) -> Dict[str, float]:
    board = state["board"]
    you = state["you"]
    width, height = board["width"], board["height"]
    head = _point(you["head"])
    nxt = _next_point(head, move)
    occupied = _blocking_cells_for_move(state, move)
    foods = [_point(food) for food in board["food"]]
    enemies = [snake for snake in board["snakes"] if snake["id"] != you["id"]]
    enemy_heads = [_point(snake["head"]) for snake in enemies]
    bigger_heads = [_point(snake["head"]) for snake in enemies if snake["length"] >= you["length"]]
    danger = _head_to_head_cells(board["snakes"], you["id"], you["length"])

    my_dist = _bfs_dist([nxt], occupied, width, height)
    enemy_dist = _bfs_dist(enemy_heads, occupied, width, height) if enemy_heads else {}
    voronoi = sum(1 for cell, md in my_dist.items() if md < enemy_dist.get(cell, _BIG))

    my_tail = _point(you["body"][-1])
    reach = _bfs_dist([nxt], occupied - {my_tail}, width, height)
    nearest_now = min((_manhattan(head, food) for food in foods), default=_BIG)
    nearest_next = min((_manhattan(nxt, food) for food in foods), default=_BIG)

    return {
        "space_capped": float(_flood_fill(nxt, occupied, width, height, limit=you["length"] + 1)),
        "open_space": float(_flood_fill(nxt, occupied, width, height, limit=width * height)),
        "voronoi": float(voronoi),
        "reaches_tail": 1.0 if my_tail in reach else 0.0,
        "escape": float(sum(
            1 for dx, dy in _NEIGHBORS
            if _in_bounds((nxt[0] + dx, nxt[1] + dy), width, height)
            and (nxt[0] + dx, nxt[1] + dy) not in occupied
        )),
        "h2h_danger": 1.0 if nxt in danger else 0.0,
        "near_bigger_head": float(min((_manhattan(nxt, h) for h in bigger_heads), default=width + height)),
        "near_enemy_head": float(min((_manhattan(nxt, h) for h in enemy_heads), default=width + height)),
        "wall_dist": float(min(nxt[0], width - 1 - nxt[0], nxt[1], height - 1 - nxt[1])),
        "food_score": float((width + height - nearest_next) * 2) if you["health"] < HUNGRY_THRESHOLD and foods else 0.0,
        "food_delta": float(nearest_now - nearest_next) if foods else 0.0,
        "is_food": 1.0 if nxt in foods else 0.0,
        "dist_to_center": abs(nxt[0] - (width - 1) / 2) + abs(nxt[1] - (height - 1) / 2),
    }


def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


_MODEL: Dict = {
    "feature_names": [
        "space_capped", "open_space", "voronoi", "reaches_tail", "escape",
        "h2h_danger", "near_bigger_head", "near_enemy_head", "wall_dist",
        "food_score", "food_delta", "is_food", "dist_to_center",
    ],
    "mean": [
        7.357954545454546, 100.9034090909091, 48.26988636363637,
        0.9943181818181818, 2.4431818181818183, 0.04261363636363636,
        9.673295454545455, 4.676136363636363, 1.625,
        0.8920454545454546, 0.14772727272727273, 0.036931818181818184,
        5.056818181818182,
    ],
    "std": [
        3.5995966185276513, 22.80542174802676, 31.41119158524981,
        0.07516338951888041, 0.6235520417417705, 0.20198444088469822,
        7.9675173248507924, 2.2532045017839604, 1.3552297691803878,
        5.861056404757769, 0.9449599886584031, 0.18859442989548575,
        2.34451950177747,
    ],
    "coef": [
        0.00010539398521136327, -1.6778512168946185, 80.89420182766183,
        9.793855564450467, 0.7884630868036275, -11.025170822665032,
        -0.7981723553489, 0.5410534990053248, 1.5629078731518526,
        7.582325762611304, 0.12463070008097832, 0.21036618806863483,
        1.836259515524985,
    ],
    "intercept": 0.0,
    "top1_accuracy": 0.9928571428571429,
}
