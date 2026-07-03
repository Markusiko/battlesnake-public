
import unittest

from logic import _legal_moves, _safe_moves, choose_move


def cell(x, y):
    return {"x": x, "y": y}


def snake(snake_id, body, health=90):
    cells = [cell(x, y) for x, y in body]
    return {
        "id": snake_id,
        "name": snake_id,
        "health": health,
        "body": cells,
        "head": cells[0],
        "length": len(cells),
    }


def state(width, height, you, enemies=None, food=None):
    enemies = enemies or []
    food = food or []
    return {
        "game": {"id": "game-id", "ruleset": {"name": "standard"}},
        "turn": 1,
        "board": {
            "height": height,
            "width": width,
            "food": [cell(x, y) for x, y in food],
            "hazards": [],
            "snakes": [you] + enemies,
        },
        "you": you,
    }


class MoveSafetyTests(unittest.TestCase):
    def test_legal_moves_allow_own_tail_when_it_will_move(self):
        you = snake("you", [(2, 2), (2, 1), (1, 1), (1, 2)])
        game_state = state(5, 5, you)

        self.assertIn("left", _legal_moves(game_state))

    def test_choose_move_avoids_head_to_head_against_equal_snake(self):
        you = snake("you", [(2, 2), (2, 1), (2, 0)])
        enemy = snake("enemy", [(2, 4), (3, 4), (4, 4)])
        game_state = state(5, 5, you, enemies=[enemy])

        self.assertNotEqual(choose_move(game_state), "up")

    def test_safe_moves_drop_one_turn_dead_end_when_other_options_exist(self):
        you = snake("you", [(1, 1), (0, 1), (0, 0)])
        blocker = snake("blocker", [(3, 0), (3, 1), (2, 2), (2, 0), (1, 0)])
        game_state = state(4, 3, you, enemies=[blocker])

        moves = _safe_moves(game_state)

        self.assertIn("up", moves)
        self.assertNotIn("right", moves)

    def test_choose_move_never_returns_wall_move_when_legal_moves_exist(self):
        you = snake("you", [(0, 0), (0, 1), (1, 1)])
        game_state = state(3, 3, you)

        self.assertIn(choose_move(game_state), {"right"})


if __name__ == "__main__":
    unittest.main()
