import unittest

import logic


def _point(x, y):
    return {"x": x, "y": y}


class CriticalFoodGateTest(unittest.TestCase):
    def test_critical_health_takes_adjacent_safe_food_on_edge(self):
        you = {
            "id": "you",
            "name": "candidate",
            "health": 8,
            "head": _point(9, 9),
            "body": [_point(9, 9), _point(9, 8), _point(9, 7)],
            "length": 3,
        }
        game_state = {
            "game": {"id": "test"},
            "turn": 12,
            "board": {
                "height": 11,
                "width": 11,
                "food": [_point(10, 9)],
                "hazards": [],
                "snakes": [you],
            },
            "you": you,
        }

        self.assertEqual(logic.choose_move(game_state), "right")


if __name__ == "__main__":
    unittest.main()
