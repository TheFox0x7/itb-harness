from __future__ import annotations

import unittest

from itb_harness.validator import validate_solution


class ValidatorTests(unittest.TestCase):
    def test_valid_when_adjacent_to_demon(self) -> None:
        result = validate_solution(
            [["H", ".", "D"]],
            ["move_right"],
            max_steps=1,
            spike_mode="static",
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.steps, 1)
        self.assertEqual(result.cost, 1)

    def test_demon_cell_itself_is_not_a_valid_destination(self) -> None:
        result = validate_solution(
            [["H", "D"]],
            ["move_right"],
            max_steps=1,
            spike_mode="static",
        )
        self.assertFalse(result.ok)
        self.assertIn("cannot move onto the demon", result.reason)

    def test_blocked_move_fails_validation(self) -> None:
        result = validate_solution(
            [["H", "X", ".", "D"]],
            ["move_right"],
            max_steps=1,
            spike_mode="static",
        )
        self.assertFalse(result.ok)
        self.assertIn("wall", result.reason)

    def test_must_finish_adjacent_to_demon(self) -> None:
        result = validate_solution(
            [["H", ".", ".", "D"]],
            ["move_right"],
            max_steps=2,
            spike_mode="static",
        )
        self.assertFalse(result.ok)
        self.assertIn("not adjacent", result.reason)

    def test_max_steps_is_obeyed(self) -> None:
        result = validate_solution(
            [["H", ".", ".", "D"]],
            ["move_right", "move_right"],
            max_steps=1,
            spike_mode="static",
        )
        self.assertFalse(result.ok)
        self.assertIn("max_steps", result.reason)

    def test_static_extended_spikes_cost_two_will(self) -> None:
        result = validate_solution(
            [["H", "T", "D"]],
            ["move_right"],
            max_steps=2,
            spike_mode="static",
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.cost, 2)

        too_short = validate_solution(
            [["H", "T", "D"]],
            ["move_right"],
            max_steps=1,
            spike_mode="static",
        )
        self.assertFalse(too_short.ok)

    def test_hidden_spikes_cost_two_will(self) -> None:
        result = validate_solution(
            [["H", "T", "D"]],
            ["move_right"],
            max_steps=2,
            spike_mode="down",
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.cost, 2)

        too_short = validate_solution(
            [["H", "T", "D"]],
            ["move_right"],
            max_steps=1,
            spike_mode="down",
        )
        self.assertFalse(too_short.ok)

    def test_alternating_extended_spikes_cost_one(self) -> None:
        result = validate_solution(
            [["H", "T", "D"]],
            ["move_right"],
            max_steps=1,
            spike_mode="up",
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.cost, 1)

    def test_spikes_alternate_each_move(self) -> None:
        result = validate_solution(
            [["H", "T", "T", "D"]],
            ["move_right", "move_right"],
            max_steps=3,
            spike_mode="down",
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.cost, 3)

        too_short = validate_solution(
            [["H", "T", "T", "D"]],
            ["move_right", "move_right"],
            max_steps=2,
            spike_mode="down",
        )
        self.assertFalse(too_short.ok)

    def test_barrier_blocks_without_key(self) -> None:
        result = validate_solution(
            [["H", "B", "D"]],
            ["move_right"],
            max_steps=1,
            spike_mode="static",
        )
        self.assertFalse(result.ok)
        self.assertIn("barrier", result.reason)

    def test_map_can_extend_beyond_visible_grid(self) -> None:
        result = validate_solution(
            [list("........H.D.")],
            ["move_right"],
            max_steps=1,
            spike_mode="static",
        )
        self.assertTrue(result.ok)

    def test_key_removes_barrier(self) -> None:
        result = validate_solution(
            [["H", "K", "B", "D"]],
            ["move_right", "move_right"],
            max_steps=2,
            spike_mode="static",
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.cost, 2)

    def test_non_move_actions_are_not_allowed(self) -> None:
        result = validate_solution(
            [["H", ".", "D"]],
            ["press_space"],
            max_steps=1,
            spike_mode="static",
        )
        self.assertFalse(result.ok)
        self.assertIn("unknown move", result.reason)

    def test_unknown_cells_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "\\?"):
            validate_solution(
                [["H", "?", "D"]],
                ["move_right"],
                max_steps=1,
                spike_mode="static",
            )

    def test_rock_push_and_skeleton_kick(self) -> None:
        rock = validate_solution(
            [["H", "R", "."], [".", ".", "D"]],
            ["move_right", "move_down"],
            max_steps=2,
            spike_mode="static",
        )
        self.assertTrue(rock.ok)

        skeleton = validate_solution(
            [["H", "S", ".", "D"]],
            ["move_right", "move_right"],
            max_steps=2,
            spike_mode="static",
        )
        self.assertTrue(skeleton.ok)


if __name__ == "__main__":
    unittest.main()
