from __future__ import annotations

import unittest

from itb_harness.board_tools import (
    apply_move_to_level_map,
    format_level_map,
    parse_level_map_args,
)


class LevelMapParsingTests(unittest.TestCase):
    def test_bracketed_rows(self) -> None:
        rows = parse_level_map_args(
            {
                "rows": [
                    "[X] [X] [X]",
                    "[H] [.] [D]",
                ]
            }
        )
        self.assertEqual(rows, [["X", "X", "X"], ["H", ".", "D"]])
        self.assertEqual(
            format_level_map(rows),
            "[X] [X] [X]\n[H] [.] [D]",
        )

    def test_compact_and_spaced_rows(self) -> None:
        self.assertEqual(
            parse_level_map_args({"rows": ["XXX", "H.D"]}),
            [["X", "X", "X"], ["H", ".", "D"]],
        )
        self.assertEqual(
            parse_level_map_args({"rows": "X X X\nH . D"}),
            [["X", "X", "X"], ["H", ".", "D"]],
        )

    def test_rows_can_be_lists_of_cells(self) -> None:
        self.assertEqual(
            parse_level_map_args({"rows": [["X", "H"], ["D", "X"]]}),
            [["X", "H"], ["D", "X"]],
        )

    def test_updates_are_applied_to_existing_map(self) -> None:
        current = [["X", "H", "X"], ["X", ".", "D"]]
        rows = parse_level_map_args(
            {
                "updates": [
                    {"row": 0, "col": 1, "symbol": "."},
                    {"row": 1, "col": 0, "symbol": "H"},
                ]
            },
            current,
        )
        self.assertEqual(rows, [["X", ".", "X"], ["H", ".", "D"]])

    def test_clear_removes_map(self) -> None:
        self.assertIsNone(
            parse_level_map_args({"clear": True}, [["X", "H"]])
        )

    def test_updates_without_rows_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rows"):
            parse_level_map_args(
                {"updates": [{"row": 0, "col": 0, "symbol": "H"}]}
            )

    def test_rows_must_be_rectangular(self) -> None:
        with self.assertRaisesRegex(ValueError, "same width"):
            parse_level_map_args({"rows": ["XXX", "HH"]})

    def test_symbols_must_be_single_characters(self) -> None:
        with self.assertRaisesRegex(ValueError, "single non-space"):
            parse_level_map_args({"rows": ["[wall] [X]"]})


class MoveMapUpdateTests(unittest.TestCase):
    def test_moves_into_empty_cell(self) -> None:
        self.assertEqual(
            apply_move_to_level_map([["H", "."]], "move_right"),
            [["." , "H"]],
        )

    def test_blocked_destination_leaves_map_unchanged(self) -> None:
        self.assertIsNone(
            apply_move_to_level_map([["H", "X"]], "move_right")
        )
        self.assertIsNone(
            apply_move_to_level_map([["H", "?"]], "move_right")
        )

    def test_kicks_skeleton_forward(self) -> None:
        self.assertEqual(
            apply_move_to_level_map([["H", "S", "."]], "move_right"),
            [["." , "H", "S"]],
        )

    def test_skeleton_is_removed_against_wall_or_rock(self) -> None:
        self.assertEqual(
            apply_move_to_level_map([["H", "S", "X"]], "move_right"),
            [["." , "H", "X"]],
        )
        self.assertEqual(
            apply_move_to_level_map([["H", "S", "R"]], "move_right"),
            [["." , "H", "R"]],
        )

    def test_pushes_rock_into_empty_or_spikes(self) -> None:
        self.assertEqual(
            apply_move_to_level_map([["H", "R", "."]], "move_right"),
            [["." , "H", "R"]],
        )
        self.assertEqual(
            apply_move_to_level_map([["H", "R", "T"]], "move_right"),
            [["." , "H", "R"]],
        )

    def test_rock_does_not_move_into_wall_or_unknown(self) -> None:
        self.assertIsNone(
            apply_move_to_level_map([["H", "R", "X"]], "move_right")
        )
        self.assertIsNone(
            apply_move_to_level_map([["H", "R", "?"]], "move_right")
        )

    def test_move_outside_map_or_without_player_is_ignored(self) -> None:
        self.assertIsNone(
            apply_move_to_level_map([["H"]], "move_left")
        )
        self.assertIsNone(
            apply_move_to_level_map([["S", "."]], "move_right")
        )


if __name__ == "__main__":
    unittest.main()
