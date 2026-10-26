from __future__ import annotations

import unittest

from itb_harness.board_tools import STRICT_BOX_LABELS, parse_boxes_args, tool_schemas
from itb_harness.loop import build_system_prompt


class ToolSchemaTests(unittest.TestCase):
    def test_tool_set_is_observation_plus_four_directions(self) -> None:
        names = [tool["function"]["name"] for tool in tool_schemas()]
        self.assertEqual(
            names,
            [
                "view_board",
                "move_up",
                "move_down",
                "move_left",
                "move_right",
                "reset_level",
                "toggle_grid",
                "toggle_walkability",
                "set_boxes",
                "add_boxes",
            ],
        )

    def test_movement_schema_has_no_arguments(self) -> None:
        tools = {
            tool["function"]["name"]: tool for tool in tool_schemas()
        }
        for name in ("move_up", "move_down", "move_left", "move_right"):
            params = tools[name]["function"]["parameters"]
            self.assertEqual(params["properties"], {})
            self.assertFalse(params["additionalProperties"])

    def test_boxes_schema_strict_labels(self) -> None:
        for name in ("set_boxes", "add_boxes"):
            free = next(
                tool for tool in tool_schemas()
                if tool["function"]["name"] == name
            )
            label = free["function"]["parameters"]["properties"]["boxes"]["items"]["properties"]["label"]
            self.assertNotIn("enum", label)

            strict = next(
                tool for tool in tool_schemas(strict_labels=True)
                if tool["function"]["name"] == name
            )
            label = strict["function"]["parameters"]["properties"]["boxes"]["items"]["properties"]["label"]
            self.assertEqual(label["enum"], list(STRICT_BOX_LABELS))

    def test_parse_boxes(self) -> None:
        boxes = parse_boxes_args(
            {
                "boxes": [
                    {"box_2d": [1, 2, 30, 40], "label": "custom thing"}
                ]
            }
        )
        self.assertEqual(boxes[0].xmin, 1)
        self.assertEqual(boxes[0].label, "custom thing")

        with self.assertRaises(ValueError):
            parse_boxes_args(
                {"boxes": [{"box_2d": [1, 2, 3, 4], "label": "unknown"}]},
                strict_labels=True,
            )

    def test_parse_scaled_boxes(self) -> None:
        boxes = parse_boxes_args(
            {
                "boxes": [
                    {"box_2d": [1, 1, 1000, 1000], "label": "rock"}
                ]
            },
            scaled_boxes=True,
            width=101,
            height=51,
        )
        self.assertEqual(
            (boxes[0].xmin, boxes[0].ymin, boxes[0].xmax, boxes[0].ymax),
            (0, 0, 100, 50),
        )

        with self.assertRaises(ValueError):
            parse_boxes_args(
                {"boxes": [{"box_2d": [0, 1, 1000, 1000], "label": "rock"}]},
                scaled_boxes=True,
                width=101,
                height=51,
            )

    def test_scaled_boxes_schema(self) -> None:
        tool = next(
            tool for tool in tool_schemas(scaled_boxes=True)
            if tool["function"]["name"] == "set_boxes"
        )
        item = tool["function"]["parameters"]["properties"]["boxes"]["items"]
        coords = item["properties"]["box_2d"]["items"]
        self.assertEqual(coords["minimum"], 1)
        self.assertEqual(coords["maximum"], 1000)

    def test_system_prompt_mentions_single_tile_movement(self) -> None:
        prompt = build_system_prompt()
        self.assertIn("move_up", prompt)
        self.assertIn("exactly one tile", prompt)
        self.assertIn("set_boxes", prompt)
        self.assertIn("add_boxes", prompt)

    def test_scaled_prompt_mentions_1_to_1000(self) -> None:
        prompt = build_system_prompt(scaled_boxes=True)
        self.assertIn("scaled from 1 to 1000", prompt)

    def test_strict_prompt_lists_categories(self) -> None:
        prompt = build_system_prompt(strict_box_labels=True)
        for label in STRICT_BOX_LABELS:
            self.assertIn(label, prompt)

    def test_system_prompt_mentions_game_rules(self) -> None:
        prompt = build_system_prompt()
        self.assertIn("Skeleton knights can be kicked", prompt)
        self.assertIn("Black tiles are edges", prompt)
        self.assertIn("obstacles are pushable", prompt)
        self.assertIn("step counter", prompt)
        self.assertIn("reset_level", prompt)

    def test_system_prompt_mentions_mandatory_workflow(self) -> None:
        prompt = build_system_prompt()
        self.assertIn("PLAN BEFORE MOVING", prompt)
        self.assertIn("MARK THE BOARD", prompt)
        self.assertIn("CHECK MOVE RESULTS", prompt)
        self.assertIn("BLOCKED", prompt)
        self.assertIn("AFTER RESET, CHANGE STRATEGY", prompt)
        self.assertIn("OUTPUT ONLY TOOL CALLS", prompt)
        self.assertIn("KEEP THINKING CONCISE", prompt)


if __name__ == "__main__":
    unittest.main()
