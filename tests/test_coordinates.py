from __future__ import annotations

import unittest

from itb_harness.board_tools import tool_schemas
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
            ],
        )

    def test_movement_schema_is_batchable(self) -> None:
        tools = {
            tool["function"]["name"]: tool for tool in tool_schemas()
        }
        for name in ("move_up", "move_down", "move_left", "move_right"):
            steps = tools[name]["function"]["parameters"]["properties"]["steps"]
            self.assertEqual(steps["minimum"], 1)
            self.assertEqual(steps["maximum"], 50)
            self.assertEqual(steps["default"], 1)

    def test_system_prompt_mentions_batched_movement(self) -> None:
        prompt = build_system_prompt()
        self.assertIn("move_up", prompt)
        self.assertIn("steps", prompt)
        self.assertIn("one tile", prompt)


if __name__ == "__main__":
    unittest.main()
