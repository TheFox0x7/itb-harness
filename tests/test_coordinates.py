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
                "reset_level",
                "level_map",
                "validate_solution",
                "execute_plan",
                "press_space",
                "toggle_map_overlay",
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

    def test_level_map_schema(self) -> None:
        tool = next(
            tool
            for tool in tool_schemas()
            if tool["function"]["name"] == "level_map"
        )
        params = tool["function"]["parameters"]
        self.assertIn("rows", params["properties"])
        self.assertIn("updates", params["properties"])
        self.assertIn("clear", params["properties"])

    def test_validate_solution_schema(self) -> None:
        tool = next(
            tool
            for tool in tool_schemas()
            if tool["function"]["name"] == "validate_solution"
        )
        params = tool["function"]["parameters"]
        self.assertEqual(
            params["required"],
            ["moves", "max_steps", "spike_mode"],
        )

    def test_system_prompt_mentions_single_tile_movement(self) -> None:
        prompt = build_system_prompt()
        self.assertIn("move_up", prompt)
        self.assertIn("exactly one tile", prompt)
        self.assertIn("level_map", prompt)
        self.assertIn("validate_solution", prompt)

    def test_system_prompt_mentions_game_rules(self) -> None:
        prompt = build_system_prompt()
        self.assertIn("Skeleton knights can be kicked", prompt)
        self.assertIn("Black tiles are edges", prompt)
        self.assertIn("obstacles are pushable", prompt)
        self.assertIn("step budget", prompt)
        self.assertIn("reset_level", prompt)

    def test_system_prompt_mentions_mandatory_workflow(self) -> None:
        prompt = build_system_prompt()
        self.assertIn("PHASE 1: RECON", prompt)
        self.assertIn("PHASE 2: PLAN", prompt)
        self.assertIn("PHASE 3: EXECUTE AND VERIFY", prompt)
        self.assertIn("PHASE 4: MAP ERROR CORRECTION", prompt)
        self.assertIn("BLOCKED", prompt)
        self.assertIn("AFTER RESET, CHANGE STRATEGY", prompt)
        self.assertIn("OUTPUT ONLY TOOL CALLS", prompt)
        self.assertIn("KEEP THINKING CONCISE", prompt)
        self.assertIn("VALIDATE BEFORE EXECUTING", prompt)
        self.assertIn("VERIFY ALIGNMENT", prompt)
        self.assertIn("FIX MAP ERRORS IMMEDIATELY", prompt)


if __name__ == "__main__":
    unittest.main()
