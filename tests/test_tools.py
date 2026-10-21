from __future__ import annotations

import unittest

from itb_harness.tools import Coord, StubActions, ToolRegistry


class ToolSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ToolRegistry()

    def test_expected_tool_names(self) -> None:
        names = [tool["function"]["name"] for tool in self.registry.mistral_tools()]
        self.assertEqual(
            names,
            [
                "move_mech",
                "prepare_attack",
                "execute_attack",
                "end_turn",
                "reset_turn",
                "inspect",
            ],
        )

    def test_coordinate_is_inlined_and_bounded(self) -> None:
        tool = next(
            tool
            for tool in self.registry.mistral_tools()
            if tool["function"]["name"] == "move_mech"
        )
        params = tool["function"]["parameters"]
        self.assertNotIn("$defs", params)
        target = params["properties"]["target"]
        self.assertEqual(target["properties"]["x"]["minimum"], 0)
        self.assertEqual(target["properties"]["x"]["maximum"], 7)
        self.assertEqual(target["required"], ["x", "y"])


class ToolExecutionTests(unittest.TestCase):
    def test_move_mech_valid(self) -> None:
        result = ToolRegistry().execute(
            "move_mech", {"mech": 1, "target": {"x": 3, "y": 4}}
        )
        self.assertTrue(result.ok)
        self.assertIn("mech 1", result.message)
        self.assertIn("(3, 4)", result.message)

    def test_move_mech_accepts_json_string(self) -> None:
        result = ToolRegistry().execute(
            "move_mech", '{"mech": 2, "target": {"x": 0, "y": 7}}'
        )
        self.assertTrue(result.ok)

    def test_out_of_range_mech_is_rejected(self) -> None:
        result = ToolRegistry().execute(
            "move_mech", {"mech": 4, "target": {"x": 0, "y": 0}}
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "invalid_arguments")

    def test_out_of_range_coordinate_is_rejected(self) -> None:
        result = ToolRegistry().execute(
            "move_mech", {"mech": 1, "target": {"x": 8, "y": 0}}
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "invalid_arguments")

    def test_execute_requires_prepare(self) -> None:
        result = ToolRegistry().execute(
            "execute_attack", {"target": {"x": 2, "y": 2}}
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "no_prepared_attack")

    def test_prepare_then_execute(self) -> None:
        actions = StubActions()
        registry = ToolRegistry(actions)
        prepared = registry.execute(
            "prepare_attack", {"mech": 1, "weapon": 1}
        )
        fired = registry.execute(
            "execute_attack", {"target": {"x": 2, "y": 2}}
        )
        self.assertTrue(prepared.ok)
        self.assertTrue(fired.ok)
        self.assertIsNone(actions.prepared)

    def test_reset_clears_prepared_attack(self) -> None:
        actions = StubActions()
        registry = ToolRegistry(actions)
        registry.execute("prepare_attack", {"mech": 1, "weapon": 1})
        registry.execute("reset_turn", {})
        result = registry.execute("execute_attack", {"target": {"x": 2, "y": 2}})
        self.assertEqual(result.error, "no_prepared_attack")

    def test_inspect_is_visual_only_stub(self) -> None:
        result = ToolRegistry().execute("inspect", {"target": {"x": 1, "y": 1}})
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "inspect_unavailable")

    def test_unknown_tool(self) -> None:
        result = ToolRegistry().execute("nope", {})
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "unknown_tool")

    def test_tool_message_shape(self) -> None:
        result = ToolRegistry().execute(
            "move_mech", {"mech": 1, "target": Coord(x=1, y=2).model_dump()}
        )
        message = result.as_tool_message("call-1", "move_mech")
        self.assertEqual(message["role"], "tool")
        self.assertEqual(message["tool_call_id"], "call-1")
        self.assertIsInstance(message["content"], str)


if __name__ == "__main__":
    unittest.main()
