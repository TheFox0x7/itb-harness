from __future__ import annotations

import unittest

from itb_harness.board_tools import tool_schemas
from itb_harness.loop import build_system_prompt


class CoordinateModeTests(unittest.TestCase):
    def _click_schema(self, normalized: bool):
        tools = tool_schemas(normalized=normalized)
        return next(
            tool for tool in tools if tool["function"]["name"] == "click"
        )["function"]["parameters"]["properties"]["point_2d"]

    def test_pixel_schema(self) -> None:
        point = self._click_schema(False)
        self.assertEqual(point["items"]["type"], "integer")
        self.assertNotIn("maximum", point["items"])

    def test_normalized_schema(self) -> None:
        point = self._click_schema(True)
        self.assertEqual(point["items"]["type"], "number")
        self.assertEqual(point["items"]["maximum"], 1000)

    def test_system_prompt_coordinate_mode(self) -> None:
        self.assertIn("screenshot pixels", build_system_prompt(False))
        self.assertIn("normalized from 0 to 1000", build_system_prompt(True))


if __name__ == "__main__":
    unittest.main()
