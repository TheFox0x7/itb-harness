from __future__ import annotations

import json
import unittest

from itb_harness.tools import ToolResult


class ToolResultTests(unittest.TestCase):
    def test_text_result(self) -> None:
        result = ToolResult(ok=True, message="moved")
        self.assertEqual(
            json.loads(result.as_content()),
            {"ok": True, "message": "moved"},
        )

    def test_tool_message_shape(self) -> None:
        result = ToolResult(ok=False, error="invalid_arguments", message="bad")
        message = result.as_tool_message("call-1", "move_up")
        self.assertEqual(message["role"], "tool")
        self.assertEqual(message["tool_call_id"], "call-1")
        self.assertEqual(message["name"], "move_up")
        self.assertIsInstance(message["content"], str)

    def test_image_result(self) -> None:
        result = ToolResult(ok=True, image_png=b"png")
        content = result.as_content()
        self.assertIsInstance(content, list)
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")

    def test_blocked_field_in_payload(self) -> None:
        result = ToolResult(ok=True, message="blocked", blocked=True)
        payload = result.payload()
        self.assertTrue(payload["blocked"])

    def test_blocked_field_omitted_when_none(self) -> None:
        result = ToolResult(ok=True, message="moved")
        payload = result.payload()
        self.assertNotIn("blocked", payload)

    def test_blocked_false_in_payload(self) -> None:
        result = ToolResult(ok=True, message="moved", blocked=False)
        payload = result.payload()
        self.assertFalse(payload["blocked"])


if __name__ == "__main__":
    unittest.main()
