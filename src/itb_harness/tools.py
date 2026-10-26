"""Tool-result plumbing shared by the observation/movement loop."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolResult:
    ok: bool
    message: str = ""
    error: str | None = None
    image_png: bytes | None = None
    state_changed: bool = False
    pixel: tuple[int, int] | None = None

    def payload(self) -> dict[str, Any]:
        """Small structured result for logs and tool messages."""

        payload: dict[str, Any] = {"ok": self.ok}
        if self.message:
            payload["message"] = self.message
        if self.error:
            payload["error"] = self.error
        return payload

    def as_content(self) -> str | list[dict[str, Any]]:
        """Mistral tool-message content, optionally including a screenshot."""

        text = json.dumps(self.payload())
        if not self.image_png:
            return text

        encoded = base64.b64encode(self.image_png).decode("ascii")
        return [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            },
        ]

    def as_tool_message(self, tool_call_id: str, name: str) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": self.as_content(),
        }
