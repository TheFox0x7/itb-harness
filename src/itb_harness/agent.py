"""Model adapters for the observer/act loop.

The loop does not depend on the Mistral SDK directly.  It talks to a small
``ChatAgent`` protocol, which makes the loop testable with scripted turns.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ModelToolCall:
    id: str
    name: str
    arguments: Any


@dataclass(frozen=True, slots=True)
class ModelTurn:
    content: str | None = None
    tool_calls: tuple[ModelToolCall, ...] = ()


class ChatAgent(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelTurn:
        """Return the next assistant turn."""


class MistralAgent:
    """Mistral chat-completions adapter with vision + tool calling."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        # temperature: float = 0.0,
    ) -> None:
        self.model = model or os.environ.get(
            "MISTRAL_MODEL", "mistral-large-4"
        )
        # self.temperature = temperature
        if client is None:
            key = api_key or os.environ.get("MISTRAL_API_KEY")
            if not key:
                raise RuntimeError(
                    "MISTRAL_API_KEY is not set. Export it or pass an API key."
                )
            from mistralai.client import Mistral
            client = Mistral(api_key=key)
        self.client = client

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelTurn:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            # "temperature": self.temperature,
        }
        if tools:
            # Kept for the future tool mode; the point-action loop passes none.
            request["tools"] = tools
            request["tool_choice"] = "required"
            request["parallel_tool_calls"] = False

        response = self.client.chat.complete(**request)
        choice = response.choices[0]
        message = choice.message
        if message is None:
            return ModelTurn()

        calls: list[ModelToolCall] = []
        for index, call in enumerate(message.tool_calls or []):
            arguments = call.function.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            calls.append(
                ModelToolCall(
                    id=call.id or f"call-{index + 1}",
                    name=call.function.name,
                    arguments=arguments,
                )
            )

        return ModelTurn(
            content=_content_text(message.content),
            tool_calls=tuple(calls),
        )


class ScriptedAgent:
    """Deterministic agent used by tests and dry-run demos."""

    def __init__(self, turns: list[ModelTurn]) -> None:
        self._turns = list(turns)
        self.index = 0
        self.calls: list[list[dict[str, Any]]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelTurn:
        self.calls.append(messages)
        if self.index >= len(self._turns):
            return ModelTurn(content="done")
        turn = self._turns[self.index]
        self.index += 1
        return turn


def _content_text(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict):
                if chunk.get("type") == "text" and chunk.get("text"):
                    parts.append(str(chunk["text"]))
            else:
                text = getattr(chunk, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts) or None
    return str(content)
