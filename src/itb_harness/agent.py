"""Model adapters for the observer/act loop.

The loop does not depend on the Mistral SDK directly.  It talks to a small
``ChatAgent`` protocol, which makes the loop testable with scripted turns.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass(frozen=True, slots=True)
class ModelToolCall:
    id: str
    name: str
    arguments: Any


@dataclass(frozen=True, slots=True)
class ModelTurn:
    content: str | None = None
    thinking: str | None = None
    tool_calls: tuple[ModelToolCall, ...] = ()
    usage: dict[str, int] | None = None


class ChatAgent(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelTurn:
        """Return the next assistant turn."""


class MistralAgent:
    """Mistral chat-completions adapter with vision + tool calling."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        prompt_cache_key: str | None = None,
        stream: bool = False,
        # temperature: float = 0.0,
    ) -> None:
        self.model = model or os.environ.get(
            "MISTRAL_MODEL", "mistral-large-4"
        )
        self.prompt_cache_key = (
            prompt_cache_key
            or os.environ.get("MISTRAL_PROMPT_CACHE_KEY")
            or f"itb-{uuid.uuid4().hex}"
        )
        self.stream = stream
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
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelTurn:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "prompt_cache_key": self.prompt_cache_key,
            # "temperature": self.temperature,
        }
        if tools:
            # Kept for the future tool mode; the point-action loop passes none.
            request["tools"] = tools
            request["tool_choice"] = "required"
            request["parallel_tool_calls"] = False

        if self.stream and hasattr(self.client.chat, "stream"):
            return self._complete_streaming(request, on_thinking)

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

        content, thinking = _split_content(message.content)
        return ModelTurn(
            content=content,
            thinking=thinking,
            tool_calls=tuple(calls),
            usage=_usage_dict(getattr(response, "usage", None)),
        )

    def _complete_streaming(
        self,
        request: dict[str, Any],
        on_thinking: Callable[[str], None] | None,
    ) -> ModelTurn:
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        usage: dict[str, int] | None = None

        for event in self.client.chat.stream(**request):
            chunk = getattr(event, "data", event)
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = _usage_dict(chunk_usage)

            for choice in getattr(chunk, "choices", None) or []:
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                content = getattr(delta, "content", None)
                _consume_stream_content(
                    content, text_parts, thinking_parts, on_thinking
                )

        return ModelTurn(
            content="".join(text_parts) or None,
            thinking="".join(thinking_parts) or None,
            usage=usage,
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
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelTurn:
        self.calls.append(messages)
        if self.index >= len(self._turns):
            return ModelTurn(content="done")
        turn = self._turns[self.index]
        self.index += 1
        return turn


def _usage_dict(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None

    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if value is not None:
            result[key] = int(value)

    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        extra = getattr(usage, "additional_properties", None) or {}
        details = extra.get("prompt_tokens_details")

    if details is not None:
        cached = (
            details.get("cached_tokens")
            if isinstance(details, dict)
            else getattr(details, "cached_tokens", None)
        )
        if cached is not None:
            result["cached_tokens"] = int(cached)

    return result or None


def _consume_stream_content(
    content: Any,
    text_parts: list[str],
    thinking_parts: list[str],
    on_thinking: Callable[[str], None] | None,
) -> None:
    if content is None:
        return
    if isinstance(content, str):
        text_parts.append(content)
        return
    if not isinstance(content, list):
        return

    for chunk in content:
        chunk_type = (
            chunk.get("type") if isinstance(chunk, dict) else getattr(chunk, "type", None)
        )
        if chunk_type == "thinking":
            thinking = _thinking_text(chunk)
            if thinking:
                thinking_parts.append(thinking)
                if on_thinking is not None:
                    on_thinking(thinking)
            continue

        text = _chunk_text(chunk)
        if text:
            text_parts.append(text)


def _split_content(content: Any) -> tuple[str | None, str | None]:
    """Separate visible answer text from reasoning/thinking chunks."""

    if content is None:
        return None, None
    if isinstance(content, str):
        return content, None
    if not isinstance(content, list):
        return str(content), None

    text_parts: list[str] = []
    thinking_parts: list[str] = []

    for chunk in content:
        chunk_type = (
            chunk.get("type") if isinstance(chunk, dict) else getattr(chunk, "type", None)
        )
        if chunk_type == "thinking":
            thinking = _thinking_text(chunk)
            if thinking:
                thinking_parts.append(thinking)
            continue

        text = _chunk_text(chunk)
        if text:
            text_parts.append(text)

    return "\n".join(text_parts) or None, "\n".join(thinking_parts) or None


def _chunk_text(chunk: Any) -> str | None:
    if isinstance(chunk, dict):
        text = chunk.get("text")
    else:
        text = getattr(chunk, "text", None)
    return str(text) if text else None


def _thinking_text(chunk: Any) -> str | None:
    if isinstance(chunk, dict):
        parts = chunk.get("thinking") or []
    else:
        parts = getattr(chunk, "thinking", None) or []

    texts = []
    for part in parts:
        text = _chunk_text(part)
        if text:
            texts.append(text)
    return "\n".join(texts) or None
