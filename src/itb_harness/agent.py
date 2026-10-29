"""Model adapters for the observer/act loop.

The loop does not depend on the Mistral SDK directly.  It talks to a small
``ChatAgent`` protocol, which makes the loop testable with scripted turns.
"""

from __future__ import annotations

import contextlib
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
        on_content: Callable[[str], None] | None = None,
    ) -> ModelTurn:
        """Return the next assistant turn."""


class MistralAgent:
    """Mistral Responses API adapter with vision + tool calling."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        prompt_cache_key: str | None = None,
        stream: bool = False,
        timeout_ms: int | None = 180_000,
        reasoning_effort: str | None = None,
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
        self.timeout_ms = timeout_ms
        self.reasoning_effort = reasoning_effort or os.environ.get(
            "MISTRAL_REASONING_EFFORT"
        )
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
        on_content: Callable[[str], None] | None = None,
    ) -> ModelTurn:
        input_items, instructions = _responses_input(messages)
        request: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "prompt_cache_key": self.prompt_cache_key,
        }
        if instructions:
            request["instructions"] = instructions
        if tools:
            request["tools"] = _responses_tools(tools)
            request["tool_choice"] = "required"
            request["parallel_tool_calls"] = False
        if self.reasoning_effort:
            request["reasoning"] = {"effort": self.reasoning_effort}
        if self.timeout_ms is not None:
            request["timeout_ms"] = self.timeout_ms

        if self.stream:
            return self._complete_streaming(
                request, on_thinking, on_content
            )

        response = self.client.responses.create(**request)
        return _turn_from_response(response)

    def _complete_streaming(
        self,
        request: dict[str, Any],
        on_thinking: Callable[[str], None] | None,
        on_content: Callable[[str], None] | None,
    ) -> ModelTurn:
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: dict[str, dict[str, Any]] = {}
        done_items: list[Any] = []
        final_response: Any | None = None

        stream = self.client.responses.create(**request, stream=True)
        with contextlib.ExitStack() as stack:
            if hasattr(stream, "__enter__"):
                stack.enter_context(stream)
            for frame in stream:
                event = _get(frame, "data")
                if event is None:
                    event = frame
                event_type = _get(event, "type")

                if event_type == "response.output_text.delta":
                    delta = _get(event, "delta")
                    if delta:
                        text_parts.append(str(delta))
                        if on_content is not None:
                            on_content(str(delta))
                elif event_type == "response.reasoning_text.delta":
                    delta = _get(event, "delta")
                    if delta:
                        thinking_parts.append(str(delta))
                        if on_thinking is not None:
                            on_thinking(str(delta))
                elif event_type == "response.output_item.added":
                    item = _get(event, "item")
                    if _get(item, "type") == "function_call":
                        _record_function_call(
                            tool_calls,
                            item,
                            _get(event, "output_index"),
                        )
                elif event_type == "response.function_call_arguments.delta":
                    key = str(
                        _get(event, "item_id")
                        or _get(event, "output_index")
                    )
                    entry = tool_calls.setdefault(
                        key, {"id": None, "name": "", "arguments": ""}
                    )
                    delta = _get(event, "delta")
                    if isinstance(delta, str):
                        entry["arguments"] += delta
                elif event_type == "response.function_call_arguments.done":
                    key = str(
                        _get(event, "item_id")
                        or _get(event, "output_index")
                    )
                    entry = tool_calls.setdefault(
                        key, {"id": None, "name": "", "arguments": ""}
                    )
                    name = _get(event, "name")
                    if name:
                        entry["name"] = str(name)
                    arguments = _get(event, "arguments")
                    if isinstance(arguments, str):
                        entry["arguments"] = arguments
                elif event_type == "response.output_item.done":
                    item = _get(event, "item")
                    if item is not None:
                        done_items.append(item)
                    if _get(item, "type") == "function_call":
                        _record_function_call(
                            tool_calls,
                            item,
                            _get(event, "output_index"),
                        )
                elif event_type == "response.completed":
                    final_response = _get(event, "response")
                elif event_type in {"response.failed", "response.incomplete"}:
                    response = _get(event, "response")
                    error = _get(response, "error")
                    message = _get(error, "message") if error else None
                    details = _get(response, "incomplete_details")
                    reason = _get(details, "reason") if details else None
                    suffix = f": {reason}" if reason else ""
                    raise RuntimeError(
                        message
                        or "model response "
                        f"{event_type.removeprefix('response.')}{suffix}"
                    )
                elif event_type in {"error", "response.error"}:
                    message = _get(event, "message")
                    raise RuntimeError(
                        message or "model response stream failed"
                    )

        if final_response is not None:
            turn = _turn_from_response(final_response)
            return ModelTurn(
                content=(
                    turn.content
                    if turn.content is not None
                    else "".join(text_parts) or None
                ),
                thinking=(
                    turn.thinking
                    if turn.thinking is not None
                    else "".join(thinking_parts) or None
                ),
                tool_calls=turn.tool_calls
                or _finalize_stream_tool_calls(tool_calls),
                usage=turn.usage,
            )

        fallback = _turn_from_response({"output": done_items})
        return ModelTurn(
            content=fallback.content or "".join(text_parts) or None,
            thinking=fallback.thinking or "".join(thinking_parts) or None,
            tool_calls=fallback.tool_calls
            or _finalize_stream_tool_calls(tool_calls),
            usage=fallback.usage,
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
        on_content: Callable[[str], None] | None = None,
    ) -> ModelTurn:
        self.calls.append(messages)
        if self.index >= len(self._turns):
            return ModelTurn(content="done")
        turn = self._turns[self.index]
        self.index += 1
        if on_thinking is not None and turn.thinking:
            on_thinking(turn.thinking)
        if on_content is not None and turn.content:
            on_content(turn.content)
        return turn


def _responses_input(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    """Convert chat-style loop messages into Responses API input items."""

    items: list[dict[str, Any]] = []
    instructions: list[str] = []

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role == "system":
            text = _message_text(content)
            if text:
                instructions.append(text)
            continue

        if role == "assistant":
            text = _message_text(content)
            if text:
                items.append({"role": "assistant", "content": text})
            for index, call in enumerate(message.get("tool_calls") or []):
                function = call.get("function") or {}
                arguments = function.get("arguments", {})
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments)
                items.append(
                    {
                        "type": "function_call",
                        "call_id": str(call.get("id") or f"call-{index + 1}"),
                        "name": function.get("name") or "",
                        "arguments": arguments,
                    }
                )
            continue

        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(message.get("tool_call_id") or ""),
                    "output": _responses_content(content),
                }
            )
            continue

        items.append(
            {
                "role": role or "user",
                "content": _responses_content(content),
            }
        )

    return items, "\n\n".join(instructions) or None


def _responses_content(content: Any) -> Any:
    """Convert chat-completions content chunks to Responses input chunks."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[dict[str, Any]] = []
    for chunk in content:
        if isinstance(chunk, str):
            parts.append({"type": "input_text", "text": chunk})
            continue

        chunk_type = _get(chunk, "type")
        if chunk_type in {"image_url", "input_image"}:
            image_url = _get(chunk, "image_url")
            url = (
                image_url
                if isinstance(image_url, str)
                else _get(image_url, "url")
            )
            if url:
                parts.append(
                    {
                        "type": "input_image",
                        "image_url": str(url),
                        "detail": "auto",
                    }
                )
            continue

        text = _get(chunk, "text")
        if text is not None:
            parts.append({"type": "input_text", "text": str(text)})

    if not parts:
        return ""
    if all(part["type"] == "input_text" for part in parts):
        return "\n".join(part["text"] for part in parts)
    return parts


def _responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten chat-completions function tools for the Responses API."""

    converted: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function")
        if tool.get("type") == "function" and isinstance(function, dict):
            converted.append(
                {
                    "type": "function",
                    "name": function.get("name") or "",
                    "description": function.get("description") or "",
                    "parameters": function.get("parameters")
                    or {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                }
            )
        else:
            converted.append(tool)
    return converted


def _turn_from_response(response: Any) -> ModelTurn:
    texts: list[str] = []
    thinking: list[str] = []
    calls: list[ModelToolCall] = []

    for item in _get(response, "output") or []:
        item_type = _get(item, "type")
        if item_type == "message":
            for part in _get(item, "content") or []:
                part_type = _get(part, "type")
                if part_type == "output_text":
                    text = _get(part, "text")
                    if text:
                        texts.append(str(text))
                elif part_type == "refusal":
                    refusal = _get(part, "refusal")
                    if refusal:
                        texts.append(str(refusal))
        elif item_type == "reasoning":
            thinking.extend(_reasoning_texts(item))
        elif item_type == "function_call":
            calls.append(_tool_call_from_item(item, len(calls)))

    return ModelTurn(
        content="\n".join(texts) or None,
        thinking="\n".join(thinking) or None,
        tool_calls=tuple(calls),
        usage=_usage_dict(_get(response, "usage")),
    )


def _reasoning_texts(item: Any) -> list[str]:
    texts: list[str] = []
    for part in _get(item, "content") or []:
        if _get(part, "type") == "reasoning_text":
            text = _get(part, "text")
            if text:
                texts.append(str(text))
    if texts:
        return texts
    for part in _get(item, "summary") or []:
        text = _get(part, "text")
        if text:
            texts.append(str(text))
    return texts


def _tool_call_from_item(item: Any, index: int) -> ModelToolCall:
    raw = _get(item, "arguments")
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            arguments: Any = {}
        else:
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError:
                arguments = raw
    elif raw is None:
        arguments = {}
    else:
        arguments = raw

    return ModelToolCall(
        id=str(
            _get(item, "call_id")
            or _get(item, "id")
            or f"call-{index + 1}"
        ),
        name=str(_get(item, "name") or ""),
        arguments=arguments,
    )


def _record_function_call(
    accumulator: dict[str, dict[str, Any]],
    item: Any,
    output_index: Any = None,
) -> None:
    key = str(_get(item, "id") or output_index or len(accumulator))
    entry = accumulator.setdefault(
        key, {"id": None, "name": "", "arguments": ""}
    )
    call_id = _get(item, "call_id")
    if call_id:
        entry["id"] = str(call_id)
    name = _get(item, "name")
    if name:
        entry["name"] = str(name)
    arguments = _get(item, "arguments")
    if isinstance(arguments, str):
        entry["arguments"] = arguments


def _finalize_stream_tool_calls(
    accumulator: dict[str, dict[str, Any]],
) -> tuple[ModelToolCall, ...]:
    result: list[ModelToolCall] = []
    for index, entry in enumerate(accumulator.values()):
        name = entry["name"]
        if not name:
            continue
        raw = entry["arguments"].strip()
        if raw:
            try:
                arguments: Any = json.loads(raw)
            except json.JSONDecodeError:
                arguments = raw
        else:
            arguments = {}
        result.append(
            ModelToolCall(
                id=entry["id"] or f"call-{index + 1}",
                name=name,
                arguments=arguments,
            )
        )
    return tuple(result)


def _message_text(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    texts: list[str] = []
    for chunk in content:
        text = (
            chunk.get("text")
            if isinstance(chunk, dict)
            else getattr(chunk, "text", None)
        )
        if text:
            texts.append(str(text))
    return "\n".join(texts) or None


def _usage_dict(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None

    result: dict[str, int] = {}
    for target, source in (
        ("prompt_tokens", "input_tokens"),
        ("completion_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        value = _get(usage, source)
        if value is not None:
            result[target] = int(value)

    details = _get(usage, "input_tokens_details")
    if details is not None:
        cached = _get(details, "cached_tokens")
        if cached is not None:
            result["cached_tokens"] = int(cached)

    return result or None


def _get(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)
