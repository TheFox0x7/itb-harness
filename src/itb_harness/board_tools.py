"""Observation and movement tools exposed to the model."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class MoveArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: int = Field(
        default=1,
        ge=1,
        le=50,
        description="Number of tiles to move in this batch.",
    )


_EMPTY_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

_MOVE_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "integer",
            "minimum": 1,
            "maximum": 50,
            "default": 1,
            "description": "Number of tiles to move in this batch.",
        }
    },
    "required": [],
    "additionalProperties": False,
}


def _tool(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def tool_schemas() -> list[dict[str, Any]]:
    """Return the tool definitions visible to the model."""

    move_schema = _MOVE_SCHEMA
    return [
        _tool(
            "view_board",
            "Capture and return the current game screenshot. Call this first "
            "and after movement when you need to see the result.",
            _EMPTY_SCHEMA,
        ),
        _tool(
            "move_up",
            "Move upward by pressing the Up arrow. Set steps to move several "
            "tiles in one batch.",
            move_schema,
        ),
        _tool(
            "move_down",
            "Move downward by pressing the Down arrow. Set steps to move "
            "several tiles in one batch.",
            move_schema,
        ),
        _tool(
            "move_left",
            "Move left by pressing the Left arrow. Set steps to move several "
            "tiles in one batch.",
            move_schema,
        ),
        _tool(
            "move_right",
            "Move right by pressing the Right arrow. Set steps to move several "
            "tiles in one batch.",
            move_schema,
        ),
    ]


def _load_object(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Arguments are not valid JSON: {exc}") from exc
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return arguments


def parse_move_args(arguments: Any) -> int:
    try:
        parsed = MoveArgs.model_validate(_load_object(arguments))
    except ValidationError as exc:
        raise ValueError(_format_validation(exc)) from exc
    return parsed.steps


def _format_validation(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts) or "Invalid arguments."
