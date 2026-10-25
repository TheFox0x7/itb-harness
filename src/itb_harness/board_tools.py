"""Board, UI, and observation tools exposed to the model.

Coordinates are screenshot pixels, origin top-left.  The model no longer gets
an automatic screenshot; it must call view_board/hover/inspect/attack_order.
"""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .tools import ToolResult


class PointArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    point_2d: list[float] = Field(min_length=2, max_length=2)


class WeaponArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: int = Field(ge=1, le=2)


class SelectUnitArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: int = Field(ge=1, le=5)


class NoteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2000)


def _point_property(normalized: bool) -> dict[str, Any]:
    if normalized:
        return {
            "type": "array",
            "items": {"type": "number", "minimum": 0, "maximum": 1000},
            "minItems": 2,
            "maxItems": 2,
            "description": (
                "Normalized [x, y] in 0..1000, origin top-left."
            ),
        }
    return {
        "type": "array",
        "items": {"type": "integer", "minimum": 0},
        "minItems": 2,
        "maxItems": 2,
        "description": (
            "Screenshot pixel [x, y], origin top-left. Must be inside the "
            "current screenshot dimensions."
        ),
    }


def _point_schema(normalized: bool) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"point_2d": _point_property(normalized)},
        "required": ["point_2d"],
        "additionalProperties": False,
    }


_EMPTY_SCHEMA = {
    "type": "object",
    "properties": {},
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


def tool_schemas(
    include_note: bool = False, normalized: bool = False
) -> list[dict[str, Any]]:
    schemas = [
        _tool(
            "view_board",
            "Capture and return the current game screenshot. Call this first, "
            "and again after an action when you need to see the result.",
            _EMPTY_SCHEMA,
        ),
        _tool(
            "click",
            (
                "Move the pointer to a normalized 0..1000 coordinate and "
                "left-click."
                if normalized
                else "Move the pointer to a screenshot pixel coordinate and "
                "left-click."
            )
            + " Returns a screenshot with the click position marked, so you "
            "can see the immediate result. Use hover first to preview the "
            "tile or target.",
            _point_schema(normalized),
        ),
        _tool(
            "hover",
            (
                "Move the pointer to a normalized 0..1000 coordinate and "
                "capture a screenshot without holding Ctrl."
                if normalized
                else "Move the pointer to a screenshot pixel coordinate and "
                "capture a screenshot without holding Ctrl."
            )
            + " Use this to preview a tile, unit, or target before acting.",
            _point_schema(normalized),
        ),
        _tool(
            "inspect",
            (
                "Hover at a normalized 0..1000 coordinate while holding Left "
                "Ctrl and capture a screenshot with detailed tooltips."
                if normalized
                else "Hover at a screenshot pixel coordinate while holding "
                "Left Ctrl and capture a screenshot with detailed tooltips."
            )
            + " Use this to inspect units, terrain, and predicted attack "
            "results.",
            _point_schema(normalized),
        ),
        _tool(
            "attack_order",
            "Hold Left Alt and capture a screenshot showing the attack-order "
            "preview. Does not click or commit anything. Hover individual "
            "targets to inspect them before committing.",
            _EMPTY_SCHEMA,
        ),
        _tool(
            "repair",
            "Press R to activate repair mode. Requires a selected mech and an "
            "available repair; hover the selected mech to preview the repair, "
            "then click it as the target.",
            _EMPTY_SCHEMA,
        ),
        _tool(
            "weapon",
            "Press 1 or 2 to select that weapon. Afterwards click the target "
            "to fire. Hover the target first to preview possible attack "
            "results.",
            {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "enum": [1, 2],
                        "description": "Weapon slot to select.",
                    }
                },
                "required": ["number"],
                "additionalProperties": False,
            },
        ),
        _tool(
            "select_unit",
            "Select unit 1-5. Keys: 1=A, 2=S, 3=D, 4=Z, 5=X. Hover the "
            "selected unit to inspect it.",
            {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "Unit number 1-5.",
                    }
                },
                "required": ["number"],
                "additionalProperties": False,
            },
        ),
        _tool(
            "end_turn",
            "Press Space to end the player turn. Only valid during the "
            "player turn; commits the current turn.",
            _EMPTY_SCHEMA,
        ),
        _tool(
            "undo_move",
            "Press Left Shift to undo the last move. Only moves can be "
            "undone; once a unit attacks, that action is permanent.",
            _EMPTY_SCHEMA,
        ),
        _tool(
            "reset_turn",
            "Press Backspace to reset the turn. Available at most once per "
            "mission and not always available. Undoes all moves and attacks "
            "made this turn.",
            _EMPTY_SCHEMA,
        ),
        _tool(
            "cancel",
            "Right-click to cancel the current selection or action.",
            _EMPTY_SCHEMA,
        ),
    ]

    if include_note:
        schemas.append(
            _tool(
                "note",
                "Save a concise note for future runs. Use this for facts "
                "worth remembering about enemies, mechs, maps, or mistakes. "
                "It does not affect the current game.",
                {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 2000,
                            "description": "The note to persist.",
                        }
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
            )
        )
    return schemas


def _load_object(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Arguments are not valid JSON: {exc}") from exc
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return arguments


def _point_values(model: PointArgs, normalized: bool) -> tuple[float, float]:
    x, y = model.point_2d
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Coordinates must be finite numbers.")
    if normalized:
        if not 0.0 <= x <= 1000.0:
            raise ValueError(f"x={x} is outside 0..1000.")
        if not 0.0 <= y <= 1000.0:
            raise ValueError(f"y={y} is outside 0..1000.")
        return float(x), float(y)
    if x < 0 or y < 0:
        raise ValueError(f"Point ({x}, {y}) is outside the screenshot.")
    return float(x), float(y)


def normalized_to_pixels(
    x: float, y: float, width: int, height: int
) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Frame has no drawable area.")
    px = int(x / 1000.0 * max(width - 1, 0) + 0.5)
    py = int(y / 1000.0 * max(height - 1, 0) + 0.5)
    return max(0, min(width - 1, px)), max(0, min(height - 1, py))


def parse_point_args(
    arguments: Any, normalized: bool = False
) -> tuple[float, float]:
    try:
        parsed = PointArgs.model_validate(_load_object(arguments))
    except ValidationError as exc:
        raise ValueError(_format_validation(exc)) from exc
    return _point_values(parsed, normalized)


def parse_weapon_args(arguments: Any) -> int:
    try:
        parsed = WeaponArgs.model_validate(_load_object(arguments))
    except ValidationError as exc:
        raise ValueError(_format_validation(exc)) from exc
    return parsed.number


def parse_select_unit_args(arguments: Any) -> int:
    try:
        parsed = SelectUnitArgs.model_validate(_load_object(arguments))
    except ValidationError as exc:
        raise ValueError(_format_validation(exc)) from exc
    return parsed.number


def parse_note_args(arguments: Any) -> str:
    try:
        parsed = NoteArgs.model_validate(_load_object(arguments))
    except ValidationError as exc:
        raise ValueError(_format_validation(exc)) from exc
    return parsed.text.strip()


def _format_validation(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts) or "Invalid arguments."


def tool_error(error: str, message: str) -> ToolResult:
    return ToolResult(ok=False, error=error, message=message)
