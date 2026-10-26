"""Observation, movement, and overlay tools exposed to the model."""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .annotate import BoxAnnotation


STRICT_BOX_LABELS = (
    "spike trap",
    "skeleton",
    "rock",
    "demon",
    "helltaker",
)


class BoxItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    box_2d: list[float] = Field(min_length=4, max_length=4)
    label: str = Field(min_length=1)


class BoxesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    boxes: list[BoxItem] = Field(default_factory=list)


_EMPTY_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def _tool(
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters or _EMPTY_SCHEMA,
        },
    }


def _boxes_schema(
    strict_labels: bool, scaled_boxes: bool = False
) -> dict[str, Any]:
    label: dict[str, Any] = {
        "type": "string",
        "minLength": 1,
        "description": "Short label describing the boxed object.",
    }
    if strict_labels:
        label["enum"] = list(STRICT_BOX_LABELS)
        label["description"] = (
            "One of: " + ", ".join(STRICT_BOX_LABELS) + "."
        )
    return {
        "type": "object",
        "properties": {
            "boxes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "box_2d": {
                            "type": "array",
                            "items": (
                                {
                                    "type": "number",
                                    "minimum": 1,
                                    "maximum": 1000,
                                }
                                if scaled_boxes
                                else {"type": "number"}
                            ),
                            "minItems": 4,
                            "maxItems": 4,
                            "description": (
                                "[xmin, ymin, xmax, ymax] in scaled "
                                "coordinates from 1 to 1000."
                                if scaled_boxes
                                else "[xmin, ymin, xmax, ymax] in screenshot "
                                "pixels."
                            ),
                        },
                        "label": label,
                    },
                    "required": ["box_2d", "label"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["boxes"],
        "additionalProperties": False,
    }


def tool_schemas(
    strict_labels: bool = False, scaled_boxes: bool = False
) -> list[dict[str, Any]]:
    """Return the tool definitions visible to the model."""

    coordinate_text = (
        "scaled coordinates from 1 to 1000"
        if scaled_boxes
        else "screenshot pixels"
    )
    return [
        _tool(
            "view_board",
            "Capture and return the current game screenshot. Call this first "
            "and after movement when you need to see the result.",
        ),
        _tool(
            "move_up",
            "Move one tile upward by pressing the Up arrow.",
        ),
        _tool(
            "move_down",
            "Move one tile downward by pressing the Down arrow.",
        ),
        _tool(
            "move_left",
            "Move one tile left by pressing the Left arrow.",
        ),
        _tool(
            "move_right",
            "Move one tile right by pressing the Right arrow.",
        ),
        _tool(
            "reset_level",
            "Restart the current level by pressing R. Use this when a plan "
            "fails or you are stuck, then try a better plan.",
        ),
        _tool(
            "toggle_grid",
            "Toggle a coordinate grid overlay on screenshots. When enabled, "
            "each tile is labeled with its row and column (e.g. r3c2) so you "
            "can reference positions precisely. Pass {\"enabled\": true} to "
            "turn on, {\"enabled\": false} to turn off.",
            {
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "True to show the grid, false to hide it.",
                    }
                },
                "required": ["enabled"],
                "additionalProperties": False,
            },
        ),
        _tool(
            "toggle_walkability",
            "Toggle a walkability overlay on screenshots. When enabled, "
            "walkable tiles are tinted green and blocked tiles are tinted "
            "red, based on floor color analysis. This helps you see which "
            "areas are open paths. Pass {\"enabled\": true} to turn on, "
            "{\"enabled\": false} to turn off.",
            {
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "True to show walkability, false to hide it.",
                    }
                },
                "required": ["enabled"],
                "additionalProperties": False,
            },
        ),
        _tool(
            "set_boxes",
            "Replace the persistent bounding-box overlay on screenshots. "
            f"Coordinates are {coordinate_text} as "
            "[xmin, ymin, xmax, ymax]. The overlay remains on every future "
            "screenshot until the next set_boxes or add_boxes call.",
            _boxes_schema(strict_labels, scaled_boxes),
        ),
        _tool(
            "add_boxes",
            "Append bounding boxes to the persistent overlay without removing "
            f"existing boxes. Coordinates are {coordinate_text} as "
            "[xmin, ymin, xmax, ymax].",
            _boxes_schema(strict_labels, scaled_boxes),
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


def _scaled_to_pixel(value: float, size: int) -> int:
    if size <= 1:
        return 0
    pixel = int(round((value - 1) / 999.0 * (size - 1)))
    return max(0, min(size - 1, pixel))


def parse_boxes_args(
    arguments: Any,
    strict_labels: bool = False,
    scaled_boxes: bool = False,
    width: int | None = None,
    height: int | None = None,
) -> list[BoxAnnotation]:
    try:
        parsed = BoxesArgs.model_validate(_load_object(arguments))
    except ValidationError as exc:
        raise ValueError(_format_validation(exc)) from exc

    boxes: list[BoxAnnotation] = []
    for item in parsed.boxes:
        xmin, ymin, xmax, ymax = item.box_2d
        if not all(math.isfinite(value) for value in item.box_2d):
            raise ValueError("Bounding-box coordinates must be finite.")
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(
                "Bounding box must satisfy xmin < xmax and ymin < ymax."
            )
        if scaled_boxes:
            if width is None or height is None:
                raise ValueError(
                    "Scaled bounding boxes require screenshot dimensions."
                )
            if not all(1 <= value <= 1000 for value in item.box_2d):
                raise ValueError(
                    "Scaled bounding-box coordinates must be in 1..1000."
                )
            xmin, ymin, xmax, ymax = (
                _scaled_to_pixel(xmin, width),
                _scaled_to_pixel(ymin, height),
                _scaled_to_pixel(xmax, width),
                _scaled_to_pixel(ymax, height),
            )
            if xmax <= xmin or ymax <= ymin:
                raise ValueError(
                    "Bounding box is too small after scaling."
                )
        label = item.label.strip()
        if not label:
            raise ValueError("Bounding-box label cannot be empty.")
        if strict_labels and label not in STRICT_BOX_LABELS:
            raise ValueError(
                "Bounding-box label must be one of: "
                + ", ".join(STRICT_BOX_LABELS)
            )
        boxes.append(
            BoxAnnotation(
                xmin=int(round(xmin)),
                ymin=int(round(ymin)),
                xmax=int(round(xmax)),
                ymax=int(round(ymax)),
                label=label,
            )
        )
    return boxes


def _format_validation(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts) or "Invalid arguments."
