"""Parse normalized point actions from the model's text response.

The model is asked for a JSON array like:

    [{"point_2d": [500, 250], "label": "click"}]

Coordinates are normalized to 0..1000, with (0, 0) at the top-left of the
screenshot.  The parser is deliberately a little forgiving (code fences,
single quotes, prose around the array), but it rejects unknown labels and
out-of-range coordinates instead of silently clicking somewhere wrong.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal


class PointParseError(ValueError):
    """The model response did not contain a valid point-action list."""


@dataclass(frozen=True, slots=True)
class PointAction:
    x: float
    y: float
    label: Literal["click", "hover"]

    @property
    def point_2d(self) -> list[float]:
        return [self.x, self.y]


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_UNQUOTED_KEY_RE = re.compile(
    r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)'
)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_POINT_KEYS = (
    "point_2d",
    "point",
    "points",
    "coordinate",
    "coordinates",
    "position",
    "target",
)
_LABEL_KEYS = ("label", "action", "type")
_LIST_KEYS = ("actions", "points", "inputs", "clicks")


def parse_point_actions(content: str) -> list[PointAction]:
    """Parse a model response into zero or more point actions."""

    if not content or not content.strip():
        raise PointParseError("The response was empty.")

    text = content.strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    candidates: list[Any] = []
    parsed = _try_parse(text)
    if parsed is not None:
        candidates.append(parsed)

    if not candidates:
        # A model may emit standalone dicts instead of an array.  Prefer
        # object extraction over array extraction: a dict contains an inner
        # point array like [30, 40], which is not an action list itself.
        objects = _extract_all_objects(text)
        if objects:
            parsed_objects = [_try_parse(obj) for obj in objects]
            parsed_objects = [item for item in parsed_objects if item is not None]
            if parsed_objects:
                candidates.append(parsed_objects)

    if not candidates:
        array = _extract_balanced(text, "[", "]")
        if array is not None:
            parsed = _try_parse(array)
            if parsed is not None:
                candidates.append(parsed)

    if not candidates:
        raise PointParseError(
            "Could not find a JSON point list. Return only an array like "
            '[{"point_2d": [500, 250], "label": "click"}].'
        )

    actions: list[PointAction] = []
    for payload in candidates:
        actions.extend(_actions_from_payload(payload))

    if not actions:
        # An explicit empty list is valid: the model has nothing to do.
        return []
    return actions


def point_to_pixels(
    action: PointAction, width: int, height: int
) -> tuple[int, int]:
    """Map normalized 0..1000 coordinates to image pixels."""

    if width <= 0 or height <= 0:
        raise ValueError("Frame has no drawable area.")
    # +0.5 makes the mapping deterministic at exact half-pixels.
    x = int(action.x / 1000.0 * max(width - 1, 0) + 0.5)
    y = int(action.y / 1000.0 * max(height - 1, 0) + 0.5)
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _try_parse(candidate: str) -> Any | None:
    candidate = candidate.strip()
    if not candidate:
        return None

    repaired = _repair_json(candidate)
    for text in (candidate, repaired):
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            pass

    return None


def _repair_json(text: str) -> str:
    # Models sometimes forget quotes around keys or leave trailing commas.
    text = _UNQUOTED_KEY_RE.sub(r'\1"\2"\3', text)
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    return text


def _actions_from_payload(payload: Any) -> list[PointAction]:
    if isinstance(payload, dict):
        for key in _LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return _normalize_many(value)
        return [_normalize_item(payload)]

    if isinstance(payload, list):
        return _normalize_many(payload)

    raise PointParseError(
        "Point response must be an object or a list of objects, "
        f"got {type(payload).__name__}."
    )


def _normalize_many(items: Iterable[Any]) -> list[PointAction]:
    actions: list[PointAction] = []
    for index, item in enumerate(items, start=1):
        try:
            actions.append(_normalize_item(item))
        except PointParseError as exc:
            raise PointParseError(f"action {index}: {exc}") from exc
    return actions


def _normalize_item(item: Any) -> PointAction:
    if not isinstance(item, dict):
        raise PointParseError("Each point must be an object.")

    point = None
    for key in _POINT_KEYS:
        if key in item:
            point = item[key]
            break

    if point is None:
        point = item

    if isinstance(point, dict):
        x_value = point.get("x")
        y_value = point.get("y")
    elif isinstance(point, (list, tuple)) and len(point) == 2:
        x_value, y_value = point
    else:
        raise PointParseError(
            'Expected "point_2d": [x, y] or an object with x and y.'
        )

    label_value = None
    for key in _LABEL_KEYS:
        if key in item:
            label_value = item[key]
            break
    if not isinstance(label_value, str):
        raise PointParseError('Missing string "label" ("click" or "hover").')

    label = label_value.strip().lower()
    if label not in {"click", "hover"}:
        raise PointParseError(
            f"Unknown label {label_value!r}; expected 'click' or 'hover'."
        )

    x = _number(x_value, "x")
    y = _number(y_value, "y")
    if not 0.0 <= x <= 1000.0:
        raise PointParseError(f"x={x} is outside 0..1000.")
    if not 0.0 <= y <= 1000.0:
        raise PointParseError(f"y={y} is outside 0..1000.")

    return PointAction(x=x, y=y, label=label)  # type: ignore[arg-type]


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PointParseError(f"{name} must be a number, got {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise PointParseError(f"{name} must be finite.")
    return number


def _extract_balanced(text: str, open_char: str, close_char: str) -> str | None:
    start = text.find(open_char)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    quote = ""
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
            continue
        if char in ("'", '"'):
            in_string = True
            quote = char
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _extract_all_objects(text: str) -> list[str]:
    objects: list[str] = []
    index = 0
    while True:
        start = text.find("{", index)
        if start < 0:
            break
        block = _extract_balanced(text[start:], "{", "}")
        if block is None:
            break
        objects.append(block)
        index = start + len(block)
    return objects
