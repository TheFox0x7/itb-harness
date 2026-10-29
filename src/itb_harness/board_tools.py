"""Observation, movement, and overlay tools exposed to the model."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class LevelMapCell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row: int = Field(ge=0)
    col: int = Field(ge=0)
    symbol: str = Field(min_length=1, max_length=8)


class LevelMapArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: str | list[str | list[str]] | None = None
    updates: list[LevelMapCell] = Field(default_factory=list)
    clear: bool = False


class ValidateSolutionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    moves: list[str] = Field(min_length=1, max_length=10_000)
    max_steps: int = Field(ge=1, le=10_000)
    spike_mode: str = Field(pattern="^(static|up|down)$")


_MAX_LEVEL_MAP_ROWS = 64
_MAX_LEVEL_MAP_COLS = 64
_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")

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


def _level_map_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Level rows. Each string may use bracketed cells like "
                    "\"[X] [X] [H]\" or plain cells like \"X X H\" or "
                    "\"XXH\". All rows must have the same number of cells."
                ),
            },
            "updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "row": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "0-based row index.",
                        },
                        "col": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "0-based column index.",
                        },
                        "symbol": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 1,
                            "description": "New single-character cell value.",
                        },
                    },
                    "required": ["row", "col", "symbol"],
                    "additionalProperties": False,
                },
                "description": "Update individual cells in the existing map.",
            },
            "clear": {
                "type": "boolean",
                "description": "Clear the current map before applying changes.",
            },
        },
        "additionalProperties": False,
    }


def _validate_solution_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "moves": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "move_up",
                        "move_down",
                        "move_left",
                        "move_right",
                    ],
                },
                "minItems": 1,
                "description": "The proposed sequence of moves to validate.",
            },
            "max_steps": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10000,
                "description": "The level's will/step limit.",
            },
            "spike_mode": {
                "type": "string",
                "enum": ["static", "up", "down"],
                "description": (
                    "static = spikes stay extended; up = start extended and "
                    "alternate each move; down = start hidden and alternate "
                    "each move. Hidden spikes cost 2 will. Alternating "
                    "extended spikes cost 1 because they retract before entry."
                ),
            },
        },
        "required": ["moves", "max_steps", "spike_mode"],
        "additionalProperties": False,
    }


def tool_schemas() -> list[dict[str, Any]]:
    """Return the tool definitions visible to the model."""

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
            "level_map",
            "Create or update your own text map of the level. This does not "
            "change the game; it is a persistent scratchpad so you can model "
            "walls and entity positions. Symbols: X=wall/blocked, .=open "
            "floor, H=Helltaker, S=skeleton, T=spike trap, D=demon, R=rock, "
            "K=key, B=barrier. Call it after moving or observing to keep the "
            "map current. All rows must have the same width. The map may "
            "extend beyond the visible 9-column grid.",
            _level_map_schema(),
        ),
        _tool(
            "validate_solution",
            "Validate a proposed move sequence against the current level_map. "
            "This does not move the character. The sequence must reach a tile "
            "adjacent to D, obey max_steps, and contain no blocked moves. "
            "Stepping on K removes all B barriers. spike_mode controls "
            "whether spikes stay extended or alternate.",
            _validate_solution_schema(),
        ),
        _tool(
            "execute_plan",
            "Validate a move sequence and then execute it on the real game. "
            "Only movement steps are allowed in a plan; press_space cannot be "
            "part of one. Execution stops if a move is blocked or the level "
            "map diverges.",
            _validate_solution_schema(),
        ),
        _tool(
            "press_space",
            "Press the spacebar. Use this only for dialogue, confirmations, "
            "and menus; pressing space outside dialogue does nothing. It is "
            "not a movement step and cannot be included in a plan.",
        ),
        _tool(
            "toggle_map_overlay",
            "Show or hide the current level_map symbols on screenshots "
            "returned to you. This helps revalidate map rows visually without "
            "changing the game or the map. Requires a level_map when enabling.",
            {
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "True to show map markers, false to hide.",
                    }
                },
                "required": ["enabled"],
                "additionalProperties": False,
            },
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


def parse_level_map_args(
    arguments: Any,
    current: list[list[str]] | None = None,
) -> list[list[str]] | None:
    """Apply a level_map tool call and return the new map, or None when clear."""

    try:
        parsed = LevelMapArgs.model_validate(_load_object(arguments))
    except ValidationError as exc:
        raise ValueError(_format_validation(exc)) from exc

    map_rows: list[list[str]] | None = None
    if current is not None and not parsed.clear:
        map_rows = [row[:] for row in current]

    if parsed.rows is not None:
        map_rows = _parse_level_rows(parsed.rows) or None

    if map_rows is None:
        if parsed.updates:
            raise ValueError(
                "Call level_map with rows before updating individual cells."
            )
        if parsed.clear or parsed.rows is not None:
            return None
        raise ValueError("Provide rows, updates, or clear=true.")

    for update in parsed.updates:
        if update.row >= len(map_rows):
            raise ValueError(
                f"Level-map row {update.row} is outside the map "
                f"(0..{len(map_rows) - 1})."
            )
        width = len(map_rows[update.row])
        if update.col >= width:
            raise ValueError(
                f"Level-map column {update.col} is outside row "
                f"{update.row} (0..{width - 1})."
            )
        map_rows[update.row][update.col] = _normalize_symbol(update.symbol)

    return map_rows


_MOVE_DELTAS = {
    "move_up": (-1, 0),
    "move_down": (1, 0),
    "move_left": (0, -1),
    "move_right": (0, 1),
}
_EMPTY = "."
_HELLTAKER = "H"
_SKELETON = "S"
_ROCK = "R"
_WALL = "X"
_SPIKE = "T"
_KEY = "K"
_BARRIER = "B"


def apply_move_to_level_map(
    rows: list[list[str]] | None,
    direction: str,
) -> list[list[str]] | None:
    """Return the map after a successful Helltaker move, or None if unchanged.

    Rules:
    - empty destination: move H there and clear the old cell;
    - key: pick it up, remove it, and remove all barriers;
    - skeleton: kick it one cell further, or remove it into an obstacle;
    - rock: push it one cell only when the next cell is empty or spikes.

    Walls, barriers, spikes, demons, and out-of-bounds outcomes do not change
    the map because the move should have been reported as blocked.
    """

    if not rows:
        return None
    delta = _MOVE_DELTAS.get(direction)
    if delta is None:
        return None

    updated = [row[:] for row in rows]
    player = _find_symbol(updated, _HELLTAKER)
    if player is None:
        return None
    player_row, player_col = player
    row, col = player_row + delta[0], player_col + delta[1]
    if not _inside(updated, row, col):
        return None

    destination = updated[row][col].upper()
    if destination in {_EMPTY, _SPIKE}:
        updated[row][col] = _HELLTAKER
        updated[player_row][player_col] = _EMPTY
        return updated

    if destination == _KEY:
        updated[row][col] = _HELLTAKER
        updated[player_row][player_col] = _EMPTY
        for cells in updated:
            for index, cell in enumerate(cells):
                if cell.upper() == _BARRIER:
                    cells[index] = _EMPTY
        return updated

    if destination == _BARRIER:
        return None

    if destination == _SKELETON:
        next_row, next_col = row + delta[0], col + delta[1]
        if (
            not _inside(updated, next_row, next_col)
            or updated[next_row][next_col].upper()
            in {_WALL, _ROCK, _BARRIER, _KEY}
        ):
            # The skeleton is destroyed against the obstacle.
            updated[row][col] = _HELLTAKER
        else:
            updated[next_row][next_col] = _SKELETON
            updated[row][col] = _HELLTAKER
        updated[player_row][player_col] = _EMPTY
        return updated

    if destination == _ROCK:
        next_row, next_col = row + delta[0], col + delta[1]
        if (
            _inside(updated, next_row, next_col)
            and updated[next_row][next_col].upper()
            in {_EMPTY, _SPIKE}
        ):
            updated[next_row][next_col] = _ROCK
            updated[row][col] = _HELLTAKER
            updated[player_row][player_col] = _EMPTY
            return updated
        return None

    return None


def _find_symbol(
    rows: list[list[str]], symbol: str
) -> tuple[int, int] | None:
    for row, cells in enumerate(rows):
        for col, cell in enumerate(cells):
            if cell.upper() == symbol:
                return row, col
    return None


def _inside(rows: list[list[str]], row: int, col: int) -> bool:
    return 0 <= row < len(rows) and 0 <= col < len(rows[row])


def parse_validate_solution_args(
    arguments: Any,
) -> tuple[list[str], int, str]:
    """Parse moves, max_steps, and spike_mode for validate_solution."""

    try:
        parsed = ValidateSolutionArgs.model_validate(_load_object(arguments))
    except ValidationError as exc:
        raise ValueError(_format_validation(exc)) from exc
    return parsed.moves, parsed.max_steps, parsed.spike_mode


def format_level_map(rows: list[list[str]]) -> str:
    """Render a level map in the bracketed grid format."""

    return "\n".join(
        " ".join(f"[{cell}]" for cell in row)
        for row in rows
    )


def _parse_level_rows(
    value: str | list[str | list[str]],
) -> list[list[str]]:
    if isinstance(value, str):
        raw_rows: list[str | list[str]] = [
            line for line in value.splitlines() if line.strip()
        ]
    elif isinstance(value, list):
        raw_rows = value
    else:
        raise ValueError("Level-map rows must be a string or a list.")

    if not raw_rows:
        return []

    rows: list[list[str]] = []
    for raw_row in raw_rows:
        if isinstance(raw_row, str):
            row = _split_level_row(raw_row)
        elif isinstance(raw_row, list):
            row = [_normalize_symbol(cell) for cell in raw_row]
        else:
            raise ValueError(
                "Each level-map row must be a string or a list of cells."
            )
        if not row:
            raise ValueError("Level-map rows cannot be empty.")
        rows.append(row)

    if not rows:
        return []

    width = len(rows[0])
    for index, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(
                "All level-map rows must have the same width; "
                f"row 0 has {width} cells but row {index} has {len(row)}."
            )
    if len(rows) > _MAX_LEVEL_MAP_ROWS:
        raise ValueError(
            f"Level map cannot exceed {_MAX_LEVEL_MAP_ROWS} rows."
        )
    if width > _MAX_LEVEL_MAP_COLS:
        raise ValueError(
            f"Level map cannot exceed {_MAX_LEVEL_MAP_COLS} columns."
        )
    return rows


def _split_level_row(line: str) -> list[str]:
    text = line.strip()
    if not text:
        return []
    bracketed = _BRACKET_RE.findall(text)
    if bracketed:
        return [_normalize_symbol(token) for token in bracketed]
    if any(char.isspace() for char in text):
        return [_normalize_symbol(token) for token in text.split()]
    return [_normalize_symbol(char) for char in text]


def _normalize_symbol(token: str) -> str:
    symbol = str(token).strip()
    match = _BRACKET_RE.fullmatch(symbol)
    if match is not None:
        symbol = match.group(1).strip()
    if len(symbol) != 1 or symbol.isspace():
        raise ValueError(
            "Level-map cells must be single non-space characters; "
            f"got {token!r}."
        )
    if symbol == "?":
        raise ValueError(
            "Level maps no longer support '?' cells; use only observed cells."
        )
    return symbol


def _format_validation(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts) or "Invalid arguments."
