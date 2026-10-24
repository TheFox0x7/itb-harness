"""Tool definitions and stubs for the Into the Breach action space.

The model sees only the screenshot.  These tools are the action side of the
loop: the harness executes a tool call, then captures a fresh frame if the
action changed the game.

For now the actual game action backend is a stub.  The schemas, validation, and
result plumbing are real so the tool-calling loop can be wired up and tested
before Lua/UI input is connected.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class Coord(BaseModel):
    """A board tile, with (0, 0) at the top-left."""

    model_config = ConfigDict(extra="forbid")

    x: int = Field(ge=0, le=7, description="Column, 0 = leftmost.")
    y: int = Field(ge=0, le=7, description="Row, 0 = topmost.")


class MoveMechArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mech: int = Field(ge=1, le=3, description="1-based mech slot.")
    target: Coord = Field(description="Destination tile on the board.")


class PrepareAttackArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mech: int = Field(ge=1, le=3, description="1-based mech slot.")
    weapon: int = Field(
        ge=1,
        le=3,
        description="1-based weapon slot in that mech's panel.",
    )


class ExecuteAttackArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: Coord = Field(description="Tile to fire at.")


class InspectArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: Coord = Field(description="Tile to inspect visually.")


class EmptyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True, slots=True)
class ActionResult:
    ok: bool
    message: str = ""
    error: str | None = None
    image_png: bytes | None = None
    state_changed: bool = False
    pixel: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class ToolResult:
    ok: bool
    message: str = ""
    error: str | None = None
    image_png: bytes | None = None
    state_changed: bool = False
    pixel: tuple[int, int] | None = None

    @classmethod
    def from_action(cls, result: ActionResult) -> "ToolResult":
        return cls(
            ok=result.ok,
            message=result.message,
            error=result.error,
            image_png=result.image_png,
            state_changed=result.state_changed,
            pixel=result.pixel,
        )

    def payload(self) -> dict[str, Any]:
        """Small structured result for logs and tool messages."""

        payload: dict[str, Any] = {"ok": self.ok}
        if self.message:
            payload["message"] = self.message
        if self.error:
            payload["error"] = self.error
        return payload

    def as_content(self) -> str | list[dict[str, Any]]:
        """Mistral tool-message content.

        Mistral accepts either a string or a list of content chunks.  The
        latter lets an inspection tool return an image without leaking state.
        """

        text = json.dumps(self.payload())
        if not self.image_png:
            return text

        import base64

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


class GameActions(Protocol):
    """The actual game-side implementation, injected into ToolRegistry."""

    def move_mech(self, mech: int, target: Coord) -> ActionResult: ...

    def prepare_attack(self, mech: int, weapon: int) -> ActionResult: ...

    def execute_attack(self, target: Coord) -> ActionResult: ...

    def end_turn(self) -> ActionResult: ...

    def reset_turn(self) -> ActionResult: ...

    def inspect(self, target: Coord) -> ActionResult: ...


class StubActions:
    """Validating, dry-run action backend.

    It enforces the tool-call sequence (an attack must be prepared before it is
    executed) but cannot know whether a move or attack is legal on the board.
    Replace this with a Lua/UI backend later; the tool layer does not change.
    """

    def __init__(self, dry_run: bool = True) -> None:
        self.dry_run = dry_run
        self.prepared: tuple[int, int] | None = None

    def _accepted(self, message: str, *, state_changed: bool = False) -> ActionResult:
        return ActionResult(
            ok=True,
            message=f"dry-run: {message}" if self.dry_run else message,
            state_changed=state_changed,
        )

    def _unavailable(self) -> ActionResult:
        return ActionResult(
            ok=False,
            error="action_backend_unavailable",
            message="No game action backend is connected yet.",
        )

    def move_mech(self, mech: int, target: Coord) -> ActionResult:
        if not self.dry_run:
            return self._unavailable()
        return self._accepted(f"would move mech {mech} to ({target.x}, {target.y})")

    def prepare_attack(self, mech: int, weapon: int) -> ActionResult:
        if not self.dry_run:
            return self._unavailable()
        self.prepared = (mech, weapon)
        return self._accepted(f"would prepare mech {mech} weapon {weapon}")

    def execute_attack(self, target: Coord) -> ActionResult:
        if self.prepared is None:
            return ActionResult(
                ok=False,
                error="no_prepared_attack",
                message="Call prepare_attack before execute_attack.",
            )
        if not self.dry_run:
            return self._unavailable()
        mech, weapon = self.prepared
        self.prepared = None
        return self._accepted(
            f"would fire mech {mech} weapon {weapon} at ({target.x}, {target.y})"
        )

    def end_turn(self) -> ActionResult:
        self.prepared = None
        if not self.dry_run:
            return self._unavailable()
        return self._accepted("would end the current turn")

    def reset_turn(self) -> ActionResult:
        self.prepared = None
        if not self.dry_run:
            return self._unavailable()
        return self._accepted("would reset the current turn")

    def inspect(self, target: Coord) -> ActionResult:
        # Deliberately image-oriented: inspection must not become a backdoor
        # for feeding structured game state to the model.
        return ActionResult(
            ok=False,
            error="inspect_unavailable",
            message=(
                "Visual inspection is not wired up yet; it needs the board "
                "grid calibration to crop the last screenshot."
            ),
        )


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[BaseModel], ActionResult]
    kind: Literal["action", "observation"] = "action"

    def parameters(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        return _inline_refs(schema)

    def as_mistral(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters(),
            },
        }


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Flatten Pydantic's ``$defs``/``$ref`` into one self-contained schema."""

    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                return walk(copy.deepcopy(defs[name]))
            return {key: walk(value) for key, value in node.items()}
        if isinstance(node, list):
            return [walk(value) for value in node]
        return node

    return walk(schema)


class ToolRegistry:
    """Maps model tool calls onto a GameActions implementation."""

    def __init__(self, actions: GameActions | None = None) -> None:
        self.actions: GameActions = actions or StubActions()
        self._tools: dict[str, ToolSpec] = {
            spec.name: spec for spec in self._build_tools()
        }

    def _build_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="move_mech",
                description=(
                    "Move one of your mechs to a board tile. The tile is the "
                    "destination. Fails if the move is not legal, the mech is "
                    "disabled, or it is not the player's turn."
                ),
                args_model=MoveMechArgs,
                handler=lambda args: self.actions.move_mech(
                    args.mech, args.target  # type: ignore[attr-defined]
                ),
            ),
            ToolSpec(
                name="prepare_attack",
                description=(
                    "Select a mech and one of its weapons, arming it for an "
                    "attack. This does not choose a target or fire."
                ),
                args_model=PrepareAttackArgs,
                handler=lambda args: self.actions.prepare_attack(
                    args.mech, args.weapon  # type: ignore[attr-defined]
                ),
            ),
            ToolSpec(
                name="execute_attack",
                description=(
                    "Fire the currently prepared weapon at a board tile. "
                    "Requires a successful prepare_attack first."
                ),
                args_model=ExecuteAttackArgs,
                handler=lambda args: self.actions.execute_attack(
                    args.target  # type: ignore[attr-defined]
                ),
            ),
            ToolSpec(
                name="end_turn",
                description=(
                    "End the current player turn. This commits all actions "
                    "taken this turn."
                ),
                args_model=EmptyArgs,
                handler=lambda args: self.actions.end_turn(),
            ),
            ToolSpec(
                name="reset_turn",
                description=(
                    "Reset the current player turn, undoing all moves and "
                    "attacks made this turn."
                ),
                args_model=EmptyArgs,
                handler=lambda args: self.actions.reset_turn(),
            ),
            ToolSpec(
                name="inspect",
                description=(
                    "Return a zoomed image crop around one board tile. This "
                    "is visual inspection only and never exposes hidden game "
                    "state. Use it to resolve details too small in the full "
                    "screenshot."
                ),
                args_model=InspectArgs,
                handler=lambda args: self.actions.inspect(
                    args.target  # type: ignore[attr-defined]
                ),
                kind="observation",
            ),
        ]

    def specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def spec(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def mistral_tools(self) -> list[dict[str, Any]]:
        return [spec.as_mistral() for spec in self._tools.values()]

    def execute(self, name: str, arguments: Any) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(
                ok=False,
                error="unknown_tool",
                message=f"Unknown tool {name!r}.",
            )

        try:
            parsed = _parse_arguments(arguments)
        except ValueError as exc:
            return ToolResult(ok=False, error="invalid_arguments", message=str(exc))

        try:
            args = spec.args_model.model_validate(parsed)
        except ValidationError as exc:
            return ToolResult(
                ok=False,
                error="invalid_arguments",
                message=_format_validation_error(exc),
            )

        try:
            result = spec.handler(args)
        except Exception as exc:  # keep the model loop alive on backend bugs
            return ToolResult(
                ok=False,
                error="tool_failed",
                message=f"{spec.name} failed: {exc}",
            )

        return ToolResult.from_action(result)


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        if not arguments.strip():
            return {}
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Tool arguments are not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments must be a JSON object.")
        return parsed
    raise ValueError("Tool arguments must be a JSON object or JSON string.")


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts) or "Invalid arguments."
