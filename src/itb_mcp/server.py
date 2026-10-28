"""MCP server exposing Helltaker game tools, resources, and prompts."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, Literal

from mcp.server.mcpserver import Image as MCPImage
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import (
    ResourceError,
    ResourceNotFoundError,
    ToolError,
)

from itb_harness.capture import (
    BackendUnavailable,
    CaptureError,
    Frame,
    get_backend,
)
from itb_harness.cli import format_windows
from itb_harness.input import InputError

from .boxes import BoxItem, to_annotations
from .config import Config
from .game_state import GameState, WindowSelectionError
from .prompts import RULES, STRATEGY, WORKFLOW, level_guide
from .runs import (
    RunNotFound,
    actions_payload,
    conversation_text,
    list_runs,
    load_run,
    safe_frame_path,
    safe_run_dir,
)

INSTRUCTIONS = """\
Play Helltaker through these tools.  The server owns screenshots, overlays,
move history, and optional run recording; you own planning and conversation.

Start with `view_board`, then plan a route to the demon before moving.  Each
`move` presses one arrow key and returns a fresh screenshot plus a BLOCKED or
successful result.  Use `toggle_grid` / `toggle_walkability` when the board is
hard to read, and `set_boxes` to mark the demon, yourself, and obstacles.
Call `start_run` before a serious attempt if you want the frames and actions
recorded for later review.
"""


@contextmanager
def _expected_tool_errors():
    """Turn anticipated game/validation failures into readable tool errors."""

    try:
        yield
    except ToolError:
        raise
    except (
        ValueError,
        BackendUnavailable,
        CaptureError,
        InputError,
        WindowSelectionError,
    ) as exc:
        raise ToolError(str(exc) or type(exc).__name__) from exc


def _frame_result(
    frame: Frame,
    text: str,
) -> list[Any]:
    return [text, MCPImage(data=frame.png, format="png")]


def _overlay_summary(state: GameState) -> str:
    return (
        f"grid={state.show_grid}, "
        f"walkability={state.show_walkability}, "
        f"boxes={len(state.box_overlay)}"
    )


def _frame_text(prefix: str, frame: Frame, state: GameState) -> str:
    return (
        f"{prefix} Screenshot attached ({frame.width}x{frame.height}); "
        f"overlays: {_overlay_summary(state)}."
    )


def build_server(
    config: Config | None = None,
    state: GameState | None = None,
) -> MCPServer:
    """Create the MCP server.

    ``state`` can be injected for tests; otherwise a lazy :class:`GameState`
    is created with ``config``.
    """

    config = config or Config.from_env()
    state = state or GameState(config)

    @asynccontextmanager
    async def lifespan(_server: MCPServer):
        try:
            yield
        finally:
            state.close()

    server = MCPServer(
        name="itb-harness",
        title="Helltaker harness",
        description=(
            "Drive the Helltaker puzzle game: screenshots, movement, overlays, "
            "run recording, and replay resources."
        ),
        instructions=INSTRUCTIONS,
        version="0.1.0",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # Game tools
    # ------------------------------------------------------------------

    @server.tool(name="view_board", structured_output=False)
    def view_board() -> list[Any]:
        """Capture and return the current game screen.

        Overlays (boxes, grid, walkability) are applied based on the current
        server state and persist across calls.
        """

        with _expected_tool_errors(), state.lock:
            source = state.last_frame_path
            frame = state.capture()
            text = _frame_text("Current board captured.", frame, state)
            state.log_action(
                tool="view_board",
                message=text,
                source_frame_path=source,
            )
            return _frame_result(frame, text)

    @server.tool(name="move", structured_output=False)
    def move(
        direction: Literal["up", "down", "left", "right"],
    ) -> list[Any]:
        """Press one arrow key and return the resulting screenshot.

        The result says BLOCKED when the frame did not change (no step was
        consumed). After a BLOCKED move, do not repeat the same direction:
        try another direction or call view_board to re-assess.
        """

        with _expected_tool_errors(), state.lock:
            source = state.last_frame_path
            frame, blocked = state.move(direction)
            history = state.format_history()

            if blocked is True:
                text = (
                    f"Move {direction} was BLOCKED — no step consumed. "
                    "Do NOT repeat this move; choose another direction or "
                    f"call view_board. {history} "
                )
            elif blocked is False:
                text = f"Moved {direction} one tile. Step consumed. {history} "
            else:
                text = (
                    f"Pressed {direction}. No previous frame was available, "
                    f"so the result was not classified. {history} "
                )

            if state.input is not None and getattr(state.input, "name", "") == "stub":
                text += " (stub input: key press recorded but not sent.)"

            text = _frame_text(text.strip(), frame, state)
            state.log_action(
                tool="move",
                arguments={"direction": direction},
                message=text,
                blocked=blocked,
                state_changed=(None if blocked is None else not blocked),
                source_frame_path=source,
            )
            return _frame_result(frame, text)

    @server.tool(name="reset_level", structured_output=False)
    def reset_level() -> list[Any]:
        """Restart the current level by pressing R and clear the box overlay.

        The game may briefly show a title/loading screen. If the returned
        screenshot looks wrong, call view_board once more.
        """

        with _expected_tool_errors(), state.lock:
            source = state.last_frame_path
            frame = state.reset_level()
            prefix = "Level reset with R; bounding-box overlay cleared."
            if state.input is not None and getattr(state.input, "name", "") == "stub":
                prefix += " Stub input: key press recorded but not sent."
            text = _frame_text(prefix, frame, state)
            state.log_action(
                tool="reset_level",
                message=text,
                state_changed=True,
                source_frame_path=source,
            )
            return _frame_result(frame, text)

    @server.tool(name="toggle_grid", structured_output=False)
    def toggle_grid(enabled: bool = True) -> list[Any]:
        """Show or hide row/column labels on each tile (e.g. r3c2)."""

        with _expected_tool_errors(), state.lock:
            source = state.last_frame_path
            state.show_grid = enabled
            frame = state.capture()
            text = _frame_text(
                f"Grid overlay {'enabled' if enabled else 'disabled'}.", frame, state
            )
            state.log_action(
                tool="toggle_grid",
                arguments={"enabled": enabled},
                message=text,
                source_frame_path=source,
            )
            return _frame_result(frame, text)

    @server.tool(name="toggle_walkability", structured_output=False)
    def toggle_walkability(enabled: bool = True) -> list[Any]:
        """Show or hide walkability tinting (green walkable, red blocked)."""

        with _expected_tool_errors(), state.lock:
            source = state.last_frame_path
            state.show_walkability = enabled
            frame = state.capture()
            text = _frame_text(
                f"Walkability overlay {'enabled' if enabled else 'disabled'}.",
                frame,
                state,
            )
            state.log_action(
                tool="toggle_walkability",
                arguments={"enabled": enabled},
                message=text,
                source_frame_path=source,
            )
            return _frame_result(frame, text)

    def _apply_boxes(
        tool: str,
        boxes: list[BoxItem],
        *,
        replace: bool,
    ) -> list[Any]:
        with _expected_tool_errors(), state.lock:
            if state.current_frame is None:
                state.capture()
            frame = state.current_frame
            assert frame is not None
            annotations = to_annotations(
                boxes,
                width=frame.width,
                height=frame.height,
                config=state.config,
            )
            if replace:
                state.set_boxes(annotations)
            else:
                state.add_boxes(annotations)

            source = state.last_frame_path
            frame = state.capture()
            if replace:
                if annotations:
                    message = f"Overlay set with {len(annotations)} boxes."
                else:
                    message = "Overlay cleared."
            else:
                message = (
                    f"Added {len(annotations)} boxes; overlay now has "
                    f"{len(state.box_overlay)}."
                )
            text = _frame_text(message, frame, state)
            state.log_action(
                tool=tool,
                arguments={"boxes": [box.model_dump() for box in boxes]},
                message=text,
                source_frame_path=source,
            )
            return _frame_result(frame, text)

    @server.tool(name="set_boxes", structured_output=False)
    def set_boxes(boxes: list[BoxItem]) -> list[Any]:
        """Replace the persistent bounding-box overlay.

        Boxes stay on every future screenshot until the next set_boxes call.
        Coordinates are [xmin, ymin, xmax, ymax] in screenshot pixels. Pass an
        empty list to clear the overlay.
        """

        return _apply_boxes("set_boxes", boxes, replace=True)

    @server.tool(name="add_boxes", structured_output=False)
    def add_boxes(boxes: list[BoxItem]) -> list[Any]:
        """Append boxes to the persistent overlay without clearing it."""

        return _apply_boxes("add_boxes", boxes, replace=False)

    # ------------------------------------------------------------------
    # Window tools
    # ------------------------------------------------------------------

    @server.tool(name="list_windows")
    def list_windows() -> str:
        """List capturable windows so a target can be chosen."""

        with _expected_tool_errors(), state.lock:
            if state.backend is None:
                state.backend = get_backend(state.config.backend)
            target_id = state.target.id if state.target else None
            return format_windows(state.backend.list_windows(), target_id)

    @server.tool(name="select_window")
    def select_window(selector: str) -> str:
        """Choose the game window by index, id, app id, or title substring."""

        with _expected_tool_errors(), state.lock:
            target = state.select_window(selector)
            return (
                f"Target set to [{target.monitor or '?'}] {target.label} "
                f"({target.width}x{target.height}, id={target.id})."
            )

    # ------------------------------------------------------------------
    # Recording tools
    # ------------------------------------------------------------------

    @server.tool(name="start_run")
    def start_run(notes: str | None = None) -> str:
        """Start recording frames and actions for a new run.

        Call this before a serious attempt. Frames are saved on every capture;
        actions are written to actions.jsonl. Use end_run when finished.
        """

        with _expected_tool_errors(), state.lock:
            recorder = state.start_run(notes)
            return (
                f"Run started: {recorder.run_id}\n"
                f"Directory: {recorder.root}\n"
                "Every capture is saved as frames/NNNN.png. Call end_run when "
                "finished."
            )

    @server.tool(name="end_run")
    def end_run() -> str:
        """Stop recording the active run and return a summary."""

        with _expected_tool_errors(), state.lock:
            summary = state.end_run()
            return "Run ended:\n" + json.dumps(summary, indent=2)

    # ------------------------------------------------------------------
    # Notes tools
    # ------------------------------------------------------------------

    @server.tool(name="read_notes")
    def read_notes() -> str:
        """Read the persistent notes file."""

        text = state.notes.read()
        return text or "Notes are empty."

    @server.tool(name="append_notes")
    def append_notes(text: str, run_id: str | None = None) -> str:
        """Append a timestamped note for future runs.

        Use this for durable facts (level layouts, trap timing, dead ends).
        """

        with _expected_tool_errors():
            state.notes.append(text, run_id=run_id or state.run_id)
            return (
                f"Note appended to {state.notes.path} "
                f"({state.notes.char_count} chars loaded)."
            )

    @server.tool(name="write_notes")
    def write_notes(text: str) -> str:
        """Replace the whole notes file with new text."""

        with _expected_tool_errors():
            state.notes.write(text)
            return f"Notes replaced ({state.notes.char_count} chars loaded)."

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------

    @server.resource(
        "helltaker://state",
        name="Game state",
        description="Current frame, overlays, move history, and run state.",
        mime_type="application/json",
    )
    def state_resource() -> str:
        return json.dumps(state.snapshot(), indent=2)

    @server.resource(
        "helltaker://runs",
        name="Recorded runs",
        description="All runs under the configured runs directory.",
        mime_type="application/json",
    )
    def runs_resource() -> str:
        return json.dumps(list_runs(state.config.runs_dir), indent=2)

    @server.resource(
        "helltaker://runs/{run_id}",
        name="Run details",
        description="Metadata, actions, turns, and frame names for one run.",
        mime_type="application/json",
    )
    def run_resource(run_id: str) -> str:
        try:
            run_dir = safe_run_dir(state.config.runs_dir, run_id)
            return json.dumps(load_run(run_dir), indent=2)
        except RunNotFound as exc:
            raise ResourceNotFoundError(str(exc)) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ResourceError(f"Could not read run {run_id}: {exc}") from exc

    @server.resource(
        "helltaker://runs/{run_id}/frames/{name}",
        name="Run frame",
        description="One recorded screenshot as a PNG.",
        mime_type="image/png",
    )
    def frame_resource(run_id: str, name: str) -> bytes:
        try:
            path = safe_frame_path(state.config.runs_dir, run_id, name)
        except RunNotFound as exc:
            raise ResourceNotFoundError(str(exc)) from exc
        return path.read_bytes()

    @server.resource(
        "helltaker://runs/{run_id}/conversation",
        name="Run transcript",
        description=(
            "Human-readable action transcript. Reasoning lives in the MCP "
            "client, so this is reconstructed from actions.jsonl."
        ),
        mime_type="text/plain",
    )
    def conversation_resource(run_id: str) -> str:
        try:
            run_dir = safe_run_dir(state.config.runs_dir, run_id)
            return conversation_text(run_dir)
        except RunNotFound as exc:
            raise ResourceNotFoundError(str(exc)) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ResourceError(f"Could not read run {run_id}: {exc}") from exc

    @server.resource(
        "helltaker://runs/{run_id}/actions",
        name="Run actions",
        description="Structured action log for one run.",
        mime_type="application/json",
    )
    def actions_resource(run_id: str) -> str:
        try:
            run_dir = safe_run_dir(state.config.runs_dir, run_id)
        except RunNotFound as exc:
            raise ResourceNotFoundError(str(exc)) from exc
        return json.dumps(actions_payload(run_dir), indent=2)

    @server.resource(
        "helltaker://notes",
        name="Persistent notes",
        description="Notes shared between runs.",
        mime_type="text/markdown",
    )
    def notes_resource() -> str:
        return state.notes.read()

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    @server.prompt(
        name="helltaker-rules",
        description="Game mechanics and the available tools.",
    )
    def helltaker_rules() -> str:
        return RULES

    @server.prompt(
        name="helltaker-strategy",
        description="General puzzle-solving strategy for Helltaker.",
    )
    def helltaker_strategy() -> str:
        return STRATEGY

    @server.prompt(
        name="helltaker-workflow",
        description="The mandatory observe/plan/act workflow.",
    )
    def helltaker_workflow() -> str:
        return WORKFLOW

    @server.prompt(
        name="helltaker-level-guide",
        description="Known notes for a specific level, or general guidance.",
    )
    def helltaker_level_guide(level: int) -> str:
        return level_guide(level, state.notes.read())

    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="itb-mcp",
        description="Run the Helltaker MCP server.",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport (default: stdio).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--backend", default=None, help="Capture backend: auto, kwin, x11."
    )
    parser.add_argument("--input", default=None, help="Input backend: x11 or stub.")
    parser.add_argument(
        "-w",
        "--window",
        default=None,
        help="Target window: index, id, app id, or title substring.",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=None,
        help="Directory for recorded runs (default: runs).",
    )
    parser.add_argument(
        "--notes",
        type=Path,
        default=None,
        help="Persistent notes file (default: notes.md).",
    )
    parser.add_argument(
        "--action-delay",
        type=float,
        default=None,
        help="Seconds to wait after a key before capturing (default: 0.35).",
    )
    parser.add_argument(
        "--strict-box-labels",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Restrict box labels to the known Helltaker categories.",
    )
    parser.add_argument(
        "--scaled-boxes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Accept box coordinates in 1..1000 instead of pixels.",
    )
    parser.add_argument(
        "--list-windows",
        action="store_true",
        help="List capturable windows and exit.",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    config = Config.from_env()
    changes: dict[str, Any] = {}
    if args.backend is not None:
        changes["backend"] = args.backend
    if args.input is not None:
        changes["input"] = args.input
    if args.window is not None:
        changes["window"] = args.window
    if args.runs is not None:
        changes["runs_dir"] = args.runs.expanduser()
    if args.notes is not None:
        changes["notes_file"] = args.notes.expanduser()
    if args.action_delay is not None:
        changes["action_delay"] = max(0.0, args.action_delay)
    if args.strict_box_labels is not None:
        changes["strict_box_labels"] = args.strict_box_labels
    if args.scaled_boxes is not None:
        changes["scaled_boxes"] = args.scaled_boxes
    return config.replaced(**changes) if changes else config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)

    if args.list_windows:
        try:
            backend = get_backend(config.backend)
        except BackendUnavailable as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        try:
            print(format_windows(backend.list_windows()))
        finally:
            backend.close()
        return 0

    server = build_server(config)
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run(
            args.transport,
            host=args.host,
            port=args.port,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
