"""Main observer/act loop.

The model is not given an automatic screenshot.  Its observation tools are:

    view_board()          capture and return the board
    click(point_2d)       click at screenshot pixel (x, y)
    hover(point_2d)       hover, wait, capture and return a plain screenshot
    inspect(point_2d)     hover with Left Ctrl, capture detailed tooltips

The loop executes one tool call per model turn, feeds the tool result back, and
lets the model decide what to do next.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .agent import ChatAgent, MistralAgent, ModelToolCall, ModelTurn
from .annotate import annotate_click, annotate_grid
from .grid import IsoGrid, calibrate_grid, default_grid, detect_drop_zone_tiles
from .board_tools import (
    parse_note_args,
    parse_point_args,
    parse_select_unit_args,
    parse_weapon_args,
    normalized_to_pixels,
    tool_schemas,
)
from .capture import (
    BackendUnavailable,
    CaptureBackend,
    CaptureError,
    Frame,
    WindowInfo,
    get_backend,
    resolve_window,
)
from .cli import format_windows
from .input import InputBackend, get_input_backend
from .notes import NotesStore
from .tools import ToolResult


UNIT_KEYS = {1: "a", 2: "s", 3: "d", 4: "z", 5: "x"}
INPUT_TOOLS = {
    "click",
    "hover",
    "inspect",
    "attack_order",
    "repair",
    "weapon",
    "select_unit",
    "end_turn",
    "undo_move",
    "reset_turn",
    "cancel",
}

SYSTEM_PROMPT = """\
You are controlling the game Into the Breach.

You do not receive screenshots automatically. Use the tools supplied with the
request:

- view_board: capture and return the current game screenshot.
- click: move to a __POINT__ and left-click. Returns a screenshot with the
  click position marked.
- hover: move to a __POINT__, wait briefly, and capture a screenshot without
  holding Ctrl. Use this to preview a tile, unit, or target before acting.
- inspect: hover at a __POINT__ while holding Left Ctrl, then capture and
  return a screenshot with detailed tooltips. Use this to inspect units,
  terrain, and predicted attack results.
- attack_order: hold Left Alt and capture the attack-order preview.
- repair: press R to activate repair mode, then hover and click the selected
  mech.
- weapon: press 1 or 2 to select a weapon, then hover the target to preview
  the attack before clicking.
- select_unit: select unit 1-5 (A/S/D/Z/X), then hover the selected unit to
  inspect it.
- end_turn, undo_move, reset_turn, cancel: turn and cancellation controls.

__COORDS__ Call view_board before your first click, and call view_board again
after a click when you need to see the result. Hover before committing an
attack; use inspect when you need detailed tooltips.

Before the mission you may see the squad selection screen and a difficulty
dialog. Click the center of the Start button in the top-right, wait for the
Difficulty Setup dialog, choose a difficulty (Normal is the default), then
click the Start button in the bottom-right of that dialog. These are UI
buttons, not board tiles; do not use tile-grid coordinates on menus. If a UI
click appears to do nothing, call view_board again before clicking again;
screen transitions can take a moment.

Use exactly one tool per turn. Do not describe an action in prose; call a tool.
If a tool returns an error, inspect the last screenshot and try a different
approach. Do not repeat a failed action blindly.

Game takes place on 4 islands you'll tackle them one at a time and can move to final game after beating two.
On the islands there are color coded zones. Red ones you can click and attempt to defend, grey ones are locked.
Other colors are not interactible.
To pick a level on an island click a red zone you want to play on then click on the smaller map to start the mission. This will take you to main game loop.

The main game has 3 phases:
- Deployment where you need to place your units on tan spots. 
Pay attention to building placements and enemies to try and get early advantage.
After you deploy all your units click confirm to deploy them. Then the enemy takes it's initial turn.
You can ONLY place the units on the tan spots. To deploy a unit you must click on the tile.
Clicking the tile again will swap units with the next one that is to be deployed or will allow you to change placement.
Yellow arrow under a unit means the unit will be deployed to that spot. A refresh icon indicates that mech will be swapped with one that is about to be deployed.

- Enemy turn. If you see that information in the screenshot you should wait and take another. You can't do anything while enemy takes it's turn.
- Your turn. Now you get to move your units. Each unit can move and attack in the same turn but an attack ends the turn.
Units can have multiple abilities to use on the lower bar of the screen. Hover or inspect them to learn more and plan to use the attacks optimally.
If you see red arrows from cracked ground, those are enemy spawn points. Leaving any unit on this spot will block the spawn and deal damage to the unit.

To attack with an unit, choose one of yours then use tool to select a weapon. Red tiles are the range of your weapon, you can hover over them to learn about effects of the attack.
Some weapons push units around which is indicated with arrows. Weapons may have additional effects as well.
To move a unit, select it and click any tile marked in lime overlay. You may undo the move at any time before attacking. Hover over the tile to see a preview of the move.

Use notes tool to note down things you learned or misunderstood about instructions so future iterations can be adjusted.

Your primary goal is to protect buildings and accomplish side objectives. Sometimes sacrifice of unit health or entire unit can be optimal,
but it will mean you lose the pilot and all the perks they had if mech is destroyed. Your power grid shows how many times you can lose a building in the run.
You lose the game it it reaches 0 bars. Grid defense is a chance to mitigate the damage to the grid completely, however it works per attack.
"""


def build_system_prompt(
    normalized: bool,
    snap_to_tile: bool = False,
    grid_overlay: bool = False,
) -> str:
    point = (
        "normalized coordinate in the range 0..1000"
        if normalized
        else "screenshot pixel coordinate"
    )
    coords = (
        "Coordinates are [x, y] normalized from 0 to 1000, with (0, 0) at "
        "the top-left and (1000, 1000) at the bottom-right."
        if normalized
        else "Coordinates are [x, y] in screenshot pixels, with (0, 0) at "
        "the top-left and the bottom-right at (width - 1, height - 1). The "
        "dimensions reported by view_board are authoritative; never infer "
        "coordinates from the monitor or window resolution."
    )
    notes: list[str] = []
    if snap_to_tile:
        notes.append(
            "Clicks are automatically snapped to the nearest board tile "
            "centre, so you do not need to be pixel-perfect when aiming at "
            "a tile. Aim for the general area of the tile you want."
        )
    if grid_overlay:
        notes.append(
            "On board screens, the screenshot has a tile grid overlay. Each "
            "tile shows its pixel coordinates (e.g. \"472,288\") at its "
            "centre. During deployment, valid drop-zone tiles are highlighted "
            "in gold — click on a highlighted tile to deploy. To click a "
            "tile, use the pixel coordinates shown at its centre. Menus and "
            "dialogs do not have the grid overlay; click their visible "
            "buttons normally."
        )
    suffix = ("\n\n" + "\n\n".join(notes)) if notes else ""
    return (
        SYSTEM_PROMPT.replace("__POINT__", point)
        .replace("__COORDS__", coords)
        + suffix
    )


@dataclass(slots=True)
class LoopConfig:
    max_steps: int = 20
    max_images: int = 4
    action_delay: float = 0.5
    show_thinking: bool = False
    normalized_coordinates: bool = False
    snap_to_tile: bool = False
    grid_overlay: bool = False
    console: bool = True
    out_dir: Path = field(default_factory=lambda: Path("runs"))
    model: str = "mistral-large-4"


@dataclass(slots=True)
class LoopResult:
    steps: int
    actions: int
    frames: int
    run_dir: Path
    stopped: str
    final_text: str | None = None


class Recorder:
    """Write frames, model turns, and tool results for one run."""

    def __init__(
        self,
        out_dir: Path,
        backend: CaptureBackend,
        input_backend: InputBackend,
        model: str,
        cache_key: str | None,
        notes: NotesStore | None,
        target: WindowInfo,
        config: LoopConfig,
    ) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.root = out_dir / f"{stamp}-agent"
        self.frames_dir = self.root / "frames"
        self.thinking_dir = self.root / "thinking"
        self.frames_dir.mkdir(parents=True, exist_ok=False)
        self.thinking_dir.mkdir(parents=True, exist_ok=True)
        self.frame_index = 0

        self.actions_path = self.root / "actions.jsonl"
        self.conversation_path = self.root / "conversation.jsonl"
        self._actions = self.actions_path.open("a", encoding="utf-8")
        self._conversation = self.conversation_path.open("a", encoding="utf-8")

        (self.root / "run.json").write_text(
            json.dumps(
                {
                    "backend": backend.name,
                    "input": input_backend.name,
                    "model": model,
                    "prompt_cache_key": cache_key,
                    "notes_file": str(notes.path) if notes is not None else None,
                    "normalized_coordinates": config.normalized_coordinates,
                    "snap_to_tile": config.snap_to_tile,
                    "grid_overlay": config.grid_overlay,
                    "max_steps": config.max_steps,
                    "max_images": config.max_images,
                    "action_delay": config.action_delay,
                    "started_at": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    "target": asdict(target),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def save_frame(self, frame: Frame) -> Path:
        self.frame_index += 1
        path = self.frames_dir / f"{self.frame_index:04d}.png"
        path.write_bytes(frame.png)
        return path

    def log_turn(self, step: int, turn: ModelTurn) -> None:
        record: dict[str, Any] = {
            "step": step,
            "content": turn.content,
            "thinking": turn.thinking,
            "usage": turn.usage,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": _jsonable(call.arguments),
                }
                for call in turn.tool_calls
            ],
        }
        if turn.thinking:
            path = self.thinking_dir / f"{step:04d}.txt"
            path.write_text(turn.thinking, encoding="utf-8")
            record["thinking_file"] = str(path.relative_to(self.root))
        self._conversation.write(json.dumps(record) + "\n")
        self._conversation.flush()

    def log_tool_call(
        self,
        step: int,
        call: ModelToolCall,
        result: ToolResult,
        frame_path: Path | None,
        source_frame_path: Path | None,
    ) -> None:
        record = {
            "step": step,
            "tool": call.name,
            "arguments": _jsonable(call.arguments),
            "ok": result.ok,
            "error": result.error,
            "message": result.message,
            "state_changed": result.state_changed,
            "pixel": list(result.pixel) if result.pixel is not None else None,
            "frame": (
                str(frame_path.relative_to(self.root))
                if frame_path is not None
                else None
            ),
            "source_frame": (
                str(source_frame_path.relative_to(self.root))
                if source_frame_path is not None
                else None
            ),
        }
        self._actions.write(json.dumps(record) + "\n")
        self._actions.flush()

    def close(self) -> None:
        self._actions.close()
        self._conversation.close()


class ThinkingPrinter:
    """Stream reasoning chunks to the terminal as they arrive."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.started = False

    def write(self, text: str) -> None:
        if not self.enabled or not text:
            return
        if not self.started:
            print("  thinking:", flush=True)
            self.started = True
        sys.stdout.write(text.replace("\n", "\n    "))
        sys.stdout.flush()

    def close(self) -> None:
        if self.started:
            print()


class HarnessLoop:
    """Tool-driven observe/act loop."""

    def __init__(
        self,
        backend: CaptureBackend,
        target: WindowInfo,
        agent: ChatAgent,
        input_backend: InputBackend,
        config: LoopConfig | None = None,
        notes: NotesStore | None = None,
    ) -> None:
        self.backend = backend
        self.target = target
        self.agent = agent
        self.input = input_backend
        self.notes = notes
        self.config = config or LoopConfig()
        self.recorder = Recorder(
            self.config.out_dir,
            backend,
            input_backend,
            getattr(agent, "model", "unknown"),
            getattr(agent, "prompt_cache_key", None),
            notes,
            target,
            self.config,
        )

        self.frame: Frame | None = None
        self.last_frame_path: Path | None = None
        self.action_count = 0
        self.step_index = 0
        self.final_text: str | None = None
        self._closed = False
        self.event_callback: Callable[[dict[str, Any]], None] | None = None
        self.grid: IsoGrid | None = None
        self._grid_calibrated = False
        self._drop_zone_tiles: list[tuple[float, float]] | None = None

        if self.config.console:
            print(f"Target  [{target.monitor or '?'}] {target.label}")
            print(f"Capture {backend.name}")
            print(f"Input   {input_backend.name}")
            if notes is not None:
                print(f"Notes   {notes.path} ({notes.char_count} chars loaded)")
            print(f"Output  {self.recorder.root}")

        system_prompt = build_system_prompt(
            self.config.normalized_coordinates,
            snap_to_tile=self.config.snap_to_tile,
            grid_overlay=self.config.grid_overlay,
        )
        if input_backend.name == "stub":
            system_prompt += (
                "\n\nInput is in stub/dry-run mode: actions are recorded but "
                "NOT sent to the game, so the board will not change."
            )
        if notes is not None:
            rendered_notes = notes.render_for_prompt()
            if rendered_notes:
                system_prompt += "\n\n" + rendered_notes

        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Start by calling view_board. Use one tool per turn."
                ),
            },
        ]

    def _emit(self, event_type: str, **data: Any) -> None:
        if self.event_callback is not None:
            self.event_callback({"type": event_type, **data})

    def _capture(self) -> Frame:
        frame = self.backend.capture_window(self.target.id)
        self.target = frame.window

        # Calibrate the grid lazily on captures until the board is found.
        # During deployment the tan drop-zone tiles are visible and provide a
        # good calibration.  Menus and dialogs do not have a board, so they
        # keep the fallback grid uncalibrated and get no misleading overlay.
        if (
            self.config.snap_to_tile or self.config.grid_overlay
        ) and (self.grid is None or not self._grid_calibrated):
            self._calibrate_grid(frame)

        # Apply the grid overlay only when a real board was detected.
        if (
            self.config.grid_overlay
            and self.grid is not None
            and self._grid_calibrated
        ):
            highlighted = None
            if self._drop_zone_tiles:
                highlighted = self._drop_zone_tiles
            frame = replace(
                frame,
                png=annotate_grid(
                    frame.png,
                    self.grid,
                    highlight_tiles=highlighted,
                ),
            )

        self.frame = frame
        path = self.recorder.save_frame(frame)
        self.last_frame_path = path
        self._emit(
            "frame",
            path=str(path.relative_to(self.recorder.root)),
            width=frame.width,
            height=frame.height,
        )
        if self.config.console:
            print(
                f"  frame {self.recorder.frame_index:04d}  "
                f"{frame.width}x{frame.height}  "
                f"[{frame.window.monitor or '?'}]  "
                f"{path.relative_to(self.recorder.root)}"
            )
        return frame

    def _calibrate_grid(self, frame: Frame) -> None:
        """Calibrate the isometric grid from the current screenshot."""
        grid = calibrate_grid(frame.png, frame.width, frame.height)
        if grid is not None:
            self.grid = grid
            self._grid_calibrated = True
            # Cache drop-zone tiles for the overlay highlight.
            self._drop_zone_tiles = detect_drop_zone_tiles(frame.png)
            if self.config.console:
                print(
                    f"  grid calibrated: origin=({grid.origin_x:.0f},"
                    f"{grid.origin_y:.0f}) tile={2*grid.tile_w2:.0f}x"
                    f"{2*grid.tile_h2:.0f} "
                    f"({len(self._drop_zone_tiles)} drop-zone tiles)"
                )
        else:
            # Fall back to the default grid (measured for 1280×720), but do
            # not draw it until a board is actually detected.
            self.grid = default_grid(frame.width, frame.height)
            self._grid_calibrated = False
            if self.config.console:
                print(
                    f"  grid default: origin=({self.grid.origin_x:.0f},"
                    f"{self.grid.origin_y:.0f}) tile="
                    f"{2*self.grid.tile_w2:.0f}x{2*self.grid.tile_h2:.0f}"
                )

    def _prune_images(self) -> None:
        # Keep the latest screenshots and replace older tool-message images
        # with text.  This changes the prompt prefix and therefore invalidates
        # prompt cache entries from that point onward; --max-images 0 keeps the
        # full append-only history for better caching.
        if self.config.max_images <= 0:
            return

        image_messages = []
        for message in self.messages:
            if message.get("role") != "tool":
                continue
            content = message.get("content")
            if isinstance(content, list) and any(
                isinstance(chunk, dict) and chunk.get("type") == "image_url"
                for chunk in content
            ):
                image_messages.append(message)

        for message in image_messages[: -self.config.max_images]:
            message["content"] = json.dumps(
                {"ok": True, "message": "[older screenshot omitted]"}
            )

    def _require_frame(self) -> Frame:
        if self.frame is None or self.last_frame_path is None:
            raise ValueError(
                "Call view_board before click, hover, or inspect."
            )
        return self.frame

    def _pause(self) -> None:
        if self.config.action_delay > 0:
            time.sleep(self.config.action_delay)

    def _validate_point(self, frame: Frame, x: int, y: int) -> None:
        if not (0 <= x < frame.width and 0 <= y < frame.height):
            raise ValueError(
                f"Point ({x}, {y}) is outside the screenshot "
                f"{frame.width}x{frame.height}."
            )

    # ------------------------------------------------------------------
    # Tile snapping
    # ------------------------------------------------------------------

    # Maximum distance (px) from a tile centre for snapping to apply.
    # Clicks farther away are assumed to target UI elements, not tiles.
    _SNAP_MAX_DIST = 60.0

    def _ensure_grid(self, frame: Frame) -> IsoGrid:
        """Return the calibrated grid, calibrating lazily if needed.

        Normally the grid is calibrated in ``_capture`` on the first
        frame.  This is a fallback for the case where a click arrives
        before any capture (shouldn't happen, but be safe).
        """
        if self.grid is not None:
            return self.grid
        self._calibrate_grid(frame)
        assert self.grid is not None
        return self.grid

    def _snap_point(
        self, frame: Frame, x: int, y: int
    ) -> tuple[int, int, int, int] | None:
        """Snap a screenshot pixel to the nearest tile centre.

        Returns ``(snapped_x, snapped_y, row, col)`` or ``None`` if the
        point is too far from any tile centre (likely a UI click).
        """
        if not self.config.snap_to_tile or not self._grid_calibrated:
            return None
        grid = self._ensure_grid(frame)
        sx, sy, row, col = grid.snap(x, y)
        dist = ((sx - x) ** 2 + (sy - y) ** 2) ** 0.5
        if dist > self._SNAP_MAX_DIST:
            return None
        return sx, sy, row, col

    def _resolve_point(
        self, call: ModelToolCall, frame: Frame
    ) -> tuple[int, int, str]:
        """Parse, convert, validate, and optionally snap a point argument."""
        x, y = parse_point_args(
            call.arguments,
            normalized=self.config.normalized_coordinates,
        )
        if self.config.normalized_coordinates:
            x, y = normalized_to_pixels(x, y, frame.width, frame.height)
        else:
            x, y = int(round(x)), int(round(y))
        self._validate_point(frame, x, y)

        snap_info = ""
        snapped = self._snap_point(frame, x, y)
        if snapped is not None:
            nx, ny, row, col = snapped
            if (nx, ny) != (x, y):
                snap_info = (
                    f" Snapped to tile ({row}, {col}) "
                    f"at pixel ({nx}, {ny})."
                )
                x, y = nx, ny
        return x, y, snap_info

    def _execute_tool(
        self, call: ModelToolCall
    ) -> tuple[ToolResult, Path | None]:
        try:
            if call.name == "note":
                if self.notes is None:
                    return (
                        ToolResult(
                            ok=False,
                            error="notes_disabled",
                            message="Notes are not enabled for this run.",
                        ),
                        None,
                    )
                text = parse_note_args(call.arguments)
                self.notes.append(text, run_id=self.recorder.root.name)
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            f"Saved note for future runs "
                            f"({len(text)} chars)."
                        ),
                    ),
                    None,
                )

            if call.name == "view_board":
                frame = self._capture()
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            "Current screenshot attached "
                            f"({frame.width}x{frame.height}). Valid "
                            f"coordinates are 0..{frame.width - 1} by "
                            f"0..{frame.height - 1}."
                        ),
                        image_png=frame.png,
                    ),
                    self.last_frame_path,
                )

            if call.name == "click":
                frame = self._require_frame()
                x, y, snap_info = self._resolve_point(call, frame)
                sx = frame.window.x + x
                sy = frame.window.y + y
                self.input.click(sx, sy)
                self.action_count += 1
                self._pause()
                frame = self._capture()
                marked_png = annotate_click(frame.png, x, y)
                self.last_frame_path.write_bytes(marked_png)
                self.frame = replace(frame, png=marked_png)
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            f"Clicked screenshot pixel ({x}, {y})."
                            f"{snap_info} Marked screenshot attached "
                            f"({frame.width}x{frame.height})."
                        ),
                        image_png=marked_png,
                        state_changed=True,
                        pixel=(x, y),
                    ),
                    self.last_frame_path,
                )

            if call.name == "hover":
                frame = self._require_frame()
                x, y, snap_info = self._resolve_point(call, frame)
                sx = frame.window.x + x
                sy = frame.window.y + y
                self.input.hover(sx, sy)
                self._pause()
                frame = self._capture()
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            f"Hovered screenshot pixel ({x}, {y}) and "
                            f"captured a {frame.width}x{frame.height} "
                            f"screenshot.{snap_info}"
                        ),
                        image_png=frame.png,
                        pixel=(x, y),
                    ),
                    self.last_frame_path,
                )

            if call.name == "inspect":
                frame = self._require_frame()
                x, y, snap_info = self._resolve_point(call, frame)
                sx = frame.window.x + x
                sy = frame.window.y + y
                self.input.key_down("ctrl")
                try:
                    self.input.hover(sx, sy)
                    self._pause()
                    frame = self._capture()
                finally:
                    self.input.key_up("ctrl")
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            f"Hovered screenshot pixel ({x}, {y}) with "
                            f"Left Ctrl and captured a {frame.width}x"
                            f"{frame.height} screenshot with details."
                            f"{snap_info}"
                        ),
                        image_png=frame.png,
                        pixel=(x, y),
                    ),
                    self.last_frame_path,
                )

            if call.name == "attack_order":
                self.input.key_down("alt")
                try:
                    self._pause()
                    frame = self._capture()
                finally:
                    self.input.key_up("alt")
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            "Attack-order preview attached "
                            f"({frame.width}x{frame.height}). Hover individual "
                            "targets to inspect them before committing."
                        ),
                        image_png=frame.png,
                    ),
                    self.last_frame_path,
                )

            if call.name == "repair":
                self.input.key("r")
                self._pause()
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            "Repair mode activated. Hover the selected mech "
                            "to preview the repair, then click it."
                        ),
                        state_changed=True,
                    ),
                    None,
                )

            if call.name == "weapon":
                number = parse_weapon_args(call.arguments)
                self.input.key(str(number))
                self._pause()
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            f"Weapon {number} selected. Hover the target to "
                            "preview possible results, then click it."
                        ),
                        state_changed=True,
                    ),
                    None,
                )

            if call.name == "select_unit":
                number = parse_select_unit_args(call.arguments)
                self.input.key(UNIT_KEYS[number])
                self._pause()
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            f"Selected unit {number} with key "
                            f"{UNIT_KEYS[number]!r}. Hover it to inspect."
                        ),
                        state_changed=True,
                    ),
                    None,
                )

            if call.name == "end_turn":
                self.input.key("space")
                self._pause()
                return (
                    ToolResult(
                        ok=True,
                        message="End-turn key pressed.",
                        state_changed=True,
                    ),
                    None,
                )

            if call.name == "undo_move":
                self.input.key("shift")
                self._pause()
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            "Undo-move key pressed. Only moves can be undone; "
                            "attacks are permanent."
                        ),
                        state_changed=True,
                    ),
                    None,
                )

            if call.name == "reset_turn":
                self.input.key("backspace")
                self._pause()
                return (
                    ToolResult(
                        ok=True,
                        message=(
                            "Reset-turn key pressed. This is available at "
                            "most once per mission and may be unavailable."
                        ),
                        state_changed=True,
                    ),
                    None,
                )

            if call.name == "cancel":
                self.input.right_click()
                self._pause()
                return (
                    ToolResult(
                        ok=True,
                        message="Right-clicked to cancel the current selection.",
                        state_changed=True,
                    ),
                    None,
                )

            return (
                ToolResult(
                    ok=False,
                    error="unknown_tool",
                    message=f"Unknown tool {call.name!r}.",
                ),
                None,
            )
        except ValueError as exc:
            message = str(exc)
            error = (
                "no_frame"
                if message.startswith("Call view_board")
                else "invalid_arguments"
            )
            return (
                ToolResult(ok=False, error=error, message=message),
                None,
            )
        except Exception as exc:
            return (
                ToolResult(
                    ok=False,
                    error="tool_failed",
                    message=str(exc),
                ),
                None,
            )

    def step_once(self) -> bool:
        """Run one model turn and at most one tool call.

        Returns False when the model stops calling tools.
        """

        self.step_index += 1
        step = self.step_index
        if self.config.console:
            print(f"\n[turn {step}]")
        self._emit("status", phase="model_waiting", step=step)
        printer = ThinkingPrinter(enabled=self.config.console)

        def on_thinking(text: str) -> None:
            printer.write(text)
            self._emit("thinking", text=text, step=step)

        def on_content(text: str) -> None:
            self._emit("content", text=text, step=step)

        try:
            turn = self.agent.complete(
                self.messages,
                tools=tool_schemas(
                    include_note=self.notes is not None,
                    normalized=self.config.normalized_coordinates,
                ),
                on_thinking=on_thinking,
                on_content=on_content,
            )
        finally:
            printer.close()

        self.recorder.log_turn(step, turn)
        if self.config.console and turn.thinking and not printer.started:
            if self.config.show_thinking:
                print(textwrap.indent(turn.thinking.strip(), "  "))
            else:
                print(f"  thinking saved -> thinking/{step:04d}.txt")
        if self.config.console and turn.usage:
            cached = turn.usage.get("cached_tokens")
            prompt_tokens = turn.usage.get("prompt_tokens")
            if cached is not None:
                print(
                    f"  cache: {cached}/"
                    f"{prompt_tokens if prompt_tokens is not None else '?'} "
                    "prompt tokens"
                )

        self.messages.append(_assistant_message(turn))

        self._emit(
            "turn",
            step=step,
            usage=turn.usage,
            content=turn.content,
            tool_call_count=len(turn.tool_calls),
        )

        if not turn.tool_calls:
            if self.config.console:
                print("  model returned no tool call; stopping")
            self._emit("status", phase="model_stopped", step=step)
            self.final_text = turn.content
            return False

        primary = turn.tool_calls[0]
        for extra in turn.tool_calls[1:]:
            extra_result = ToolResult(
                ok=False,
                error="one_tool_per_turn",
                message="Call exactly one tool per turn.",
            )
            self.messages.append(
                extra_result.as_tool_message(extra.id, extra.name)
            )
            self.recorder.log_tool_call(
                step,
                extra,
                extra_result,
                None,
                self.last_frame_path,
            )

        self._emit(
            "status", phase="tool", step=step, tool=primary.name
        )
        source_frame = self.last_frame_path
        result, next_frame = self._execute_tool(primary)
        if (
            self.input.name == "stub"
            and primary.name in INPUT_TOOLS
            and result.ok
        ):
            result = replace(
                result,
                state_changed=False,
                message=(
                    f"{result.message} "
                    "(stub input: action was recorded but not sent)"
                ),
            )
        self.messages.append(
            result.as_tool_message(primary.id, primary.name)
        )
        self.recorder.log_tool_call(
            step, primary, result, next_frame, source_frame
        )
        self._emit(
            "tool_result",
            step=step,
            tool=primary.name,
            ok=result.ok,
            error=result.error,
            message=result.message,
            pixel=list(result.pixel) if result.pixel is not None else None,
            frame=(
                str(next_frame.relative_to(self.recorder.root))
                if next_frame is not None
                else None
            ),
        )
        self._emit("status", phase="turn_done", step=step)

        if self.config.console:
            if result.ok:
                print(f"  tool {primary.name}: {result.message}")
            else:
                print(
                    f"  tool {primary.name}: error={result.error} "
                    f"{result.message}"
                )
        self._prune_images()
        return True

    def run(self, max_steps: int | None = None) -> LoopResult:
        limit = self.config.max_steps if max_steps is None else max_steps
        stopped = "max_steps"

        try:
            while limit is None or limit <= 0 or self.step_index < limit:
                if not self.step_once():
                    stopped = "model_stopped"
                    break
        except KeyboardInterrupt:
            stopped = "interrupted"
        finally:
            self.close()

        return LoopResult(
            steps=self.step_index,
            actions=self.action_count,
            frames=self.recorder.frame_index,
            run_dir=self.recorder.root,
            stopped=stopped,
            final_text=self.final_text,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.recorder.close()


def _assistant_message(turn: ModelTurn) -> dict[str, Any]:
    tool_calls = []
    for call in turn.tool_calls:
        arguments = call.arguments
        if not isinstance(arguments, str):
            arguments = json.dumps(_jsonable(arguments))
        tool_calls.append(
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": arguments},
            }
        )
    message: dict[str, Any] = {
        "role": "assistant",
        "content": turn.content,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _resolve_target(backend: CaptureBackend, spec: str) -> WindowInfo:
    return resolve_window(backend.list_windows(), spec)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="itb-agent",
        description="Run the tool-driven observer/act loop.",
    )
    parser.add_argument(
        "--backend", default="auto", help="Capture backend: auto, kwin, x11."
    )
    parser.add_argument(
        "-w",
        "--window",
        help="Target window: index, id, app id, or title substring.",
    )
    parser.add_argument(
        "--input",
        default="stub",
        help="Pointer input backend: stub (default) or x11.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs"),
        help="Directory for run output (default: runs).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help=(
            "Headless maximum model turns/tool calls (default: 20). "
            "Ignored with --web."
        ),
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Start the interactive browser UI instead of running headless.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--normalize-coordinates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use normalized 0..1000 coordinates for click/hover/inspect "
            "instead of screenshot pixels (default: pixel coordinates)."
        ),
    )
    parser.add_argument(
        "--snap-to-tile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Snap click/hover/inspect coordinates to the nearest board tile "
            "centre. "
            "The grid is calibrated from the drop-zone tiles during deployment "
            "and reused for the rest of the game (default: off)."
        ),
    )
    parser.add_argument(
        "--grid-overlay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Draw the isometric tile grid with pixel-coordinate labels on "
            "board screenshots after the grid is calibrated. Drop-zone tiles "
            "are highlighted in gold during deployment. Menus and dialogs "
            "are left clean (default: off)."
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=4,
        help=(
            "Screenshots kept in conversation history; 0 keeps everything "
            "(better prompt-cache reuse, larger context)."
        ),
    )
    parser.add_argument(
        "--action-delay",
        type=float,
        default=0.5,
        help="Seconds to wait after clicks, keys, and hovers (default: 0.5).",
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream model thinking to the terminal (default: on).",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help=(
            "With --no-stream, print the saved thinking after the response. "
            "Streaming mode already prints it live."
        ),
    )
    parser.add_argument(
        "--notes",
        action="store_true",
        help="Enable the optional note tool and load persistent notes.",
    )
    parser.add_argument(
        "--notes-file",
        type=Path,
        default=Path("notes.md"),
        help="Persistent notes file (default: notes.md).",
    )
    parser.add_argument(
        "--notes-max-chars",
        type=int,
        default=6000,
        help="Maximum note characters loaded into the prompt (default: 6000).",
    )
    parser.add_argument(
        "--model-timeout",
        type=float,
        default=180.0,
        help="Mistral request timeout in seconds (default: 180).",
    )
    parser.add_argument(
        "--cache-key",
        default=None,
        help=(
            "Stable Mistral prompt_cache_key; defaults to "
            "MISTRAL_PROMPT_CACHE_KEY or a per-run UUID."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Mistral model (default: MISTRAL_MODEL or mistral-large-4).",
    )
    parser.add_argument(
        "--list",
        dest="list_windows",
        action="store_true",
        help="List windows and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        backend = get_backend(args.backend)
    except BackendUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    input_backend: InputBackend | None = None
    try:
        if args.list_windows:
            print(format_windows(backend.list_windows()))
            return 0

        if not args.window:
            print("error: --window is required", file=sys.stderr)
            print(format_windows(backend.list_windows()), file=sys.stderr)
            return 2

        try:
            target = _resolve_target(backend, args.window)
        except (ValueError, CaptureError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        try:
            input_backend = get_input_backend(args.input)
        except BackendUnavailable as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        try:
            agent = MistralAgent(
                model=args.model,
                prompt_cache_key=args.cache_key,
                stream=args.stream,
                timeout_ms=max(1, int(args.model_timeout * 1000)),
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"error: could not initialize model client: {exc}", file=sys.stderr)
            return 2

        notes = (
            NotesStore(
                args.notes_file.expanduser(),
                max_chars=args.notes_max_chars,
            )
            if args.notes
            else None
        )

        config = LoopConfig(
            max_steps=max(1, args.steps),
            max_images=args.max_images,
            action_delay=max(0.0, args.action_delay),
            show_thinking=args.show_thinking,
            normalized_coordinates=args.normalize_coordinates,
            snap_to_tile=args.snap_to_tile,
            grid_overlay=args.grid_overlay,
            console=not args.web,
            out_dir=args.out.expanduser(),
            model=agent.model,
        )
        if input_backend.name == "stub":
            print(
                "warning: input backend is 'stub'; clicks and keys are "
                "recorded but NOT sent to the game. Use --input x11 for real "
                "input.",
                file=sys.stderr,
            )

        loop = HarnessLoop(
            backend, target, agent, input_backend, config, notes=notes
        )

        if args.web:
            import uvicorn

            from .session import SessionRunner
            from .web.server import create_app

            runner = SessionRunner(loop)
            app = create_app(config.out_dir, live=runner)
            runner.start()
            print(f"Live UI: http://{args.host}:{args.port}")
            print("Manual step mode; press Step in the browser to begin.")
            try:
                uvicorn.run(
                    app,
                    host=args.host,
                    port=args.port,
                    log_level="info",
                )
            finally:
                runner.stop()
                runner.join(timeout=5.0)
            return 0

        try:
            result = loop.run()
        except Exception as exc:
            print(f"error: loop failed: {exc}", file=sys.stderr)
            return 1
        print(
            f"\nStopped: {result.stopped}  steps={result.steps}  "
            f"clicks={result.actions}  frames={result.frames}"
        )
        print(f"Run output: {result.run_dir}")
        return 0
    finally:
        if input_backend is not None:
            input_backend.close()
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
