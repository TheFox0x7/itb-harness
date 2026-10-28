"""Game-side state shared by every MCP tool call.

Only game state lives here: the current frame, overlays, move history, and
the optional run recorder.  Conversation history belongs to the MCP client.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from itb_harness.annotate import BoxAnnotation
from itb_harness.capture import (
    CaptureBackend,
    Frame,
    WindowInfo,
    get_backend,
    resolve_window,
)
from itb_harness.input import InputBackend, get_input_backend
from itb_harness.notes import NotesStore

from .config import Config
from .frame_diff import frames_differ
from .overlays import apply_overlays
from .recorder import Recorder


class WindowSelectionError(ValueError):
    """The target game window could not be chosen."""


@dataclass(frozen=True, slots=True)
class MoveRecord:
    direction: str
    blocked: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {"direction": self.direction, "blocked": self.blocked}


class GameState:
    """Mutable game state for one MCP server process."""

    def __init__(
        self,
        config: Config,
        *,
        backend: CaptureBackend | None = None,
        input_backend: InputBackend | None = None,
        target: WindowInfo | None = None,
        notes: NotesStore | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.backend = backend
        self.input = input_backend
        self.target = target
        self.notes = notes or NotesStore(
            config.notes_file,
            max_chars=config.notes_max_chars,
            title="Helltaker harness notes",
        )
        self._sleep = sleep
        self.lock = threading.RLock()

        self.current_frame: Frame | None = None
        self._raw_png: bytes | None = None
        self._prev_raw_png: bytes | None = None
        self.last_frame_path: Path | None = None

        self.show_grid = False
        self.show_walkability = False
        self.box_overlay: list[BoxAnnotation] = []

        self.move_history: list[MoveRecord] = []
        self.move_count = 0
        self.steps_consumed = 0
        self.action_count = 0

        self.run_id: str | None = None
        self.recorder: Recorder | None = None

    # ------------------------------------------------------------------
    # Backend / target lifecycle
    # ------------------------------------------------------------------

    def ensure_ready(self) -> None:
        """Create the capture backend and resolve a target window on first use.

        Input is created separately so observation still works when input
        simulation is unavailable (for example a read-only capture setup).
        """

        with self.lock:
            if self.backend is None:
                self.backend = get_backend(self.config.backend)
            if self.target is None:
                self.target = self._select_target()

    def ensure_input(self) -> None:
        """Create the input backend on first use."""

        with self.lock:
            if self.input is None:
                self.input = get_input_backend(self.config.input)

    def _select_target(self) -> WindowInfo:
        assert self.backend is not None
        windows = self.backend.list_windows()
        if self.config.window:
            return resolve_window(windows, self.config.window)

        matches = [
            window
            for window in windows
            if "helltaker" in f"{window.label} {window.app_id}".casefold()
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            labels = ", ".join(window.label for window in windows[:12])
            raise WindowSelectionError(
                "No Helltaker window found. Set ITB_MCP_WINDOW to a window "
                "index/title, call select_window, or list_windows to see "
                f"available windows: {labels or '(none)'}"
            )
        labels = ", ".join(window.label for window in matches)
        raise WindowSelectionError(
            f"Multiple Helltaker windows found ({labels}); choose one with "
            "ITB_MCP_WINDOW or select_window."
        )

    def select_window(self, selector: str) -> WindowInfo:
        """Point the server at a different game window."""

        with self.lock:
            if self.backend is None:
                self.backend = get_backend(self.config.backend)
            target = resolve_window(self.backend.list_windows(), selector)
            self.target = target
            if self.recorder is not None:
                self.recorder.update_target(target)
            return target

    def close(self) -> None:
        with self.lock:
            if self.recorder is not None:
                self.recorder.close()
                self.recorder = None
            if self.input is not None:
                self.input.close()
                self.input = None
            if self.backend is not None:
                self.backend.close()
                self.backend = None

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def capture(self) -> Frame:
        """Capture, annotate, optionally save, and return the current frame."""

        with self.lock:
            self.ensure_ready()
            assert self.backend is not None and self.target is not None

            raw = self.backend.capture_window(self.target.id)
            self.target = raw.window
            # Keep the pre-annotation pixels for blocked-move detection.
            self._prev_raw_png = self._raw_png
            self._raw_png = raw.png

            png = apply_overlays(
                raw.png,
                raw.width,
                raw.height,
                boxes=self.box_overlay,
                show_grid=self.show_grid,
                show_walkability=self.show_walkability,
            )
            frame = replace(raw, png=png)
            self.current_frame = frame

            if self.recorder is not None:
                self.last_frame_path = self.recorder.save_frame(frame.png)
            return frame

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def move(self, direction: str) -> tuple[Frame, bool | None]:
        """Press one direction key, then capture the result."""

        with self.lock:
            self.ensure_ready()
            self.ensure_input()
            assert self.input is not None

            self.input.key(direction)
            self.action_count += 1
            self._pause()

            frame = self.capture()
            blocked: bool | None = None
            if self._prev_raw_png is not None and self._raw_png is not None:
                blocked = not frames_differ(self._prev_raw_png, self._raw_png)

            self.move_count += 1
            if blocked is False:
                self.steps_consumed += 1
            self.record_move(direction, blocked)
            return frame, blocked

    def reset_level(self) -> Frame:
        """Press R, clear boxes, and capture the restarted level."""

        with self.lock:
            self.ensure_ready()
            self.ensure_input()
            assert self.input is not None

            self.input.key("r")
            self.action_count += 1
            self.box_overlay = []
            self._pause()

            frame = self.capture()
            # Never compare a post-reset frame with one from before the reset.
            self._prev_raw_png = None
            return frame

    def _pause(self) -> None:
        if self.config.action_delay > 0:
            self._sleep(self.config.action_delay)

    def set_boxes(self, boxes: list[BoxAnnotation]) -> None:
        with self.lock:
            self.box_overlay = list(boxes)

    def add_boxes(self, boxes: list[BoxAnnotation]) -> None:
        with self.lock:
            self.box_overlay = [*self.box_overlay, *boxes]

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def record_move(self, direction: str, blocked: bool | None) -> None:
        self.move_history.append(MoveRecord(direction, blocked))
        if len(self.move_history) > 10:
            self.move_history = self.move_history[-10:]

    def format_history(self) -> str:
        parts = []
        for entry in self.move_history[-5:]:
            if entry.blocked is True:
                mark = "✗"
            elif entry.blocked is False:
                mark = "✓"
            else:
                mark = "?"
            parts.append(f"{entry.direction}{mark}")
        return f"Recent moves: {' '.join(parts)}." if parts else ""

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def start_run(self, notes: str | None = None) -> Recorder:
        with self.lock:
            self.ensure_ready()
            self.ensure_input()
            if self.recorder is not None:
                raise ValueError(
                    "A run is already active. Call end_run before starting another one."
                )
            assert (
                self.backend is not None
                and self.input is not None
                and self.target is not None
            )
            self.config.runs_dir.mkdir(parents=True, exist_ok=True)
            self.recorder = Recorder.create(
                self.config.runs_dir,
                backend=self.backend,
                input_backend=self.input,
                target=self.target,
                notes_file=self.notes.path,
                initial_notes=notes,
                config=self.config,
            )
            self.run_id = self.recorder.run_id
            self.last_frame_path = None
            self.move_history = []
            self.move_count = 0
            self.steps_consumed = 0
            self.action_count = 0
            return self.recorder

    def end_run(self) -> dict[str, Any]:
        with self.lock:
            if self.recorder is None:
                raise ValueError("No run is active.")
            summary = self.recorder.close()
            self.recorder = None
            self.last_frame_path = None
            return summary

    def log_action(
        self,
        *,
        tool: str,
        arguments: dict[str, Any] | None = None,
        ok: bool = True,
        error: str | None = None,
        message: str = "",
        blocked: bool | None = None,
        state_changed: bool | None = None,
        source_frame_path: Path | None = None,
    ) -> None:
        recorder = self.recorder
        if recorder is None:
            return
        recorder.log_action(
            tool=tool,
            arguments=arguments,
            ok=ok,
            error=error,
            message=message,
            blocked=blocked,
            state_changed=state_changed,
            frame_path=self.last_frame_path,
            source_frame_path=source_frame_path,
        )

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            last_frame = None
            if self.recorder is not None and self.last_frame_path is not None:
                try:
                    last_frame = str(
                        self.last_frame_path.relative_to(self.recorder.root)
                    )
                except ValueError:
                    last_frame = str(self.last_frame_path)

            current = None
            if self.current_frame is not None:
                current = {
                    "width": self.current_frame.width,
                    "height": self.current_frame.height,
                    "captured_at": self.current_frame.captured_at,
                }

            return {
                "run_id": self.run_id,
                "recording": self.recorder is not None,
                "step_count": self.steps_consumed,
                "move_count": self.move_count,
                "action_count": self.action_count,
                "overlays": {
                    "grid": self.show_grid,
                    "walkability": self.show_walkability,
                },
                "box_overlay_count": len(self.box_overlay),
                "boxes": [
                    {
                        "box_2d": [
                            box.xmin,
                            box.ymin,
                            box.xmax,
                            box.ymax,
                        ],
                        "label": box.label,
                    }
                    for box in self.box_overlay
                ],
                "move_history": [entry.to_dict() for entry in self.move_history],
                "frame_count": (self.recorder.frame_index if self.recorder else 0),
                "last_frame": last_frame,
                "current_frame": current,
                "window": asdict(self.target) if self.target else None,
                "notes_file": str(self.notes.path),
                "runs_dir": str(self.config.runs_dir),
            }
