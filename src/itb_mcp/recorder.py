"""Run recording for MCP sessions.

Unlike the headless harness recorder, this does not log model turns: the MCP
client owns the conversation.  It records frames plus one JSON line per game
tool call so a run can be replayed or inspected later.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from itb_harness.capture import CaptureBackend, WindowInfo
from itb_harness.input import InputBackend

from .config import Config


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


class Recorder:
    """Write frames and an action log for one MCP run."""

    def __init__(
        self,
        root: Path,
        *,
        backend: CaptureBackend,
        input_backend: InputBackend,
        target: WindowInfo,
        notes_file: Path | None,
        initial_notes: str | None,
        config: Config,
    ) -> None:
        self.root = root
        self.frames_dir = root / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.frame_index = 0
        self.action_index = 0
        self.started_monotonic = time.monotonic()
        self._closed = False

        self.actions_path = root / "actions.jsonl"
        self._actions = self.actions_path.open("a", encoding="utf-8")
        self.metadata_path = root / "run.json"

        self.metadata: dict[str, Any] = {
            "kind": "mcp",
            "backend": backend.name,
            "input": input_backend.name,
            "model": "external MCP client",
            "notes_file": str(notes_file) if notes_file is not None else None,
            "action_delay": config.action_delay,
            "strict_box_labels": config.strict_box_labels,
            "scaled_boxes": config.scaled_boxes,
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "target": asdict(target),
        }
        if initial_notes and initial_notes.strip():
            self.metadata["initial_notes"] = initial_notes.strip()
        self._write_metadata()

    @classmethod
    def create(
        cls,
        out_dir: Path,
        *,
        backend: CaptureBackend,
        input_backend: InputBackend,
        target: WindowInfo,
        notes_file: Path | None,
        initial_notes: str | None,
        config: Config,
    ) -> Recorder:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        root = out_dir / f"{stamp}-mcp"
        root.mkdir(parents=True, exist_ok=False)
        return cls(
            root,
            backend=backend,
            input_backend=input_backend,
            target=target,
            notes_file=notes_file,
            initial_notes=initial_notes,
            config=config,
        )

    @property
    def run_id(self) -> str:
        return self.root.name

    def _write_metadata(self) -> None:
        self.metadata_path.write_text(
            json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8"
        )

    def save_frame(self, png: bytes) -> Path:
        self.frame_index += 1
        path = self.frames_dir / f"{self.frame_index:04d}.png"
        path.write_bytes(png)
        return path

    def update_target(self, target: WindowInfo) -> None:
        self.metadata["target"] = asdict(target)
        self._write_metadata()

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
        frame_path: Path | None = None,
        source_frame_path: Path | None = None,
    ) -> None:
        self.action_index += 1
        record = {
            "step": self.action_index,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "tool": tool,
            "arguments": _jsonable(arguments or {}),
            "ok": ok,
            "error": error,
            "message": message,
            "blocked": blocked,
            "state_changed": state_changed,
            "frame": self._relative(frame_path),
            "source_frame": self._relative(source_frame_path),
        }
        self._actions.write(json.dumps(record) + "\n")
        self._actions.flush()

    def _relative(self, path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def close(self) -> dict[str, Any]:
        """Stop recording and return a run summary."""

        if self._closed:
            return self.summary()
        self._closed = True
        self._actions.close()

        ended_at = datetime.now(UTC)
        self.metadata["ended_at"] = ended_at.isoformat(timespec="seconds")
        self.metadata["frames"] = self.frame_index
        self.metadata["actions"] = self.action_index
        self.metadata["duration_seconds"] = round(
            time.monotonic() - self.started_monotonic, 3
        )
        self._write_metadata()
        return self.summary()

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.root),
            "frames": self.frame_index,
            "actions": self.action_index,
            "started_at": self.metadata.get("started_at"),
            "ended_at": self.metadata.get("ended_at"),
            "duration_seconds": self.metadata.get("duration_seconds"),
        }
