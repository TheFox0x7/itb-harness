"""Load recorded runs from disk for MCP resources.

The on-disk format matches the headless harness (``run.json``,
``actions.jsonl``, ``frames/*.png``), so runs produced by either front end can
be listed and replayed here.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

_RUN_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_FRAME_RE = re.compile(r"^[A-Za-z0-9_.-]+\.png$")


class RunNotFound(ValueError):
    """The requested run or frame does not exist."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _basename(value: Any) -> str | None:
    if not value:
        return None
    return Path(str(value)).name


def list_runs(runs_dir: Path) -> list[dict[str, Any]]:
    """List recorded runs, newest first."""

    runs_dir = runs_dir.expanduser()
    if not runs_dir.exists():
        return []

    result: list[dict[str, Any]] = []
    for path in runs_dir.iterdir():
        if not path.is_dir():
            continue
        run_file = path / "run.json"
        if not run_file.exists():
            continue
        try:
            metadata = json.loads(run_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        frames_dir = path / "frames"
        frame_count = len(list(frames_dir.glob("*.png"))) if frames_dir.exists() else 0
        actions = _read_jsonl(path / "actions.jsonl")
        result.append(
            {
                "id": path.name,
                "started_at": metadata.get("started_at"),
                "ended_at": metadata.get("ended_at"),
                "kind": metadata.get("kind", "agent"),
                "model": metadata.get("model"),
                "backend": metadata.get("backend"),
                "input": metadata.get("input"),
                "frames": frame_count,
                "actions": len(actions),
                "mtime": path.stat().st_mtime,
            }
        )
    return sorted(result, key=lambda item: item["mtime"], reverse=True)


def safe_run_dir(runs_dir: Path, run_id: str) -> Path:
    """Resolve a run id, rejecting traversal and unknown runs."""

    if not _RUN_RE.fullmatch(run_id):
        raise RunNotFound(f"Unknown run: {run_id}")
    root = runs_dir.expanduser().resolve()
    run_dir = (root / run_id).resolve()
    if run_dir.parent != root or not (run_dir / "run.json").is_file():
        raise RunNotFound(f"Unknown run: {run_id}")
    return run_dir


def safe_frame_path(runs_dir: Path, run_id: str, name: str) -> Path:
    """Resolve a frame inside a known run."""

    run_dir = safe_run_dir(runs_dir, run_id)
    if not _FRAME_RE.fullmatch(name):
        raise RunNotFound(f"Unknown frame: {name}")
    path = (run_dir / "frames" / name).resolve()
    if path.parent != (run_dir / "frames").resolve() or not path.is_file():
        raise RunNotFound(f"Unknown frame: {name}")
    return path


def _action_from_record(record: dict[str, Any]) -> dict[str, Any]:
    action = dict(record)
    try:
        action["step"] = int(action.get("step", 0))
    except (TypeError, ValueError):
        action["step"] = 0
    action["frame_name"] = _basename(action.get("frame"))
    action["source_frame_name"] = _basename(action.get("source_frame"))
    return action


def build_turns(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group actions into per-step turns for resource consumers.

    MCP runs have no model transcript, so each turn carries the action that
    was recorded at that step.
    """

    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in actions:
        action = _action_from_record(record)
        by_step[action["step"]].append(action)

    return [
        {
            "step": step,
            "content": None,
            "thinking": None,
            "usage": None,
            "guidance": [],
            "actions": by_step[step],
        }
        for step in sorted(by_step)
    ]


def load_run(run_dir: Path) -> dict[str, Any]:
    """Return metadata, frames, actions, and turns for one run."""

    run_file = run_dir / "run.json"
    if not run_file.is_file():
        raise RunNotFound(f"Unknown run: {run_dir.name}")
    metadata = json.loads(run_file.read_text(encoding="utf-8"))

    frames_dir = run_dir / "frames"
    frames = (
        sorted(path.name for path in frames_dir.glob("*.png") if path.is_file())
        if frames_dir.exists()
        else []
    )
    actions = _read_jsonl(run_dir / "actions.jsonl")

    return {
        "id": run_dir.name,
        "run": metadata,
        "frames": frames,
        "actions": actions,
        "turns": build_turns(actions),
    }


def actions_payload(run_dir: Path) -> list[dict[str, Any]]:
    return _read_jsonl(run_dir / "actions.jsonl")


def conversation_text(run_dir: Path, run: dict[str, Any] | None = None) -> str:
    """Render a human-readable transcript of the recorded actions.

    The server never sees the client's reasoning, so this is an action
    transcript rather than a full chat log.
    """

    loaded = run or load_run(run_dir)
    metadata = loaded["run"]
    lines = [
        f"Run: {loaded['id']}",
        f"Started: {metadata.get('started_at', '?')}",
        f"Kind: {metadata.get('kind', 'agent')}",
        "",
        "# Actions",
    ]
    if not loaded["actions"]:
        lines.append("(no actions recorded)")
    for record in loaded["actions"]:
        action = _action_from_record(record)
        arguments = record.get("arguments")
        rendered_args = (
            json.dumps(arguments, sort_keys=True)
            if isinstance(arguments, dict) and arguments
            else ""
        )
        call = f"{action['tool']}({rendered_args})" if rendered_args else action["tool"]
        status = "ok"
        if record.get("blocked"):
            status = "BLOCKED"
        elif record.get("ok") is False:
            status = f"error: {record.get('error') or record.get('message')}"
        message = record.get("message") or ""
        frame = action.get("frame_name")
        suffix = f" [{frame}]" if frame else ""
        lines.append(f"{action['step']:04d}  {call} -> {status}. {message}{suffix}")
    return "\n".join(lines) + "\n"
