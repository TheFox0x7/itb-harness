"""Runtime configuration for the MCP server.

Configuration comes from environment variables so a client (Vibe, Claude
Desktop, ...) can launch ``itb-mcp`` without extra arguments.  Command-line
flags in :mod:`itb_mcp.server` override these values.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


def _env_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Config:
    """Settings shared by the game state, recorder, and server."""

    backend: str = "auto"
    input: str = "x11"
    window: str | None = None
    runs_dir: Path = Path("runs")
    notes_file: Path = Path("notes.md")
    notes_max_chars: int = 6000
    action_delay: float = 0.35
    strict_box_labels: bool = False
    scaled_boxes: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Config:
        """Build a config from ``ITB_MCP_*`` environment variables."""

        values = os.environ if env is None else env
        return cls(
            backend=values.get("ITB_MCP_BACKEND", "auto").strip() or "auto",
            input=values.get("ITB_MCP_INPUT", "x11").strip() or "x11",
            window=(values.get("ITB_MCP_WINDOW") or "").strip() or None,
            runs_dir=Path(values.get("ITB_MCP_RUNS", "runs")).expanduser(),
            notes_file=Path(values.get("ITB_MCP_NOTES", "notes.md")).expanduser(),
            notes_max_chars=max(
                1000, int(values.get("ITB_MCP_NOTES_MAX_CHARS", "6000"))
            ),
            action_delay=max(0.0, float(values.get("ITB_MCP_ACTION_DELAY", "0.35"))),
            strict_box_labels=_env_bool(values.get("ITB_MCP_STRICT_BOX_LABELS")),
            scaled_boxes=_env_bool(values.get("ITB_MCP_SCALED_BOXES")),
        )

    def replaced(self, **changes: object) -> Config:
        """Return a copy with ``changes`` applied (used by CLI flags)."""

        from dataclasses import replace

        return replace(self, **changes)  # type: ignore[arg-type]
