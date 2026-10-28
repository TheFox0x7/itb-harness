"""Shared fakes for MCP server tests."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image

from itb_harness.capture import Frame, MonitorInfo, WindowInfo
from itb_harness.input import StubInput
from itb_mcp.config import Config
from itb_mcp.game_state import GameState

WINDOW = WindowInfo(
    id="{game}",
    title="Helltaker",
    app_id="Helltaker",
    pid=1234,
    x=0,
    y=0,
    width=64,
    height=48,
    monitor="DP-4",
    active=True,
    minimized=False,
    fullscreen=True,
)


def make_png(
    color: tuple[int, int, int] = (0, 0, 0),
    size: tuple[int, int] = (64, 48),
) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeBackend:
    """Return a fixed frame, or a sequence of frames, on each capture."""

    name = "fake"

    def __init__(
        self,
        png: bytes | None = None,
        frames: list[bytes] | None = None,
        window: WindowInfo | None = None,
    ) -> None:
        self.window = window or WINDOW
        self._frames = list(frames or [])
        self._default = png or make_png()
        self._index = 0
        self.captures = 0

    def list_monitors(self) -> list[MonitorInfo]:
        return [
            MonitorInfo(
                id="1",
                name="DP-4",
                x=0,
                y=0,
                width=self.window.width,
                height=self.window.height,
            )
        ]

    def list_windows(self) -> list[WindowInfo]:
        return [self.window]

    def capture_window(self, window_id: str) -> Frame:
        self.captures += 1
        if self._frames:
            png = self._frames[min(self._index, len(self._frames) - 1)]
            self._index += 1
        else:
            png = self._default
        return Frame(
            window=self.window,
            png=png,
            width=self.window.width,
            height=self.window.height,
            captured_at="2026-10-26T00:00:00+00:00",
            latency_ms=1.0,
        )

    def close(self) -> None:
        pass


def make_config(tmp: Path, **changes: object) -> Config:
    values: dict[str, object] = {
        "runs_dir": tmp / "runs",
        "notes_file": tmp / "notes.md",
        "action_delay": 0.0,
    }
    values.update(changes)
    return Config(**values)  # type: ignore[arg-type]


def make_state(
    tmp: Path,
    *,
    frames: list[bytes] | None = None,
    png: bytes | None = None,
    config: Config | None = None,
    backend: FakeBackend | None = None,
    input_backend: StubInput | None = None,
) -> tuple[GameState, FakeBackend, StubInput]:
    config = config or make_config(tmp)
    backend = backend or FakeBackend(png=png, frames=frames)
    pointer = input_backend or StubInput()
    state = GameState(
        config,
        backend=backend,
        input_backend=pointer,
        target=WINDOW,
        sleep=lambda _seconds: None,
    )
    return state, backend, pointer
