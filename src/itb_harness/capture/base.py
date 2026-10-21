"""Backend-neutral types for the observation side of the harness.

The loop only needs two things from a capture backend:

* enumerate the windows that exist right now
* capture one specific window as a PNG

Everything else (KWin, X11, Windows Graphics Capture, macOS ScreenCaptureKit,
...) is an implementation detail behind :class:`CaptureBackend`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class CaptureError(RuntimeError):
    """Base class for capture failures."""


class BackendUnavailable(CaptureError):
    """No usable capture backend exists on this machine."""


class WindowNotFound(CaptureError):
    """The selected window closed or could not be found."""


class CaptureFailed(CaptureError):
    """The backend could not produce a frame."""


@dataclass(frozen=True, slots=True)
class MonitorInfo:
    id: str
    name: str
    x: int
    y: int
    width: int
    height: int
    scale: float = 1.0

    @property
    def geometry(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


@dataclass(frozen=True, slots=True)
class WindowInfo:
    id: str
    title: str
    app_id: str
    pid: int | None
    x: int
    y: int
    width: int
    height: int
    monitor: str | None
    active: bool
    minimized: bool
    fullscreen: bool

    @property
    def geometry(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    @property
    def label(self) -> str:
        title = self.title.strip() or self.app_id or self.id
        return title


@dataclass(frozen=True, slots=True)
class Frame:
    window: WindowInfo
    png: bytes
    width: int
    height: int
    captured_at: str
    latency_ms: float


@runtime_checkable
class CaptureBackend(Protocol):
    name: str

    def list_monitors(self) -> list[MonitorInfo]:
        """Return the compositor's current outputs."""

    def list_windows(self) -> list[WindowInfo]:
        """Return all windows the compositor is willing to expose."""

    def capture_window(self, window_id: str) -> Frame:
        """Capture the client area of one window as PNG bytes."""

    def close(self) -> None:
        """Release any helper processes or handles."""


def _intersection_area(
    window: WindowInfo, monitor: MonitorInfo
) -> int:
    left = max(window.x, monitor.x)
    top = max(window.y, monitor.y)
    right = min(window.x + window.width, monitor.x + monitor.width)
    bottom = min(window.y + window.height, monitor.y + monitor.height)
    if right <= left or bottom <= top:
        return 0
    return (right - left) * (bottom - top)


def resolve_window(windows: list[WindowInfo], spec: str) -> WindowInfo:
    """Resolve a 1-based index, exact id, or title/app-id substring."""

    raw = spec.strip()
    if not raw:
        raise ValueError("Empty window selector.")
    if not windows:
        raise ValueError("No windows are available.")

    if raw.isdigit():
        index = int(raw)
        if not 1 <= index <= len(windows):
            raise ValueError(
                f"Window index {index} out of range (1-{len(windows)})."
            )
        return windows[index - 1]

    wanted = raw.strip("{}")
    exact = [
        window
        for window in windows
        if window.id.strip("{}") == wanted
        or window.id == raw
        or window.label == raw
    ]
    if exact:
        return exact[0]

    needle = raw.casefold()
    matches = [
        window
        for window in windows
        if needle in window.label.casefold()
        or needle in window.app_id.casefold()
        or needle in window.id.casefold()
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"No window matches {raw!r}.")

    names = "\n".join(f"  {window.label}" for window in matches[:12])
    raise ValueError(
        f"{len(matches)} windows match {raw!r}; be more specific:\n{names}"
    )


def monitor_for_window(
    window: WindowInfo, monitors: list[MonitorInfo]
) -> str | None:
    """Return the output holding most of ``window``.

    Fullscreen/borderless game windows normally intersect exactly one output.
    For straddling windows, the output with the largest overlap wins.
    """

    if not monitors:
        return None

    best = max(monitors, key=lambda monitor: _intersection_area(window, monitor))
    if _intersection_area(window, best) > 0:
        return best.name

    # No overlap (e.g. a stale/minimized geometry): fall back to nearest centre.
    wx = window.x + window.width / 2
    wy = window.y + window.height / 2
    nearest = min(
        monitors,
        key=lambda monitor: (wx - (monitor.x + monitor.width / 2)) ** 2
        + (wy - (monitor.y + monitor.height / 2)) ** 2,
    )
    return nearest.name
