"""Capture backend selection.

Cross-platform note: there is no single reliable Python API for *window*
capture across Wayland, macOS, and Windows.  The pragmatic design is a small
backend interface (this module) with one native implementation per platform.
"""

from __future__ import annotations

import os
import sys

from .base import (
    BackendUnavailable,
    CaptureBackend,
    CaptureError,
    CaptureFailed,
    Frame,
    MonitorInfo,
    WindowInfo,
    WindowNotFound,
    monitor_for_window,
    resolve_window,
)

__all__ = [
    "BackendUnavailable",
    "CaptureBackend",
    "CaptureError",
    "CaptureFailed",
    "Frame",
    "MonitorInfo",
    "WindowInfo",
    "WindowNotFound",
    "get_backend",
    "monitor_for_window",
    "resolve_window",
]


def _kde_session() -> bool:
    desktop = os.environ.get("XDG_CURRENT_DESKTOP", "")
    session = os.environ.get("XDG_SESSION_DESKTOP", "")
    return (
        "kde" in desktop.lower()
        or "plasma" in desktop.lower()
        or "kde" in session.lower()
        or "plasma" in session.lower()
        or os.environ.get("KDE_FULL_SESSION") == "true"
    )


def _kwin_available() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    if not _kde_session():
        return False
    try:
        from .kwin import KWinBackend

        return KWinBackend.available()
    except Exception:
        return False


def _x11_available() -> bool:
    try:
        from .x11 import X11Backend

        return X11Backend.available()
    except Exception:
        return False


def _get_x11() -> CaptureBackend:
    try:
        from .x11 import X11Backend
    except Exception as exc:
        raise BackendUnavailable(
            "The X11 backend needs python-xlib: `uv sync --extra x11`."
        ) from exc
    return X11Backend()


def _get_kwin() -> CaptureBackend:
    from .kwin import KWinBackend

    return KWinBackend()


def _auto_backend() -> CaptureBackend:
    """Pick a backend suitable for the current environment.

    gamescope runs its game on a nested Xwayland display.  If that display is
    exported in DISPLAY (and we are not just looking at the host's :0), the
    generic X11 backend is the right choice.  On a KDE Wayland desktop, KWin is
    still the best choice for the host desktop.
    """

    override = os.environ.get("ITB_CAPTURE_BACKEND", "").strip().lower()
    if override:
        return get_backend(override)

    # Explicit gamescope environment, if inherited by the harness.
    if os.environ.get("GAMESCOPE_WAYLAND_DISPLAY") or os.environ.get(
        "GAMESCOPE_LIMITER_FILE"
    ):
        if _x11_available():
            return _get_x11()
        raise BackendUnavailable(
            "gamescope detected, but the X11 backend is unavailable. "
            "Run `uv sync --extra x11` and ensure DISPLAY points at the "
            "gamescope Xwayland display."
        )

    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    display = os.environ.get("DISPLAY", "")
    # A non-default DISPLAY under a Wayland session is usually a nested
    # Xwayland server (gamescope, Xvfb, Xephyr).
    if display and session_type == "wayland" and display not in {":0", ":0.0"}:
        if _x11_available():
            return _get_x11()

    if _kwin_available():
        return _get_kwin()
    if _x11_available():
        return _get_x11()

    raise BackendUnavailable(
        "No capture backend found. On KDE Plasma 6 Wayland, run "
        "`uv sync --extra kde`; for gamescope/X11, run "
        "`uv sync --extra x11` and point DISPLAY at the nested X server."
    )


def get_backend(name: str = "auto") -> CaptureBackend:
    """Create a capture backend by name.

    ``auto`` discovers KWin on KDE Wayland, and X11 for gamescope/Xvfb/plain
    Xorg.  ``kwin`` and ``x11`` force a specific backend.
    """

    name = (name or "auto").strip().lower()

    if name == "auto":
        return _auto_backend()

    if name in {"kwin", "kde"}:
        if _kwin_available():
            return _get_kwin()
        raise BackendUnavailable(
            "The KWin backend requires KDE Plasma on Linux and the "
            "`kwcapture` package (`uv sync --extra kde`)."
        )

    if name == "x11":
        if _x11_available():
            return _get_x11()
        raise BackendUnavailable(
            "The X11 backend requires python-xlib (`uv sync --extra x11`) "
            "and a valid DISPLAY."
        )

    raise BackendUnavailable(
        f"Unknown capture backend {name!r}. Available backends: auto, kwin, x11."
    )
