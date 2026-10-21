"""Pointer input backends for point actions.

Coordinates passed to these backends are absolute screen coordinates on the
selected X11/gamescope display.  The loop maps normalized model coordinates to
window pixels and then adds the window origin before calling click/hover.

The default backend is a stub so running the loop cannot accidentally click the
desktop.  Use ``--input x11`` when you intentionally want real X11 input.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import Literal, Protocol

from .capture import BackendUnavailable


class InputError(RuntimeError):
    """Pointer input failed."""


@dataclass(frozen=True, slots=True)
class InputEvent:
    kind: Literal["click", "hover"]
    x: int
    y: int


class InputBackend(Protocol):
    name: str

    def click(self, x: int, y: int) -> None:
        """Move to absolute (x, y) and left-click."""

    def hover(self, x: int, y: int) -> None:
        """Move to absolute (x, y) without clicking."""

    def close(self) -> None:
        """Release any display connection."""


class StubInput:
    """Record requests without touching the pointer."""

    name = "stub"

    def __init__(self) -> None:
        self.events: list[InputEvent] = []

    def click(self, x: int, y: int) -> None:
        self.events.append(InputEvent("click", x, y))

    def hover(self, x: int, y: int) -> None:
        self.events.append(InputEvent("hover", x, y))

    def close(self) -> None:
        pass


class X11Input:
    """Absolute pointer input over XTest on an X11 display."""

    name = "x11"

    def __init__(self, display_name: str | None = None) -> None:
        try:
            from Xlib import X, display as xdisplay
            from Xlib.ext import xtest
        except Exception as exc:  # pragma: no cover - optional dependency
            raise BackendUnavailable(
                "python-xlib is not installed. Run `uv sync --extra x11`."
            ) from exc

        name = display_name or os.environ.get("DISPLAY")
        if not name:
            raise BackendUnavailable(
                "No DISPLAY is set. Point DISPLAY at the gamescope/X11 display."
            )

        try:
            self._display = xdisplay.Display(name)
        except Exception as exc:
            raise BackendUnavailable(
                f"Could not connect to X display {name!r}: {exc}"
            ) from exc

        if not self._display.query_extension("XTEST"):
            self._display.close()
            raise BackendUnavailable("The X server does not expose XTEST.")

        self._X = X
        self._xtest = xtest
        self._root = self._display.screen().root

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("DISPLAY")) and (
            importlib.util.find_spec("Xlib") is not None
        )

    def _move(self, x: int, y: int) -> None:
        try:
            self._root.warp_pointer(x, y)
            self._display.sync()
        except Exception as exc:
            raise InputError(f"Could not move pointer: {exc}") from exc

    def click(self, x: int, y: int) -> None:
        self._move(x, y)
        try:
            self._xtest.fake_input(self._display, self._X.ButtonPress, 1)
            self._xtest.fake_input(self._display, self._X.ButtonRelease, 1)
            self._display.sync()
        except Exception as exc:
            raise InputError(f"Could not click: {exc}") from exc

    def hover(self, x: int, y: int) -> None:
        self._move(x, y)

    def close(self) -> None:
        try:
            self._display.close()
        except Exception:
            pass


def get_input_backend(name: str = "stub") -> InputBackend:
    name = (name or "stub").strip().lower()
    if name == "stub":
        return StubInput()
    if name == "x11":
        if not X11Input.available():
            raise BackendUnavailable(
                "The X11 input backend needs python-xlib and DISPLAY "
                "(`uv sync --extra x11`)."
            )
        return X11Input()
    raise BackendUnavailable(
        f"Unknown input backend {name!r}. Available: stub, x11."
    )
