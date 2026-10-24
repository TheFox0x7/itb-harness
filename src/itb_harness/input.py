"""Pointer and keyboard input backends.

Coordinates passed to the pointer methods are absolute screen coordinates on
the selected X11/gamescope display.  The default backend records requests
without touching the desktop; ``--input x11`` performs real XTest input.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import Literal, Protocol

from .capture import BackendUnavailable


class InputError(RuntimeError):
    """Input simulation failed."""


@dataclass(frozen=True, slots=True)
class InputEvent:
    kind: str
    x: int | None = None
    y: int | None = None
    key: str | None = None


class InputBackend(Protocol):
    name: str

    def click(self, x: int, y: int) -> None:
        """Move to absolute (x, y) and left-click."""

    def right_click(self) -> None:
        """Right-click at the current pointer position."""

    def hover(self, x: int, y: int) -> None:
        """Move to absolute (x, y) without clicking."""

    def key(self, key: str) -> None:
        """Press and release a key."""

    def key_down(self, key: str) -> None:
        """Hold a key down."""

    def key_up(self, key: str) -> None:
        """Release a key."""

    def close(self) -> None:
        """Release any display connection."""


class StubInput:
    """Record requests without touching the pointer or keyboard."""

    name = "stub"

    def __init__(self) -> None:
        self.events: list[InputEvent] = []

    def click(self, x: int, y: int) -> None:
        self.events.append(InputEvent("click", x, y))

    def right_click(self) -> None:
        self.events.append(InputEvent("right_click"))

    def hover(self, x: int, y: int) -> None:
        self.events.append(InputEvent("hover", x, y))

    def key(self, key: str) -> None:
        self.events.append(InputEvent("key", key=key))

    def key_down(self, key: str) -> None:
        self.events.append(InputEvent("key_down", key=key))

    def key_up(self, key: str) -> None:
        self.events.append(InputEvent("key_up", key=key))

    def close(self) -> None:
        pass


class X11Input:
    """Pointer and keyboard input over XTest on an X11 display."""

    name = "x11"

    def __init__(self, display_name: str | None = None) -> None:
        try:
            from Xlib import X, XK, display as xdisplay
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
        self._XK = XK
        self._xtest = xtest
        self._root = self._display.screen().root
        self._keycodes: dict[str, int] = {}

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("DISPLAY")) and (
            importlib.util.find_spec("Xlib") is not None
        )

    def _keycode(self, key: str) -> int:
        if key in self._keycodes:
            return self._keycodes[key]

        named = {
            "space": self._XK.XK_space,
            "backspace": self._XK.XK_BackSpace,
            "shift": self._XK.XK_Shift_L,
            "ctrl": self._XK.XK_Control_L,
            "alt": self._XK.XK_Alt_L,
        }
        keysym = named.get(key)
        if keysym is None:
            keysym = self._XK.string_to_keysym(key)
        if not keysym:
            raise InputError(f"Unknown key {key!r}.")

        code = self._display.keysym_to_keycode(keysym)
        if not code:
            raise InputError(f"No keycode for key {key!r}.")
        self._keycodes[key] = int(code)
        return int(code)

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

    def right_click(self) -> None:
        try:
            self._xtest.fake_input(self._display, self._X.ButtonPress, 3)
            self._xtest.fake_input(self._display, self._X.ButtonRelease, 3)
            self._display.sync()
        except Exception as exc:
            raise InputError(f"Could not right-click: {exc}") from exc

    def hover(self, x: int, y: int) -> None:
        self._move(x, y)

    def key(self, key: str) -> None:
        code = self._keycode(key)
        try:
            self._xtest.fake_input(self._display, self._X.KeyPress, code)
            self._xtest.fake_input(self._display, self._X.KeyRelease, code)
            self._display.sync()
        except Exception as exc:
            raise InputError(f"Could not press {key!r}: {exc}") from exc

    def key_down(self, key: str) -> None:
        code = self._keycode(key)
        try:
            self._xtest.fake_input(self._display, self._X.KeyPress, code)
            self._display.sync()
        except Exception as exc:
            raise InputError(f"Could not hold {key!r}: {exc}") from exc

    def key_up(self, key: str) -> None:
        code = self._keycode(key)
        try:
            self._xtest.fake_input(self._display, self._X.KeyRelease, code)
            self._display.sync()
        except Exception as exc:
            raise InputError(f"Could not release {key!r}: {exc}") from exc

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
