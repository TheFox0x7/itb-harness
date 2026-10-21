"""Generic X11 capture backend.

This works against any X server: a normal Xorg session, XWayland, Xvfb, or the
nested Xwayland that gamescope starts for the game.  It is deliberately not
tied to KDE.

Why window capture instead of root capture: under gamescope, the nested
Xwayland root is black; gamescope owns compositing.  The game window itself
still has readable pixels, so we capture that window.
"""

from __future__ import annotations

import importlib.util
import io
import os
import time
from dataclasses import replace
from datetime import datetime, timezone

from .base import (
    BackendUnavailable,
    CaptureFailed,
    Frame,
    MonitorInfo,
    WindowInfo,
    WindowNotFound,
    monitor_for_window,
)


class X11Backend:
    """Window capture over the X11 protocol, via python-xlib."""

    name = "x11"

    def __init__(self, display_name: str | None = None) -> None:
        try:
            from Xlib import X, display as xdisplay
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

        self._X = X
        self._screen = self._display.screen()
        self._root = self._screen.root
        self._atoms: dict[str, int] = {}

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("DISPLAY")) and (
            importlib.util.find_spec("Xlib") is not None
        )

    # ------------------------------------------------------------------
    # X11 helpers
    # ------------------------------------------------------------------

    def _atom(self, name: str) -> int:
        if name not in self._atoms:
            self._atoms[name] = self._display.intern_atom(name)
        return self._atoms[name]

    def _prop(self, window: object, name: str):
        try:
            return window.get_full_property(self._atom(name), self._X.AnyPropertyType)
        except Exception:
            return None

    def _wm_name(self, window: object) -> str:
        prop = self._prop(window, "_NET_WM_NAME")
        if prop is not None and prop.value:
            value = prop.value
            if isinstance(value, bytes):
                return value.decode("utf-8", "replace")
            return str(value)

        try:
            name = window.get_wm_name()
        except Exception:
            return ""
        if isinstance(name, bytes):
            return name.decode("utf-8", "replace")
        return str(name or "")

    def _wm_class(self, window: object) -> str:
        try:
            wm_class = window.get_wm_class()
        except Exception:
            return ""
        if not wm_class:
            return ""
        # (instance, class); the class is the useful app id.
        value = wm_class[1] if len(wm_class) > 1 else wm_class[0]
        if isinstance(value, bytes):
            return value.decode("utf-8", "replace")
        return str(value or "")

    def _pid(self, window: object) -> int | None:
        prop = self._prop(window, "_NET_WM_PID")
        if prop is None or not prop.value:
            return None
        try:
            return int(prop.value[0])
        except Exception:
            return None

    def _absolute_xy(self, window: object) -> tuple[int, int]:
        x = y = 0
        current = window
        while current is not None and current.id != self._root.id:
            try:
                geom = current.get_geometry()
            except Exception:
                break
            x += geom.x
            y += geom.y
            try:
                current = current.query_tree().parent
            except Exception:
                break
        return x, y

    def _is_active(self, window: object) -> bool:
        prop = self._prop(self._root, "_NET_ACTIVE_WINDOW")
        if prop is not None and prop.value:
            try:
                return int(prop.value[0]) == int(window.id)
            except Exception:
                pass
        try:
            focus = self._display.get_input_focus().focus
            return int(focus.id) == int(window.id)
        except Exception:
            return False

    def _is_fullscreen(self, window: object, width: int, height: int) -> bool:
        prop = self._prop(window, "_NET_WM_STATE")
        if prop is not None:
            fullscreen_atom = self._atom("_NET_WM_STATE_FULLSCREEN")
            try:
                if fullscreen_atom in {int(value) for value in prop.value}:
                    return True
            except Exception:
                pass
        screen = self._screen
        return width >= screen.width_in_pixels and height >= screen.height_in_pixels

    def _to_window(self, window: object, monitors: list[MonitorInfo]) -> WindowInfo:
        geom = window.get_geometry()
        x, y = self._absolute_xy(window)
        attrs = window.get_attributes()
        provisional = WindowInfo(
            id=hex(int(window.id)),
            title=self._wm_name(window),
            app_id=self._wm_class(window),
            pid=self._pid(window),
            x=x,
            y=y,
            width=int(geom.width),
            height=int(geom.height),
            monitor=None,
            active=self._is_active(window),
            minimized=attrs.map_state != self._X.IsViewable,
            fullscreen=self._is_fullscreen(window, geom.width, geom.height),
        )
        return replace(
            provisional, monitor=monitor_for_window(provisional, monitors)
        )

    def _window_candidates(self) -> list[object]:
        """Return mapped, named windows.

        Prefer EWMH's _NET_CLIENT_LIST when the server supports it.  Gamescope's
        Xwayland does not, so fall back to walking the window tree.
        """

        prop = self._prop(self._root, "_NET_CLIENT_LIST")
        if prop is not None and prop.value:
            result = []
            for value in prop.value:
                try:
                    window = self._display.create_resource_object(
                        "window", int(value)
                    )
                    attrs = window.get_attributes()
                    if attrs.map_state == self._X.IsViewable:
                        result.append(window)
                except Exception:
                    continue
            if result:
                return result

        result: list[object] = []

        def walk(window: object) -> None:
            try:
                attrs = window.get_attributes()
                geom = window.get_geometry()
                if (
                    attrs.map_state == self._X.IsViewable
                    and geom.width > 1
                    and geom.height > 1
                    and (self._wm_name(window) or self._wm_class(window))
                ):
                    result.append(window)
            except Exception:
                pass

            try:
                children = window.query_tree().children
            except Exception:
                return
            for child in children:
                walk(child)

        walk(self._root)
        return result

    # ------------------------------------------------------------------
    # CaptureBackend
    # ------------------------------------------------------------------

    def list_monitors(self) -> list[MonitorInfo]:
        monitors: list[MonitorInfo] = []
        try:
            reply = self._root.xrandr_get_monitors()
            for monitor in reply.monitors:
                try:
                    name = self._display.get_atom_name(monitor.name)
                except Exception:
                    name = f"monitor-{monitor.name}"
                monitors.append(
                    MonitorInfo(
                        id=str(monitor.name),
                        name=str(name),
                        x=int(monitor.x),
                        y=int(monitor.y),
                        width=int(monitor.width_in_pixels),
                        height=int(monitor.height_in_pixels),
                        scale=1.0,
                    )
                )
        except Exception:
            monitors = []

        if not monitors:
            screen = self._screen
            monitors.append(
                MonitorInfo(
                    id="0",
                    name=f"X11-{screen.width_in_pixels}x{screen.height_in_pixels}",
                    x=0,
                    y=0,
                    width=int(screen.width_in_pixels),
                    height=int(screen.height_in_pixels),
                    scale=1.0,
                )
            )
        return monitors

    def list_windows(self) -> list[WindowInfo]:
        try:
            monitors = self.list_monitors()
            windows = [
                self._to_window(window, monitors)
                for window in self._window_candidates()
            ]
        except Exception as exc:
            raise CaptureFailed(f"X11 window enumeration failed: {exc}") from exc

        # De-duplicate (some servers expose both a frame and a client).
        unique: dict[str, WindowInfo] = {}
        for window in windows:
            unique.setdefault(window.id, window)
        return sorted(
            unique.values(),
            key=lambda window: (-(window.width * window.height), window.title.lower()),
        )

    def capture_window(self, window_id: str) -> Frame:
        try:
            wid = int(window_id, 0)
            window = self._display.create_resource_object("window", wid)
        except Exception as exc:
            raise WindowNotFound(f"Invalid X11 window id {window_id!r}.") from exc

        try:
            attrs = window.get_attributes()
            if attrs.map_state != self._X.IsViewable:
                raise CaptureFailed("The target window is not viewable/minimized.")
            geom = window.get_geometry()
        except CaptureFailed:
            raise
        except Exception as exc:
            raise WindowNotFound(
                f"Window {window_id!r} is no longer open: {exc}"
            ) from exc

        if geom.width <= 1 or geom.height <= 1:
            raise CaptureFailed("The target window has no drawable area.")

        started = time.perf_counter()
        try:
            image = window.get_image(
                0,
                0,
                geom.width,
                geom.height,
                self._X.ZPixmap,
                0xFFFFFFFF,
            )
        except Exception as exc:
            raise CaptureFailed(f"X11 window capture failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover
            raise CaptureFailed(
                "Pillow is required to encode frames; run `uv sync`."
            ) from exc

        if int(image.depth) not in (24, 32):
            raise CaptureFailed(
                f"Unsupported X11 image depth {image.depth}; expected 24 or 32."
            )

        try:
            pil = Image.frombytes(
                "RGB",
                (int(geom.width), int(geom.height)),
                image.data,
                "raw",
                "BGRX",
            )
            buffer = io.BytesIO()
            pil.save(buffer, format="PNG", compress_level=1)
            png = buffer.getvalue()
        except Exception as exc:
            raise CaptureFailed(f"Could not encode X11 frame: {exc}") from exc

        monitors = self.list_monitors()
        window_info = self._to_window(window, monitors)

        return Frame(
            window=window_info,
            png=png,
            width=int(geom.width),
            height=int(geom.height),
            captured_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        try:
            self._display.close()
        except Exception:
            pass

    def __enter__(self) -> "X11Backend":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
