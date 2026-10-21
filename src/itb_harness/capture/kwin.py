"""KDE Plasma / KWin backend.

This talks to KWin's ``org.kde.KWin.ScreenShot2`` service through the small
``kwcapture`` helper.  It is deliberately isolated here: the rest of the
harness never imports kwcapture or sees its types.
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
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


class KWinBackend:
    """Window capture on KDE Plasma 6 Wayland."""

    name = "kwin"

    def __init__(self) -> None:
        try:
            import kwcapture  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - depends on the desktop
            raise BackendUnavailable(
                "kwcapture is not installed. Run `uv sync --extra kde`."
            ) from exc

        self._kc = kwcapture
        self._captures: dict[str, object] = {}

    @staticmethod
    def available() -> bool:
        if not sys.platform.startswith("linux"):
            return False
        return importlib.util.find_spec("kwcapture") is not None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_monitors(self) -> list[MonitorInfo]:
        try:
            monitors = self._kc.list_monitors()
        except Exception as exc:
            raise CaptureFailed(f"KWin monitor enumeration failed: {exc}") from exc

        result: list[MonitorInfo] = []
        for monitor in monitors:
            result.append(
                MonitorInfo(
                    id=str(monitor.id),
                    name=monitor.name,
                    x=int(monitor.x),
                    y=int(monitor.y),
                    width=int(monitor.width),
                    height=int(monitor.height),
                    scale=float(monitor.scale or 1.0),
                )
            )
        return result

    def _to_window(self, raw: object, monitors: list[MonitorInfo]) -> WindowInfo:
        width = int(raw.width)
        height = int(raw.height)
        provisional = WindowInfo(
            id=str(getattr(raw, "id", None) or "{" + getattr(raw, "uuid", "") + "}"),
            title=str(getattr(raw, "name", "") or ""),
            app_id=str(getattr(raw, "app_id", "") or ""),
            pid=None,
            x=int(raw.x),
            y=int(raw.y),
            width=width,
            height=height,
            monitor=None,
            active=bool(getattr(raw, "active", False)),
            minimized=bool(getattr(raw, "minimized", False)),
            fullscreen=bool(getattr(raw, "fullscreen", False)),
        )
        return replace(
            provisional, monitor=monitor_for_window(provisional, monitors)
        )

    def list_windows(self) -> list[WindowInfo]:
        try:
            monitors = self.list_monitors()
            windows = self._kc.list_windows()
        except CaptureFailed:
            raise
        except Exception as exc:
            raise CaptureFailed(f"KWin window enumeration failed: {exc}") from exc

        return [self._to_window(window, monitors) for window in windows]

    def _find_raw(self, window_id: str) -> object:
        """Find a live kwcapture window by KWin UUID."""

        wanted = window_id.strip()
        wanted_uuid = wanted.strip("{}")
        try:
            windows = self._kc.list_windows()
        except Exception as exc:
            raise CaptureFailed(f"KWin window enumeration failed: {exc}") from exc

        for window in windows:
            ids = {
                str(getattr(window, "id", "")),
                "{" + str(getattr(window, "uuid", "")) + "}",
                str(getattr(window, "uuid", "")),
            }
            if wanted in ids or wanted_uuid in {value.strip("{}") for value in ids}:
                return window

        raise WindowNotFound(f"Window {window_id!r} is no longer open.")

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture_window(self, window_id: str) -> Frame:
        raw = self._find_raw(window_id)
        if bool(getattr(raw, "minimized", False)):
            raise CaptureFailed("The target window is minimized; restore it first.")

        raw_id = str(
            getattr(raw, "id", None) or "{" + getattr(raw, "uuid", "") + "}"
        )
        capture = self._captures.get(raw_id)
        if capture is None:
            try:
                capture = self._kc.Capture(window=raw_id)
            except Exception as exc:
                raise CaptureFailed(f"Could not attach to window: {exc}") from exc
            self._captures[raw_id] = capture

        started = time.perf_counter()
        try:
            array = capture.shot()  # type: ignore[attr-defined]
        except Exception as exc:
            self._captures.pop(raw_id, None)
            try:
                capture.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            raise CaptureFailed(f"KWin capture failed: {exc}") from exc

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        try:
            stats = capture.stats()  # type: ignore[attr-defined]
            latency_ms = float(stats.get("total_ms", elapsed_ms))
        except Exception:
            latency_ms = elapsed_ms

        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover
            raise CaptureFailed(
                "Pillow is required to encode frames; run `uv sync`."
            ) from exc

        try:
            image = Image.fromarray(array).convert("RGB")
            buffer = io.BytesIO()
            # Fast encode: these are intermediate frames for a model, not
            # archival images.  The model provider will re-encode anyway.
            image.save(buffer, format="PNG", compress_level=1)
            png = buffer.getvalue()
        except Exception as exc:
            raise CaptureFailed(f"Could not encode frame: {exc}") from exc

        monitors = self.list_monitors()
        window = self._to_window(raw, monitors)

        return Frame(
            window=window,
            png=png,
            width=int(image.width),
            height=int(image.height),
            captured_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        for capture in self._captures.values():
            try:
                capture.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._captures.clear()

    def __enter__(self) -> "KWinBackend":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
