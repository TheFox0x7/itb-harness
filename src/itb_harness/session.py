"""Interactive session control and live event bus for the observer/act loop.

The loop is synchronous and may block on capture, model calls, or input.  A
SessionRunner owns it on a worker thread and exposes thread-safe controls plus a
small event stream for the web UI.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Protocol


class SteppableLoop(Protocol):
    step_index: int
    action_count: int

    @property
    def recorder(self) -> Any: ...

    def step_once(self) -> bool: ...

    def add_guidance(self, text: str) -> bool: ...

    def close(self) -> None: ...


class SessionRunner:
    def __init__(self, loop: SteppableLoop) -> None:
        self.loop = loop
        self.run_id = loop.recorder.root.name

        self.state = "paused"
        self.auto = False
        self.error: str | None = None
        self.reason: str | None = None

        self.phase = "idle"
        self.thinking = ""
        self.content = ""
        self.first_token_ms: int | None = None
        self._model_started: float | None = None
        self._last_activity = time.monotonic()

        self._step_requested = False
        self._stop_requested = False
        self._busy = False
        self._closed = False

        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None

        self._events: deque[dict[str, Any]] = deque(maxlen=5000)
        self._subscribers: list[queue.Queue[dict[str, Any]]] = []
        self._seq = 0

        setattr(loop, "event_callback", self._handle_loop_event)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._cv:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run, name="itb-session", daemon=True
            )
            self._thread.start()

    def step(self) -> bool:
        with self._cv:
            if self.state in {"stopped", "stopping", "error"}:
                return False
            if self._step_requested or self._busy or self.auto:
                return False
            self._step_requested = True
            self.state = "running"
            self.phase = "queued"
            self._cv.notify_all()
        self._emit_status()
        return True

    def resume(self) -> bool:
        with self._cv:
            if self.state in {"stopped", "stopping", "error"}:
                return False
            self.auto = True
            self.state = "running"
            self.phase = "queued"
            self._cv.notify_all()
        self._emit_status()
        return True

    def pause(self) -> bool:
        with self._cv:
            if self.state in {"stopped", "stopping", "error"}:
                return False
            self.auto = False
            self.state = "pausing" if self._busy else "paused"
            self.phase = "pausing" if self._busy else "paused"
            self._cv.notify_all()
        self._emit_status()
        return True

    def steer(self, text: str) -> bool:
        """Queue a steering message for the next model turn."""
        text = text.strip()
        if not text:
            return False
        with self._lock:
            if self.state in {"stopped", "stopping", "error"}:
                return False
        add_guidance = getattr(self.loop, "add_guidance", None)
        if not callable(add_guidance) or not add_guidance(text):
            return False
        self._broadcast({"type": "guidance_queued", "text": text})
        return True

    def stop(self) -> bool:
        with self._cv:
            if self.state in {"stopped", "stopping"}:
                return False
            self._stop_requested = True
            self.auto = False
            self.state = "stopping"
            self.phase = "stopping"
            self._cv.notify_all()
        self._emit_status()
        return True

    def set_overlay(self, name: str, enabled: bool) -> bool:
        """Toggle a board overlay (grid or walkability) on the live loop."""
        with self._lock:
            if self.state in {"stopped", "stopping", "error"}:
                return False
            loop = self.loop
        if name == "grid":
            loop._show_grid = enabled
        elif name == "walkability":
            loop._show_walkability = enabled
        else:
            return False
        # Capture a fresh frame with the new overlay state so the
        # next view_board / movement tool shows it immediately.
        try:
            loop._capture()
        except Exception:
            pass
        self._broadcast({
            "type": "overlay",
            "name": name,
            "enabled": enabled,
        })
        return True

    def join(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            age_ms = int((time.monotonic() - self._last_activity) * 1000)
            return {
                "active": True,
                "run_id": self.run_id,
                "state": self.state,
                "phase": self.phase,
                "busy": self._busy,
                "steps": self.loop.step_index,
                "actions": self.loop.action_count,
                "frames": self.loop.recorder.frame_index,
                "input": getattr(getattr(self.loop, "input", None), "name", None),
                "last_activity_age_ms": age_ms,
                "first_token_ms": self.first_token_ms,
                "pending_guidance": getattr(
                    self.loop, "pending_guidance", 0
                ),
                "thinking": self.thinking,
                "content": self.content,
                "error": self.error,
                "reason": self.reason,
                "thread_alive": bool(
                    self._thread is not None and self._thread.is_alive()
                ),
                "overlays": {
                    "grid": getattr(self.loop, "_show_grid", False),
                    "walkability": getattr(self.loop, "_show_walkability", False),
                },
            }

    # ------------------------------------------------------------------
    # Event bus
    # ------------------------------------------------------------------

    def subscribe(self, after: int = 0) -> queue.Queue[dict[str, Any]]:
        subscriber: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._lock:
            for event in self._events:
                if int(event.get("seq", 0)) > after:
                    subscriber.put_nowait(event)
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            try:
                self._subscribers.remove(subscriber)
            except ValueError:
                pass

    def _broadcast(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._seq += 1
            payload = {
                "seq": self._seq,
                "at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                **event,
            }
            self._events.append(payload)
            for subscriber in list(self._subscribers):
                subscriber.put_nowait(payload)

    def _emit_status(self) -> None:
        with self._lock:
            event = {
                "type": "status",
                "phase": self.phase,
                "state": self.state,
                "busy": self._busy,
            }
        self._broadcast(event)

    def _handle_loop_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        now = time.monotonic()

        with self._lock:
            self._last_activity = now
            if event_type == "status":
                phase = str(event.get("phase") or "")
                if phase:
                    self.phase = phase
                if phase == "model_waiting":
                    self.thinking = ""
                    self.content = ""
                    self.first_token_ms = None
                    self._model_started = now
                elif phase in {"model_stopped", "turn_done"}:
                    pass
            elif event_type == "thinking":
                text = str(event.get("text") or "")
                if text:
                    if self.first_token_ms is None and self._model_started:
                        self.first_token_ms = int(
                            (now - self._model_started) * 1000
                        )
                    self.phase = "thinking"
                    self.thinking += text
            elif event_type == "content":
                text = str(event.get("text") or "")
                if text:
                    if self.first_token_ms is None and self._model_started:
                        self.first_token_ms = int(
                            (now - self._model_started) * 1000
                        )
                    self.phase = "content"
                    self.content += text
            elif event_type in {"frame", "turn", "tool_result"}:
                self._last_activity = now

        self._broadcast(event)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            with self._cv:
                while (
                    not self._stop_requested
                    and not self.auto
                    and not self._step_requested
                ):
                    if self.state not in {"paused", "pausing"}:
                        self.state = "paused"
                        self.phase = "idle"
                    self._cv.wait()

                if self._stop_requested:
                    self.state = "stopped"
                    self.phase = "stopped"
                    self.reason = "user_stop"
                    break

                self._step_requested = False
                self.state = "running"
                self.phase = "queued"
                self._busy = True
            self._emit_status()

            try:
                keep_going = self.loop.step_once()
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI
                with self._cv:
                    self._busy = False
                    self.state = "error"
                    self.phase = "error"
                    self.error = str(exc)
                self._broadcast(
                    {"type": "error", "error": str(exc), "phase": "error"}
                )
                break

            with self._cv:
                self._busy = False
                if not keep_going:
                    self.state = "stopped"
                    self.phase = "stopped"
                    self.reason = "model_stopped"
                    break
                if not self.auto:
                    self.state = "paused"
                    self.phase = "idle"
            self._emit_status()

        if self.state == "stopped":
            self._broadcast(
                {
                    "type": "stopped",
                    "reason": self.reason,
                    "state": self.state,
                }
            )
        self.loop.close()
        with self._lock:
            self._closed = True
