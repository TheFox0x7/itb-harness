"""Interactive session control for the observer/act loop.

The loop is synchronous and may block on capture, model calls, or input.  A
SessionRunner owns it on a worker thread and exposes a small thread-safe control
surface for the web UI: step, resume, pause, stop, and snapshot.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol


class SteppableLoop(Protocol):
    step_index: int
    action_count: int

    @property
    def recorder(self) -> Any: ...

    def step_once(self) -> bool: ...

    def close(self) -> None: ...


class SessionRunner:
    def __init__(self, loop: SteppableLoop) -> None:
        self.loop = loop
        self.run_id = loop.recorder.root.name

        self.state = "paused"
        self.auto = False
        self.error: str | None = None
        self.reason: str | None = None

        self._step_requested = False
        self._stop_requested = False
        self._busy = False
        self._closed = False

        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None

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
            self._cv.notify_all()
            return True

    def resume(self) -> bool:
        with self._cv:
            if self.state in {"stopped", "stopping", "error"}:
                return False
            self.auto = True
            self.state = "running"
            self._cv.notify_all()
            return True

    def pause(self) -> bool:
        with self._cv:
            if self.state in {"stopped", "stopping", "error"}:
                return False
            self.auto = False
            self.state = "pausing" if self._busy else "paused"
            self._cv.notify_all()
            return True

    def stop(self) -> bool:
        with self._cv:
            if self.state in {"stopped", "stopping"}:
                return False
            self._stop_requested = True
            self.auto = False
            self.state = "stopping"
            self._cv.notify_all()
            return True

    def join(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": True,
                "run_id": self.run_id,
                "state": self.state,
                "busy": self._busy,
                "steps": self.loop.step_index,
                "actions": self.loop.action_count,
                "frames": self.loop.recorder.frame_index,
                "error": self.error,
                "reason": self.reason,
                "thread_alive": bool(
                    self._thread is not None and self._thread.is_alive()
                ),
            }

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
                    self._cv.wait()

                if self._stop_requested:
                    self.state = "stopped"
                    self.reason = "user_stop"
                    break

                self._step_requested = False
                self.state = "running"
                self._busy = True

            try:
                keep_going = self.loop.step_once()
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI
                with self._cv:
                    self._busy = False
                    self.state = "error"
                    self.error = str(exc)
                break

            with self._cv:
                self._busy = False
                if not keep_going:
                    self.state = "stopped"
                    self.reason = "model_stopped"
                    break
                if not self.auto:
                    self.state = "paused"

        self.loop.close()
        with self._lock:
            self._closed = True
