from __future__ import annotations

import tempfile
import time
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from itb_harness.agent import ModelToolCall, ModelTurn, ScriptedAgent
from itb_harness.capture import Frame, MonitorInfo, WindowInfo
from itb_harness.input import StubInput
from itb_harness.loop import HarnessLoop, LoopConfig
from itb_harness.session import SessionRunner

def tiny_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (10, 10), (0, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


WINDOW = WindowInfo(
    id="{game}",
    title="Into the Breach",
    app_id="Breach",
    pid=1,
    x=0,
    y=0,
    width=100,
    height=100,
    monitor="gamescope",
    active=True,
    minimized=False,
    fullscreen=True,
)


class FakeBackend:
    name = "fake"

    def __init__(self) -> None:
        self.captures = 0

    def list_monitors(self):
        return [MonitorInfo("1", "gamescope", 0, 0, 100, 100)]

    def list_windows(self):
        return [WINDOW]

    def capture_window(self, window_id: str) -> Frame:
        self.captures += 1
        return Frame(WINDOW, tiny_png(), 100, 100, "now", 1.0)

    def close(self) -> None:
        pass


def call(name: str, arguments: dict | None = None, id: str = "call-1"):
    return ModelToolCall(id, name, arguments or {})


def wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for session state")


class SessionRunnerTests(unittest.TestCase):
    def test_step_pauses_and_runs_one_turn(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(tool_calls=(call("click", {"point_2d": [50, 50]}),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop = HarnessLoop(
                FakeBackend(),
                WINDOW,
                ScriptedAgent(turns),
                StubInput(),
                LoopConfig(max_steps=20, action_delay=0, out_dir=Path(tmp)),
            )
            runner = SessionRunner(loop)
            runner.start()
            self.assertEqual(runner.snapshot()["state"], "paused")

            self.assertTrue(runner.step())
            wait_for(lambda: runner.snapshot()["state"] == "paused")
            self.assertEqual(runner.snapshot()["steps"], 1)
            self.assertEqual(runner.snapshot()["frames"], 1)

            self.assertTrue(runner.step())
            wait_for(lambda: runner.snapshot()["steps"] == 2)
            self.assertEqual(runner.snapshot()["actions"], 1)
            self.assertEqual(runner.snapshot()["frames"], 2)

            self.assertTrue(runner.step())
            wait_for(lambda: runner.snapshot()["state"] == "stopped")
            self.assertEqual(runner.snapshot()["reason"], "model_stopped")
            self.assertEqual(runner.snapshot()["steps"], 3)
            self.assertEqual(runner.snapshot()["frames"], 2)

    def test_resume_runs_until_model_stops(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("view_board", id="1"),)),
            ModelTurn(tool_calls=(call("view_board", id="2"),)),
            ModelTurn(tool_calls=(call("view_board", id="3"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop = HarnessLoop(
                FakeBackend(),
                WINDOW,
                ScriptedAgent(turns),
                StubInput(),
                LoopConfig(action_delay=0, out_dir=Path(tmp)),
            )
            runner = SessionRunner(loop)
            runner.start()
            self.assertTrue(runner.resume())
            wait_for(lambda: runner.snapshot()["state"] == "stopped")
            snapshot = runner.snapshot()
            self.assertEqual(snapshot["steps"], 4)
            self.assertEqual(snapshot["frames"], 3)
            self.assertEqual(snapshot["reason"], "model_stopped")

    def test_stop_while_paused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop = HarnessLoop(
                FakeBackend(),
                WINDOW,
                ScriptedAgent([ModelTurn(content="done")]),
                StubInput(),
                LoopConfig(out_dir=Path(tmp)),
            )
            runner = SessionRunner(loop)
            runner.start()
            self.assertTrue(runner.stop())
            wait_for(lambda: runner.snapshot()["state"] == "stopped")
            self.assertEqual(runner.snapshot()["reason"], "user_stop")
            runner.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
