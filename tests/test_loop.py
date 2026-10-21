from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from itb_harness.agent import MistralAgent, ModelTurn, ScriptedAgent
from itb_harness.capture import Frame, MonitorInfo, WindowInfo
from itb_harness.input import StubInput
from itb_harness.loop import HarnessLoop, LoopConfig


WINDOW = WindowInfo(
    id="{game}",
    title="Into the Breach",
    app_id="Breach",
    pid=1234,
    x=0,
    y=0,
    width=1920,
    height=1080,
    monitor="DP-4",
    active=True,
    minimized=False,
    fullscreen=True,
)


class FakeBackend:
    name = "fake"

    def __init__(self) -> None:
        self.captures = 0
        self.window = WINDOW

    def list_monitors(self) -> list[MonitorInfo]:
        return [
            MonitorInfo(
                id="1", name="DP-4", x=0, y=0, width=1920, height=1080
            )
        ]

    def list_windows(self) -> list[WindowInfo]:
        return [self.window]

    def capture_window(self, window_id: str) -> Frame:
        self.captures += 1
        return Frame(
            window=self.window,
            png=f"frame-{self.captures}".encode(),
            width=self.window.width,
            height=self.window.height,
            captured_at="2026-10-21T00:00:00+00:00",
            latency_ms=1.0,
        )

    def close(self) -> None:
        pass


class FakeMistralClient:
    def __init__(self, content: str = '[]') -> None:
        self.content = content
        self.requests: list[dict[str, object]] = []
        self.chat = SimpleNamespace(complete=self._complete)

    def _complete(self, **kwargs: object):
        self.requests.append(kwargs)
        message = SimpleNamespace(content=self.content, tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )


def run_loop(
    turns: list[ModelTurn],
    tmp: Path,
    max_steps: int = 5,
    max_images: int = 4,
    max_actions: int = 12,
    action_delay: float = 0.0,
) -> tuple[HarnessLoop, object, StubInput]:
    backend = FakeBackend()
    agent = ScriptedAgent(turns)
    pointer = StubInput()
    loop = HarnessLoop(
        backend,
        WINDOW,
        agent,
        pointer,
        LoopConfig(
            max_steps=max_steps,
            max_images=max_images,
            max_actions_per_turn=max_actions,
            action_delay=action_delay,
            out_dir=tmp,
        ),
    )
    result = loop.run()
    return loop, result, pointer


class LoopTests(unittest.TestCase):
    def test_actions_are_executed_in_order_and_capture_once(self) -> None:
        turns = [
            ModelTurn(
                content=(
                    '[{"point_2d": [500, 500], "label": "click"},'
                    ' {"point_2d": [250, 750], "label": "hover"}]'
                )
            ),
            ModelTurn(content="[]"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp))
            self.assertEqual(result.steps, 2)
            self.assertEqual(result.actions, 2)
            self.assertEqual(result.frames, 2)
            self.assertEqual(result.stopped, "model_stopped")
            self.assertEqual(
                [(event.kind, event.x, event.y) for event in pointer.events],
                [
                    ("click", 960, 540),
                    ("hover", 480, 809),
                ],
            )
            records = [
                json.loads(line)
                for line in (result.run_dir / "actions.jsonl").read_text().splitlines()
            ]
            self.assertEqual([record["label"] for record in records], ["click", "hover"])
            self.assertTrue(all(record["ok"] for record in records))
            self.assertEqual(records[0]["next_frame"], "frames/0002.png")

    def test_parse_error_does_not_click_or_recapture(self) -> None:
        turns = [ModelTurn(content="not json"), ModelTurn(content="[]")]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp))
            self.assertEqual(result.actions, 0)
            self.assertEqual(result.frames, 1)
            self.assertEqual(pointer.events, [])
            self.assertEqual(result.steps, 2)

    def test_empty_list_stops(self) -> None:
        turns = [ModelTurn(content="[]")]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp))
            self.assertEqual(result.stopped, "model_stopped")
            self.assertEqual(result.frames, 1)
            self.assertEqual(pointer.events, [])

    def test_too_many_actions_are_rejected(self) -> None:
        content = (
            '[{"point_2d": [1, 1], "label": "click"},'
            ' {"point_2d": [2, 2], "label": "click"},'
            ' {"point_2d": [3, 3], "label": "click"}]'
        )
        turns = [ModelTurn(content=content), ModelTurn(content="[]")]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp), max_actions=2)
            self.assertEqual(result.actions, 0)
            self.assertEqual(result.frames, 1)
            self.assertEqual(pointer.events, [])

    def test_history_prunes_old_images(self) -> None:
        turns = [
            ModelTurn(content='[{"point_2d": [100, 100], "label": "click"}]'),
            ModelTurn(content='[{"point_2d": [200, 200], "label": "click"}]'),
            ModelTurn(content='[{"point_2d": [300, 300], "label": "click"}]'),
            ModelTurn(content="[]"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(
                turns, Path(tmp), max_images=2
            )
            self.assertEqual(result.frames, 4)
            image_messages = [
                message
                for message in loop.messages
                if message.get("role") == "user"
                and isinstance(message.get("content"), list)
                and any(
                    chunk.get("type") == "image_url"
                    for chunk in message["content"]
                    if isinstance(chunk, dict)
                )
            ]
            self.assertLessEqual(len(image_messages), 2)

    def test_mistral_adapter_does_not_force_tools(self) -> None:
        client = FakeMistralClient(
            content='[{"point_2d": [1, 2], "label": "click"}]'
        )
        agent = MistralAgent(model="test", client=client)
        turn = agent.complete([], None)
        self.assertEqual(
            turn.content, '[{"point_2d": [1, 2], "label": "click"}]'
        )
        self.assertEqual(turn.tool_calls, ())
        self.assertNotIn("tools", client.requests[0])
        self.assertNotIn("tool_choice", client.requests[0])


if __name__ == "__main__":
    unittest.main()
