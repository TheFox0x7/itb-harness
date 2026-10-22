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
    def __init__(self, content: object = "[]", usage: object = None) -> None:
        self.content = content
        self.usage = usage
        self.requests: list[dict[str, object]] = []
        self.chat = SimpleNamespace(complete=self._complete)

    def _complete(self, **kwargs: object):
        self.requests.append(kwargs)
        message = SimpleNamespace(content=self.content, tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], usage=self.usage
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
        self.assertTrue(client.requests[0]["prompt_cache_key"])

    def test_mistral_streams_thinking_and_accumulates_content(self) -> None:
        def event(content, usage=None):
            return SimpleNamespace(
                data=SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=content)
                        )
                    ],
                    usage=usage,
                )
            )

        usage = SimpleNamespace(
            prompt_tokens=20,
            completion_tokens=3,
            total_tokens=23,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        events = [
            event([
                {
                    "type": "thinking",
                    "thinking": [{"type": "text", "text": "look "}],
                }
            ]),
            event([
                {
                    "type": "thinking",
                    "thinking": [{"type": "text", "text": "left"}],
                }
            ]),
            event([{"type": "text", "text": "[]"}], usage=usage),
        ]

        class StreamingClient:
            def __init__(self) -> None:
                self.requests = []
                self.chat = SimpleNamespace(stream=self._stream)

            def _stream(self, **kwargs):
                self.requests.append(kwargs)
                return iter(events)

        client = StreamingClient()
        agent = MistralAgent(
            model="test", client=client, stream=True
        )
        chunks: list[str] = []
        turn = agent.complete([], None, on_thinking=chunks.append)
        self.assertEqual(chunks, ["look ", "left"])
        self.assertEqual(turn.thinking, "look left")
        self.assertEqual(turn.content, "[]")
        self.assertEqual(turn.usage["cached_tokens"], 0)
        self.assertIn("prompt_cache_key", client.requests[0])

    def test_mistral_reuses_prompt_cache_key_and_reads_cached_tokens(self) -> None:
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=5,
            total_tokens=105,
            prompt_tokens_details=SimpleNamespace(cached_tokens=64),
        )
        client = FakeMistralClient(content="[]", usage=usage)
        agent = MistralAgent(
            model="test", client=client, prompt_cache_key="test-run"
        )
        first = agent.complete([], None)
        second = agent.complete([], None)
        self.assertEqual(first.usage["cached_tokens"], 64)
        self.assertEqual(second.usage["prompt_tokens"], 100)
        self.assertEqual(
            client.requests[0]["prompt_cache_key"],
            client.requests[1]["prompt_cache_key"],
        )
        self.assertEqual(client.requests[0]["prompt_cache_key"], "test-run")

    def test_thinking_is_saved_even_when_not_printed(self) -> None:
        turns = [
            ModelTurn(content="[]", thinking="The mech is exposed on the left.")
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp))
            thinking_file = result.run_dir / "thinking" / "0001.txt"
            self.assertEqual(
                thinking_file.read_text(),
                "The mech is exposed on the left.",
            )
            record = json.loads(
                (result.run_dir / "conversation.jsonl")
                .read_text()
                .splitlines()[0]
            )
            self.assertEqual(record["thinking"], "The mech is exposed on the left.")
            self.assertEqual(pointer.events, [])

    def test_mistral_adapter_splits_thinking_from_content(self) -> None:
        client = FakeMistralClient(
            content=[
                {
                    "type": "thinking",
                    "thinking": [{"type": "text", "text": "reasoning"}],
                },
                {"type": "text", "text": "[]"},
            ]
        )
        agent = MistralAgent(model="test", client=client)
        turn = agent.complete([], None)
        self.assertEqual(turn.content, "[]")
        self.assertEqual(turn.thinking, "reasoning")


if __name__ == "__main__":
    unittest.main()
