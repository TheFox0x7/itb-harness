from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from itb_harness.agent import (
    MistralAgent,
    ModelToolCall,
    ModelTurn,
    ScriptedAgent,
)
from itb_harness.capture import Frame, MonitorInfo, WindowInfo
from itb_harness.input import InputEvent, StubInput
from itb_harness.loop import HarnessLoop, LoopConfig
from itb_harness.notes import NotesStore


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
            captured_at="2026-10-22T00:00:00+00:00",
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


def call(name: str, arguments: dict | None = None, id: str = "call-1"):
    return ModelToolCall(id=id, name=name, arguments=arguments or {})


def run_loop(
    turns: list[ModelTurn],
    tmp: Path,
    max_steps: int = 5,
    max_images: int = 4,
    action_delay: float = 0.0,
    notes: NotesStore | None = None,
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
            action_delay=action_delay,
            out_dir=tmp,
        ),
        notes=notes,
    )
    result = loop.run()
    return loop, result, pointer


class LoopTests(unittest.TestCase):
    def test_view_click_view_flow(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(tool_calls=(call("click", {"point_2d": [960, 540]}),)),
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp))
            self.assertEqual(result.steps, 4)
            self.assertEqual(result.actions, 1)
            self.assertEqual(result.frames, 2)
            self.assertEqual(result.stopped, "model_stopped")
            self.assertEqual(
                [(event.kind, event.x, event.y) for event in pointer.events],
                [("click", 960, 540)],
            )
            records = [
                json.loads(line)
                for line in (result.run_dir / "actions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [record["tool"] for record in records],
                ["view_board", "click", "view_board"],
            )
            self.assertEqual(records[0]["frame"], "frames/0001.png")
            self.assertIsNone(records[1]["frame"])
            self.assertEqual(records[1]["source_frame"], "frames/0001.png")
            self.assertEqual(records[2]["frame"], "frames/0002.png")

    def test_click_before_view_is_rejected(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("click", {"point_2d": [1, 1]}),)),
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp))
            self.assertEqual(result.actions, 0)
            self.assertEqual(result.frames, 1)
            self.assertEqual(pointer.events, [])
            records = [
                json.loads(line)
                for line in (result.run_dir / "actions.jsonl").read_text().splitlines()
            ]
            self.assertFalse(records[0]["ok"])
            self.assertEqual(records[0]["error"], "no_frame")
            self.assertIn("view_board", records[0]["message"])

    def test_inspect_hovers_and_captures(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(
                tool_calls=(call("inspect", {"point_2d": [480, 809]}),)
            ),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp))
            self.assertEqual(result.actions, 0)
            self.assertEqual(result.frames, 2)
            self.assertEqual(
                [(event.kind, event.x, event.y) for event in pointer.events],
                [("hover", 480, 809)],
            )

    def test_keyboard_and_modifier_tools(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(tool_calls=(call("select_unit", {"number": 1}),)),
            ModelTurn(tool_calls=(call("weapon", {"number": 2}),)),
            ModelTurn(tool_calls=(call("repair"),)),
            ModelTurn(
                tool_calls=(
                    call(
                        "inspect",
                        {"point_2d": [960, 540], "with_details": True},
                    ),
                )
            ),
            ModelTurn(tool_calls=(call("attack_order"),)),
            ModelTurn(tool_calls=(call("end_turn"),)),
            ModelTurn(tool_calls=(call("undo_move"),)),
            ModelTurn(tool_calls=(call("reset_turn"),)),
            ModelTurn(tool_calls=(call("cancel"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(
                turns, Path(tmp), max_steps=12
            )
            self.assertEqual(result.frames, 3)
            self.assertEqual(
                pointer.events,
                [
                    InputEvent("key", key="a"),
                    InputEvent("key", key="2"),
                    InputEvent("key", key="r"),
                    InputEvent("hover", 960, 540),
                    InputEvent("key_down", key="ctrl"),
                    InputEvent("key_up", key="ctrl"),
                    InputEvent("key_down", key="alt"),
                    InputEvent("key_up", key="alt"),
                    InputEvent("key", key="space"),
                    InputEvent("key", key="shift"),
                    InputEvent("key", key="backspace"),
                    InputEvent("right_click"),
                ],
            )

    def test_out_of_bounds_pixel_is_rejected(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(tool_calls=(call("click", {"point_2d": [1920, 10]}),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp))
            self.assertEqual(pointer.events, [])
            records = [
                json.loads(line)
                for line in (result.run_dir / "actions.jsonl").read_text().splitlines()
            ]
            self.assertFalse(records[1]["ok"])
            self.assertEqual(records[1]["error"], "invalid_arguments")
            self.assertIn("outside the screenshot", records[1]["message"])

    def test_invalid_weapon_and_unit_are_rejected(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("weapon", {"number": 3}),)),
            ModelTurn(tool_calls=(call("select_unit", {"number": 6}),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp))
            self.assertEqual(pointer.events, [])
            records = [
                json.loads(line)
                for line in (result.run_dir / "actions.jsonl").read_text().splitlines()
            ]
            self.assertFalse(records[0]["ok"])
            self.assertFalse(records[1]["ok"])
            self.assertEqual(records[0]["error"], "invalid_arguments")
            self.assertEqual(records[1]["error"], "invalid_arguments")

    def test_extra_tool_calls_are_rejected(self) -> None:
        turns = [
            ModelTurn(
                tool_calls=(
                    call("view_board", id="1"),
                    call("view_board", id="2"),
                )
            ),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp))
            self.assertEqual(result.frames, 1)
            tool_messages = [
                message
                for message in loop.messages
                if message.get("role") == "tool"
            ]
            self.assertEqual(len(tool_messages), 2)
            self.assertIn("one_tool_per_turn", str(tool_messages[0]["content"]))

    def test_thinking_is_saved_even_when_not_printed(self) -> None:
        turns = [
            ModelTurn(
                content="done",
                thinking="The mech is exposed on the left.",
            )
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

    def test_history_prunes_old_tool_images(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("view_board", id="1"),)),
            ModelTurn(tool_calls=(call("view_board", id="2"),)),
            ModelTurn(tool_calls=(call("view_board", id="3"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp), max_images=2)
            self.assertEqual(result.frames, 3)
            image_tool_messages = [
                message
                for message in loop.messages
                if message.get("role") == "tool"
                and isinstance(message.get("content"), list)
                and any(
                    chunk.get("type") == "image_url"
                    for chunk in message["content"]
                    if isinstance(chunk, dict)
                )
            ]
            self.assertLessEqual(len(image_tool_messages), 2)

    def test_note_tool_persists_for_future_runs(self) -> None:
        turns = [
            ModelTurn(
                tool_calls=(
                    call(
                        "note",
                        {"text": "Firefly: attacks in a line; use a shield."},
                    ),
                )
            ),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            notes_path = Path(tmp) / "notes.md"
            notes = NotesStore(notes_path)
            loop, result, pointer = run_loop(
                turns, Path(tmp), notes=notes
            )
            self.assertEqual(result.frames, 0)
            self.assertEqual(result.actions, 0)
            saved = notes_path.read_text()
            self.assertIn("Firefly", saved)
            self.assertIn("shield", saved)

    def test_prior_notes_are_loaded_into_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            notes_path = Path(tmp) / "notes.md"
            notes = NotesStore(notes_path)
            notes.append("Old fact: the train mission is fragile.", run_id="old")
            loop, result, pointer = run_loop(
                [ModelTurn(content="done")], Path(tmp), notes=notes
            )
            system = loop.messages[0]
            self.assertEqual(system["role"], "system")
            self.assertIn("Old fact: the train mission is fragile.", system["content"])
            self.assertIn("Persistent notes from previous runs", system["content"])

    def test_mistral_adapter_does_not_force_tools_when_none(self) -> None:
        client = FakeMistralClient(content="hello")
        agent = MistralAgent(model="test", client=client)
        turn = agent.complete([], None)
        self.assertEqual(turn.content, "hello")
        self.assertNotIn("tools", client.requests[0])
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
        agent = MistralAgent(model="test", client=client, stream=True)
        chunks: list[str] = []
        turn = agent.complete([], None, on_thinking=chunks.append)
        self.assertEqual(chunks, ["look ", "left"])
        self.assertEqual(turn.thinking, "look left")
        self.assertEqual(turn.content, "[]")
        self.assertEqual(turn.usage["cached_tokens"], 0)
        self.assertIn("prompt_cache_key", client.requests[0])

    def test_mistral_streams_tool_call_deltas(self) -> None:
        def event(tool_calls=None):
            return SimpleNamespace(
                data=SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None, tool_calls=tool_calls
                            )
                        )
                    ],
                    usage=None,
                )
            )

        def tool_delta(index, call_id, name, arguments):
            return SimpleNamespace(
                index=index,
                id=call_id,
                function=SimpleNamespace(name=name, arguments=arguments),
            )

        events = [
            event([tool_delta(0, "call-1", "view_board", "{")]),
            event([tool_delta(0, None, None, "}")]),
        ]

        class StreamingClient:
            def __init__(self) -> None:
                self.chat = SimpleNamespace(stream=self._stream)

            def _stream(self, **kwargs):
                return iter(events)

        agent = MistralAgent(
            model="test", client=StreamingClient(), stream=True
        )
        turn = agent.complete([], [{"type": "function"}])
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.tool_calls[0].name, "view_board")
        self.assertEqual(turn.tool_calls[0].arguments, {})


if __name__ == "__main__":
    unittest.main()
