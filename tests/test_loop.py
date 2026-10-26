from __future__ import annotations

import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

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


def tiny_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (10, 10), (0, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


WINDOW = WindowInfo(
    id="{game}",
    title="Helltaker",
    app_id="Helltaker",
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

    def __init__(self, png: bytes | None = None) -> None:
        self.captures = 0
        self.window = WINDOW
        self._png = png or tiny_png()

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
            png=self._png,
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
    max_steps: int = 10,
    max_images: int = 4,
    action_delay: float = 0.0,
    notes: NotesStore | None = None,
    strict_box_labels: bool = False,
    scaled_boxes: bool = False,
    guidance: list[str] | None = None,
    png: bytes | None = None,
    window: WindowInfo | None = None,
) -> tuple[HarnessLoop, object, StubInput]:
    win = window or WINDOW
    backend = FakeBackend(png=png)
    backend.window = win
    agent = ScriptedAgent(turns)
    pointer = StubInput()
    loop = HarnessLoop(
        backend,
        win,
        agent,
        pointer,
        LoopConfig(
            max_steps=max_steps,
            max_images=max_images,
            action_delay=action_delay,
            strict_box_labels=strict_box_labels,
            scaled_boxes=scaled_boxes,
            out_dir=tmp,
        ),
        notes=notes,
    )
    for text in guidance or []:
        loop.add_guidance(text)
    result = loop.run()
    return loop, result, pointer


class LoopTests(unittest.TestCase):
    def test_view_move_view_flow(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(tool_calls=(call("move_up"),)),
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp))
            self.assertEqual(result.steps, 4)
            self.assertEqual(result.actions, 1)
            self.assertEqual(result.frames, 3)
            self.assertEqual(result.stopped, "model_stopped")
            self.assertEqual(
                pointer.events,
                [InputEvent("key", key="up")],
            )
            records = [
                json.loads(line)
                for line in (result.run_dir / "actions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [record["tool"] for record in records],
                ["view_board", "move_up", "view_board"],
            )
            self.assertEqual(records[1]["arguments"], {})
            self.assertEqual(records[1]["frame"], "frames/0002.png")
            self.assertEqual(records[1]["source_frame"], "frames/0001.png")
            self.assertFalse(records[1]["state_changed"])
            self.assertIn("not sent", records[1]["message"])

    def test_reset_level_presses_r(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(tool_calls=(call("reset_level"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp))
            self.assertEqual(result.actions, 1)
            self.assertEqual(result.frames, 2)
            self.assertEqual(pointer.events, [InputEvent("key", key="r")])
            records = [
                json.loads(line)
                for line in (result.run_dir / "actions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(records[1]["tool"], "reset_level")
            self.assertEqual(records[1]["frame"], "frames/0002.png")
            self.assertIn("Level reset", records[1]["message"])

    def test_reset_level_clears_overlay(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(
                tool_calls=(
                    call(
                        "set_boxes",
                        {
                            "boxes": [
                                {"box_2d": [1, 1, 6, 6], "label": "demon"}
                            ]
                        },
                    ),
                )
            ),
            ModelTurn(tool_calls=(call("reset_level"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp))
            self.assertEqual(result.frames, 3)
            self.assertEqual(loop.box_overlay, [])
            raw = (result.run_dir / "frames" / "0001.png").read_bytes()
            self.assertEqual(
                (result.run_dir / "frames" / "0003.png").read_bytes(),
                raw,
            )
            records = [
                json.loads(line)
                for line in (result.run_dir / "actions.jsonl").read_text().splitlines()
            ]
            self.assertIn("overlay cleared", records[2]["message"])

    def test_all_directions_move_one_tile(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(tool_calls=(call("move_up"),)),
            ModelTurn(tool_calls=(call("move_down"),)),
            ModelTurn(tool_calls=(call("move_left"),)),
            ModelTurn(tool_calls=(call("move_right"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp))
            self.assertEqual(result.actions, 4)
            self.assertEqual(result.frames, 5)
            self.assertEqual(
                [event.key for event in pointer.events],
                ["up", "down", "left", "right"],
            )
            records = [
                json.loads(line)
                for line in (result.run_dir / "actions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [record["arguments"] for record in records[1:]],
                [{}, {}, {}, {}],
            )

    def test_boxes_overlay_persists_and_replaces(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(
                tool_calls=(
                    call(
                        "set_boxes",
                        {
                            "boxes": [
                                {
                                    "box_2d": [1, 1, 6, 6],
                                    "label": "skeleton",
                                }
                            ]
                        },
                    ),
                )
            ),
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(tool_calls=(call("move_up"),)),
            ModelTurn(tool_calls=(call("set_boxes", {"boxes": []}),)),
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp))
            self.assertEqual(result.frames, 6)
            self.assertEqual(len(loop.box_overlay), 0)
            frames = result.run_dir / "frames"
            raw = (frames / "0001.png").read_bytes()
            self.assertNotEqual((frames / "0002.png").read_bytes(), raw)
            self.assertNotEqual((frames / "0003.png").read_bytes(), raw)
            self.assertNotEqual((frames / "0004.png").read_bytes(), raw)
            self.assertEqual((frames / "0006.png").read_bytes(), raw)

    def test_add_boxes_appends_without_clearing(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(
                tool_calls=(
                    call(
                        "set_boxes",
                        {
                            "boxes": [
                                {"box_2d": [1, 1, 4, 4], "label": "skeleton"}
                            ]
                        },
                    ),
                )
            ),
            ModelTurn(
                tool_calls=(
                    call(
                        "add_boxes",
                        {
                            "boxes": [
                                {"box_2d": [5, 5, 8, 8], "label": "demon"}
                            ]
                        },
                    ),
                )
            ),
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp))
            self.assertEqual(result.frames, 4)
            self.assertEqual(
                [box.label for box in loop.box_overlay],
                ["skeleton", "demon"],
            )
            records = [
                json.loads(line)
                for line in (result.run_dir / "actions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(records[2]["tool"], "add_boxes")
            self.assertIn("overlay now has 2", records[2]["message"])

    def test_scaled_boxes_convert_to_pixels(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(
                tool_calls=(
                    call(
                        "set_boxes",
                        {
                            "boxes": [
                                {
                                    "box_2d": [1, 1, 1000, 1000],
                                    "label": "helltaker",
                                }
                            ]
                        },
                    ),
                )
            ),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(
                turns, Path(tmp), scaled_boxes=True
            )
            box = loop.box_overlay[0]
            self.assertEqual(
                (box.xmin, box.ymin, box.xmax, box.ymax),
                (0, 0, WINDOW.width - 1, WINDOW.height - 1),
            )

    def test_boxes_require_a_frame(self) -> None:
        turns = [
            ModelTurn(
                tool_calls=(
                    call(
                        "set_boxes",
                        {
                            "boxes": [
                                {"box_2d": [1, 1, 4, 4], "label": "rock"}
                            ]
                        },
                    ),
                )
            ),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp))
            self.assertEqual(result.frames, 0)
            self.assertEqual(loop.box_overlay, [])
            record = json.loads(
                (result.run_dir / "actions.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(record["error"], "invalid_arguments")
            self.assertIn("view_board", record["message"])

    def test_strict_box_labels_reject_unknown(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(
                tool_calls=(
                    call(
                        "set_boxes",
                        {
                            "boxes": [
                                {"box_2d": [1, 1, 4, 4], "label": "unknown"}
                            ]
                        },
                    ),
                )
            ),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(
                turns, Path(tmp), strict_box_labels=True
            )
            self.assertEqual(loop.box_overlay, [])
            record = json.loads(
                (result.run_dir / "actions.jsonl").read_text().splitlines()[1]
            )
            self.assertEqual(record["error"], "invalid_arguments")
            self.assertIn("must be one of", record["message"])

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
                thinking="The exit is behind the statue.",
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp))
            thinking_file = result.run_dir / "thinking" / "0001.txt"
            self.assertEqual(
                thinking_file.read_text(),
                "The exit is behind the statue.",
            )
            record = json.loads(
                (result.run_dir / "conversation.jsonl")
                .read_text()
                .splitlines()[0]
            )
            self.assertEqual(
                record["thinking"], "The exit is behind the statue."
            )
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

    def test_prior_notes_are_loaded_into_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            notes_path = Path(tmp) / "notes.md"
            notes = NotesStore(notes_path)
            notes.append("Old fact: the spikes are deadly.", run_id="old")
            loop, result, pointer = run_loop(
                [ModelTurn(content="done")], Path(tmp), notes=notes
            )
            system = loop.messages[0]
            self.assertEqual(system["role"], "system")
            self.assertIn("Old fact: the spikes are deadly.", system["content"])
            self.assertIn("Persistent notes from previous runs", system["content"])

    def test_guidance_is_added_to_next_prompt(self) -> None:
        turns = [ModelTurn(content="done")]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(
                turns,
                Path(tmp),
                guidance=["go left toward the rock", "avoid the spikes"],
            )
            user_messages = [
                message["content"]
                for message in loop.messages
                if message.get("role") == "user"
            ]
            combined = "\n".join(user_messages)
            self.assertIn("go left toward the rock", combined)
            self.assertIn("avoid the spikes", combined)
            self.assertEqual(loop.pending_guidance, 0)
            guidance_records = [
                json.loads(line)
                for line in (
                    result.run_dir / "guidance.jsonl"
                ).read_text().splitlines()
            ]
            self.assertEqual(
                [record["text"] for record in guidance_records],
                ["go left toward the rock", "avoid the spikes"],
            )
            self.assertTrue(
                all(record["step"] == 1 for record in guidance_records)
            )

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
        contents: list[str] = []
        turn = agent.complete(
            [],
            None,
            on_thinking=chunks.append,
            on_content=contents.append,
        )
        self.assertEqual(chunks, ["look ", "left"])
        self.assertEqual(contents, ["[]"])
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
