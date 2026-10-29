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
from itb_harness.loop import HarnessLoop, LoopConfig, _frames_differ
from itb_harness.notes import NotesStore


def message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict)
        )
    return str(content)


def tiny_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (10, 10), (0, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def different_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (10, 10), (255, 255, 255)).save(buffer, format="PNG")
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


class SequenceBackend(FakeBackend):
    """Returns a different PNG on each capture, cycling through a list."""

    def __init__(self, pngs: list[bytes]) -> None:
        super().__init__(png=pngs[0] if pngs else tiny_png())
        self._pngs = pngs
        self._index = 0

    def capture_window(self, window_id: str) -> Frame:
        self.captures += 1
        if self._pngs:
            self._png = self._pngs[self._index % len(self._pngs)]
            self._index += 1
        return Frame(
            window=self.window,
            png=self._png,
            width=self.window.width,
            height=self.window.height,
            captured_at="2026-10-22T00:00:00+00:00",
            latency_ms=1.0,
        )


class FakeMistralClient:
    def __init__(
        self,
        content: object = "[]",
        usage: object = None,
        output: list[object] | None = None,
    ) -> None:
        self.content = content
        self.usage = usage
        self.output = output
        self.requests: list[dict[str, object]] = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: object):
        self.requests.append(kwargs)
        output = self.output
        if output is None:
            output = [
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text", text=self.content
                        )
                    ],
                )
            ]
        return SimpleNamespace(
            id="resp-1", output=output, usage=self.usage
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

    def test_move_tools_do_not_return_screenshots(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(tool_calls=(call("move_up"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, _result, _pointer = run_loop(turns, Path(tmp))
            tool_messages = [
                message
                for message in loop.messages
                if message.get("role") == "tool"
                and message.get("name") == "move_up"
            ]
            self.assertEqual(len(tool_messages), 1)
            self.assertIsInstance(tool_messages[0]["content"], str)
            self.assertNotIn("Screenshot attached", tool_messages[0]["content"])
            self.assertIn("BLOCKED", tool_messages[0]["content"])

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

    def test_initial_messages_include_hazard_reference_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loop, _result, _pointer = run_loop(
                [ModelTurn(content="done")], Path(tmp)
            )
            references = loop.messages[1]["content"]
            text = message_text(references)
            self.assertIn("Spikes - extended", text)
            self.assertIn("Spikes - hidden", text)
            self.assertIn("Passable floor - paired tiles", text)
            self.assertIn("Passable floor - single tile", text)
            self.assertIn("Rock", text)
            self.assertIn("Helltaker", text)
            self.assertIn("Impassable rocks", text)
            self.assertIn("Impassable wall", text)
            images = [
                part
                for part in references
                if isinstance(part, dict) and part.get("type") == "image_url"
            ]
            self.assertEqual(len(images), 8)

    def test_guidance_is_added_to_next_prompt(self) -> None:
        turns = [ModelTurn(content="done")]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(
                turns,
                Path(tmp),
                guidance=["go left toward the rock", "avoid the spikes"],
            )
            user_messages = [
                message_text(message["content"])
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
        turn = agent.complete(
            [
                {"role": "system", "content": "be nice"},
                {"role": "user", "content": "hi"},
            ],
            None,
        )
        self.assertEqual(turn.content, "hello")
        request = client.requests[0]
        self.assertNotIn("tools", request)
        self.assertTrue(request["prompt_cache_key"])
        self.assertEqual(request["instructions"], "be nice")
        self.assertEqual(request["input"], [{"role": "user", "content": "hi"}])

    def test_mistral_converts_history_and_tools_for_responses(self) -> None:
        client = FakeMistralClient(content="")
        agent = MistralAgent(model="test", client=client)
        image_url = "data:image/png;base64,AAAA"
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "start"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "view_board",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "view_board",
                "content": [
                    {"type": "text", "text": '{"ok": true}'},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            },
        ]
        agent.complete(
            messages,
            [
                {
                    "type": "function",
                    "function": {
                        "name": "view_board",
                        "description": "look",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                }
            ],
        )

        request = client.requests[0]
        self.assertEqual(request["instructions"], "system prompt")
        self.assertEqual(
            request["tools"],
            [
                {
                    "type": "function",
                    "name": "view_board",
                    "description": "look",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                }
            ],
        )
        self.assertEqual(request["tool_choice"], "required")
        self.assertFalse(request["parallel_tool_calls"])
        self.assertEqual(
            request["input"],
            [
                {"role": "user", "content": "start"},
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "view_board",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": [
                        {"type": "input_text", "text": '{"ok": true}'},
                        {
                            "type": "input_image",
                            "image_url": image_url,
                            "detail": "auto",
                        },
                    ],
                },
            ],
        )

    def test_mistral_parses_responses_function_calls(self) -> None:
        output = [
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(type="output_text", text="moving")
                ],
            ),
            SimpleNamespace(
                type="function_call",
                call_id="call-9",
                name="move_up",
                arguments='{"steps": 1}',
            ),
        ]
        client = FakeMistralClient(output=output)
        agent = MistralAgent(model="test", client=client)
        turn = agent.complete([{"role": "user", "content": "go"}], None)
        self.assertEqual(turn.content, "moving")
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.tool_calls[0].id, "call-9")
        self.assertEqual(turn.tool_calls[0].name, "move_up")
        self.assertEqual(turn.tool_calls[0].arguments, {"steps": 1})

    def test_mistral_streams_thinking_and_accumulates_content(self) -> None:
        def event(**data):
            return SimpleNamespace(data=SimpleNamespace(**data))

        usage = SimpleNamespace(
            input_tokens=20,
            output_tokens=3,
            total_tokens=23,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
            output_tokens_details=SimpleNamespace(reasoning_tokens=2),
        )
        output = [
            SimpleNamespace(
                type="reasoning",
                content=[
                    SimpleNamespace(
                        type="reasoning_text", text="look left"
                    )
                ],
                summary=[],
            ),
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(type="output_text", text="[]")
                ],
            ),
        ]
        events = [
            event(type="response.reasoning_text.delta", delta="look "),
            event(type="response.reasoning_text.delta", delta="left"),
            event(type="response.output_text.delta", delta="[]"),
            event(
                type="response.completed",
                response=SimpleNamespace(output=output, usage=usage),
            ),
        ]

        class StreamingClient:
            def __init__(self) -> None:
                self.requests = []
                self.responses = SimpleNamespace(create=self._create)

            def _create(self, **kwargs):
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
        self.assertTrue(client.requests[0]["stream"])

    def test_mistral_streams_tool_call_deltas(self) -> None:
        def event(**data):
            return SimpleNamespace(data=SimpleNamespace(**data))

        events = [
            event(
                type="response.output_item.added",
                output_index=0,
                item=SimpleNamespace(
                    type="function_call",
                    id="fc-1",
                    call_id="call-1",
                    name="view_board",
                    arguments="",
                ),
            ),
            event(
                type="response.function_call_arguments.delta",
                item_id="fc-1",
                output_index=0,
                delta="{",
            ),
            event(
                type="response.function_call_arguments.delta",
                item_id="fc-1",
                output_index=0,
                delta="}",
            ),
        ]

        class StreamingClient:
            def __init__(self) -> None:
                self.responses = SimpleNamespace(create=self._create)

            def _create(self, **kwargs):
                return iter(events)

        agent = MistralAgent(
            model="test", client=StreamingClient(), stream=True
        )
        turn = agent.complete(
            [],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "view_board",
                        "description": "",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.tool_calls[0].id, "call-1")
        self.assertEqual(turn.tool_calls[0].name, "view_board")
        self.assertEqual(turn.tool_calls[0].arguments, {})


class FrameDiffTests(unittest.TestCase):
    def test_identical_frames_do_not_differ(self) -> None:
        png = tiny_png()
        self.assertFalse(_frames_differ(png, png))

    def test_different_frames_differ(self) -> None:
        self.assertTrue(_frames_differ(tiny_png(), different_png()))

    def test_different_sizes_differ(self) -> None:
        buf_a = BytesIO()
        Image.new("RGB", (10, 10), (0, 0, 0)).save(buf_a, format="PNG")
        buf_b = BytesIO()
        Image.new("RGB", (20, 20), (0, 0, 0)).save(buf_b, format="PNG")
        self.assertTrue(_frames_differ(buf_a.getvalue(), buf_b.getvalue()))

    def test_invalid_png_assumes_differ(self) -> None:
        self.assertTrue(_frames_differ(b"not a png", b"also not a png"))


class BlockedMoveTests(unittest.TestCase):
    def test_blocked_move_when_frames_identical(self) -> None:
        """When the screenshot doesn't change after a move, it was blocked."""
        png = tiny_png()
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(tool_calls=(call("move_up"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            win = WINDOW
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
                    max_steps=10,
                    max_images=4,
                    action_delay=0.0,
                    out_dir=Path(tmp),
                ),
            )
            result = loop.run()
            records = [
                json.loads(line)
                for line in (result.run_dir / "actions.jsonl").read_text().splitlines()
            ]
            move_record = records[1]
            self.assertEqual(move_record["tool"], "move_up")
            self.assertTrue(move_record["blocked"])
            self.assertIn("BLOCKED", move_record["message"])
            self.assertFalse(move_record["state_changed"])

    def test_move_verification_retries_before_blocking(self) -> None:
        """A delayed animation should not be reported as a blocked move."""
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(tool_calls=(call("move_up"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            backend = SequenceBackend(
                [tiny_png(), tiny_png(), different_png()]
            )
            backend.window = WINDOW
            loop = HarnessLoop(
                backend,
                WINDOW,
                ScriptedAgent(turns),
                StubInput(),
                LoopConfig(
                    max_steps=10,
                    action_delay=0.0,
                    move_verify_attempts=2,
                    move_verify_delay=0.0,
                    out_dir=Path(tmp),
                ),
            )
            result = loop.run()
            records = [
                json.loads(line)
                for line in (result.run_dir / "actions.jsonl")
                .read_text()
                .splitlines()
            ]
            move_record = records[1]
            self.assertEqual(move_record["tool"], "move_up")
            self.assertFalse(move_record["blocked"])
            self.assertIn("Step consumed", move_record["message"])

    def test_successful_move_when_frames_differ(self) -> None:
        """When the screenshot changes after a move, a step was consumed."""
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(tool_calls=(call("move_up"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            win = WINDOW
            # Alternate between two different PNGs so the move changes the frame
            backend = SequenceBackend([tiny_png(), different_png()])
            backend.window = win
            agent = ScriptedAgent(turns)
            pointer = StubInput()
            loop = HarnessLoop(
                backend,
                win,
                agent,
                pointer,
                LoopConfig(
                    max_steps=10,
                    max_images=4,
                    action_delay=0.0,
                    out_dir=Path(tmp),
                ),
            )
            result = loop.run()
            records = [
                json.loads(line)
                for line in (result.run_dir / "actions.jsonl").read_text().splitlines()
            ]
            move_record = records[1]
            self.assertEqual(move_record["tool"], "move_up")
            self.assertFalse(move_record["blocked"])
            self.assertIn("Step consumed", move_record["message"])

    def test_reset_clears_prev_frame(self) -> None:
        """After reset, the next move compares against the post-reset frame."""
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(tool_calls=(call("reset_level"),)),
            ModelTurn(tool_calls=(call("move_up"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            win = WINDOW
            # Sequence: view_board → reset frame → move frame (different from reset)
            backend = SequenceBackend([tiny_png(), different_png(), tiny_png()])
            backend.window = win
            agent = ScriptedAgent(turns)
            pointer = StubInput()
            loop = HarnessLoop(
                backend,
                win,
                agent,
                pointer,
                LoopConfig(
                    max_steps=10,
                    max_images=4,
                    action_delay=0.0,
                    out_dir=Path(tmp),
                ),
            )
            result = loop.run()
            records = [
                json.loads(line)
                for line in (result.run_dir / "actions.jsonl").read_text().splitlines()
            ]
            move_record = records[2]
            self.assertEqual(move_record["tool"], "move_up")
            # After reset, the move compares against the reset frame.
            # Since the move frame differs from the reset frame, it's not blocked.
            self.assertFalse(move_record["blocked"])
            self.assertIn("Step consumed", move_record["message"])

    def test_tool_result_includes_blocked_in_message(self) -> None:
        """The tool message sent to the model should mention BLOCKED status."""
        png = tiny_png()
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(tool_calls=(call("move_up"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            win = WINDOW
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
                    max_steps=10,
                    max_images=4,
                    action_delay=0.0,
                    out_dir=Path(tmp),
                ),
            )
            loop.run()
            # Find the tool message for the move_up call
            tool_messages = [
                m for m in loop.messages
                if m.get("role") == "tool" and m.get("name") == "move_up"
            ]
            self.assertEqual(len(tool_messages), 1)
            content = tool_messages[0]["content"]
            if isinstance(content, list):
                text = content[0]["text"]
            else:
                text = content
            payload = json.loads(text)
            self.assertTrue(payload["blocked"])
            self.assertIn("BLOCKED", payload["message"])


class LevelMapTests(unittest.TestCase):
    def test_level_map_is_stored_and_echoed_with_tool_results(self) -> None:
        turns = [
            ModelTurn(
                tool_calls=(
                    call("level_map", {"rows": ["XXX", "XHD"]}),
                )
            ),
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp))
            self.assertEqual(loop.level_map, [["X", "X", "X"], ["X", "H", "D"]])
            tool_messages = [
                message
                for message in loop.messages
                if message.get("role") == "tool"
            ]
            self.assertIn(
                "[X] [X] [X]",
                str(tool_messages[0]["content"]),
            )
            view_text = next(
                chunk["text"]
                for chunk in tool_messages[1]["content"]
                if chunk.get("type") == "text"
            )
            self.assertIn("Current level map:", view_text)
            self.assertIn("[X] [H] [D]", view_text)
            map_records = [
                json.loads(line)
                for line in (result.run_dir / "level_map.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                map_records[0],
                {"step": 1, "rows": [["X", "X", "X"], ["X", "H", "D"]]},
            )

    def test_level_map_supports_cell_updates(self) -> None:
        turns = [
            ModelTurn(
                tool_calls=(
                    call("level_map", {"rows": ["XHX", "X.D"]}),
                )
            ),
            ModelTurn(
                tool_calls=(
                    call(
                        "level_map",
                        {"updates": [{"row": 0, "col": 1, "symbol": "."}]},
                    ),
                )
            ),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp))
            self.assertEqual(loop.level_map, [["X", ".", "X"], ["X", ".", "D"]])

    def test_validate_solution_accepts_valid_plan(self) -> None:
        turns = [
            ModelTurn(
                tool_calls=(
                    call("level_map", {"rows": ["XD", "H."]}),
                )
            ),
            ModelTurn(
                tool_calls=(
                    call(
                        "validate_solution",
                        {
                            "moves": ["move_right"],
                            "max_steps": 1,
                            "spike_mode": "static",
                        },
                    ),
                )
            ),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            _loop, result, _pointer = run_loop(turns, Path(tmp))
            records = [
                json.loads(line)
                for line in (result.run_dir / "actions.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(records[1]["tool"], "validate_solution")
            self.assertTrue(records[1]["ok"])
            self.assertIn("Solution valid", records[1]["message"])
            self.assertIn("adjacent to the demon", records[1]["message"])

    def test_execute_plan_runs_moves_and_updates_map(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(
                tool_calls=(
                    call("level_map", {"rows": ["H.D"]}),
                )
            ),
            ModelTurn(
                tool_calls=(
                    call(
                        "execute_plan",
                        {
                            "moves": ["move_right"],
                            "max_steps": 1,
                            "spike_mode": "static",
                        },
                    ),
                )
            ),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            backend = SequenceBackend([tiny_png(), different_png()])
            backend.window = WINDOW
            loop = HarnessLoop(
                backend,
                WINDOW,
                ScriptedAgent(turns),
                StubInput(),
                LoopConfig(max_steps=10, action_delay=0, out_dir=Path(tmp)),
            )
            result = loop.run()
            self.assertEqual(loop.action_count, 1)
            self.assertEqual(loop.level_map, [[".", "H", "D"]])
            records = [
                json.loads(line)
                for line in (result.run_dir / "actions.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(records[2]["tool"], "execute_plan")
            self.assertTrue(records[2]["ok"])
            self.assertIn("Executed 1/1 planned moves", records[2]["message"])

    def test_toggle_map_overlay_returns_marked_screenshot(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("level_map", {"rows": ["XHD"]}),)),
            ModelTurn(
                tool_calls=(
                    call("toggle_map_overlay", {"enabled": True}),
                )
            ),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, _result, _pointer = run_loop(turns, Path(tmp))
            self.assertTrue(loop.show_level_map_overlay)
            messages = [
                message
                for message in loop.messages
                if message.get("role") == "tool"
                and message.get("name") == "toggle_map_overlay"
            ]
            self.assertEqual(len(messages), 1)
            self.assertIsInstance(messages[0]["content"], list)
            self.assertTrue(
                any(
                    part.get("type") == "image_url"
                    for part in messages[0]["content"]
                    if isinstance(part, dict)
                )
            )

    def test_press_space_is_not_a_move(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("press_space"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp))
            self.assertEqual(pointer.events, [InputEvent("key", key="space")])
            self.assertEqual(result.frames, 0)
            self.assertEqual(loop.action_count, 1)
            records = [
                json.loads(line)
                for line in (result.run_dir / "actions.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(records[0]["tool"], "press_space")
            self.assertIn("Pressed space", records[0]["message"])

    def test_validate_solution_rejects_blocked_plan(self) -> None:
        turns = [
            ModelTurn(
                tool_calls=(
                    call("level_map", {"rows": ["HXD"]}),
                )
            ),
            ModelTurn(
                tool_calls=(
                    call(
                        "validate_solution",
                        {
                            "moves": ["move_right"],
                            "max_steps": 1,
                            "spike_mode": "static",
                        },
                    ),
                )
            ),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            _loop, result, _pointer = run_loop(turns, Path(tmp))
            records = [
                json.loads(line)
                for line in (result.run_dir / "actions.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(records[1]["error"], "invalid_solution")
            self.assertIn("wall", records[1]["message"])

    def test_successful_move_updates_level_map(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(
                tool_calls=(
                    call("level_map", {"rows": ["X.X", "XHX"]}),
                )
            ),
            ModelTurn(tool_calls=(call("move_up"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            backend = SequenceBackend([tiny_png(), different_png()])
            backend.window = WINDOW
            loop = HarnessLoop(
                backend,
                WINDOW,
                ScriptedAgent(turns),
                StubInput(),
                LoopConfig(max_steps=10, action_delay=0, out_dir=Path(tmp)),
            )
            result = loop.run()
            self.assertEqual(
                loop.level_map,
                [["X", "H", "X"], ["X", ".", "X"]],
            )
            records = [
                json.loads(line)
                for line in (result.run_dir / "level_map.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                records[-1],
                {
                    "step": 3,
                    "rows": [["X", "H", "X"], ["X", ".", "X"]],
                },
            )

    def test_reset_level_clears_level_map(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("level_map", {"rows": ["XH"]}),)),
            ModelTurn(tool_calls=(call("reset_level"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(turns, Path(tmp))
            self.assertIsNone(loop.level_map)
            record = json.loads(
                (result.run_dir / "actions.jsonl").read_text().splitlines()[1]
            )
            self.assertIn("level map cleared", record["message"])


class MoveHistoryTests(unittest.TestCase):
    def test_history_in_tool_message(self) -> None:
        """Movement tool results should include recent move history."""
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(tool_calls=(call("move_up"),)),
            ModelTurn(tool_calls=(call("move_up"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            win = WINDOW
            backend = SequenceBackend([tiny_png(), different_png(), tiny_png(), different_png()])
            backend.window = win
            agent = ScriptedAgent(turns)
            pointer = StubInput()
            loop = HarnessLoop(
                backend,
                win,
                agent,
                pointer,
                LoopConfig(
                    max_steps=10,
                    max_images=4,
                    action_delay=0.0,
                    out_dir=Path(tmp),
                ),
            )
            loop.run()
            # Check that move history was recorded
            self.assertEqual(len(loop._move_history), 2)
            # Check that the second move's message includes history
            tool_messages = [
                m for m in loop.messages
                if m.get("role") == "tool" and m.get("name") == "move_up"
            ]
            self.assertEqual(len(tool_messages), 2)
            content = tool_messages[1]["content"]
            if isinstance(content, list):
                text = content[0]["text"]
            else:
                text = content
            payload = json.loads(text)
            self.assertIn("Recent moves:", payload["message"])

    def test_history_shows_blocked(self) -> None:
        """Blocked moves should show ✗ in history."""
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(tool_calls=(call("move_up"),)),  # blocked (same frame)
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            win = WINDOW
            backend = FakeBackend(png=tiny_png())
            backend.window = win
            agent = ScriptedAgent(turns)
            pointer = StubInput()
            loop = HarnessLoop(
                backend,
                win,
                agent,
                pointer,
                LoopConfig(
                    max_steps=10,
                    max_images=4,
                    action_delay=0.0,
                    out_dir=Path(tmp),
                ),
            )
            loop.run()
            tool_messages = [
                m for m in loop.messages
                if m.get("role") == "tool" and m.get("name") == "move_up"
            ]
            content = tool_messages[0]["content"]
            if isinstance(content, list):
                text = content[0]["text"]
            else:
                text = content
            payload = json.loads(text)
            # First move has no history yet (or just itself)
            # The blocked status should be in the message
            self.assertTrue(payload["blocked"])


if __name__ == "__main__":
    unittest.main()
