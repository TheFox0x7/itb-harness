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
from itb_harness.grid import default_grid
from itb_harness.input import InputEvent, StubInput
from itb_harness.loop import HarnessLoop, LoopConfig
from itb_harness.notes import NotesStore


def tiny_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (10, 10), (0, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def deployment_png(width: int = 1280, height: int = 720) -> bytes:
    """Synthetic deployment screenshot with tan drop-zone tiles."""
    from PIL import ImageDraw

    grid = default_grid(width, height)
    img = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    # Board diamond (bright area).
    cx = grid.origin_x
    cy = grid.origin_y + 7 * grid.tile_h2
    half_w = int(4 * 2 * grid.tile_w2)
    half_h = int(4 * 2 * grid.tile_h2)
    draw.polygon(
        [(cx, cy - half_h), (cx + half_w, cy), (cx, cy + half_h), (cx - half_w, cy)],
        fill=(60, 65, 55),
    )

    # Tan drop-zone tiles with gaps (so erosion separates them).
    tan = (140, 140, 90)
    for row, col in [(3, 3), (3, 4), (4, 4), (4, 5), (5, 5)]:
        tx, ty = grid.tile_center(row, col)
        tw = int(2 * grid.tile_w2 * 0.88)
        th = int(2 * grid.tile_h2 * 0.88)
        draw.polygon(
            [(tx, ty - th // 2), (tx + tw // 2, ty), (tx, ty + th // 2), (tx - tw // 2, ty)],
            fill=tan,
        )

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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


WINDOW_1280 = WindowInfo(
    id="{game}",
    title="Into the Breach",
    app_id="Breach",
    pid=1234,
    x=86,
    y=48,
    width=1280,
    height=720,
    monitor="gamescope",
    active=True,
    minimized=False,
    fullscreen=False,
)


def run_loop(
    turns: list[ModelTurn],
    tmp: Path,
    max_steps: int = 5,
    max_images: int = 4,
    action_delay: float = 0.0,
    notes: NotesStore | None = None,
    normalized: bool = False,
    snap_to_tile: bool = False,
    grid_overlay: bool = False,
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
            normalized_coordinates=normalized,
            snap_to_tile=snap_to_tile,
            grid_overlay=grid_overlay,
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
            self.assertEqual(result.frames, 3)
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
            self.assertEqual(records[1]["frame"], "frames/0002.png")
            self.assertEqual(records[1]["source_frame"], "frames/0001.png")
            self.assertFalse(records[1]["state_changed"])
            self.assertIn("not sent", records[1]["message"])
            self.assertNotIn("at screen", records[1]["message"])
            self.assertEqual(records[2]["frame"], "frames/0003.png")

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

    def test_inspect_hovers_with_details_and_captures(self) -> None:
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
                pointer.events,
                [
                    InputEvent("key_down", key="ctrl"),
                    InputEvent("hover", 480, 809),
                    InputEvent("key_up", key="ctrl"),
                ],
            )

    def test_hover_only_hovers(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(
                tool_calls=(call("hover", {"point_2d": [480, 809]}),)
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
                    call("inspect", {"point_2d": [960, 540]}),
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
                    InputEvent("key_down", key="ctrl"),
                    InputEvent("hover", 960, 540),
                    InputEvent("key_up", key="ctrl"),
                    InputEvent("key_down", key="alt"),
                    InputEvent("key_up", key="alt"),
                    InputEvent("key", key="space"),
                    InputEvent("key", key="shift"),
                    InputEvent("key", key="backspace"),
                    InputEvent("right_click"),
                ],
            )

    def test_normalized_coordinates_are_converted_to_pixels(self) -> None:
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(tool_calls=(call("click", {"point_2d": [500, 500]}),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(
                turns, Path(tmp), normalized=True
            )
            self.assertEqual(
                [(event.kind, event.x, event.y) for event in pointer.events],
                [("click", 960, 540)],
            )
            records = [
                json.loads(line)
                for line in (result.run_dir / "actions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(records[1]["arguments"]["point_2d"], [500, 500])
            self.assertEqual(records[1]["pixel"], [960, 540])

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

    def test_snap_to_tile_snaps_click(self) -> None:
        """With snap_to_tile, a click near a tile centre is snapped to it."""
        grid = default_grid(1280, 720)
        # Target tile (4, 4) centre.
        tx, ty = grid.tile_center(4, 4)
        # Click 15px away from the centre (within snap threshold).
        click_x, click_y = tx + 15, ty - 10

        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(
                tool_calls=(call("click", {"point_2d": [click_x, click_y]}),)
            ),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(
                turns,
                Path(tmp),
                snap_to_tile=True,
                png=deployment_png(),
                window=WINDOW_1280,
            )
            # The click should have been snapped to the tile centre.
            # Screen coords = window offset + screenshot pixel.
            expected_sx = WINDOW_1280.x + tx
            expected_sy = WINDOW_1280.y + ty
            self.assertEqual(
                [(event.kind, event.x, event.y) for event in pointer.events],
                [("click", expected_sx, expected_sy)],
            )
            # The action record should show the snapped pixel.
            records = [
                json.loads(line)
                for line in (result.run_dir / "actions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(records[1]["pixel"], [tx, ty])
            self.assertIn("Snapped to tile (4, 4)", records[1]["message"])

    def test_snap_to_tile_disabled_by_default(self) -> None:
        """Without snap_to_tile, clicks are not modified."""
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(tool_calls=(call("click", {"point_2d": [500, 400]}),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(
                turns,
                Path(tmp),
                png=deployment_png(),
                window=WINDOW_1280,
            )
            # Click should be at the exact requested pixel (no snapping).
            self.assertEqual(
                [(event.kind, event.x, event.y) for event in pointer.events],
                [("click", WINDOW_1280.x + 500, WINDOW_1280.y + 400)],
            )
            records = [
                json.loads(line)
                for line in (result.run_dir / "actions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(records[1]["pixel"], [500, 400])
            self.assertNotIn("Snapped", records[1]["message"])

    def test_snap_to_tile_far_click_not_snapped(self) -> None:
        """Clicks far from any tile (e.g. UI buttons) are not snapped."""
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            # Click in the bottom-left corner (UI area, far from tiles).
            ModelTurn(tool_calls=(call("click", {"point_2d": [50, 700]}),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(
                turns,
                Path(tmp),
                snap_to_tile=True,
                png=deployment_png(),
                window=WINDOW_1280,
            )
            # Click should NOT be snapped (too far from any tile).
            self.assertEqual(
                [(event.kind, event.x, event.y) for event in pointer.events],
                [("click", WINDOW_1280.x + 50, WINDOW_1280.y + 700)],
            )
            records = [
                json.loads(line)
                for line in (result.run_dir / "actions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(records[1]["pixel"], [50, 700])
            self.assertNotIn("Snapped", records[1]["message"])

    def test_grid_overlay_annotates_screenshots(self) -> None:
        """With grid_overlay, screenshots returned to the model have the
        tile grid drawn on them (different bytes than the raw capture)."""
        raw_png = deployment_png()
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(
                turns,
                Path(tmp),
                grid_overlay=True,
                png=raw_png,
                window=WINDOW_1280,
            )
            # The saved frame should differ from the raw capture (overlay
            # adds grid lines and labels).
            saved = (result.run_dir / "frames" / "0001.png").read_bytes()
            self.assertNotEqual(saved, raw_png)
            # Grid should be calibrated.
            self.assertIsNotNone(loop.grid)
            # Drop zone tiles should be cached for highlighting.
            self.assertIsNotNone(loop._drop_zone_tiles)
            self.assertGreater(len(loop._drop_zone_tiles), 0)

    def test_grid_overlay_skips_uncalibrated_screens(self) -> None:
        """Menus without detectable board tiles should not get an overlay."""
        raw_png = tiny_png()
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(
                turns,
                Path(tmp),
                grid_overlay=True,
                png=raw_png,
                window=WINDOW_1280,
            )
            saved = (result.run_dir / "frames" / "0001.png").read_bytes()
            self.assertEqual(saved, raw_png)
            self.assertFalse(loop._grid_calibrated)

    def test_grid_overlay_disabled_by_default(self) -> None:
        """Without grid_overlay, screenshots are not annotated."""
        raw_png = deployment_png()
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(
                turns,
                Path(tmp),
                png=raw_png,
                window=WINDOW_1280,
            )
            saved = (result.run_dir / "frames" / "0001.png").read_bytes()
            self.assertEqual(saved, raw_png)

    def test_grid_overlay_and_snap_combined(self) -> None:
        """Grid overlay + snap-to-tile work together."""
        grid = default_grid(1280, 720)
        tx, ty = grid.tile_center(4, 4)
        turns = [
            ModelTurn(tool_calls=(call("view_board"),)),
            ModelTurn(
                tool_calls=(call("click", {"point_2d": [tx + 10, ty - 8]}),)
            ),
            ModelTurn(content="done"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            loop, result, pointer = run_loop(
                turns,
                Path(tmp),
                snap_to_tile=True,
                grid_overlay=True,
                png=deployment_png(),
                window=WINDOW_1280,
            )
            # Click should be snapped to the tile centre.
            self.assertEqual(
                [(event.kind, event.x, event.y) for event in pointer.events],
                [("click", WINDOW_1280.x + tx, WINDOW_1280.y + ty)],
            )
            # Screenshots should have the overlay.
            saved = (result.run_dir / "frames" / "0001.png").read_bytes()
            self.assertNotEqual(saved, deployment_png())

    def test_system_prompt_includes_grid_overlay_note(self) -> None:
        """The system prompt mentions the grid overlay when enabled."""
        from itb_harness.loop import build_system_prompt

        prompt = build_system_prompt(
            normalized=False, grid_overlay=True
        )
        self.assertIn("tile grid overlay", prompt)
        self.assertIn("pixel coordinates", prompt)

        prompt_off = build_system_prompt(normalized=False, grid_overlay=False)
        self.assertNotIn("tile grid overlay", prompt_off)


if __name__ == "__main__":
    unittest.main()
