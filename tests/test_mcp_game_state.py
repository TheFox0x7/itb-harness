from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mcp_support import (
    WINDOW,
    FakeBackend,
    make_config,
    make_png,
    make_state,
)

from itb_harness.annotate import BoxAnnotation
from itb_harness.capture import WindowInfo
from itb_harness.input import StubInput
from itb_mcp.game_state import GameState, WindowSelectionError


class GameStateTests(unittest.TestCase):
    def test_first_move_has_no_blocked_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state, _, pointer = make_state(Path(tmp))
            _, blocked = state.move("up")
            self.assertIsNone(blocked)
            self.assertEqual(len(pointer.events), 1)
            self.assertEqual(state.move_count, 1)

    def test_blocked_move_detected_from_identical_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            same = make_png((90, 0, 0))
            state, _, _ = make_state(Path(tmp), frames=[same, same])
            state.capture()
            _, blocked = state.move("left")
            self.assertTrue(blocked)
            self.assertEqual(state.format_history(), "Recent moves: left✗.")

    def test_successful_move_detected_from_changed_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state, _, _ = make_state(
                Path(tmp),
                frames=[make_png((0, 0, 0)), make_png((255, 0, 0))],
            )
            state.capture()
            _, blocked = state.move("right")
            self.assertFalse(blocked)
            self.assertEqual(state.steps_consumed, 1)
            self.assertEqual(state.format_history(), "Recent moves: right✓.")

    def test_history_is_capped_at_ten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state, _, _ = make_state(Path(tmp))
            for index in range(12):
                direction = ("up", "down", "left", "right")[index % 4]
                state.record_move(direction, index % 2 == 0)
            self.assertEqual(len(state.move_history), 10)
            self.assertEqual(state.move_history[0].direction, "left")
            self.assertEqual(state.move_history[-1].direction, "right")
            # format_history only shows the five most recent moves.
            self.assertEqual(
                state.format_history(),
                "Recent moves: right✓ up✗ down✓ left✗ right✓.",
            )

    def test_reset_clears_boxes_and_keeps_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state, _, pointer = make_state(Path(tmp))
            state.set_boxes([BoxAnnotation(1, 1, 10, 10, "demon")])
            state.record_move("up", False)
            state.reset_level()
            self.assertEqual(state.box_overlay, [])
            self.assertEqual([event.key for event in pointer.events], ["r"])
            self.assertEqual(len(state.move_history), 1)

    def test_snapshot_reports_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state, _, _ = make_state(Path(tmp))
            state.show_grid = True
            state.show_walkability = True
            state.set_boxes([BoxAnnotation(1, 2, 3, 4, "skeleton")])
            state.record_move("up", True)
            state.capture()

            snapshot = state.snapshot()
            self.assertTrue(snapshot["overlays"]["grid"])
            self.assertTrue(snapshot["overlays"]["walkability"])
            self.assertEqual(snapshot["box_overlay_count"], 1)
            self.assertEqual(snapshot["boxes"][0]["box_2d"], [1, 2, 3, 4])
            self.assertEqual(
                snapshot["move_history"], [{"direction": "up", "blocked": True}]
            )
            self.assertEqual(snapshot["current_frame"]["width"], 64)
            self.assertFalse(snapshot["recording"])

    def test_auto_selects_single_helltaker_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(Path(tmp), window=None)
            state = GameState(
                config,
                backend=FakeBackend(),
                input_backend=StubInput(),
                target=None,
                sleep=lambda _seconds: None,
            )
            state.ensure_ready()
            self.assertEqual(state.target, WINDOW)
            state.close()

    def test_auto_selection_rejects_missing_window(self) -> None:
        other = WindowInfo(
            id="{other}",
            title="A text editor",
            app_id="editor",
            pid=1,
            x=0,
            y=0,
            width=64,
            height=48,
            monitor="DP-4",
            active=True,
            minimized=False,
            fullscreen=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(Path(tmp), window=None)
            state = GameState(
                config,
                backend=FakeBackend(window=other),
                input_backend=StubInput(),
                target=None,
                sleep=lambda _seconds: None,
            )
            with self.assertRaises(WindowSelectionError):
                state.ensure_ready()

    def test_capture_does_not_require_input_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(Path(tmp))
            state = GameState(
                config,
                backend=FakeBackend(),
                target=WINDOW,
                sleep=lambda _seconds: None,
            )
            frame = state.capture()
            self.assertEqual(frame.width, 64)
            self.assertIsNone(state.input)
            state.close()

    def test_configured_window_selector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(Path(tmp), window="Helltaker")
            state = GameState(
                config,
                backend=FakeBackend(),
                input_backend=StubInput(),
                target=None,
                sleep=lambda _seconds: None,
            )
            state.ensure_ready()
            self.assertEqual(state.target, WINDOW)

    def test_start_and_end_run_records_frames_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state, _, _ = make_state(root)
            state.record_move("up", False)
            state.start_run("first attempt")
            self.assertEqual(state.move_count, 0)
            self.assertEqual(state.move_history, [])

            state.capture()
            state.log_action(
                tool="view_board",
                message="ok",
                source_frame_path=None,
            )
            summary = state.end_run()

            self.assertEqual(summary["frames"], 1)
            self.assertEqual(summary["actions"], 1)
            self.assertFalse(state.snapshot()["recording"])

            run_dir = Path(summary["run_dir"])
            self.assertTrue((run_dir / "frames" / "0001.png").is_file())
            metadata = json.loads((run_dir / "run.json").read_text())
            self.assertEqual(metadata["kind"], "mcp")
            self.assertEqual(metadata["initial_notes"], "first attempt")
            self.assertEqual(metadata["frames"], 1)

            actions = [
                json.loads(line)
                for line in (run_dir / "actions.jsonl").read_text().splitlines()
            ]
            self.assertEqual(actions[0]["tool"], "view_board")
            self.assertEqual(actions[0]["frame"], "frames/0001.png")

    def test_start_run_twice_requires_end_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state, _, _ = make_state(Path(tmp))
            state.start_run()
            with self.assertRaises(ValueError):
                state.start_run()
            state.end_run()
            with self.assertRaises(ValueError):
                state.end_run()


if __name__ == "__main__":
    unittest.main()
