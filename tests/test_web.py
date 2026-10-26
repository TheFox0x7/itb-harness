from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from itb_harness.web.server import list_runs, load_run


class ReplayLoaderTests(unittest.TestCase):
    def make_run(self, root: Path) -> Path:
        run = root / "20261024-120000-000000-agent"
        (run / "frames").mkdir(parents=True)
        (run / "frames" / "0001.png").write_bytes(b"png-1")
        (run / "frames" / "0002.png").write_bytes(b"png-2")
        (run / "run.json").write_text(
            json.dumps(
                {
                    "backend": "x11",
                    "input": "x11",
                    "model": "test-model",
                    "started_at": "2026-10-24T12:00:00+00:00",
                    "target": {"title": "Breach"},
                }
            )
        )
        (run / "conversation.jsonl").write_text(
            json.dumps(
                {
                    "step": 1,
                    "content": "thinking",
                    "thinking": "I see the board.",
                    "usage": {"prompt_tokens": 10},
                    "tool_calls": [],
                }
            )
            + "\n"
        )
        actions = [
            {
                "step": 1,
                "tool": "view_board",
                "arguments": {},
                "ok": True,
                "frame": "frames/0001.png",
                "source_frame": None,
            },
            {
                "step": 2,
                "tool": "click",
                "arguments": {"point_2d": [500, 500]},
                "pixel": [960, 540],
                "ok": True,
                "frame": None,
                "source_frame": "frames/0001.png",
            },
            {
                "step": 3,
                "tool": "inspect",
                "arguments": {"point_2d": [480, 809], "with_details": True},
                "ok": True,
                "frame": "frames/0002.png",
                "source_frame": "frames/0001.png",
            },
        ]
        (run / "actions.jsonl").write_text(
            "\n".join(json.dumps(action) for action in actions) + "\n"
        )
        (run / "guidance.jsonl").write_text(
            json.dumps({"step": 2, "text": "push the rock left"}) + "\n"
        )
        return run

    def test_load_run_builds_timeline_and_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_run(root)
            loaded = load_run(run)
            self.assertEqual(loaded["id"], run.name)
            self.assertEqual(loaded["frames"], ["0001.png", "0002.png"])
            self.assertEqual(len(loaded["turns"]), 3)

            click = loaded["turns"][1]["actions"][0]
            self.assertEqual(click["point_2d"], [960.0, 540.0])
            self.assertEqual(click["marker_frames"], ["0001.png"])

            inspect = loaded["turns"][2]["actions"][0]
            self.assertEqual(
                inspect["marker_frames"], ["0001.png", "0002.png"]
            )
            self.assertTrue(inspect["with_details"])
            self.assertEqual(
                loaded["turns"][1]["guidance"], ["push the rock left"]
            )

    def test_hover_action_has_markers_without_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.make_run(root)
            actions_path = run / "actions.jsonl"
            actions = [
                json.loads(line)
                for line in actions_path.read_text().splitlines()
            ]
            actions.append(
                {
                    "step": 4,
                    "tool": "hover",
                    "arguments": {"point_2d": [100, 120]},
                    "ok": True,
                    "frame": "frames/0002.png",
                    "source_frame": "frames/0002.png",
                }
            )
            actions_path.write_text(
                "\n".join(json.dumps(action) for action in actions) + "\n"
            )

            loaded = load_run(run)
            hover = loaded["turns"][3]["actions"][0]
            self.assertEqual(hover["marker_frames"], ["0002.png"])
            self.assertFalse(hover["with_details"])

    def test_list_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_run(root)
            runs = list_runs(root)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["frame_count"], 2)
            self.assertEqual(runs[0]["model"], "test-model")


if __name__ == "__main__":
    unittest.main()
