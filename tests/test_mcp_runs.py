from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mcp_support import make_png

from itb_mcp.runs import (
    RunNotFound,
    actions_payload,
    conversation_text,
    list_runs,
    load_run,
    safe_frame_path,
    safe_run_dir,
)


def make_run(runs_dir: Path, run_id: str = "20260101-000000-000000-mcp") -> Path:
    run_dir = runs_dir / run_id
    (run_dir / "frames").mkdir(parents=True)
    (run_dir / "frames" / "0001.png").write_bytes(make_png())
    (run_dir / "frames" / "0002.png").write_bytes(make_png((255, 0, 0)))
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "kind": "mcp",
                "model": "external MCP client",
                "started_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )
    actions = [
        {
            "step": 1,
            "tool": "view_board",
            "arguments": {},
            "ok": True,
            "message": "looked",
            "blocked": None,
            "frame": "frames/0001.png",
        },
        {
            "step": 2,
            "tool": "move",
            "arguments": {"direction": "up"},
            "ok": True,
            "message": "Moved up",
            "blocked": False,
            "frame": "frames/0002.png",
        },
    ]
    (run_dir / "actions.jsonl").write_text(
        "\n".join(json.dumps(action) for action in actions) + "\n"
    )
    return run_dir


class RunLoadingTests(unittest.TestCase):
    def test_list_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            make_run(runs_dir)
            runs = list_runs(runs_dir)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["kind"], "mcp")
            self.assertEqual(runs[0]["frames"], 2)
            self.assertEqual(runs[0]["actions"], 2)

    def test_list_runs_missing_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(list_runs(Path(tmp) / "missing"), [])

    def test_load_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = make_run(Path(tmp) / "runs")
            loaded = load_run(run_dir)
            self.assertEqual(loaded["id"], run_dir.name)
            self.assertEqual(loaded["frames"], ["0001.png", "0002.png"])
            self.assertEqual(len(loaded["actions"]), 2)
            self.assertEqual([turn["step"] for turn in loaded["turns"]], [1, 2])
            self.assertEqual(loaded["turns"][1]["actions"][0]["frame_name"], "0002.png")

    def test_conversation_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = make_run(Path(tmp) / "runs")
            text = conversation_text(run_dir)
            self.assertIn("view_board", text)
            self.assertIn('move({"direction": "up"}', text)
            self.assertIn("Moved up", text)
            self.assertIn("[0002.png]", text)

    def test_actions_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = make_run(Path(tmp) / "runs")
            self.assertEqual(len(actions_payload(run_dir)), 2)

    def test_safe_run_dir_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            make_run(runs_dir)
            with self.assertRaises(RunNotFound):
                safe_run_dir(runs_dir, "../escape")
            with self.assertRaises(RunNotFound):
                safe_run_dir(runs_dir, "does-not-exist")

    def test_safe_frame_path_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            run_dir = make_run(runs_dir)
            path = safe_frame_path(runs_dir, run_dir.name, "0001.png")
            self.assertTrue(path.is_file())
            for name in ("../run.json", "0001.jpg", "missing.png"):
                with self.assertRaises(RunNotFound):
                    safe_frame_path(runs_dir, run_dir.name, name)


if __name__ == "__main__":
    unittest.main()
