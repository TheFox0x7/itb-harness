from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mcp_support import WINDOW, FakeBackend, make_config, make_png

from itb_harness.input import StubInput
from itb_mcp.recorder import Recorder


class RecorderTests(unittest.TestCase):
    def test_create_save_and_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = make_config(root)
            backend = FakeBackend()
            recorder = Recorder.create(
                config.runs_dir,
                backend=backend,
                input_backend=StubInput(),
                target=WINDOW,
                notes_file=config.notes_file,
                initial_notes="hello",
                config=config,
            )

            path = recorder.save_frame(make_png((1, 2, 3)))
            self.assertEqual(path.name, "0001.png")
            self.assertTrue(path.is_file())

            recorder.log_action(
                tool="move",
                arguments={"direction": "up"},
                message="Moved",
                blocked=False,
                state_changed=True,
                frame_path=path,
                source_frame_path=None,
            )

            summary = recorder.close()
            self.assertEqual(summary["frames"], 1)
            self.assertEqual(summary["actions"], 1)
            self.assertEqual(summary["run_id"], recorder.run_id)

            metadata = json.loads(
                (recorder.root / "run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["kind"], "mcp")
            self.assertEqual(metadata["initial_notes"], "hello")
            self.assertEqual(metadata["target"]["title"], "Helltaker")
            self.assertIn("ended_at", metadata)

            actions = [
                json.loads(line)
                for line in (recorder.root / "actions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(actions[0]["tool"], "move")
            self.assertEqual(actions[0]["arguments"], {"direction": "up"})
            self.assertEqual(actions[0]["frame"], "frames/0001.png")

    def test_close_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(Path(tmp))
            recorder = Recorder.create(
                config.runs_dir,
                backend=FakeBackend(),
                input_backend=StubInput(),
                target=WINDOW,
                notes_file=None,
                initial_notes=None,
                config=config,
            )
            first = recorder.close()
            second = recorder.close()
            self.assertEqual(first["frames"], second["frames"])
            self.assertIsNone(
                json.loads((recorder.root / "run.json").read_text())["notes_file"]
            )

    def test_update_target_rewrites_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(Path(tmp))
            recorder = Recorder.create(
                config.runs_dir,
                backend=FakeBackend(),
                input_backend=StubInput(),
                target=WINDOW,
                notes_file=None,
                initial_notes=None,
                config=config,
            )
            from dataclasses import replace

            other = replace(WINDOW, id="{other}", title="Other")
            recorder.update_target(other)
            metadata = json.loads(
                (recorder.root / "run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["target"]["id"], "{other}")
            recorder.close()


if __name__ == "__main__":
    unittest.main()
