from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from itb_harness.board_tools import parse_note_args, tool_schemas
from itb_harness.notes import NotesStore


class NotesStoreTests(unittest.TestCase):
    def test_append_and_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.md"
            store = NotesStore(path)
            self.assertEqual(store.read(), "")
            store.append("Enemy: Firefly shoots in a line.", run_id="run-1")
            store.append("Mech: Boulder has armor.", run_id="run-2")
            text = path.read_text()
            self.assertIn("Firefly", text)
            self.assertIn("Boulder", text)
            self.assertIn("[run-1]", text)
            self.assertIn("[run-2]", text)

    def test_render_wraps_notes_as_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = NotesStore(Path(tmp) / "notes.md")
            self.assertEqual(store.render_for_prompt(), "")
            store.append("Remember the acid tiles.", run_id="run")
            rendered = store.render_for_prompt()
            self.assertIn("Persistent notes from previous runs", rendered)
            self.assertIn("<notes>", rendered)
            self.assertIn("Remember the acid tiles.", rendered)

    def test_old_notes_are_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = NotesStore(Path(tmp) / "notes.md", max_chars=1000)
            store.append("old " + "x" * 1200, run_id="old")
            store.append("newest note", run_id="new")
            text = store.read()
            self.assertLessEqual(len(text), 1000 + len("[older notes truncated]\n"))
            self.assertTrue(text.startswith("[older notes truncated]"))
            self.assertIn("newest note", text)

    def test_rejects_empty_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = NotesStore(Path(tmp) / "notes.md")
            with self.assertRaises(ValueError):
                store.append("   ")


class NoteToolTests(unittest.TestCase):
    def test_note_schema_is_optional(self) -> None:
        base = [tool["function"]["name"] for tool in tool_schemas()]
        with_note = [
            tool["function"]["name"] for tool in tool_schemas(include_note=True)
        ]
        self.assertNotIn("note", base)
        self.assertIn("note", with_note)

    def test_parse_note_args(self) -> None:
        self.assertEqual(
            parse_note_args({"text": "  useful fact  "}),
            "useful fact",
        )
        with self.assertRaises(ValueError):
            parse_note_args({"text": ""})
        with self.assertRaises(ValueError):
            parse_note_args({"text": "x" * 2001})


if __name__ == "__main__":
    unittest.main()
