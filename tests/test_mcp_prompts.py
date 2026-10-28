from __future__ import annotations

import unittest

from itb_mcp.prompts import RULES, STRATEGY, WORKFLOW, level_guide


class PromptTests(unittest.TestCase):
    def test_prompt_texts_cover_core_rules(self) -> None:
        self.assertIn("skeleton", RULES.lower())
        self.assertIn("BLOCKED", RULES)
        self.assertIn("Count your steps", STRATEGY)
        self.assertIn("Plan before moving", WORKFLOW)
        # The old harness-only constraint must not leak into MCP prompts.
        self.assertNotIn("OUTPUT ONLY TOOL CALLS", WORKFLOW.upper())

    def test_level_guide_extracts_matching_section(self) -> None:
        notes = """\
# Notes

## Level 8 attempt

Kick the left skeleton first. The trap alternates every move.

## Level 9 attempt

Go right immediately.
"""
        text = level_guide(8, notes)
        self.assertIn("Level 8 attempt", text)
        self.assertIn("Kick the left skeleton", text)
        self.assertNotIn("Level 9 attempt", text)
        self.assertNotIn("Go right immediately", text)

    def test_level_guide_without_notes(self) -> None:
        text = level_guide(12, "")
        self.assertIn("No level-specific notes", text)

    def test_level_guide_with_unrelated_notes(self) -> None:
        text = level_guide(3, "## General\nPillars are decoration.")
        self.assertIn("No level-specific notes", text)


if __name__ == "__main__":
    unittest.main()
