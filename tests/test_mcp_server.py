from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from mcp.server.mcpserver.exceptions import ToolError
from mcp_support import make_config, make_png, make_state

from itb_mcp.server import build_server


def resource_text(result: object) -> str:
    contents = list(result)  # type: ignore[arg-type]
    content = contents[0].content
    assert isinstance(content, str)
    return content


class ServerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def make_server(self, **config_changes: object):
        same = make_png((90, 0, 0))
        frames = [
            same,
            same,
            make_png((255, 0, 0)),
            make_png((0, 255, 0)),
            make_png((0, 0, 255)),
        ]
        config = make_config(self.root, **config_changes)
        state, backend, pointer = make_state(self.root, config=config, frames=frames)
        self.addCleanup(state.close)
        return build_server(config, state), state, backend, pointer

    async def test_tool_catalog(self) -> None:
        server, _, _, _ = self.make_server()
        names = {tool.name for tool in await server.list_tools()}
        self.assertEqual(
            names,
            {
                "view_board",
                "move",
                "reset_level",
                "toggle_grid",
                "toggle_walkability",
                "set_boxes",
                "add_boxes",
                "list_windows",
                "select_window",
                "start_run",
                "end_run",
                "read_notes",
                "append_notes",
                "write_notes",
            },
        )

    async def test_view_board_returns_text_and_image(self) -> None:
        server, state, _, _ = self.make_server()
        result = await server.call_tool("view_board", {})
        self.assertFalse(result.is_error)
        self.assertEqual([item.type for item in result.content], ["text", "image"])
        self.assertIn("64x48", result.content[0].text)
        self.assertEqual(result.content[1].mime_type, "image/png")
        self.assertIsNotNone(state.current_frame)

    async def test_move_reports_blocked_and_updates_history(self) -> None:
        server, state, _, pointer = self.make_server()
        await server.call_tool("view_board", {})
        result = await server.call_tool("move", {"direction": "up"})
        self.assertEqual([item.type for item in result.content], ["text", "image"])
        self.assertTrue(result.content[0].text.startswith("Move up was BLOCKED"))
        self.assertEqual(state.move_count, 1)
        self.assertEqual(state.format_history(), "Recent moves: up✗.")
        self.assertEqual([event.key for event in pointer.events], ["up"])

    async def test_move_success_and_history(self) -> None:
        same = make_png((90, 0, 0))
        frames = [
            same,
            make_png((255, 0, 0)),
            make_png((0, 255, 0)),
        ]
        config = make_config(self.root)
        state, _, _ = make_state(self.root, config=config, frames=frames)
        self.addCleanup(state.close)
        server = build_server(config, state)

        await server.call_tool("view_board", {})
        result = await server.call_tool("move", {"direction": "right"})
        self.assertIn("Step consumed", result.content[0].text)
        self.assertEqual(state.format_history(), "Recent moves: right✓.")

    async def test_overlay_toggles(self) -> None:
        server, state, _, _ = self.make_server()
        await server.call_tool("toggle_grid", {"enabled": True})
        self.assertTrue(state.show_grid)
        await server.call_tool("toggle_walkability", {"enabled": True})
        self.assertTrue(state.show_walkability)
        await server.call_tool("toggle_grid", {"enabled": False})
        self.assertFalse(state.show_grid)

    async def test_set_boxes_and_reset(self) -> None:
        server, state, _, _ = self.make_server()
        await server.call_tool("view_board", {})
        result = await server.call_tool(
            "set_boxes",
            {"boxes": [{"box_2d": [1, 2, 30, 20], "label": "demon"}]},
        )
        self.assertIn("Overlay set with 1 boxes", result.content[0].text)
        self.assertEqual(len(state.box_overlay), 1)
        self.assertEqual(state.box_overlay[0].label, "demon")

        await server.call_tool(
            "add_boxes",
            {"boxes": [{"box_2d": [5, 5, 15, 15], "label": "skeleton"}]},
        )
        self.assertEqual(len(state.box_overlay), 2)

        result = await server.call_tool("reset_level", {})
        self.assertIn("Level reset", result.content[0].text)
        self.assertEqual(state.box_overlay, [])

    async def test_set_boxes_rejects_out_of_bounds(self) -> None:
        server, _, _, _ = self.make_server()
        with self.assertRaises(ToolError) as caught:
            await server.call_tool(
                "set_boxes",
                {"boxes": [{"box_2d": [0, 0, 500, 500], "label": "demon"}]},
            )
        self.assertIn("outside the screenshot", str(caught.exception))

    async def test_strict_labels_reject_unknown_category(self) -> None:
        server, _, _, _ = self.make_server(strict_box_labels=True)
        with self.assertRaises(ToolError):
            await server.call_tool(
                "set_boxes",
                {"boxes": [{"box_2d": [1, 1, 10, 10], "label": "cat"}]},
            )

    async def test_recording_and_run_resources(self) -> None:
        server, state, _, _ = self.make_server()
        started = await server.call_tool("start_run", {"notes": "test run"})
        self.assertIn("Run started:", started.content[0].text)
        run_id = state.run_id
        self.assertIsNotNone(run_id)

        await server.call_tool("view_board", {})
        await server.call_tool("move", {"direction": "up"})
        ended = await server.call_tool("end_run", {})
        self.assertIn("Run ended:", ended.content[0].text)

        runs = json.loads(resource_text(await server.read_resource("helltaker://runs")))
        self.assertEqual(runs[0]["id"], run_id)
        self.assertEqual(runs[0]["kind"], "mcp")

        run = json.loads(
            resource_text(await server.read_resource(f"helltaker://runs/{run_id}"))
        )
        self.assertEqual(run["id"], run_id)
        self.assertEqual(len(run["actions"]), 2)
        self.assertEqual(len(run["frames"]), 2)

        actions = json.loads(
            resource_text(
                await server.read_resource(f"helltaker://runs/{run_id}/actions")
            )
        )
        self.assertEqual(actions[1]["tool"], "move")

        conversation = resource_text(
            await server.read_resource(f"helltaker://runs/{run_id}/conversation")
        )
        self.assertIn("move(", conversation)

        frame_contents = list(
            await server.read_resource(f"helltaker://runs/{run_id}/frames/0001.png")
        )
        self.assertEqual(frame_contents[0].mime_type, "image/png")
        self.assertIsInstance(frame_contents[0].content, bytes)

    async def test_state_resource(self) -> None:
        server, state, _, _ = self.make_server()
        await server.call_tool("view_board", {})
        state.record_move("up", True)
        payload = json.loads(
            resource_text(await server.read_resource("helltaker://state"))
        )
        self.assertEqual(payload["move_count"], 0)
        self.assertEqual(
            payload["move_history"], [{"direction": "up", "blocked": True}]
        )
        self.assertEqual(payload["overlays"]["grid"], False)
        self.assertIsNotNone(payload["current_frame"])

    async def test_notes_tools_and_resource(self) -> None:
        server, _state, _, _ = self.make_server()
        self.assertEqual(
            (await server.call_tool("read_notes", {})).content[0].text,
            "Notes are empty.",
        )
        await server.call_tool("append_notes", {"text": "Level 1 is easy."})
        note_text = resource_text(await server.read_resource("helltaker://notes"))
        self.assertIn("Level 1 is easy.", note_text)
        self.assertIn("Helltaker harness notes", note_text)

        await server.call_tool("write_notes", {"text": "# Fresh\nOnly this."})
        self.assertEqual(
            resource_text(await server.read_resource("helltaker://notes")),
            "# Fresh\nOnly this.",
        )

    async def test_prompts(self) -> None:
        server, _, _, _ = self.make_server()
        names = {prompt.name for prompt in await server.list_prompts()}
        self.assertEqual(
            names,
            {
                "helltaker-rules",
                "helltaker-strategy",
                "helltaker-workflow",
                "helltaker-level-guide",
            },
        )
        prompt = await server.get_prompt("helltaker-level-guide", {"level": 4})
        self.assertIn("level 4", prompt.messages[0].content.text.lower())

    async def test_resource_templates_are_registered(self) -> None:
        server, _, _, _ = self.make_server()
        templates = {
            template.uri_template for template in await server.list_resource_templates()
        }
        self.assertIn("helltaker://runs/{run_id}", templates)
        self.assertIn("helltaker://runs/{run_id}/frames/{name}", templates)

    async def test_window_tools(self) -> None:
        server, state, _, _ = self.make_server()
        listing = await server.call_tool("list_windows", {})
        self.assertIn("Helltaker", listing.content[0].text)

        selected = await server.call_tool("select_window", {"selector": "Helltaker"})
        self.assertIn("Target set to", selected.content[0].text)
        self.assertEqual(state.target.title, "Helltaker")

    async def test_parallel_moves_are_serialized(self) -> None:
        same = make_png((90, 0, 0))
        frames = [
            make_png((0, 0, 0)),
            same,
            same,
            make_png((255, 0, 0)),
            make_png((0, 255, 0)),
        ]
        config = make_config(self.root)
        state, _, pointer = make_state(self.root, config=config, frames=frames)
        self.addCleanup(state.close)
        server = build_server(config, state)

        await server.call_tool("view_board", {})
        results = await asyncio.gather(
            server.call_tool("move", {"direction": "up"}),
            server.call_tool("move", {"direction": "left"}),
        )
        self.assertEqual(len(state.move_history), 2)
        self.assertEqual({event.key for event in pointer.events}, {"up", "left"})
        for result in results:
            self.assertFalse(result.is_error)


if __name__ == "__main__":
    unittest.main()
