from __future__ import annotations

import unittest

from itb_harness.capture import MonitorInfo, WindowInfo, monitor_for_window
from itb_harness.cli import format_windows


def window(x: int, y: int, width: int, height: int) -> WindowInfo:
    return WindowInfo(
        id="{test}",
        title="Test",
        app_id="test",
        pid=None,
        x=x,
        y=y,
        width=width,
        height=height,
        monitor=None,
        active=False,
        minimized=False,
        fullscreen=False,
    )


MONITORS = [
    MonitorInfo(id="1", name="eDP-1", x=0, y=0, width=1920, height=1080),
    MonitorInfo(id="2", name="DP-4", x=1920, y=0, width=1920, height=1080),
]


class MonitorSelectionTests(unittest.TestCase):
    def test_left_monitor(self) -> None:
        self.assertEqual(
            monitor_for_window(window(0, 0, 1920, 1036), MONITORS), "eDP-1"
        )

    def test_right_monitor(self) -> None:
        self.assertEqual(
            monitor_for_window(window(2421, 58, 1280, 720), MONITORS), "DP-4"
        )

    def test_straddling_window_uses_larger_overlap(self) -> None:
        self.assertEqual(
            monitor_for_window(window(1800, 0, 400, 400), MONITORS), "DP-4"
        )

    def test_no_overlap_falls_back_to_nearest(self) -> None:
        self.assertEqual(
            monitor_for_window(window(5000, 5000, 100, 100), MONITORS), "DP-4"
        )


class WindowFormattingTests(unittest.TestCase):
    def test_selected_window_is_marked(self) -> None:
        selected = window(0, 0, 1920, 1080)
        other = WindowInfo(
            id="{other}",
            title="Other",
            app_id="other",
            pid=None,
            x=1920,
            y=0,
            width=1920,
            height=1080,
            monitor="DP-4",
            active=True,
            minimized=False,
            fullscreen=True,
        )
        rendered = format_windows([selected, other], selected.id)
        lines = rendered.splitlines()
        self.assertIn("* 1", lines[1])
        self.assertIn(" 2", lines[2])
        self.assertIn("A", lines[2])
        self.assertIn("F", lines[2])


if __name__ == "__main__":
    unittest.main()
