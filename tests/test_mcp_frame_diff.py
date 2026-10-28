from __future__ import annotations

import unittest

from mcp_support import make_png

from itb_mcp.frame_diff import frame_difference, frames_differ


class FrameDiffTests(unittest.TestCase):
    def test_identical_frames_do_not_differ(self) -> None:
        png = make_png((10, 20, 30))
        self.assertFalse(frames_differ(png, png))
        self.assertEqual(frame_difference(png, png), 0.0)

    def test_different_frames_differ(self) -> None:
        dark = make_png((0, 0, 0))
        bright = make_png((255, 255, 255))
        self.assertTrue(frames_differ(dark, bright))
        self.assertGreater(frame_difference(dark, bright), 0.9)

    def test_invalid_png_counts_as_changed(self) -> None:
        self.assertTrue(frames_differ(b"not a png", b"also not a png"))

    def test_size_mismatch_counts_as_changed(self) -> None:
        small = make_png((0, 0, 0), size=(10, 10))
        large = make_png((0, 0, 0), size=(20, 20))
        self.assertTrue(frames_differ(small, large))


if __name__ == "__main__":
    unittest.main()
