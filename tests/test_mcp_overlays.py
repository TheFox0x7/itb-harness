from __future__ import annotations

import unittest
from io import BytesIO

from mcp_support import make_png
from PIL import Image

from itb_harness.annotate import BoxAnnotation
from itb_mcp.overlays import apply_overlays


def image_size(png: bytes) -> tuple[int, int]:
    return Image.open(BytesIO(png)).size


class OverlayTests(unittest.TestCase):
    def test_no_overlays_returns_original_bytes(self) -> None:
        png = make_png((120, 0, 0))
        result = apply_overlays(png, 64, 48)
        self.assertEqual(result, png)

    def test_boxes_change_pixels_and_keep_size(self) -> None:
        png = make_png((120, 0, 0))
        result = apply_overlays(
            png,
            64,
            48,
            boxes=[BoxAnnotation(4, 4, 30, 20, "demon")],
        )
        self.assertNotEqual(result, png)
        self.assertEqual(image_size(result), (64, 48))

    def test_grid_and_walkability_render(self) -> None:
        png = make_png((120, 0, 0))
        grid = apply_overlays(png, 64, 48, show_grid=True)
        walkability = apply_overlays(png, 64, 48, show_walkability=True)
        combined = apply_overlays(png, 64, 48, show_grid=True, show_walkability=True)
        for result in (grid, walkability, combined):
            self.assertNotEqual(result, png)
            self.assertEqual(image_size(result), (64, 48))


if __name__ == "__main__":
    unittest.main()
