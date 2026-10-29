from __future__ import annotations

import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from itb_harness.grid_overlay import (
    CELL_HEIGHT,
    CELL_WIDTH,
    annotate_helltaker_grid,
    find_floor_anchor,
    red_floor_mask,
)


def png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FloorAnchorTests(unittest.TestCase):
    def test_finds_bottom_row_of_wide_red_run(self) -> None:
        mask = np.zeros((200, 200), dtype=bool)
        mask[100:150, 40:120] = True  # 80 px wide run

        self.assertEqual(find_floor_anchor(mask), 149)

    def test_ignores_runs_narrower_than_one_cell(self) -> None:
        mask = np.zeros((200, 200), dtype=bool)
        mask[100:150, 40 : 40 + CELL_WIDTH - 1] = True

        self.assertIsNone(find_floor_anchor(mask))

    def test_red_floor_filter_matches_helltaker_colors(self) -> None:
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[0, 0] = (140, 28, 49)  # #8c1c31
        image[0, 1] = (112, 12, 41)  # #700c29
        image[1, 0] = (200, 200, 200)
        image[1, 1] = (80, 80, 80)

        mask = red_floor_mask(image)
        self.assertTrue(mask[0, 0])
        self.assertTrue(mask[0, 1])
        self.assertFalse(mask[1, 0])
        self.assertFalse(mask[1, 1])


class GridOverlayTests(unittest.TestCase):
    def test_overlay_is_anchored_to_red_floor(self) -> None:
        image = Image.new("RGB", (800, 300), (0, 0, 0))
        pixels = np.asarray(image).copy()
        pixels[100:150, 300:400] = (112, 12, 41)
        image = Image.fromarray(pixels)

        result = Image.open(BytesIO(annotate_helltaker_grid(png_bytes(image))))
        self.assertEqual(result.size, (800, 300))

        # Anchor row is 149, so the bottom grid line is at y=150. A line one
        # cell above it is at y=150-72=78.
        bottom_line = result.getpixel((100, 150))
        upper_line = result.getpixel((100, 78))
        blank = result.getpixel((100, 20))
        self.assertGreater(bottom_line[2], blank[2])
        self.assertGreater(upper_line[2], blank[2])
        self.assertGreater(bottom_line[0], blank[0])

        # Nine 68 px columns centered on an 800 px image start at x=94.
        left_line = result.getpixel((94, 50))
        self.assertGreater(left_line[2], blank[2])

    def test_cell_dimensions(self) -> None:
        self.assertEqual(CELL_WIDTH, 68)
        self.assertEqual(CELL_HEIGHT, 72)


if __name__ == "__main__":
    unittest.main()
