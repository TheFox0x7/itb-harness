from __future__ import annotations

import unittest
from io import BytesIO

from PIL import Image

from itb_harness.board_overlay import (
    HexGrid,
    annotate_combined,
    annotate_grid_overlay,
    annotate_walkability,
    default_grid,
)


def make_png(width: int = 200, height: int = 200, color: tuple = (0, 0, 0)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def make_red_floor_png(width: int = 200, height: int = 200) -> bytes:
    """Create a PNG with red floor in the center (walkable area)."""
    img = Image.new("RGB", (width, height), (0, 0, 0))
    # Draw a red rectangle in the center
    for y in range(50, 150):
        for x in range(50, 150):
            img.putpixel((x, y), (140, 50, 40))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class HexGridTests(unittest.TestCase):
    def test_tile_center(self) -> None:
        grid = HexGrid(origin_x=100, origin_y=50, tile_w=80, tile_h=60)
        x, y = grid.tile_center(0, 0)
        self.assertEqual((x, y), (100, 50))
        x, y = grid.tile_center(1, 0)
        self.assertEqual((x, y), (100, 110))
        x, y = grid.tile_center(0, 1)
        self.assertEqual((x, y), (180, 80))  # odd col offset by half tile_h

    def test_snap(self) -> None:
        grid = HexGrid(origin_x=100, origin_y=50, tile_w=80, tile_h=60)
        row, col = grid.snap(100, 50)
        self.assertEqual((row, col), (0, 0))
        row, col = grid.snap(180, 80)
        self.assertEqual((row, col), (0, 1))

    def test_default_grid(self) -> None:
        grid = default_grid(1280, 768)
        self.assertEqual(grid.n_cols, 5)
        self.assertEqual(grid.n_rows, 12)
        # Origin should be within the screen
        x, y = grid.tile_center(0, 0)
        self.assertTrue(0 <= x <= 1280)
        self.assertTrue(0 <= y <= 768)


class GridOverlayTests(unittest.TestCase):
    def test_returns_valid_png(self) -> None:
        png = make_png()
        result = annotate_grid_overlay(png)
        img = Image.open(BytesIO(result))
        self.assertEqual(img.size, (200, 200))

    def test_overlay_changes_image(self) -> None:
        png = make_png(color=(50, 50, 50))
        result = annotate_grid_overlay(png)
        self.assertNotEqual(result, png)

    def test_no_labels(self) -> None:
        png = make_png()
        result = annotate_grid_overlay(png, show_labels=False)
        img = Image.open(BytesIO(result))
        self.assertEqual(img.size, (200, 200))


class WalkabilityTests(unittest.TestCase):
    def test_returns_valid_png(self) -> None:
        png = make_red_floor_png()
        result = annotate_walkability(png)
        img = Image.open(BytesIO(result))
        self.assertEqual(img.size, (200, 200))

    def test_overlay_changes_image(self) -> None:
        png = make_red_floor_png()
        result = annotate_walkability(png)
        self.assertNotEqual(result, png)

    def test_all_black_no_walkable(self) -> None:
        png = make_png(color=(0, 0, 0))
        result = annotate_walkability(png)
        # Should still produce a valid PNG (all tiles marked as blocked)
        img = Image.open(BytesIO(result))
        self.assertEqual(img.size, (200, 200))


class CombinedOverlayTests(unittest.TestCase):
    def test_both_overlays(self) -> None:
        png = make_red_floor_png()
        result = annotate_combined(png, show_grid=True, show_walkability=True)
        img = Image.open(BytesIO(result))
        self.assertEqual(img.size, (200, 200))

    def test_grid_only(self) -> None:
        png = make_png()
        result = annotate_combined(png, show_grid=True, show_walkability=False)
        img = Image.open(BytesIO(result))
        self.assertEqual(img.size, (200, 200))

    def test_walkability_only(self) -> None:
        png = make_red_floor_png()
        result = annotate_combined(png, show_grid=False, show_walkability=True)
        img = Image.open(BytesIO(result))
        self.assertEqual(img.size, (200, 200))


if __name__ == "__main__":
    unittest.main()
