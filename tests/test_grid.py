"""Tests for the isometric grid detection and click snapping."""

from __future__ import annotations

import unittest
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from itb_harness.annotate import annotate_grid
from itb_harness.grid import (
    IsoGrid,
    calibrate_grid,
    default_grid,
    detect_drop_zone_tiles,
)


def make_deployment_png(
    width: int = 1280,
    height: int = 720,
    grid: IsoGrid | None = None,
    tan_tiles: list[tuple[int, int]] | None = None,
) -> bytes:
    """Create a synthetic deployment screenshot with tan drop-zone tiles.

    Draws a dark background, a bright board diamond, and tan tiles at the
    given grid positions.
    """
    if grid is None:
        grid = default_grid(width, height)
    if tan_tiles is None:
        tan_tiles = [(3, 3), (3, 4), (4, 4), (4, 5), (5, 5)]

    img = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(img)

    # Draw the board diamond (bright area).
    cx = grid.origin_x
    cy = grid.origin_y + 7 * grid.tile_h2  # vertical centre
    half_w = 4 * 2 * grid.tile_w2
    half_h = 4 * 2 * grid.tile_h2
    draw.polygon(
        [
            (cx, cy - half_h),
            (cx + half_w, cy),
            (cx, cy + half_h),
            (cx - half_w, cy),
        ],
        fill=(60, 65, 55),
    )

    # Draw tan drop-zone tiles with a small gap between them (simulates
    # the grid lines in the real game so erosion can separate them).
    tan = (140, 140, 90)
    for row, col in tan_tiles:
        tx, ty = grid.tile_center(row, col)
        tw = int(2 * grid.tile_w2 * 0.88)
        th = int(2 * grid.tile_h2 * 0.88)
        draw.polygon(
            [
                (tx, ty - th // 2),
                (tx + tw // 2, ty),
                (tx, ty + th // 2),
                (tx - tw // 2, ty),
            ],
            fill=tan,
        )

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class IsoGridTests(unittest.TestCase):
    def test_tile_center_roundtrip(self) -> None:
        grid = default_grid(1280, 720)
        for row in range(8):
            for col in range(8):
                x, y = grid.tile_center(row, col)
                sx, sy, r2, c2 = grid.snap(x, y)
                self.assertEqual((r2, c2), (row, col))
                self.assertAlmostEqual(sx, x, delta=1)
                self.assertAlmostEqual(sy, y, delta=1)

    def test_snap_clamps_to_board(self) -> None:
        grid = default_grid(1280, 720)
        # Far outside the board should clamp to a valid edge tile.
        sx, sy, row, col = grid.snap(-100, -100)
        self.assertTrue(0 <= row <= 7)
        self.assertTrue(0 <= col <= 7)
        sx, sy, row, col = grid.snap(2000, 2000)
        self.assertTrue(0 <= row <= 7)
        self.assertTrue(0 <= col <= 7)
        # Snapping a tile centre should return that tile.
        for row in range(8):
            for col in range(8):
                tx, ty = grid.tile_center(row, col)
                _sx, _sy, r2, c2 = grid.snap(tx, ty)
                self.assertEqual((r2, c2), (row, col))

    def test_scaled_grid(self) -> None:
        g1 = default_grid(1280, 720)
        g2 = g1.scaled(640, 360)
        self.assertAlmostEqual(g2.origin_x, g1.origin_x / 2)
        self.assertAlmostEqual(g2.origin_y, g1.origin_y / 2)
        self.assertAlmostEqual(g2.tile_w2, g1.tile_w2 / 2)
        self.assertAlmostEqual(g2.tile_h2, g1.tile_h2 / 2)

    def test_all_centers_on_screen(self) -> None:
        grid = default_grid(1280, 720)
        centers = grid.all_centers()
        self.assertEqual(len(centers), 64)
        for x, y, row, col in centers:
            self.assertTrue(0 <= x < 1280, f"tile ({row},{col}) x={x} off screen")
            self.assertTrue(0 <= y < 720, f"tile ({row},{col}) y={y} off screen")


class DetectDropZoneTilesTests(unittest.TestCase):
    def test_detects_tan_tiles(self) -> None:
        png = make_deployment_png()
        tiles = detect_drop_zone_tiles(png)
        self.assertEqual(len(tiles), 5)
        # Each detected tile should be close to a known tile centre.
        grid = default_grid(1280, 720)
        expected = [(3, 3), (3, 4), (4, 4), (4, 5), (5, 5)]
        for (row, col) in expected:
            ex, ey = grid.tile_center(row, col)
            closest = min(tiles, key=lambda t: (t[0] - ex) ** 2 + (t[1] - ey) ** 2)
            dist = ((closest[0] - ex) ** 2 + (closest[1] - ey) ** 2) ** 0.5
            self.assertLess(dist, 15, f"tile ({row},{col}) detection off by {dist:.1f}px")

    def test_no_tiles_returns_empty(self) -> None:
        # Plain dark image — no tan tiles.
        img = Image.new("RGB", (1280, 720), (20, 20, 20))
        buf = BytesIO()
        img.save(buf, format="PNG")
        tiles = detect_drop_zone_tiles(buf.getvalue())
        self.assertEqual(tiles, [])


class CalibrateGridTests(unittest.TestCase):
    def test_calibrate_from_synthetic(self) -> None:
        grid_true = default_grid(1280, 720)
        png = make_deployment_png(grid=grid_true)
        grid = calibrate_grid(png, 1280, 720)
        self.assertIsNotNone(grid)
        assert grid is not None
        # Origin should be close to the true grid.
        self.assertAlmostEqual(grid.origin_x, grid_true.origin_x, delta=10)
        self.assertAlmostEqual(grid.origin_y, grid_true.origin_y, delta=15)
        self.assertAlmostEqual(grid.tile_w2, grid_true.tile_w2, delta=5)
        self.assertAlmostEqual(grid.tile_h2, grid_true.tile_h2, delta=5)

    def test_calibrate_returns_none_without_tiles(self) -> None:
        img = Image.new("RGB", (1280, 720), (20, 20, 20))
        buf = BytesIO()
        img.save(buf, format="PNG")
        grid = calibrate_grid(buf.getvalue(), 1280, 720)
        self.assertIsNone(grid)

    def test_calibrated_grid_snaps_correctly(self) -> None:
        grid_true = default_grid(1280, 720)
        png = make_deployment_png(grid=grid_true)
        grid = calibrate_grid(png, 1280, 720)
        self.assertIsNotNone(grid)
        assert grid is not None
        # Snapping a tile centre should land on itself.
        for row, col in [(3, 3), (4, 4), (5, 5)]:
            ex, ey = grid_true.tile_center(row, col)
            sx, sy, r2, c2 = grid.snap(ex, ey)
            self.assertEqual((r2, c2), (row, col))
            self.assertAlmostEqual(sx, ex, delta=5)
            self.assertAlmostEqual(sy, ey, delta=5)


class SnapModelClicksTests(unittest.TestCase):
    """Verify that the model's failed clicks from the last run would be
    snapped to valid tile centres."""

    def test_snap_failed_clicks(self) -> None:
        grid = default_grid(1280, 720)
        # These are the model's actual failed clicks from run 215958.
        failed = [
            (972, 410),  # was: outside drop zone, right side
            (703, 345),  # was: forest tile
            (767, 503),  # was: gray tile
        ]
        for px, py in failed:
            sx, sy, row, col = grid.snap(px, py)
            # Snapped position should be a valid tile centre.
            tx, ty = grid.tile_center(row, col)
            self.assertEqual((sx, sy), (tx, ty))
            # Should be within a reasonable distance.
            dist = ((sx - px) ** 2 + (sy - py) ** 2) ** 0.5
            self.assertLess(dist, 60, f"click ({px},{py}) snapped too far: {dist:.0f}px")


class AnnotateGridTests(unittest.TestCase):
    """Tests for the grid overlay annotation."""

    def test_annotate_grid_returns_valid_png(self) -> None:
        grid = default_grid(1280, 720)
        png = make_deployment_png(grid=grid)
        result = annotate_grid(png, grid)
        img = Image.open(BytesIO(result))
        self.assertEqual(img.size, (1280, 720))
        self.assertEqual(img.mode, "RGB")

    def test_annotate_grid_draws_labels(self) -> None:
        """Labels should add yellow-ish pixels near tile centres."""
        grid = default_grid(1280, 720)
        png = make_deployment_png(grid=grid)
        result = annotate_grid(png, grid, show_labels=True)
        arr = np.array(Image.open(BytesIO(result)).convert("RGB"))

        # Check that a tile centre has label text (yellow pixels nearby).
        cx, cy = grid.tile_center(4, 4)
        region = arr[cy + 4 : cy + 20, cx - 30 : cx + 30]
        has_yellow = np.any(
            (region[:, :, 0] > 200)
            & (region[:, :, 1] > 200)
            & (region[:, :, 2] < 200)
        )
        self.assertTrue(has_yellow, "no label text found near tile centre")

    def test_annotate_grid_no_labels(self) -> None:
        """Without labels, no yellow text pixels near tile centres."""
        grid = default_grid(1280, 720)
        png = make_deployment_png(grid=grid)
        result = annotate_grid(png, grid, show_labels=False)
        arr = np.array(Image.open(BytesIO(result)).convert("RGB"))

        cx, cy = grid.tile_center(4, 4)
        region = arr[cy + 4 : cy + 20, cx - 30 : cx + 30]
        has_yellow = np.any(
            (region[:, :, 0] > 200)
            & (region[:, :, 1] > 200)
            & (region[:, :, 2] < 200)
        )
        self.assertFalse(has_yellow, "unexpected label text found")

    def test_annotate_grid_highlight_tiles(self) -> None:
        """Highlighted tiles should have a gold tint at their centres."""
        grid = default_grid(1280, 720)
        png = make_deployment_png(grid=grid)
        tiles = detect_drop_zone_tiles(png)
        self.assertGreater(len(tiles), 0)

        result = annotate_grid(png, grid, highlight_tiles=tiles)
        arr = np.array(Image.open(BytesIO(result)).convert("RGB"))

        # A highlighted tile centre should be brighter / more gold than
        # the original tan fill.
        tx, ty = tiles[0]
        sx, sy, _row, _col = grid.snap(int(tx), int(ty))
        pixel = arr[sy, sx]
        # Gold tint pushes R and G up relative to B.
        self.assertGreater(int(pixel[0]), 150)
        self.assertGreater(int(pixel[1]), 150)

    def test_annotate_grid_diamonds(self) -> None:
        """Diamond outlines should brighten pixels at tile edges."""
        grid = default_grid(1280, 720)
        png = make_deployment_png(grid=grid)
        original = np.array(Image.open(BytesIO(png)).convert("RGB")).astype(int)
        result = annotate_grid(png, grid, show_labels=False, show_centers=False)
        arr = np.array(Image.open(BytesIO(result)).convert("RGB")).astype(int)

        # The top vertex of a tile diamond should be brighter than the
        # original (the white outline adds brightness).
        cx, cy = grid.tile_center(4, 4)
        th2 = int(grid.tile_h2)
        top_y = cy - th2
        region_new = arr[top_y - 2 : top_y + 3, cx - 3 : cx + 4]
        region_old = original[top_y - 2 : top_y + 3, cx - 3 : cx + 4]
        # At least one pixel should be noticeably brighter.
        diff = region_new.mean(axis=2) - region_old.mean(axis=2)
        self.assertGreater(diff.max(), 15, "no diamond outline brightness increase")


if __name__ == "__main__":
    unittest.main()
