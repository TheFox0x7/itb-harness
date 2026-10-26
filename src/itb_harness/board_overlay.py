"""Board analysis overlays for Helltaker screenshots.

Provides grid coordinate labels and walkability highlighting so the model
can reference positions and see which tiles are open.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(slots=True)
class HexGrid:
    """Hexagonal grid parameters for a Helltaker board.

    Tiles are arranged in columns. Adjacent columns are offset vertically
    by half a tile height (brick-like pattern). Row 0 is the top of the
    screen, column 0 is the leftmost walkable column.
    """

    origin_x: float  # screen x of column 0 tile center
    origin_y: float  # screen y of row 0 tile center
    tile_w: float  # horizontal spacing between column centers
    tile_h: float  # vertical spacing between row centers (within a column)
    n_cols: int = 5
    n_rows: int = 12

    def tile_center(self, row: int, col: int) -> tuple[int, int]:
        """Screen center of tile (row, col)."""
        # Brick offset: odd columns are shifted down by half a tile
        offset = (col % 2) * (self.tile_h / 2.0)
        x = self.origin_x + col * self.tile_w
        y = self.origin_y + row * self.tile_h + offset
        return int(round(x)), int(round(y))

    def snap(self, px: int, py: int) -> tuple[int, int]:
        """Snap a pixel to the nearest tile (row, col)."""
        # Inverse of tile_center — approximate
        col = round((px - self.origin_x) / self.tile_w)
        offset = (col % 2) * (self.tile_h / 2.0)
        row = round((py - self.origin_y - offset) / self.tile_h)
        col = max(0, min(self.n_cols - 1, col))
        row = max(0, min(self.n_rows - 1, row))
        return row, col


def default_grid(width: int, height: int) -> HexGrid:
    """Return a default grid scaled for the screenshot size.

    Calibrated from 1280x768 Helltaker screenshots. The play area spans
    roughly x=350..1050 with ~5 columns of hexagonal tiles.
    """
    sx = width / 1280.0
    sy = height / 768.0
    return HexGrid(
        origin_x=450 * sx,
        origin_y=45 * sy,
        tile_w=140 * sx,
        tile_h=90 * sy,
        n_cols=5,
        n_rows=12,
    )


def annotate_grid_overlay(
    png: bytes,
    grid: HexGrid | None = None,
    show_labels: bool = True,
    show_dots: bool = True,
) -> bytes:
    """Draw a coordinate grid overlay on a Helltaker screenshot.

    Draws a small dot at each tile center and labels it with "r{row}c{col}"
    so the model can reference positions precisely.
    """
    image = Image.open(BytesIO(png)).convert("RGB")
    width, height = image.size
    if grid is None:
        grid = default_grid(width, height)

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    scale = min(width / 1280.0, height / 768.0)

    dot_r = max(2, int(round(3 * scale)))
    font = ImageFont.load_default()

    for row in range(grid.n_rows):
        for col in range(grid.n_cols):
            cx, cy = grid.tile_center(row, col)
            if not (0 <= cx < width and 0 <= cy < height):
                continue

            # Dot at tile center
            if show_dots:
                draw.ellipse(
                    [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
                    fill=(255, 255, 100, 180),
                )

            # Label below the dot
            if show_labels:
                text = f"r{row}c{col}"
                bbox = draw.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                lx = max(2, min(width - tw - 4, cx - tw // 2))
                ly = max(2, min(height - th - 4, cy + dot_r + 2))
                draw.rectangle(
                    [lx - 2, ly - 1, lx + tw + 2, ly + th + 1],
                    fill=(0, 0, 0, 160),
                )
                draw.text(
                    (lx, ly), text, fill=(255, 255, 120, 255), font=font
                )

    image = Image.alpha_composite(image.convert("RGBA"), overlay)
    out = BytesIO()
    image.convert("RGB").save(out, format="PNG", compress_level=1)
    return out.getvalue()


def annotate_walkability(
    png: bytes,
    grid: HexGrid | None = None,
    walkable_color: tuple[int, int, int, int] = (80, 255, 80, 40),
    blocked_color: tuple[int, int, int, int] = (255, 60, 60, 30),
) -> bytes:
    """Highlight walkable vs blocked tiles based on floor color analysis.

    Red hexagonal floor tiles are walkable. Black/dark areas are walls.
    The overlay tints each grid tile green (walkable) or red (blocked)
    based on the average color at the tile center.
    """
    image = Image.open(BytesIO(png)).convert("RGB")
    width, height = image.size
    arr = np.array(image).astype(int)
    if grid is None:
        grid = default_grid(width, height)

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    scale = min(width / 1280.0, height / 768.0)

    # Sample radius for color analysis at each tile center
    sample_r = max(5, int(round(15 * scale)))

    for row in range(grid.n_rows):
        for col in range(grid.n_cols):
            cx, cy = grid.tile_center(row, col)
            if not (0 <= cx < width and 0 <= cy < height):
                continue

            # Sample a small region around the tile center
            x0 = max(0, cx - sample_r)
            x1 = min(width, cx + sample_r)
            y0 = max(0, cy - sample_r)
            y1 = min(height, cy + sample_r)
            if x1 <= x0 or y1 <= y0:
                continue

            region = arr[y0:y1, x0:x1]
            r = region[:, :, 0].mean()
            g = region[:, :, 1].mean()
            b = region[:, :, 2].mean()
            brightness = region.mean()

            # Walkable: red floor (moderate R, low G, low B, decent brightness)
            # Blocked: black/dark (low brightness) or non-red
            is_walkable = (
                brightness > 40
                and r > 60
                and r > g + 20
                and r > b + 20
            )

            color = walkable_color if is_walkable else blocked_color
            # Draw a small filled circle at the tile center
            dot_r = max(8, int(round(20 * scale)))
            draw.ellipse(
                [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
                fill=color,
            )

    image = Image.alpha_composite(image.convert("RGBA"), overlay)
    out = BytesIO()
    image.convert("RGB").save(out, format="PNG", compress_level=1)
    return out.getvalue()


def annotate_combined(
    png: bytes,
    grid: HexGrid | None = None,
    show_grid: bool = True,
    show_walkability: bool = True,
) -> bytes:
    """Apply both grid labels and walkability highlighting."""
    result = png
    if show_walkability:
        result = annotate_walkability(result, grid)
    if show_grid:
        result = annotate_grid_overlay(result, grid)
    return result
