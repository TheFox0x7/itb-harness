"""Helltaker grid overlay for screenshots sent to the model.

The grid uses 68x72 pixel cells. The bottom edge is anchored to the first row
(scanned upward) that contains at least one continuous cell-width of red floor
pixels. Rows then propagate upward, leaving a partial row at the top when the
anchored height is not a multiple of 72.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CELL_WIDTH = 68
CELL_HEIGHT = 72
COLUMNS = 9
MIN_RED_RUN = CELL_WIDTH

# Same filter as the web viewer grid in web/static/app.js.
_FLOOR_RED = (120, 220, 255, 97)
_MAP_COLORS = {
    "H": (126, 240, 255, 255),
    "S": (242, 242, 242, 255),
    "T": (255, 107, 95, 255),
    "D": (255, 141, 240, 255),
    "R": (255, 179, 199, 255),
    "X": (154, 167, 180, 255),
    "K": (255, 209, 102, 255),
    "B": (255, 159, 67, 255),
}
_MAP_DEFAULT_COLOR = (255, 209, 102, 255)


def red_floor_mask(arr: np.ndarray) -> np.ndarray:
    """Boolean mask for the Helltaker floor reds around #8c1c31/#700c29."""

    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    return (
        (r >= 80)
        & (r <= 200)
        & (g <= 80)
        & (b <= 110)
        & ((r - g) >= 45)
        & ((r - b) >= 30)
    )


def find_floor_anchor(mask: np.ndarray) -> int | None:
    """Return the bottom row of the first wide-enough red run, or None."""

    height = mask.shape[0]
    for y in range(height - 1, -1, -1):
        row = mask[y].astype(np.int8)
        padded = np.concatenate((np.zeros(1, dtype=np.int8), row, np.zeros(1, dtype=np.int8)))
        diff = np.diff(padded)
        starts = np.flatnonzero(diff == 1)
        if starts.size == 0:
            continue
        ends = np.flatnonzero(diff == -1)
        if int((ends - starts).max()) >= MIN_RED_RUN:
            return y
    return None


def _draw_level_map(
    draw: ImageDraw.ImageDraw,
    rows: list[list[str]],
    *,
    width: int,
    height: int,
    bottom: int,
) -> None:
    columns = max(
        (len(row) for row in rows if isinstance(row, list)),
        default=0,
    )
    if not columns or not rows:
        return

    left = width / 2.0 - (columns * CELL_WIDTH) / 2.0
    font = ImageFont.load_default(size=20)

    for row_index, cells in enumerate(rows):
        if not isinstance(cells, list):
            continue
        for col_index, value in enumerate(cells):
            symbol = str(value).strip()
            if not symbol:
                continue
            x = int(
                round(
                    left
                    + col_index * CELL_WIDTH
                    + CELL_WIDTH / 2.0
                )
            )
            y = int(
                round(
                    bottom
                    - (len(rows) - 1 - row_index) * CELL_HEIGHT
                    - CELL_HEIGHT / 2.0
                )
            )
            if not (0 <= x < width and 0 <= y < height):
                continue

            if symbol == ".":
                draw.ellipse(
                    [x - 3, y - 3, x + 3, y + 3],
                    fill=(220, 230, 240, 128),
                )
                continue

            bbox = draw.textbbox((0, 0), symbol, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.rectangle(
                [
                    x - text_width // 2 - 5,
                    y - text_height // 2 - 3,
                    x + text_width // 2 + 5,
                    y + text_height // 2 + 3,
                ],
                fill=(8, 12, 16, 184),
            )
            draw.text(
                (x, y),
                symbol,
                fill=_MAP_COLORS.get(symbol, _MAP_DEFAULT_COLOR),
                font=font,
                anchor="mm",
            )


def annotate_helltaker_grid(
    png: bytes,
    level_map: list[list[str]] | None = None,
) -> bytes:
    """Return a PNG with the anchored Helltaker grid drawn on top.

    When ``level_map`` is provided, its symbols are drawn into the matching
    grid cells.
    """

    image = Image.open(BytesIO(png)).convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        return png

    anchor = find_floor_anchor(red_floor_mask(np.asarray(image)))
    bottom = height if anchor is None else min(height, anchor + 1)

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    left = width / 2.0 - (COLUMNS * CELL_WIDTH) / 2.0
    right = left + COLUMNS * CELL_WIDTH

    for col in range(COLUMNS + 1):
        x = int(round(left + col * CELL_WIDTH))
        if 0 <= x < width:
            draw.line([(x, 0), (x, bottom)], fill=_FLOOR_RED, width=1)

    # Anchor the bottom row to the detected floor edge and propagate upward.
    for y in range(bottom, -1, -CELL_HEIGHT):
        draw.line(
            [(max(0, int(round(left))), y), (min(width - 1, int(round(right))), y)],
            fill=_FLOOR_RED,
            width=1,
        )

    if level_map:
        _draw_level_map(
            draw,
            level_map,
            width=width,
            height=height,
            bottom=bottom,
        )

    image = Image.alpha_composite(image.convert("RGBA"), overlay)
    out = BytesIO()
    image.convert("RGB").save(out, format="PNG", compress_level=1)
    return out.getvalue()
