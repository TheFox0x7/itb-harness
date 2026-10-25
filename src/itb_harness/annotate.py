"""Small image annotation helpers for tool feedback."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from .grid import IsoGrid


def annotate_click(png: bytes, x: int, y: int, label: str = "click") -> bytes:
    """Return a PNG with a visible marker at the clicked screenshot pixel."""

    image = Image.open(BytesIO(png)).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    radius = max(12, int(min(width, height) * 0.018))
    x = max(0, min(width - 1, int(x)))
    y = max(0, min(height - 1, int(y)))

    # High-contrast ring + crosshair.
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        outline=(20, 20, 20),
        width=5,
    )
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        outline=(255, 70, 60),
        width=3,
    )
    arm = radius + 7
    for start, end in (
        ((x - arm, y), (x - radius, y)),
        ((x + radius, y), (x + arm, y)),
        ((x, y - arm), (x, y - radius)),
        ((x, y + radius), (x, y + arm)),
    ):
        draw.line([start, end], fill=(255, 70, 60), width=3)

    # Small label near the marker.
    font = ImageFont.load_default()
    text = f"{label} ({x}, {y})"
    box = draw.textbbox((0, 0), text, font=font)
    text_width = box[2] - box[0]
    text_height = box[3] - box[1]
    label_x = x + radius + 10
    label_y = y - text_height - 10
    if label_x + text_width + 8 > width:
        label_x = x - radius - text_width - 18
    if label_y < 4:
        label_y = y + radius + 8

    draw.rectangle(
        [label_x - 4, label_y - 3, label_x + text_width + 4, label_y + text_height + 3],
        fill=(15, 15, 15),
    )
    draw.text((label_x, label_y), text, fill=(255, 120, 110), font=font)

    out = BytesIO()
    image.save(out, format="PNG", compress_level=1)
    return out.getvalue()


def annotate_grid(
    png: bytes,
    grid: IsoGrid,
    highlight_tiles: list[tuple[float, float]] | None = None,
    show_labels: bool = True,
    show_diamonds: bool = True,
    show_centers: bool = True,
) -> bytes:
    """Overlay the isometric tile grid on a screenshot.

    Draws diamond outlines for each tile, center markers, and pixel-
    coordinate labels so the model can read off exact click targets.
    ``highlight_tiles`` is an optional list of ``(x, y)`` pixel positions
    (e.g. drop-zone tiles) to tint with a semi-transparent gold fill.
    """
    image = Image.open(BytesIO(png)).convert("RGB")
    width, height = image.size

    # Work on an RGBA overlay so we can use alpha blending.
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Scale factor for line widths / font sizes relative to 1280×720.
    scale = min(width / 1280.0, height / 720.0)

    # --- Highlight tiles (e.g. drop zone) --------------------------------
    if highlight_tiles:
        for hx, hy in highlight_tiles:
            # Find the nearest tile centre and fill its diamond.
            _sx, _sy, row, col = grid.snap(int(hx), int(hy))
            cx, cy = grid.tile_center(row, col)
            tw2 = int(grid.tile_w2 * 0.92)
            th2 = int(grid.tile_h2 * 0.92)
            draw.polygon(
                [
                    (cx, cy - th2),
                    (cx + tw2, cy),
                    (cx, cy + th2),
                    (cx - tw2, cy),
                ],
                fill=(255, 215, 80, 55),
            )

    # --- Tile diamonds ---------------------------------------------------
    if show_diamonds:
        line_w = max(1, int(round(scale)))
        for row in range(grid.board_size):
            for col in range(grid.board_size):
                cx, cy = grid.tile_center(row, col)
                tw2 = int(grid.tile_w2)
                th2 = int(grid.tile_h2)
                draw.polygon(
                    [
                        (cx, cy - th2),
                        (cx + tw2, cy),
                        (cx, cy + th2),
                        (cx - tw2, cy),
                    ],
                    outline=(255, 255, 255, 90),
                    width=line_w,
                )

    # --- Tile centre markers ---------------------------------------------
    if show_centers:
        arm = max(3, int(round(4 * scale)))
        for row in range(grid.board_size):
            for col in range(grid.board_size):
                cx, cy = grid.tile_center(row, col)
                draw.line(
                    [(cx - arm, cy), (cx + arm, cy)],
                    fill=(255, 255, 255, 120),
                    width=1,
                )
                draw.line(
                    [(cx, cy - arm), (cx, cy + arm)],
                    fill=(255, 255, 255, 120),
                    width=1,
                )

    # Composite the overlay onto the image.
    image = Image.alpha_composite(image.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(image)

    # --- Pixel-coordinate labels -----------------------------------------
    if show_labels:
        font = ImageFont.load_default()
        for row in range(grid.board_size):
            for col in range(grid.board_size):
                cx, cy = grid.tile_center(row, col)
                text = f"{cx},{cy}"
                box = draw.textbbox((0, 0), text, font=font)
                tw = box[2] - box[0]
                th = box[3] - box[1]
                lx = cx - tw // 2
                ly = cy + int(6 * scale)
                # Clamp inside the image.
                lx = max(2, min(width - tw - 4, lx))
                ly = max(2, min(height - th - 4, ly))
                # Dark background for readability.
                draw.rectangle(
                    [lx - 2, ly - 1, lx + tw + 2, ly + th + 1],
                    fill=(0, 0, 0, 160),
                )
                draw.text((lx, ly), text, fill=(255, 255, 160, 255), font=font)

    out = BytesIO()
    image.convert("RGB").save(out, format="PNG", compress_level=1)
    return out.getvalue()
