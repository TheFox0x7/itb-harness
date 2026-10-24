"""Small image annotation helpers for tool feedback."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw, ImageFont


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
