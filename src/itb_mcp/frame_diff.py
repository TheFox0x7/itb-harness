"""Detect whether a movement key actually changed the game.

A blocked move leaves the screenshot almost identical (the character does not
move and no step is consumed); a successful move changes the character
position and usually the step counter.  Comparing mean absolute pixel
difference is cheap and robust enough for that distinction.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

DEFAULT_THRESHOLD = 0.005


def frames_differ(
    png_a: bytes,
    png_b: bytes,
    threshold: float = DEFAULT_THRESHOLD,
) -> bool:
    """Return True when two PNG screenshots differ enough to be a move.

    Undecodable or differently-sized images count as changed so a broken
    capture never reports a false BLOCKED result.
    """

    try:
        img_a = Image.open(BytesIO(png_a)).convert("RGB")
        img_b = Image.open(BytesIO(png_b)).convert("RGB")
    except (OSError, ValueError, SyntaxError):
        return True

    if img_a.size != img_b.size:
        return True

    arr_a = np.asarray(img_a, dtype=np.int16)
    arr_b = np.asarray(img_b, dtype=np.int16)
    diff = np.abs(arr_a - arr_b).mean()
    # Max possible pixel difference is 255, so normalise to 0..1.
    return bool(diff / 255.0 > threshold)


def frame_difference(
    png_a: bytes,
    png_b: bytes,
) -> float:
    """Return the normalised mean absolute pixel difference (0..1)."""

    try:
        img_a = Image.open(BytesIO(png_a)).convert("RGB")
        img_b = Image.open(BytesIO(png_b)).convert("RGB")
    except (OSError, ValueError, SyntaxError):
        return 1.0
    if img_a.size != img_b.size:
        return 1.0

    arr_a = np.asarray(img_a, dtype=np.int16)
    arr_b = np.asarray(img_b, dtype=np.int16)
    return float(np.abs(arr_a - arr_b).mean() / 255.0)
