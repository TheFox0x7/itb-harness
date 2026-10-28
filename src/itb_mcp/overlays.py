"""Apply persistent annotations to screenshots."""

from __future__ import annotations

from collections.abc import Sequence

from itb_harness.annotate import BoxAnnotation, annotate_boxes
from itb_harness.board_overlay import annotate_combined, default_grid


def apply_overlays(
    png: bytes,
    width: int,
    height: int,
    *,
    boxes: Sequence[BoxAnnotation] = (),
    show_grid: bool = False,
    show_walkability: bool = False,
) -> bytes:
    """Draw boxes first, then the grid/walkability layers.

    Boxes are drawn underneath the grid so coordinate labels stay readable.
    """

    if boxes:
        png = annotate_boxes(png, list(boxes))
    if show_grid or show_walkability:
        png = annotate_combined(
            png,
            grid=default_grid(width, height),
            show_grid=show_grid,
            show_walkability=show_walkability,
        )
    return png
