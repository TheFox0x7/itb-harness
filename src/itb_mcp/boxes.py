"""Bounding-box input model and conversion for MCP tools."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from itb_harness.annotate import BoxAnnotation
from itb_harness.board_tools import parse_boxes_args

from .config import Config


class BoxItem(BaseModel):
    """One labelled box passed to ``set_boxes`` / ``add_boxes``."""

    model_config = ConfigDict(extra="forbid")

    box_2d: list[float] = Field(
        min_length=4,
        max_length=4,
        description=(
            "[xmin, ymin, xmax, ymax] in screenshot pixels (or scaled 1..1000 "
            "coordinates when the server is configured that way)."
        ),
    )
    label: str = Field(
        min_length=1,
        description=(
            'Short label such as "demon", "skeleton", "helltaker", '
            '"spike trap", or "rock".'
        ),
    )


def to_annotations(
    items: list[BoxItem],
    *,
    width: int,
    height: int,
    config: Config,
) -> list[BoxAnnotation]:
    """Validate MCP box items and convert them to pixel annotations."""

    payload = {"boxes": [item.model_dump() for item in items]}
    annotations = parse_boxes_args(
        payload,
        strict_labels=config.strict_box_labels,
        scaled_boxes=config.scaled_boxes,
        width=width,
        height=height,
    )
    for box in annotations:
        if not (
            0 <= box.xmin < box.xmax <= width and 0 <= box.ymin < box.ymax <= height
        ):
            raise ValueError(
                f"Bounding box is outside the screenshot {width}x{height}."
            )
    return annotations
