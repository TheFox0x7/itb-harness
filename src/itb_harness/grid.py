"""Isometric grid detection and click snapping for Into the Breach.

The board is a regular 8×8 isometric grid.  Tile centres follow a linear
transform:

    x = origin_x + (col - row) * tile_w2
    y = origin_y + (col + row) * tile_h2

where ``tile_w2`` / ``tile_h2`` are half the diamond width / height in
screen pixels.  During deployment the tan drop-zone tiles are visible and
can be detected by colour masking; their centroids are used to calibrate
the grid at runtime.  The calibrated grid then persists for the rest of
the game so that clicks can be snapped to the nearest tile centre.

The default parameters below were measured from 1280×720 screenshots and
scale linearly with the window size.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

# Default grid for a 1280×720 screenshot (measured from recorded runs).
# The board diamond spans roughly x=194..1084 at the vertical centre.
_DEFAULT_1280x720 = dict(
    origin_x=639.0,
    origin_y=78.0,
    tile_w2=55.0,
    tile_h2=42.0,
)


@dataclass(slots=True)
class IsoGrid:
    """Isometric grid parameters for an 8×8 ITB board."""

    origin_x: float  # screen x of tile (0, 0) centre
    origin_y: float  # screen y of tile (0, 0) centre
    tile_w2: float  # half tile width  (screen px)
    tile_h2: float  # half tile height (screen px)
    board_size: int = 8

    # ------------------------------------------------------------------
    # forward / inverse transforms
    # ------------------------------------------------------------------

    def tile_center(self, row: int, col: int) -> tuple[int, int]:
        """Screen centre of tile ``(row, col)``."""
        x = self.origin_x + (col - row) * self.tile_w2
        y = self.origin_y + (col + row) * self.tile_h2
        return int(round(x)), int(round(y))

    def snap(
        self, px: int, py: int
    ) -> tuple[int, int, int, int]:
        """Snap a screenshot pixel to the nearest tile centre.

        Returns ``(snapped_x, snapped_y, row, col)``.
        """
        cr = (px - self.origin_x) / self.tile_w2  # col - row
        cs = (py - self.origin_y) / self.tile_h2  # col + row
        col = round((cs + cr) / 2)
        row = round((cs - cr) / 2)
        col = max(0, min(self.board_size - 1, col))
        row = max(0, min(self.board_size - 1, row))
        sx, sy = self.tile_center(row, col)
        return sx, sy, row, col

    def all_centers(self) -> list[tuple[int, int, int, int]]:
        """All tile centres as ``(x, y, row, col)``."""
        out: list[tuple[int, int, int, int]] = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                x, y = self.tile_center(row, col)
                out.append((x, y, row, col))
        return out

    # ------------------------------------------------------------------
    # scaling
    # ------------------------------------------------------------------

    def scaled(self, width: int, height: int) -> IsoGrid:
        """Return a copy scaled for a ``width``×``height`` screenshot."""
        sx = width / 1280.0
        sy = height / 720.0
        return IsoGrid(
            origin_x=self.origin_x * sx,
            origin_y=self.origin_y * sy,
            tile_w2=self.tile_w2 * sx,
            tile_h2=self.tile_h2 * sy,
            board_size=self.board_size,
        )


# ----------------------------------------------------------------------
# Drop-zone tile detection
# ----------------------------------------------------------------------

def _tan_mask(arr: np.ndarray) -> np.ndarray:
    """Boolean mask of tan / khaki drop-zone tiles."""
    r = arr[:, :, 0].astype(int)
    g = arr[:, :, 1].astype(int)
    b = arr[:, :, 2].astype(int)
    return (
        (r > 100)
        & (r >= g - 10)
        & (g > 90)
        & (b < 120)
        & (r - b > 30)
        & (r - b < 80)
    )


def _erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Binary erosion (separates tiles connected by thin grid lines)."""
    out = mask
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        windows = sliding_window_view(padded, (3, 3))
        out = windows.all(axis=(2, 3))
    return out


def _connected_components(
    mask: np.ndarray, min_size: int = 200
) -> list[tuple[float, float, int]]:
    """Return ``(cx, cy, size)`` for each connected component."""
    visited = np.zeros_like(mask, dtype=bool)
    components: list[tuple[float, float, int]] = []
    h, w = mask.shape

    ys, xs = np.where(mask & ~visited)
    for y0, x0 in zip(ys, xs):
        if visited[y0, x0]:
            continue
        # BFS flood fill
        queue: deque[tuple[int, int]] = deque([(int(y0), int(x0))])
        visited[y0, x0] = True
        pixels: list[tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            for dy, dx in (
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1),
            ):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        if len(pixels) >= min_size:
            pys = np.array([p[0] for p in pixels])
            pxs = np.array([p[1] for p in pixels])
            components.append(
                (float(pxs.mean()), float(pys.mean()), len(pixels))
            )
    return components


def detect_drop_zone_tiles(
    png_bytes: bytes,
) -> list[tuple[float, float]]:
    """Detect tan drop-zone tiles in a screenshot.

    Returns a list of ``(center_x, center_y)`` pixel coordinates.
    """
    img = Image.open(BytesIO(png_bytes)).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    mask = _tan_mask(arr)
    eroded = _erode(mask, iterations=1)
    # min_size=1000 filters out small UI elements (mech panel icons etc.)
    components = _connected_components(eroded, min_size=1000)
    # Keep only tiles inside the approximate board area.  The board diamond
    # spans roughly x=190..1090, y=30..710 at 1280×720; scale for other sizes.
    sx, sy = w / 1280.0, h / 720.0
    x_lo, x_hi = 190 * sx, 1090 * sx
    y_lo, y_hi = 30 * sy, 710 * sy
    return [
        (cx, cy)
        for cx, cy, _size in components
        if x_lo <= cx <= x_hi and y_lo <= cy <= y_hi
    ]


# ----------------------------------------------------------------------
# Grid calibration
# ----------------------------------------------------------------------

def _estimate_tile_spacing(
    centers: list[tuple[float, float]],
) -> tuple[float, float]:
    """Estimate ``(tile_w2, tile_h2)`` from nearest-neighbour distances.

    In isometric space adjacent tiles differ by ``(+tile_w2, +tile_h2)``
    (col+1) or ``(-tile_w2, +tile_h2)`` (row+1).  We collect all pairwise
    deltas, keep those with ``dy > 0`` and small ``|dx|``, and take the
    median of ``|dx|`` and ``dy``.
    """
    n = len(centers)
    dxs: list[float] = []
    dys: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[j][0] - centers[i][0]
            dy = centers[j][1] - centers[i][1]
            if dy > 5 and abs(dx) < 80 and dy < 80:
                dxs.append(abs(dx))
                dys.append(dy)
    if not dxs:
        # Fallback: use the default ratio
        return _DEFAULT_1280x720["tile_w2"], _DEFAULT_1280x720["tile_h2"]
    return float(np.median(dxs)), float(np.median(dys))


def _fit_origin(
    centers: list[tuple[float, float]],
    tile_w2: float,
    tile_h2: float,
    board_size: int = 8,
) -> tuple[float, float, float] | None:
    """Fit ``(origin_x, origin_y, total_error)`` from tile centres.

    Tries every plausible ``(row, col)`` assignment for the first tile and
    picks the one that places all tiles on (or very near) integer grid
    positions with the lowest total error.
    """
    if not centers:
        return None

    best: tuple[float, float, float] | None = None

    # Use the first tile as the anchor and try all grid positions.
    ax, ay = centers[0]
    for r0 in range(board_size):
        for c0 in range(board_size):
            ox = ax - (c0 - r0) * tile_w2
            oy = ay - (c0 + r0) * tile_h2
            total_err = 0.0
            valid = True
            for tx, ty in centers:
                cr = (tx - ox) / tile_w2
                cs = (ty - oy) / tile_h2
                col = (cs + cr) / 2
                row = (cs - cr) / 2
                if row < -0.6 or row > board_size - 0.4:
                    valid = False
                    break
                if col < -0.6 or col > board_size - 0.4:
                    valid = False
                    break
                px = ox + (col - row) * tile_w2
                py = oy + (col + row) * tile_h2
                total_err += ((px - tx) ** 2 + (py - ty) ** 2) ** 0.5
            if valid and (best is None or total_err < best[2]):
                best = (ox, oy, total_err)

    return best


def _detect_board_bounds(
    png_bytes: bytes,
) -> tuple[float, float] | None:
    """Detect the board diamond's leftmost and rightmost screen x.

    The board is a bright diamond on a dark background.  We find the row
    with the widest bright span (near the diamond's vertical centre) and
    return ``(left_x, right_x)`` at that row.
    """
    img = Image.open(BytesIO(png_bytes)).convert("RGB")
    arr = np.array(img).astype(int)
    brightness = arr.mean(axis=2)
    h, w = brightness.shape

    # Board tiles are brighter than the surrounding dark frame / UI.
    board_mask = brightness > 40

    # For each row, find the widest contiguous bright span.
    best_width = 0
    best_row = h // 2
    best_left = 0
    best_right = 0
    for y in range(0, h, 2):
        row = board_mask[y]
        xs = np.where(row)[0]
        if len(xs) < 100:
            continue
        # Find the widest contiguous run.
        splits = np.where(np.diff(xs) > 10)[0]
        runs = np.split(xs, splits + 1)
        widest = max(runs, key=len)
        if len(widest) > best_width:
            best_width = len(widest)
            best_row = y
            best_left = int(widest[0])
            best_right = int(widest[-1])

    if best_width < 200:
        return None
    return float(best_left), float(best_right)


def calibrate_grid(
    png_bytes: bytes,
    width: int,
    height: int,
) -> IsoGrid | None:
    """Calibrate the isometric grid from a deployment screenshot.

    Detects the tan drop-zone tiles and the board bounds, then fits the
    grid parameters.  Returns ``None`` if calibration fails.
    """
    centers = detect_drop_zone_tiles(png_bytes)
    if len(centers) < 2:
        return None

    tile_w2, tile_h2 = _estimate_tile_spacing(centers)

    # Detect board bounds to pin down origin_x and tile_w2.
    bounds = _detect_board_bounds(png_bytes)
    if bounds is not None:
        left_x, right_x = bounds
        # Board diamond width = board_size * tile_w  →  tile_w2 = width / (2*N)
        board_w = right_x - left_x
        tile_w2 = board_w / (2 * 8)  # 8 tiles wide
        origin_x = (left_x + right_x) / 2.0
    else:
        # Fall back to fitting origin_x from the tiles alone.
        fit = _fit_origin(centers, tile_w2, tile_h2)
        if fit is None:
            return None
        origin_x, _oy, _err = fit

    # Now find origin_y.  Try every plausible assignment and pick the one
    # that (a) best fits the tiles and (b) keeps the board mostly on-screen.
    best: tuple[float, float] | None = None  # (origin_y, total_error)
    ax, ay = centers[0]
    for r0 in range(8):
        for c0 in range(8):
            oy = ay - (c0 + r0) * tile_h2
            total_err = 0.0
            valid = True
            for tx, ty in centers:
                cr = (tx - origin_x) / tile_w2
                cs = (ty - oy) / tile_h2
                col = (cs + cr) / 2
                row = (cs - cr) / 2
                if row < -0.6 or row > 7.4 or col < -0.6 or col > 7.4:
                    valid = False
                    break
                px = origin_x + (col - row) * tile_w2
                py = oy + (col + row) * tile_h2
                total_err += ((px - tx) ** 2 + (py - ty) ** 2) ** 0.5
            if not valid:
                continue
            # Sanity: board vertical centre should be near screen centre.
            board_vcentre = oy + 7 * tile_h2
            screen_vcentre = height / 2.0
            penalty = abs(board_vcentre - screen_vcentre) * 0.1
            score = total_err + penalty
            if best is None or score < best[1]:
                best = (oy, score)

    if best is None:
        return None
    origin_y = best[0]

    return IsoGrid(
        origin_x=origin_x,
        origin_y=origin_y,
        tile_w2=tile_w2,
        tile_h2=tile_h2,
    )


def default_grid(width: int, height: int) -> IsoGrid:
    """Return the default grid scaled for ``width``×``height``."""
    g = IsoGrid(**_DEFAULT_1280x720)
    return g.scaled(width, height)
