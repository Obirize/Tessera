"""The one-piece base plate: a floor with walls of the line colour between the stones, so every stone drops
into its own pocket. Bigger than the printer it is cut straight into panels; a stone lying across the cut
holds both sides together.

Coordinates as everywhere in geometry: mm, x right, y up; the plate's underside is z = 0, stones stand on FLOOR.
"""
from __future__ import annotations

import numpy as np
import shapely
from manifold3d import CrossSection, FillRule, Manifold

from .layout import _polys

FLOOR = 1.6          # thickness under the stones
WALL_H = 1.4         # how far the walls rise above the floor (at most half the stone)
CLEAR = 0.15         # play around a stone, every side
MIN_WALL = 0.8       # thinner walls cannot be printed: those pockets open into each other
SEAM = 0.1           # every panel is this much smaller at a cut
NUMBER_T = 0.4       # depth of the number on a pocket's floor


def wall_height(m) -> float:
    return min(WALL_H, 0.5 * float(m.settings.stone_h_mm))


def cell(frame, w_mm, h_mm, nx, ny, col, row):
    """The part of the plate in one panel; row 0 is the top."""
    cw, ch = w_mm / nx, h_mm / ny
    x0, x1 = col * cw + (SEAM if col else 0), (col + 1) * cw - (SEAM if col < nx - 1 else 0)
    y0, y1 = h_mm - (row + 1) * ch + (SEAM if row < ny - 1 else 0), h_mm - row * ch - (SEAM if row else 0)
    return frame.intersection(shapely.box(x0, y0, x1, y1))


def frame_of(m):
    """The artwork's outline in the 3D frame (y up)."""
    w, h = m.size_mm
    if m.outline is None:
        return shapely.box(0, 0, w, h)
    return shapely.transform(m.outline, lambda c: np.column_stack([c[:, 0], h - c[:, 1]]))


def panel_count(m) -> int:
    """How many panels the artwork really has: cells its outline does not reach are none."""
    from .layout import panel_xy
    w, h = m.size_mm
    nx, ny = panel_xy(w, h, m.settings.panel_max_mm)
    if m.outline is None:
        return nx * ny
    fr = frame_of(m)
    return sum(cell(fr, w, h, nx, ny, c, r).area > 1.0 for r in range(ny) for c in range(nx))


def walls(region, stones):
    """The walls of one panel (2D). stones: every Stone that reaches into it."""
    if not stones:
        return region
    pockets = shapely.union_all([st.poly.buffer(CLEAR, join_style="mitre", mitre_limit=2.0) for st in stones])
    r = MIN_WALL / 2
    w = region.difference(pockets).buffer(-r, quad_segs=3).buffer(r - 0.005, quad_segs=3).intersection(region)
    return w


def _section(g):
    rings = []
    for p in _polys(g):
        if p.area > 0.05:
            rings.append(np.asarray(p.exterior.coords)[:-1])
            rings += [np.asarray(q.coords)[:-1] for q in p.interiors]
    return CrossSection(rings, FillRule.EvenOdd)


def build_plate(L, pn, numbers: bool = False) -> Manifold:
    """Solid of one panel of the base plate. numbers: each pocket carries the number of its stone."""
    region = pn.region
    here = [st for st in L.stones if region.intersects(st.poly)]
    body = _section(region).extrude(FLOOR)
    w = walls(region, here)
    if not w.is_empty:
        body = body + _section(w).extrude(FLOOR + wall_height(L.mosaic))
    if numbers:
        from .text import place_text
        cuts = []
        for st in here:
            tg = None
            if st.width >= 3.0 and st.panel == pn.index:
                tg, _ = place_text(st.poly.intersection(region), [f"{st.number}"], (4.0, 3.0, 2.2), 0.0, False, 0.3)
            if tg is not None:
                cuts.append(tg)
        if cuts:
            body = body - _section(shapely.union_all(cuts)).extrude(NUMBER_T + 1).translate((0, 0, FLOOR - NUMBER_T))
    return body


def plate_volume(m) -> float:
    """mm3 of the whole plate (fast: from areas)."""
    free = max(0.0, m.plate.area - sum(p.area for p in m.polys))
    return m.plate.area * FLOOR + free * wall_height(m)
