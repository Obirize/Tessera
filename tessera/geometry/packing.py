"""Plate packing: MaxRects (best short side fit) on oriented bounding boxes.

Each item is a 2D outline (shapely). It is rotated so its minimum-area rectangle is axis
aligned (and optionally by 90 deg), then packed with `spacing` between rectangles. The plate's
excluded corner (X1C purge chute, bed_exclude_area 18x28 mm at the front-left) is kept free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely import affinity


@dataclass
class Placement:
    item: int
    plate: int
    angle: float          # degrees, applied about the outline's (0, 0)
    dx: float             # translation applied after rotation (plate-local mm)
    dy: float


@dataclass
class _Plate:
    free: list = field(default_factory=list)   # (x, y, w, h)
    used_area: float = 0.0


def _oriented(g):
    """Angle (deg) that makes the minimum rotated rectangle axis-aligned."""
    try:
        rr = g.minimum_rotated_rectangle
        c = np.asarray(rr.exterior.coords)
        e = c[1] - c[0]
        return -math.degrees(math.atan2(e[1], e[0]))
    except Exception:
        return 0.0


class MaxRects:
    def __init__(self, w, h, exclude=()):
        self.w, self.h = w, h
        self.plates: list[_Plate] = []
        self.exclude = list(exclude)
        self.first_open = 0           # plates before this index are closed (panel order)

    def _new_plate(self):
        p = _Plate([(0.0, 0.0, self.w, self.h)])
        for ex in self.exclude:
            self._split(p, ex)
        self.plates.append(p)
        return p

    @staticmethod
    def _split(p: _Plate, r):
        rx, ry, rw, rh = r
        F = np.asarray(p.free, float).reshape(-1, 4)
        x, y, w, h = F.T
        hit = ~((rx >= x + w) | (rx + rw <= x) | (ry >= y + h) | (ry + rh <= y))
        H = F[hit]
        hx, hy, hw, hh = H.T
        new = np.concatenate([np.column_stack([hx, hy, rx - hx, hh]),
                              np.column_stack([np.full(len(H), rx + rw), hy, hx + hw - rx - rw, hh]),
                              np.column_stack([hx, hy, hw, ry - hy]),
                              np.column_stack([hx, np.full(len(H), ry + rh), hw, hy + hh - ry - rh])])
        new = np.unique(new[(new[:, 2] > 1e-6) & (new[:, 3] > 1e-6)], axis=0)
        A = np.concatenate([F[~hit], new])
        # only the new rectangles can be contained in another one
        e = 1e-9
        B = new[:, None, :]
        O = A[None, :, :]
        inside = ((O[..., 0] <= B[..., 0] + e) & (O[..., 1] <= B[..., 1] + e)
                  & (B[..., 0] + B[..., 2] <= O[..., 0] + O[..., 2] + e)
                  & (B[..., 1] + B[..., 3] <= O[..., 1] + O[..., 3] + e))
        k = np.arange(len(new))
        inside[k, len(A) - len(new) + k] = False
        p.free = np.concatenate([F[~hit], new[~inside.any(1)]])

    def _find(self, p: _Plate, w, h):
        F = np.asarray(p.free, float).reshape(-1, 4)
        if not len(F):
            return None
        best = None
        for rot, (iw, ih) in ((False, (w, h)), (True, (h, w))):
            dx, dy = F[:, 2] - iw, F[:, 3] - ih
            ok = np.flatnonzero((dx >= -1e-9) & (dy >= -1e-9))
            if not len(ok):
                continue
            a, b = np.minimum(dx[ok], dy[ok]), np.maximum(dx[ok], dy[ok])
            q = np.lexsort((b, a))[0]
            if best is None or (a[q], b[q]) < best[0]:
                best = ((a[q], b[q]), float(F[ok[q], 0]), float(F[ok[q], 1]), rot)
        return best

    def insert(self, w, h):
        """Returns (plate, x, y, rotated) or None if it can never fit."""
        if min(w, h) > min(self.w, self.h) or max(w, h) > max(self.w, self.h):
            return None
        for i, p in enumerate(self.plates):
            if i < self.first_open:
                continue
            f = self._find(p, w, h)
            if f:
                _, x, y, rot = f
                iw, ih = (h, w) if rot else (w, h)
                self._split(p, (x, y, iw, ih))
                p.used_area += w * h
                return i, x, y, rot
        p = self._new_plate()
        f = self._find(p, w, h)
        if not f:
            self.plates.pop()
            return None
        _, x, y, rot = f
        iw, ih = (h, w) if rot else (w, h)
        self._split(p, (x, y, iw, ih))
        p.used_area += w * h
        return len(self.plates) - 1, x, y, rot


def pack_outlines(outlines, plate_mm=256.0, margin=3.0, spacing=3.0,
                  exclude=((0, 0, 18, 28),), groups=None):
    """Pack outlines onto as few plates as possible. Returns (placements, n_plates, unplaced).

    The usable square is [margin, plate-margin]; rectangles are inflated by `spacing`.
    groups (e.g. panel per outline): plates are filled group by group in ascending order; only
    the last plate of the previous group stays open, so plates come out sorted by panel."""
    usable = plate_mm - 2 * margin + spacing
    ex = [(x - margin, y - margin, w + spacing, h + spacing) for x, y, w, h in exclude]
    mr = MaxRects(usable, usable, ex)
    info = []
    for i, g in enumerate(outlines):
        ang = _oriented(g)
        gr = affinity.rotate(g, ang, origin=(0, 0))
        x0, y0, x1, y1 = gr.bounds
        info.append((i, ang, x0, y0, x1 - x0, y1 - y0))
    big_first = lambda t: -(max(t[4], t[5]) * 1000 + t[4] * t[5])  # noqa: E731
    if groups is None:
        order = sorted(info, key=big_first)
    else:
        order = sorted(info, key=lambda t: (groups[t[0]], big_first(t)))
    placements, unplaced = [], []
    cur = None
    for i, ang, x0, y0, w, h in order:
        if groups is not None and groups[i] != cur:
            cur = groups[i]
            mr.first_open = max(0, len(mr.plates) - 1)
        r = mr.insert(w + spacing, h + spacing)
        if r is None:
            # try the unrotated outline (an axis-aligned panel-sized part may fit only that way)
            g = outlines[i]
            gx0, gy0, gx1, gy1 = g.bounds
            r = mr.insert(gx1 - gx0 + spacing, gy1 - gy0 + spacing)
            if r is None:
                unplaced.append(i)
                continue
            ang, x0, y0, w, h = 0.0, gx0, gy0, gx1 - gx0, gy1 - gy0
        plate, x, y, rot = r
        a = ang
        if rot:
            a = ang + 90.0
            # rotating (x0..x0+w, y0..y0+h) by +90 deg about origin gives x in [-y1, -y0]
            nx0, ny0 = -(y0 + h), x0
        else:
            nx0, ny0 = x0, y0
        placements.append(Placement(i, plate, a, margin + x - nx0, margin + y - ny0))
    return placements, len(mr.plates), unplaced
