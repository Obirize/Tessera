"""Shared toolkit for the cat-symbol iterations (pure Python + shapely, no Blender).

A design is a list of Stone records. Each stone is a hand-designed polygon in the unit frame
[-1, 1]^2 (y up). Regions are cut into stones with explicit, drawn cut lines (polygonize), so
every gap is a deliberate line of the drawing, not noise.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw
from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import polygonize, unary_union

# ------------------------------------------------------------------ palette (sRGB 0-255)
PAL = {
    "base": (16, 15, 14),        # grout / base plate (ink)
    "ink": (40, 37, 34),         # ink stone (lighter than the base so it still reads as stone)
    "ink2": (31, 29, 27),
    "paper": (240, 231, 214),
    "paper2": (229, 217, 195),
    "paper3": (218, 203, 178),
    "w0": (252, 250, 245),       # brightest fur
    "w1": (243, 239, 230),
    "w2": (228, 222, 210),
    "w3": (205, 197, 184),       # fur shadow
    "red": (184, 48, 38),
    "red2": (158, 38, 32),
    "red3": (204, 72, 52),
    "gold": (236, 184, 84),      # metallic in the renderer
    "gold2": (214, 160, 66),
    "blue": (58, 136, 206),
    "blue2": (120, 182, 232),
    "amber": (226, 158, 34),
    "amber2": (244, 196, 80),
    "pupil": (20, 18, 17),
    "rose": (214, 146, 136),
    "rose2": (196, 120, 112),
}
METAL = {"gold", "gold2"}


@dataclass
class Stone:
    poly: Polygon
    col: str
    region: str
    gap: float = 1.0            # gap multiplier
    h: float = 1.0              # height multiplier
    order: float = 0.0          # animation order hint (0 first .. 1 last)


@dataclass
class Design:
    key: str
    name: str
    plate: Polygon
    stones: list = field(default_factory=list)
    gap: float = 0.024
    corner: float = 0.012
    plate_col: str = "base"


# ------------------------------------------------------------------ geometry helpers
def catmull(pts, n=10, closed=True):
    """Smooth curve through control points (centripetal-ish uniform Catmull-Rom)."""
    p = np.asarray(pts, float)
    if closed:
        p = np.vstack([p[-1], p, p[0], p[1]])
    else:
        p = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    if not closed:
        out.append(p[-2])
    return [tuple(q) for q in out]


def smooth_poly(pts, n=10):
    return Polygon(catmull(pts, n)).buffer(0)


def mirror_pts(pts):
    return [(-x, y) for x, y in pts]


def sym(poly):
    return affinity.scale(poly, -1, 1, origin=(0, 0))


def circle(c, r, n=96):
    return Point(c).buffer(r, quad_segs=n // 4)


def ellipse(c, rx, ry, ang=0.0):
    e = affinity.scale(Point(0, 0).buffer(1, quad_segs=24), rx, ry)
    e = affinity.rotate(e, ang, origin=(0, 0))
    return affinity.translate(e, *c)


def almond(c, a, b, ang=0.0, pinch=0.75, top=1.0, bot=1.0):
    """Cat-eye almond: pointed ends, fuller top/bottom controlled separately."""
    pts = []
    for t in np.linspace(0, 2 * math.pi, 64, endpoint=False):
        ct, st = math.cos(t), math.sin(t)
        x = a * math.copysign(abs(ct) ** pinch, ct)
        y = b * st * (top if st > 0 else bot)
        pts.append((x, y))
    p = Polygon(pts)
    p = affinity.rotate(p, ang, origin=(0, 0))
    return affinity.translate(p, *c)


def rounded_square(h, r):
    return box(-h + r, -h + r, h - r, h - r).buffer(r, quad_segs=16)


def polyline(pts, smooth=False, n=8):
    return LineString(catmull(pts, n, closed=False) if smooth else pts)


def ext(line: LineString, d=0.07):
    """Extend a polyline at both ends so it fully crosses the region boundary."""
    c = list(line.coords)
    (x0, y0), (x1, y1) = c[0], c[1]
    (xa, ya), (xb, yb) = c[-2], c[-1]
    l0 = math.hypot(x1 - x0, y1 - y0) or 1
    l1 = math.hypot(xb - xa, yb - ya) or 1
    s = (x0 - (x1 - x0) / l0 * d, y0 - (y1 - y0) / l0 * d)
    e = (xb + (xb - xa) / l1 * d, yb + (yb - ya) / l1 * d)
    return LineString([s] + c + [e])


def as_polys(g):
    if g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]


def cut(region, lines, min_area=2e-4):
    """Partition `region` with drawn cut lines; returns faces inside the region."""
    region = region.buffer(0)
    bnd = [region.boundary]
    lines = [ext(l) if isinstance(l, LineString) else l for l in lines]
    noded = unary_union(bnd + lines)
    faces = []
    for f in polygonize(noded):
        if f.area < min_area:
            continue
        rp = f.representative_point()
        if region.contains(rp):
            faces.append(f.intersection(region).buffer(0))
    out = []
    for f in faces:
        out += [p for p in as_polys(f) if p.area >= min_area]
    return out


def rays(center, angles_deg, r=3.0):
    cx, cy = center
    return [LineString([(cx, cy), (cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))])
            for a in angles_deg]


def qbez(p0, p1, bulge, n=8):
    """Quadratic curve p0->p1 bowed sideways by `bulge` (fraction of the chord, + = left)."""
    (x0, y0), (x1, y1) = p0, p1
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    cx, cy = mx - dy * bulge, my + dx * bulge
    out = []
    for t in np.linspace(0, 1, n):
        a, b, c = (1 - t) ** 2, 2 * (1 - t) * t, t * t
        out.append((a * x0 + b * cx + c * x1, a * y0 + b * cy + c * y1))
    return out


def flame_band(seq, inner, closing, bulge_out=-0.18, bulge_in=0.08, bulge_cut=-0.10, mirror=True):
    """Fur band made of flame-shaped tufts (left half designed, right half mirrored).

    seq: [start, tip, valley, tip, valley, ..., last_tip] outer outline of the LEFT half, running
         down/counter-clockwise; last_tip sits on the axis when mirror=True.
    inner(valley) -> point inside the head where the dividing line between two tufts ends.
    closing: points that close the band behind the head (left to right).
    """
    half = [seq[0]]
    for i in range(len(seq) - 1):
        rising = (i % 2 == 0)
        half += qbez(seq[i], seq[i + 1], bulge_out if rising else bulge_in)[1:]
    lines = []
    for i in range(2, len(seq) - 1, 2):
        v = seq[i]
        lines.append(LineString(qbez(inner(v), v, bulge_cut, 10)))
    if mirror:
        right = [(-x, y) for x, y in half[::-1]][1:]
        outline = half + right
        lines += [LineString([(-x, y) for x, y in l.coords]) for l in lines]
    else:
        outline = half
    band = Polygon(outline + list(closing)).buffer(0)
    return band, lines


def arcs(center, radii, n=180):
    return [circle(center, r, n).exterior for r in radii]


def split_holes(p, depth=0):
    """A stone must never surround another stone: cut through every hole."""
    if not p.interiors or depth > 6:
        return [p]
    h = Polygon(p.interiors[0])
    c = h.centroid
    minx, miny, maxx, maxy = p.bounds
    line = LineString([(c.x, miny - 1), (c.x, maxy + 1)])
    parts = cut(p.buffer(0), [line])
    out = []
    for q in parts:
        out += split_holes(q, depth + 1)
    return out


def merge_thin(faces, w):
    """Faces that would vanish once the gap is cut are merged into their best neighbour."""
    faces = list(faces)
    changed = True
    while changed:
        changed = False
        for i, f in enumerate(faces):
            if not f.buffer(-w).is_empty or len(faces) == 1:
                continue
            best, bl = None, 0.0
            for j, g in enumerate(faces):
                if j == i:
                    continue
                l = f.buffer(1e-4).intersection(g.boundary).length
                if l > bl:
                    best, bl = j, l
            if best is None:
                continue
            faces[best] = unary_union([faces[best], f.buffer(1e-5)]).buffer(-1e-5)
            if not isinstance(faces[best], Polygon):
                faces[best] = max(as_polys(faces[best]), key=lambda q: q.area)
            faces.pop(i)
            changed = True
            break
    return faces


# ------------------------------------------------------------------ design builder
class Builder:
    """Priority painter: regions added first win; later regions only get what is left."""

    def __init__(self, design: Design):
        self.d = design
        self.taken = Polygon()

    def region(self, name, shape, colour, lines=(), gap=1.0, h=1.0, order=0.5, min_area=6e-4):
        shape = shape.buffer(0).difference(self.taken).buffer(0)
        shape = shape.intersection(self.d.plate).buffer(0)
        if shape.is_empty:
            return []
        self.taken = unary_union([self.taken, shape]).buffer(0)
        faces = []
        for part in as_polys(shape):
            faces += cut(part, list(lines), min_area) if lines else [part]
        faces = [q for f in faces for q in split_holes(f)]
        faces = merge_thin(faces, self.d.gap * gap / 2 + self.d.corner + 0.008)
        out = []
        for i, f in enumerate(faces):
            c = colour(f, i) if callable(colour) else colour
            if c is None:
                continue
            st = Stone(f, c, name, gap, h, order)
            out.append(st)
            self.d.stones.append(st)
        return out


def shrink(st: Stone, d: Design):
    g = d.gap * st.gap / 2
    r = d.corner
    p = st.poly.buffer(-(g + r), join_style="mitre", mitre_limit=3).buffer(r, quad_segs=3)
    p = p.buffer(0)
    ps = [q.simplify(0.0015) for q in as_polys(p) if q.area > 4e-4]
    return ps


# ------------------------------------------------------------------ outputs
def export(d: Design, path):
    stones = []
    for s in d.stones:
      for p in shrink(s, d):
        ext_ = list(p.exterior.coords)[:-1]
        if Polygon(ext_).exterior.is_ccw is False:
            ext_ = ext_[::-1]
        c = p.centroid
        stones.append({"pts": [[round(x, 5), round(y, 5)] for x, y in ext_], "col": s.col,
                       "region": s.region, "h": s.h, "order": s.order,
                       "cx": round(c.x, 5), "cy": round(c.y, 5), "area": round(p.area, 6)})
    plate = list(d.plate.exterior.coords)[:-1]
    data = {"key": d.key, "name": d.name, "plate": [[round(x, 5), round(y, 5)] for x, y in plate],
            "plate_col": d.plate_col, "palette": {k: list(v) for k, v in PAL.items()},
            "metal": sorted(METAL), "stones": stones}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return len(stones)


def preview(d: Design, path, size=640):
    s = size / 2.2
    img = Image.new("RGB", (size, size), (236, 228, 212))
    dr = ImageDraw.Draw(img)

    def tr(pts):
        return [(size / 2 + x * s, size / 2 - y * s) for x, y in pts]

    dr.polygon(tr(list(d.plate.exterior.coords)), fill=PAL[d.plate_col])
    for st in d.stones:
        for p in shrink(st, d):
          dr.polygon(tr(list(p.exterior.coords)), fill=PAL[st.col])
    img.save(path)
    return img
