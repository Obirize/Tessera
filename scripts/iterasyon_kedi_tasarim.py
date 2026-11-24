"""Five hand-designed cat symbol concepts for Tessera (stone layouts).

Run:  .venv/Scripts/python scripts/iterasyon_kedi_tasarim.py <out_dir> [1 2 3 4 5]
Writes <out_dir>/kedi_<n>.json (stones, for Blender) and <out_dir>/kedi_<n>_onizleme.png (flat check).

Every stone is drawn: regions are cut by explicit lines (brow, muzzle, tuft partings, rows...).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shapely import affinity  # noqa: E402
from shapely.geometry import LineString, Point, Polygon, box  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

from iterasyon_kedi_ortak import (Builder, Design, almond, circle, ellipse, export,  # noqa: E402
                                  flame_band, mirror_pts, polyline, preview, qbez, rays,
                                  rounded_square, smooth_poly, sym)


def tone(seq):
    return lambda f, i: seq[i % len(seq)]


class Local:
    """Add regions designed in a local frame (scaled by s, moved to off) to a Builder."""

    def __init__(self, b, s=1.0, off=(0.0, 0.0)):
        self.b, self.s, self.off = b, s, off

    def g(self, geom):
        return affinity.translate(affinity.scale(geom, self.s, self.s, origin=(0, 0)), *self.off)

    def region(self, name, shape, colour, lines=(), **kw):
        if callable(colour):
            f = colour

            def colour(p, i, f=f):
                c = p.centroid
                q = Point((c.x - self.off[0]) / self.s, (c.y - self.off[1]) / self.s)
                return f(_Cen(q), i)
        return self.b.region(name, self.g(shape), colour, [self.g(l) for l in lines], **kw)


class _Cen:
    def __init__(self, c):
        self.centroid = c


def M(sg, pts):
    return [(sg * x, y) for x, y in pts]


def plume(spine, widths, cuts_t, slant=0.06, tip=0.05, bulge=0.22):
    """Fluffy Angora tail: a scalloped plume around a spine, cut into slanted tufts.

    spine: control points base -> tip; widths: half-width at each control point;
    cuts_t: parameters (0..1) of the tuft partings; tufts point towards the tail tip.
    Returns (polygon, cut_lines).
    """
    import numpy as np
    from iterasyon_kedi_ortak import catmull
    P = np.array(catmull(spine, 16, closed=False))
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    u = seg / seg[-1]
    W = np.interp(u, np.linspace(0, 1, len(widths)), widths)
    T = np.gradient(P, axis=0)
    T /= np.hypot(T[:, 0], T[:, 1])[:, None]
    N = np.c_[-T[:, 1], T[:, 0]]

    def at(t, side):
        x = np.interp(t, u, P[:, 0]); y = np.interp(t, u, P[:, 1])
        tx = np.interp(t, u, T[:, 0]); ty = np.interp(t, u, T[:, 1])
        nx, ny = -ty, tx
        w = np.interp(t, u, W)
        return (x + side * nx * w, y + side * ny * w), (tx, ty), (side * nx, side * ny)

    knots = [0.0] + list(cuts_t) + [1.0]
    sides = []
    for side in (1, -1):
        pts = [at(0.0, side)[0]]
        for k in range(1, len(knots)):
            t1 = knots[k] - (slant if side > 0 else 0.0)
            t1 = min(max(t1, 0.0), 1.0)
            e, tg, nn = at(t1, side)
            tipp = (e[0] + tg[0] * tip * 0.5 + nn[0] * tip, e[1] + tg[1] * tip * 0.5 + nn[1] * tip)
            pts += qbez(pts[-1], tipp, bulge * side, 6)[1:]
            if k < len(knots) - 1:
                pts.append(e)
        sides.append(pts)
    tip_pt = (P[-1][0] + T[-1][0] * 0.06, P[-1][1] + T[-1][1] * 0.06)
    outline = sides[0] + [tip_pt] + sides[1][::-1]
    poly = Polygon(outline).buffer(0.004).buffer(-0.004)
    lines = []
    for t in cuts_t:
        a = at(min(max(t - slant, 0), 1), 1)[0]
        b_ = at(t, -1)[0]
        lines.append(LineString([a, b_]))
    return poly, lines


# =====================================================================================
# Angora head (front view) in a local frame: head ~ x +-0.6, ears to y 0.92
# =====================================================================================
def angora_head(L, detail="full", ruff=True, right_shade=True, eye_gap=1.7, inner_col="rose",
                nose_col="rose", order0=0.6):
    """detail: full (forehead M, muzzle, cheeks, ear tufts) | small (few big stones)."""
    shade = (lambda c: {"w0": "w1", "w1": "w2", "w2": "w2", "w3": "w3"}[c]) if right_shade else (lambda c: c)

    # eyes: slanted almonds; the slit pupil only touches the top lid, so the iris stays one stone
    ec = (-0.245, 0.10)
    if detail == "full":
        eL = almond(ec, 0.17, 0.118, ang=-14, pinch=0.78, top=1.12, bot=0.92)
        pL = ellipse((ec[0] + 0.012, ec[1] + 0.05), 0.024, 0.12, ang=-3)
    else:
        ec = (-0.285, 0.09)
        eL = almond(ec, 0.175, 0.135, ang=-14, pinch=0.8, top=1.1, bot=0.92)
        pL = ellipse((ec[0] + 0.014, ec[1] + 0.07), 0.045, 0.15, ang=-3)
    L.region("pupil", pL.intersection(eL), "pupil", order=0.95)
    L.region("pupil", sym(pL).intersection(sym(eL)), "pupil", order=0.95)
    L.region("eye", eL, "blue", gap=eye_gap, order=0.9)
    L.region("eye", sym(eL), "amber", gap=eye_gap, order=0.9)

    nose = Polygon([(-0.06, -0.11), (0.06, -0.11), (0.0, -0.18)]).buffer(0.018)
    if detail != "full":
        nose = Polygon([(-0.085, -0.10), (0.085, -0.10), (0.0, -0.20)]).buffer(0.02)
    L.region("nose", nose, nose_col, order=0.85)

    hr = [(0.0, 0.46), (0.22, 0.45), (0.42, 0.37), (0.55, 0.22), (0.60, 0.02), (0.55, -0.18),
          (0.42, -0.34), (0.22, -0.44), (0.0, -0.47)]
    head = smooth_poly(mirror_pts(hr[::-1])[:-1] + hr[:-1], n=8)
    lines = [LineString([(0, 0.7), (0, 0.18)]), LineString([(0, -0.2), (0, -0.7)])]
    for sg in (-1, 1):
        if detail == "full":
            lines += [
                polyline(M(sg, [(0.0, 0.18), (0.10, 0.30), (0.18, 0.50)]), smooth=True),
                polyline(M(sg, [(0.36, 0.08), (0.50, 0.02), (0.58, -0.12), (0.62, -0.2)]), smooth=True),
                polyline(M(sg, [(0.10, 0.02), (0.15, -0.12), (0.27, -0.20), (0.26, -0.31), (0.12, -0.35),
                                (-0.02, -0.33)]), smooth=True),
            ]
        else:
            lines += [polyline(M(sg, [(0.10, -0.04), (0.16, -0.16), (0.30, -0.22), (0.28, -0.34), (0.12, -0.38),
                                      (-0.02, -0.36)]), smooth=True)]

    def face_col(p, i):
        x, y = p.centroid.x, p.centroid.y
        if abs(x) < 0.12 and y > 0.2:
            c = "w0"
        elif abs(x) < 0.26 and y < -0.05:
            c = "w0" if y > -0.36 else "w2"
        elif abs(x) > 0.38 and y < 0.05:
            c = "w1"
        else:
            c = "w0"
        return shade(c) if x > 0 else c
    L.region("face", head, face_col, lines, order=order0)

    for sg in (1, -1):

        if detail == "full":
            inner = smooth_poly(M(sg, [(-0.54, 0.34), (-0.56, 0.74), (-0.32, 0.47)]), n=3)
            L.region("ear_in", inner, inner_col, order=order0 - 0.15)
        if ruff is True:
            ear = Polygon(M(sg, [(-0.12, 0.44), (-0.36, 0.66), (-0.60, 0.92), (-0.68, 0.56), (-0.64, 0.14),
                                 (-0.3, 0.2)]))
        else:
            ear = Polygon(M(sg, [(-0.12, 0.44), (-0.36, 0.66), (-0.60, 0.92), (-0.65, 0.56), (-0.56, 0.24),
                                 (-0.3, 0.2)]))
        if detail == "full":
            L.region("ear", ear, "w1" if sg > 0 else shade("w1"), order=order0 - 0.2)
        else:   # two stones: white outer half, rose inner half
            axis = LineString(M(sg, [(-0.62, 0.98), (-0.40, 0.30)]))
            L.region("ear", ear, lambda p, i, sg=sg: inner_col if sg * p.centroid.x > -0.47 else "w1",
                     [axis], order=order0 - 0.2)

    if ruff == "cheeks":
        seq = [(-0.54, 0.26),
               (-0.76, 0.04), (-0.60, -0.06),
               (-0.68, -0.38), (-0.44, -0.34),
               (-0.30, -0.64), (-0.14, -0.46),
               (0.0, -0.62)]
        band, rl = flame_band(seq, lambda v: (v[0] * 0.55, v[1] * 0.55), [(0.54, 0.26), (0, 0.3)],
                              bulge_out=-0.30, bulge_in=0.10, bulge_cut=-0.10)
        L.region("ruff", band, lambda p, i: ("w1" if p.centroid.x < 0 else "w2") if i % 2 else
                 ("w0" if p.centroid.x < 0 else "w1"), rl, order=order0 - 0.3)
    elif ruff:
        seq = [(-0.58, 0.30),
               (-0.80, 0.12), (-0.64, 0.02),
               (-0.84, -0.22), (-0.62, -0.22),
               (-0.74, -0.52), (-0.48, -0.42),
               (-0.50, -0.74), (-0.26, -0.54),
               (-0.22, -0.84), (0.0, -0.66)]
        band, rl = flame_band(seq, lambda v: (v[0] * 0.55, v[1] * 0.55 - 0.04), [(0.58, 0.30), (0, 0.3)],
                              bulge_out=-0.20, bulge_in=0.06, bulge_cut=-0.10)

        def ruff_col(p, i):
            x, y = p.centroid.x, p.centroid.y
            k = int(round((math.degrees(math.atan2(y, abs(x))) + 90) / 22))
            c = ["w1", "w0", "w1", "w0", "w2", "w0"][k % 6]
            return shade(c) if x > 0.05 else c
        L.region("ruff", band, ruff_col, rl, order=order0 - 0.3)
    return head


# =====================================================================================
# 1. Madalyon - bold faceted front head, ink field, cream ring, red accents
# =====================================================================================
def concept_1():
    d = Design("1", "Madalyon", circle((0, 0), 0.97, 180), gap=0.026, corner=0.010)
    b = Builder(d)
    L = Local(b, 0.92, (0.0, 0.03))

    ec = (-0.255, 0.10)
    eL = almond(ec, 0.185, 0.12, ang=-16, pinch=0.6, top=1.1, bot=0.9)
    pL = ellipse((ec[0] + 0.012, ec[1] + 0.05), 0.026, 0.125, ang=-3)
    L.region("pupil", pL.intersection(eL), "pupil", order=0.95)
    L.region("pupil", sym(pL).intersection(sym(eL)), "pupil", order=0.95)
    L.region("eye", eL, "blue", gap=1.8, order=0.9)
    L.region("eye", sym(eL), "amber", gap=1.8, order=0.9)
    L.region("nose", Polygon([(-0.085, -0.075), (0.085, -0.075), (0.0, -0.175)]), "red", order=0.85)

    hp = [(0, 0.48), (-0.20, 0.46), (-0.44, 0.36), (-0.58, 0.20), (-0.62, 0.0), (-0.54, -0.20),
          (-0.34, -0.36), (0, -0.43)]
    head = Polygon(hp[:-1] + [(-x, y) for x, y in hp[::-1]][1:-1])
    lines = [LineString([(0, 0.6), (0, 0.18)]), LineString([(0, -0.16), (0, -0.6)])]
    for sg in (-1, 1):
        lines += [LineString(M(sg, q)) for q in [
            [(0.22, 0.46), (0.11, 0.25), (0.0, 0.17)],
            [(0.58, 0.20), (0.40, 0.16)],
            [(0.62, 0.0), (0.36, -0.02)],
            [(0.08, 0.02), (0.20, -0.20), (0.34, -0.36)],
            [(0.36, -0.02), (0.20, -0.20)],
        ]]

    def facet(p, i):
        x, y = p.centroid.x, p.centroid.y
        lit = -0.6 * x + 0.75 * y
        if y < -0.25:
            return "w2" if x > 0 else "w1"
        return "w0" if lit > 0.12 else ("w1" if lit > -0.12 else "w2")
    L.region("face", head, facet, lines, order=0.6)

    for sg in (1, -1):
        ear = Polygon(M(sg, [(-0.12, 0.46), (-0.56, 0.94), (-0.66, 0.20), (-0.4, 0.3)]))
        L.region("ear", ear, lambda p, i, sg=sg: "rose" if sg * p.centroid.x > -0.44 else ("w1" if sg > 0 else "w2"),
                 [LineString(M(sg, [(-0.56, 0.94), (-0.40, 0.30)]))], order=0.4)

    seq = [(-0.60, 0.30),
           (-0.86, 0.12), (-0.64, 0.02),
           (-0.86, -0.24), (-0.58, -0.22),
           (-0.70, -0.52), (-0.42, -0.40),
           (-0.42, -0.74), (-0.20, -0.50),
           (0.0, -0.80)]
    ruff, rl = flame_band(seq, lambda v: (v[0] * 0.5, v[1] * 0.5), [(0.6, 0.3), (0, 0.2)],
                          bulge_out=0.0, bulge_in=0.0, bulge_cut=0.0)

    def rc(p, i):
        x, y = p.centroid.x, p.centroid.y
        k = int((math.atan2(y, abs(x)) + 2) * 2.6)
        return ["w1", "w0", "w2", "w0"][k % 4] if x < 0 else ["w2", "w1", "w3", "w1"][k % 4]
    L.region("ruff", ruff, rc, rl, order=0.3)

    ring = circle((0, 0), 0.92, 180).difference(circle((0, 0), 0.80, 180))
    b.region("ring", ring, lambda p, i: "paper" if i % 2 == 0 else "paper2",
             rays((0, 0), [i * 360 / 20 + 4.5 for i in range(20)]), order=0.0)
    b.region("bg", circle((0, 0), 0.80, 180), tone(["ink", "ink2"]),
             rays((0, 0), [i * 30 + 15 for i in range(12)]), order=0.1)
    return d


# =====================================================================================
# 2. Oturan - sitting, front view, plumed tail wrapped round the paws, in an arched niche
# =====================================================================================
def arch(w, y0, y1):
    return unary_union([box(-w, y0, w, y1), circle((0, y1), w, 160)]).buffer(0)


def concept_2():
    plate = arch(0.80, -0.96, 0.16)
    d = Design("2", "Oturan", plate, gap=0.022, corner=0.009)
    b = Builder(d)

    head = Local(b, 0.46, (0.0, 0.36))
    angora_head(head, detail="small", ruff="cheeks", eye_gap=1.0, order0=0.7)

    # chest: a wide bib of fur ending in four soft tufts over the legs
    seq = [(-0.27, 0.22),
           (-0.32, 0.0), (-0.22, -0.02),
           (-0.21, -0.20), (-0.10, -0.10),
           (-0.04, -0.24), (0.04, -0.10)]
    seq = seq[:-1] + [(0.0, -0.10)]
    bib, bl = flame_band(seq, lambda v: (v[0] * 0.6, 0.10), [(0.27, 0.22), (0, 0.26)],
                         bulge_out=-0.22, bulge_in=0.10, bulge_cut=-0.06)
    b.region("bib", bib, lambda p, i: "w0" if p.centroid.x <= 0.0 else "w1", bl, order=0.6)

    # tail: a plume wrapped round the front of the paws, tip curling up on the left
    tail, tc = plume([(0.44, -0.66), (0.36, -0.82), (0.06, -0.86), (-0.30, -0.82), (-0.52, -0.70),
                      (-0.60, -0.50)],
                     [0.04, 0.06, 0.07, 0.075, 0.075, 0.05], [0.28, 0.44, 0.60, 0.75, 0.88], slant=0.04, tip=0.045)
    b.region("tail", tail, tone(["w2", "w0", "w1", "w0", "w1", "w0"]), tc, order=0.58)

    # paws resting just behind the tail
    for sg in (-1, 1):
        b.region("paw", ellipse((sg * 0.085, -0.70), 0.08, 0.05), "w0" if sg < 0 else "w1", order=0.55)

    # body: one soft pear, drawn by haunch and leg lines
    body_l = [(-0.17, 0.27), (-0.28, 0.14), (-0.31, -0.06), (-0.36, -0.30), (-0.46, -0.52),
              (-0.48, -0.72), (-0.40, -0.84), (0.0, -0.85)]
    body = smooth_poly(body_l[:-1] + [(-x, y) for x, y in body_l[::-1]][1:-1], n=8)
    bc = [LineString([(0, -0.2), (0, -0.9)])]
    for sg in (-1, 1):
        bc += [polyline(M(sg, [(0.15, -0.86), (0.16, -0.60), (0.15, -0.36), (0.12, -0.1)]), smooth=True),
               polyline(M(sg, [(0.155, -0.70), (0.26, -0.52), (0.38, -0.40), (0.56, -0.40)]), smooth=True)]

    def body_col(p, i):
        x, y = p.centroid.x, p.centroid.y
        if abs(x) < 0.16:
            return "w0" if x < 0 else "w1"
        if y < -0.45:
            return "w1" if x < 0 else "w2"
        return "w2" if x < 0 else "w3"
    b.region("body", body, body_col, bc, order=0.45)

    # niche: gold border row + ink field
    inner = arch(0.70, -0.86, 0.16)
    border = plate.buffer(-0.025).difference(inner)
    bord_cuts = rays((0, 0.16), [i * 180 / 9 for i in range(1, 9)]) + [
        LineString([(-1, y), (-0.6, y)]) for y in (-0.15, -0.55)] + [
        LineString([(1, y), (0.6, y)]) for y in (-0.15, -0.55)] + [
        LineString([(x, -1), (x, -0.8)]) for x in (-0.4, 0.0, 0.4)] + [
        LineString([(-1, 0.16), (-0.6, 0.16)]), LineString([(1, 0.16), (0.6, 0.16)])]
    b.region("border", border, tone(["gold", "gold2"]), bord_cuts, order=0.0)
    fcuts = rays((0, 0.16), [30, 70, 110, 150]) + [LineString([(-1, -0.30), (1, -0.30)]),
                                                  LineString([(-0.7, 0.16), (0.7, 0.16)])]
    b.region("bg", inner, tone(["ink", "ink2", "ink"]), fcuts, order=0.1)
    return d


# =====================================================================================
# 3. Yalın - ultra minimal odd-eyed face on a red square tile
# =====================================================================================
def concept_3():
    d = Design("3", "Yalın", rounded_square(0.94, 0.24), gap=0.028, corner=0.014)
    b = Builder(d)
    angora_head(Local(b, 1.0, (0.0, 0.0)), detail="full", ruff=True)
    bg_lines = [LineString([(0, 0.4), (0, 1.0)]),
                LineString([(-0.60, 0.92), (-1.0, 0.70)]), LineString([(0.60, 0.92), (1.0, 0.70)]),
                LineString([(-0.80, 0.12), (-1.0, 0.12)]), LineString([(0.80, 0.12), (1.0, 0.12)]),
                LineString([(-0.74, -0.52), (-1.0, -0.52)]), LineString([(0.74, -0.52), (1.0, -0.52)]),
                LineString([(0, -0.6), (0, -1.0)])]
    b.region("bg", rounded_square(0.86, 0.18), lambda p, i: ["red", "red2", "red3"][i % 3], bg_lines, order=0.0)
    return d


# =====================================================================================
# 4. Bizans - white Angora bust on a gold ground laid in rows that follow the figure
# =====================================================================================
def concept_4():
    d = Design("4", "Bizans", circle((0, 0), 0.97, 180), gap=0.022, corner=0.009)
    b = Builder(d)
    R_IN = 0.84
    field = circle((0, 0), R_IN, 180)

    L = Local(b, 0.66, (0.0, 0.16))
    angora_head(L, detail="full", ruff="cheeks", order0=0.7)

    # bust: shoulders and long chest fur flowing down out of the roundel
    seq = [(-0.30, 0.12),
           (-0.52, 0.0), (-0.38, -0.06),
           (-0.62, -0.26), (-0.44, -0.30),
           (-0.68, -0.58), (-0.48, -0.58),
           (-0.70, -0.92), (-0.50, -0.86),
           (-0.56, -1.2)]
    bust, bl = flame_band(seq, lambda v: (v[0] * 0.45, v[1] * 0.6 + 0.02), [], bulge_out=-0.22,
                          bulge_in=0.10, bulge_cut=-0.10, mirror=True)
    # long locks of fur: S-curves from the chin down
    for sg in (-1, 1):
        bl += [polyline(M(sg, [(0.08, -0.16), (0.18, -0.40), (0.24, -0.70), (0.30, -1.0)]), smooth=True)]
    bl.append(LineString([(0, -0.2), (0, -1.2)]))

    def bust_col(p, i):
        x, y = p.centroid.x, p.centroid.y
        c = ["w0", "w1", "w0", "w2", "w1"][i % 5]
        if abs(x) < 0.1:
            c = "w0"
        return {"w0": "w1", "w1": "w2", "w2": "w2"}[c] if x > 0.03 else c
    b.region("bust", bust, bust_col, bl, order=0.5)

    # gold rows following the silhouette (andamento), brick-staggered, equal stone lengths
    cat = b.taken.buffer(0)
    guide = cat.buffer(0.16, quad_segs=16).buffer(-0.16, quad_segs=16)    # notches closed: calm rows
    step, length = 0.12, 0.22
    k = 0
    r0 = 0.0
    while r0 < 1.2:
        r1 = r0 + step
        inner_ = cat if r0 == 0 else guide.buffer(r0, quad_segs=16)
        band = guide.buffer(r1, quad_segs=16).difference(inner_).intersection(field)
        if not band.is_empty:
            mid = guide.buffer((r0 + r1) / 2, quad_segs=16)
            rings = [mid.exterior] if hasattr(mid, "exterior") else [g.exterior for g in mid.geoms]
            cuts = []
            for ring in rings:
                n = max(4, int(round(ring.length / length)))
                off = 0.5 if k % 2 else 0.0
                for j in range(n):
                    t = (j + off) / n
                    p = ring.interpolate(t, normalized=True)
                    q = ring.interpolate((t + 0.002) % 1, normalized=True)
                    dx, dy = q.x - p.x, q.y - p.y
                    nn = math.hypot(dx, dy) or 1
                    nx, ny = -dy / nn * step, dx / nn * step
                    cuts.append(LineString([(p.x - nx, p.y - ny), (p.x + nx, p.y + ny)]))
            b.region(f"gold{k}", band, lambda p, i: ["gold", "gold2"][i % 2], cuts, order=0.1 + 0.02 * k)
        r0 = r1
        k += 1

    ring = circle((0, 0), 0.935, 180).difference(field)
    b.region("ring", ring, lambda p, i: "red" if i % 2 == 0 else "red2",
             rays((0, 0), [i * 360 / 22 + 3 for i in range(22)]), order=0.0)
    return d


# =====================================================================================
# 5. Konstrüktivist - cat in profile, head turned, red sun disc, black diagonal, cream ground
# =====================================================================================
def concept_5():
    d = Design("5", "Konstrüktivist", rounded_square(0.94, 0.07), gap=0.022, corner=0.009)
    b = Builder(d)
    field = rounded_square(0.875, 0.04)

    head = Local(b, 0.42, (-0.16, 0.30))
    angora_head(head, detail="small", ruff="cheeks", eye_gap=1.0, order0=0.75)

    # tail: plume rising along the right edge, tufts pointing to the tip
    tail, tc = plume([(0.34, -0.78), (0.56, -0.70), (0.66, -0.44), (0.64, -0.14), (0.54, 0.08), (0.42, 0.16)],
                     [0.035, 0.055, 0.072, 0.08, 0.072, 0.04], [0.26, 0.44, 0.60, 0.74, 0.87], slant=0.05,
                     tip=0.045)
    b.region("tail", tail, tone(["w2", "w1", "w0", "w1", "w0", "w1"]), tc, order=0.45)

    # body in profile, facing left; the chest ruff is drawn as tufts
    body = smooth_poly([(-0.20, 0.14), (0.0, 0.16), (0.14, 0.04), (0.26, -0.16), (0.38, -0.40), (0.42, -0.64),
                        (0.36, -0.82), (0.0, -0.83), (-0.36, -0.83), (-0.44, -0.79), (-0.38, -0.70),
                        (-0.36, -0.44), (-0.40, -0.18), (-0.36, 0.04)], n=8)
    bc = [polyline([(-0.21, -0.30), (-0.22, -0.60), (-0.23, -0.9)], smooth=True),
          polyline([(-0.12, -0.9), (-0.04, -0.62), (0.08, -0.40), (0.24, -0.28), (0.40, -0.24)], smooth=True),
          polyline([(0.08, -0.40), (0.14, -0.10), (0.10, 0.3)], smooth=True),
          polyline([(-0.5, -0.06), (-0.34, -0.16), (-0.20, -0.30), (0.0, -0.36), (0.08, -0.40)], smooth=True),
          polyline([(0.30, -0.6), (0.6, -0.6)], smooth=False)]
    b.region("body", body, lambda p, i: {0: "w1", 1: "w0", 2: "w2", 3: "w1", 4: "w0", 5: "w2"}[i % 6], bc,
             order=0.4)

    # red sun disc behind the head
    sc, sr = (0.12, 0.40), 0.48
    sun_cuts = [LineString([(-1, sc[1] + dy), (1, sc[1] + dy)]) for dy in (-0.27, -0.12, 0.03, 0.18, 0.31)]
    sun_cuts += [LineString([(sc[0] + dx, -1), (sc[0] + dx, 1)]) for dx in (0.0,)]
    b.region("sun", circle(sc, sr, 160), lambda p, i: ["red", "red2", "red3"][i % 3], sun_cuts, order=0.2)

    # black diagonal band (a constructivist beam) and a thin red counter-diagonal
    k = 0.5
    hw = 0.23
    band = Polygon([(-1.2, -0.28 + k * -1.2 - hw), (1.2, -0.28 + k * 1.2 - hw),
                    (1.2, -0.28 + k * 1.2 + hw), (-1.2, -0.28 + k * -1.2 + hw)])
    bcut = [LineString([(x + 0.12, -0.25 + k * x - 0.5), (x - 0.12, -0.25 + k * x + 0.5)])
            for x in (-0.62, -0.18, 0.26, 0.66)]
    b.region("band", band, tone(["ink", "ink2"]), bcut, order=0.1)

    # cream ground: a few large slabs
    gl = [LineString([(-1, 0.62), (1, 0.62)]), LineString([(0.66, 0.62), (0.66, 1)]),
          LineString([(-0.2, 0.62), (-0.2, 1)]), LineString([(-1, -0.12), (-0.5, -0.12)]),
          LineString([(0.55, -1), (0.55, -0.5)]), LineString([(0.55, -0.5), (1, -0.5)]),
          LineString([(-0.1, -1), (-0.1, -0.84)])]
    b.region("ground", field, tone(["paper2", "paper3", "paper2", "paper"]), gl, order=0.0)
    return d


CONCEPTS = {"1": concept_1, "2": concept_2, "3": concept_3, "4": concept_4, "5": concept_5}

if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    which = sys.argv[2:] or list(CONCEPTS)
    for k in which:
        d = CONCEPTS[k]()
        n = export(d, os.path.join(out, f"kedi_{k}.json"))
        preview(d, os.path.join(out, f"kedi_{k}_onizleme.png"))
        print(k, n, "stones")
