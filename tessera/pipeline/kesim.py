"""Cut-first stones (R8): the picture is drawn by the CUT of the stones.

Colour areas come first (smooth, about 0.2 mm/px), stones second:
- every stone is one colour; a colour boundary is always a cut
- a small shape (eye, lip, nostril, window) stays one custom-cut stone
- large areas are divided by the chosen texture (``layout``):
      "mozaik"  rows that follow the contours      "kirik"  irregular broken stones
      "blok"    staggered blocks                    "iri"    few large pieces
- ray areas become planks with a constant angle, so they grow away from the centre
- AMS mode only: small shapes are printed INTO the stones, so a stone may hold several colours

build(...) -> dict(polys, colour, stone, zone, label, lab, res). Polygons are mm, x right, y DOWN, already
shrunk by half the joint. The same polygons are used for preview, 3D, template and export.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import shapely
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

from ..model import Vocab

LAYOUTS = {"mozaik": "Mozaik", "kirik": "Kırık taş", "blok": "Blok", "iri": "İri parça"}
MIN_W = 1.2          # printable feature width, mm (3 lines of a 0.4 nozzle)
MIN_A = 5.0          # smallest loose stone, mm2
FEATURE_A, FEATURE_W, FEATURE_L = 150.0, 14.0, 26.0   # up to this a shape stays one stone (mm2, mm, mm)


def resolution(width_mm: float) -> float:
    return max(0.2, width_mm / 2400.0)


# ------------------------------------------------------------------ colour areas
def _lab(rgb):
    from .palette import rgb_to_lab
    return np.asarray(rgb_to_lab(np.asarray(rgb, np.uint8).reshape(-1, 3)), float).reshape(-1, 3)


def image_colours(img: np.ndarray, pal: np.ndarray):
    """The picture's own colours and the palette colour each one becomes (distinct when possible)."""
    from . import renk
    found = renk.find(img)
    if len(found) > len(pal):
        found = renk.find(img, len(pal))
    rgb = np.array([c["rgb"] for c in found])
    cost = np.linalg.norm(_lab(rgb)[:, None] - _lab(pal)[None], axis=2)
    to = cost.argmin(1)
    if len(found) <= len(pal):
        r, c = linear_sum_assignment(cost)
        to = np.empty(len(found), int)
        to[r] = c
    return found, to


def drop_specks(lab: np.ndarray, n: int, min_px: float) -> np.ndarray:
    """Areas under min_px take the colour that surrounds them most."""
    H, W = lab.shape
    for c in range(n):
        cc, k = ndi.label(lab == c)
        if not k:
            continue
        sz = np.bincount(cc.ravel())
        obj = ndi.find_objects(cc)
        for i in np.flatnonzero(sz[1:] < min_px) + 1:
            sl = obj[i - 1]
            y0, y1 = max(0, sl[0].start - 1), min(H, sl[0].stop + 1)
            x0, x1 = max(0, sl[1].start - 1), min(W, sl[1].stop + 1)
            m = cc[y0:y1, x0:x1] == i
            ring = ndi.binary_dilation(m) & ~m
            v = lab[y0:y1, x0:x1]
            if ring.any():
                v[m] = np.bincount(v[ring]).argmax()
    return lab


def widen_thin(lab: np.ndarray, order, res: float) -> np.ndarray:
    """Lines thinner than MIN_W are widened to about MIN_W (taken from the neighbours)."""
    w = max(3, int(round(MIN_W / res)) | 1)
    out = lab.copy()
    for c in order:
        M = (lab == c).view(np.uint8)
        core = ndi.maximum_filter(ndi.minimum_filter(M, size=w), size=w + 2)
        thin = (M > 0) & (core == 0)
        cc, k = ndi.label(thin, np.ones((3, 3)))
        if not k:
            continue
        keep = np.zeros(k + 1, bool)
        for i, sl in enumerate(ndi.find_objects(cc), 1):
            keep[i] = math.hypot(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) * res > 2.5
        grow = ndi.maximum_filter(keep[cc].view(np.uint8), size=3) > 0
        out[grow & (out != c)] = c
    return out


# ------------------------------------------------------------------ dividing one colour area
def _lloyd(pts, seeds, it):
    for _ in range(it):
        a = cKDTree(seeds).query(pts)[1]
        n = np.bincount(a, minlength=len(seeds)).astype(float)
        ok = n > 0
        s = np.stack([np.bincount(a, pts[:, i], len(seeds)) for i in (0, 1)], 1)
        seeds = seeds.copy()
        seeds[ok] = s[ok] / n[ok, None]
    return seeds


def _poisson(pts, r, rng):
    if not len(pts):
        return pts
    left = np.ones(len(pts), bool)
    tree = cKDTree(pts)
    out = []
    for i in rng.permutation(len(pts)):
        if left[i]:
            out.append(i)
            left[tree.query_ball_point(pts[i], r)] = False
    return pts[out]


def _sub(pts, n, rng):
    return pts if len(pts) < n else pts[rng.choice(len(pts), n, replace=False)]


def lay_rows(mask, s0, res, rng):
    """Rows of stones parallel to the contour; cuts cross the rows squarely."""
    d = (ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] * res).astype(np.float32)
    bounds, t, b, top = [0.0], s0, 0.0, float(d.max())
    while b < top:
        b += t
        bounds.append(b)
        t = min(t * 1.06, s0 * 2)
    row = np.where(mask, np.searchsorted(bounds, d, side="left") - 1, -1).astype(np.int16)
    for k in range(1, len(bounds) - 1):          # inner contours are rounded off: no spiky cores
        th = bounds[k + 1] - bounds[k]
        inner = row >= k
        box = ndi.find_objects(inner.view(np.uint8).astype(np.int8))
        if not box or box[0] is None:
            break
        sl = box[0]
        core = d[sl] >= bounds[k] + 0.5 * th     # eroded set, read straight from the distance field
        if core.any():
            keep = ndi.distance_transform_edt(~np.pad(core, 1))[1:-1, 1:-1] * res <= 0.5 * th
            keep &= inner[sl]
            cc, n = ndi.label(keep)
            small = np.bincount(cc.ravel()) * res ** 2 < th * th
            small[0] = False
            keep &= ~small[cc]
        else:
            keep = np.zeros(inner[sl].shape, bool)
        row[sl][inner[sl] & ~keep] = k - 1
    out = np.full(mask.shape, -1, np.int32)
    nxt = 0
    for k in range(int(row.max()) + 1):
        rk = row == k
        box = ndi.find_objects(rk.view(np.uint8).astype(np.int8))
        if not box or box[0] is None:
            continue
        sk = box[0]
        dk = ndi.distance_transform_edt(np.pad(rk[sk], 1))[1:-1, 1:-1] * res
        th = bounds[min(k + 1, len(bounds) - 1)] - bounds[k]
        ridge_k = dk >= 0.55 * ndi.maximum_filter(dk, size=int(max(3, th / res)) | 1)
        cc, n = ndi.label(rk[sk], np.ones((3, 3)))
        for i, sl in enumerate(ndi.find_objects(cc), 1):
            m = cc[sl] == i
            pts = np.argwhere(m).astype(float)
            dd = dk[sl]
            wl = 2 * float(np.percentile(dd[m], 90))
            L = max(th, min(wl, 2 * th)) * 1.15
            o = out[sk][sl]
            if max(1, round(len(pts) * res ** 2 / (L * max(min(wl, 1.8 * th), 0.6 * th)))) == 1:
                o[m] = nxt
                nxt += 1
                continue
            sub = _sub(pts, 7000, rng)
            line = np.argwhere(m & ridge_k[sl] & (dd >= 0.3 * dd[m].max())).astype(float)
            if len(line) > 5:
                seeds = _poisson(line, L * 0.72 / res, rng)
                tree = cKDTree(line)
                for _ in range(8):               # seeds stay on the middle line of the row
                    seeds = line[tree.query(_lloyd(sub, seeds, 1))[1]]
                seeds = np.unique(seeds, axis=0)
            else:
                seeds = _lloyd(sub, _poisson(sub, L * 0.72 / res, rng), 6)
            o[m] = cKDTree(seeds).query(pts)[1] + nxt
            nxt += len(seeds)
    return out, nxt


def lay_broken(mask, s0, res, rng, relax):
    pts = np.argwhere(mask).astype(float)
    sub = _sub(pts, 60000, rng)
    seeds = _lloyd(sub, _poisson(sub, s0 * 0.8 / res, rng), relax)
    out = np.full(mask.shape, -1, np.int32)
    out[mask] = cKDTree(seeds).query(pts, workers=-1)[1]
    return out, len(seeds)


def lay_blocks(mask, s0, res, oy, ox):
    ys, xs = np.nonzero(mask)
    h = s0 / res
    r = ((ys + oy) / h).astype(int)
    c = ((xs + ox) / (1.7 * h) + (r % 2) * 0.5).astype(int)
    out = np.full(mask.shape, -1, np.int32)
    inv = np.unique(r.astype(np.int64) * 100000 + c, return_inverse=True)[1]
    out[ys, xs] = inv
    return out, int(inv.max()) + 1


def split_wedges(m, oy, ox, centre, res):
    """Separate rays that touch each other near the centre."""
    ys, xs = np.nonzero(m)
    r = np.zeros(m.shape, np.float32)
    r[ys, xs] = np.hypot(ys + oy - centre[1], xs + ox - centre[0])
    best, keep = 1, None
    for r0 in np.linspace(0, np.percentile(r[ys, xs], 85), 12):
        cc, _ = ndi.label(m & (r >= r0))
        big = np.flatnonzero(np.bincount(cc.ravel())[1:] * res ** 2 > 120) + 1
        if len(big) > best:
            best, keep = len(big), np.where(np.isin(cc, big), cc, 0)
    if keep is None:
        return [m]
    idx = ndi.distance_transform_edt(keep == 0, return_distances=False, return_indices=True)
    near = np.where(m, keep[idx[0], idx[1]], 0)
    return [near == v for v in np.unique(near[near > 0])]


def lay_planks(mask, oy, ox, centre, scale=1.0):
    """Radial planks: constant angle across, length grows with the distance from the centre."""
    ys, xs = np.nonzero(mask)
    dy, dx = ys + oy - centre[1], xs + ox - centre[0]
    th = np.arctan2(dy, dx)
    m0 = math.atan2(np.sin(th).mean(), np.cos(th).mean())
    th = (th - m0 + math.pi) % (2 * math.pi) - math.pi
    r = np.hypot(dx, dy)
    far = r >= np.median(r)                      # the wedge angle is read away from the crowded centre
    lo, hi = np.percentile(th[far], [2, 98])
    span = max(hi - lo, 1e-3)
    n = max(1, int(round(math.degrees(span) / (2.8 * scale))))
    strip = np.clip(((th - lo) / span * n).astype(int), 0, n - 1)
    rb = np.floor(np.log(np.maximum(r, 1)) / (span / n * 2.6) + (strip % 2) * 0.5 + 0.13).astype(int)
    inv = np.unique(strip.astype(np.int64) * 100000 + rb - rb.min(), return_inverse=True)[1]
    out = np.full(mask.shape, -1, np.int32)
    out[ys, xs] = inv
    return out, int(inv.max()) + 1


def _split_components(L):
    out = np.zeros_like(L)
    nxt = 0
    for i, sl in enumerate(ndi.find_objects(L + 1)):
        if sl is None:
            continue
        m = L[sl] == i
        cc, k = ndi.label(m)
        out[sl][m] = cc[m] - 1 + nxt
        nxt += k
    return out


def _merge_small(L, colour, min_px, passes=3):
    """Fragments join the same-colour neighbour with the longest shared border."""
    for _ in range(passes):
        n = int(L.max()) + 1
        sz = np.bincount(L.ravel(), minlength=n)
        pa, pb = [], []
        for s1, s2 in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                       ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
            k = (L[s1] != L[s2]) & (colour[s1] == colour[s2])
            pa.append(L[s1][k])
            pb.append(L[s2][k])
        a = np.concatenate(pa + pb)
        b = np.concatenate(pb + pa)
        k = sz[a] < min_px
        a, b = a[k], b[k]
        if not len(a):
            break
        pair, cnt = np.unique(a.astype(np.int64) * n + b, return_counts=True)
        x, y = pair // n, pair % n
        o = np.lexsort((cnt, x))
        x, y = x[o], y[o]
        last = np.r_[x[1:] != x[:-1], True]
        mp = np.arange(n)
        for s, t in zip(x[last], y[last]):
            if sz[t] > sz[s] or (sz[t] == sz[s] and t > s):
                mp[s] = t
        for _ in range(4):
            mp = mp[mp]
        L = np.unique(mp[L], return_inverse=True)[1].reshape(L.shape).astype(np.int32)
    return L


# ------------------------------------------------------------------ outlines
_SEG = {1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)], 5: [(3, 0), (1, 2)], 6: [(0, 2)], 7: [(3, 2)],
        8: [(2, 3)], 9: [(2, 0)], 10: [(0, 1), (2, 3)], 11: [(2, 1)], 12: [(1, 3)], 13: [(1, 0)], 14: [(0, 3)]}


def outline(m, oy, ox, res, sig=1.2):
    """Smooth polygon (mm, y down) of a mask: marching squares on a lightly blurred mask."""
    f = ndi.gaussian_filter(np.pad(m.astype(np.float32), 5), sig, truncate=3.0) - 0.5
    ins = (f > 0).view(np.uint8)
    case = ins[:-1, :-1] + 2 * ins[:-1, 1:] + 4 * ins[1:, 1:] + 8 * ins[1:, :-1]
    W = f.shape[1]
    nx = {}
    for cv, segs in _SEG.items():
        ys, xs = np.nonzero(case == cv)
        if not len(ys):
            continue
        base = ys * W + xs                       # edge id: 2*pixel (horizontal) or 2*pixel+1 (vertical)
        e = {0: 2 * base, 1: 2 * (base + 1) + 1, 2: 2 * (base + W), 3: 2 * base + 1}
        for a, b in segs:
            nx.update(zip(e[a].tolist(), e[b].tolist()))
    ff = f.ravel()

    def pt(e):
        p = e >> 1
        y, x = divmod(p, W)
        if e & 1:
            a, b = ff[p], ff[p + W]
            return x, y + a / (a - b)
        a, b = ff[p], ff[p + 1]
        return x + a / (a - b), y

    loops = []
    while nx:
        e = next(iter(nx))
        lp = []
        while e in nx:
            lp.append(pt(e))
            e = nx.pop(e)
        if len(lp) > 6:
            loops.append(np.asarray(lp) * res + ((ox - 4.5) * res, (oy - 4.5) * res))
    if not loops:
        return None
    ps = sorted((Polygon(q).buffer(0) for q in loops), key=lambda p: -p.area)
    P = ps[0]
    for h in ps[1:]:
        if h.area > 0.6:
            P = P.difference(h) if P.contains(h.representative_point()) else P.union(h)
    return P


def _parts(g):
    return [p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon"]


# ------------------------------------------------------------------ the whole thing
def build(img: np.ndarray, width_mm: float, pal: np.ndarray, *, layout="mozaik", zone=None, centre=None,
          gap=0.4, size=1.0, seed=7, ams=False, report=None, colours=None, to=None, zoom=1.0,
          labels=None) -> dict:
    """img: source picture (uint8, any resolution). zone/centre: Vocab map (any resolution) and ray
    centre as 0..1 fractions. colours/to: the picture's colours (renk.find) and the palette index of
    each; found here when missing. zoom: > 1 when img is only a part of the picture. labels: index map
    into colours when it is already known."""
    from . import renk
    report = report or (lambda f, m: None)
    rng = np.random.default_rng(seed)
    res = resolution(width_mm)
    W = int(round(width_mm / res))
    H = max(8, int(round(img.shape[0] * W / img.shape[1])))
    res = width_mm / W
    report(0.03, "Renkler ayrılıyor")
    if colours is None:
        colours, to = image_colours(img, pal)
    im = None
    if labels is not None:
        lab = np.asarray(labels)
    elif colours[0].get("yol") == "ton":
        from . import sade
        lab = sade.label(img, colours, width_mm)
    else:
        cw = int(min(W, np.clip(img.shape[1], 900 / zoom, 1500 / zoom)))
        ch = max(8, int(round(H * cw / W)))
        im = np.asarray(Image.fromarray(img).resize((cw, ch), Image.BICUBIC if cw >= img.shape[1] else Image.BOX))
        lab = renk.label(im, colours, cw * zoom)
    report(0.12, "Renkler ayrılıyor")
    lab = np.asarray(to)[renk.enlarge(lab, len(colours), (W, H))].astype(np.int16)
    del im
    n = len(pal)
    # the picture's own lightness (not the chosen paint) decides the order, so repainting moves no stone
    lum = np.zeros(n)
    for i in range(n):
        own = [c["rgb"] for c, t in zip(colours, to) if t == i]
        lum[i] = np.mean(own or [pal[i]], axis=0) @ np.array([.3, .59, .11])
    lab = drop_specks(lab, n, 0.8 / res ** 2)
    lab = widen_thin(lab, [int(i) for i in np.argsort(np.abs(lum - 128))], res)
    lab = drop_specks(lab, n, MIN_A / res ** 2)
    report(0.25, "Taşlar kesiliyor")

    z = np.zeros((H, W), np.int8)
    if zone is not None:
        z = np.asarray(Image.fromarray(zone.astype(np.uint8)).resize((W, H), Image.NEAREST)).astype(np.int8)
    c_px = (centre[0] * W, centre[1] * H) if centre is not None else None
    ray = (z == int(Vocab.RAYS)) if c_px is not None else np.zeros((H, W), bool)

    full = lab
    if ams:                                      # isolated small shapes are printed into the stones
        lab = drop_specks(lab.copy(), n, FEATURE_A / res ** 2)

    jobs = []
    for c in range(n):
        cc, k = ndi.label(lab == c)
        for i, sl in enumerate(ndi.find_objects(cc), 1):
            jobs.append((c, sl, cc[sl] == i, int(rng.integers(1 << 31))))

    def work(job):
        _, sl, m, sd = job
        r = np.random.default_rng(sd)
        area = int(m.sum()) * res ** 2
        dm = ndi.distance_transform_edt(np.pad(m, 1)).max() * res
        one = np.where(m, 0, -1).astype(np.int32)
        if area < FEATURE_A and 2 * dm < FEATURE_W and max(m.shape) * res < FEATURE_L:
            return one, 1
        if ray[sl][m].mean() > 0.5:
            out, nxt = np.full(m.shape, -1, np.int32), 0
            for w in split_wedges(m, sl[0].start, sl[1].start, c_px, res):
                o, q = lay_planks(w, sl[0].start, sl[1].start, c_px, size * (1.5 if layout == "iri" else 1.0))
                out[w] = o[w] + nxt
                nxt += q
            left = m & (out < 0)
            if left.any():
                out[left] = nxt
                nxt += 1
            return out, nxt
        s0 = float(np.clip(3.8 + 0.17 * dm, 4.6, 10.0)) * size
        if layout == "kirik":
            return lay_broken(m, s0 * 1.25, res, r, 2)
        if layout == "iri":
            return lay_broken(m, s0 * 2.4, res, r, 8)
        if layout == "blok":
            return lay_blocks(m, s0 * 1.15, res, sl[0].start, sl[1].start)
        if m.size > 1_200_000:                   # huge areas: rows at half resolution (joints only)
            half = m[::2, ::2]
            o, q = lay_rows(half, s0, res * 2, r)
            hi = np.where(half, 0, 1)
            idx = ndi.distance_transform_edt(hi, return_distances=False, return_indices=True)
            o = o[idx[0], idx[1]]
            up = np.repeat(np.repeat(o, 2, 0), 2, 1)[: m.shape[0], : m.shape[1]]
            return np.where(m, up, -1), q
        return lay_rows(m, s0, res, r)

    L = np.full((H, W), -1, np.int32)
    nxt = done = 0
    total = sum(j[2].size for j in jobs) or 1
    with ThreadPoolExecutor(6) as ex:
        for (c, sl, m, _), (o, q) in zip(jobs, ex.map(work, jobs)):
            L[sl][m] = o[m] + nxt
            nxt += q
            done += m.size
            report(0.25 + 0.40 * done / total, "Taşlar kesiliyor")
    L = _merge_small(_split_components(L), lab, 14.0 / res ** 2)
    report(0.72, "Taşların kenarı çiziliyor")

    # the joint around a small feature is taken from its big neighbour, so features keep their width
    sz = np.bincount(L.ravel()) * res ** 2
    small = (sz < 60)[L]
    if small.any():
        near = (ndi.maximum_filter(small.view(np.uint8), size=3) > 0) & ~small
        ys, xs = np.nonzero(near)
        Lg = L.copy()
        for oy, ox in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            yy, xx = np.clip(ys + oy, 0, H - 1), np.clip(xs + ox, 0, W - 1)
            hit = small[yy, xx]
            Lg[ys[hit], xs[hit]] = L[yy[hit], xx[hit]]
        L = Lg

    polys, pcol, pstone, pzone = [], [], [], []
    objs = ndi.find_objects(L + 1)
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        m = L[sl] == i
        oy, ox = sl[0].start, sl[1].start
        P = outline(m, oy, ox, res)
        if P is None:
            continue
        S = shapely.simplify(P.buffer(-gap / 2, quad_segs=3), 0.05)
        if S.is_empty:
            S = P.buffer(-gap / 5, quad_segs=3)
        zz = int(np.bincount(z[sl][m].clip(0)).argmax())
        cols = np.bincount(full[sl][m], minlength=n)
        if ams and (cols * res ** 2 > 2.0).sum() > 1:
            main = int(cols.argmax())
            rest = S
            for c in np.flatnonzero(cols * res ** 2 > 2.0):
                if c == main:
                    continue
                Q = outline(m & (full[sl] == c), oy, ox, res, 0.9)
                if Q is None:
                    continue
                Q = shapely.simplify(Q, 0.05).buffer(0).intersection(S)
                for g in _parts(Q):
                    if g.area > 1.0:
                        polys.append(g), pcol.append(int(c)), pstone.append(i), pzone.append(zz)
                        rest = rest.difference(g)
            S, col = rest, main
        else:
            col = int(cols.argmax())
        for g in _parts(S):
            if g.area >= 1.5:
                polys.append(g), pcol.append(col), pstone.append(i), pzone.append(zz)
        if i % 200 == 0:
            report(0.72 + 0.26 * i / len(objs), "Taşların kenarı çiziliyor")
    report(1.0, "Taşlar hazır")
    return {"polys": polys, "colour": np.array(pcol), "stone": np.array(pstone), "zone": np.array(pzone),
            "label": L, "lab": full, "res": res, "size": (W * res, H * res)}
