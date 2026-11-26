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

import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

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


THREADS = max(2, min(8, os.cpu_count() or 2))   # numpy, scipy and shapely let go of the GIL while they work


@contextmanager
def pool(n: int = THREADS):
    """A few worker threads; when the caller gives up (a newer draft) the jobs not begun are dropped."""
    ex = ThreadPoolExecutor(n)
    try:
        yield ex
    except BaseException:
        ex.shutdown(wait=True, cancel_futures=True)
        raise
    ex.shutdown(wait=True)


def big_first(ex, fn, items, size):
    """fn over items on the pool ex, the biggest begun first; the results come in the items' order."""
    fut = [None] * len(items)
    for i in sorted(range(len(items)), key=lambda i: -size(items[i])):
        fut[i] = ex.submit(fn, items[i])
    return (f.result() for f in fut)


def resolution(width_mm: float) -> float:
    return max(0.2, width_mm / 2400.0)


# ------------------------------------------------------------------ colour areas
def image_colours(img: np.ndarray, pal: np.ndarray):
    """The picture's own colours and the palette colour each one becomes (distinct when possible)."""
    from . import renk
    from .palette import rgb_to_lab
    found = renk.find(img)
    if len(found) > len(pal):
        found = renk.find(img, len(pal))
    a, b = (rgb_to_lab(np.asarray(v, np.uint8)) for v in ([c["rgb"] for c in found], pal))
    cost = np.linalg.norm(a[:, None] - b[None], axis=2)
    to = cost.argmin(1)
    if len(found) <= len(pal):
        r, c = linear_sum_assignment(cost)
        to = np.empty(len(found), int)
        to[r] = c
    return found, to


def drop_specks(lab: np.ndarray, n: int, min_px: float, out: int = -1) -> np.ndarray:
    """Areas under min_px take the colour that surrounds them most. out: label of what lies outside the
    artwork; nothing is given to it, only a speck with no other neighbour leaves."""
    H, W = lab.shape
    if not lab.flags.c_contiguous:
        return _drop_specks_one_by_one(lab, n, min_px, out)
    flat = lab.reshape(-1)
    for c in range(n):
        cc, k = ndi.label(lab == c)
        if not k:
            continue
        small = np.bincount(cc.ravel()) < min_px
        small[0] = False
        if not small.any():
            continue
        # all specks of this colour at once: they never touch each other, so the order does not matter
        ccf = cc.reshape(-1)
        P = np.flatnonzero(small[ccf])
        ids = ccf[P].astype(np.int64)
        py, px = np.divmod(P, W)
        pairs = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):          # the ring: 4-neighbours outside the speck
            yy, xx = py + dy, px + dx
            ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            q = yy[ok] * W + xx[ok]
            i = ids[ok]
            ok = (ccf[q] != i) & (flat[q] != out)
            pairs.append(i[ok] * flat.size + q[ok])
        pq = np.unique(np.concatenate(pairs))                       # a ring pixel counts once per speck
        gi, gv = pq // flat.size, flat[pq % flat.size].astype(np.int64)
        M = int(gv.max()) + 1 if len(gv) else 1
        key, cnt = np.unique(gi * M + gv, return_counts=True)
        gi, gv = key // M, key % M
        o = np.lexsort((gv, -cnt, gi))                              # most votes, the lowest colour on a tie
        first = o[np.r_[True, gi[o][1:] != gi[o][:-1]]] if len(o) else o
        to = np.full(k + 1, -1, np.int64)
        to[gi[first]] = gv[first]
        new = to[ids]
        has = new >= 0
        flat[P[has]] = new[has]
        if out >= 0:
            flat[P[~has]] = out
    return lab


def _drop_specks_one_by_one(lab, n, min_px, out):
    """The same, speck by speck (for a map that is a view into a bigger one)."""
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
            v = lab[y0:y1, x0:x1]
            ring = ndi.binary_dilation(m) & ~m & (v != out)
            if ring.any():
                v[m] = np.bincount(v[ring]).argmax()
            elif out >= 0:
                v[m] = out
    return lab


def cut_slivers(lab, before, n, res, w_mm=2.0):
    """Thin strips the outer cut leaves of something wide join the neighbouring colour; lines that were thin
    anyway stay. lab: n marks the outside; before: the same map uncut."""
    inside = lab != n
    r = w_mm / 2 / res
    q = int(r) + 2

    def wide(M):
        core = ndi.distance_transform_edt(np.pad(M, q)) > r
        return (ndi.distance_transform_edt(~core) <= r + 1.5)[q:-q, q:-q]
    near = ndi.distance_transform_edt(inside) * res < w_mm * 2
    rim = ndi.binary_dilation(~inside, iterations=2)
    bad = np.zeros(lab.shape, bool)

    def thin_of(c):
        out = np.zeros(lab.shape, bool)
        cc, k = ndi.label(ndi.binary_dilation((lab == c) & near, iterations=3))      # only where it meets the rim
        for i, sl in enumerate(ndi.find_objects(cc), 1):
            t = cc[sl] == i
            if (rim[sl] & t).any():
                g = tuple(slice(max(0, s.start - 3 * q), s.stop + 3 * q) for s in sl)
                tg = np.zeros(lab[g].shape, bool)
                tg[tuple(slice(a.start - b.start, a.stop - b.start) for a, b in zip(sl, g))] = t
                M = lab[g] == c
                out[g] |= tg & M & near[g] & ~wide(M) & wide(before[g] == c)
        return out
    with ThreadPoolExecutor(6) as ex:
        thins = list(ex.map(thin_of, range(n)))
    for thin in thins:
        cc, k = ndi.label(thin)
        if k:
            ok = np.zeros(k + 1, bool)
            ok[np.unique(cc[rim & thin])] = True       # only strips that lie along the outline
            ok[0] = False
            bad |= ok[cc]
    good = inside & ~bad
    if bad.any() and good.any():
        i = ndi.distance_transform_edt(~good, return_distances=False, return_indices=True)
        lab = np.where(bad, lab[i[0], i[1]], lab)
    return lab


def widen_thin(lab: np.ndarray, order, res: float) -> np.ndarray:
    """Lines thinner than MIN_W are widened to about MIN_W (taken from the neighbours)."""
    w = max(3, int(round(MIN_W / res)) | 1)
    out = lab.copy()

    def grow_of(c):                              # read from lab only: the colours can be looked at side by side
        M = (lab == c).view(np.uint8)
        core = ndi.maximum_filter(ndi.minimum_filter(M, size=w), size=w + 2)
        thin = (M > 0) & (core == 0)
        cc, k = ndi.label(thin, np.ones((3, 3)))
        if not k:
            return None
        keep = np.zeros(k + 1, bool)
        for i, sl in enumerate(ndi.find_objects(cc), 1):
            keep[i] = math.hypot(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) * res > 2.5
        return ndi.maximum_filter(keep[cc].view(np.uint8), size=3) > 0
    with pool(4) as ex:                          # four: every one holds a few full-size maps
        for c, grow in zip(order, ex.map(grow_of, order)):
            if grow is not None:
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
        used = np.zeros(n, bool)                 # numbered again 0, 1, 2 ... (as np.unique would, without a sort)
        used[mp[sz > 0]] = True
        L = (np.cumsum(used) - 1)[mp][L].astype(np.int32)
    return L


# ------------------------------------------------------------------ outlines
_SEG = {1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)], 5: [(3, 0), (1, 2)], 6: [(0, 2)], 7: [(3, 2)],
        8: [(2, 3)], 9: [(2, 0)], 10: [(0, 1), (2, 3)], 11: [(2, 1)], 12: [(1, 3)], 13: [(1, 0)], 14: [(0, 3)]}


def outline(m, oy, ox, res, sig=1.2):
    """Smooth polygon (mm, y down) of a mask: marching squares on a lightly blurred mask."""
    return _whole([Polygon(q).buffer(0) for q in _loops(m, oy, ox, res, sig)])


def outlines(items, res, sig=1.2, tick=None) -> list:
    """outline() of many masks [(m, oy, ox), ...], the same polygons; the GEOS work goes in big batches to a
    few threads (one call per batch lets go of the GIL once, small calls one by one would fight over it).
    tick(share done) now and then."""
    loops = []
    for k, (m, oy, ox) in enumerate(items):
        loops.append(_loops(m, oy, ox, res, sig))
        if tick and k % 200 == 199:
            tick(k / len(items))
    ps = batch(lambda a: shapely.buffer(a, 0), [Polygon(q) for lp in loops for q in lp])
    out, k = [], 0
    for lp in loops:
        out.append(_whole(list(ps[k:k + len(lp)])))
        k += len(lp)
    return out


def batch(fn, geoms) -> np.ndarray:
    """fn (a vectorised shapely call) over a list of geometries, in a few large parts side by side."""
    a = np.empty(len(geoms), object)
    a[:] = geoms
    if len(a) < 64:
        return fn(a) if len(a) else a
    with pool() as ex:
        return np.concatenate(list(ex.map(fn, np.array_split(a, THREADS))))


def shrunk(Ps, gap, edge) -> np.ndarray:
    """The outlines Ps shrunk by half the joint (a stone too small for that by a fifth of it) and kept inside the
    artwork's edge (prepared polygon or None); as one by one, the GEOS work in batches."""
    S = batch(lambda a: shapely.simplify(shapely.buffer(a, -gap / 2, quad_segs=3), 0.05), Ps)
    for k in np.flatnonzero(shapely.is_empty(S)):
        S[k] = Ps[k].buffer(-gap / 5, quad_segs=3)
    if edge is not None and len(S):
        k = np.flatnonzero(~shapely.covers(edge, S))
        if len(k):
            S[k] = shapely.intersection(S[k], edge)      # the exact curve instead of the pixel one
    return S


def _whole(ps):
    """The polygon of a mask from its rings (buffered): the largest, holes cut, islands joined."""
    if not ps:
        return None
    ps = sorted(ps, key=lambda p: -p.area)
    P = ps[0]
    for h in ps[1:]:
        if h.area > 0.6:
            P = P.difference(h) if P.contains(h.representative_point()) else P.union(h)
    return P


def _loops(m, oy, ox, res, sig):
    """The closed lines around a mask (mm, y down) as point arrays, in the order the walk finds them."""
    f = ndi.gaussian_filter(np.pad(m.astype(np.float32), 5), sig, truncate=3.0) - 0.5
    ins = (f > 0).view(np.uint8)
    case = ins[:-1, :-1] + 2 * ins[:-1, 1:] + 4 * ins[1:, 1:] + 8 * ins[1:, :-1]
    W = f.shape[1]
    cells = np.flatnonzero((case > 0) & (case < 15))
    cv_all = case.ravel()[cells]
    o = np.argsort(cv_all, kind="stable")        # by case, row by row inside one case: the order the walk relies on
    cells, cv_all = cells[o], cv_all[o]
    cut = np.searchsorted(cv_all, np.arange(1, 16))
    ka, kb = [], []
    for cv, segs in _SEG.items():
        c = cells[cut[cv - 1]:cut[cv]]
        if not len(c):
            continue
        ys, xs = np.divmod(c, W - 1)
        base = ys * W + xs                       # edge id: 2*pixel (horizontal) or 2*pixel+1 (vertical)
        e = {0: 2 * base, 1: 2 * (base + 1) + 1, 2: 2 * (base + W), 3: 2 * base + 1}
        for a, b in segs:
            ka.append(e[a])
            kb.append(e[b])
    ff = f.ravel()
    off = ((ox - 4.5) * res, (oy - 4.5) * res)
    K = np.concatenate(ka) if ka else np.zeros(0, np.int64)
    V = np.concatenate(kb) if kb else K
    if np.unique(K).size == K.size:
        loops = _chains(K, V, ff, W, res, off)
    else:                                        # an edge met twice (never seen): the plain walk
        nx = {}
        for a, b in zip(ka, kb):
            nx.update(zip(a.tolist(), b.tolist()))

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
                loops.append(np.asarray(lp) * res + off)
    return loops


def _chains(K, V, ff, W, res, off):
    """The closed lines of marching squares: K[i] -> V[i] are the segments (edge ids, each K once), walked
    from the first unwalked edge in K's order, exactly as a dict walk would. Points as the scalar code gives
    them (float32 crossing, whole pixel on the other axis)."""
    o = np.argsort(K)
    j = np.minimum(np.searchsorted(K[o], V), len(K) - 1)
    nxt = np.where(K[o][j] == V, o[j], -1).tolist()
    p = K >> 1
    y, x = np.divmod(p, W)
    vert = (K & 1).astype(bool)
    a = ff[p]
    b = np.where(vert, ff[np.minimum(p + W, len(ff) - 1)], ff[np.minimum(p + 1, len(ff) - 1)])
    t = a / (a - b)                              # float32, as in the scalar code
    px = np.where(vert, x, x.astype(np.float32) + t).astype(np.float64)
    py = np.where(vert, y.astype(np.float32) + t, y).astype(np.float64)
    xy = np.stack([px, py], 1)
    seen = bytearray(len(K))
    loops = []
    for s in range(len(K)):
        if seen[s]:
            continue
        lp = []
        e = s
        while e >= 0 and not seen[e]:
            seen[e] = 1
            lp.append(e)
            e = nxt[e]
        if len(lp) > 6:
            loops.append(xy[lp] * res + off)
    return loops


def _parts(g):
    return [p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon"]


# ------------------------------------------------------------------ the whole thing
_AREAS: dict = {}        # the newest picture's colour areas: another texture or stone size starts from them


def _areas(img, width_mm, size, colours, to, lum, zoom, labels) -> np.ndarray:
    """Palette index of every fine pixel (size = W, H): areas that can be cut."""
    from . import renk, sade
    key = None
    if labels is None and zoom == 1.0:
        oz = json.dumps([[c["oz"], c.get("yol", "")] for c in colours])
        key = (hashlib.blake2b(img.tobytes()).digest(), img.shape, width_mm, size, oz, to.tobytes(), lum.tobytes())
        if key in _AREAS:
            return _AREAS[key].copy()
    W, H = size
    res, n = width_mm / W, len(lum)
    if labels is not None:
        lab = np.asarray(labels)
    elif sade.tonal(colours):
        lab = sade.label(img, colours, width_mm)
    else:
        cw = int(min(W, np.clip(img.shape[1], 900 / zoom, 1500 / zoom)))
        ch = max(8, int(round(H * cw / W)))
        im = np.asarray(Image.fromarray(img).resize((cw, ch), Image.BICUBIC if cw >= img.shape[1] else Image.BOX))
        lab = renk.label(im, colours, cw * zoom)
    lab = to[renk.enlarge(lab, len(colours), (W, H))].astype(np.int16)
    lab = drop_specks(lab, n, 0.8 / res ** 2)
    lab = widen_thin(lab, [int(i) for i in np.argsort(np.abs(lum - 128))], res)
    lab = drop_specks(lab, n, MIN_A / res ** 2)
    if key is not None:
        _AREAS.clear()
        _AREAS[key] = lab.copy()
    return lab


def build(img: np.ndarray, width_mm: float, pal: np.ndarray, *, layout="mozaik", zone=None, centre=None,
          gap=0.4, size=1.0, seed=7, ams=False, report=None, colours=None, to=None, zoom=1.0,
          labels=None, pieces=None, base=None, shape=None) -> dict:
    """img: source picture (uint8, any resolution). zone/centre: Vocab map (any resolution) and ray
    centre as 0..1 fractions. colours/to: the picture's colours (renk.find) and the palette index of
    each; found here when missing. zoom: > 1 when img is only a part of the picture. labels: index map
    into colours when it is already known. pieces: how many stones the picture should have ("sekil": every shape
    one stone, None: the texture decides). base: palette colour that makes no stones; it becomes the one plate
    the stones sit in. shape: outline of the artwork (polygon, mm, y down), None = the whole rectangle."""
    from . import parca
    report = report or (lambda f, m: None)
    rng = np.random.default_rng(seed)
    res = resolution(width_mm)
    W = int(round(width_mm / res))
    H = max(8, int(round(img.shape[0] * W / img.shape[1])))
    res = width_mm / W
    report(0.03, "Renkler ayrılıyor")
    if colours is None:
        colours, to = image_colours(img, pal)
    n = len(pal)
    # the picture's own lightness (not the chosen paint) decides the order, so repainting moves no stone
    lum = np.zeros(n)
    for i in range(n):
        own = [c["rgb"] for c, t in zip(colours, to) if t == i]
        lum[i] = np.mean(own or [pal[i]], axis=0) @ np.array([.3, .59, .11])
    lab = _areas(img, width_mm, (W, H), colours, np.asarray(to), lum, zoom, labels)
    edge = None
    if shape is not None:                        # what lies outside is label n: no colour, never a stone
        from .kadraj import mask
        uncut = lab.copy()
        lab[~mask(shape, (W, H), res)] = n
        lab = drop_specks(cut_slivers(lab, uncut, n, res), n, MIN_A / res ** 2, n)
        edge = shape.buffer(-gap / 2, quad_segs=90)
        shapely.prepare(edge)
    report(0.25, "Taşlar kesiliyor")

    z = np.zeros((H, W), np.int8)
    if zone is not None:
        z = np.asarray(Image.fromarray(zone.astype(np.uint8)).resize((W, H), Image.NEAREST)).astype(np.int8)
    c_px = (centre[0] * W, centre[1] * H) if centre is not None else None
    ray = (z == int(Vocab.RAYS)) if c_px is not None else np.zeros((H, W), bool)

    full = lab
    if ams and not pieces:
        base = None                              # printed together anyway: the lines stay stones,
        # and isolated small shapes are printed into the stones
        lab = drop_specks(lab.copy(), n, FEATURE_A / res ** 2, n if edge is not None else -1)

    jobs = []
    for c in range(n):
        cc, k = ndi.label(lab == c)
        for i, sl in enumerate(ndi.find_objects(cc), 1):
            jobs.append((c, sl, cc[sl] == i, int(rng.integers(1 << 31))))
    more = {"areas": sum(j[0] != base for j in jobs), "contour": parca.contour(full, n)}

    def work(job):
        if job[0] == base:
            return np.where(job[2], 0, -1).astype(np.int32), 1
        return divide(job)

    def divide(job):
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

    if pieces:
        lines = None
        if ams and base is not None:             # the base colour is printed with the panel, not assembled
            lines, k = np.full((H, W), -1, np.int32), 0
            for c, sl, m, sd in jobs:
                if c == base:
                    o, q = divide((c, sl, m, sd))
                    lines[sl][m] = o[m] + k
                    k += q
        return _counted(lab, z, n, res, pal, colours, pieces, base, lines, layout, gap, seed, report, edge) | more

    L = np.full((H, W), -1, np.int32)
    nxt = done = 0
    total = sum(j[2].size for j in jobs) or 1
    with pool() as ex:                           # the biggest areas first: they take longest
        for (c, sl, m, _), (o, q) in zip(jobs, big_first(ex, work, jobs, lambda j: j[2].size)):
            L[sl][m] = o[m] + nxt
            nxt += q
            done += m.size
            report(0.25 + 0.40 * done / total, "Taşlar kesiliyor")
    outside = lab == n
    L[outside] = nxt
    L = _merge_small(_split_components(L), lab, 14.0 / res ** 2)
    out = np.zeros(int(L.max()) + 1, bool)       # labels that lie outside the artwork
    out[np.unique(L[outside])] = True
    report(0.72, "Taşların kenarı çiziliyor")

    # the joint around a small feature is taken from its big neighbour, so features keep their width
    sz = np.bincount(L.ravel()) * res ** 2
    small = ((sz < 60) & ~out)[L]
    if small.any():
        near = (ndi.maximum_filter(small.view(np.uint8), size=3) > 0) & ~small & ~outside
        ys, xs = np.nonzero(near)
        Lg = L.copy()
        for oy, ox in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            yy, xx = np.clip(ys + oy, 0, H - 1), np.clip(xs + ox, 0, W - 1)
            hit = small[yy, xx]
            Lg[ys[hit], xs[hit]] = L[yy[hit], xx[hit]]
        L = Lg

    polys, pcol, pstone, pzone = [], [], [], []
    objs = ndi.find_objects(L + 1)
    todo = []
    for i, sl in enumerate(objs):
        if sl is None or out[i]:
            continue
        m = L[sl] == i
        if base is not None and np.bincount(full[sl][m], minlength=n)[:n].argmax() == base:
            continue
        todo.append((i, sl, m))
    Ps = outlines([(m, sl[0].start, sl[1].start) for _, sl, m in todo], res,
                  tick=lambda f: report(0.72 + 0.2 * f, "Taşların kenarı çiziliyor"))
    todo = [t + (P,) for t, P in zip(todo, Ps) if P is not None]
    for (i, sl, m, P), S in zip(todo, shrunk([t[3] for t in todo], gap, edge)):
        oy, ox = sl[0].start, sl[1].start
        zz = int(np.bincount(z[sl][m].clip(0)).argmax())
        cols = np.bincount(full[sl][m], minlength=n)[:n]
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
    report(1.0, "Taşlar hazır")
    return {"polys": polys, "colour": np.array(pcol), "stone": np.array(pstone), "zone": np.array(pzone),
            "label": L, "lab": full, "res": res, "size": (W * res, H * res)} | more


def _counted(lab, z, n, res, pal, colours, pieces, base, lines, layout, gap, seed, report, edge) -> dict:
    """Exactly as many stones as asked for (the same result as build's). lines: stones the base colour is made
    of all the same; they come on top of the number asked for."""
    from . import parca, sade
    from .palette import rgb_to_lab
    r = parca.make(lab, n, rgb_to_lab(np.asarray(pal, np.uint8)), res,
                   None if pieces == "sekil" else int(pieces), base=base, layout=layout,
                   tonal=sade.tonal(colours), gap=gap, seed=seed, edge=edge,
                   report=lambda f, m: report(0.25 + 0.74 * f, m))
    S, polys, pc, ps = r["label"], r["polys"], r["colour"], r["stone"]
    k = int(S.max()) + 1
    if lines is not None:
        more = np.where(S < 0, lines, -1)
        q, c, i = parca.outlines(more, np.zeros(S.shape, np.int8), [base], res, gap, edge)
        S = np.where(more >= 0, more + k, S)
        polys, pc, ps = polys + q, np.r_[pc, c], np.r_[ps, i + k]
        k = int(S.max()) + 1
    inside = S >= 0
    zc = np.bincount(S[inside].astype(np.int64) * 64 + z[inside].clip(0), minlength=k * 64).reshape(k, 64).argmax(1)
    H, W = lab.shape
    report(1.0, "Taşlar hazır")
    return {"polys": polys, "colour": pc, "stone": np.arange(len(polys)), "zone": zc[ps],
            "label": S, "lab": r["lab"], "res": res, "size": (W * res, H * res)}


def count_areas(lab, n, base=None) -> int:
    return sum(ndi.label(lab == c)[1] for c in range(n) if c != base)
