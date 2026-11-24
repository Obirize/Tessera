"""Tessellations that respect colour edges.

- ``andamento``: contour-following Voronoi stones. Seeds are laid on offset
  curves of the colour/zone boundaries (distance-transform level sets at 0.5 and
  1.5 tile widths), on the medial ridge of features thinner than a tile, then on a
  jittered grid inside; one Lloyd step evens them out. Each pixel takes the nearest
  seed *of its own colour*, so a row of stones hugs every contour (keystone row).
- ``rays_adaptive``: sunburst planks whose angular cuts snap to radial colour edges.
- ``blocks_adaptive``: square blocks whose grid lines snap to straight colour edges
  (window columns and floors of buildings).
- ``split_components`` / ``merge_small``: every tile is one 4-connected region;
  slivers left where a tile crosses a colour edge are merged into the neighbouring
  tile of the same colour, while isolated thin features keep their own tile.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


# ------------------------------------------------------------------ distance to edges
def edge_distance(key: np.ndarray) -> np.ndarray:
    """Distance (px) of every pixel to the nearest pixel lying on a boundary of ``key``
    (pixels with a 4-neighbour of different key, plus the image border)."""
    b = np.zeros(key.shape, bool)
    dh = key[:, 1:] != key[:, :-1]
    dv = key[1:, :] != key[:-1, :]
    b[:, 1:] |= dh
    b[:, :-1] |= dh
    b[1:, :] |= dv
    b[:-1, :] |= dv
    b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = True
    return ndi.distance_transform_edt(~b).astype(np.float32)


# ------------------------------------------------------------------ andamento
def _nms(cand: np.ndarray, rng, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Thin a candidate mask: keep pixels whose random priority is the max in a size x size window."""
    if not cand.any():
        return np.zeros(0, int), np.zeros(0, int)
    pr = np.where(cand, rng.random(cand.shape, dtype=np.float32), -1.0)
    keep = cand & (pr >= ndi.maximum_filter(pr, size=size))
    return np.nonzero(keep)


def _poisson(cands, shape, r_cell):
    """Greedy Poisson-disc acceptance. ``cands``: list of (ys, xs, cols, r) arrays in priority
    order. A candidate is accepted when no accepted seed of the same colour lies within r."""
    grid: dict = {}
    out_y, out_x, out_c = [], [], []
    cs = float(r_cell)
    for ys, xs, cols, r in cands:
        r2 = r * r
        reach = int(math.ceil(r / cs))
        for y, x, c in zip(ys.tolist(), xs.tolist(), cols.tolist()):
            gy, gx = int(y // cs), int(x // cs)
            ok = True
            for yy in range(gy - reach, gy + reach + 1):
                for xx in range(gx - reach, gx + reach + 1):
                    lst = grid.get((c, yy, xx))
                    if lst:
                        for (py, px) in lst:
                            if (py - y) ** 2 + (px - x) ** 2 < r2:
                                ok = False
                                break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                grid.setdefault((c, gy, gx), []).append((y, x))
                out_y.append(y)
                out_x.append(x)
                out_c.append(c)
    return np.array(out_y, np.float64), np.array(out_x, np.float64), np.array(out_c, np.int64)


def _nearest_seed(mask: np.ndarray, sy, sx, ids) -> np.ndarray:
    """For pixels in ``mask`` return the id of the nearest seed (Euclidean, unconstrained), computed
    on the bounding box of mask+seeds only."""
    h, w = mask.shape
    yy, xx = np.nonzero(mask)
    iy = np.clip(np.rint(sy).astype(int), 0, h - 1)
    ix = np.clip(np.rint(sx).astype(int), 0, w - 1)
    y0, y1 = min(yy.min(), iy.min()), max(yy.max(), iy.max()) + 1
    x0, x1 = min(xx.min(), ix.min()), max(xx.max(), ix.max()) + 1
    inp = np.ones((y1 - y0, x1 - x0), bool)
    inp[iy - y0, ix - x0] = False
    idm = np.full(inp.shape, -1, np.int64)
    idm[iy - y0, ix - x0] = ids
    _, (ny, nx) = ndi.distance_transform_edt(inp, return_indices=True)
    out = np.full((h, w), -1, np.int64)
    sub = mask[y0:y1, x0:x1]
    out[y0:y1, x0:x1][sub] = idm[ny[sub], nx[sub]]
    return out


def andamento(colour: np.ndarray, region: np.ndarray, dist: np.ndarray, s: float, rng,
              lloyd: int = 1, rows: int = 2) -> np.ndarray:
    """Contour-following stones inside ``region``. Returns (H, W) int64 local ids (valid in region)."""
    h, w = colour.shape
    loc = np.zeros((h, w), np.int64)
    if not region.any():
        return loc
    s = max(3.0, float(s))
    nms = 3 if s < 12 else 5
    cands = []
    # 1) keystone rows on offset curves of every colour/zone boundary
    lv = [(i + 0.5) * s - 0.5 for i in range(rows)]
    for i, lev in enumerate(lv):
        band = region & (np.abs(dist - lev) < 0.6)
        ys, xs = _nms(band, rng, nms)
        cands.append((ys, xs, colour[ys, xs], 0.82 * s))
        if i == 0:
            # 2) medial ridge of features too thin for a full row (whiskers, outlines)
            ridge = region & (dist < lev) & (dist >= ndi.maximum_filter(dist, size=3) - 1e-4)
            ys, xs = _nms(ridge, rng, nms)
            cands.append((ys, xs, colour[ys, xs], 1.0 * s))
    # 3) jittered grid in the interior
    gy, gx = np.mgrid[0:h // s + 1, 0:w // s + 1]
    py = ((gy + 0.5 + rng.uniform(-0.3, 0.3, gy.shape)) * s).ravel()
    px = ((gx + 0.5 + rng.uniform(-0.3, 0.3, gx.shape)) * s).ravel()
    ok = (py < h) & (px < w)
    py, px = py[ok].astype(int), px[ok].astype(int)
    ok = region[py, px] & (dist[py, px] >= lv[-1] + 0.4 * s)
    cands.append((py[ok], px[ok], colour[py[ok], px[ok]], 0.82 * s))
    sy, sx, sc = _poisson(cands, (h, w), 0.82 * s)
    if len(sy) == 0:
        return loc
    ids = np.arange(len(sy))
    cols = np.unique(colour[region])
    for it in range(lloyd + 1):
        loc[:] = -1
        for c in cols:
            m = region & (colour == c)
            sel = sc == c
            if not sel.any():
                continue
            loc[m] = _nearest_seed(m, sy[sel], sx[sel], ids[sel])[m]
        if it == lloyd:
            break
        yy, xx = np.nonzero(region & (loc >= 0))
        lab = loc[yy, xx]
        cnt = np.bincount(lab, minlength=len(sy))
        okc = cnt > 0
        my = np.bincount(lab, yy, minlength=len(sy))
        mx = np.bincount(lab, xx, minlength=len(sy))
        sy[okc] = my[okc] / cnt[okc]
        sx[okc] = mx[okc] / cnt[okc]
    loc[loc < 0] = len(sy)  # colours without seeds: one tile per component (split later)
    return loc


# ------------------------------------------------------------------ adaptive 1-D cuts
def adaptive_cuts(hist: np.ndarray, nominal: float, thresh: float, min_frac: float = 0.45,
                  circular: bool = False) -> np.ndarray:
    """Cut positions (bin indices) along a 1-D axis of len(hist) bins: strong peaks of ``hist``
    first, then each interval subdivided into ~``nominal``-long parts."""
    n = len(hist)
    hs = ndi.gaussian_filter1d(hist.astype(np.float64), 1.0, mode="wrap" if circular else "nearest")
    pk = np.nonzero((hs >= thresh) & (hs >= ndi.maximum_filter1d(hs, 3, mode="wrap" if circular else "nearest")))[0]
    chosen: list[int] = []
    min_sp = max(1.0, min_frac * nominal)
    for p in pk[np.argsort(-hs[pk])]:
        if all(min(abs(p - q), n - abs(p - q)) >= min_sp if circular else abs(p - q) >= min_sp for q in chosen):
            if circular or (p >= min_sp and n - p >= min_sp):
                chosen.append(int(p))
    chosen.sort()
    cuts: list[float] = []
    if circular:
        if not chosen:
            k = max(1, int(round(n / nominal)))
            return np.arange(k) * n / k
        for a, b in zip(chosen, chosen[1:] + [chosen[0] + n]):
            k = max(1, int(round((b - a) / nominal)))
            cuts.extend(a + (b - a) * np.arange(k) / k)
        return np.sort(np.mod(cuts, n))
    bounds = [0] + chosen + [n]
    for a, b in zip(bounds[:-1], bounds[1:]):
        k = max(1, int(round((b - a) / nominal)))
        cuts.extend(a + (b - a) * np.arange(k) / k)
    return np.array(cuts[1:])


def rays_adaptive(colour: np.ndarray, region: np.ndarray, centre, plank_px: float, deg: float = 2.2,
                  hub_frac: float = 0.9, min_strip_px: float = 15.0) -> np.ndarray:
    """Radial planks whose strips follow the colour wedges of a sunburst.

    The zone is cut into thin rings around ``centre``; in each ring every run of one colour is
    the cross-section of a ray wedge. Runs that overlap one-to-one from ring to ring are chained
    into a wedge. Each wedge is split into a fixed number of strips (~``deg`` wide) measured
    *within the run*, so strip edges follow the wedge's own colour edges even when the rays are
    not exactly radial. Planks are then cut along the radius with a brick-like stagger."""
    h, w = colour.shape
    out = np.zeros((h, w), np.int64)
    if not region.any():
        return out
    yy, xx = np.nonzero(region)
    dx, dy = xx - centre[0], yy - centre[1]
    r = np.hypot(dx, dy)
    th = np.degrees(np.arctan2(dy, dx)) % 360.0
    col = colour[yy, xx].astype(np.int64)
    # put the 0/360 seam on a strong colour edge so that no wedge is cut in two
    e = np.zeros((h, w), bool)
    e[:, 1:] |= colour[:, 1:] != colour[:, :-1]
    e[1:, :] |= colour[1:, :] != colour[:-1, :]
    ee = e[yy, xx]
    if ee.any():
        hist = np.bincount((th[ee]).astype(np.int64) % 360, minlength=360)
        th = (th - float(hist.argmax())) % 360.0
    ring = (r / max(plank_px * 0.5, 2.0)).astype(np.int64)
    rkey = col * 100000 + ring
    order = np.lexsort((th, rkey))
    th_s, rk_s, r_s = th[order], rkey[order], r[order]
    gap_thr = np.degrees(2.5 / np.maximum(r_s, 1.0)) + 0.05
    new = np.ones(len(order), bool)
    new[1:] = (rk_s[1:] != rk_s[:-1]) | (th_s[1:] - th_s[:-1] > gap_thr[1:])
    run = np.cumsum(new) - 1
    starts = np.nonzero(new)[0]
    ends = np.append(starts[1:], len(order)) - 1
    lo, hi, rk = th_s[starts], th_s[ends], rk_s[starts]
    nr = len(starts)
    # one-to-one overlaps between consecutive rings -> wedges
    loG, hiG = rk * 1000.0 + lo, rk * 1000.0 + hi

    def overlaps(dr):
        q_lo, q_hi = (rk + dr) * 1000.0 + lo, (rk + dr) * 1000.0 + hi
        first = np.searchsorted(hiG, q_lo, "left")
        last = np.searchsorted(loG, q_hi, "right")
        return first, last - first

    first_up, cnt_up = overlaps(-1)     # runs in the inner ring overlapping each run
    _, cnt_dn = overlaps(+1)            # runs in the outer ring overlapping each run
    ok = cnt_up == 1
    a_idx = np.nonzero(ok)[0]
    p_idx = first_up[ok]
    keep = cnt_dn[p_idx] == 1
    a_idx, p_idx = a_idx[keep], p_idx[keep]
    g = coo_matrix((np.ones(len(a_idx)), (a_idx, p_idx)), shape=(nr, nr))
    _, wedge = connected_components(g, directed=False)
    ext = np.maximum(hi - lo, 1e-6)
    npx = (ends - starts + 1).astype(np.float64)
    wsum = np.bincount(wedge, ext * npx)
    wn = np.bincount(wedge, npx)
    nstrip = np.maximum(1, np.rint(wsum / np.maximum(wn, 1) / deg)).astype(np.int64)
    ns_run = nstrip[wedge]
    # straight wedge sides: fit lo(r), hi(r) = a + b / r per wedge (a straight line in the image
    # that need not pass through the centre); strip cuts interpolate between them -> straight too
    rr = np.maximum(0.5 * (r_s[starts] + r_s[ends]), 1.0)
    u = 1.0 / rr
    nw = int(wedge.max()) + 1
    s0 = np.bincount(wedge, npx, nw)
    s1 = np.bincount(wedge, npx * u, nw)
    s2 = np.bincount(wedge, npx * u * u, nw)

    def fit(y):
        t0 = np.bincount(wedge, npx * y, nw)
        t1 = np.bincount(wedge, npx * y * u, nw)
        det = s0 * s2 - s1 * s1
        okd = np.abs(det) > 1e-12 * np.maximum(s0 * s2, 1e-30)
        bb = np.where(okd, (s0 * t1 - s1 * t0) / np.where(okd, det, 1), 0.0)
        aa = (t0 - bb * s1) / np.maximum(s0, 1e-9)
        return aa, bb

    la, lb = fit(lo)
    ha, hb = fit(hi)
    wr = wedge[run]
    lo_px = la[wr] + lb[wr] / np.maximum(r_s, 1.0)
    hi_px = ha[wr] + hb[wr] / np.maximum(r_s, 1.0)
    span = np.maximum(hi_px - lo_px, 1e-6)
    f = (th_s - lo_px) / span
    # no needle planks: near the centre (where the wedge is narrow) use fewer, wider strips
    ns_eff = np.minimum(ns_run[run], np.maximum(1, np.floor(np.radians(span) * r_s / max(min_strip_px, 1.0)))).astype(np.int64)
    j = np.clip(np.floor(f * ns_eff).astype(np.int64), 0, ns_eff - 1)
    sid = (wr.astype(np.int64) * 64 + ns_eff) * 64 + j
    stagger = (sid * 37 % 11) / 11.0
    plank = (r_s / max(plank_px, 4) + stagger).astype(np.int64)
    val = sid * 10000 + plank
    # hub: when the centre lies in the zone, the middle becomes one disc per colour
    hub_r = hub_frac * plank_px
    if hub_r > 0 and r.min() < hub_r:
        val = np.where(r_s < hub_r, -1 - col[order], val)
    out[yy[order], xx[order]] = val
    return out


def rays_logpolar(colour: np.ndarray, region: np.ndarray, centre, deg: float = 2.2, aspect: float = 3.5,
                  r_min_px: float = 6.0, min_strip_px: float = 5.0) -> np.ndarray:
    """Round-5 rays (the first version's look, with stronger perspective): log-polar planks.

    - Angular width is constant (``deg``), the radial length is proportional to r
      (length = aspect x width), so planks are small near the sun and grow toward the edges.
    - Joints are staggered from strip to strip.
    - Strips follow each colour wedge: in every thin ring, a run of one colour is split into the
      wedge's fixed number of strips measured inside the run, so a ray-colour boundary only runs
      along the length of a plank, never across it in steps.
    """
    h, w = colour.shape
    out = np.zeros((h, w), np.int64)
    if not region.any():
        return out
    yy, xx = np.nonzero(region)
    dx, dy = xx - centre[0], yy - centre[1]
    r = np.maximum(np.hypot(dx, dy), 1e-3)
    th = np.degrees(np.arctan2(dy, dx)) % 360.0
    col = colour[yy, xx].astype(np.int64)
    # seam on a strong colour edge so that no wedge is split at 0/360
    e = np.zeros((h, w), bool)
    e[:, 1:] |= colour[:, 1:] != colour[:, :-1]
    e[1:, :] |= colour[1:, :] != colour[:-1, :]
    ee = e[yy, xx]
    if ee.any():
        hist = np.bincount((th[ee]).astype(np.int64) % 360, minlength=360)
        th = (th - float(hist.argmax())) % 360.0
    q = np.radians(deg) * aspect                      # radial length / r
    lq = np.log1p(q)
    lr = np.log(np.maximum(r, r_min_px))
    # thin rings (a quarter plank) to read the wedge cross-sections
    ring = np.floor(lr / (lq / 4)).astype(np.int64)
    rkey = col * 1000000 + ring
    order = np.lexsort((th, rkey))
    th_s, rk_s, r_s, lr_s = th[order], rkey[order], r[order], lr[order]
    gap_thr = np.degrees(2.5 / np.maximum(r_s, 1.0)) + 0.05
    new = np.ones(len(order), bool)
    new[1:] = (rk_s[1:] != rk_s[:-1]) | (th_s[1:] - th_s[:-1] > gap_thr[1:])
    run = np.cumsum(new) - 1
    starts = np.nonzero(new)[0]
    ends = np.append(starts[1:], len(order)) - 1
    lo, hi, rk = th_s[starts], th_s[ends], rk_s[starts]
    nr = len(starts)
    loG, hiG = rk * 1000.0 + lo, rk * 1000.0 + hi

    def overlaps(dr):
        q_lo, q_hi = (rk + dr) * 1000.0 + lo, (rk + dr) * 1000.0 + hi
        first = np.searchsorted(hiG, q_lo, "left")
        last = np.searchsorted(loG, q_hi, "right")
        return first, last - first

    first_up, cnt_up = overlaps(-1)
    _, cnt_dn = overlaps(+1)
    ok = cnt_up == 1
    a_idx = np.nonzero(ok)[0]
    p_idx = first_up[ok]
    keep = cnt_dn[p_idx] == 1
    a_idx, p_idx = a_idx[keep], p_idx[keep]
    g = coo_matrix((np.ones(len(a_idx)), (a_idx, p_idx)), shape=(nr, nr))
    _, wedge = connected_components(g, directed=False)
    ext = np.maximum(hi - lo, 1e-6)
    npx = (ends - starts + 1).astype(np.float64)
    # strips per wedge from its typical (outer-half) angular extent
    wsum = np.bincount(wedge, ext * npx)
    wn = np.bincount(wedge, npx)
    nstrip = np.maximum(1, np.rint(wsum / np.maximum(wn, 1) / deg)).astype(np.int64)
    ns_run = nstrip[wedge]
    # where a wedge tapers (or near the sun) fewer strips, never thinner than min_strip_px
    ns_run = np.minimum(ns_run, np.maximum(1, np.floor(np.radians(ext) * np.maximum(r_s[starts], 1.0)
                                                       / max(min_strip_px, 1.0)))).astype(np.int64)
    f = (th_s - lo[run]) / ext[run]
    j = np.clip(np.floor(f * ns_run[run]).astype(np.int64), 0, ns_run[run] - 1)
    sid = (wedge[run].astype(np.int64) * 64 + ns_run[run]) * 64 + j
    stagger = ((sid * 37) % 11) / 11.0
    plank = np.floor(lr_s / lq + stagger).astype(np.int64)
    out[yy[order], xx[order]] = sid * 100000 + plank
    return out


def focal_rings(colour: np.ndarray, region: np.ndarray, s: float) -> np.ndarray:
    """Ring-and-slice discs for FOCAL zones: a centre disc, then rings ~``s`` wide cut into
    slices ~``s`` long (staggered from ring to ring), around each zone component's centre."""
    h, w = colour.shape
    s = max(3.0, float(s))
    out = np.zeros((h, w), np.int64)
    cc, n = ndi.label(region)
    for i, sl in enumerate(ndi.find_objects(cc)):
        if sl is None:
            continue
        sub = cc[sl] == i + 1
        yy, xx = np.nonzero(sub)
        cy, cx = yy.mean(), xx.mean()
        r = np.hypot(yy - cy, xx - cx)
        th = (np.arctan2(yy - cy, xx - cx) / (2 * np.pi)) % 1.0
        ring = (r / s + 0.5).astype(np.int64)            # ring 0 = centre disc of diameter s
        nsl = np.maximum(1, np.rint(2 * np.pi * ring)).astype(np.int64)
        nsl[ring == 0] = 1
        sl_i = ((th + 0.5 * (ring % 2) / nsl) * nsl).astype(np.int64) % nsl
        o = out[sl]
        o[yy, xx] = (i * 4096 + ring) * 4096 + sl_i
    return out


def grid_angle(colour: np.ndarray, mask: np.ndarray) -> float:
    """Dominant grid angle (radians, in [-45, 45) deg) of the colour edges inside ``mask``,
    from a 4-fold symmetric orientation average. 0 when there is no clear direction."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 50:
        return 0.0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    col = colour[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1]
    X = Y = tot = 0.0
    for c in np.unique(col[m]):
        g = ndi.gaussian_filter((col == c).astype(np.float32), 1.2)
        gy, gx = ndi.sobel(g, 0), ndi.sobel(g, 1)
        mag2 = (gx * gx + gy * gy) * m
        ang = np.arctan2(gy, gx)
        X += float((mag2 * np.cos(4 * ang)).sum())
        Y += float((mag2 * np.sin(4 * ang)).sum())
        tot += float(mag2.sum())
    if tot <= 0 or np.hypot(X, Y) / tot < 0.25:
        return 0.0
    phi = 0.25 * np.arctan2(Y, X)
    return float(phi) if abs(np.degrees(phi)) > 4 else 0.0


def blocks_adaptive(colour: np.ndarray, region: np.ndarray, s: float, angle: bool = True) -> np.ndarray:
    """Square blocks per connected zone component. The grid turns to the component's dominant
    edge direction (oblique bands, 45 deg lattices) and its lines snap to straight colour edges."""
    h, w = colour.shape
    s = max(3.0, float(s))
    out = np.zeros((h, w), np.int64)
    cc, n = ndi.label(region)
    for i, sl in enumerate(ndi.find_objects(cc)):
        if sl is None:
            continue
        sub = cc[sl] == i + 1
        col = colour[sl]
        phi = grid_angle(col, sub) if angle else 0.0
        yy, xx = np.nonzero(sub)
        cph, sph = np.cos(phi), np.sin(phi)
        u = xx * cph + yy * sph
        v = -xx * sph + yy * cph
        ui = np.rint(u - u.min()).astype(np.int64)
        vi = np.rint(v - v.min()).astype(np.int64)
        # colour edges along the rotated axes, as histograms over u and v
        e = np.zeros(sub.shape, bool)
        e[:, 1:] |= (col[:, 1:] != col[:, :-1]) & sub[:, 1:] & sub[:, :-1]
        e[1:, :] |= (col[1:, :] != col[:-1, :]) & sub[1:, :] & sub[:-1, :]
        ee = e[yy, xx]
        # an edge pixel votes for a u-cut when its local edge runs along v (and vice versa)
        g = ndi.gaussian_filter(col.astype(np.float32), 0.8)
        gy, gx = ndi.sobel(g, 0)[yy, xx], ndi.sobel(g, 1)[yy, xx]
        gu = np.abs(gx * cph + gy * sph)
        gv = np.abs(-gx * sph + gy * cph)
        hu = np.bincount(ui[ee & (gu > gv)], minlength=ui.max() + 1)
        hv = np.bincount(vi[ee & (gv >= gu)], minlength=vi.max() + 1)
        hh, ww = vi.max() + 1, ui.max() + 1
        xc = adaptive_cuts(hu, s, thresh=max(0.08 * hh, 1.6 * s))
        yc = adaptive_cuts(hv, s, thresh=max(0.08 * ww, 1.6 * s))
        bx = np.searchsorted(xc, ui, side="right")
        by = np.searchsorted(yc, vi, side="right")
        o = out[sl]
        o[yy, xx] = (i * 4096 + by) * 4096 + bx
    return out


# ------------------------------------------------------------------ connectivity
def split_components(lab: np.ndarray) -> np.ndarray:
    """Give every 4-connected fragment its own id (tiles must be connected to be printable)."""
    out = np.zeros_like(lab)
    nxt = 0
    objs = ndi.find_objects(lab + 1)
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        sub = lab[sl] == i
        cc, k = ndi.label(sub)
        o = out[sl]
        o[sub] = cc[sub] - 1 + nxt
        nxt += k
    return out


def _adjacency(lab: np.ndarray, n: int):
    a = np.concatenate([lab[:, :-1].ravel(), lab[:-1, :].ravel()])
    b = np.concatenate([lab[:, 1:].ravel(), lab[1:, :].ravel()])
    d = a != b
    a, b = a[d], b[d]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    code, cnt = np.unique(lo.astype(np.int64) * n + hi, return_counts=True)
    return code // n, code % n, cnt


def merge_small(lab: np.ndarray, pix_key: np.ndarray, pix_min_area: np.ndarray, passes: int = 2,
                pix_thin: Optional[np.ndarray] = None, thin_scale: float = 0.35,
                pix_colour: Optional[np.ndarray] = None) -> np.ndarray:
    """Merge tiles smaller than their minimum area into the neighbour with the same ``pix_key``
    (colour + zone) that shares the longest border. With ``pix_colour``, a small tile that has
    no such neighbour (a sliver cut off by a zone border) joins the same-colour neighbour of the
    other zone instead. Tiles with no same-colour neighbour at all (isolated features such as an
    eye or a nostril) are kept. Tiles that are mostly ``pix_thin`` (segments of a thin line) use
    a smaller minimum, so a long outline stays cut into several printable pieces."""
    flat_key, flat_min = pix_key.ravel(), pix_min_area.ravel()
    flat_col = pix_colour.ravel() if pix_colour is not None else None
    for _ in range(passes):
        n = int(lab.max()) + 1
        fl = lab.ravel()
        area = np.bincount(fl, minlength=n)
        tkey = np.zeros(n, flat_key.dtype)
        tkey[fl] = flat_key
        tmin = np.zeros(n, np.float64)
        tmin[fl] = flat_min
        if pix_thin is not None:
            thin_frac = np.bincount(fl, pix_thin.ravel().astype(np.float64), n) / np.maximum(area, 1)
            tmin = np.where(thin_frac > 0.5, tmin * thin_scale, tmin)
        a0, b0, ln0 = _adjacency(lab, n)

        def edges(sel):
            a, b, ln = a0[sel], b0[sel], ln0[sel]
            src = np.concatenate([a, b])
            dst = np.concatenate([b, a])
            ln2 = np.concatenate([ln, ln])
            small = area[src] < tmin[src]
            return src[small], dst[small], ln2[small]

        src, dst, ln2 = edges(tkey[a0] == tkey[b0])
        if flat_col is not None:
            tcol = np.zeros(n, flat_col.dtype)
            tcol[fl] = flat_col
            s2, d2, l2 = edges((tcol[a0] == tcol[b0]) & (tkey[a0] != tkey[b0]))
            lonely = ~np.isin(s2, src)
            # same-key partners always win (sorted first by the large bonus)
            src = np.concatenate([src, s2[lonely]])
            dst = np.concatenate([dst, d2[lonely]])
            ln2 = np.concatenate([ln2 + 10 ** 9, l2[lonely]])
        if len(src) == 0:
            break
        order = np.lexsort((-ln2, src))
        src, dst = src[order], dst[order]
        first = np.ones(len(src), bool)
        first[1:] = src[1:] != src[:-1]
        src, dst = src[first], dst[first]
        g = coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
        _, comp = connected_components(g, directed=False)
        _, lab_new = np.unique(comp[lab], return_inverse=True)
        lab = lab_new.reshape(lab.shape)
    return lab


def snap_zones(zone: np.ndarray, colour: np.ndarray, sliver_px: np.ndarray, protect=None) -> np.ndarray:
    """Zones follow colour regions: the part of a colour region that a zone border cuts off is
    given to the region's main zone when it is only a sliver (smaller than ``sliver_px[zone]``)
    or a small share (< 20 %) of a small region. Big painted zones are left alone."""
    k = int(colour.max()) + 1
    comp = np.zeros(zone.shape, np.int64)
    nxt = 0
    for c in range(k):
        cc, m = ndi.label(colour == c)
        sel = cc > 0
        comp[sel] = cc[sel] + nxt
        nxt += m
    nz = int(zone.max()) + 1
    pair = comp * nz + zone
    pa = np.bincount(pair.ravel(), minlength=(nxt + 1) * nz).reshape(nxt + 1, nz)
    carea = pa.sum(1)
    major = pa.argmax(1)
    share = pa / np.maximum(carea[:, None], 1)
    thr = np.asarray(sliver_px, np.float64)[:nz][None, :]
    small_region = carea[:, None] < 4 * thr
    move = (pa > 0) & ((pa < thr) | (small_region & (share < 0.2)))
    move[np.arange(nxt + 1), major] = False
    if protect is not None:
        for z in protect:
            if z < nz:
                move[:, z] = False                # never take pixels away from these zones
                major_bad = major == z
                move[major_bad, :] = False        # ...and never grow them
    mv = move[comp, zone]
    out = zone.copy()
    out[mv] = major[comp[mv]]
    return out
