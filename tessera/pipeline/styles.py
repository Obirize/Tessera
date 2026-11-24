"""R6-1: four stone styles, every stone a separate printed piece.

Raster ports of the approved look-dev scripts (scripts/iterasyon_tas_v1..v4.py):

- "klasik" (V1, default): irregular Voronoi stones; rays and sun as polar planks that grow with the
  distance from the centre (plank length 4 + 0.10 r mm, width 4.8 + 0.03 r mm, brick offset,
  random ring phase per ray so joints never line up); figures raised on a rounded body with facets.
- "smalti" (V2): every colour region laid in rows that follow its outline (iso-bands of the
  distance to the colour edge, 8.2 mm; 7.4 mm in figures/focal), cut every ~12 mm (10.6) along the
  row, +-5 deg tilt per stone.
- "freud" (V3): brick courses of 14.5 mm, blocks 12.5-17.5 mm wide, each with a sculpted top
  (quarter-round or ramp) falling down the "form" field (colour pillow + rounded body).
- "orman" (V4): three height levels; big cut-paper ray planks (wavy cuts), chunky flagstones,
  10 mm square blocks for buildings, big organic pieces for figures.

Zones still steer the vocabulary inside each style (rays, figure, focal, blocks, stones).

style_cells(...) -> (H, W) int64 cell id (stones = cell x colour, split into connected parts)
style_relief(...) -> per tile (a, b, c, profile, dir, bevel)
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ..model import Vocab

STYLES = ("klasik", "smalti", "freud", "orman")
STYLE_NAMES_TR = {"klasik": "Klasik taş", "smalti": "Smalti", "freud": "Freud blok", "orman": "Orman katmanı"}
PROFILE_FLAT, PROFILE_RAMP, PROFILE_QUARTER = 0, 1, 2


# ------------------------------------------------------------------ helpers
def _noise(shape, sigma, rng):
    """Smooth unit-variance noise. Large blurs are computed on a coarse grid and interpolated up
    (same character, a fraction of the cost)."""
    f = int(max(1, min(8, sigma // 4)))
    h, w = shape
    sh = (h // f + 2, w // f + 2)
    a = ndi.gaussian_filter(rng.normal(size=sh).astype(np.float32), sigma / f)
    if f > 1:
        yy = (np.arange(h) / f).astype(np.float32)
        xx = (np.arange(w) / f).astype(np.float32)
        a = ndi.map_coordinates(a, np.meshgrid(yy, xx, indexing="ij"), order=1)
    return a / (a.std() + 1e-9)


def _jitter_seeds(h, w, s, rng, jitter=0.45, mask=None, pad=1):
    """Jittered-grid seeds (y, x) about ``s`` px apart, optionally only near ``mask``."""
    gy, gx = int(h // s) + 2, int(w // s) + 2
    cy, cx = np.mgrid[0:gy, 0:gx]
    sy = ((cy + 0.5 + rng.uniform(-jitter, jitter, cy.shape)) * s).ravel()
    sx = ((cx + 0.5 + rng.uniform(-jitter, jitter, cx.shape)) * s).ravel()
    if mask is not None:
        near = np.zeros((gy, gx), bool)
        ys, xs = np.nonzero(mask)
        if len(ys):
            near[(ys // s).astype(int), (xs // s).astype(int)] = True
        near = ndi.binary_dilation(near, np.ones((2 * pad + 1,) * 2, bool)).ravel()
        sy, sx = sy[near], sx[near]
    return sy, sx


def _voronoi_mask(mask, s, rng, lloyd=2, warp=None):
    """Voronoi cells (Lloyd-relaxed jittered grid) for the pixels of ``mask``; ``warp`` = (wy, wx)
    displacement fields for organic (warped) cuts. Returns (H, W) int64, valid inside mask."""
    h, w = mask.shape
    out = np.zeros((h, w), np.int64)
    yy, xx = np.nonzero(mask)
    if len(yy) == 0:
        return out
    sy, sx = _jitter_seeds(h, w, s, rng, mask=mask)
    py = yy.astype(np.float32) + (warp[0][yy, xx] if warp is not None else 0)
    px = xx.astype(np.float32) + (warp[1][yy, xx] if warp is not None else 0)
    pts = np.c_[py, px]
    for it in range(lloyd + 1):
        _, j = cKDTree(np.c_[sy, sx]).query(pts, workers=-1)
        if it == lloyd:
            break
        c = np.bincount(j, minlength=len(sy))
        ok = c > 0
        sy[ok] = np.bincount(j, py, len(sy))[ok] / c[ok]
        sx[ok] = np.bincount(j, px, len(sy))[ok] / c[ok]
    out[yy, xx] = j + 1
    return out


def _wedges(colour, region, cx, cy):
    """Ray wedges: per-colour angular runs in thin rings, chained one-to-one from ring to ring.
    Returns (yy, xx, r, th, wedge id per pixel) for the pixels of region."""
    yy, xx = np.nonzero(region)
    r = np.maximum(np.hypot(xx - cx, yy - cy), 1e-3)
    th = np.degrees(np.arctan2(yy - cy, xx - cx)) % 360.0
    col = colour[yy, xx].astype(np.int64)
    e = np.zeros(colour.shape, bool)
    e[:, 1:] |= colour[:, 1:] != colour[:, :-1]
    e[1:, :] |= colour[1:, :] != colour[:-1, :]
    ee = e[yy, xx]
    seam = 0.0
    if ee.any():
        seam = float(np.bincount(th[ee].astype(np.int64) % 360, minlength=360).argmax())
    th = (th - seam) % 360.0
    ring = (np.log(np.maximum(r, 4.0)) / 0.03).astype(np.int64)
    rkey = col * 1000000 + ring
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
    wid = np.empty(len(order), np.int64)
    wid[order] = wedge[run]
    return yy, xx, r, th, wid


def polar_planks(colour, region, centre, mm, rng, ring_fn, width_fn, brick=True, r0_mm=0.0):
    """Annular-sector planks (V1): rings from ``ring_fn`` (mm edges, random phase per wedge),
    every wedge cross-section in a ring split into equal angular parts ~width_fn(r) mm wide,
    alternate rings offset by half a plank (brick)."""
    h, w = colour.shape
    out = np.zeros((h, w), np.int64)
    if not region.any():
        return out
    yy, xx, r, th, wid = _wedges(colour, region, centre[0], centre[1])
    rmm = r * mm
    nw = int(wid.max()) + 1
    phase = rng.uniform(0, 1, nw)
    edges = np.asarray(ring_fn())
    # ring index with a per-wedge phase (joints of neighbouring rays never line up)
    step = np.diff(edges)
    k0 = np.clip(np.searchsorted(edges, rmm, side="right") - 1, 0, len(edges) - 2)
    shift = phase[wid] * step[k0] * (rmm > r0_mm + 1e-6)
    ring = np.clip(np.searchsorted(edges, rmm + shift, side="right") - 1, 0, len(edges) - 2)
    key = wid * 4096 + ring
    _, grp = np.unique(key, return_inverse=True)
    ng = int(grp.max()) + 1
    lo = np.full(ng, np.inf)
    hi = np.full(ng, -np.inf)
    np.minimum.at(lo, grp, th)
    np.maximum.at(hi, grp, th)
    span = np.maximum(hi - lo, 1e-6)
    rmid = np.bincount(grp, rmm, ng) / np.maximum(np.bincount(grp, minlength=ng), 1)
    n = np.maximum(1, np.rint(np.radians(span) * rmid / width_fn(rmid))).astype(np.int64)
    off = np.where((ring % 2 == 1) & brick, 0.5, 0.0)
    f = (th - lo[grp]) / span[grp] * n[grp] + off
    j = np.floor(f).astype(np.int64)
    out[yy, xx] = key * 512 + j + 1
    return out


# ------------------------------------------------------------------ per style cells
def style_cells(style: str, colour, zone, settings, rng, centre, mm):
    """-> (cells (H, W) int64, info). Stones are later cut by colour and split into parts."""
    h, w = colour.shape
    vp = settings.vocab
    gap = float(settings.gap_mm)
    smin = float(getattr(settings, "min_stone_mm", 4.0))
    t_min = smin + gap + 2.3          # smallest nominal stone that still prints >= smin wide
    cells = np.zeros((h, w), np.int64)
    present = [int(v) for v in np.unique(zone)]
    cx, cy = centre

    def tpx(v, floor_mm=t_min):
        return max(vp[v].tile_mm if v in vp else 9.6, floor_mm) / mm

    if style == "klasik":
        def ray_ring():
            e = [0.0, 40.0 + rng.uniform(0, 14)]
            while e[-1] < 4000:
                rr = e[-1]
                e.append(rr + (4.0 + 0.10 * rr) * rng.uniform(0.85, 1.15))
            return e

        for v in present:
            m = zone == v
            if v == Vocab.RAYS:
                t = polar_planks(colour, m, centre, mm, rng, ray_ring, lambda r: 4.8 + 0.03 * r)
            elif v == Vocab.FOCAL:
                fc = _focal_centre(m, centre)
                t = polar_planks(colour, m, fc, mm, rng,
                                 lambda: [0.0, 6.0] + [6.0 + 7.4 * i for i in range(1, 400)],
                                 lambda r: np.full_like(r, 7.4), brick=True)
            elif v == Vocab.BLOCKS:
                s = tpx(v)
                yy, xx = np.mgrid[0:h, 0:w]
                t = (yy // s).astype(np.int64) * 100000 + (xx // s).astype(np.int64) + 1
            elif v == Vocab.FLAT:
                t = np.ones((h, w), np.int64)
            else:
                floor = 7.5 if v == Vocab.FIGURE else t_min
                t = _voronoi_mask(m, tpx(v, floor), rng, lloyd=2)
            cells[m] = t[m] + v * 10 ** 12
    elif style == "smalti":
        cells = _smalti_rows(colour, zone, mm, rng)
    elif style == "freud":
        ch = 14.5 / mm
        yy, xx = np.mgrid[0:h, 0:w]
        row = (yy // ch).astype(np.int64)
        nrow = int(row.max()) + 1
        edges = []
        for _ in range(nrow):
            x = -rng.uniform(0, 14)
            e = [x]
            while x < w * mm + 20:
                x += rng.uniform(12.5, 17.5)
                e.append(x)
            edges.append(np.array(e) / mm)
        col = np.zeros((h, w), np.int64)
        for rr in range(nrow):
            sel = row == rr
            col[sel] = np.searchsorted(edges[rr], xx[sel], side="right")
        cells = row * 100000 + col + 1
    elif style == "orman":
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.hypot(xx - cx, yy - cy) * mm
        th = np.arctan2(yy - cy, xx - cx)
        rw = r + 2.2 * np.sin(th * 9.0 + r * 0.05) + 1.5 * _noise((h, w), 25 * 0.4 / mm, rng)
        for c in range(int(colour.max()) + 1):
            cc, nc = ndi.label(colour == c)
            offc = rng.uniform(0, 1, nc + 1)
            rw = np.where(colour == c, rw + offc[cc] * 30.0 * (0.4 + r / 400.0), rw)
        edges = [0.0]
        while edges[-1] < 4000:
            edges.append(edges[-1] * 1.28 + 26)
        ring = np.digitize(rw, edges)
        ang = np.floor((th + math.pi) / (2 * math.pi / 34) + 0.5 * (ring % 2)).astype(np.int64)
        rays = ring.astype(np.int64) * 1000 + ang + 1
        sg = 20 * 0.4 / mm
        wy, wx = (2.0 / mm) * _noise((h, w), sg, rng), (2.0 / mm) * _noise((h, w), sg, rng)
        sg3 = 45 * 0.4 / mm
        wy3, wx3 = (7.0 / mm) * _noise((h, w), sg3, rng), (7.0 / mm) * _noise((h, w), sg3, rng)
        for v in present:
            m = zone == v
            if v == Vocab.RAYS:
                t = rays
            elif v == Vocab.BLOCKS:
                b = 10.0 / mm
                t = (np.floor(yy / b) * 100000 + np.floor(xx / b)).astype(np.int64) + 1
            elif v == Vocab.FOCAL:
                t = _voronoi_mask(m, 19.0 / mm, rng, lloyd=0, warp=(wy, wx))
            elif v == Vocab.FIGURE:
                t = _voronoi_mask(m, 34.0 / mm, rng, lloyd=0, warp=(wy3, wx3))
            elif v == Vocab.FLAT:
                t = np.ones((h, w), np.int64)
            else:
                t = _voronoi_mask(m, 13.0 / mm, rng, lloyd=0, warp=(wy, wx))
            cells[m] = t[m] + v * 10 ** 12
    else:
        raise ValueError(f"Bilinmeyen taş stili: {style!r}. Geçerli: {', '.join(STYLES)}.")
    return cells


def _focal_centre(m, centre):
    """Sun centre: the ray centre when it lies inside this focal zone, else its centroid."""
    h, w = m.shape
    x, y = int(round(centre[0])), int(round(centre[1]))
    if 0 <= x < w and 0 <= y < h and m[y, x]:
        return centre
    ys, xs = np.nonzero(m)
    return (float(xs.mean()), float(ys.mean()))


def _smalti_rows(colour, zone, mm, rng):
    """V2: rows following each colour region's outline, cut every ~lp along the row."""
    h, w = colour.shape
    cells = np.zeros((h, w), np.int64)
    k = int(colour.max()) + 1
    comp = np.zeros((h, w), np.int64)
    nxt = 0
    for c in range(k):
        cc, n = ndi.label(colour == c)
        comp[cc > 0] = cc[cc > 0] + nxt
        nxt += n
    # distance (mm) to the colour edge or the image border
    b = np.zeros((h, w), bool)
    b[:, 1:] |= comp[:, 1:] != comp[:, :-1]
    b[:, :-1] |= comp[:, 1:] != comp[:, :-1]
    b[1:, :] |= comp[1:, :] != comp[:-1, :]
    b[:-1, :] |= comp[1:, :] != comp[:-1, :]
    b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = True
    d = ndi.distance_transform_edt(~b) * mm
    # row height / plank length by the region's main zone
    zc = np.bincount((comp * 8 + zone).ravel(), minlength=(nxt + 1) * 8).reshape(nxt + 1, 8)
    zmain = zc.argmax(1)
    fine = np.isin(zmain, [int(Vocab.FIGURE), int(Vocab.FOCAL)])
    rh = np.where(fine, 7.4, 8.2)[comp]
    lp = np.where(fine, 10.6, 12.2)[comp]
    row = np.floor(d / rh).astype(np.int64)
    # seeds on each row's centre line, ~lp apart (greedy along a random order)
    mid = np.abs(d - (row + 0.5) * rh) < 0.5 * mm + 0.25
    ys, xs = np.nonzero(mid)
    key = comp[ys, xs] * 4096 + row[ys, xs]
    perm = rng.permutation(len(ys))
    ys, xs, key = ys[perm], xs[perm], key[perm]
    order = np.argsort(key, kind="stable")
    ys, xs, key = ys[order], xs[order], key[order]
    cut = np.nonzero(np.diff(key))[0] + 1
    seeds_y, seeds_x, seeds_k = [], [], []
    for a, bnd in zip(np.r_[0, cut], np.r_[cut, len(key)]):
        gy, gx = ys[a:bnd], xs[a:bnd]
        # random sequential picking along a line leaves gaps of r..2r (mean ~1.33 r): aim for lp
        sp = 0.65 * float(lp[gy[0], gx[0]]) / mm
        chosen = []
        taken = np.zeros(len(gy), bool)
        tree = cKDTree(np.c_[gy, gx]) if len(gy) > 1 else None
        for i in range(len(gy)):
            if taken[i]:
                continue
            chosen.append(i)
            if tree is not None:
                taken[tree.query_ball_point([gy[i], gx[i]], sp * rng.uniform(0.85, 1.15))] = True
        seeds_y.extend(gy[chosen].tolist())
        seeds_x.extend(gx[chosen].tolist())
        seeds_k.extend([key[a]] * len(chosen))
    seeds_y, seeds_x, seeds_k = map(np.asarray, (seeds_y, seeds_x, seeds_k))
    # every pixel -> nearest seed of its own (region, row)
    py, px = np.nonzero(np.ones((h, w), bool))
    pk = comp[py, px] * 4096 + row[py, px]
    o = np.argsort(pk, kind="stable")
    so = np.argsort(seeds_k, kind="stable")
    seeds_y, seeds_x, seeds_k = seeds_y[so], seeds_x[so], seeds_k[so]
    uk, s_start = np.unique(seeds_k, return_index=True)
    s_end = np.r_[s_start[1:], len(seeds_k)]
    pk_s = pk[o]
    p_cut = np.r_[0, np.nonzero(np.diff(pk_s))[0] + 1, len(pk_s)]
    flat = np.zeros(h * w, np.int64)
    pos = np.searchsorted(uk, pk_s[p_cut[:-1]])
    for gi, (a, bnd) in enumerate(zip(p_cut[:-1], p_cut[1:])):
        idx = o[a:bnd]
        kk = pk_s[a]
        p = pos[gi]
        if p < len(uk) and uk[p] == kk:
            sa, sb = s_start[p], s_end[p]
            if sb - sa == 1:
                j = np.zeros(len(idx), np.int64)
            else:
                _, j = cKDTree(np.c_[seeds_y[sa:sb], seeds_x[sa:sb]]).query(np.c_[py[idx], px[idx]])
            flat[idx] = kk * 4096 + j + 1
        else:
            flat[idx] = kk * 4096          # a sliver row without a seed: one stone
    cells = flat.reshape(h, w)
    return cells


# ------------------------------------------------------------------ relief per style
def _tilt(rng, deg, n):
    a = rng.uniform(0, 2 * math.pi, n)
    t = math.tan(math.radians(deg)) * np.sqrt(rng.uniform(0.15, 1.0, n))
    return t * np.cos(a), t * np.sin(a)


def _person_dome(zone, mm):
    fig = zone == Vocab.FIGURE
    dome = np.zeros(zone.shape, np.float32)
    if not fig.any():
        return dome
    cc, n = ndi.label(fig)
    d = ndi.distance_transform_edt(fig) * mm
    area = np.bincount(cc.ravel(), minlength=n + 1) * mm * mm
    R = np.clip(0.3 * np.sqrt(area), 30.0, 70.0)
    dome = np.sqrt(np.clip(d / R[cc], 0, 1)).astype(np.float32) * fig
    return dome


def style_relief(style, settings, colour, zone, lab, tz, tc, cx, cy, mm, rng, pal):
    """Per tile: a (height at centroid, mm), b, c (top slope, mm/mm in raster x / y-down),
    profile (0 flat, 1 ramp, 2 quarter-round), direction (radians, raster coords, the way the top
    falls from high to low), hi/lo (profile heights)."""
    n = len(tz)
    h, w = lab.shape
    b = np.zeros(n)
    c = np.zeros(n)
    prof = np.zeros(n, np.int8)
    dirs = np.zeros(n)
    lum = pal @ np.array([0.30, 0.59, 0.11], np.float32)
    order = np.argsort(lum)
    darkest, lightest = int(order[0]), int(order[-1])
    pix = (np.clip(np.rint(cy).astype(int), 0, h - 1), np.clip(np.rint(cx).astype(int), 0, w - 1))
    edge = np.zeros(colour.shape, bool)
    edge[:, 1:] |= colour[:, 1:] != colour[:, :-1]
    edge[1:, :] |= colour[1:, :] != colour[:-1, :]
    dreg = ndi.distance_transform_edt(~edge) * mm
    dome = _person_dome(zone, mm)
    s1 = 3.0 * 0.4 / mm
    if style == "klasik":
        fig = ndi.gaussian_filter(dome * 4.2 + np.sqrt(np.clip(dreg / 9.0, 0, 1)) * 1.6, s1)
        fig = np.where(zone == Vocab.FIGURE, fig, 0).astype(np.float32)
        gy, gx = np.gradient(fig, mm)
        tx, ty = _tilt(rng, 3.0, n)
        a = 3.0 + rng.uniform(-0.6, 0.6, n)
        isf = tz == Vocab.FIGURE
        a = np.where(isf, 4.0 + fig[pix] + rng.uniform(-0.4, 0.4, n), a)
        a = np.where(tz == Vocab.FOCAL, 4.6 + rng.uniform(-0.6, 0.6, n), a)
        a = np.where(tz == Vocab.BLOCKS, 3.6 + rng.uniform(-0.5, 0.5, n), a)
        b = np.where(isf, gx[pix] * 0.8, 0) + tx
        c = np.where(isf, gy[pix] * 0.8, 0) + ty
    elif style == "smalti":
        tx, ty = _tilt(rng, 5.0, n)
        a = 3.2 + rng.uniform(-0.5, 0.5, n) + np.where(tz == Vocab.FIGURE, 1.4, np.where(tz == Vocab.FOCAL, 0.8, 0.0))
        b, c = tx, ty
    elif style == "freud":
        pillow = np.clip(dreg / 22.0, 0, 1) ** 0.6
        F = 0.5 * pillow + np.where(zone == Vocab.FIGURE, dome, 0) + np.where(zone == Vocab.FOCAL, 0.45, 0)
        F = ndi.gaussian_filter(F.astype(np.float32), 5 * 0.4 / mm)
        gy, gx = np.gradient(F, mm)
        fmax = float(F.max()) or 1.0
        f = F[pix] / fmax
        gxx, gyy = gx[pix], gy[pix]
        gm = np.hypot(gxx, gyy)
        fall = np.arctan2(-gyy, -gxx) + rng.normal(0, 0.35, n)
        woven = rng.integers(0, 4, n) * (math.pi / 2) + rng.normal(0, 0.15, n)
        dirs = np.where(gm > 0.004, fall, woven)
        hi = np.minimum(4.0 + 6.0 * f + rng.uniform(-0.6, 0.6, n), 10.0)
        lo = np.maximum(2.0, hi * rng.uniform(0.25, 0.42, n))
        prof = np.where(rng.random(n) < 0.72, PROFILE_QUARTER, PROFILE_RAMP).astype(np.int8)
        tx, ty = _tilt(rng, 2.0, n)
        a = (hi + lo) / 2
        # mean plane of the profile (for exporters that ignore the profile): fall over ~13 mm
        sl = (hi - lo) / 13.0
        b = -sl * np.cos(dirs) + tx
        c = -sl * np.sin(dirs) + ty
        return a, b, c, prof, dirs, hi, lo
    else:  # orman
        cream = tc == lightest
        black = tc == darkest
        a = np.select([tz == Vocab.FIGURE, tz == Vocab.FOCAL, tz == Vocab.BLOCKS, tz == Vocab.STONES],
                      [9.0 + np.where(cream, 0.9, np.where(black, -0.4, 0.0)) + rng.uniform(-0.25, 0.25, n),
                       6.4 + np.where(cream, 1.3, 0.0) + rng.uniform(-0.15, 0.15, n),
                       5.6 + np.where(cream, 0.8, 0.0) + rng.uniform(-0.5, 0.5, n),
                       5.0 + rng.uniform(-0.6, 0.6, n)],
                      np.where(cream, 3.8, 2.6) + rng.uniform(-0.2, 0.2, n))
        tilt = np.select([tz == Vocab.FIGURE, tz == Vocab.FOCAL, tz == Vocab.BLOCKS, tz == Vocab.STONES],
                         [1.6, 1.2, 1.5, 2.5], 1.2)
        tx, ty = _tilt(rng, 1.0, n)
        b, c = tx * tilt, ty * tilt
    return a, b, c, prof, dirs, a, a
