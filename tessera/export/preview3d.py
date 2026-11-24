"""Small numpy z-buffer renderer for the real meshes of one panel (UI "Parçalar" view).

Orthographic camera looking down at the panel from an angle, flat shading. Triangles are
rasterised by dense barycentric sampling (no per-triangle Python loop), then a z-buffer keeps
the nearest sample per pixel.
"""
from __future__ import annotations

import math

import numpy as np


def render_meshes(meshes, size: int = 900, az_deg: float = -30.0, el_deg: float = 50.0,
                  light=(-0.45, 0.35, 0.82), bg=(236, 236, 240)) -> np.ndarray:
    """meshes: list of (vertices (n,3), triangles (m,3), rgb). Returns uint8 (size, size, 3)."""
    a, e = math.radians(az_deg), math.radians(el_deg)
    u = np.array([math.cos(e) * math.sin(a), -math.cos(e) * math.cos(a), math.sin(e)])
    f = -u
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    up = np.cross(r, f)
    allv = np.vstack([np.asarray(v, float) for v, _, _ in meshes])
    c0 = (allv.min(0) + allv.max(0)) / 2
    P = allv - c0
    X, Y = P @ r, P @ up
    scale = (size * 0.94) / max(np.ptp(X), np.ptp(Y), 1e-6)
    L = np.asarray(light, float)
    L /= np.linalg.norm(L)
    xs, ys, ds, cs = [], [], [], []
    rng = np.random.default_rng(1)
    for v, t, rgb in meshes:
        v = np.asarray(v, float) - c0
        t = np.asarray(t, np.int64)
        if len(t) == 0:
            continue
        sx = v @ r * scale + size / 2
        sy = size / 2 - v @ up * scale
        dz = v @ f
        tri = v[t]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        vis = (n @ f) < 0
        t = t[vis]
        n = n[vis]
        rgb_arr = np.asarray(rgb, float)
        if rgb_arr.ndim == 2:                       # per-triangle colours
            rgb_arr = rgb_arr[vis]
        if len(t) == 0:
            continue
        shade = np.clip(n @ L, 0, 1) * 0.72 + 0.28
        col = np.clip(shade[:, None] * (rgb_arr if rgb_arr.ndim == 2 else rgb_arr[None, :]), 0, 255)
        p0 = np.stack([sx[t[:, 0]], sy[t[:, 0]]], 1)
        p1 = np.stack([sx[t[:, 1]], sy[t[:, 1]]], 1)
        p2 = np.stack([sx[t[:, 2]], sy[t[:, 2]]], 1)
        e1, e2 = p1 - p0, p2 - p0
        area = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) / 2
        ns = np.clip(np.ceil(area * 3.0), 1, 200000).astype(np.int64) + 2
        idx = np.repeat(np.arange(len(t)), ns)
        r1 = rng.random(len(idx))
        r2 = rng.random(len(idx))
        flip = r1 + r2 > 1
        r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
        w0 = 1 - r1 - r2
        d0, d1, d2 = dz[t[:, 0]], dz[t[:, 1]], dz[t[:, 2]]
        X_ = w0 * p0[idx, 0] + r1 * p1[idx, 0] + r2 * p2[idx, 0]
        Y_ = w0 * p0[idx, 1] + r1 * p1[idx, 1] + r2 * p2[idx, 1]
        D_ = w0 * d0[idx] + r1 * d1[idx] + r2 * d2[idx]
        xs.append(X_)
        ys.append(Y_)
        ds.append(D_)
        cs.append(col[idx])
    img = np.empty((size, size, 3), np.uint8)
    img[:] = bg
    if not xs:
        return img
    X_ = np.concatenate(xs).astype(np.int64)
    Y_ = np.concatenate(ys).astype(np.int64)
    D_ = np.concatenate(ds)
    C_ = np.concatenate(cs)
    ok = (X_ >= 0) & (X_ < size) & (Y_ >= 0) & (Y_ < size)
    X_, Y_, D_, C_ = X_[ok], Y_[ok], D_[ok], C_[ok]
    order = np.argsort(-D_, kind="stable")          # far first, near last wins
    pix = Y_[order] * size + X_[order]
    flat = img.reshape(-1, 3)
    flat[pix] = C_[order].astype(np.uint8)
    # close single-pixel holes
    hole = np.all(img == np.asarray(bg, np.uint8), axis=2)
    if hole.any():
        from scipy import ndimage as ndi
        filled = np.stack([ndi.maximum_filter(np.where(hole, 0, img[:, :, k]), 3) for k in range(3)], 2)
        near = hole & (ndi.uniform_filter(hole.astype(np.float32), 3) * 9 <= 3.5)
        img[near] = filled[near]
    return img


def heightfield_mesh(m, x0, y0, x1, y1, backing_t: float = 1.2, max_cells: int = 220,
                     lattice_rgb=(40, 38, 36)):
    """Fast stand-in for a panel: the mosaic height raster as a coloured grid mesh (mm)."""
    mm = m.settings.mm_per_px
    H, W = m.label.shape
    c0, c1 = int(max(0, x0 / mm)), int(min(W, math.ceil(x1 / mm)))
    r0, r1 = int(max(0, H - y1 / mm)), int(min(H, math.ceil(H - y0 / mm)))
    step = max(1, int(math.ceil(max(c1 - c0, r1 - r0) / max_cells)))
    hgt = m.height[r0:r1:step, c0:c1:step].astype(np.float64)
    col = m.colour[r0:r1:step, c0:c1:step]
    gap = m.gap[r0:r1:step, c0:c1:step]
    h, w = hgt.shape
    pal = np.array([f.rgb for f in m.settings.palette], float)
    rows, cols = np.mgrid[0:h, 0:w]
    x = (c0 + cols * step + step / 2) * mm
    y = (H - (r0 + rows * step + step / 2)) * mm
    z = backing_t + hgt
    v = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    idx = np.arange(h * w).reshape(h, w)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    t = np.concatenate([np.column_stack([a, c, b]), np.column_stack([b, c, d])])
    rgb = pal[col[:-1, :-1].ravel()]
    rgb[gap[:-1, :-1].ravel()] = lattice_rgb
    rgb = np.concatenate([rgb, rgb])
    return v, t, rgb
