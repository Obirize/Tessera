"""R6-3: no stone narrower than ``min_stone_mm``, and no lost details.

Colour features that are too thin to become their own stone are classified as "lines" and handled
with mosaic craft instead of a dark backing:

- "row":    the feature is widened into one row of stones at least min_stone (+ gap) wide that
            follows it (opus vermiculatum outline). Dots become one small round stone.
- "groove": the feature's colour is given to the stones around it, and its centre line is recorded
            as a carved V-groove on those stones (``Mosaic.grooves``); it reads as a shadow line.
- "step":   reserved (a height step between stones); currently handled as "row".
- "auto":   dark, elongated features (ink lines: eyelids, lips, outlines) -> groove; light features
            (highlights) and short blobs (pupils, nostrils) -> row.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage as ndi

_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def _thin(mask: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """Zhang-Suen thinning (vectorised) -> 1 px skeleton."""
    img = np.pad(mask.astype(np.uint8), 1)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            P = [img[1 + dy:img.shape[0] - 1 + dy, 1 + dx:img.shape[1] - 1 + dx] for dy, dx in
                 [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]]
            c = img[1:-1, 1:-1]
            B = sum(p.astype(np.int16) for p in P)
            seq = P + [P[0]]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int16) for i in range(8))
            p2, p4, p6, p8 = P[0], P[2], P[4], P[6]
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = (c == 1) & (B >= 2) & (B <= 6) & (A == 1) & cond
            if rm.any():
                c[rm] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1].astype(bool)


def _trace(skel: np.ndarray) -> list[list[tuple[int, int]]]:
    """Skeleton pixels -> polylines (lists of (y, x)), walking from end points, then loops."""
    pts = set(zip(*np.nonzero(skel)))
    if not pts:
        return []

    def nbrs(p):
        return [(p[0] + dy, p[1] + dx) for dy, dx in _N8 if (p[0] + dy, p[1] + dx) in pts]

    deg = {p: len(nbrs(p)) for p in pts}
    seen: set = set()
    lines = []
    starts = [p for p in pts if deg[p] == 1] + [p for p in pts if deg[p] != 1]
    for s in starts:
        if s in seen:
            continue
        line = [s]
        seen.add(s)
        cur = s
        while True:
            nx = [q for q in nbrs(cur) if q not in seen]
            if not nx:
                break
            # prefer 4-neighbours (straighter walk)
            nx.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
            cur = nx[0]
            seen.add(cur)
            line.append(cur)
        if len(line) >= 2:
            lines.append(line)
    return lines


def classify_lines(colour: np.ndarray, pal: np.ndarray, min_px: int):
    """Colour components whose widest inscribed width is below ``min_px`` px.
    Returns list of dicts {c, sl (slice), mask (bool crop), width_px, length_px, dark}."""
    lum = pal @ np.array([0.30, 0.59, 0.11], np.float32)
    darkest = int(lum.argmin())
    feats = []
    for c in range(len(pal)):
        cc, n = ndi.label(colour == c)
        if n == 0:
            continue
        d = ndi.distance_transform_edt(cc > 0)
        mx = np.asarray(ndi.maximum(d, cc, np.arange(1, n + 1)))
        width = 2 * mx - 1
        thin_ids = np.nonzero(width < min_px)[0] + 1
        if thin_ids.size == 0:
            continue
        objs = ndi.find_objects(cc)
        area = np.bincount(cc.ravel(), minlength=n + 1)
        for i in thin_ids:
            sl = objs[i - 1]
            if sl is None:
                continue
            m = cc[sl] == i
            w = float(width[i - 1])
            feats.append({"c": c, "sl": sl, "mask": m, "width_px": max(w, 1.0),
                          "length_px": float(area[i]) / max(w, 1.0), "dark": c == darkest,
                          "lum": float(lum[c])})
    return feats


def handle_lines(colour: np.ndarray, pal: np.ndarray, mm: float, min_stone_mm: float, gap_mm: float,
                 mode: str = "auto", groove_depth_mm: float = 1.0):
    """Apply the line mode. Returns (colour, grooves list in mm (x right, y down from the top-left),
    groove depth raster (H, W) mm, stats dict)."""
    h, w = colour.shape
    min_px = int(np.ceil((min_stone_mm + gap_mm) / mm))
    feats = classify_lines(colour, pal, min_px)
    out = colour.copy()
    depth = np.zeros((h, w), np.float32)
    grooves = []
    n_row = n_groove = 0
    lum = pal @ np.array([0.30, 0.59, 0.11], np.float32)
    rng_l = float(lum.max() - lum.min()) or 1.0
    k = len(pal)
    # rows first (they need room), then grooves
    order = sorted(range(len(feats)), key=lambda i: feats[i]["length_px"] > 3 * feats[i]["width_px"])
    for i in order:
        f = feats[i]
        m_ = mode
        if m_ == "auto":
            elong = f["length_px"] >= 3.0 * f["width_px"] and f["length_px"] * mm >= 4.0
            dark = f["lum"] <= float(lum.min()) + 0.25 * rng_l
            m_ = "groove" if (elong and dark) else "row"
        if m_ == "step":
            m_ = "row"
        sl, m = f["sl"], f["mask"]
        y0, x0 = sl[0].start, sl[1].start
        if m_ == "row":
            # widen into one row of stones min_px wide (grow by the missing half width each side)
            r = int(np.ceil((min_px - f["width_px"]) / 2.0))
            pad = r + 1
            Y0, Y1 = max(0, y0 - pad), min(h, sl[0].stop + pad)
            X0, X1 = max(0, x0 - pad), min(w, sl[1].stop + pad)
            big = np.zeros((Y1 - Y0, X1 - X0), bool)
            big[y0 - Y0:y0 - Y0 + m.shape[0], x0 - X0:x0 - X0 + m.shape[1]] = m
            if r > 0:
                dd = ndi.distance_transform_edt(~big)
                big = dd <= r
            out[Y0:Y1, X0:X1][big] = f["c"]
            n_row += 1
        else:
            # groove: the feature takes the colour around it; its centre line is carved
            pad = 2
            Y0, Y1 = max(0, y0 - pad), min(h, sl[0].stop + pad)
            X0, X1 = max(0, x0 - pad), min(w, sl[1].stop + pad)
            big = np.zeros((Y1 - Y0, X1 - X0), bool)
            big[y0 - Y0:y0 - Y0 + m.shape[0], x0 - X0:x0 - X0 + m.shape[1]] = m
            sub = out[Y0:Y1, X0:X1]
            ring = ndi.binary_dilation(big, np.ones((3, 3), bool)) & ~big
            if ring.any():
                votes = np.bincount(sub[ring], minlength=k)
                votes[f["c"]] = 0
                fill = int(votes.argmax()) if votes.sum() else int(f["c"])
            else:
                fill = int(f["c"])
            # nearest non-feature colour for each pixel (keeps two-sided lines honest)
            _, (iy, ix) = ndi.distance_transform_edt(big, return_indices=True)
            nb = sub[iy, ix]
            nb = np.where(nb == f["c"], fill, nb)
            sub[big] = nb[big]
            skel = _thin(big)
            wmm = max(1.0, f["width_px"] * mm)
            dmm = min(groove_depth_mm, 0.8 * wmm)
            for line in _trace(skel):
                pts = [[round((x + X0 + 0.5) * mm, 2), round((y + Y0 + 0.5) * mm, 2)] for y, x in line]
                grooves.append({"pts": pts, "width": round(wmm, 2), "depth": round(dmm, 2), "renk": int(f["c"])})
            # V profile in the height raster
            if skel.any():
                dsk = ndi.distance_transform_edt(~skel) * mm
                v = np.clip(1.0 - dsk / (wmm / 2.0), 0, 1) * dmm
                depth[Y0:Y1, X0:X1] = np.maximum(depth[Y0:Y1, X0:X1], v.astype(np.float32))
            n_groove += 1
    return out, grooves, depth, {"cizgi_sira": n_row, "cizgi_oluk": n_groove}
