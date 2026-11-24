"""Automatic zone suggestion without AI models.

suggest_zones(img, settings) -> (zone_map, info)

Works on a small copy of the picture (long side ~420 px) with classic image cues:

- RAYS  : a sunburst. Every strong edge votes along its own direction (a Hough-like
          accumulator); a sunburst's edges all pass through one point -> sharp peak = centre.
          Pixels whose edges point at that centre form the ray zone.
- BLOCKS: rectilinear, building-like areas: dense edges that are almost all horizontal or
          vertical, with both directions present (window grids, facades).
- FIGURE: the subject: colour saliency + dense edges of every orientation (faces, clothing),
          weighted by a centre/bottom prior; the largest such composition(s).
- FOCAL : a small round focal object: the disc at the ray centre (sun) when there is a
          sunburst, otherwise the strongest bright round blob.

The result is only a starting point; the user corrects it with the brush. Reasons are
returned in Turkish in ``info["nedenler"]``.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from ..model import Settings, Vocab, VOCAB_NAMES_TR
from .palette import rgb_to_lab

WORK_SIDE = 600
SUNBURST_MIN = 0.15   # share of edges pointing at one centre needed to call it a sunburst


def _small(img: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    f = min(1.0, WORK_SIDE / max(h, w))
    if f < 1.0:
        im = Image.fromarray(img[:, :, :3]).resize((max(8, int(w * f)), max(8, int(h * f))), Image.BOX)
        return np.asarray(im), f
    return img[:, :, :3], 1.0


def _gradients(L: np.ndarray):
    g = ndi.gaussian_filter(L, 0.7)
    gy, gx = ndi.sobel(g, 0), ndi.sobel(g, 1)
    mag = np.hypot(gx, gy)
    return gx, gy, mag


def detect_sunburst(L: np.ndarray, gx, gy, mag, rng) -> tuple[Optional[tuple[float, float]], float]:
    """Hough-like voting: each strong edge pixel votes along its edge direction.
    Returns (centre (x, y) in px of ``L``, strength = peak / 99.9th percentile)."""
    h, w = L.shape
    thr = np.percentile(mag, 90)
    ys, xs = np.nonzero(mag > max(thr, 1e-3))
    if len(ys) < 200:
        return None, 0.0
    sel = rng.choice(len(ys), min(8000, len(ys)), replace=False)
    ys, xs = ys[sel], xs[sel]
    m = mag[ys, xs]
    dx, dy = -gy[ys, xs] / m, gx[ys, xs] / m          # edge direction (perpendicular to gradient)
    # extend the vote area a bit beyond the frame (a sun may sit on the edge)
    pad = 0.25
    f = 2
    H, W = int(h * (1 + 2 * pad) / f) + 1, int(w * (1 + 2 * pad) / f) + 1
    L_ = 1.4 * max(h, w)
    ts = np.arange(-L_, L_, f * 0.7, dtype=np.float32)
    px = (xs[:, None] + dx[:, None] * ts[None] + w * pad) / f
    py = (ys[:, None] + dy[:, None] * ts[None] + h * pad) / f
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    acc = np.bincount(py[ok].astype(np.int64) * W + px[ok].astype(np.int64), minlength=H * W)
    acc = ndi.gaussian_filter(acc.reshape(H, W).astype(np.float32), 1.0)
    cy, cx = np.unravel_index(int(acc.argmax()), acc.shape)
    centre = (cx * f + f / 2 - w * pad, cy * f + f / 2 - h * pad)
    # strength = share of edge pixels whose edge line passes close to that centre
    # (random edges: a few %, a sunburst: a large share of all edges)
    d = np.abs((centre[0] - xs) * dy - (centre[1] - ys) * dx)
    strength = float(np.mean(d < 0.015 * max(h, w)))
    return centre, strength


def _box(a: np.ndarray, s: int) -> np.ndarray:
    return ndi.uniform_filter(a.astype(np.float32), s)


def _clean(mask: np.ndarray, close: int, min_frac: float) -> np.ndarray:
    if not mask.any():
        return mask
    m = ndi.binary_closing(mask, np.ones((close, close), bool), border_value=0)
    m = ndi.binary_opening(m, np.ones((max(3, close // 2),) * 2, bool))
    m = ndi.binary_fill_holes(m)
    cc, n = ndi.label(m)
    if n == 0:
        return m
    sz = np.bincount(cc.ravel())
    sz[0] = 0
    keep = sz >= min_frac * m.size
    return keep[cc]


def suggest_zones(img: np.ndarray, settings: Optional[Settings] = None, *, seed: int = 7) -> tuple[np.ndarray, dict]:
    """-> (zone_map (H, W) int32 at the size of ``img``, info dict with Turkish reasons).

    info keys: ``nedenler`` (list of Turkish sentences), ``isin_merkezi`` ((x, y) as fractions of
    width/height, or None), ``varsayilan`` (default vocab int), ``kapsama`` ({vocab name: share})."""
    H0, W0 = img.shape[:2]
    small, f = _small(np.asarray(img))
    h, w = small.shape[:2]
    rng = np.random.default_rng(seed)
    L = small.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    gx, gy, mag = _gradients(L)
    strong = mag > max(np.percentile(mag, 80), 8.0)
    win = max(9, int(round(max(h, w) / 22)) | 1)
    reasons: list[str] = []
    zone = np.full((h, w), int(Vocab.STONES), np.int32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = np.arctan2(gy, gx)

    # ---------------------------------------------------------------- RAYS
    centre, strength = detect_sunburst(L, gx, gy, mag, rng)
    rays = np.zeros((h, w), bool)
    centre_frac = None
    if centre is not None and strength > SUNBURST_MIN:
        rx, ry = xx - centre[0], yy - centre[1]
        rr = np.hypot(rx, ry) + 1e-6
        # an edge "points at the centre" when its gradient is perpendicular to the radius
        cosg = np.abs((gx * rx + gy * ry) / (rr * np.maximum(mag, 1e-6)))
        radial = strong & (cosg < 0.22)
        frac = _box(radial, win) / np.maximum(_box(strong, win), 1e-3)
        dens = _box(strong, win)
        rays = (frac > 0.55) & (dens > 0.03)
        rays = _clean(rays, win, 0.01)
        # keep only ray areas that reach towards the centre (dress folds elsewhere also look radial)
        cc, n = ndi.label(rays)
        if n:
            dmin = ndi.minimum(np.hypot(xx - centre[0], yy - centre[1]), cc, np.arange(1, n + 1))
            ok = np.concatenate([[False], np.asarray(dmin) < 0.3 * max(h, w)])
            rays = ok[cc]
        share = rays.mean()
        if share > 0.04:
            centre_frac = (float(centre[0] / w), float(centre[1] / h))
            zone[rays] = int(Vocab.RAYS)
            reasons.append(f"{VOCAB_NAMES_TR[Vocab.RAYS]}: kenarların çoğu tek bir noktada buluşuyor "
                           f"(merkez yaklaşık %{centre_frac[0] * 100:.0f} yatay, %{centre_frac[1] * 100:.0f} dikey). "
                           f"Bu bir güneş ışını deseni; alanın %{share * 100:.0f}'i ışın olarak işaretlendi.")
        else:
            rays[:] = False

    # ---------------------------------------------------------------- BLOCKS
    # facades: vertical edges (walls, window columns) plus one other dominant, parallel family
    # (floors and window rows; sloped when the building is drawn in perspective)
    vert = strong & (np.abs(np.cos(ang)) > math.cos(math.radians(8)))
    other = strong & ~vert
    sv, so, ss = _box(vert, win), _box(other, win), _box(strong, win)
    c2 = _box(other * np.cos(2 * ang), win)
    s2 = _box(other * np.sin(2 * ang), win)
    coh = np.hypot(c2, s2) / np.maximum(so, 1e-3)
    axis = vert | (other & (coh > 0.6))
    # the second family must run roughly across (floors, window rows: within ~35 deg of horizontal,
    # perspective allowed); a single family of diagonal stripes (hair, folds) is not a facade
    phi = 0.5 * np.arctan2(s2, c2)          # mean gradient direction of the other family
    across = np.abs(np.sin(phi)) > math.cos(math.radians(35))
    blocks = (sv / np.maximum(ss, 1e-3) > 0.2) & (sv > 0.025) & (so > 0.025) & (coh > 0.6) & across & ~rays
    blocks = _clean(blocks, max(5, win // 2), 0.004)
    if blocks.mean() > 0.01:
        zone[blocks] = int(Vocab.BLOCKS)
        n_b = int(ndi.label(blocks)[1])
        reasons.append(f"{VOCAB_NAMES_TR[Vocab.BLOCKS]}: {n_b} bölgede dik duvar çizgileri ve onlara dik sıralar var "
                       f"(bina, pencere ızgarası gibi); alanın %{blocks.mean() * 100:.0f}'i.")

    # ---------------------------------------------------------------- FIGURE
    lab = rgb_to_lab(small)
    lab_s = np.stack([ndi.gaussian_filter(lab[..., c], 2.0) for c in range(3)], -1)
    sal = np.sqrt(((lab_s - lab_s.reshape(-1, 3).mean(0)) ** 2).sum(-1))
    sal /= max(np.percentile(sal, 99), 1e-6)
    # edges of every orientation (not explained by rays or a grid): faces, hands, folds
    free = strong & ~axis
    if centre is not None and rays.any():
        free &= ~(cosg < 0.22)
    busy = _box(free, win) / max(_box(free, win).max(), 1e-6)
    prior = np.exp(-((xx / w - 0.5) ** 2) / (2 * 0.35 ** 2)) * (0.55 + 0.45 * yy / h)
    score = (0.45 * np.clip(sal, 0, 1.5) + 0.75 * busy) * prior
    score = ndi.gaussian_filter(score, win / 3)
    score[blocks] *= 0.6
    score[rays] *= 0.5
    thr = np.percentile(score, 68)
    fig = _clean(score > thr, win, 0.02)
    if fig.any():
        # keep the largest one or two compositions
        cc, n = ndi.label(fig)
        sz = np.bincount(cc.ravel())
        sz[0] = 0
        keep = np.argsort(-sz)[:3]
        keep = [k for k in keep if sz[k] > 0.15 * sz.max() and k > 0]
        fig = np.isin(cc, keep)
    if blocks.any() and fig.any():
        # facades never sit inside a figure (hair stripes, clothing folds)
        cc, n = ndi.label(blocks)
        inside = np.bincount(cc.ravel(), fig.ravel(), n + 1) / np.maximum(np.bincount(cc.ravel(), minlength=n + 1), 1)
        drop = inside > 0.3
        drop[0] = False
        if drop.any():
            gone = drop[cc]
            zone[gone] = int(Vocab.STONES)
            blocks &= ~gone
            if not blocks.any():
                reasons = [r for r in reasons if not r.startswith(VOCAB_NAMES_TR[Vocab.BLOCKS])]
    if fig.mean() > 0.03:
        zone[fig] = int(Vocab.FIGURE)
        reasons.append(f"{VOCAB_NAMES_TR[Vocab.FIGURE]}: görselin ortasında ve altında yoğun, her yöne giden çizgiler ve "
                       f"çevresinden ayrışan renkler var (yüz, el, giysi gibi); alanın %{fig.mean() * 100:.0f}'i "
                       f"öne çıkan figür olarak işaretlendi.")
    else:
        fig[:] = False

    # ---------------------------------------------------------------- FOCAL
    focal = np.zeros((h, w), bool)
    if centre_frac is not None and 0 <= centre[0] < w and 0 <= centre[1] < h:
        # the sun: the disc around the ray centre inside which edges stop being radial
        rr_c = np.hypot(xx - centre[0], yy - centre[1])
        rbins = (rr_c / 2).astype(int)
        nb = int(rbins.max()) + 1
        rad_frac = np.bincount(rbins.ravel(), (rays & ~fig).ravel(), nb) / np.maximum(np.bincount(rbins.ravel(), minlength=nb), 1)
        r0 = 0
        for i in range(2, nb):
            if rad_frac[i] > 0.5:
                r0 = i * 2
                break
        r0 = int(np.clip(r0, 0.03 * max(h, w), 0.2 * max(h, w)))
        focal = rr_c < r0
        reasons.append(f"{VOCAB_NAMES_TR[Vocab.FOCAL]}: ışınların çıktığı yuvarlak nesne (güneş), yarıçapı yaklaşık %{r0 / w * 100:.0f} genişlik; "
                       f"en yüksek kabartma burada olacak.")
    else:
        # a compact, round, high-contrast object that is not part of a figure: candidates are the
        # strongest Laplacian-of-Gaussian blobs (bright or dark) over a few scales; a candidate
        # must be one even colour inside and clearly different from the ring around it
        cands = []
        for sig in np.array([0.02, 0.03, 0.045, 0.065]) * max(h, w):
            resp = np.abs(ndi.gaussian_laplace(L, sig)) * sig ** 2
            pk = (resp == ndi.maximum_filter(resp, size=int(2 * sig) | 1)) & (resp > 25)
            ys_, xs_ = np.nonzero(pk)
            for y_, x_ in zip(ys_, xs_):
                cands.append((float(resp[y_, x_]), int(x_), int(y_), float(sig)))
        cands.sort(reverse=True)
        lab_px = lab
        for val, cx_, cy_, sig in cands[:40]:
            r0 = 1.414 * sig
            if cx_ < r0 or cy_ < r0 or cx_ > w - r0 or cy_ > h - r0:
                continue
            rr_c = np.hypot(xx - cx_, yy - cy_)
            disc, ring = rr_c < 0.8 * r0, (rr_c >= 1.1 * r0) & (rr_c < 1.6 * r0)
            if fig[disc].mean() > 0.3:
                continue
            mean_in = lab_px[disc].mean(0)
            even = (np.sqrt(((lab_px[disc] - mean_in) ** 2).sum(-1)) < 18).mean()
            contrast = np.sqrt(((lab_px[ring].mean(0) - mean_in) ** 2).sum())
            if even > 0.7 and contrast > 25:
                focal = rr_c < r0
                reasons.append(f"{VOCAB_NAMES_TR[Vocab.FOCAL]}: tek renkli, yuvarlak ve çevresinden ayrışan küçük "
                               f"bir nesne bulundu; en yüksek kabartma onun olacak.")
                break
    if focal.any():
        zone[focal] = int(Vocab.FOCAL)

    if not reasons:
        reasons.append(f"Belirgin ışın, bina ya da figür bulunamadı; her yer '{VOCAB_NAMES_TR[Vocab.STONES]}' olarak bırakıldı. "
                       "Fırçayla istediğin bölgeyi değiştirebilirsin.")
    # default vocabulary = the one covering most of the background
    varsayilan = int(Vocab.RAYS) if (zone == Vocab.RAYS).mean() > 0.3 else int(Vocab.STONES)
    out = np.asarray(Image.fromarray(zone.astype(np.uint8)).resize((W0, H0), Image.NEAREST)).astype(np.int32)
    kaps = {VOCAB_NAMES_TR[Vocab(v)]: round(float((out == v).mean()), 3) for v in np.unique(out)}
    info = {"nedenler": reasons, "isin_merkezi": centre_frac, "varsayilan": varsayilan, "kapsama": kaps}
    return out, info
