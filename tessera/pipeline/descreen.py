"""Descreening: turn printed halftone / screentone dot patterns into flat grey tone.

Comic ink art shades greys with regular dot screens. Quantised pixel by pixel,
a 30 % dot screen becomes black speckle on white and the grey tone is lost.
This module finds screened areas and low-passes *only those*, so that

- the dots average out to the grey they were meant to show, and
- solid ink lines (inside or outside screened areas) stay sharp.

Detection is local: a screen produces many tone transitions in both x and y
inside a small window (about 4 / period per pixel), while an ink line produces
only two per crossing. The dominant period is estimated from that density and
sets the blur radius (sigma = 0.55 period removes the dot frequency almost
completely). Ink inside screened areas is protected: dark connected components
much larger than a dot are copied back from the original.
"""
from __future__ import annotations

from typing import Union

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

DescreenMode = Union[str, bool, None]


def _lum(img: np.ndarray) -> np.ndarray:
    a = img.astype(np.float32)
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114 if a.ndim == 3 else a


def screen_map(img: np.ndarray, *, contrast: float = 24.0, window: int = 0) -> tuple[np.ndarray, float]:
    """Return (soft screen mask in [0, 1], estimated period px). Period is 0 when nothing found."""
    L = _lum(img)
    h, w = L.shape
    d = L - ndi.gaussian_filter(L, 2.0)
    zx = np.zeros((h, w), np.float32)
    zy = np.zeros((h, w), np.float32)
    zx[:, 1:] = ((np.sign(d[:, 1:]) != np.sign(d[:, :-1])) & (np.abs(L[:, 1:] - L[:, :-1]) > contrast))
    zy[1:, :] = ((np.sign(d[1:, :]) != np.sign(d[:-1, :])) & (np.abs(L[1:, :] - L[:-1, :]) > contrast))
    win = window or max(15, int(round(min(h, w) / 60)))
    tx = ndi.uniform_filter(zx, win)
    ty = ndi.uniform_filter(zy, win)
    t = tx + ty
    # a screen has transitions in both directions; an isolated ink line mostly in one
    both = np.minimum(tx, ty) / np.maximum(t, 1e-6)
    score = np.clip((t - 0.16) / 0.12, 0, 1) * np.clip((both - 0.18) / 0.12, 0, 1)
    score = ndi.uniform_filter(score, win)            # fill the dot centres, bridge small holes
    mask = np.clip((score - 0.25) / 0.35, 0, 1).astype(np.float32)
    sel = mask > 0.5
    period = float(np.median(4.0 / np.maximum(t[sel], 1e-3))) if sel.sum() > 50 else 0.0
    return mask, period


_DIRS = np.arange(16) * np.pi / 16


def _shift(m: np.ndarray, oy: int, ox: int) -> np.ndarray:
    """out[y, x] = m[y + oy, x + ox] (zero outside)."""
    h, w = m.shape
    out = np.zeros_like(m)
    ys, yd = (slice(oy, h), slice(0, h - oy)) if oy >= 0 else (slice(0, h + oy), slice(-oy, h))
    xs, xd = (slice(ox, w), slice(0, w - ox)) if ox >= 0 else (slice(0, w + ox), slice(-ox, w))
    out[yd, xd] = m[ys, xs]
    return out


def _line_score(mask: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Max over 16 directions of the fraction of ``mask`` on a centred segment of ``length`` px,
    and the index of the best direction."""
    length = max(3, length | 1)
    hl = length // 2
    m = mask.astype(np.float32)
    best = np.zeros(m.shape, np.float32)
    arg = np.zeros(m.shape, np.int8)
    for i, ang in enumerate(_DIRS):
        dy, dx = np.sin(ang), np.cos(ang)
        acc = np.zeros(m.shape, np.float32)
        for t in range(-hl, hl + 1):
            acc += _shift(m, int(round(t * dy)), int(round(t * dx)))
        acc /= length
        upd = acc > best
        best[upd] = acc[upd]
        arg[upd] = i
    return best, arg


def _isolated(line: np.ndarray, arg: np.ndarray, period: float) -> np.ndarray:
    """Keep line pixels with no parallel line about one period away (screens at ~50 % form
    periodic chains of touching dots; a real ink line stands alone)."""
    keep = line.copy()
    offs = sorted({int(round(f * period)) for f in (0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0)} - {0})
    for i, ang in enumerate(_DIRS):
        sel = line & (arg == i)
        if not sel.any():
            continue
        ny, nx = np.cos(ang), -np.sin(ang)    # normal of the line direction
        near = np.zeros(line.shape, bool)
        for o in offs:
            for sgn in (1, -1):
                near |= _shift(line, int(round(sgn * o * ny)), int(round(sgn * o * nx)))
        keep &= ~(sel & near)
    return keep


def descreen(img: np.ndarray, mode: DescreenMode = "auto", *, min_area_frac: float = 0.004,
             max_period_px: float = 16.0, return_mask: bool = False):
    """Low-pass halftone areas of ``img`` (uint8 RGB or grey). ``mode``: "auto" (only when a
    meaningful screened area is found), True (always apply the detected mask), False (no-op).

    ``max_period_px``: patterns coarser than this are real content (window grids, tiles), not a
    print screen. Callers pass ~5 mm expressed in pixels of ``img``: a dot pattern finer than
    that cannot be printed as separate pieces anyway."""
    if mode is False or mode is None or (isinstance(mode, str) and mode.lower() in ("false", "off", "kapali", "no")):
        return (img, np.zeros(img.shape[:2], np.float32)) if return_mask else img
    mask, period = screen_map(img)
    frac = float((mask > 0.5).mean())
    # coarse screens (dots ~6 mm apart): look again at half resolution, where the window holds
    # enough periods; keep whichever reading covers clearly more area
    h, w = img.shape[:2]
    if min(h, w) >= 200:
        half = np.asarray(Image.fromarray(img).resize((w // 2, h // 2), Image.BOX))
        m2, p2 = screen_map(half)
        f2 = float((m2 > 0.5).mean())
        if p2 > 0 and f2 > max(1.5 * frac, frac + 0.05) and 2 * p2 <= max_period_px:
            mask = np.asarray(Image.fromarray(m2).resize((w, h), Image.BILINEAR))
            period, frac = 2 * p2, f2
    if period <= 0 or period > max_period_px or (mode == "auto" and frac < min_area_frac):
        return (img, np.zeros(img.shape[:2], np.float32)) if return_mask else img
    period = float(np.clip(period, 2.0, max(2.0, max_period_px)))
    sigma = 0.55 * period
    a = img.astype(np.float32)
    # protect real ink: pixels on long straight-ish dark runs (lines) or in solid blacks. Screen
    # dots are short in every direction, so they are not protected.
    L = _lum(img)
    dark = L < 110
    local_dark = ndi.uniform_filter(dark.astype(np.float32), int(2 * period) | 1)
    score, arg = _line_score(dark, int(round(1.7 * period)))
    line = (score >= 0.86) & (local_dark < 0.6)
    solid = ndi.binary_opening(dark, np.ones((max(3, int(period * 1.6)),) * 2, bool))
    # real lines are sparse; a screen near 50 % also forms "lines" of touching dots, but in
    # regular parallel rows
    line = _isolated(line, arg, period)
    line &= ndi.uniform_filter(line.astype(np.float32), int(4 * period) | 1) < 0.13
    ink_core = line | solid
    core = dark & ink_core
    ink = ndi.binary_dilation(core, np.ones((3, 3), bool))   # + anti-alias fringe: kept as is
    # normalised low-pass over screened pixels without the ink core (ink does not darken the tone)
    hard = (mask > 0.5) & ~core
    wgt = ndi.gaussian_filter(hard.astype(np.float32), sigma)
    if a.ndim == 3:
        blur = np.stack([ndi.gaussian_filter(a[..., c] * hard, sigma) for c in range(a.shape[2])], -1)
        blur = blur / np.maximum(wgt, 1e-3)[..., None]
    else:
        blur = ndi.gaussian_filter(a * hard, sigma) / np.maximum(wgt, 1e-3)
    m = mask * (~core) * (wgt > 0.05)
    m = ndi.gaussian_filter(m, 0.7)
    m = m[..., None] if a.ndim == 3 else m
    out = a * (1 - m) + blur * m
    # anti-alias fringe of ink: a mix of ink and tone, never lighter than the tone
    fr = (ink & ~core & (mask > 0.5))
    fr = fr[..., None] if a.ndim == 3 else fr
    out = np.where(fr, np.minimum(a, blur), out)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return (out, mask) if return_mask else out
