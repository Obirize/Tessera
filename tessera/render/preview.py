"""Relit previews of a height map (top view and oblique 'lying on a table' view).

render_top(m, az_deg, el_deg, *, preview_scale=1.0, light_size_deg=5.0)
render_oblique(top, m, elev_deg)   # works with a full or reduced-scale ``top``

Lighting model: Lambert + weak Blinn specular, ambient occlusion from a local height
average, and soft shadows from a light of finite angular size (penumbra grows with
the distance to the occluder). Shifts are clamped at the borders (no wrap-around,
which used to paint a shadow band along the top edge).
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from ..model import Mosaic

# R6-2: a real gap shows the wall / board behind the stones, never a dark backing
WALL_RGB = (178, 174, 166)


def _shift_clamped(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """out[y, x] = a[clip(y + dy), clip(x + dx)] (edge values repeated, never wrapped)."""
    h, w = a.shape
    ys = np.clip(np.arange(h) + dy, 0, h - 1)
    xs = np.clip(np.arange(w) + dx, 0, w - 1)
    return a[ys[:, None], xs[None, :]]


def _scaled(m: Mosaic, scale: float, lattice_rgb=WALL_RGB):
    """Height (mm), albedo (0..255) and mm-per-px at ``scale`` of the mosaic raster."""
    H = m.height.astype(np.float32)
    A = m.albedo(lattice_rgb).astype(np.float32)
    mm = m.settings.mm_per_px
    if scale >= 0.999:
        return H, A, mm
    h, w = H.shape
    nw, nh = max(8, int(round(w * scale))), max(8, int(round(h * scale)))
    Hs = np.asarray(Image.fromarray(H).resize((nw, nh), Image.BOX), np.float32)
    As = np.stack([np.asarray(Image.fromarray(A[..., c]).resize((nw, nh), Image.BOX), np.float32)
                   for c in range(3)], -1)
    return Hs, As, mm * w / nw


def render_top(m: Mosaic, az_deg: float = -130, el_deg: float = 40, amb: float = 0.30,
               lattice_rgb=None, *, preview_scale: float = 1.0, light_size_deg: float = 5.0,
               max_shadow_mm: float = 40.0) -> np.ndarray:
    """az: direction TO the light in image coords (deg, y down). -130 = upper-left.
    ``preview_scale`` < 1 renders a smaller image (e.g. 0.5 -> half size, ~4x faster) for
    interactive UI previews. ``light_size_deg`` = angular radius of the light (0 = hard shadows)."""
    if lattice_rgb is None:
        lattice_rgb = WALL_RGB
    H, A, mm = _scaled(m, preview_scale, lattice_rgb)
    H = H.astype(np.float64)
    az, el = math.radians(az_deg), math.radians(max(1.0, min(89.0, el_deg)))
    hd = np.array([math.cos(az), math.sin(az)])
    L = np.array([hd[0] * math.cos(el), hd[1] * math.cos(el), math.sin(el)])
    gy, gx = np.gradient(H, mm)
    n = np.sqrt(gx ** 2 + gy ** 2 + 1)
    nx, ny, nz = -gx / n, -gy / n, 1 / n
    diff = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)

    # soft shadows: steepest occluder slope towards the light vs. the light's elevation
    hmax = float(H.max())
    tan_el = math.tan(el)
    maxk = int(min(max_shadow_mm / mm, (hmax + 0.5) / max(tan_el * mm, 1e-6))) + 1
    slope = np.full(H.shape, -np.inf)
    for k in range(1, maxk + 1):
        sy, sx = int(round(k * hd[1])), int(round(k * hd[0]))
        np.maximum(slope, (_shift_clamped(H, sy, sx) - H) / (k * mm), out=slope)
    occ_ang = np.arctan(np.maximum(slope, -10.0))
    half = math.radians(max(light_size_deg, 0.01))
    vis = np.clip((el - occ_ang) / (2 * half) + 0.5, 0.0, 1.0)     # 1 = fully lit
    vis = ndi.uniform_filter(vis, 3)

    ao_r = max(3, int(round(4.4 / mm)) | 1)
    ao = 1 - np.clip((ndi.uniform_filter(H, ao_r) - H) * 0.09, 0, 0.45)
    hv = L + np.array([0, 0, 1.0])
    hv /= np.linalg.norm(hv)
    spec = 0.07 * np.clip(nx * hv[0] + ny * hv[1] + nz * hv[2], 0, 1) ** 30
    # sky fill: shadowed areas still receive some diffuse light (no pitch-black shadows)
    light = (amb * ao + 0.82 * diff * (0.2 + 0.8 * vis) * ao)[..., None]
    out = A / 255.0 * light + (spec * vis)[..., None]
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def render_oblique(top_rgb: np.ndarray, m: Mosaic, elev_deg: float = 45, side_dark: float = 0.5,
                   bg=(236, 234, 230)) -> np.ndarray:
    """Painter's-algorithm view of the piece lying on a table, seen from the front.
    ``top_rgb`` may be a reduced-scale render (preview_scale); the height map follows it."""
    H = m.height.astype(np.float32)
    mm = m.settings.mm_per_px
    if top_rgb.shape[:2] != H.shape:
        th, tw = top_rgb.shape[:2]
        mm = mm * H.shape[1] / tw
        H = np.asarray(Image.fromarray(H).resize((tw, th), Image.BOX), np.float32)
    e = math.radians(elev_deg)
    h, w = H.shape
    hp = H / mm * math.cos(e)
    base = np.arange(h) * math.sin(e)
    off = int(hp.max()) + 10
    out_h = int(h * math.sin(e) + off + 12)
    can = np.empty((out_h, w, 3), np.uint8)
    can[:] = bg
    side = (top_rgb.astype(np.float32) * side_dark).astype(np.uint8)
    xs = np.arange(w)
    for y in range(h):
        top = (base[y] - hp[y] + off).astype(int)
        bot = int(base[y] + off)
        span = bot - top
        mx = int(span.max())
        for k in range(1, mx + 1):
            sel = span >= k
            can[np.minimum(top[sel] + k, out_h - 1), xs[sel]] = side[y, sel]
        can[top, xs] = top_rgb[y]
    return can
