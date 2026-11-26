"""Shaded top view of cut-first stones (R8), drawn from the exact outlines: what you see is what prints.
Joints show the wall, darkened by their depth; no backing is assumed."""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from ..geometry.plate import wall_height
from ..model import Mosaic
from ..pipeline.kadraj import draw

WALL = (214, 208, 198)
BANDS = max(2, min(8, os.cpu_count() or 2))   # threads that shade the picture side by side


def render_cut(m: Mosaic, width_px: int = 1500, az: float = -130.0, el: float = 48.0, wall=WALL) -> np.ndarray:
    w_mm, h_mm = m.size_mm
    ss = 2
    px = width_px / w_mm * ss
    W, H = int(round(w_mm * px)), int(round(h_mm * px))
    idm = Image.new("I", (W, H), 0)
    d = ImageDraw.Draw(idm)
    env = np.array([p.envelope.area for p in m.polys])
    for i in np.argsort(-env).tolist():
        p = m.polys[i]
        d.polygon([(x * px, y * px) for x, y in p.exterior.coords], fill=i + 1)
        for q in p.interiors:
            d.polygon([(x * px, y * px) for x, y in q.coords], fill=0)
    I = np.asarray(idm).astype(np.int32)
    top = np.concatenate([[0.0], m.tile_plane[:, 0]]).astype(np.float32)
    a, q = math.radians(az), math.radians(el)
    steps = max(2, int(float(top.max()) * px / math.tan(q)))
    ks = np.unique(np.linspace(1, steps, min(steps, 10)).astype(int))
    plate = wall_height(m) if m.plate is not None else None
    if m.plate is not None:
        wall = m.settings.palette[m.plate_colour].rgb
    pal = np.array([wall] + [f.rgb for f in m.settings.palette], np.float32)
    val = np.concatenate([[0.0], np.random.default_rng(3).uniform(-.025, .025, len(m.polys))]).astype(np.float32)
    cmap = np.concatenate([[0], np.asarray(m.tile_colour) + 1])
    o = None
    if m.outline is not None:                # outside the artwork: the bare wall
        o = Image.new("L", (W, H), 0)
        draw(ImageDraw.Draw(o), m.outline, lambda x, y: (x * px, y * px))
        o = np.asarray(o)
    # the picture is shaded in bands of rows on a few threads; every band reads enough rows around it that
    # its own rows come out exactly as in one piece (shadow reach, blurs, joint and wall distances)
    au = max(3, int(2.2 * px) | 1)
    rim = int(math.ceil(0.55 * px)) + 1
    reach = max(4 + max(abs(int(round(k * math.sin(a)))) for k in ks) + 3, 1 + 3, au // 2) + 1
    mg = reach + 2 * rim + 4
    n = max(1, min(BANDS, H // (4 * mg)))
    cuts = np.linspace(0, H, n + 1).astype(int)

    def band(j):
        y0, y1 = int(cuts[j]), int(cuts[j + 1])
        b0, b1 = max(0, y0 - mg), min(H, y1 + mg)
        Ib = I[b0:b1]
        col = _shade(Ib, top, px, a, q, ks, plate, au)
        col = pal[cmap[Ib]] * (1 + val[Ib])[..., None] * col[..., None]
        col = col[y0 - b0:y1 - b0]
        if o is not None:
            col[o[y0:y1] == 0] = WALL
        return np.clip(col, 0, 255).astype(np.uint8)
    with ThreadPoolExecutor(BANDS) as ex:
        out = Image.fromarray(np.concatenate(list(ex.map(band, range(n)))))
    return np.asarray(out.resize((W // ss, H // ss), Image.BOX))


def _dist(mask: np.ndarray) -> np.ndarray:
    """Distance to the nearest False pixel; infinite in a band without one (it lies farther than anything asked)."""
    return ndi.distance_transform_edt(mask) if not mask.all() else np.full(mask.shape, np.inf)


def _shade(I, top, px, a, q, ks, plate, au):
    """Light of every pixel of the stone map I (0 = joint): bevel, plate walls, shadow and the closed-in dark."""
    H, W = I.shape
    e = _dist(I > 0).astype(np.float32) / px
    Hm = np.where(I > 0, top[I] - np.clip(0.5 - e, 0, .5) ** 2 * 2.4, 0).astype(np.float32)
    if plate is not None:                    # the plate's walls stand between the stones where there is room
        core = _dist(I == 0) / px >= 0.55
        Hm[(I == 0) & (_dist(~core) / px <= 0.55)] = plate
    Hs = ndi.gaussian_filter(Hm, .7)
    gy, gx = np.gradient(Hs, 1 / px)
    L = (math.cos(a) * math.cos(q), math.sin(a) * math.cos(q), math.sin(q))
    dif = np.clip((-gx * L[0] - gy * L[1] + L[2]) / np.sqrt(gx ** 2 + gy ** 2 + 1), 0, 1)
    sh = np.zeros_like(Hm)
    for k in ks:
        ys = np.clip(np.arange(H) + int(round(k * math.sin(a))), 0, H - 1)
        xs = np.clip(np.arange(W) + int(round(k * math.cos(a))), 0, W - 1)
        sh = np.maximum(sh, Hs[ys][:, xs] - Hs - k / px * math.tan(q))
    sh = ndi.gaussian_filter(np.clip(sh * 1.6, 0, 1), 1.0)
    ao = np.clip(1 - (ndi.uniform_filter(Hm, au) - Hm) * .22, .12, 1)
    return ao * (.38 + .82 * dif * (1 - .75 * sh))
