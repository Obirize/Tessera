"""Shaded top view of cut-first stones (R8), drawn from the exact outlines: what you see is what prints.
Joints show the wall, darkened by their depth; no backing is assumed."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from ..geometry.plate import wall_height
from ..model import Mosaic

WALL = (214, 208, 198)


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
    e = ndi.distance_transform_edt(I > 0).astype(np.float32) / px
    Hm = np.where(I > 0, top[I] - np.clip(0.5 - e, 0, .5) ** 2 * 2.4, 0).astype(np.float32)
    if m.plate is not None:                  # the plate's walls stand between the stones where there is room
        core = ndi.distance_transform_edt(I == 0) / px >= 0.55
        Hm[(I == 0) & (ndi.distance_transform_edt(~core) / px <= 0.55)] = wall_height(m)
    Hs = ndi.gaussian_filter(Hm, .7)
    gy, gx = np.gradient(Hs, 1 / px)
    a, q = math.radians(az), math.radians(el)
    L = (math.cos(a) * math.cos(q), math.sin(a) * math.cos(q), math.sin(q))
    dif = np.clip((-gx * L[0] - gy * L[1] + L[2]) / np.sqrt(gx ** 2 + gy ** 2 + 1), 0, 1)
    sh = np.zeros_like(Hm)
    steps = max(2, int(float(top.max()) * px / math.tan(q)))
    for k in np.unique(np.linspace(1, steps, min(steps, 10)).astype(int)):
        ys = np.clip(np.arange(H) + int(round(k * math.sin(a))), 0, H - 1)
        xs = np.clip(np.arange(W) + int(round(k * math.cos(a))), 0, W - 1)
        sh = np.maximum(sh, Hs[ys[:, None], xs[None]] - Hs - k / px * math.tan(q))
    sh = ndi.gaussian_filter(np.clip(sh * 1.6, 0, 1), 1.0)
    ao = np.clip(1 - (ndi.uniform_filter(Hm, max(3, int(2.2 * px) | 1)) - Hm) * .22, .12, 1)
    if m.plate is not None:
        wall = m.settings.palette[m.plate_colour].rgb
    pal = np.array([wall] + [f.rgb for f in m.settings.palette], np.float32)
    val = np.concatenate([[0.0], np.random.default_rng(3).uniform(-.025, .025, len(m.polys))]).astype(np.float32)
    ci = np.concatenate([[0], np.asarray(m.tile_colour) + 1])[I]
    col = pal[ci] * (1 + val[I])[..., None] * (ao * (.38 + .82 * dif * (1 - .75 * sh)))[..., None]
    if m.outline is not None:                # outside the artwork: the bare wall
        o = Image.new("L", (W, H), 0)
        ImageDraw.Draw(o).polygon([(x * px, y * px) for x, y in m.outline.exterior.coords], fill=255)
        col[np.asarray(o) == 0] = WALL
    out = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8))
    return np.asarray(out.resize((W // ss, H // ss), Image.BOX))
