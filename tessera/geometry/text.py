"""Text -> shapely polygons (mm) via Pillow rasterisation, for engraved part ids."""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import shapely
import shapely.ops
from PIL import Image, ImageDraw, ImageFont
from shapely import affinity

_RES = 0.05  # mm per raster px
_FONTS = [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"]


@lru_cache(maxsize=16)
def _font(px: int):
    for p in _FONTS:
        try:
            if os.path.isabs(p) and not os.path.exists(p):
                continue
            return ImageFont.truetype(p, px)
        except OSError:
            continue
    return ImageFont.load_default(size=px)


def ui_font(px: int):
    """Font for guide images (Turkish glyphs)."""
    return _font(int(px))


@lru_cache(maxsize=4096)
def text_polygon(text: str, cap_h_mm: float, mirror: bool = False):
    """Polygon of `text` with capital height ~cap_h_mm, centred on (0, 0), y up, mm.
    mirror=True flips x (for text read from below)."""
    cap_px = max(8, int(round(cap_h_mm / _RES)))
    font = _font(int(cap_px / 0.716))
    l, t, r, b = font.getbbox(text)
    w, h = r - l + 8, b - t + 8
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).text((4 - l, 4 - t), text, fill=255, font=font)
    a = np.asarray(im) > 127
    H, W = a.shape
    d = np.zeros((H, W + 2), bool)
    d[:, 1:-1] = a
    diff = np.diff(d.astype(np.int8), axis=1)
    rs, cs = np.nonzero(diff == 1)
    re_, ce = np.nonzero(diff == -1)
    if len(rs) == 0:
        return shapely.Polygon()
    boxes = shapely.box(cs, rs, ce, rs + 1)
    g = shapely.union_all(boxes).simplify(0.6)
    g = affinity.scale(g, _RES, -_RES, origin=(0, 0))
    if mirror:
        g = affinity.scale(g, -1, 1, origin=(0, 0))
    x0, y0, x1, y1 = g.bounds
    return affinity.translate(g, -(x0 + x1) / 2, -(y0 + y1) / 2)


def place_text(region, candidates, heights, angle_deg: float, mirror: bool, margin: float = 0.6):
    """Find a placement of one of the candidate strings inside `region`.
    Returns (geometry, text) or (None, None). Tries the long-axis angle and +90 deg."""
    if region is None or region.is_empty:
        return None, None
    inner = region.buffer(-margin)
    if inner.is_empty:
        return None, None
    try:
        pts = [shapely.ops.polylabel(max(getattr(inner, "geoms", [inner]), key=lambda q: q.area), 0.2)]
    except Exception:
        pts = []
    pts.append(inner.representative_point())
    for hgt in heights:
        for txt in candidates:
            base = text_polygon(txt, hgt, mirror)
            if base.is_empty:
                continue
            x0, y0, x1, y1 = base.bounds
            rect = shapely.box(x0 - 0.2, y0 - 0.2, x1 + 0.2, y1 + 0.2)
            for ang in (angle_deg, angle_deg + 90.0):
                rr = affinity.rotate(rect, ang, origin=(0, 0))
                for p in pts:
                    cand = affinity.translate(rr, p.x, p.y)
                    if inner.contains(cand):
                        g = affinity.translate(affinity.rotate(base, ang, origin=(0, 0)), p.x, p.y)
                        return g, txt
    return None, None
