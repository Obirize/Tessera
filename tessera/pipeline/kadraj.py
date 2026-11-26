"""Which part of the picture becomes the mosaic, and the outline it gets.

kadraj: {"don": 0|90|180|270 (clockwise), "ayna": bool, "kutu": [x0, y0, x1, y1]}, the box as fractions of the
        picture after it was turned; None = the whole picture as it is
sekil:  {"tur": "dikdortgen"|"yuvarlak"|"daire"|"oval", "r": corner radius as a share of the shorter side,
         "zemin": the background taken away (pipeline.zemin)}; None = rectangle
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from shapely import affinity
from shapely.geometry import Point, box

WHOLE = [0.0, 0.0, 1.0, 1.0]
SHAPES = ("dikdortgen", "yuvarlak", "daire", "oval")
MIN_SIDE = 0.04          # smallest box, as a fraction of the picture


# ------------------------------------------------------------------ the wishes, cleaned
def clean_frame(k):
    """A usable kadraj, or None when it is the whole picture as it is."""
    if not isinstance(k, dict):
        return None
    try:
        don = int(k.get("don") or 0) % 360
        x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in (k.get("kutu") or WHOLE))
    except (TypeError, ValueError):
        return None
    don = don if don in (90, 180, 270) else 0
    x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
    if x1 - x0 < MIN_SIDE or y1 - y0 < MIN_SIDE:
        x0, y0, x1, y1 = WHOLE
    b = [round(v, 5) for v in (x0, y0, x1, y1)]
    ayna = bool(k.get("ayna"))
    return None if b == WHOLE and not don and not ayna else {"don": don, "ayna": ayna, "kutu": b}


def clean_shape(s):
    from .zemin import clean
    if not isinstance(s, dict):
        return None
    out = {"tur": s.get("tur") if s.get("tur") in SHAPES else "dikdortgen"}
    if out["tur"] == "yuvarlak":
        try:
            out["r"] = round(min(0.5, max(0.02, float(s.get("r", 0.15)))), 3)
        except (TypeError, ValueError):
            out["r"] = 0.15
    z = clean(s.get("zemin"))
    if z is not None:
        out["zemin"] = z
    return None if out == {"tur": "dikdortgen"} else out


# ------------------------------------------------------------------ the picture
def turned(im: Image.Image, k) -> Image.Image:
    """The picture turned and mirrored, not yet cut."""
    don = (k or {}).get("don", 0)
    if don:
        im = im.transpose({90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}[don])
    return im.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if (k or {}).get("ayna") else im


def pixels(k, w: int, h: int) -> tuple:
    """The box in whole pixels of the turned picture (w, h)."""
    x0, y0, x1, y1 = (k or {}).get("kutu") or WHOLE
    a, b = round(x0 * w), round(y0 * h)
    return a, b, max(a + 2, round(x1 * w)), max(b + 2, round(y1 * h))


def outline(sekil, w: float, h: float):
    """The outer shape as an exact polygon in a w x h box (y down), None = the whole box."""
    t = (sekil or {}).get("tur")
    if t in ("daire", "oval"):
        return affinity.scale(Point(w / 2, h / 2).buffer(1.0, quad_segs=180), w / 2, h / 2)
    if t == "yuvarlak":
        r = min(0.499, sekil["r"]) * min(w, h)
        return box(r, r, w - r, h - r).buffer(r, quad_segs=48)
    return None


def fill_outside(a: np.ndarray, sekil, keep=None) -> np.ndarray:
    """What lies outside the shape, and the background taken away (keep: what stays, any size), takes the
    colours at its edge, so it has no say in colours or zones."""
    from .zemin import bleed
    W, H = a.shape[1::-1]
    p, inside, rim = outline(sekil, W, H), None, 0
    if p is not None:
        inside = mask(p, (W, H), 1.0)
    if keep is not None and not keep.all():
        k = np.asarray(Image.fromarray(keep.astype(np.uint8) * 255).resize((W, H), Image.BILINEAR)) > 127
        inside = k if inside is None else inside & k
        rim = max(1, round(2 * W / keep.shape[1]))         # the blended edge of the background goes too
    return a if inside is None else bleed(a, inside, rim)


def polygons(g) -> list:
    """The polygons of any shapely geometry."""
    if g is None or g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    return [q for p in getattr(g, "geoms", []) for q in polygons(p)]


def draw(d, g, f, fill=255, hole=0) -> None:
    """A geometry (mm) into an ImageDraw; f: mm -> px. Holes are drawn with ``hole``."""
    for p in polygons(g):
        d.polygon([f(x, y) for x, y in p.exterior.coords], fill=fill)
        for q in p.interiors:
            d.polygon([f(x, y) for x, y in q.coords], fill=hole)


def mask(poly, size, res: float) -> np.ndarray:
    """Pixels (size = W, H; res mm each) whose larger part lies inside the outline (any geometry, holes too)."""
    from PIL import ImageDraw
    W, H = size
    k, band = 4, 512
    out = np.zeros((H, W), bool)
    for y0 in range(0, H, band):
        h = min(band, H - y0)
        im = Image.new("L", (W * k, h * k))
        draw(ImageDraw.Draw(im), poly, lambda x, y: (x / res * k - .5, (y / res - y0) * k - .5))
        out[y0:y0 + h] = np.asarray(im.resize((W, h), Image.BOX)) >= 128
    return out


@lru_cache(maxsize=2)
def _prepared(path: str, mtime: float, kadraj: str, sekil: str) -> np.ndarray:
    from ..resim import open_rgb
    k, s = json.loads(kadraj), json.loads(sekil)
    im = turned(open_rgb(path), k)
    a = fill_outside(np.asarray(im.crop(pixels(k, *im.size))), s, kept(path, k, s))
    a.setflags(write=False)
    return a


def kept(path: str, kadraj, sekil):
    """What stays of the frame when the background is taken away (bool, zemin.SIDE), None = all of it."""
    if (sekil or {}).get("zemin") is None:
        return None
    from .zemin import keep
    k = keep(path, kadraj, sekil["zemin"])[0]
    return None if k.all() else k


def inside(path: str, kadraj, sekil, size):
    """Where the stones are, in a picture of the frame of this size (W, H); None = everywhere."""
    W, H = size
    p, k, m = outline(sekil, W, H), kept(path, kadraj, sekil), None
    if p is not None:
        m = mask(p, (W, H), 1.0)
    if k is not None:
        k = np.asarray(Image.fromarray(k.astype(np.uint8) * 255).resize((W, H), Image.BILINEAR)) > 127
        m = k if m is None else m & k
    return m


def picture(path: str, kadraj=None, sekil=None):
    """What the pipeline sees: the path itself when nothing is cut, else the prepared picture."""
    if kadraj is None and (sekil is None or sekil == {"tur": "dikdortgen"}):
        return path
    return _prepared(path, os.path.getmtime(path), json.dumps(kadraj), json.dumps(sekil))


# ------------------------------------------------------------------ points of the original picture
def to_frame(x, y, k) -> tuple:
    """Fractions of the original picture -> fractions of the frame."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    k = k or {}
    for _ in range(k.get("don", 0) // 90):
        x, y = 1 - y, x
    if k.get("ayna"):
        x = 1 - x
    x0, y0, x1, y1 = k.get("kutu") or WHOLE
    return (x - x0) / (x1 - x0), (y - y0) / (y1 - y0)


def zones_to_frame(shapes: list, k, aspect: float) -> list:
    """Zone shapes of a picture (aspect = width / height) in its frame; rectangles and circles become polygons."""
    if k is None:
        return shapes
    out = []
    for s in shapes:
        if "polygon" in s:
            p = np.asarray(s["polygon"], float)
        elif "rect" in s:
            x0, y0, x1, y1 = s["rect"]
            p = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        else:
            cx, cy, r = s["circle"]
            t = np.linspace(0, 2 * np.pi, 96, endpoint=False)
            p = np.stack([cx + r * np.cos(t), cy + r * np.sin(t) * aspect], 1)
        u, v = to_frame(p[:, 0], p[:, 1], k)
        out.append({**{a: b for a, b in s.items() if a not in ("polygon", "rect", "circle")},
                    "polygon": np.stack([u, v], 1).tolist()})
    return out
