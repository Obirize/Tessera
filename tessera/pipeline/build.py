"""2.5D pipeline: image -> Mosaic.

build_mosaic(image, settings, zone_map=None, **optional) -> Mosaic

Steps: resize -> detail-preserving palette quantisation (colour.py) -> zones ->
per-vocabulary tessellation that respects colour edges (tiling.py) -> split by
colour / merge slivers -> hairline gaps (kept off thin features) -> heights.
The algorithm team owns this file; keep the signature stable.
"""
from __future__ import annotations

import math
import time
import zlib
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi
from shapely.geometry import box

from ..model import VOCAB_NAMES_TR, Mode, Mosaic, Settings, Vocab
from ..resim import open_rgb  # noqa: F401 (kept outside the pipeline: the window opens pictures at start)
from .colour import absorb_islands, quantize_detail, thin_masks
from .descreen import descreen as descreen_image
from .lines import handle_lines
from .relief import compute_relief, relief_budget  # noqa: F401
from .kesim import LAYOUTS
from .styles import STYLE_NAMES_TR, STYLES, style_cells, style_relief
from .tiling import (andamento as andamento_tiles, blocks_adaptive, edge_distance, focal_rings,  # noqa: F401
                     merge_small, rays_adaptive, rays_logpolar, snap_zones, split_components)  # noqa: F401


# ------------------------------------------------------------------ io
SCREEN_MAX_MM = 9.0   # dot patterns finer than this (in the finished mosaic) count as print screens
SCREEN_MIN_AREA = 0.04   # a dotted print covers real areas; a smaller "screen" is an ornament


def screen_free(img: np.ndarray, mode, width_mm: float, **kw):
    """img (the whole picture, width_mm wide in the end) without its print dots."""
    return descreen_image(img, mode, max_period_px=SCREEN_MAX_MM / (width_mm / img.shape[1]),
                          min_area_frac=SCREEN_MIN_AREA, **kw)


def open_capped(path, side: int = 1600) -> tuple:
    """(picture at most ``side`` px wide, its original width)"""
    img = np.asarray(open_rgb(path))
    native = img.shape[1]
    if native > side:
        img = np.asarray(Image.fromarray(img).resize((side, max(8, round(img.shape[0] * side / native))), Image.BOX))
    return img, native


_SCREEN_MEMO: dict = {}   # fingerprint of a loaded working image -> screen mask at working size


def _fingerprint(a: np.ndarray) -> tuple:
    flat = np.ascontiguousarray(a).reshape(-1)
    return a.shape, zlib.crc32(flat[:: max(1, flat.size // 65536)].tobytes())


def _screen_mask_for(img: np.ndarray) -> Optional[np.ndarray]:
    return _SCREEN_MEMO.get(_fingerprint(img))


def descreen_mode(settings: Settings, override=None):
    """Resolve the descreen setting: keyword override > settings.descreen > "auto"."""
    return getattr(settings, "descreen", "auto") if override is None else override


def working_mm_per_px(width_mm: float, preview: bool = False) -> float:
    """Working raster resolution for an artwork ``width_mm`` wide (lead decision R2-3).
    Build/export: max(0.4, width/2600). Preview: max(0.8, width/1300). Keeps big pieces
    (up to 300 cm) within time and memory budgets."""
    w = float(width_mm)
    return max(0.8, w / 1300.0) if preview else max(0.4, w / 2600.0)


def _floor_resolution(settings: Settings) -> None:
    """Automatic working resolution for very wide pieces (R2-3). A request of 0.8 mm/px or coarser
    is treated as a preview. Updates ``settings.mm_per_px`` in place so all mm maths agree."""
    want = settings.mm_per_px
    floor = working_mm_per_px(settings.width_mm, preview=want >= 0.8 - 1e-9)
    if floor > want + 1e-9:
        settings.mm_per_px = round(floor, 4)


def load_image(path_or_img, settings: Settings, *, descreen=None,
               progress: Optional[Callable[[float, str], None]] = None) -> np.ndarray:
    """Load and resize so that width_mm / mm_per_px pixels wide. Returns uint8 RGB.

    - Transparent areas become white.
    - Halftone / screentone dot patterns are smoothed into flat grey *before* resizing (where the
      dots are still resolved); ``descreen``: "auto" (default), True or False.
    - Very wide pieces get a coarser working resolution (``working_mm_per_px``); the new value
      is written back to ``settings.mm_per_px``.
    - Corrupt / unsupported files raise ValueError with a Turkish message."""
    if progress:
        progress(0.0, "Görsel yükleniyor")
    im = open_rgb(path_or_img)
    _floor_resolution(settings)
    w_px = max(8, int(round(settings.width_mm / settings.mm_per_px)))
    h_px = max(8, int(round(im.height * w_px / im.width)))
    mode = descreen_mode(settings, descreen)
    if mode is not False:
        if im.width > 2.5 * w_px:   # keep descreening fast on huge scans
            im = im.resize((int(2.5 * w_px), max(8, int(round(im.height * 2.5 * w_px / im.width)))), Image.BOX)
        if progress:
            progress(0.4, "Tram çözülüyor")
        arr, smask = screen_free(np.asarray(im), mode, settings.width_mm, return_mask=True)
        im = Image.fromarray(arr)
    else:
        smask = None
    if (w_px, h_px) != im.size:
        # area averaging when shrinking: keeps the exact ink coverage of thin lines and has no
        # Lanczos ringing (light halos next to dark lines would turn into false thin features)
        im = im.resize((w_px, h_px), Image.BOX if w_px < im.width else Image.LANCZOS)
    out = np.asarray(im).copy()
    if progress:
        progress(1.0, "Görsel hazır")
    if smask is not None and smask.max() > 0.5:
        if len(_SCREEN_MEMO) > 8:
            _SCREEN_MEMO.clear()
        sm = np.asarray(Image.fromarray(smask).resize((w_px, h_px), Image.BOX)) > 0.25
        # include the rim of the screened area, where leftover dot "beads" sit
        _SCREEN_MEMO[_fingerprint(out)] = ndi.binary_dilation(sm, np.ones((5, 5), bool))
    return out


def zones_from_shapes(shape_hw, shapes: list[dict], default: int = int(Vocab.STONES)) -> np.ndarray:
    """Rasterise zone shapes. Each shape: {"vocab": int, "polygon": [[x,y],...]} or
    {"vocab": int, "rect": [x0,y0,x1,y1]} or {"vocab": int, "circle": [cx,cy,r]}.
    Coordinates are in fractions of width/height (0..1) so they survive resizing.
    Later shapes paint over earlier ones."""
    h, w = shape_hw
    zone = np.full((h, w), default, np.int32)
    for s in shapes:
        img = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(img)
        if "polygon" in s:
            d.polygon([(x * w, y * h) for x, y in s["polygon"]], fill=255)
        elif "rect" in s:
            x0, y0, x1, y1 = s["rect"]
            d.rectangle((x0 * w, y0 * h, x1 * w, y1 * h), fill=255)
        elif "circle" in s:
            cx, cy, r = s["circle"]  # cx, r as fraction of width; cy as fraction of height
            d.ellipse((cx * w - r * w, cy * h - r * w, cx * w + r * w, cy * h + r * w), fill=255)
        zone[np.asarray(img) > 0] = int(s["vocab"])
    return zone


# ------------------------------------------------------------------ palette (legacy)
def quantize(img: np.ndarray, palette_rgb: np.ndarray) -> np.ndarray:
    """Legacy quantiser: nearest palette colour + 3x3 majority (loses thin lines).
    Kept for ``build_mosaic(detail=False)``; see colour.quantize_detail."""
    blur = np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(1.0))).astype(np.float32)
    wts = np.array([0.30, 0.59, 0.11], np.float32) ** 0.5
    d = (((blur[:, :, None, :] - palette_rgb[None, None]) * wts) ** 2).sum(-1)
    col = d.argmin(-1)
    return majority3(col, len(palette_rgb))


def majority3(lbl: np.ndarray, n: int) -> np.ndarray:
    best, cnt = lbl.copy(), np.full(lbl.shape, -1, np.int32)
    for v in range(n):
        s = ndi.uniform_filter((lbl == v).astype(np.float32), 3, mode="nearest")
        upd = s > cnt
        best[upd], cnt[upd] = v, s[upd]
    return best


# ------------------------------------------------------------------ legacy tessellations
def voronoi(h, w, s, rng, lloyd=2, jitter=0.38, mask=None):
    """Voronoi stones on a jittered grid with Lloyd relaxation (seeds clamped to their grid
    cell): the first version's stones. Labels are computed only for ``mask`` pixels (nearest seed
    via a KD-tree), the rest of the returned array is 0."""
    from scipy.spatial import cKDTree
    s = max(3, int(round(s)))
    gy, gx = h // s + 2, w // s + 2
    cy, cx = np.mgrid[0:gy, 0:gx]
    sy = ((cy + 0.5 + rng.uniform(-jitter, jitter, cy.shape)) * s).ravel()
    sx = ((cx + 0.5 + rng.uniform(-jitter, jitter, cx.shape)) * s).ravel()
    cyf, cxf = cy.ravel(), cx.ravel()
    m = np.ones((h, w), bool) if mask is None else mask
    yy, xx = np.nonzero(m)
    out = np.zeros((h, w), np.int64)
    if len(yy) == 0:
        return out
    pts = np.c_[yy, xx].astype(np.float32)
    # only seeds near the mask matter
    near = np.zeros(gy * gx, bool)
    near[np.unique((yy // s) * gx + (xx // s))] = True
    near = ndi.binary_dilation(near.reshape(gy, gx), np.ones((3, 3), bool)).ravel()
    idx = np.nonzero(near)[0]
    lab = None
    for it in range(lloyd + 1):
        tree = cKDTree(np.c_[sy[idx], sx[idx]])
        _, j = tree.query(pts, workers=-1)
        lab = idx[j]
        if it == lloyd:
            break
        c = np.bincount(lab, minlength=gy * gx)
        my = np.bincount(lab, yy, minlength=gy * gx)
        mx = np.bincount(lab, xx, minlength=gy * gx)
        ok = c > 0
        sy[ok] = np.clip(my[ok] / c[ok], cyf[ok] * s + 1, (cyf[ok] + 1) * s - 1)
        sx[ok] = np.clip(mx[ok] / c[ok], cxf[ok] * s + 1, (cxf[ok] + 1) * s - 1)
    out[yy, xx] = lab
    return out


def rays(h, w, centre, plank_px, deg=2.2):
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - centre[0], yy - centre[1])
    th = np.degrees(np.arctan2(yy - centre[1], xx - centre[0])) % 360
    abin = (th / deg).astype(np.int64)
    stagger = (abin * 37 % 11) / 11.0
    return abin * 100000 + ((r / max(plank_px, 4)) + stagger).astype(np.int64)


def blocks(h, w, s):
    s = max(3, int(round(s)))
    yy, xx = np.mgrid[0:h, 0:w]
    return (yy // s) * 100000 + xx // s


# ------------------------------------------------------------------ main
def _as_rgb_array(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    return np.asarray(open_rgb(x))


def _source_for(image, img: np.ndarray, source, max_scale: float = 2.0, descreen="auto",
                mm_per_px: float = 0.4) -> Optional[np.ndarray]:
    """High-resolution copy of the picture for supersampled quantisation (at most ``max_scale``
    times the working size, to stay fast). None when no larger source is available."""
    if source is None and not isinstance(image, np.ndarray):
        source = image
    src = _as_rgb_array(source)
    if src is None:
        return None
    h, w = img.shape[:2]
    if src.shape[1] < w * 1.2:
        return None
    f = min(max_scale, src.shape[1] / w)
    tw, th = int(round(w * f)), int(round(h * f))
    if (tw, th) != (src.shape[1], src.shape[0]):
        src = np.asarray(Image.fromarray(src).resize((tw, th), Image.BOX))
    mm_src = mm_per_px * w / tw
    if descreen is not False:
        src = descreen_image(src, descreen, max_period_px=SCREEN_MAX_MM / mm_src)
    return src


def _check_palette(settings: Settings) -> np.ndarray:
    pal = getattr(settings, "palette", None) or []
    if len(pal) < 2:
        raise ValueError("En az 2 renk gerekli.")
    try:
        arr = np.array([tuple(int(v) for v in f.rgb[:3]) for f in pal], np.float32)
    except (TypeError, ValueError, AttributeError) as e:
        raise ValueError("Renk okunamadı: her rengin (R, G, B) değeri olmalı.") from e
    if arr.ndim != 2 or arr.shape[1] != 3 or (arr < 0).any() or (arr > 255).any():
        raise ValueError("Renk geçersiz: R, G, B değerleri 0–255 arasında olmalı.")
    return arr


def _check_zone(zone_map, settings: Settings, hw) -> np.ndarray:
    h, w = hw
    valid = sorted(int(v) for v in settings.vocab)
    dv = int(settings.default_vocab)
    if dv not in valid:
        raise ValueError(f"Geçersiz varsayılan taş tipi: {dv}. Geçerli değerler: {valid}.")
    if zone_map is None:
        return np.full((h, w), dv, np.int32)
    z = np.asarray(zone_map)
    if z.ndim != 2:
        raise ValueError("Bölge haritası iki boyutlu olmalı (yükseklik × genişlik).")
    if not np.issubdtype(z.dtype, np.integer):
        if not np.isfinite(z).all() or (np.rint(z) != z).any():
            raise ValueError("Bölge haritasında tam sayı olmayan değerler var.")
    z = z.astype(np.int64)
    bad = np.setdiff1d(np.unique(z), valid)
    if bad.size:
        raise ValueError(f"Bölge haritasında geçersiz taş tipi değeri: {bad[:5].tolist()}. "
                         f"Geçerli değerler: {valid}.")
    if z.shape != (h, w):
        z = np.asarray(Image.fromarray(z.astype(np.uint8)).resize((w, h), Image.NEAREST))
    return z.astype(np.int32)


def build_mosaic(image, settings: Settings, zone_map: Optional[np.ndarray] = None, *,
                 source=None, detail: bool = True, andamento: Optional[bool] = None, descreen=None,
                 progress: Optional[Callable[[float, str], None]] = None,
                 timings: Optional[dict] = None, budget: bool = True, labels=None, zoom: float = 1.0) -> Mosaic:
    """Image -> Mosaic.

    Optional keyword arguments (all default to the best quality):
    - ``source``: the original picture (path, PIL image or array) at a higher resolution than
      ``image``; colours are then decided on it, which keeps sub-pixel lines. Used automatically
      when ``image`` itself is a path or PIL image.
    - ``detail``: thin-feature preserving quantisation.
    - ``andamento``: contour-following stones in STONES/FIGURE zones. Off by default since round 5
      (the reference look uses evenly relaxed Voronoi stones); ``settings.andamento`` or True turns
      it on.
    - ``descreen``: "auto" | True | False - smooth halftone dot screens into grey before
      quantising (default: ``settings.descreen`` if present, else "auto"). Only screened areas are
      low-passed; ink lines stay sharp.
    - ``progress``: callback(fraction, message_tr); fractions rise monotonically from 0 to 1 over
      loading/descreening, colours, tiling (one update per zone type), merging, relief and done.
    - ``timings``: dict that receives seconds per stage.
    - ``labels``, ``zoom``: for a detail of a picture (cut-first textures only): index map into
      ``settings.source_colours``; how many times wider the whole picture is.
    - ``budget``: apply the relief hierarchy and weight budget (R2-2; see relief.py).
    For a fast UI preview, run with ``settings.mm_per_px = 0.8``.
    """
    tm = timings if timings is not None else {}
    t_prev = time.perf_counter()

    last = [0.0]

    def report(frac, msg):
        """progress(fraction, message_tr), fractions never go backwards."""
        if progress:
            f = min(1.0, max(last[0], float(frac)))
            last[0] = f
            progress(f, msg)

    def mark(name, frac, msg):
        nonlocal t_prev
        t = time.perf_counter()
        tm[name] = round(t - t_prev, 3)
        t_prev = t
        report(frac, msg)

    report(0.0, "Görsel hazırlanıyor")
    rng = np.random.default_rng(settings.seed)
    ds_mode = descreen_mode(settings, descreen)
    if isinstance(image, np.ndarray):
        img = image if (image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8) else np.asarray(open_rgb(image))
    else:
        img = load_image(image, settings, descreen=ds_mode,
                         progress=(lambda f, msg: report(0.1 * f, msg)) if progress else None)
    report(0.1, "Renkler ayrılıyor")
    h, w = img.shape[:2]
    mm = settings.mm_per_px
    pal_full = _check_palette(settings)
    # the same colour chosen twice is one filament: pixels go to its first occurrence
    _, first_idx = np.unique(pal_full, axis=0, return_index=True)
    to_full = np.sort(first_idx)
    pal = pal_full[to_full]
    k = len(pal)
    if k < 2:
        raise ValueError("En az 2 farklı renk gerekli: bütün renkler aynı seçilmiş.")
    if getattr(settings, "style", None) in LAYOUTS and not getattr(settings, "merge_same_colour", False):
        return _build_kesim(image, img, source, settings, zone_map, ds_mode, pal, to_full, mark, report,
                            labels, zoom)
    min_feat_mm = float(getattr(settings, "min_feature_mm", 1.6))
    min_part_mm2 = float(getattr(settings, "min_part_mm2", 12.0))
    bc = getattr(settings, "backing_colour", None)
    lum_u = pal @ np.array([0.30, 0.59, 0.11], np.float32)
    if bc is not None and 0 <= int(bc) < len(pal_full):
        backing_u = int(np.argmin(((pal - pal_full[int(bc)]) ** 2).sum(1)))
    else:
        backing_u = int(lum_u.argmin())
    feat_px = max(2, int(round(min_feat_mm / mm)))

    src = _source_for(image, img, source, descreen=ds_mode, mm_per_px=mm) if detail else None
    screen = _screen_mask_for(img) if ds_mode is not False else None
    if src is not None or ds_mode is False:
        qimg = img
    else:
        qimg, sm = descreen_image(img, ds_mode, max_period_px=SCREEN_MAX_MM / mm, return_mask=True)
        if sm.max() > 0.5:
            sm = ndi.binary_dilation(sm > 0.25, np.ones((5, 5), bool))
            screen = sm if screen is None else (screen | sm)
    if detail:
        colour = quantize_detail(qimg, pal, mm, source=src, min_feature_mm=min_feat_mm, screen_mask=screen,
                                 min_part_mm2=min_part_mm2, backing=backing_u)
    else:
        colour = quantize(qimg, pal)
    colour = colour.astype(np.int32)
    mark("renk", 0.3, "Taşlar diziliyor")

    zone = _check_zone(zone_map, settings, (h, w))
    vp = settings.vocab
    tile_px = {int(v): (max(vp[v].tile_mm / mm, 3) if vp[v].tile_mm > 0 else 6.0 / mm) for v in vp}
    sliver = np.array([0.5 * tile_px.get(v, 15.0) ** 2 for v in range(max(tile_px) + 1)])
    # the focal disc keeps its painted shape (the first version's sun)
    zone = snap_zones(zone, colour, sliver, protect=(int(Vocab.FOCAL),))
    style = getattr(settings, "style", None)
    if style in STYLES:
        return _build_styled(style, img, settings, colour, zone, pal, to_full, rng, mm, mark, report)

    ckey = zone.astype(np.int64) * 64 + colour      # tiles never cross a colour or zone edge
    loc = np.zeros((h, w), np.int64)
    min_area = np.zeros((h, w), np.float32)
    present = np.unique(zone)
    dist = None
    if andamento is None:
        andamento = bool(getattr(settings, "andamento", False))
    if andamento and np.isin(present, [Vocab.STONES, Vocab.FIGURE]).any():
        dist = edge_distance(ckey)
    for vi, v in enumerate(present):
        report(0.3 + 0.3 * vi / max(len(present), 1), f"Taşlar diziliyor: {VOCAB_NAMES_TR.get(Vocab(int(v)), '')}")
        m = zone == v
        p = vp[int(v)]
        tpx = tile_px[int(v)]
        if v == Vocab.RAYS:
            centre = settings.rays_center or (w / 2, h * 0.12)
            # log-polar planks: constant angle, length proportional to r (tile_mm sets the
            # length/width ratio: 23 mm -> 3.5:1, the first version's proportions)
            aspect = max(1.5, p.tile_mm / 6.6)
            t = rays_logpolar(colour, m, centre, deg=2.2, aspect=aspect, r_min_px=3.0 / mm, min_strip_px=2.0 / mm)
            # crumbs are judged against the local plank size (small near the sun, big at the edge)
            yy_m, xx_m = np.nonzero(m)
            rr = np.hypot(xx_m - centre[0], yy_m - centre[1])
            plank_area = (np.radians(2.2) * rr) ** 2 * aspect
            min_area[yy_m, xx_m] = np.maximum(0.3 * plank_area, 6.0)
        elif v == Vocab.BLOCKS:
            t = blocks_adaptive(colour, m, tpx, angle=False)
            min_area[m] = 0.3 * tpx * tpx
        elif v == Vocab.FOCAL:
            # the first version's sun: a fine square grid (focal_rings stays available)
            t = blocks(h, w, tpx)
            min_area[m] = 0.3 * tpx * tpx
        elif v == Vocab.FLAT:
            t = np.zeros((h, w), np.int64)
            min_area[m] = 0.3 * tpx * tpx
        elif dist is not None:
            t = andamento_tiles(colour, m, dist, tpx, rng, lloyd=min(p.lloyd, 1))
            min_area[m] = 0.3 * tpx * tpx
        else:
            t = voronoi(h, w, tpx, rng, lloyd=p.lloyd, mask=m)
            min_area[m] = 0.3 * tpx * tpx
        loc[m] = t[m]
    del dist
    mark("taslar", 0.6, "Kırıntılar birleştiriliyor")

    key = (zone.astype(np.int64) * 10 ** 10 + loc) * 64 + colour
    del loc
    _, lab = np.unique(key, return_inverse=True)
    del key
    lab = split_components(lab.reshape(h, w))
    thin = thin_masks(colour, k, feat_px + 1)
    lab = merge_small(lab, ckey, min_area, pix_thin=thin, pix_colour=colour)
    # tiles narrower than the minimum feature that are cut from a wider colour region (thin
    # wedge strips, slivers between seeds) join their same-colour neighbour
    lab = _merge_narrow(lab, ckey, feat_px)
    del ckey, min_area
    n = int(lab.max()) + 1
    flat = lab.ravel()
    # a tile that absorbed a sliver from the next zone takes its main zone everywhere
    nzv = int(zone.max()) + 1
    zc = np.bincount(flat.astype(np.int64) * nzv + zone.ravel(), minlength=n * nzv).reshape(n, nzv)
    tz = zc.argmax(1)
    zone = tz[lab].astype(np.int32)
    mark("birlestir", 0.75, "Kabartma hesaplanıyor")

    # hairline gaps, gap_mm wide; put on the wider side of a colour edge so that thin
    # features (whiskers, outlines) keep their full width
    gap = np.zeros((h, w), bool)
    for axis in (0, 1):
        a = (slice(None), slice(None, -1)) if axis == 1 else (slice(None, -1), slice(None))
        b = (slice(None), slice(1, None)) if axis == 1 else (slice(1, None), slice(None))
        d = lab[a] != lab[b]
        to_a = d & (colour[a] != colour[b]) & thin[b] & ~thin[a]
        gap[b] |= d & ~to_a
        gap[a] |= to_a
    gw = max(1, int(round(float(settings.gap_mm) / mm)))
    for _ in range(gw - 1):
        g2 = gap.copy()
        g2[:, 1:] |= gap[:, :-1]
        g2[1:, :] |= gap[:-1, :]
        gap |= g2 & ~thin
    inner2 = ndi.binary_erosion(~gap, iterations=1, border_value=1)

    cnt = np.maximum(np.bincount(flat, minlength=n), 1)
    tc = np.zeros(n, int)
    tc[flat] = colour.ravel()
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = np.bincount(flat, xx.ravel(), n) / cnt
    cy = np.bincount(flat, yy.ravel(), n) / cnt

    base, b, c = compute_relief(settings, pal, lab, zone, colour, tz, tc, np.bincount(flat, minlength=n), rng,
                                budget=budget)
    height = (base[lab] + (b[lab] * (xx - cx[lab]) + c[lab] * (yy - cy[lab])) * mm).astype(np.float32)
    del xx, yy
    height -= np.where(~inner2, np.float32(settings.bevel_mm), np.float32(0.0))
    height[gap] = 0.0
    np.maximum(height, 0.0, out=height)
    # report colours in the caller's palette numbering
    colour = to_full[colour].astype(np.int32)
    tc = to_full[tc]
    mark("yukseklik", 0.95, "Son kontroller")

    mos = Mosaic(settings=settings, image_rgb=img, colour=colour, zone=zone, label=lab, gap=gap,
                 height=height.astype(np.float32), tile_colour=tc, tile_zone=tz,
                 tile_area_px=np.bincount(flat, minlength=n), tile_centroid=np.stack([cx, cy], 1),
                 tile_plane=np.stack([base, b, c], 1))
    mos.stats = quick_stats(mos)
    mark("istatistik", 1.0, "Hazır")
    return mos


_split_components = split_components  # backwards compatibility


# ------------------------------------------------------------------ R6: separate stones
WALL_RGB = (178, 174, 166)      # what shows in a real gap: the wall / board, never a dark backing


def stone_widths_mm(lab: np.ndarray, gap: np.ndarray, mm: float) -> np.ndarray:
    """Printed width of every stone: 2 x the largest inscribed radius of its non-gap pixels."""
    d = ndi.distance_transform_edt(~gap)
    n = int(lab.max()) + 1
    mx = np.asarray(ndi.maximum(d, lab, np.arange(n)))
    return 2.0 * mx * mm


def _gap_band(lab: np.ndarray, gw: int) -> np.ndarray:
    gap = np.zeros(lab.shape, bool)
    gap[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    gap[1:, :] |= lab[1:, :] != lab[:-1, :]
    # grow to gw px, alternating sides so the band stays centred on the joint
    for i in range(gw - 1):
        g2 = gap.copy()
        if i % 2 == 0:
            g2[:, :-1] |= gap[:, 1:]
            g2[:-1, :] |= gap[1:, :]
        else:
            g2[:, 1:] |= gap[:, :-1]
            g2[1:, :] |= gap[:-1, :]
        gap = g2
    return gap


def _build_kesim(image, img, source, settings, zone_map, ds_mode, pal, to_full, mark, report, labels, zoom):
    """R8: cut-first stones. The outlines are exact polygons; the rasters only serve numbers and zones."""
    from . import kadraj, kesim
    h, w = img.shape[:2]
    mm = settings.mm_per_px
    src = _as_rgb_array(source if source is not None or isinstance(image, np.ndarray) else image)
    if src is None or src.shape[1] < img.shape[1]:
        src = img
    if src.shape[1] > 2400:
        src = np.asarray(Image.fromarray(src).resize((2400, int(round(src.shape[0] * 2400 / src.shape[1]))), Image.BOX))
    if ds_mode is not False:
        src = screen_free(src, ds_mode, w * settings.mm_per_px)
    mark("girdi", 0.06, "Renkler ayrılıyor")
    zone = _check_zone(zone_map, settings, (h, w))
    rc = settings.rays_center
    centre = (rc[0] / w, rc[1] / h) if rc else None
    ams = settings.mode == Mode.AMS
    cols = getattr(settings, "source_colours", None) or None
    to = None
    if cols:                                     # palette position -> position among the distinct colours
        pf = _check_palette(settings)
        same = np.array([int(np.flatnonzero((pal == c).all(1))[0]) for c in pf])
        to = same[np.clip(np.asarray([int(c["hedef"]) for c in cols]), 0, len(pf) - 1)]
    base = settings.plate_colour
    if base is not None:
        base = int(np.flatnonzero((pal == _check_palette(settings)[base]).all(1))[0])
    shape = kadraj.outline(settings.shape, w * mm, h * mm)
    cut = getattr(settings, "cutout", None)      # what stays when the background is taken away (zemin)
    if cut is not None:
        from .kesim import resolution
        from .zemin import silhouette
        sil = silhouette(cut, w * mm, h * mm, resolution(w * mm))
        if sil is not None:
            shape = sil if shape is None else sil.intersection(shape)
    r = kesim.build(src, w * mm, pal, colours=cols, to=to, zoom=zoom, labels=labels,
                    pieces=settings.pieces, base=base, shape=shape,
                    layout=settings.style, zone=zone, centre=centre,
                    gap=float(settings.gap_mm), size=float(getattr(settings, "stone_scale", 1.0)),
                    seed=int(settings.seed), ams=ams, report=lambda f, m: report(0.06 + 0.88 * f, m))
    polys, pc, pz = r["polys"], to_full[r["colour"]], r["zone"]
    n = len(polys)
    if not n:
        raise ValueError("Görselden taş çıkmadı. Daha büyük bir boyut ya da farklı renkler dene.")
    rng = np.random.default_rng(settings.seed)
    fig = np.isin(pz, (int(Vocab.FIGURE), int(Vocab.FOCAL)))
    # one height per physical stone (AMS pieces of one stone share it)
    ns = int(r["stone"].max()) + 1
    jit = np.round(rng.uniform(0, 1, ns) * float(settings.h_jitter_mm) / 0.2) * 0.2
    top = float(settings.stone_h_mm) + np.where(fig, float(settings.relief_mm), 0.0) + jit[r["stone"]]
    cen = np.array([(p.centroid.x, p.centroid.y) for p in polys]) / mm
    area = np.array([p.area for p in polys])
    # raster view of the same polygons
    k = max(1, int(np.ceil(mm / 0.25)))
    idm = Image.new("I", (w * k, h * k), 0)
    dr = ImageDraw.Draw(idm)
    f = k / mm
    for i in np.argsort(-np.array([p.envelope.area for p in polys])).tolist():
        p = polys[i]
        dr.polygon([(x * f, y * f) for x, y in p.exterior.coords], fill=i + 1)
        for q in p.interiors:
            dr.polygon([(x * f, y * f) for x, y in q.coords], fill=0)
    raw = np.asarray(idm)[k // 2::k, k // 2::k][:h, :w]
    idx = ndi.distance_transform_edt(raw == 0, return_distances=False, return_indices=True)
    lab = (raw[idx[0], idx[1]] - 1).astype(np.int64)
    gap = raw == 0
    height = np.where(gap, 0.0, top[lab]).astype(np.float32)
    mos = Mosaic(settings=settings, image_rgb=img, colour=pc[lab].astype(np.int32), zone=pz[lab].astype(np.int32),
                 label=lab, gap=gap, height=height, tile_colour=pc, tile_zone=pz,
                 tile_area_px=np.maximum(1, np.round(area / mm ** 2)).astype(int), tile_centroid=cen,
                 tile_plane=np.stack([top, np.zeros(n), np.zeros(n)], 1))
    mos.polys = polys
    mos.outline = shape
    if base is not None and not ams:             # with the AMS everything is printed together anyway
        mos.plate = shape or box(0, 0, w * mm, h * mm)
    mos.style = settings.style
    mos.stats = quick_stats(mos)
    mos.stats.update({"stil": LAYOUTS[settings.style], "tas_sayisi": len(set(r["stone"].tolist())), "parca": n,
                      "alan": r["areas"], "cizgi": None if r["contour"] is None else int(to_full[r["contour"]])})
    mark("istatistik", 1.0, "Hazır")
    return mos


def _build_styled(style, img, settings, colour, zone, pal, to_full, rng, mm, mark, report):
    """R6-1/R6-3: every stone a separate piece, in one of the four styles."""
    h, w = colour.shape
    k = len(pal)
    gap_mm = float(settings.gap_mm)
    smin = float(getattr(settings, "min_stone_mm", 4.0))
    line_mode = str(getattr(settings, "line_mode", "auto"))
    raw_px = int(np.ceil((smin + gap_mm) / mm))          # raw tile width that prints >= smin
    # 1) lines thinner than a stone: rows or carved grooves (no detail is dropped)
    grooves, gdepth, lstats = [], np.zeros((h, w), np.float32), {"cizgi_sira": 0, "cizgi_oluk": 0}
    island_px = int(round(0.6 * raw_px * raw_px))
    for _ in range(3):
        # widening one line can make its neighbour too thin: repeat until nothing is left
        colour, gv, gd, ls = handle_lines(colour, pal, mm, smin, gap_mm, line_mode)
        grooves += gv
        gdepth = np.maximum(gdepth, gd)
        for kk in lstats:
            lstats[kk] += ls[kk]
        colour = absorb_islands(colour, k, island_px, passes=2)
        if ls["cizgi_sira"] + ls["cizgi_oluk"] == 0:
            break
    report(0.4, "Taşlar diziliyor")
    # 2) style cells -> stones (cell x colour, connected parts)
    centre = settings.rays_center or (w / 2, h * 0.12)
    cells = style_cells(style, colour, zone, settings, rng, centre, mm)
    report(0.55, "Taşlar diziliyor")
    key = cells * 64 + colour
    _, lab = np.unique(key, return_inverse=True)
    del key, cells
    lab = split_components(lab.reshape(h, w))
    # 3) too small / too narrow stones join a same-colour neighbour (any zone)
    min_area = np.full((h, w), np.float32(0.9 * raw_px * raw_px))
    lab = merge_small(lab, colour.astype(np.int64), min_area, passes=3)
    lab = _merge_narrow(lab, colour.astype(np.int64), raw_px, passes=4)
    n = int(lab.max()) + 1
    flat = lab.ravel()
    nzv = int(zone.max()) + 1
    zc = np.bincount(flat.astype(np.int64) * nzv + zone.ravel(), minlength=n * nzv).reshape(n, nzv)
    tz = zc.argmax(1)
    zone = tz[lab].astype(np.int32)
    tc = np.zeros(n, int)
    tc[flat] = colour.ravel()
    mark("taslar", 0.7, "Kabartma hesaplanıyor")
    # 4) real gaps
    gw = max(1, int(round(gap_mm / mm)))
    gap = _gap_band(lab, gw)
    cnt = np.maximum(np.bincount(flat, minlength=n), 1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = np.bincount(flat, xx.ravel(), n) / cnt
    cy = np.bincount(flat, yy.ravel(), n) / cnt
    # 5) relief per stone: height + tilt (+ sculpted profile for freud)
    a, b, c, prof, dirs, hi, lo = style_relief(style, settings, colour, zone, lab, tz, tc, cx, cy, mm, rng, pal)
    dx = (xx - cx[lab]) * mm
    dy = (yy - cy[lab]) * mm
    height = (a[lab] + b[lab] * dx + c[lab] * dy).astype(np.float32)
    if (prof > 0).any():
        cd, sd = np.cos(dirs), np.sin(dirs)
        u = dx * cd[lab] + dy * sd[lab]                       # along the fall direction
        umin = np.full(n, np.inf)
        umax = np.full(n, -np.inf)
        np.minimum.at(umin, flat, u.ravel())
        np.maximum.at(umax, flat, u.ravel())
        span = np.maximum(umax - umin, 1e-3)
        sfr = np.clip((u - umin[lab]) / span[lab], 0, 1)      # 0 = high edge, 1 = low edge
        hi_p, lo_p = hi[lab], lo[lab]
        ramp = hi_p - (hi_p - lo_p) * sfr
        quarter = lo_p + (hi_p - lo_p) * np.sqrt(np.clip(1 - sfr * sfr, 0, 1))
        pp = prof[lab]
        height = np.where(pp == 1, ramp, np.where(pp == 2, quarter, height)).astype(np.float32)
    del yy, xx, dx, dy
    # bevelled (rounded) top edges, then the carved grooves
    bev = {"klasik": 0.85, "smalti": 0.6, "freud": 0.9, "orman": 1.0}.get(style, 0.8)
    dgap = ndi.distance_transform_edt(~gap).astype(np.float32) * mm
    t = np.clip(dgap / bev, 0, 1)
    height -= (0.8 * bev) * (1 - t) ** 2
    height -= gdepth
    height[gap] = 0.0
    np.maximum(height, 0.0, out=height)
    mark("yukseklik", 0.95, "Son kontroller")

    colour_out = to_full[colour].astype(np.int32)
    tc_out = to_full[tc]
    mos = Mosaic(settings=settings, image_rgb=img, colour=colour_out, zone=zone, label=lab, gap=gap,
                 height=height, tile_colour=tc_out, tile_zone=tz,
                 tile_area_px=np.bincount(flat, minlength=n), tile_centroid=np.stack([cx, cy], 1),
                 tile_plane=np.stack([a, b, c], 1))
    # optional per-stone extras (documented in docs/MIMARI.md)
    mos.tile_profile = prof.astype(np.int8)          # 0 flat, 1 ramp, 2 quarter-round
    mos.tile_dir = dirs.astype(np.float32)           # radians, raster x right / y down, high -> low
    mos.tile_hilo = np.stack([hi, lo], 1).astype(np.float32)   # profile heights (mm)
    mos.grooves = grooves                            # carved V-grooves: pts mm (x right, y down)
    mos.groove_depth = gdepth                        # (H, W) mm, already cut into ``height``
    mos.style = style
    mos.stats = quick_stats(mos)
    widths = stone_widths_mm(lab, gap, mm)
    mos.stats.update({"stil": STYLE_NAMES_TR.get(style, style), "tas_sayisi": int(n),
                      "en_dar_tas_mm": round(float(widths.min()), 2) if n else 0.0,
                      "dar_tas_sayisi": int((widths < smin - 1e-6).sum()), **lstats})
    mark("istatistik", 1.0, "Hazır")
    return mos


def _merge_narrow(lab: np.ndarray, ckey: np.ndarray, feat_px: int, passes: int = 2) -> np.ndarray:
    """Merge tiles whose largest inscribed width is below ``feat_px`` into the same-key neighbour
    with the longest shared border (tiles that are a whole thin colour feature have no such
    neighbour and stay)."""
    for _ in range(passes):
        pad = np.pad(lab, 1, constant_values=-1)
        c = pad[1:-1, 1:-1]
        bnd = (pad[:-2, 1:-1] != c) | (pad[2:, 1:-1] != c) | (pad[1:-1, :-2] != c) | (pad[1:-1, 2:] != c)
        d = ndi.distance_transform_edt(~bnd)
        n = int(lab.max()) + 1
        mx = np.asarray(ndi.maximum(d, lab, np.arange(n)))
        width = 2 * np.floor(mx) + 1
        narrow = width < feat_px
        if not narrow.any():
            break
        before = n
        lab = merge_small(lab, ckey, np.where(narrow[lab], np.float32(1e12), np.float32(0)), passes=1)
        if int(lab.max()) + 1 == before:
            break
    return lab


def quick_stats(m: Mosaic) -> dict:
    s = m.settings
    mm2 = s.mm_per_px ** 2
    real = m.tile_area_px * mm2 >= 3.2   # ignore crumbs under ~3 mm2 (20 px at 0.4 mm/px)
    per = {}
    vol_tot = 0.0
    for ci, f in enumerate(s.palette):
        sel = real & (m.tile_colour == ci)
        pix = (m.colour == ci) & ~m.gap
        area = pix.sum() * mm2
        grams = float(m.height[pix].sum() * mm2 / 1000 * 1.24 * 0.6)
        vol_tot += grams
        per[f.name] = {"taş": int(sel.sum()), "gram": round(grams),
                       "tabla": max(1, math.ceil(area / (s.plate_mm ** 2 * 0.6))) if area else 0}
    w_mm, h_mm = m.size_mm
    return {"boyut_mm": [round(w_mm), round(h_mm)], "taş_toplam": int(real.sum()), "renkler": per,
            "gram_tahmini": round(vol_tot + w_mm * h_mm * s.base_thickness_mm / 1000 * 1.24 * 0.35),
            "saat_tahmini": round((vol_tot + w_mm * h_mm * s.base_thickness_mm / 1000 * 1.24 * 0.35) / 22)}
