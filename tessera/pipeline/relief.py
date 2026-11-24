"""Relief hierarchy (lead decisions R2-2, R2-7, R2-9; art team requests).

compute_relief(...) -> (a, b, c) per tile: top z = a + b*dx_mm + c*dy_mm.
relief_budget(m) -> dict for the UI's budget bar.

Rules:
- Plateau heights snap to a 1.5 mm grid, so neighbouring steps are 0 / 1.5 / 3 mm and
  cast readable shadows. A small per-tile jitter (at most 0.25 mm) keeps it alive.
- FIGURE: the whole figure zone is one smooth dome (all colours) and every tile is a tilted facet
  of it, as in the first version; tone changes never change the level.
- FLAT (cut sheets): colours stack as layers, largest area at the bottom, each next layer
  one step (1.5 mm, ``settings.flat_layer_mm``) higher.
- Area hierarchy (not flattening): ~70 % of the area at most 3 mm, ~25 % at most 8 mm and the
  top ~5 % (focal point first, then figures) stays free above 8 mm.
  Priority: FOCAL > FIGURE > BLOCKS > STONES > FLAT > RAYS.
- FOCAL always ends one step above everything else and at least 9 mm high (R2-9).
- Weight is NOT solved here by flattening (R2-9): savings come from the lattice backing and
  hollow tall tiles (3D team). Grams and kg/m2 come from the 3D team's single formula (R2-7).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage as ndi

from ..model import Vocab

STEP_MM = 1.5
MIN_TOP_MM = 1.5
FOCAL_MIN_MM = 9.0       # focal accent stays in the band above 8 mm (R2-9)
BANDS = ((3.0, 0.70), (8.0, 0.95))          # (cap mm, cumulative area share from the bottom)
PRIORITY = {int(Vocab.FOCAL): 0, int(Vocab.FIGURE): 1, int(Vocab.BLOCKS): 3, int(Vocab.STONES): 4,
            int(Vocab.FLAT): 5, int(Vocab.RAYS): 6}


def _snap(h: np.ndarray, step: float, floor: bool = False) -> np.ndarray:
    q = np.floor(h / step + 1e-6) if floor else np.rint(h / step)
    return np.maximum(q * step, MIN_TOP_MM)


def darkest_colour(pal: np.ndarray) -> int:
    return int((pal @ np.array([0.30, 0.59, 0.11], np.float32)).argmin())


def compute_relief(settings, pal: np.ndarray, lab: np.ndarray, zone: np.ndarray, colour: np.ndarray,
                   tz: np.ndarray, tc: np.ndarray, area: np.ndarray, rng, *, budget: bool = True):
    n = len(tz)
    mm = settings.mm_per_px
    vp = settings.vocab
    step = float(getattr(settings, "height_step_mm", STEP_MM))
    flat = lab.ravel()
    cnt = np.maximum(area, 1)
    nv = max(vp) + 1
    base = np.array([vp[int(z)].base_height_mm if int(z) in vp else 3.0 for z in range(nv)])[tz].astype(float)
    jit = np.array([vp[int(z)].jitter_mm if int(z) in vp else 0.0 for z in range(nv)])[tz]
    darkest = darkest_colour(pal)
    ink = tc == darkest

    # colour layering inside zones
    base += np.where((tz == Vocab.STONES) & ink, step, 0)
    base -= np.where((tz == Vocab.BLOCKS) & ink, step, 0)
    isf = tz == Vocab.FIGURE
    flat_z = tz == Vocab.FLAT
    if flat_z.any():
        layer_mm = float(getattr(settings, "flat_layer_mm", STEP_MM))
        fa = np.bincount(tc[flat_z], area[flat_z], minlength=len(pal))
        rank = np.empty(len(pal), int)
        rank[np.argsort(-fa)] = np.arange(len(pal))            # biggest area = bottom sheet
        base += np.where(flat_z, rank[tc] * layer_mm, 0)
    base = _snap(base, step)

    # rounded bodies (the first version's look): the whole FIGURE zone, every colour, is one
    # smooth dome; each tile is a tilted facet of it. Tone changes inside a figure never change
    # the level.
    bump_t = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    if isf.any() and settings.figure_bump_mm > 0:
        body = zone == Vocab.FIGURE
        sig = 15.0 / mm                                   # ~3 passes of a 30 mm box blur
        dome = ndi.gaussian_filter(body.astype(np.float32), sig) * body
        dome = dome / max(float(dome.max()), 1e-6) * settings.figure_bump_mm
        gyb, gxb = np.gradient(dome)
        bump_t = np.where(isf, np.bincount(flat, dome.ravel(), n) / cnt, 0)
        b = np.where(isf, np.bincount(flat, gxb.ravel(), n) / cnt / mm, 0)
        c = np.where(isf, np.bincount(flat, gyb.ravel(), n) / cnt / mm, 0)

    is_focal = tz == Vocab.FOCAL

    def tops():
        return base + bump_t

    if budget:
        # area bands by priority: the highest-priority ~5 % may rise above 8 mm, the next ~25 %
        # up to 8 mm, the rest up to 3 mm
        pr = np.array([PRIORITY.get(int(z), 4) for z in range(nv)])[tz]
        order = np.lexsort((-tops(), pr))
        cum = np.cumsum(area[order]) / max(area.sum(), 1)
        cap = np.full(n, np.inf)
        free = 1.0 - BANDS[1][1]
        mid = BANDS[1][1] - BANDS[0][1]
        cap[order[cum > free]] = BANDS[1][0]
        cap[order[cum > free + mid]] = BANDS[0][0]
        cap[is_focal] = np.inf
        over = tops() > cap
        if over.any():
            nb = np.minimum(base, _snap(cap - bump_t, step, floor=True))
            base = np.where(over, nb, base)
            t = tops()
            scale = np.where(over & (bump_t > 1e-6) & (t > cap),
                             np.clip((cap - base) / np.maximum(bump_t, 1e-6), 0, 1), 1.0)
            # the dome is lowered, but each tile keeps its facet tilt (the faceted look)
            bump_t = bump_t * scale

    # jitter only downwards, so a capped tile never creeps over its band limit
    base = base - rng.uniform(0, 1, n) * np.minimum(jit, 0.25)
    if is_focal.any():
        others = tops()[~is_focal]
        top_other = float(others.max()) if others.size else 0.0
        # the focal point always reads as the summit: one step above the rest and in the band
        # above 8 mm (R2-9), when it is a small accent (<= ~5 % of the area)
        floor_h = FOCAL_MIN_MM if area[is_focal].sum() <= 0.08 * area.sum() else 0.0
        target = max(_snap(np.array([top_other + step]), step)[0], floor_h)
        base = np.where(is_focal, np.maximum(target, base), base)
        b = np.where(is_focal, 0, b)
        c = np.where(is_focal, 0, c)
        bump_t = np.where(is_focal, 0, bump_t)
    return base + bump_t, b, c


def _mass(m) -> tuple[Optional[float], str]:
    """Grams from the 3D team's single formula (R2-7). -> (grams or None, source name)."""
    try:
        try:
            from ..export.mass import mass_model  # type: ignore[attr-defined]
        except ImportError:
            from ..export.project import mass_model  # type: ignore[attr-defined]
        r = mass_model(m)
        g = r.get("gram", r.get("grams")) if isinstance(r, dict) else r
        if g is not None:
            return float(g), "mass_model"
    except Exception:
        pass
    try:
        from ..export.api import estimate
        return float(estimate(m)["gram"]), "estimate"
    except Exception:
        return None, "yok"


def relief_budget(m) -> dict:
    """Numbers for the UI budget bar.

    Returns {"bantlar": {"0–3 mm": f, "3–8 mm": f, "8 mm üstü": f} (area shares by tile top),
    "gram": int|None, "kg_m2": float|None (from the 3D team's estimate, R2-7), "hedef_kg_m2": 3.0,
    "durum": "yeşil" (≤3) | "sarı" (≤4) | "kırmızı" | "bilinmiyor", "ortalama_mm", "en_yuksek_mm",
    "odak_en_yuksek": bool, "kaynak": formula used, "mesaj": Turkish sentence}."""
    s = m.settings
    area = m.tile_area_px.astype(float)
    top = m.tile_plane[:, 0].astype(float)
    tot = max(area.sum(), 1.0)
    low = float(area[top <= 3.0 + 1e-6].sum() / tot)
    midb = float(area[(top > 3.0 + 1e-6) & (top <= 8.0 + 1e-6)].sum() / tot)
    high = max(0.0, 1.0 - low - midb)
    mean_top = float((top * area).sum() / tot)
    foc = m.tile_zone == Vocab.FOCAL
    focal_top = (not foc.any()) or (not (~foc).any()) or float(top[foc].min()) >= float(top[~foc].max()) - 1e-6
    grams, src = _mass(m)
    w_mm, h_mm = m.size_mm
    kg = grams / 1000.0 / (w_mm * h_mm / 1e6) if grams is not None else None
    target = float(getattr(s, "max_kg_m2", 3.0))
    if kg is None:
        durum, msg = "bilinmiyor", "Ağırlık şu an hesaplanamadı."
    elif kg <= target + 0.05:
        durum, msg = "yeşil", f"Bütçe uygun: {kg:.1f} kg/m²."
    elif kg <= 4.0:
        durum, msg = "sarı", f"Biraz ağır: {kg:.1f} kg/m² (hedef {target:.0f}). Asmak için sağlam bir arka levha kullan."
    else:
        durum, msg = "kırmızı", f"Ağır: {kg:.1f} kg/m² (hedef {target:.0f}). Yükseklikleri ya da figür tümseğini azalt."
    return {"bantlar": {"0–3 mm": round(low, 3), "3–8 mm": round(midb, 3), "8 mm üstü": round(high, 3)},
            "gram": None if grams is None else int(round(grams)),
            "kg_m2": None if kg is None else round(kg, 2), "hedef_kg_m2": target, "durum": durum,
            "ortalama_mm": round(mean_top, 2), "en_yuksek_mm": round(float(top.max()) if top.size else 0.0, 2),
            "odak_en_yuksek": bool(focal_top), "kaynak": src, "mesaj": msg}
