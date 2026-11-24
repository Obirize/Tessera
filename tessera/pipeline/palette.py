"""Filament library and palette suggestion.

- ``load_filaments()``: Bambu PLA Basic / PLA Matte library from tessera/data/filaments.json.
- ``auto_palette(img, k)``: k colours straight from the picture (no library). Anti-alias edge
  pixels and halftone dots are ignored, so no slot is wasted on in-between greys.
- ``suggest_palette(img, library, k)``: the best k filaments *you own* for this picture, ordered
  dark -> light, each with a role: yapı (structure, darkest), ışık (light, lightest), vurgu
  (accent, most saturated of the rest) and alan (field).
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Optional, Union

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from ..model import Filament
from .descreen import descreen

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "filaments.json")

ROLE_STRUCTURE, ROLE_LIGHT, ROLE_ACCENT, ROLE_FIELD = "yapı", "ışık", "vurgu", "alan"
ROLE_HELP_TR = {
    ROLE_STRUCTURE: "Çizgiler ve gölgeler (en koyu renk)",
    ROLE_LIGHT: "Işık ve boşluk (en açık renk)",
    ROLE_ACCENT: "Anlam ve dikkat (en canlı renk)",
    ROLE_FIELD: "Geniş alanlar ve ara tonlar",
}


# ------------------------------------------------------------------ colour maths
def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.strip().lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_lab(rgb) -> np.ndarray:
    """sRGB uint8 (..., 3) -> CIE Lab (D65)."""
    c = np.asarray(rgb, np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


# ------------------------------------------------------------------ library
def load_filaments(path: Optional[str] = None, *, owned_ids: Optional[Iterable[str]] = None,
                   series: Optional[Iterable[str]] = None) -> list[dict]:
    """Library entries (dicts with id, seri, yuzey, ad, ad_en, hex, onizleme_hex, renk_kodu).
    ``owned_ids`` marks entries as owned (``owned`` key); ``series`` filters, e.g. ["PLA Matte"]."""
    with open(path or DATA_PATH, encoding="utf-8") as fh:
        d = json.load(fh)
    items = d["filamentler"] if isinstance(d, dict) else d
    ser = set(series) if series else None
    own = set(str(i) for i in owned_ids) if owned_ids is not None else None
    out = []
    for e in items:
        if ser and e.get("seri") not in ser:
            continue
        e = dict(e)
        e["owned"] = (str(e["id"]) in own) if own is not None else bool(e.get("owned", True))
        out.append(e)
    return out


def to_filament(entry: Union[dict, Filament]) -> Filament:
    """Library entry -> model.Filament (rgb = realistic preview colour)."""
    if isinstance(entry, Filament):
        return entry
    rgb = _hex_rgb(entry.get("onizleme_hex") or entry["hex"])
    return Filament(entry["ad"], rgb, entry.get("yuzey", "matte"), "Bambu", bool(entry.get("owned", True)))


# ------------------------------------------------------------------ image sampling
def _flat_samples(img: np.ndarray, max_px: int = 30000, seed: int = 1) -> np.ndarray:
    """RGB samples from flat areas only: anti-aliased edges and halftone dots are excluded."""
    a = np.asarray(img)[:, :, :3]
    if a.shape[0] * a.shape[1] > 1_500_000:
        f = (a.shape[0] * a.shape[1] / 1_500_000) ** 0.5
        a = np.asarray(Image.fromarray(a).resize((int(a.shape[1] / f), int(a.shape[0] / f)), Image.BOX))
    a = descreen(a, "auto", max_period_px=max(4.0, a.shape[1] / 120))
    L = a.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    # local contrast: max-min in a 3x3 window, on every channel
    spread = np.zeros(L.shape, np.float32)
    for c in range(3):
        ch = a[..., c].astype(np.float32)
        spread = np.maximum(spread, ndi.maximum_filter(ch, 3) - ndi.minimum_filter(ch, 3))
    # keep flat pixels and the extremes of each edge (cores of ink lines); drop the in-between
    # anti-alias pixels that would otherwise claim a grey slot of their own
    lo, hi = ndi.minimum_filter(L, 3), ndi.maximum_filter(L, 3)
    rng_l = np.maximum(hi - lo, 1e-3)
    extreme = ((L - lo) < 0.15 * rng_l) | ((hi - L) < 0.15 * rng_l)
    flat = (spread < 28) | extreme
    px = a[flat] if flat.sum() > 500 else a.reshape(-1, 3)
    rng = np.random.default_rng(seed)
    if len(px) > max_px:
        px = px[rng.choice(len(px), max_px, replace=False)]
    return px.astype(np.float64)


def _kmeans(x: np.ndarray, k: int, seed: int = 1, iters: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic k-means++ in whatever space ``x`` is in. -> (centres, labels)."""
    rng = np.random.default_rng(seed)
    k = max(1, min(k, len(x)))
    cent = [x[rng.integers(len(x))]]
    d2 = ((x - cent[0]) ** 2).sum(1)
    for _ in range(1, k):
        p = d2 / d2.sum() if d2.sum() > 0 else None
        cent.append(x[rng.choice(len(x), p=p)])
        d2 = np.minimum(d2, ((x - cent[-1]) ** 2).sum(1))
    cent = np.array(cent)
    lab = np.zeros(len(x), int)
    for _ in range(iters):
        lab = ((x[:, None] - cent[None]) ** 2).sum(-1).argmin(1)
        new = np.array([x[lab == i].mean(0) if (lab == i).any() else cent[i] for i in range(k)])
        if np.allclose(new, cent):
            break
        cent = new
    return cent, lab


def _blend_slots(cent: np.ndarray, share: np.ndarray, max_share: float = 0.05) -> np.ndarray:
    """bool per centre: a rarely used colour that is (nearly) a mix of two other centres, i.e. an
    edge halo between them (pink between red and cream), not a colour of its own."""
    k = len(cent)
    out = np.zeros(k, bool)
    for m_ in range(k):
        if share[m_] >= max_share:
            continue
        for a_ in range(k):
            for b_ in range(a_ + 1, k):
                if m_ in (a_, b_) or out[a_] or out[b_]:
                    continue
                v = cent[b_] - cent[a_]
                vv = float(v @ v)
                if vv < 1:
                    continue
                t = float((cent[m_] - cent[a_]) @ v) / vv
                d = float(np.sqrt(((cent[m_] - cent[a_] - t * v) ** 2).sum()))
                if 0.1 < t < 0.9 and d < 0.2 * np.sqrt(vv):
                    out[m_] = True
    return out


def auto_palette(img: np.ndarray, k: int, *, seed: int = 1) -> list[Filament]:
    """Up to k colours taken from the picture itself, most used first. Clustering runs in Lab on
    flat pixels only, so anti-alias greys and screentone dots do not get a slot of their own.
    A rarely used colour that is just a blend of two others (an edge halo) is dropped, so fewer
    than k colours can come back when the picture really has fewer."""
    px = _flat_samples(img, seed=seed)
    cent_lab, lab = _kmeans(rgb_to_lab(px.astype(np.uint8)), k, seed)
    cnt = np.bincount(lab, minlength=len(cent_lab))
    halo = _blend_slots(cent_lab, cnt / max(cnt.sum(), 1))
    if (~halo).sum() >= 2:
        cnt = np.where(halo, -1, cnt)
    out = []
    for i, j in enumerate([j for j in np.argsort(-cnt) if cnt[j] >= 0]):
        sel = lab == j
        rgb = px[sel].mean(0) if sel.any() else px.mean(0)
        out.append(Filament(f"Renk {i + 1}", tuple(int(round(v)) for v in rgb)))
    return out


# ------------------------------------------------------------------ suggestion
def suggest_palette(img: np.ndarray, library: Iterable[Union[dict, Filament]], k: int = 4, *,
                    owned_only: bool = True, seed: int = 1) -> list[dict]:
    """Pick the k library filaments that reproduce ``img`` best.

    Fewer than k come back when an extra filament would add almost nothing (two near-equal greys).
    Returns a list ordered dark -> light of dicts:
    ``{"filament": Filament, "id": product code or None, "rol": "yapı"|"ışık"|"vurgu"|"alan",
       "rol_aciklama": str, "pay": share of the picture (0..1)}``.
    Cost = mean Lab distance from each (flat) image pixel to its nearest chosen filament, with
    chroma weighted a bit higher so a red poster keeps its red. Greedy forward selection then
    swap refinement."""
    lib = list(library)
    if owned_only:
        own = [e for e in lib if (e.get("owned", True) if isinstance(e, dict) else e.owned)]
        lib = own if len(own) >= 2 else lib
    if not lib:
        raise ValueError("Kütüphanede filament yok")
    fils = [to_filament(e) for e in lib]
    ids = [str(e["id"]) if isinstance(e, dict) and "id" in e else None for e in lib]
    k = max(1, min(k, len(fils)))

    px = _flat_samples(img, seed=seed)
    # summarise the picture with weighted clusters (fast cost evaluation)
    cl_lab, cl = _kmeans(rgb_to_lab(px.astype(np.uint8)), min(24, len(px)), seed)
    wts = np.bincount(cl, minlength=len(cl_lab)).astype(np.float64)
    wts /= wts.sum()
    f_lab = rgb_to_lab(np.array([f.rgb for f in fils], np.uint8))
    scale = np.array([1.0, 1.25, 1.25])
    dist = np.sqrt((((cl_lab[:, None] - f_lab[None]) * scale) ** 2).sum(-1))   # (clusters, filaments)

    def cost(sel):
        return float((dist[:, sel].min(1) * wts).sum())

    chosen: list[int] = []
    for _ in range(k):
        best = min((i for i in range(len(fils)) if i not in chosen), key=lambda i: cost(chosen + [i]))
        chosen.append(best)
    improved = True
    while improved:
        improved = False
        for a in range(k):
            for i in range(len(fils)):
                if i in chosen:
                    continue
                trial = chosen[:a] + [i] + chosen[a + 1:]
                if cost(trial) < cost(chosen) - 1e-9:
                    chosen, improved = trial, True

    # drop filaments that add (almost) nothing: a second, nearly equal grey only creates a new
    # border full of crumbs (keep at least 2 colours)
    while len(chosen) > 2:
        base_cost = cost(chosen)
        gains = [cost(chosen[:i] + chosen[i + 1:]) - base_cost for i in range(len(chosen))]
        sh = np.bincount(dist[:, chosen].argmin(1), weights=wts, minlength=len(chosen))
        i = int(np.argmin(gains))
        if gains[i] < max(1.5, 0.12 * base_cost) or sh[i] < 0.03:
            chosen = chosen[:i] + chosen[i + 1:]
        else:
            break
    k = len(chosen)
    share = np.bincount(dist[:, chosen].argmin(1), weights=wts, minlength=k)
    labs = f_lab[chosen]
    order = np.argsort(labs[:, 0])
    roles = [ROLE_FIELD] * k
    if k >= 1:
        roles[order[0]] = ROLE_STRUCTURE
    if k >= 2:
        roles[order[-1]] = ROLE_LIGHT
    if k >= 3:
        rest = [i for i in range(k) if roles[i] == ROLE_FIELD]
        chroma = np.hypot(labs[rest, 1], labs[rest, 2])
        if chroma.max() > 20:
            roles[rest[int(chroma.argmax())]] = ROLE_ACCENT
    out = []
    for j in order:
        i = chosen[j]
        out.append({"filament": fils[i], "id": ids[i], "rol": roles[j], "rol_aciklama": ROLE_HELP_TR[roles[j]],
                    "pay": round(float(share[j]), 3)})
    return out
