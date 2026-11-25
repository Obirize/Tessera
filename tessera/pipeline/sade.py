"""Pictures made of tones (photographs, oil paintings): simplified into something that can be a mosaic.

A mosaic has a handful of paints and no soft transitions. So, before anything is cut,
1. the tones are spread out (a dark painting would otherwise be one dark field),
2. strokes, fur and grain are smoothed away, edges stay,
3. the few paints (renk.TONES) that show the picture best are chosen; small parts that stand out (lips, a pearl,
   lights) count more than their size,
4. shapes that are too small for how little they stand out are given to their neighbour.

kind() decides which pictures need it; flat art never comes through here.

find(img, width_mm)            -> like renk.find, every colour marked "yol": "ton"
label(img, colours, width_mm)  -> index map of the simplified picture
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .renk import TONES, WORK as FLAT_WORK, _stats, blur, cells, guided, lab32, paint_lab, shrink, spread_in
from .renk import label as flat_label
from .tiling import _adjacency, split_components

WORK = 1000             # px, the size everything here happens at
SOFT_ALWAYS_FLAT = 30.0  # up to this share (%) of soft transitions a picture is flat art, no questions asked
MAX_PAINTS = 8


def tonal(colours: list) -> bool:
    """Whether these colours were found by this module."""
    return bool(colours) and colours[0].get("yol") == "ton"


# ------------------------------------------------------------------ what kind of picture is it
def kind(img: np.ndarray, colours) -> tuple:
    """("net" | "cizim" | "ton", softness %). colours: () -> renk.find(img), only asked for when needed.
    net    flat art: posters, logos, vector drawings       -> cut as it is
    cizim  drawn or printed with lines, on toned paper     -> cut as it is
    ton    photographs and paintings                       -> simplified first
    Softness is the share of the picture that is neither a flat area nor a sharp edge: gradients, strokes, grain."""
    im = shrink(img, FLAT_WORK)
    lab = lab32(blur(im.astype(np.float32), 1.0))
    rng = spread_in(lab[..., 0], 5)
    soft = 100 * float(((rng > 3.5) & (rng < 25)).mean())
    if soft < SOFT_ALWAYS_FLAT:
        return "net", soft
    rng = spread_in(ndi.gaussian_filter(lab32(im)[..., 0], 0.7), 5)
    if 100 * float((rng > 25).sum() / max((rng > 3.5).sum(), 1)) < 22:      # hardly a sharp edge anywhere
        return "ton", soft
    # how far the picture is from the few colours found in it
    colours = colours()
    L = flat_label(im, colours)
    mu = np.array([lab[L == i].mean(0) if (L == i).any() else (0, 0, 0) for i in range(len(colours))])
    far = float(np.percentile(np.linalg.norm(lab - mu[L], axis=2), 90))
    return "cizim" if far < 21.5 else "ton", soft


def thin_lines(img: np.ndarray, width_mm: float) -> tuple:
    """(share of the picture that is thin dark line, typical line width mm)"""
    im = shrink(img, 1600)
    mm = width_mm / im.shape[1]
    g = lab32(im)[..., 0]
    r = max(3, int(round(2.4 / mm)) | 1)
    dark = ndi.grey_closing(g, r) - g > 18
    if dark.mean() < 0.004:
        return 0.0, 9.0
    dt = ndi.distance_transform_edt(dark)
    core = (dt >= ndi.maximum_filter(dt, 3)) & (dt > 0)
    return float(dark.mean()), float(np.percentile((2 * dt[core] - 0.6) * mm, 60))


# ------------------------------------------------------------------ simplifying
def spread(img: np.ndarray) -> np.ndarray:
    """Tones spread over the whole range (float RGB); colours keep their saturation."""
    x = img.astype(np.float32)
    L = x @ np.array([.299, .587, .114], np.float32)
    lo, hi = np.percentile(L, [1.5, 99.0])
    lo, hi = min(lo, 40.0), max(hi, 115.0)
    t = np.clip((L - lo) / max(hi - lo, 1.0), 0, 1)
    mid = float(np.clip(np.median(t), 0.02, 0.98))
    g = float(np.clip(np.log(0.28) / np.log(mid), 0.84, 1.0)) if mid < 0.28 else 1.0
    out = 8 + 240 * t ** g
    k = out / np.maximum(L, 6.0)
    y = x * k[..., None]
    over = np.maximum(y.max(-1) / 255.0, 1.0)         # too bright in one channel: towards white, not clipped
    y = y / over[..., None] + (out - out / over)[..., None] * 0.6
    return np.clip(y, 0, 255)


def smooth(x: np.ndarray, r: int) -> np.ndarray:
    """Edge-keeping smoothing (guided by its own colour channels together)."""
    for _ in range(3):
        I = x @ np.array([.4, .4, .2], np.float32)
        st = _stats(I, r)
        x = np.stack([guided(I, st, x[..., c], r, 70.0) for c in range(3)], -1)
    return x


def prepare(img: np.ndarray, width_mm: float) -> tuple:
    """Simplified picture in Lab (WORK px), mm per px."""
    im = shrink(img, WORK)
    mm = width_mm / im.shape[1]
    x = spread(im)
    x = np.stack([ndi.median_filter(x[..., i], 3) for i in range(3)], -1)
    x = smooth(x, int(np.clip(round(1.3 / mm), 2, 6)))
    return vivid(lab32(x)), mm


def vivid(lab: np.ndarray) -> np.ndarray:
    """A muted hue becomes a clear one (a mosaic speaks in clear colours); greys stay grey."""
    c = np.hypot(lab[..., 1], lab[..., 2])
    t = np.clip((c - 5.0) / 12.0, 0, 1) * np.maximum(np.clip((lab[..., 0] - 12) / 16, 0, 1), np.clip((c - 11) / 10, 0, 1))
    k = 1 + 0.55 * t * t * (3 - 2 * t) * np.clip((58 - c) / 30, 0, 1)
    k = k + 0.9 * np.clip((c - 7) / 8, 0, 1) * np.clip((32 - lab[..., 0]) / 16, 0, 1)     # a dark blue is blue
    return np.stack([lab[..., 0], lab[..., 1] * k, lab[..., 2] * k], -1).astype(np.float32)


def importance(lab: np.ndarray, mm: float) -> np.ndarray:
    """How much every pixel stands out from what lies around it (1 = not at all)."""
    s = 9.0 / mm
    far = blur(lab, s, truncate=2.0)
    d = np.linalg.norm((lab - far) * np.array([0.7, 1.3, 1.3]), axis=2)
    return 1 + np.clip((d - 12) / 14, 0, 4)


WT = np.array([1.15, 0.95, 0.95])        # light and dark carry the picture, hue comes second


def choose(lab: np.ndarray, w: np.ndarray, n=None) -> list:
    """Which of the TONES show this picture best: greedy, then swapped while that helps."""
    P = paint_lab(TONES)
    inv, _ = cells(lab.reshape(-1, 3), 4.0)
    ww = np.bincount(inv, w.ravel())
    cell = np.stack([np.bincount(inv, (lab[..., i] * w).ravel()) for i in range(3)], 1) / ww[:, None]
    D = np.linalg.norm((cell[:, None] - P[None]) * WT, axis=2) ** 1.4
    ww = ww / ww.sum()

    def cost(sel):
        return float((D[:, sel].min(1) * ww).sum())
    near = np.linalg.norm((P[:, None] - P[None]) * WT, axis=2) < 16.5     # two paints nobody can tell apart: one is enough

    def free(sel):
        return [j for j in range(len(P)) if not near[j][sel].any()]
    sel: list = []
    base = None
    while len(sel) < (n or MAX_PAINTS):
        cand = free(sel)
        if not cand:
            break
        i = min(cand, key=lambda j: cost(sel + [j]))
        c = cost(sel + [i])
        if n is None and len(sel) >= 4 and base - c < 0.03 * base:
            break
        sel.append(i)
        better = True
        while better:
            better = False
            for a in range(len(sel)):
                for j in free(sel[:a] + sel[a + 1:]):
                    t = sel[:a] + [j] + sel[a + 1:]
                    ct = cost(t)
                    if ct < c - 1e-9:
                        sel, c, better = t, ct, True
        base = c
    return sel


def find(img: np.ndarray, width_mm: float, n=None) -> list:
    """Like renk.find, but for tones: [{"rgb", "pay", "oz", "yol": "ton", "ad", "hex"}]."""
    lab, mm = prepare(img, width_mm)
    sel = choose(lab, importance(lab, mm), n)
    P = paint_lab(TONES)
    cols = [{"oz": [[float(v) for v in P[i]]], "yol": "ton", "ad": TONES[i][0], "hex": TONES[i][1]} for i in sel]
    L = label_lab(lab, cols, mm)
    im = shrink(img, WORK)
    out = []
    for i, c in enumerate(cols):
        s = L == i
        if s.mean() < 0.0004:
            continue
        c["rgb"] = [int(v) for v in np.median(im[s], axis=0)]
        c["pay"] = round(float(s.mean()), 4)
        out.append(c)
    out.sort(key=lambda c: -c["pay"])
    return out


def merge_weak(L: np.ndarray, P: np.ndarray, mm: float) -> np.ndarray:
    """Every shape has to earn its place: the less it differs from its neighbour, the larger it must be."""
    for _ in range(4):
        lab = split_components(L.astype(np.int32))       # connected shapes of one paint
        n = int(lab.max()) + 1
        area = np.bincount(lab.ravel(), minlength=n) * mm * mm
        col = np.zeros(n, int)
        col[lab.ravel()] = L.ravel()
        lo, hi, cnt = _adjacency(lab, n)
        x, y, cnt = np.r_[lo, hi], np.r_[hi, lo], np.r_[cnt, cnt]
        d = np.linalg.norm((P[col[x]] - P[col[y]]) * WT, axis=1)
        need = np.clip(9.0 * (46.0 / np.maximum(d, 8.0)) ** 2.1, 9.0, 300.0)      # mm2
        # the neighbour it is most like, weighted by how long they touch
        o = np.lexsort((y, cnt / np.maximum(d, 6.0) ** 1.3, x))
        x, y, need = x[o], y[o], need[o]
        last = np.r_[x[1:] != x[:-1], True]
        x, y, need = x[last], y[last], need[last]
        weak = (area[x] < need) & ((area[y] > area[x]) | ((area[y] == area[x]) & (y > x)))
        if not weak.any():
            break
        new = col.copy()
        new[x[weak]] = col[y[weak]]
        L = new[lab].astype(np.int16)
    return L


def label_lab(lab: np.ndarray, colours: list, mm: float) -> np.ndarray:
    P = np.array([c["oz"][0] for c in colours], np.float32)
    s = max(0.6, 0.45 / mm)
    # where nothing in the picture says "edge", neighbours agree on one paint; at an edge they need not
    r = max(2, int(round(3.0 / mm)))
    m = 2 * r + 1
    G = [ndi.gaussian_filter(lab[..., i], 0.6) for i in range(3)]
    mG = [ndi.uniform_filter(g, m) for g in G]
    var = sum(ndi.uniform_filter(g * g, m) - q * q for g, q in zip(G, mG))
    stay = np.clip(var / (var + 14.0), 0, 1)                    # 0 = calm, 1 = an edge runs through here
    stay = ndi.grey_closing(stay, 3)
    out = []
    for p in P:
        c = np.minimum(np.linalg.norm((lab - p) * WT, axis=2), 70.0).astype(np.float32)
        out.append(stay * ndi.gaussian_filter(c, s, truncate=2.5) + (1 - stay) * ndi.gaussian_filter(c, r * 0.6, truncate=2.5))
    cost = np.stack(out)
    return merge_weak(cost.argmin(0).astype(np.int16), P.astype(float), mm)


def label(img: np.ndarray, colours: list, width_mm: float) -> np.ndarray:
    """Which colour every part of the picture gets (a WORK-sized index map)."""
    lab, mm = prepare(img, width_mm)
    return label_lab(lab, colours, mm)
