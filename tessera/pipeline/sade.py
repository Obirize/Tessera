"""Pictures made of tones (photographs, oil paintings): simplified into something that can be a mosaic.

A mosaic has a handful of paints and no soft transitions. So, before anything is cut,
1. the tones are spread out (a dark painting would otherwise be one dark field),
2. strokes, fur and grain are smoothed away, edges stay,
3. the few MAIN colours that show the picture best are chosen; small parts that stand out (lips, a pearl,
   lights) count more than their size,
4. shapes that are too small for how little they stand out are given to their neighbour.

simplify(...) is only used for pictures that need it (see kind()); flat art never comes through here.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from . import renk
from .palette import rgb_to_lab

WORK = 1000             # px, the size everything here happens at
SOFT_ALWAYS_FLAT = 30.0  # up to this share (%) of soft transitions a picture is flat art, no questions asked
MAX_PAINTS = 8


def _lab(rgb) -> np.ndarray:
    rgb = np.asarray(rgb)
    return np.asarray(rgb_to_lab(np.clip(rgb, 0, 255).astype(np.uint8).reshape(-1, 3)), np.float32).reshape(rgb.shape)


# ------------------------------------------------------------------ what kind of picture is it
def softness(img: np.ndarray) -> float:
    """Share (%) of the picture that is neither a flat area nor a sharp edge: gradients, strokes, grain."""
    im = renk._shrink(img, 900).astype(np.float32)
    x = np.stack([ndi.gaussian_filter(im[..., i], 1.0) for i in range(3)], -1)
    g = _lab(x)[..., 0]
    rng = ndi.maximum_filter(g, 5) - ndi.minimum_filter(g, 5)
    return 100 * float(((rng > 3.5) & (rng < 25)).mean())


def kind(img: np.ndarray, colours) -> tuple:
    """("net" | "cizim" | "ton", numbers). colours: () -> renk.find(img), only asked for when needed.
    net    flat art: posters, logos, vector drawings       -> cut as it is
    cizim  drawn or printed with lines, on toned paper     -> cut as it is
    ton    photographs and paintings                       -> simplified first"""
    soft = softness(img)
    if soft < SOFT_ALWAYS_FLAT:
        return "net", {"yumusak": round(soft, 1)}
    im = renk._shrink(img, 900)
    g = ndi.gaussian_filter(_lab(im)[..., 0], 0.7)
    rng = ndi.maximum_filter(g, 5) - ndi.minimum_filter(g, 5)
    hard = 100 * float((rng > 25).sum() / max((rng > 3.5).sum(), 1))
    if hard < 22:                                # hardly a sharp edge anywhere
        return "ton", {"yumusak": round(soft, 1), "sert": round(hard, 1)}
    # how far the picture is from the few colours found in it
    colours = colours()
    L = renk.label(im, colours)
    lab = _lab(np.stack([ndi.gaussian_filter(im[..., i].astype(np.float32), 1.0) for i in range(3)], -1))
    mu = np.array([lab[L == i].mean(0) if (L == i).any() else (0, 0, 0) for i in range(len(colours))])
    far = float(np.percentile(np.linalg.norm(lab - mu[L], axis=2), 90))
    k = "cizim" if far < 21.5 else "ton"
    return k, {"yumusak": round(soft, 1), "sert": round(hard, 1), "uzak": round(far, 1)}


def thin_lines(img: np.ndarray, width_mm: float) -> tuple:
    """(share of the picture that is thin dark line, typical line width mm)"""
    im = renk._shrink(img, 1600)
    mm = width_mm / im.shape[1]
    g = _lab(im)[..., 0]
    r = max(3, int(round(2.4 / mm)) | 1)
    dark = ndi.minimum_filter(ndi.maximum_filter(g, r), r) - g > 18
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


def smooth(x: np.ndarray, r: int, it: int = 3, eps: float = 70.0) -> np.ndarray:
    """Edge-keeping smoothing (guided by its own colour channels together)."""
    m = 2 * r + 1
    for _ in range(it):
        I = x @ np.array([.4, .4, .2], np.float32)
        mI = ndi.uniform_filter(I, m)
        v = np.maximum(ndi.uniform_filter(I * I, m) - mI * mI, 0)
        out = []
        for c in range(3):
            p = x[..., c]
            mp = ndi.uniform_filter(p, m)
            a = (ndi.uniform_filter(I * p, m) - mI * mp) / (v + eps)
            b = mp - a * mI
            out.append(ndi.uniform_filter(a, m) * I + ndi.uniform_filter(b, m))
        x = np.stack(out, -1)
    return x


def prepare(img: np.ndarray, width_mm: float) -> tuple:
    """Simplified picture in Lab (WORK px), mm per px."""
    im = renk._shrink(img, WORK)
    mm = width_mm / im.shape[1]
    x = spread(im)
    x = np.stack([ndi.median_filter(x[..., i], 3) for i in range(3)], -1)
    x = smooth(x, int(np.clip(round(1.3 / mm), 2, 6)))
    return vivid(_lab(x)), mm


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
    far = np.stack([ndi.gaussian_filter(lab[..., i], s, truncate=2.0) for i in range(3)], -1)
    d = np.linalg.norm((lab - far) * np.array([0.7, 1.3, 1.3]), axis=2)
    return 1 + np.clip((d - 12) / 14, 0, 4)


# tones between the main colours: a face or a dark painting cannot do without them
TONES = renk.MAIN + [("Ten rengi", "#E3B892"), ("Koyu kahve", "#3A281D"), ("Kiremit", "#A34A32"),
                     ("Zeytin yeşili", "#6C6A3B"), ("Duman mavisi", "#6C8AA5")]


def _paints():
    h = [x for _, x in TONES]
    return np.asarray(rgb_to_lab(np.array([renk.hex_rgb(v) for v in h], np.uint8)), float).reshape(-1, 3)


WT = np.array([1.15, 0.95, 0.95])        # light and dark carry the picture, hue comes second


def choose(lab: np.ndarray, w: np.ndarray, n=None) -> list:
    """Which MAIN paints show this picture best: greedy, then swapped while that helps."""
    P = _paints()
    q = np.round(lab.reshape(-1, 3) / 4.0).astype(np.int64)
    q -= q.min(0)
    m = q.max(0) + 1
    key = (q[:, 0] * m[1] + q[:, 1]) * m[2] + q[:, 2]
    _, inv = np.unique(key, return_inverse=True)
    inv = inv.ravel()
    ww = np.bincount(inv, w.ravel())
    cell = np.stack([np.bincount(inv, (lab[..., i] * w).ravel()) for i in range(3)], 1) / ww[:, None]
    D = np.linalg.norm((cell[:, None] - P[None]) * WT, axis=2) ** 1.4
    ww = ww / ww.sum()

    def cost(sel):
        return float((D[:, sel].min(1) * ww).sum())
    near = np.linalg.norm((P[:, None] - P[None]) * WT, axis=2) < 16.5     # two paints nobody can tell apart: one is enough

    def free(sel):
        return [j for j in range(len(P)) if j not in sel and not near[j][sel].any()]
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
                    if j not in sel:
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
    P = _paints()
    cols = [{"oz": [[float(v) for v in P[i]]], "yol": "ton", "ad": TONES[i][0], "hex": TONES[i][1]} for i in sel]
    L = label_lab(lab, cols, mm)
    im = renk._shrink(img, WORK)
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


def merge_weak(L: np.ndarray, P: np.ndarray, mm: float, min_mm2: float = 9.0) -> np.ndarray:
    """Every shape has to earn its place: the less it differs from its neighbour, the larger it must be."""
    for _ in range(4):
        # connected shapes of one paint
        lab = np.zeros(L.shape, np.int32)
        n = 0
        for c in range(len(P)):
            cc, k = ndi.label(L == c)
            lab[cc > 0] = cc[cc > 0] + n
            n += k
        lab -= 1
        area = np.bincount(lab.ravel(), minlength=n) * mm * mm
        col = np.zeros(n, int)
        col[lab.ravel()] = L.ravel()
        pa, pb = [], []
        for s1, s2 in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                       ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
            k = lab[s1] != lab[s2]
            pa.append(lab[s1][k])
            pb.append(lab[s2][k])
        a, b = np.concatenate(pa + pb), np.concatenate(pb + pa)
        pair, cnt = np.unique(a.astype(np.int64) * n + b, return_counts=True)
        x, y = pair // n, pair % n
        d = np.linalg.norm((P[col[x]] - P[col[y]]) * WT, axis=1)
        need = np.clip(min_mm2 * (46.0 / np.maximum(d, 8.0)) ** 2.1, min_mm2, 300.0)
        # the neighbour it is most like, weighted by how long they touch
        score = cnt / np.maximum(d, 6.0) ** 1.3
        o = np.lexsort((score, x))
        x, y, need, last = x[o], y[o], need[o], np.r_[x[o][1:] != x[o][:-1], True]
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
    stay = ndi.minimum_filter(ndi.maximum_filter(stay, 3), 3)
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
