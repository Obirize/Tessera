"""The picture's own main colours: which ones are there, and which pixel belongs to which.

A pixel is not judged alone but against its surroundings:
- soft light and shadow on one surface is not a colour of its own (lightness counts less than hue),
- a thin or small shape that stands out from what surrounds it (an ornament, a rope, an eye) is,
- the blended pixels along an edge belong to one of the two sides, never to a third colour.

find(img)            -> list of {"rgb", "pay", "oz"}: the main colours, largest first
label(img, colours)  -> index map, same height and width as img
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from ..renkler import MAIN, TONES  # noqa: F401 (kept outside the pipeline: the window needs them at start)
from .palette import hex_rgb, rgb_to_lab

W_L, W_LOCAL, W_C = 0.55, 0.6, 1.7      # weights: lightness, local contrast, hue
JOIN = 15.0                             # colours nearer than this are one colour
MAX_N = 8
WORK = 900                              # px, picture size used for finding the colours


def paint_lab(paints: list) -> np.ndarray:
    return rgb_to_lab(np.array([hex_rgb(h) for _, h in paints], np.uint8))


def _hue_gain(L) -> np.ndarray:
    """The same paint is less colourful in the shade: this brings the hue of a dark colour to good light."""
    return np.clip(45.0 / np.maximum(L, 16.0), 1.0, None) ** 0.7


def match(found: list) -> list:
    """Which main colour stands for each found colour; different colours stay different.
    -> [(name, hex)]"""
    from scipy.optimize import linear_sum_assignment
    raw, b = rgb_to_lab(np.array([c["rgb"] for c in found], np.uint8)), paint_lab(MAIN)
    k = _hue_gain(raw[:, 0])
    a = raw * np.stack([np.ones(len(raw)), k, k], 1)
    ca, cb = np.hypot(a[:, 1], a[:, 2]), np.hypot(b[:, 1], b[:, 2])
    warm_black = (raw[:, 0] < 19) & (np.hypot(raw[:, 1], raw[:, 2]) < 11) & (raw[:, 2] > -2)
    a[warm_black] = raw[warm_black]              # nearly black and a little brown or red: black
    ca[warm_black] = 0
    full = np.linalg.norm(a[:, None] - b[None], axis=2)
    # a coloured one: the hue decides, then lightness; how vivid it is counts half
    dot = (a[:, None, 1:] * b[None, :, 1:]).sum(2)
    cosh = np.clip(dot / np.maximum(np.outer(ca, cb), 1e-6), -1, 1)
    dh = 2 * np.sqrt(np.outer(ca, cb)) * np.sqrt((1 - cosh) / 2)
    hue = np.sqrt((0.45 * np.subtract.outer(a[:, 0], b[:, 0])) ** 2 + (0.5 * np.subtract.outer(ca, cb)) ** 2
                  + (1.3 * dh) ** 2)
    hue = np.where(cb[None] < 8, hue + 14, hue)
    full = np.where(cb[None] > np.where(ca < 5, 12, 24)[:, None], full + 60, full)      # a grey never becomes a colour
    cost = np.where((ca > 9)[:, None], hue, full)
    r, c = linear_sum_assignment(np.hstack([cost, cost + 45]))  # only when nothing else fits, one is used twice
    c = c % len(MAIN)
    out = [None] * len(found)
    for i, j in zip(r, c):
        out[i] = MAIN[j]
    return out


def name_of(rgb) -> str:
    """The nearest main colour's name (for a colour picked freely)."""
    return match([{"rgb": rgb}])[0][0]


def paint(found: list) -> list:
    """Gives every found colour the main colour that stands for it."""
    for c, (name, h) in zip(found, match(found)):
        c["ad"], c["hex"] = name, h
    return found


def side_by_side(fn, items, k: int = 4):
    """fn over items on k threads (numpy, scipy and PIL let go of the GIL), the results in order; at most k
    of them wait at a time."""
    items = list(items)
    with ThreadPoolExecutor(k) as ex:
        fut = [ex.submit(fn, x) for x in items[:k]]
        for i in range(len(items)):
            if i + k < len(items):
                fut.append(ex.submit(fn, items[i + k]))
            r, fut[i] = fut[i].result(), None
            yield r


def lab32(img: np.ndarray) -> np.ndarray:
    """A picture (uint8, or float 0..255) in Lab, float32."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return rgb_to_lab(img).astype(np.float32)


def blur(img: np.ndarray, sigma: float, **kw) -> np.ndarray:
    return ndi.gaussian_filter(img, (sigma, sigma, 0), **kw)


def spread_in(g: np.ndarray, size: int) -> np.ndarray:
    """Difference between the lightest and the darkest value around every pixel."""
    return ndi.maximum_filter(g, size) - ndi.minimum_filter(g, size)


def _stats(I: np.ndarray, r: int) -> tuple:
    mI = ndi.uniform_filter(I, 2 * r + 1)
    return mI, np.maximum(ndi.uniform_filter(I * I, 2 * r + 1) - mI * mI, 0)


def guided(I: np.ndarray, stats: tuple, p: np.ndarray, r: int, eps: float = 12.0) -> np.ndarray:
    """p smoothed along the surfaces of I (guided filter). stats: _stats(I, r)."""
    m = 2 * r + 1
    mI, var = stats
    mp = ndi.uniform_filter(p, m)
    a = (ndi.uniform_filter(I * p, m) - mI * mp) / (var + eps)
    b = mp - a * mI
    return ndi.uniform_filter(a, m) * I + ndi.uniform_filter(b, m)


def cells(x: np.ndarray, step: float) -> tuple:
    """Points (n, d) sorted into a grid: (cell of every point, points per cell)."""
    q = np.round(x / step).astype(np.int64)
    q -= q.min(0)
    _, inv, cnt = np.unique(np.ravel_multi_index(q.T, q.max(0) + 1), return_inverse=True, return_counts=True)
    return inv.ravel(), cnt


def sharpen_edges(img: np.ndarray, r: int) -> np.ndarray:
    """Every blended edge pixel becomes a copy of the darker or the lighter side (whichever it is
    closer to), taken from within r px. Flat areas and the cores of lines do not change."""
    L = img.astype(np.float32) @ np.array([.299, .587, .114], np.float32)
    lo, hi = ndi.minimum_filter(L, 2 * r + 1), ndi.maximum_filter(L, 2 * r + 1)
    edge = (hi - lo) > 14
    ys, xs = np.nonzero(edge)
    if not len(ys):
        return img
    H, W = L.shape

    def source(part):                            # every edge pixel on its own: the pixels go in four parts
        ys, xs = part
        v = L[ys, xs]
        want = np.where(hi[ys, xs] - v < v - lo[ys, xs], hi[ys, xs], lo[ys, xs])
        best = np.full(len(ys), 1e9, np.float32)
        by, bx = ys.copy(), xs.copy()
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                y, x = np.clip(ys + dy, 0, H - 1), np.clip(xs + dx, 0, W - 1)
                c = np.abs(L[y, x] - want) + 0.4 * (dy * dy + dx * dx) ** 0.5
                m = c < best
                best[m], by[m], bx[m] = c[m], y[m], x[m]
        return by, bx
    cut = np.linspace(0, len(ys), 5).astype(int)
    with ThreadPoolExecutor(4) as ex:
        got = list(ex.map(source, [(ys[a:b], xs[a:b]) for a, b in zip(cut[:-1], cut[1:])]))
    out = img.copy()
    out[ys, xs] = img[np.concatenate([g[0] for g in got]), np.concatenate([g[1] for g in got])]
    return out


def features(img: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """(H, W, 4): lightness, what stands out locally, and the two hue axes; all in Lab units.
    scale: img width / WORK, so that sizes mean the same at every resolution."""
    lab = lab32(img)
    sig = 0.9 * scale
    L = ndi.gaussian_filter(lab[..., 0], 0.6 * sig)
    # hue is read relative to lightness and smoothed along surfaces only, so a bright red does not tint
    # the black next to it
    k = _hue_gain(L)
    r = max(2, int(round(2.6 * sig)))
    st = _stats(L, r)
    f = max(1, int(round(scale)))                 # the wide windows run on a coarse copy
    Lc = L[::f, ::f]
    w = max(5, int(0.028 * WORK * scale / f)) | 1
    with ThreadPoolExecutor(4) as ex:             # four maps that do not need each other, side by side
        a, b = (ex.submit(lambda i: k * guided(L, st, lab[..., i], r), i) for i in (1, 2))
        op = ex.submit(lambda: ndi.gaussian_filter(ndi.grey_opening(Lc, w), w / 4))
        cl = ex.submit(lambda: ndi.gaussian_filter(ndi.grey_closing(Lc, w), w / 4))
        a, b, op, cl = a.result(), b.result(), op.result(), cl.result()
    if f > 1:
        hw = (L.shape[1], L.shape[0])
        op, cl = (np.asarray(Image.fromarray(x).resize(hw, Image.BILINEAR)) for x in (op, cl))
    out = np.clip(L - op, 0, 40) - np.clip(cl - L, 0, 40)
    out = np.clip(out * np.sqrt(k), -40, 40)      # contrast in the shade counts as much as in the light
    return np.stack([W_L * L, W_LOCAL * out, W_C * a, W_C * b], -1).astype(np.float32)


def _kmeans(x, w, k, rng, it=18) -> np.ndarray:
    """Cluster of every point, numbered 0.. without gaps."""
    cen = [x[rng.choice(len(x), p=w / w.sum())]]
    d = ((x - cen[0]) ** 2).sum(1)
    for _ in range(1, k):
        p = d * w
        s = p.sum()
        if s <= 0:
            break
        cen.append(x[rng.choice(len(x), p=p / s)])
        d = np.minimum(d, ((x - cen[-1]) ** 2).sum(1))
    cen = np.array(cen)
    for _ in range(it):
        lab = ((x[:, None] - cen[None]) ** 2).sum(-1).argmin(1)
        for i in range(len(cen)):
            m = lab == i
            if m.any():
                cen[i] = np.average(x[m], axis=0, weights=w[m])
    lab = ((x[:, None] - cen[None]) ** 2).sum(-1).argmin(1)
    return np.unique(lab, return_inverse=True)[1]


SEEN = np.array([1.0, 0.35, 1.0, 1.0])  # what makes two colours different to the eye: local contrast little


def _valley(x, w, mu_a, mu_b) -> float:
    """How empty it is between two clusters: 0 = one continuous run (shading), 1 = two separate paints."""
    v = (mu_b - mu_a) * SEEN
    L = float(v @ v)
    if L < 1e-6:
        return 0.0
    t = ((x - mu_a) * SEEN) @ v / L
    h = np.bincount(np.clip(((t + 0.5) * 12).astype(int), 0, 23), w, 24)     # -0.5 .. 1.5
    h = np.convolve(h, [.25, .5, .25], "same")
    pa, pb = h[3:9].max(), h[15:21].max()
    return float(np.clip(1 - h[9:15].min() / max(min(pa, pb), 1e-9), 0, 1))


def _merge(cell, w, lab, mu, cnt, seen, true, tint, n: Optional[int]):
    """Joins the nearest clusters until every colour is clearly different (or only n are left).
    -> group index of every cluster; tint is updated in place"""
    grp = np.arange(len(mu))
    mu, cnt, seen, true = mu.copy(), cnt.copy(), seen.copy(), true.copy()
    alive = np.ones(len(mu), bool)
    tot = cnt.sum()
    while alive.sum() > 2:
        ids = np.flatnonzero(alive)
        d = np.linalg.norm((mu[ids][:, None] - mu[ids][None]) * SEEN, axis=2)
        small = np.minimum.outer(cnt[ids], cnt[ids]) / tot
        # a tiny colour has to be more different to count; next to nothing is not a colour at all
        eff = d / np.where(small < 0.0012, 4.0, np.where(small < 0.004, 1.4, 1.0))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if 6.0 < d[i, j] < 1.5 * JOIN and small[i, j] > 0.004:
                    m = (grp[lab] == ids[i]) | (grp[lab] == ids[j])
                    eff[i, j] *= 0.75 + 0.95 * _valley(cell[m], w[m], mu[ids[i]], mu[ids[j]])
                eff[j, i] = eff[i, j]
        # a small colour and its shaded side: the same hue, only darker
        ab = seen[ids][:, 1:]
        ch = np.hypot(ab[:, 0], ab[:, 1])
        cosh = (ab @ ab.T) / np.maximum(np.outer(ch, ch), 1e-6)
        shade = ((cosh > 0.94) & (np.minimum.outer(ch, ch) > 6.5) & (np.abs(np.subtract.outer(ch, ch)) < 26)
                 & (np.abs(np.subtract.outer(seen[ids][:, 0], seen[ids][:, 0])) < 36) & (small < 0.03)
                 & (np.maximum.outer(cnt[ids], cnt[ids]) / tot < 0.08))
        eff = np.where(shade, 0.55 * eff, eff)
        # the same paint to the eye (an edge tint, print noise): joined before anything else
        de = np.linalg.norm(seen[ids][:, None] - seen[ids][None], axis=2)
        dt = np.linalg.norm(true[ids][:, None] - true[ids][None], axis=2)
        edgy = np.maximum.outer(tint[ids], tint[ids])     # lives along strong edges only: a tint, not a paint
        de = np.where(edgy, dt, de)
        same = (de < 7.5) | ((de < 13) & (small < 0.02)) | ((de < 18) & (small < 0.004))
        eff = np.where(same, 0.02 * de + 0.01 * eff / JOIN, eff)
        np.fill_diagonal(eff, 1e9)
        i, j = np.unravel_index(int(eff.argmin()), eff.shape)
        if n is None:
            if eff[i, j] > JOIN and alive.sum() <= MAX_N:
                break
        elif alive.sum() <= n:
            break
        a, b = ids[i], ids[j]
        if cnt[a] < cnt[b]:
            a, b = b, a
        mu[a] = (mu[a] * cnt[a] + mu[b] * cnt[b]) / (cnt[a] + cnt[b])
        true[a] = (true[a] * cnt[a] + true[b] * cnt[b]) / (cnt[a] + cnt[b])
        tint[a] &= tint[b]
        seen[a] = (seen[a] * cnt[a] + seen[b] * cnt[b]) / (cnt[a] + cnt[b])
        cnt[a] += cnt[b]
        alive[b] = False
        grp[grp == b] = a
    return grp


def shrink(img: np.ndarray, side: int) -> np.ndarray:
    k = side / max(img.shape[:2])
    if k >= 1:
        return img
    return np.asarray(Image.fromarray(img).resize((max(8, round(img.shape[1] * k)), max(8, round(img.shape[0] * k))),
                                                  Image.LANCZOS))


def prepare(img: np.ndarray, full_w: Optional[float] = None) -> tuple:
    """(sharpened picture, its features, scale). full_w: see label()."""
    scale = max((full_w or img.shape[1]) / WORK, 0.6)
    img = sharpen_edges(img, 2)
    return img, features(img, scale), scale


def find(img: np.ndarray, n: Optional[int] = None, seed: int = 7) -> list:
    """The main colours of a picture (uint8 RGB). n: exactly that many, None: as many as there are."""
    im, F, _ = prepare(shrink(img, WORK))
    x = F.reshape(-1, 4)
    rng = np.random.default_rng(seed)
    # one entry per occupied cell; large areas do not drown the small ones
    inv, cnt = cells(x, 2.5)
    ok = cnt >= max(2, len(x) * 1e-5)

    def per_cell(*channels):
        return (np.stack([np.bincount(inv, c.ravel()) for c in channels], 1) / cnt[:, None])[ok]
    cell, cw = per_cell(*x.T), cnt[ok].astype(float)
    xl = lab32(im)
    wide = sum(spread_in(xl[..., i], 7) ** 2 for i in range(3)) ** 0.5
    lab = _kmeans(cell, cw ** 0.5, min(26, len(cell)), rng)
    real = np.bincount(lab, cw)

    def per_cluster(v):
        return np.array([np.average(v[lab == i], axis=0, weights=cw[lab == i]) for i in range(len(real))])
    mu = per_cluster(cell)
    true = per_cluster(per_cell(*np.moveaxis(xl, -1, 0)))
    tint = per_cluster(per_cell(wide > 42))[:, 0] > 0.6
    seen = per_cluster(cell[:, [0, 2, 3]] / np.array([W_L, W_C, W_C]))     # plain Lab, hue as strong as in good light
    grp = _merge(cell, cw, lab, mu, real, seen, true, tint, n)
    # a shaded piece of a small, coloured shape belongs to that shape, not to the dark surface around it
    soft = np.array([0.35, 1.0, 1.0, 1.0])
    ids = np.unique(grp)
    core = {g: np.average(mu[grp == g], axis=0, weights=real[grp == g]) for g in ids}
    size = {g: real[grp == g].sum() / real.sum() for g in ids}
    for i in range(len(real)):
        g = grp[i]
        if real[i] > 0.5 * real[grp == g].sum() or tint[i]:
            continue
        for h in ids:
            if (h != g and size[h] < 0.08 and size[h] < size[g] and np.hypot(*core[h][2:]) > 11
                    and np.linalg.norm((mu[i] - core[h]) * soft) < 0.9 * np.linalg.norm((mu[i] - core[g]) * soft)):
                grp[i] = g = h
    parts = [list(mu[grp == g]) for g in np.unique(grp)]
    L = label_features(F, parts, 1.0)
    while n is None and len(parts) > 2:              # a colour that ends up with next to no pixels is dropped
        cover = np.bincount(L.ravel(), minlength=len(parts)) / L.size
        if cover.min() >= 0.0035:
            break
        parts.pop(int(cover.argmin()))
        L = label_features(F, parts, 1.0)
    out = []
    for i, c in enumerate(parts):
        s = L == i
        if not s.any():
            continue
        core = ndi.binary_erosion(s)
        px = im[core if core.sum() > 60 else s]
        out.append({"rgb": [int(v) for v in np.median(px, axis=0)], "pay": round(float(s.mean()), 4),
                    "oz": [[round(float(v), 2) for v in q] for q in c]})
    out.sort(key=lambda o: -o["pay"])
    return out


def label_features(F: np.ndarray, parts: list, scale: float) -> np.ndarray:
    """parts: for every colour the list of its feature centres (a colour may have a lit and a shaded one)."""
    I = F[..., 0] / W_L
    r = max(1, int(round(1.6 * scale)))
    st = _stats(I, r)

    def cost_of(cs):
        d = None
        for c in cs:
            e = ((F - np.asarray(c, np.float32)) ** 2).sum(-1)
            d = e if d is None else np.minimum(d, e)
        return guided(I, st, np.minimum(np.sqrt(d), 45.0), r)
    with ThreadPoolExecutor(4) as ex:            # numpy and scipy let go of the GIL: the colours side by side
        cost = np.stack(list(ex.map(cost_of, parts)))
    out = cost.argmin(0).astype(np.int16)
    # lines that fade for a moment are followed through (a rope in the shade stays one rope)
    best = cost.min(0)
    n = out.size
    for i in np.argsort([-(out == k).sum() for k in range(len(parts))])[1:]:
        s = out == i
        side = np.sign(np.mean([c[1] for c in parts[i]]))
        if s.sum() > 0.2 * n or not s.any():
            continue
        line = F[..., 1] * side
        near = ndi.binary_dilation(s, iterations=max(2, int(round(4 * scale)))) & (line > 1.0)
        weak = s | ((cost[i] - best < 13.0) & near)
        cc, m = ndi.label(weak, np.ones((3, 3)))
        sz = np.bincount(cc.ravel(), minlength=m + 1).astype(float)
        st_ = np.bincount(cc.ravel(), s.ravel(), m + 1)
        keep = st_ > 0.2 * sz
        keep[0] = False
        out[keep[cc]] = i
    return out


def label(img: np.ndarray, colours: list, full_w: Optional[float] = None) -> np.ndarray:
    """Which of the colours every pixel belongs to. full_w: the whole picture's width in these pixels
    when img is only a part of it."""
    _, F, scale = prepare(img, full_w)
    return label_features(F, [c["oz"] for c in colours], scale)


def enlarge(lab: np.ndarray, n: int, size_wh) -> np.ndarray:
    """Label map at another size, with smooth borders instead of pixel steps."""
    w, h = size_wh
    if lab.shape == (h, w):
        return lab
    best = np.full((h, w), -1.0, np.float32)
    out = np.zeros((h, w), np.int16)

    def big(c):
        m = lab == c
        if not m.any():
            return None
        f = ndi.gaussian_filter(m.astype(np.float32), 0.55, truncate=3.0)
        return np.asarray(Image.fromarray(f).resize((w, h), Image.BICUBIC))
    for c, f in enumerate(side_by_side(big, range(n))):     # made side by side, laid on top of each other in order
        if f is None:
            continue
        k = f > best
        out[k], best[k] = c, f[k]
    return out
