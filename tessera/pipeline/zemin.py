"""Taking the background away from a drawing or poster: what is left becomes the mosaic's outline.

The background is found on the picture's own colour areas: the colour that runs along the frame's edge, and
the areas of it that the edge reaches. A clear verdict says when this cannot be trusted; then nothing is taken
away and the user picks the areas by clicking.

zemin: {"tiklar": [[x, y, "tek"|"renk"], ...],   clicks as fractions of the frame (the cut part of the picture);
                                                 "tek" turns one area, "renk" every area of that colour
        "cizgiler": [["kalsin"|"gitsin", [[x, y], ...]], ...]}   photographs: lines drawn over what stays / goes
       None = the background stays.

Photographs and paintings (experimental) are cut differently: colour models of figure and background and a
minimum cut, steered by the lines the user draws.
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from typing import Optional

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

SIDE = 1000              # px, longest side the background is looked for at
NOMINAL_MM = 300.0       # the sizes below are meant at this width: the result does not change with the size
MAX_CLICKS = 200
MAX_LINES, MAX_POINTS = 60, 400
LINE_W = 0.02            # a drawn line is this wide, as a share of the frame's longer side
EIGHT = np.ones((3, 3), bool)
RANK = {"iyi": 0, "orta": 1, "zor": 2}


# ------------------------------------------------------------------ the wish, cleaned
def clean(z):
    """A usable zemin, or None."""
    if not isinstance(z, dict):
        return None
    out = []
    for t in (z.get("tiklar") or [])[:MAX_CLICKS]:
        try:
            x, y = (min(1.0, max(0.0, float(v))) for v in t[:2])
        except (TypeError, ValueError, IndexError):
            continue
        out.append([round(x, 4), round(y, 4), "renk" if len(t) > 2 and t[2] == "renk" else "tek"])
    lines = []
    for t in (z.get("cizgiler") or [])[:MAX_LINES]:
        try:
            pts = [[round(min(1.0, max(0.0, float(x))), 4), round(min(1.0, max(0.0, float(y))), 4)]
                   for x, y in t[1][:MAX_POINTS]]
        except (TypeError, ValueError, IndexError):
            continue
        if pts:
            lines.append(["kalsin" if t[0] == "kalsin" else "gitsin", pts])
    return {"tiklar": out, "cizgiler": lines} if lines else {"tiklar": out}


# ------------------------------------------------------------------ small tools
def _disk(m, r, grow):
    if r < 0.5:
        return m
    if grow:
        return ndi.distance_transform_edt(~m) <= r
    return ndi.distance_transform_edt(m) > r


def _ring(hw, t):
    r = np.zeros(hw, bool)
    r[:t] = r[-t:] = True
    r[:, :t] = r[:, -t:] = True
    return r


def _touching(m, seed):
    """The connected parts of m that hold a seed pixel."""
    cc, n = ndi.label(m)
    keep = np.zeros(n + 1, bool)
    keep[np.unique(cc[seed & m])] = True
    keep[0] = False
    return keep[cc]


# ------------------------------------------------------------------ finding it
def flat(img, lab, close_mm=1.2, speck_mm2=30.0):
    """-> (keep, info). img: the picture at SIDE, lab: its colour areas (same size)."""
    from .renk import lab32
    h, w = lab.shape
    mm = NOMINAL_MM / w
    t = max(2, round(0.004 * w))
    R = _ring((h, w), t)
    n = int(lab.max()) + 1
    share = np.bincount(lab[R], minlength=n) / R.sum()
    main = int(share.argmax())
    L = ndi.gaussian_filter(lab32(img), (1, 1, 0))
    mu = np.array([np.median(L[lab == i], axis=0) if (lab == i).any() else (0, 0, 0) for i in range(n)])
    bg = [i for i in range(n) if i == main or (share[i] > 0.02 and np.linalg.norm(mu[i] - mu[main]) < 16)]
    M = np.isin(lab, bg)
    raw = _touching(M, R)
    # a gap in an outline narrower than close_mm does not let the background in
    r = close_mm / 2 / mm
    q = int(r) + 2
    core = _disk(np.pad(M, q, mode="edge"), r, False)[q:-q, q:-q]
    core = _touching(core, R)
    out = _touching(_disk(core, r + 1.0, True) & raw, core)
    # crumbs inside the background (dust, a signature) go too
    cc, k = ndi.label(~out, EIGHT)
    small = np.bincount(cc.ravel()) * mm * mm < speck_mm2
    small[0] = False
    out |= small[cc]
    # see-through loops: enclosed, the same paint, and only a thin cord away from the outside all around
    near = ndi.distance_transform_edt(~out) * mm
    cc, k = ndi.label(M & ~out)
    tone = np.median(L[out], axis=0) if out.any() else 0
    loop = np.zeros_like(out)
    for i, sl in enumerate(ndi.find_objects(cc), 1):
        m = cc[sl] == i
        if m.sum() * mm * mm < 4:
            continue
        pm = np.pad(m, 1)
        rim = (pm & ~ndi.binary_erosion(pm))[1:-1, 1:-1]
        if (near[sl][rim] < 4.5).mean() > 0.8 and np.linalg.norm(np.median(L[sl][m], axis=0) - tone) < 8:
            loop[sl] |= m
    info = {"border": float(share[bg].sum()), "leak": float(raw.sum() / max(out.sum(), 1)),
            "inside": float((M & ~out).sum() / max(out.sum(), 1)),
            "pieces": int((np.bincount(ndi.label(out)[0].ravel())[1:] / (h * w) > 0.002).sum())}
    return ~(out | loop), info


def judge(img, keep, info) -> tuple:
    """How far the result can be trusted: ("iyi" | "orta" | "zor", [reasons in plain words])."""
    from .renk import lab32
    h, w = keep.shape
    L = ndi.gaussian_filter(lab32(img), (0.8, 0.8, 0))
    out = ~keep
    share = float(out.mean())
    d_in, d_out = ndi.distance_transform_edt(keep), ndi.distance_transform_edt(out)
    # contrast across the cut line: the same place a little inside and a little outside
    g = ndi.gaussian_filter(keep.astype(np.float32), 2.0)
    gy, gx = np.gradient(g)
    nrm = np.hypot(gy, gx) + 1e-6
    ys, xs = np.nonzero(keep & (d_in <= 1.0))
    ok = (ys > 6) & (ys < h - 7) & (xs > 6) & (xs < w - 7)
    ys, xs = ys[ok], xs[ok]
    weak = 1.0
    if len(ys):
        ny, nx = gy[ys, xs] / nrm[ys, xs], gx[ys, xs] / nrm[ys, xs]
        a = L[np.clip((ys + 4 * ny).round().astype(int), 0, h - 1), np.clip((xs + 4 * nx).round().astype(int), 0, w - 1)]
        b = L[np.clip((ys - 4 * ny).round().astype(int), 0, h - 1), np.clip((xs - 4 * nx).round().astype(int), 0, w - 1)]
        de = np.maximum(np.linalg.norm(a - b, axis=1), np.linalg.norm(L[ys, xs] - b, axis=1))
        weak = float((de < 9).mean())
    Lg = L[..., 0]
    far = out & (d_out > 8)
    busy = float((ndi.maximum_filter(Lg, 7) - ndi.minimum_filter(Lg, 7))[far].mean()) if far.any() else 99.0
    tort = len(ys) / (4 * np.sqrt(max(keep.sum(), 1)))
    cc, n = ndi.label(keep, EIGHT)
    sz = np.sort(np.bincount(cc.ravel())[1:])[::-1] / max(keep.sum(), 1) if n else np.array([0.0])
    why = []
    if share < 0.04:
        why.append("Kaldırılacak bir arka plan bulunamadı.")
    if share > 0.93:
        why.append("Neredeyse her şey arka plan sayıldı.")
    if info.get("leak", 1.0) > 1.25:
        why.append("Arka plan rengi resmin çizgilerinin arasına da akıyor.")
    if info.get("border", 1.0) < 0.45:
        why.append("Kenarlarda tek bir arka plan rengi yok.")
    if weak > 0.22:
        why.append("Kesilen yerin bir kısmında belirgin bir kenar yok.")
    if busy > 9:
        why.append("Kaldırılan yer düz değil, desenli.")
    if tort > 4.5:
        why.append("Dış çizgi çok girintili çıkıntılı çıkıyor.")
    if sz[0] < 0.5:
        why.append("Resim birçok parçaya bölünüyor.")
    if info.get("inside", 0.0) > 0.3:
        why.append("Arka planın rengi resmin içinde de var.")
    if info.get("pieces", 1) > 8:
        why.append("Arka plan birçok ayrı parçadan oluşuyor.")
    if info.get("stable", 1.0) < 0.9:
        why.append("Sonuç her denemede değişiyor.")
    if not why:
        return "iyi", []
    sure = len(why) == 1 and share >= 0.04 and info.get("leak", 1.0) < 1.6 and weak < 0.35
    return ("orta" if sure else "zor"), why


# ------------------------------------------------------------------ photographs (experimental)
def _kmeans(x, k, rng, it=12):
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(it):
        a = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.array([x[a == i].mean(0) if (a == i).any() else c[i] for i in range(k)])
    return a


class _Colours:
    """A handful of Gaussian colours: how well a pixel fits the background, or the figure."""

    def __init__(self, x, k, rng):
        x = x if len(x) < 6000 else x[rng.choice(len(x), 6000, replace=False)]
        a = _kmeans(x, min(k, len(x)), rng)
        self.p = []
        for i in range(k):
            q = x[a == i]
            if len(q) > 12:
                S = np.cov(q.T) + np.eye(x.shape[1]) * 6
                self.p.append((len(q) / len(x), q.mean(0), np.linalg.inv(S), np.linalg.slogdet(S)[1]))

    def cost(self, x):
        best = np.full(x.shape[:-1], 1e6, np.float64)
        for w, m, P, d in self.p:
            v = x - m
            best = np.minimum(best, 0.5 * ((v @ P) * v).sum(-1) + 0.5 * d - np.log(w))
        return best


def _cut(n, a, b, wab, to_bg, to_fg):
    """The cheapest cut of a graph of n pixels: pairs (a, b) with the price wab of parting them, to_bg / to_fg
    the price of calling a pixel background / figure. -> bool, True = figure."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order, maximum_flow
    S, T = n, n + 1
    i = np.arange(n)
    lo = np.minimum(to_bg, to_fg)
    rows = np.concatenate([a, b, np.full(n, S), i])
    cols = np.concatenate([b, a, i, np.full(n, T)])
    val = np.concatenate([wab, wab, to_bg - lo, to_fg - lo])
    val = np.minimum(np.round(val * 60.0), 2 ** 24).astype(np.int32)
    k = val > 0
    G = coo_matrix((val[k], (rows[k], cols[k])), shape=(n + 2,) * 2).tocsr()
    res = G - maximum_flow(G, S, T).flow
    res.data = np.where(res.data > 0, res.data, 0)
    res.eliminate_zeros()
    m = np.zeros(n + 2, bool)
    m[breadth_first_order(res, S, return_predecessors=False)] = True
    return m[:n]


def _mincut(Dbg, Dfg, Wh, Wv):
    """The figure as the cheapest cut between "background" and "figure" (D*: cost of calling a pixel so)."""
    h, w = Dbg.shape
    idx = np.arange(h * w).reshape(h, w)
    a = np.concatenate([idx[:, :-1].ravel(), idx[:-1].ravel()])
    b = np.concatenate([idx[:, 1:].ravel(), idx[1:].ravel()])
    return _cut(h * w, a, b, np.concatenate([Wh.ravel(), Wv.ravel()]), Dbg.ravel(), Dfg.ravel()).reshape(h, w)


STEPS = ((0, 1, 1.0), (1, 0, 1.0), (1, 1, 0.5 ** 0.5), (1, -1, 0.5 ** 0.5))   # neighbours, diagonals too


def _band_cut(band, sure, Dbg, Dfg, X, gamma):
    """The cut again, only inside a band: what lies outside it is settled (sure: figure there). D*: the prices
    for the band's pixels; X: the colours (whole picture) the price of parting two neighbours comes from."""
    h, w = band.shape
    n = int(band.sum())
    idx = np.full((h, w), -1, np.int64)
    idx[band] = np.arange(n)
    to_bg, to_fg = Dbg.astype(np.float64).copy(), Dfg.astype(np.float64).copy()
    pairs = []
    for dy, dx, f in STEPS:
        p = (slice(0, h - dy), slice(max(0, -dx), w - max(0, dx)))
        q = (slice(dy, h), slice(max(0, dx), w + min(0, dx)))
        one = band[p] | band[q]
        d = ((X[p][one] - X[q][one]) ** 2).sum(-1)
        pairs.append((idx[p][one], idx[q][one], sure[p][one], sure[q][one], f, d))
    both = np.concatenate([d[(ip >= 0) & (iq >= 0)] for ip, iq, _, _, _, d in pairs])
    beta = 1 / (2 * float(both.mean()) + 1e-6) if len(both) else 0.0
    A, B, Wab = [], [], []
    for ip, iq, sp, sq, f, d in pairs:
        wgt = f * (gamma * np.exp(-beta * d) + 0.5)
        inner = (ip >= 0) & (iq >= 0)
        A.append(ip[inner]), B.append(iq[inner]), Wab.append(wgt[inner])
        # a pair half outside: parting it is a price on the inside pixel's choice
        for me, other, so in ((ip, iq, sq), (iq, ip, sp)):
            k = (me >= 0) & (other < 0)
            np.add.at(to_bg, me[k & so], wgt[k & so])
            np.add.at(to_fg, me[k & ~so], wgt[k & ~so])
    m = _cut(n, np.concatenate(A), np.concatenate(B), np.concatenate(Wab), to_bg, to_fg)
    res = sure.copy()
    res[band] = m
    return res


def photo(img, keep_hint=None, drop_hint=None, side=260, gamma=18.0, rounds=4, seed=3, fine=True):
    """-> (keep, info). The figure of a photograph: colour models of figure and background, refined by a
    minimum cut, its edge then laid along the picture's own edges and (fine) cut again at full size in a band
    around it. hints: bool masks (img size) the user drew; the green lines also pull what lies with them."""
    from .renk import _stats, guided, lab32, shrink
    rng = np.random.default_rng(seed)
    im = shrink(img, side)
    h, w = im.shape[:2]
    X = lab32(im)
    sm = ndi.gaussian_filter(X, (1.2, 1.2, 0))
    tex = np.sqrt(np.maximum(ndi.gaussian_filter(X[..., 0] ** 2, 2.5) - ndi.gaussian_filter(X[..., 0], 2.5) ** 2, 0))
    F = np.dstack([sm, 1.2 * tex])
    t = max(2, round(0.03 * min(h, w)))
    R = _ring((h, w), t)
    R[-t:] = False                                 # a figure often stands on the lower edge
    yy, xx = np.mgrid[0:h, 0:w]
    ell = ((xx - w / 2) / (0.36 * w)) ** 2 + ((yy - 0.55 * h) / (0.42 * h)) ** 2 < 1
    def small(m):
        if m is None or not m.any():
            return None
        return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.BOX)) > 0
    ph, pb = small(keep_hint), small(drop_hint)
    dh = ((X[:, 1:] - X[:, :-1]) ** 2).sum(-1)
    dv = ((X[1:] - X[:-1]) ** 2).sum(-1)
    beta = 1 / (2 * np.mean(np.concatenate([dh.ravel(), dv.ravel()])) + 1e-6)
    Wh, Wv = gamma * np.exp(-beta * dh) + 0.5, gamma * np.exp(-beta * dv) + 0.5
    fg, same = ell.copy(), []
    if ph is not None:
        fg |= ph
    lean = _lean(sm, ph, pb if pb is not None else R) if ph is not None else None
    for it in range(rounds):
        sel_b = (~fg | R) if it else (R | ~_disk(ell, 0.10 * w, True))
        if pb is not None:
            sel_b |= pb
        if ph is not None:
            sel_b &= ~ph
        sel_f = fg if fg.sum() > 200 else ell
        if pb is not None:
            sel_f = sel_f & ~pb
        cb, cf = _Colours(F[sel_b], 5, rng).cost(F), _Colours(F[sel_f], 5, rng).cost(F)
        lo = np.minimum(cb, cf)
        cb, cf = np.clip(cb - lo, 0, 12), np.clip(cf - lo, 0, 12)
        if lean is not None:                       # what lies with the green lines leans to the figure
            cb, cf = cb + LEAN * np.maximum(lean, 0), cf + LEAN * np.maximum(-lean, 0)
        cf = np.where(R if ph is None else R & ~ph, 500.0, cf)       # the frame's edge is background
        if ph is not None:
            cb = np.where(ph, 500.0, cb)
        if pb is not None:
            cf = np.where(pb, 500.0, cf)
        new = _mincut(cb, cf, Wh, Wv)
        same.append(float((new & fg).sum() / max((new | fg).sum(), 1)))
        fg = new
        if fg.mean() < 0.01 or fg.mean() > 0.98:
            break
    fg = _main(fg, ph, pb)
    H, W = img.shape[:2]
    s = np.asarray(Image.fromarray((fg * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)).astype(np.float32) / 255
    L = lab32(img)
    I = ndi.gaussian_filter(L[..., 0], 0.8)
    guess = ndi.gaussian_filter(np.clip(guided(I, _stats(I, 6), s, 6, 30.0), 0, 1), 1.2) > 0.5
    info = {"stable": same[-1] if same else 0.0, "rough": guess}
    if fine and guess.any() and not guess.all():
        return _edge(L, guess, keep_hint, drop_hint, W / w, rng), info
    return guess, info


LEAN, KAPPA = 6.0, 4.0     # pull of the lines; how long a way crossing an edge is


def _lean(X, ph, pb):
    """-1..1: how much nearer a pixel is to the green lines (1) than to the red ones (-1), walking over the
    picture, where crossing an edge of its own is a long way."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    h, w = ph.shape
    idx = np.arange(h * w).reshape(h, w)
    a = np.concatenate([idx[:, :-1].ravel(), idx[:-1].ravel()])
    b = np.concatenate([idx[:, 1:].ravel(), idx[1:].ravel()])
    step = np.concatenate([np.linalg.norm(X[:, 1:] - X[:, :-1], axis=-1).ravel(),
                           np.linalg.norm(X[1:] - X[:-1], axis=-1).ravel()])
    G = coo_matrix((1.0 + KAPPA * step, (a, b)), shape=(h * w,) * 2).tocsr()
    far = []
    for m in (ph, pb):
        src = np.flatnonzero(m)
        if not len(src):
            return None
        far.append(dijkstra(G, directed=False, indices=src, min_only=True).reshape(h, w))
    df, db = far
    return (db ** 2 - df ** 2) / (db ** 2 + df ** 2 + 1e-6)


def _main(fg, ph=None, pb=None):
    """The biggest part, the parts the user marked green, their holes filled (not one marked red)."""
    cc, n = ndi.label(fg)
    if not n:
        return fg
    sz = np.bincount(cc.ravel())
    sz[0] = 0
    keep = cc == sz.argmax()
    if ph is not None:                             # every part the user marked stays
        keep |= np.isin(cc, np.unique(cc[ph])) & (cc > 0)
    hole = ndi.binary_fill_holes(keep) & ~keep
    if pb is not None:                             # a hole the user marked red stays open
        cc, n = ndi.label(hole)
        hole &= ~np.isin(cc, np.unique(cc[pb & hole]))
    return keep | hole


def _edge(L, guess, keep_hint, drop_hint, k, rng, gamma=12.0):
    """The edge cut again at the picture's own size, only in a band around the first guess (that was cut
    k times smaller): colours learnt just inside and just outside it, the price of a cut low where the
    picture has an edge of its own."""
    H, W = guess.shape
    r = max(3, round(2.5 * k))
    d = np.where(guess, ndi.distance_transform_edt(guess), ndi.distance_transform_edt(~guess))
    band = d <= r
    # where the figure stopped at the frame's edge strip (taken as background), the band reaches the edge
    t = max(2, round(0.03 * min(H, W)))
    yy, xx = np.ogrid[0:H, 0:W]
    to_edge = np.minimum(np.minimum(yy, H - 1 - yy), np.minimum(xx, W - 1 - xx))
    band |= (to_edge < t + 2) & ~guess & (d <= t + r + 2)
    ph = keep_hint if keep_hint is not None and keep_hint.shape == guess.shape else None
    pb = drop_hint if drop_hint is not None and drop_hint.shape == guess.shape else None
    # colour only: a texture measure is high right next to any strong edge, on both sides of it
    sm = ndi.gaussian_filter(L, (0.8, 0.8, 0))
    near = (d > r) & (d <= 5 * r)
    sel_f, sel_b = guess & near, ~guess & near & (to_edge >= t + 2)
    if ph is not None:
        sel_f, sel_b = sel_f | ph, sel_b & ~ph
    if pb is not None:
        sel_f, sel_b = sel_f & ~pb, sel_b | pb
    if sel_f.sum() < 300 or sel_b.sum() < 300:
        return guess
    Fb = sm[band]
    cb, cf = _Colours(sm[sel_b], 5, rng).cost(Fb), _Colours(sm[sel_f], 5, rng).cost(Fb)
    lo = np.minimum(cb, cf)
    cb, cf = np.clip(cb - lo, 0, 12), np.clip(cf - lo, 0, 12)
    if ph is not None:
        cb = np.where(ph[band], 500.0, cb)
    if pb is not None:
        cf = np.where(pb[band], 500.0, cf)
    out = _band_cut(band, guess, cb, cf, sm, gamma)
    # a pixel's worth of fringe off, then only what hangs together with the figure, without crumbs or pinholes
    out = ndi.gaussian_filter(out.astype(np.float32), 1.0) > 0.5
    return _main(out, ph, pb)


# ------------------------------------------------------------------ the picture and its areas
def frame_picture(path: str, kadraj) -> np.ndarray:
    """The cut part of the picture, longest side at most SIDE."""
    from ..resim import open_rgb
    from .kadraj import pixels, turned
    im = turned(open_rgb(path), kadraj)
    im = im.crop(pixels(kadraj, *im.size))
    k = min(1.0, SIDE / max(im.size))
    if k < 1:
        im = im.resize((max(8, round(im.width * k)), max(8, round(im.height * k))), Image.BOX)
    return np.asarray(im)


@lru_cache(maxsize=2)
def _found(path: str, mtime: float, kadraj: str) -> tuple:
    """(picture, colour areas or None for a photograph, automatic keep, verdict) of one frame."""
    from . import renk, sade
    img = frame_picture(path, json.loads(kadraj))
    own = renk.find(img)
    if sade.kind(img, lambda: own)[0] == "ton":   # a photograph or painting: no flat areas to go by
        lab = None
        keep, info = photo(img)
        # the fine edge pass lays any cut along some edge: the cut before it must pass the test too
        puan, why = max(judge(img, keep, info), judge(img, info.pop("rough"), info), key=lambda v: RANK[v[0]])
    else:
        lab = renk.label(img, own).astype(np.int32)
        keep, info = flat(img, lab)
        puan, why = judge(img, keep, info)
    if puan == "zor":                            # rather nothing than a wrong cut: the user picks
        keep = np.ones_like(keep)
    for a in (img, lab, keep):
        if a is not None:
            a.setflags(write=False)
    return img, lab, keep, puan, tuple(why)


def drawn(lines, kind: str, size) -> Optional[np.ndarray]:
    """The lines of one kind, as a mask of the given size (W, H)."""
    from PIL import ImageDraw
    W, H = size
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    r = max(2, round(LINE_W * max(W, H) / 2))
    for t, pts in lines:
        if t != kind:
            continue
        xy = [(x * (W - 1), y * (H - 1)) for x, y in pts]
        if len(xy) > 1:
            d.line(xy, fill=255, width=2 * r, joint="curve")
        for x, y in xy:
            d.ellipse((x - r, y - r, x + r, y + r), fill=255)
    m = np.asarray(im) > 0
    return m if m.any() else None


def click(keep, lab, x, y, whole: bool):
    """Turn the area under a click: the connected area of that colour in the same state, or (whole) every
    area of that colour in the same state."""
    h, w = keep.shape
    px, py = min(w - 1, int(x * w)), min(h - 1, int(y * h))
    same = (lab == lab[py, px]) & (keep == keep[py, px])
    if not whole:
        cc, _ = ndi.label(same)
        same = cc == cc[py, px]
    out = keep.copy()
    out[same] = not keep[py, px]
    return ndi.median_filter(out, 3)


def drop_threads(keep, lab, line: int):
    """After areas are taken away, the lines that only parted them would stand alone: a line pixel stays when
    a kept area that is not line lies within one line width of it."""
    mm = NOMINAL_MM / keep.shape[1]
    L = keep & (lab == line)
    body = keep & ~L
    cc, _ = ndi.label(body)
    body &= (np.bincount(cc.ravel()) * mm * mm >= 60)[cc]
    if not body.any():
        return keep
    core = ndi.distance_transform_edt(np.pad(lab == line, 1, mode="edge"))[1:-1, 1:-1]
    ridge = (lab == line) & (core >= ndi.maximum_filter(core, 3))
    if not ridge.any():
        return keep
    width = 2 * float(np.percentile(core[ridge], 60))
    near = ndi.distance_transform_edt(~body) <= 1.15 * width
    out = body | (L & near)
    cc, _ = ndi.label(out, EIGHT)
    sz = np.bincount(cc.ravel())
    ok = sz > 0.02 * sz[1:].max()
    ok[0] = False
    return ok[cc]


@lru_cache(maxsize=4)
def _kept(path: str, mtime: float, kadraj: str, zemin: str) -> tuple:
    from .parca import contour
    img, lab, keep, puan, why = _found(path, mtime, kadraj)
    z = json.loads(zemin)
    tiklar, lines = z["tiklar"], z.get("cizgiler") or []
    if lab is None:                              # a photograph: the lines steer the cut
        if lines:
            size = img.shape[1::-1]
            keep, _ = photo(img, drawn(lines, "kalsin", size), drawn(lines, "gitsin", size))
            keep.setflags(write=False)
        return keep, report(keep, puan, why, bool(lines), True)
    if tiklar:
        k = keep.copy()
        for x, y, how in tiklar:
            k = click(k, lab, x, y, how == "renk")
        n = int(lab.max()) + 1
        line = contour(lab, n)
        if line is not None and not k.all():
            k = drop_threads(k, lab, line)
        keep = k
        keep.setflags(write=False)
    return keep, report(keep, puan, why, bool(tiklar))


def report(keep, puan, why, picked: bool, foto: bool = False) -> dict:
    """What the window says about the background."""
    share = float(1 - keep.mean())
    if foto:
        fix = "Yanlış yerleri çizerek düzelt: yeşil çizgi kalsın, kırmızı çizgi gitsin."
        if picked:
            soz = ["Arka plan senin çizgilerine göre kaldırıldı.", fix]
            puan = "orta"
        elif puan == "iyi":
            soz = ["Fotoğrafta arka plan bulundu ve kaldırıldı.", fix]
        elif puan == "orta":
            soz = ["Arka plan kaldırıldı ama emin değilim: " + why[0].rstrip(".").lower() + ".", fix]
        else:
            soz = ["Bu fotoğrafta arka planı kendim bulamadım: " + why[0].rstrip(".").lower() + ".",
                   "Kalacak yerin üstüne yeşil, gidecek yerin üstüne kırmızı bir çizgi çek."]
        soz.append("Fotoğrafta arka plan kaldırma deneme aşamasında; kenarlar her zaman kusursuz çıkmaz.")
        return {"puan": puan, "soz": soz, "pay": round(share, 3), "yol": "foto"}
    if picked:
        soz = ["Arka plan senin seçimine göre kaldırıldı."] if share > 0.005 else ["Arka plandan hiçbir yer seçilmedi."]
        puan = "iyi"
    elif puan == "iyi":
        soz = ["Arka plan bulundu ve kaldırıldı. Yanlış giden ya da kalan bir yer varsa üstüne tıkla."]
    elif puan == "orta":
        soz = ["Arka plan kaldırıldı ama emin değilim: " + why[0].rstrip(".").lower() + ".",
               "Yanlış giden ya da kalan bir yer varsa üstüne tıkla."]
    else:
        soz = ["Arka planı kendim bulamadım: " + why[0].rstrip(".").lower() + ".",
               "Kaldırmak istediğin yere tıkla; Shift ile tıklarsan o renkteki her yer gider."]
    cc, n = ndi.label(keep, EIGHT)
    if share > 0.005 and n > 1:
        sz = np.bincount(cc.ravel())[1:] / max(keep.sum(), 1)
        k = int((sz > 0.01).sum())
        if k > 1:
            soz.append(f"Kalan resim {k} ayrı parça; her biri ayrı yapıştırılır.")
    return {"puan": puan, "soz": soz, "pay": round(share, 3), "yol": "duz"}


def keep(path: str, kadraj, zemin) -> tuple:
    """(kept area of the frame at SIDE, report)."""
    z = clean(zemin) or {"tiklar": []}
    return _kept(path, os.path.getmtime(path), json.dumps(kadraj), json.dumps(z))


def overlay(keep) -> bytes:
    """PNG for the window: what is taken away, covered and striped; what stays, clear."""
    import io
    h, w = keep.shape
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.zeros((h, w, 4), np.uint8)
    a[..., :3] = np.where((((xx + yy) // 5) % 3 == 0)[..., None], (150, 140, 128), (214, 208, 198))
    a[..., 3] = np.where(keep, 0, 225)
    edge = keep & ~ndi.binary_erosion(keep, iterations=2, border_value=1)
    a[edge] = (200, 30, 30, 255)                  # the new outline in the window's red
    b = io.BytesIO()
    Image.fromarray(a, "RGBA").save(b, "PNG", optimize=False, compress_level=3)
    return b.getvalue()


# ------------------------------------------------------------------ what is left, as an outline
def smooth_inside(alpha, size, res, sig_mm=0.5, min_w=1.6, hole_mm2=30.0, island_mm2=30.0):
    """The kept area at the cutting size: a smooth edge, no cord thinner than can be printed, no crumbs,
    no pinholes."""
    W, H = size
    f = ndi.gaussian_filter(alpha.astype(np.float32), 0.55)
    m = np.asarray(Image.fromarray(f).resize((W, H), Image.BICUBIC)) > 0.5
    sm = ndi.gaussian_filter(m.astype(np.float32), sig_mm / res, mode="nearest") > 0.5
    r = min_w / 2 / res
    q = int(3 * r) + 2
    P = np.pad(m, q, mode="edge")
    core = ndi.distance_transform_edt(P) > r
    opened = ndi.distance_transform_edt(~core) <= r + 1
    thin = (P & ~opened)[q:-q, q:-q]                    # cords: they stay, as wide as a stone must be
    cc, k = ndi.label(thin, EIGHT)
    if k:
        ok = np.zeros(k + 1, bool)
        for i, sl in enumerate(ndi.find_objects(cc), 1):
            ok[i] = math.hypot(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) * res > 4.0
        thin = ok[cc]
        d = ndi.distance_transform_edt(m)
        sk = thin & (d >= ndi.maximum_filter(d, 3) - 0.3)
        sm |= ndi.distance_transform_edt(~sk) <= r
    for flip, lim in ((False, island_mm2), (True, hole_mm2)):
        x = ~sm if flip else sm
        cc, k = ndi.label(x)
        bad = np.bincount(cc.ravel()) * res ** 2 < lim
        bad[0] = False
        sm = sm ^ bad[cc]
    return sm


def silhouette(keep, w_mm: float, h_mm: float, res: float):
    """The outline of what is kept (shapely, mm, y down), None when nothing is taken away."""
    import shapely
    from .kesim import outline
    if keep is None or keep.all():
        return None
    W, H = max(8, round(w_mm / res)), max(8, round(h_mm / res))
    res = w_mm / W
    m = smooth_inside(keep, (W, H), res)
    if not m.any():
        raise ValueError("Arka plan kaldırılınca geriye basılacak bir şey kalmıyor. Başka bir yere tıkla.")
    P = outline(m, 0, 0, res, 0.8)
    P = shapely.simplify(P, 0.05).buffer(0)
    return P.intersection(shapely.box(0, 0, w_mm, h_mm))


def bleed(rgb: np.ndarray, alpha: np.ndarray, rim: int = 2) -> np.ndarray:
    """The kept colours run out over what is taken away, so the background leaves no colour of its own
    (not even the blended rim along the cut)."""
    core = ndi.binary_erosion(alpha, iterations=rim) if rim else alpha
    if rim:                                       # thin cords have no core: they stay, but not a stray pixel
        cord = alpha & ~ndi.binary_dilation(core, iterations=rim + 1)   # in a corner, it would colour a whole area
        cc, _ = ndi.label(cord, EIGHT)
        big = np.bincount(cc.ravel()) >= (2 * rim + 1) ** 2
        big[0] = False
        core |= big[cc]
    if not core.any():
        return rgb
    i = ndi.distance_transform_edt(~core, return_distances=False, return_indices=True)
    return np.where(core[..., None], rgb, rgb[i[0], i[1]]).astype(np.uint8)
