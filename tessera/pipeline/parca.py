"""Exactly N stones from a map of colour areas.

1. reduce()    too many areas: the least visible shape goes to its neighbour, or the line between two areas of one
               colour goes (both become one area); cheapest change first. A small shape that stands out (a nose,
               an eye: features()) keeps its colour; for it the number may come out a few stones higher
2. allot()     every area gets its number of stones (highest averages: stone sizes as even as possible)
3. divide()    few stones: balanced cuts across the shape (Fiedler level sets); many: the program's own rows,
               stone size searched per area, then trimmed to the exact number
4. outlines()  polygons

base: a palette colour that makes no stones (it is the one-piece plate the stones sit in) and is not counted.
Label n in a colour map is what lies outside the artwork: never a stone, never given away, never grown.
"""
import heapq
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import shapely
from PIL import Image
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix, csc_matrix, identity
from scipy.sparse.linalg import splu

from . import kesim, renk
from .tiling import _adjacency

WT = np.array([1.15, 0.95, 0.95])
PLATE = 235.0            # largest stone side that fits a 256 mm printer
HAND_A = 16.0            # smallest stone that texture cuts may make, mm2 (about 4 x 4 mm)
HAND_W = 4.0
ROWS_N, ROWS_W = 40, 5.5
# a small shape that stands out (nose, eye): at least this big (and 1/2000 of the picture) and wide (mm2, mm), this
# far from every neighbour's colour; giving it away costs as much as giving away a shape of FEAT_AS mm2 (one stone
# that stays whole)
FEAT_A, FEAT_W, FEAT_DE, FEAT_AS = 8.0, 2.0, 25.0, kesim.FEATURE_A


def comps(lab, n, skip=None):
    """(component id of every pixel, colour of every component); id -1 where lab == skip"""
    out = np.full(lab.shape, -1, np.int32)
    col = []
    for c in range(n):
        if c == skip:
            continue
        cc, k = ndi.label(lab == c)
        m = cc > 0
        out[m] = cc[m] - 1 + len(col)
        col += [c] * k
    return out, np.array(col, int)


# ------------------------------------------------------------------ 1. fewer areas
def features(L, col, n, plab, mm, base=None, x=None, y=None, dE=None):
    """Small shapes that stand out: a colour clearly unlike every neighbour's, big enough to see. L, col: comps
    (lab, n + 1); plab: Lab values with one more row for the outside. x, y, dE: the neighbour pairs, when known."""
    K = len(col)
    area = np.bincount(L.ravel(), minlength=K) * mm * mm
    if x is None:
        lo, hi, _ = _adjacency(L, K)
        x, y = np.r_[lo, hi], np.r_[hi, lo]
        k = (col[x] < n) & (col[y] < n)
        x, y = x[k], y[k]
        dE = np.linalg.norm((plab[col[x]] - plab[col[y]]) * WT, axis=1)
    near = np.full(K, np.inf)
    np.minimum.at(near, x, dE)
    a0 = max(FEAT_A, 0.0005 * area[col < n].sum())             # crumbs of a large picture are no features
    out = (col < n) & (area >= a0) & (area < FEAT_AS) & (near >= FEAT_DE) & np.isfinite(near)
    if base is not None:
        out &= col != base
    box = ndi.find_objects(L + 1)
    for i in np.flatnonzero(out):
        m = np.pad(L[box[i]] == i, 1)
        out[i] = 2 * ndi.distance_transform_edt(m).max() * mm >= FEAT_W
    return out


def _ops(lab, n, plab, mm, width_mm, base):
    """Every possible change, cheapest first: (cost, kind, data, a feature is given away). Works on the coarse map."""
    L, col = comps(lab, n + 1)
    K = len(col)
    area = np.bincount(L.ravel(), minlength=K) * mm * mm
    lo, hi, cnt = _adjacency(L, K)
    x, y, c = np.r_[lo, hi], np.r_[hi, lo], np.r_[cnt, cnt].astype(float)
    plab = np.vstack([plab, [[0.0, 0.0, 0.0]]])
    art = col < n
    total = area[art].sum()
    k = art[x] & art[y]
    x, y, c = x[k], y[k], c[k]
    dE = np.linalg.norm((plab[col[x]] - plab[col[y]]) * WT, axis=1)
    feat = features(L, col, n, plab, mm, base, x, y, dE)
    cy, cx = (np.bincount(L.ravel(), g.ravel(), K) for g in np.mgrid[0:lab.shape[0], 0:lab.shape[1]].astype(float))
    npx = np.bincount(L.ravel(), minlength=K)
    cen = np.stack([cy / npx, cx / npx], 1) * mm
    share = np.bincount(col, area, n + 1) / total
    # many look-alikes next to each other (windows, a lattice): each one matters less
    crowd = np.ones(K)
    for i in range(K):
        near = (col == col[i]) & (np.linalg.norm(cen - cen[i], axis=1) < 4 * math.sqrt(area[i]) + 0.05 * width_mm)
        near &= (area < 2.5 * area[i]) & (area > 0.4 * area[i])
        crowd[i] = max(1, near.sum())
    mid = np.array(lab.shape) * mm / 2
    off = np.abs((cen - mid) / mid).max(1)                     # 0 = centre, 1 = frame
    rare = np.clip((0.08 / np.maximum(share[col], 1e-4)) ** 0.4, 1.0, 2.5)
    size = lambda a, p: (a / total * 1000.0) ** p         # share of the picture in thousandths
    weight = rare / crowd ** 0.4 * np.where(off < 0.6, 1.25, 1.0)
    ops = []
    o = np.lexsort((c / np.maximum(dE, 6.0) ** 1.3, x))
    xs, ys, ds = x[o], y[o], dE[o]
    last = np.r_[xs[1:] != xs[:-1], True] if len(xs) else np.zeros(0, bool)
    # an area larger than any feature, or the last area of a colour, that would clearly change colour: that hurts
    # more than a lost feature
    left = np.bincount(col[art], minlength=n + 1)
    big = (area >= FEAT_AS) | (left[col] == 1)
    dear = set()
    for i, j, d in zip(xs[last], ys[last], ds[last]):
        if base is not None and col[i] == base:
            continue                                           # the plate is no stone: giving it away gains nothing
        a = max(area[i], FEAT_AS) if feat[i] else area[i]          # a nose weighs like a whole stone
        ops.append((size(a, 0.5) * d * weight[i], 0, (int(i), int(j)), bool(feat[i])))
        if big[i] and not feat[i] and d >= FEAT_DE:
            dear.add(len(ops) - 1)
    # ... so those go only after every feature (all dearer by one factor: their order stays)
    top = max((t[0] for t in ops if t[3]), default=0.0)
    low = min((ops[q][0] for q in dear), default=np.inf)
    m = max(1.0, 1.001 * top / low)
    ops = [(t[0] * m,) + t[1:] if q in dear else t for q, t in enumerate(ops)]
    # the line between two areas of one colour
    T = max(2.0, (0.022 if base is None else 0.03) * math.sqrt(total))
    def lines_of(job):
        a, sl = job
        found = []
        if sl is None or not art[a]:
            return found
        sl = tuple(slice(max(0, s.start - 2), s.stop + 2) for s in sl)
        v = L[sl]
        m = v == a
        nb = np.unique(v[ndi.binary_dilation(m) & ~m])
        if len(nb) < 2 or len(np.unique(col[nb])) == len(nb):
            return found
        d1, (iy, ix) = ndi.distance_transform_edt(m, return_indices=True)
        halo = np.where(m, v[iy, ix], -1)
        key = np.full(m.shape, -1, np.int64)                   # the middle of the line, with the pair it separates
        for ax in (0, 1):
            s1 = (slice(None), slice(None, -1)) if ax else (slice(None, -1), slice(None))
            s2 = (slice(None), slice(1, None)) if ax else (slice(1, None), slice(None))
            k = m[s1] & m[s2] & (halo[s1] != halo[s2])
            kk = np.minimum(halo[s1], halo[s2]).astype(np.int64)[k] * K + np.maximum(halo[s1], halo[s2])[k]
            key[s1][k] = kk
            key[s2][k] = kk
        mid_px = key >= 0
        if not mid_px.any():
            return found
        jy, jx = ndi.distance_transform_edt(~mid_px, return_distances=False, return_indices=True)
        w = 2 * d1[jy, jx] * mm
        kn = key[jy, jx]
        good = m & (w < T) & (col[kn // K] == col[kn % K]) & art[kn // K]
        if not good.any():
            return found
        ids, inv = np.unique(kn[good], return_inverse=True)
        sa = np.bincount(inv) * mm * mm
        ww = np.bincount(inv, w[good]) / np.bincount(inv)
        for q, pair in enumerate(ids):
            i, j = int(pair // K), int(pair % K)
            if sa[q] < 1.5 * ww[q] ** 2 or sa[q] < 6 * mm * mm:
                continue
            d = float(np.linalg.norm((plab[col[a]] - plab[col[i]]) * WT))
            seg = good & (kn == pair)
            found.append((size(sa[q], 0.5) * d * max(rare[i], 1.0), 1, (i, j, a, sl, seg, float(np.percentile(w[seg], 90)) / mm), False))
        return found

    with ThreadPoolExecutor(6) as ex:
        for f in ex.map(lines_of, enumerate(ndi.find_objects(L + 1))):
            ops += f
    ops.sort(key=lambda t: t[0])
    return ops, L, col


def bridge(sm, L, i, j, a, sl, seg, w, colour):
    """The line between the areas i and j takes their colour - only what lies between the two: where they end
    the line ends too, in a smooth curve."""
    r = 0.55 * w + 1.0
    pad = int(3 * r) + 2
    ys, xs = np.nonzero(seg)
    y0, x0 = sl[0].start + int(ys.min()), sl[1].start + int(xs.min())
    y1, x1 = sl[0].start + int(ys.max()) + 1, sl[1].start + int(xs.max()) + 1
    q = slice(max(0, y0 - pad), y1 + pad), slice(max(0, x0 - pad), x1 + pad)
    v = L[q]

    def closed(m):
        d = np.pad(ndi.distance_transform_edt(~m) <= r, 1)
        return ndi.distance_transform_edt(d)[1:-1, 1:-1] > r
    fill = closed((v == i) | (v == j)) & ~closed(v == i) & ~closed(v == j) & (v == a)
    if fill.any():
        sm[q][fill] = colour
    else:
        sm[sl][seg] = colour


def reduce(lab, n, plab, res, K, base=None, extra=None):
    """(colour map with at most K + over areas (not counting the base colour), over). over > 0 only where the
    next change would give away a small shape that stands out; never more than extra (None: see below)."""
    if kesim.count_areas(lab, n, base) <= K:
        return lab, 0
    width = lab.shape[1] * res
    f = max(1, int(min(0.8, math.sqrt(lab.size * res * res / 140_000)) / res + 0.05))
    sm = lab[f // 2::f, f // 2::f].astype(np.int16).copy()
    mm = res * f
    sm = kesim.drop_specks(sm, n, max(2, 3.0 / mm ** 2), n)
    start = sm.copy()
    # A nose or an eye is not given away to reach the number: up to `extra` stones more than asked for (the
    # window shows the real number) are better than a lost feature. Asked for even fewer, the least visible goes.
    if extra is None:
        extra = max(1, min(3, K // 10))
    keep = stuck = False
    prev = None
    for _ in range(30):
        k = kesim.count_areas(sm, n, base)
        if prev is not None and k >= prev[0]:
            # the lines taken away cut other areas in two (back and forth without end): undo, and from now on
            # a line goes only where that leaves fewer areas
            k, sm = prev
            stuck = True
        if k <= K or keep:
            break
        ops, L, col = _ops(sm, n, plab, mm, width, base)
        if not ops:
            break
        prev = (k, sm.copy())
        todo = max(1, (k - K + 1) // 2) if k - K > 3 else k - K
        used, top, first, took = set(), 0.0, None, 0
        for cost, kind, data, feat in ops:
            if todo <= 0 and cost > 1.12 * top:                # look-alikes (both eyes) go together
                break
            if used and cost > (2.0 if k < 2 * K else 2.8) * first:                    # the cheap ones are taken: look again first
                break
            if data[0] in used or data[1] in used:
                continue
            if feat and k - took <= K + extra:                 # the cheapest change left gives a feature away
                keep = True
                break
            if kind == 0:
                i, j = data
                sm[L == i] = col[j]
            else:
                i, j, a, sl, seg, w = data
                was = sm.copy() if stuck else None
                bridge(sm, L, i, j, a, sl, seg, w, col[i])
                if stuck and kesim.count_areas(sm, n, base) >= kesim.count_areas(was, n, base):
                    sm = was
                    continue
            used |= {i, j}
            first = cost if first is None else first
            took += 1
            top = cost
            todo -= 1
        if not took and not keep:
            break
        sm = kesim.drop_specks(sm, n, max(2, 3.0 / mm ** 2), n)
    changed = ndi.binary_dilation(sm != start, iterations=2)
    H, W = lab.shape
    changed = ndi.binary_dilation(changed, iterations=max(1, int(0.8 / mm)))
    g0, pol = np.full(sm.shape, -1, np.float32), sm.copy()
    for c in range(n + 1):
        g = ndi.gaussian_filter((sm == c).astype(np.float32), min(0.7, 0.004 * width) / mm)
        kk = g > g0
        pol[kk], g0[kk] = c, g[kk]
    big = renk.enlarge(pol, n + 1, (W, H))
    ch = np.asarray(Image.fromarray(changed.view(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
    out = same_edge(np.where(ch, big, lab).astype(np.int16), lab, n)
    out = kesim.drop_specks(out, n, kesim.MIN_A / res ** 2, n)
    # crumbs the change left behind: the smallest areas go until the number fits; features only last, and not
    # below K + extra
    pl = np.vstack([plab, [[0.0, 0.0, 0.0]]])
    for _ in range(3):
        k = kesim.count_areas(out, n, base)
        if k <= K:
            break
        L, col = comps(out, n + 1)                    # the outside's areas come last
        sz = np.bincount(L.ravel(), minlength=len(col)).astype(float)
        art = col < n
        if base is not None:
            sz[col == base] = 1e12
        t = min(np.sort(sz[art])[k - K - 1] + 1, 60 / res ** 2)
        feat = features(L, col, n, pl, res, base) & (sz < t)
        if not feat.any():
            out = kesim.drop_specks(out, n, t, n)
            continue
        o = np.lexsort((sz, feat))
        o = o[art[o] & (sz[o] < t)][: k - K]
        keep |= bool((feat[o] & (k - np.arange(len(o)) <= K + extra)).any())
        o = o[~feat[o] | (k - np.arange(len(o)) > K + extra)]
        if not len(o):
            break
        out = _give(out, L, o, n)
    over = min(extra, kesim.count_areas(out, n, base) - K) if keep else 0
    return out, max(0, over)


def _give(lab, L, ids, n):
    """The areas ids of L take the colour that surrounds them most (not the outside)."""
    H, W = lab.shape
    box = ndi.find_objects(L + 1)
    for i in ids:
        sl = box[i]
        sl = slice(max(0, sl[0].start - 1), min(H, sl[0].stop + 1)), slice(max(0, sl[1].start - 1), min(W, sl[1].stop + 1))
        m = L[sl] == i
        v = lab[sl]
        ring = ndi.binary_dilation(m) & ~m & (v != n)
        if ring.any():
            v[m] = np.bincount(v[ring]).argmax()
    return lab


def same_edge(lab, before, n):
    """The outside exactly as it was before."""
    was = before == n
    if not was.any():
        return lab
    lost = (lab == n) & ~was
    if lost.any():
        i = ndi.distance_transform_edt(lab == n, return_distances=False, return_indices=True)
        lab = np.where(lost, lab[i[0], i[1]], lab)
    lab[was] = n
    return lab


def calm(lab, n, res, a_star, strength=0.06):
    """Ragged borders of large areas are smoothed (photographs); areas smaller than a stone keep theirs."""
    s = strength * math.sqrt(a_star)
    if s < 0.5:
        return lab
    f = max(1, int(s / 2 / res))
    sm = lab[f // 2::f, f // 2::f]
    best = np.full(sm.shape, -1, np.float32)
    out = sm.copy()
    for c in range(n + 1):
        m = sm == c
        if m.any():
            g = ndi.gaussian_filter(m.astype(np.float32), s / res / f)
            k = g > best
            out[k], best[k] = c, g[k]
    out = renk.enlarge(out.astype(np.int16), n + 1, (lab.shape[1], lab.shape[0]))
    L, col = comps(lab, n + 1)
    sz = np.bincount(L.ravel()) * res ** 2
    thin = np.zeros(len(sz), bool)
    for i, sl in enumerate(ndi.find_objects(L + 1)):
        m = np.pad(L[sl] == i, 1)
        thin[i] = 2 * ndi.distance_transform_edt(m).max() * res < 6 * s
    keep = ((sz < 0.5 * a_star) | thin)[L]
    out = same_edge(np.where(keep, lab, out), lab, n)
    return kesim.drop_specks(out, n, max(kesim.MIN_A, 0.04 * a_star) / res ** 2, n)


# ------------------------------------------------------------------ 2. how many stones for which area
def allot(A, N, cap, need):
    n = need.copy()
    left = N - int(n.sum())
    h = [(-A[i] / math.sqrt(n[i] * (n[i] + 1)), i) for i in range(len(A)) if n[i] < cap[i]]
    heapq.heapify(h)
    while left > 0 and h:
        _, i = heapq.heappop(h)
        n[i] += 1
        left -= 1
        if n[i] < cap[i]:
            heapq.heappush(h, (-A[i] / math.sqrt(n[i] * (n[i] + 1)), i))
    return n, left


# ------------------------------------------------------------------ 3. dividing
def _largest(m):
    cc, k = ndi.label(m)
    if k <= 1:
        return m
    return cc == np.bincount(cc.ravel())[1:].argmax() + 1


def fiedler(m):
    """(field over m, cut thresholds and their lengths) - the field runs smoothly along the shape."""
    px = int(m.sum())
    p = np.pad(m, 1)
    per = int((p[1:] != p[:-1]).sum() + (p[:, 1:] != p[:, :-1]).sum())
    step = max(1, min(int(math.sqrt(px / 900)), int(0.8 * 2 * px / max(per, 1))))
    H, W = m.shape
    h, w = -(-H // step), -(-W // step)
    q = np.zeros((h * step, w * step), np.float32)
    q[:H, :W] = m
    cov = q.reshape(h, step, w, step).mean((1, 3))
    g = _largest(cov >= 0.4)
    if g.sum() < 4:
        g = _largest(cov > 0)
    idx = np.full(g.shape, -1, np.int64)
    k = int(g.sum())
    if k < 4:
        return None
    idx[g] = np.arange(k)
    e = np.concatenate([np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()]),
                        np.stack([idx[:-1].ravel(), idx[1:].ravel()])], 1)
    e = e[:, (e >= 0).all(0)]
    A = coo_matrix((np.ones(e.shape[1]), (e[0], e[1])), (k, k))
    A = (A + A.T).tocsc()
    D = np.asarray(A.sum(0)).ravel()
    Lp = csc_matrix(identity(k) * 1e-6 - A)
    Lp.setdiag(D + 1e-6)
    lu = splu(Lp, permc_spec="MMD_AT_PLUS_A", options=dict(SymmetricMode=True), diag_pivot_thresh=0.0)
    pts = np.argwhere(g).astype(float)
    cp = pts - pts.mean(0)
    ev = np.linalg.eigh(cp.T @ cp)[1][:, -1]
    f = cp @ ev
    for _ in range(12):
        f = f - f.mean()
        f = f / np.linalg.norm(f)
        f = lu.solve(f)
    f = f - f.mean()
    return f, g, e, step


def bisect(m, n):
    """(mask of one side, stones for it, stones for the rest): a short cut that keeps stone sizes even."""
    r = fiedler(m)
    if r is None:
        return None
    f, g, e, step = r
    k = len(f)
    o = np.argsort(f, kind="stable")
    rank = np.empty(k, int)
    rank[o] = np.arange(k)
    a, b = np.minimum(rank[e[0]], rank[e[1]]), np.maximum(rank[e[0]], rank[e[1]])
    cut = np.cumsum(np.bincount(a, minlength=k) - np.bincount(b, minlength=k)).astype(float)    # after position p
    q = (np.arange(k) + 1) / k
    n1 = np.clip(np.round(q * n), 1, n - 1)
    bad = np.maximum(np.abs(q * n / n1 - 1), np.abs((1 - q) * n / (n - n1) - 1))
    side = np.minimum(n1, n - n1) / n
    score = (cut + 1.0) * (1 + 6 * bad ** 2) * (1 + 2.5 * np.maximum(0.0, 0.36 - side))
    score[-1] = 1e18
    score[bad > 0.45] += 1e9
    p = int(score.argmin())
    t = 0.5 * (f[o[p]] + f[o[min(p + 1, k - 1)]])
    H, W = m.shape
    F = np.zeros(g.shape, np.float32)
    F[g] = f
    # values just outside the coarse shape: the nearest inside (so the border is not pulled to zero)
    iy, ix = ndi.distance_transform_edt(~g, return_distances=False, return_indices=True)
    F = F[iy, ix]
    big = np.asarray(Image.fromarray(F).resize((g.shape[1] * step, g.shape[0] * step), Image.BICUBIC))[:H, :W]
    s = m & (big <= t)
    s = _largest(s)
    rest = m & ~s
    rest = _largest(rest)
    s = m & ~rest                                  # fragments go to the other side; still connected
    if not s.any() or not rest.any():
        return None
    return s, int(n1[p]), int(n - n1[p])


def cut_few(m, n, limit=10 ** 9):
    """Exactly n stones by repeated balanced cuts. limit: largest side in px."""
    out = np.where(m, 0, -1).astype(np.int32)
    nxt = 1
    todo = [(0, n)]
    while todo:
        i, k = todo.pop()
        if k <= 1:
            continue
        sl = ndi.find_objects(out + 1, max_label=i + 1)[i]
        if sl is None:
            continue
        s = out[sl] == i
        r = bisect(s, k)
        if r is None:
            continue
        a, n1, n2 = r
        out[sl][s & ~a] = nxt
        todo += [(i, n1), (nxt, n2)]
        nxt += 1
    return out, nxt


def trim(out, q, n, res):
    """Texture cells -> exactly n: the smallest joins the neighbour it fits best, the largest is halved."""
    # loose fragments first
    S = kesim._split_components(out) if (out >= 0).all() else None
    if S is None:
        S = np.full(out.shape, -1, np.int32)
        nx = 0
        for i, sl in enumerate(ndi.find_objects(out + 1)):
            if sl is None:
                continue
            s = out[sl] == i
            cc, k = ndi.label(s)
            S[sl][s] = cc[s] - 1 + nx
            nx += k
        q = nx
    else:
        q = int(S.max()) + 1
    P = np.pad(S, 1, constant_values=-1) + 1
    lo, hi, cnt = _adjacency(P, q + 1)
    area = np.bincount(P.ravel(), minlength=q + 1).astype(float)
    nb = [dict() for _ in range(q + 1)]
    per = np.zeros(q + 1)
    for a, b, c in zip(lo.tolist(), hi.tolist(), cnt.tolist()):
        per[a] += c
        per[b] += c
        if a:
            nb[a][b] = c
            nb[b][a] = c
    parent = np.arange(q + 1)
    alive = np.ones(q + 1, bool)
    alive[0] = False
    k = q
    A0 = area[1:].sum() / max(n, 1)
    h = [(area[i], i) for i in range(1, q + 1)]
    heapq.heapify(h)
    while h:
        a, i = heapq.heappop(h)
        if not alive[i] or a != area[i] or not nb[i]:
            continue
        if k <= n and a > 0.18 * A0:
            break
        # best partner: long common border, compact union, not too large
        def score(j):
            c = nb[i][j]
            u = area[i] + area[j]
            pu = per[i] + per[j] - 2 * c
            return pu / math.sqrt(u) * (1 + 0.35 * max(0.0, u / A0 - 1.0))
        j = min(nb[i], key=score)
        c = nb[i].pop(j)
        nb[j].pop(i)
        per[j] = per[i] + per[j] - 2 * c
        area[j] += area[i]
        for t, v in nb[i].items():
            nb[t].pop(i)
            nb[t][j] = nb[t].get(j, 0) + v
            nb[j][t] = nb[j].get(t, 0) + v
        alive[i] = False
        parent[i] = j
        k -= 1
        heapq.heappush(h, (area[j], j))
    for _ in range(40):
        p2 = parent[parent]
        if (p2 == parent).all():
            break
        parent = p2
    ids = np.full(q + 1, -1)
    ids[np.flatnonzero(alive)] = np.arange(int(alive.sum()))
    O = ids[parent][P[1:-1, 1:-1]].astype(np.int32)
    O[S < 0] = -1
    k = int(alive.sum())
    if k < n:
        sz = np.bincount(O[O >= 0], minlength=k).astype(float)
        box = list(ndi.find_objects(O + 1))
        tried = 0
        while k < n and tried < 200:
            i = int(sz.argmax())
            tried += 1
            s = O[box[i]] == i
            r = bisect(s, 2)
            if r is None:
                sz[i] = 0
                continue
            O[box[i]][s & ~r[0]] = k
            b2 = box[i]
            box.append(b2)
            na = float(r[0].sum())
            sz = np.append(sz, sz[i] - na)
            sz[i] = na
            k += 1
    return O, k


def rows_n(m, n, res, seed, layout="mozaik"):
    """About n texture stones in the area, then exactly n."""
    A = m.sum() * res * res
    s = math.sqrt(A / n)
    f = max(1, int(min(s / 12, 0.8) / res))
    sm = m[f // 2::f, f // 2::f]
    best = None
    s0 = s * 0.92
    tries = []

    def run(mask, s0, mm):
        r = np.random.default_rng(seed)
        if layout == "kirik":
            o, q = kesim.lay_broken(mask, s0 * 1.05, mm, r, 2)
        elif layout == "blok":
            o, q = kesim.lay_blocks(mask, s0 * 0.77, mm, 0, 0)
        else:
            o, q = kesim.lay_rows(mask, s0, mm, r)
        return o, len(np.unique(o[o >= 0]))
    for it in range(3):
        o, q = run(sm, s0, res * f)
        tries.append((s0, q))
        if best is None or abs(math.log(q / n)) < abs(math.log(best[1] / n)):
            best = (s0, q, o)
        if abs(q / n - 1) < 0.04:
            break
        if len(tries) >= 2 and tries[-1][1] != tries[-2][1] and tries[-1][0] != tries[-2][0]:
            p = math.log(tries[-1][1] / tries[-2][1]) / math.log(tries[-1][0] / tries[-2][0])
            p = min(-0.7, max(-2.5, p))
        else:
            p = -1.7
        s0 = s0 * (n / q) ** (1 / p)
    s0, q, o = best
    if f > 2:                                      # the joints are drawn finer than they were searched
        f2 = max(1, int(min(s / 12, 0.4) / res))
        if f2 < f:
            f = f2
            sm = m[f // 2::f, f // 2::f]
            o, q = run(sm, s0, res * f)
    o, k = trim(o, q, n, res * f)
    if f > 1:
        hi = o < 0
        iy, ix = ndi.distance_transform_edt(hi, return_distances=False, return_indices=True)
        o = o[iy, ix]
        up = np.repeat(np.repeat(o, f, 0), f, 1)[: m.shape[0], : m.shape[1]]
        if up.shape != m.shape:
            up = np.pad(up, ((0, m.shape[0] - up.shape[0]), (0, m.shape[1] - up.shape[1])), mode="edge")
        o = np.where(m, smooth_labels(up, m, f), -1)
    return o, k


def smooth_labels(L, m, f):
    """Stair steps of an enlarged label map: every pixel takes the majority around it."""
    if f < 2:
        return L
    out = L.copy()
    box = ndi.find_objects(L + 1)
    best = np.zeros(L.shape, np.float32)
    s = 0.6 * f
    pad = int(3 * s) + 1
    for i, sl in enumerate(box):
        if sl is None:
            continue
        sl = tuple(slice(max(0, t.start - pad), t.stop + pad) for t in sl)
        g = ndi.gaussian_filter((L[sl] == i).astype(np.float32), s, truncate=2.5)
        k = g > best[sl]
        out[sl][k] = i
        best[sl][k] = g[k]
    return np.where(m, out, -1)


def defragment(S, L):
    """Loose bits of a stone go to the neighbour stone of the same area; the number of stones stays."""
    H, W = S.shape
    for _ in range(3):
        bad = np.zeros(S.shape, bool)
        for i, sl in enumerate(ndi.find_objects(S + 1)):
            if sl is None:
                continue
            s = S[sl] == i
            cc, k = ndi.label(s)
            if k > 1:
                sz = np.bincount(cc.ravel())
                sz[0] = 0
                bad[sl] |= s & (cc != sz.argmax())
        if not bad.any():
            break
        cc, k = ndi.label(bad)
        for i, sl in enumerate(ndi.find_objects(cc), 1):
            sl = tuple(slice(max(0, t.start - 1), t.stop + 1) for t in sl)
            b = cc[sl] == i
            ring = ndi.binary_dilation(b) & ~b
            ar = L[sl][b][0]
            own = S[sl][b][0]
            v = S[sl][ring & (L[sl] == ar) & (S[sl] != own)]
            if len(v):
                S[sl][b] = np.bincount(v).argmax()
    return S


def divide(lab, n, res, N, base=None, layout="mozaik", seed=7):
    L, col = comps(lab, n, base)
    K = len(col)
    A = np.bincount(L[L >= 0], minlength=K) * res ** 2
    box = ndi.find_objects(L + 1)
    need, cap, wid = np.ones(K, int), np.ones(K, int), np.zeros(K)
    for i, sl in enumerate(box):
        m = np.pad(L[sl] == i, 1)
        d = ndi.distance_transform_edt(m[::2, ::2] if m.size > 400_000 else m)
        r2 = res * (2 if m.size > 400_000 else 1)
        wid[i] = 2 * d.max() * r2
        mw = 4 * float(d[d > 0].mean()) * r2            # mean width (a ramp 0..w/2 averages w/4)
        sides = [(t.stop - t.start) * res / PLATE for t in sl]
        need[i] = max(1, math.ceil(sides[0]) * math.ceil(sides[1]))
        holes = ndi.label(~m)[1] > 1
        if not holes and mw < 3.0:                      # a long thin stone breaks
            need[i] = max(need[i], math.ceil(A[i] / mw / max(45.0, 28 * mw)))
        amin = HAND_A if mw >= HAND_W else max(6.0, mw * 6.0)
        cap[i] = max(need[i], int(A[i] / amin))
    nn, left = allot(A, N, cap, need)
    jobs = [(i, box[i], L[box[i]] == i, int(nn[i])) for i in range(K)]

    def work(j):
        i, sl, m, k = j
        if k <= 1:
            return np.where(m, 0, -1).astype(np.int32), 1
        s = math.sqrt(A[i] / k)
        if layout in ("mozaik", "kirik", "blok") and (k > 150 or (k >= ROWS_N and wid[i] > ROWS_W * s)):
            return rows_n(m, k, res, seed + i, layout)
        return cut_few(m, k)
    S = np.full(lab.shape, -1, np.int32)
    nxt = 0
    with ThreadPoolExecutor(6) as ex:
        for (i, sl, m, k), (o, q) in zip(jobs, ex.map(work, jobs)):
            S[sl][m] = o[m] + nxt
            nxt += q
    S = fit(defragment(S, L), res)
    return S, L, col, dict(areas=K, need=int(need.sum()), cap=int(cap.sum()), left=left)


def fit(S, res):
    """Stones too long for the printer's plate are cut across."""
    todo = list(range(int(S.max()) + 1))
    box = ndi.find_objects(S + 1)
    while todo:
        i = todo.pop()
        sl = box[i]
        if sl is None or max(t.stop - t.start for t in sl) * res <= PLATE:
            continue
        s = S[sl] == i
        r = bisect(s, 2)
        if r is None:
            continue
        j = len(box)
        S[sl][s & ~r[0]] = j
        box += ndi.find_objects(np.where(S[sl] == j, 1, 0))
        box[j] = tuple(slice(a.start + b.start, a.start + b.stop) for a, b in zip(sl, box[j]))
        q = ndi.find_objects((S[sl] == i).astype(np.int8))[0]
        box[i] = tuple(slice(a.start + b.start, a.start + b.stop) for a, b in zip(sl, q))
        todo += [i, j]
    return S


# ------------------------------------------------------------------ 4. outlines
def outlines(S, L, col, res, gap=0.4, edge=None):
    """(polygons, their colours, the stone in S each one comes from)"""
    polys, pc, ps = [], [], []
    for i, sl in enumerate(ndi.find_objects(S + 1)):
        if sl is None:
            continue
        m = S[sl] == i
        P = kesim.outline(m, sl[0].start, sl[1].start, res)
        if P is None:
            continue
        T = shapely.simplify(P.buffer(-gap / 2, quad_segs=3), 0.05)
        if T.is_empty:
            T = P.buffer(-gap / 5, quad_segs=3)
        if edge is not None and not edge.covers(T):
            T = T.intersection(edge)             # the exact curve instead of the pixel one
        c = int(col[L[sl][m][0]])
        for j, g in enumerate(sorted(kesim._parts(T), key=lambda g: -g.area)):
            if g.area >= (1.5 if j == 0 else 5.0):            # a neck can break: only a real second part counts
                polys.append(g)
                pc.append(c)
                ps.append(i)
    return polys, np.array(pc, int), np.array(ps, int)


def make(lab, n, plab, res, N, *, base=None, layout="mozaik", tonal=False, gap=0.4, seed=7, report=None, tries=3, edge=None):
    """N stones from the colour map lab (n palette colours, plab their Lab values); N = None: every shape is
    one stone (only what does not fit the printer is cut). -> dict(polys, colour, stone, label (S), lab, areas)"""
    report = report or (lambda f, m: None)
    area = (lab != n).sum() * res ** 2
    first, K, Nx, best, cur = lab, N, N, None, None
    if N is None:
        tries, cur, Nx, N = 1, lab, 0, 0
    over = None                                                # stones more than asked for, to keep features
    for it in range(tries):
        report(it / tries, "Taşlar kesiliyor")
        if cur is None:
            cur, o = reduce(first, n, plab, res, K, base, over)
            if tonal and best is None:
                first = cur = calm(cur, n, res, area / N)
                cur, o = reduce(cur, n, plab, res, K, base, over)
            if over is None:                                   # fixed at the number asked for; no more later
                over = o
        over = over or 0
        T = N + over
        S, L, col, info = divide(cur, n, res, Nx + over, base, layout, seed)
        report((it + 0.7) / tries, "Taşların kenarı çiziliyor")
        polys, pc, ps = outlines(S, L, col, res, gap, edge)
        e = len(polys) - T
        if best is None or abs(e) < abs(len(best["polys"]) - best["T"]):
            best = dict(polys=polys, colour=pc, stone=ps, label=S, lab=cur, areas=info["areas"], T=T)
        if it == tries - 1 or abs(e) <= 0.01 * T or (e < 0 and info["left"] > 0):
            break
        if e > 0 and info["need"] >= Nx + over - e:            # every area is whole already: fewer areas
            K -= e
            cur = None
            if K < 1:
                break
        else:
            Nx = max(1, Nx - e)
    report(1.0, "Taşlar hazır")
    best.pop("T")
    return best


# ------------------------------------------------------------------ what suits this picture
def contour(lab, n):
    """The colour that draws the lines of the picture (leaded-glass look), or None: it lies on nearly every
    border and is nearly one piece, so one printed plate can carry all the stones."""
    sm = lab[::3, ::3]
    a = np.concatenate([sm[:, :-1].ravel(), sm[:-1].ravel()])
    b = np.concatenate([sm[:, 1:].ravel(), sm[1:].ravel()])
    k = (a != b) & (a < n) & (b < n)
    if k.sum() < 200:
        return None
    a, b = a[k], b[k]
    for c in range(n):
        if ((a == c) | (b == c)).mean() < 0.9:
            continue
        m = sm == c
        cc, q = ndi.label(ndi.binary_dilation(m, iterations=1))
        areas = sum(ndi.label(sm == o)[1] for o in range(n) if o != c)
        if areas >= 6 and m.sum() < 0.6 * (sm < n).sum() and np.bincount(cc[m])[1:].max() >= 0.9 * m.sum():
            return c
    return None
