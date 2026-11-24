"""Detail-preserving palette quantisation.

The old quantiser blurred the image (sigma 1 px) and ran a 3x3 majority vote,
which erased every feature thinner than ~3 px: eyes, nostrils, whiskers and
outline strokes. This module instead:

1. **Unmixes** each pixel as a blend of the best *pair* of palette colours
   (anti-aliased edges are blends of the two colours either side, so a grey
   pixel between black and cream never becomes red). This gives a soft
   per-colour coverage map. When a higher-resolution source is available the
   coverage is computed there and area-averaged down (supersampled voting).
2. **Rescues ridges**: a pixel whose coverage of colour c is a clear local
   maximum across a line (in any of 4 directions) gets colour c even when the
   coverage is below 50 %. This recovers 1 px anti-aliased lines.
3. **Removes specks**: connected components smaller than ``min_area`` px take
   the colour of their nearest surviving neighbour.
4. **Enforces a minimum feature width**: parts of a colour region that do not
   survive a k x k opening (k = min feature width in px) are grown by one pixel
   into neighbouring regions that are wide enough to give the pixel away, so a
   whisker becomes a printable 3 px (1.2 mm) line instead of vanishing.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

PURE_DIST = 38.0  # RGB distance under which a pixel is treated as an unmixed palette colour


def _open_sq(m: np.ndarray, k: int) -> np.ndarray:
    """Binary opening with a k x k square (separable min/max filters; much faster than
    ndi.binary_opening with a square structure). Outside the image counts as 'inside'."""
    return ndi.grey_opening(m.astype(np.uint8), size=(k, k), mode="nearest").astype(bool)


def _dil3(m: np.ndarray) -> np.ndarray:
    return ndi.maximum_filter(m.astype(np.uint8), size=3).astype(bool)



def coverage(img: np.ndarray, pal: np.ndarray, short_blend: float = 0.02, pure_dist: float = PURE_DIST) -> np.ndarray:
    """Soft per-colour coverage (H, W, K) float32 in [0, 1] via best-pair unmixing.

    ``short_blend`` > 0 prefers the shortest blend when several pairs explain a pixel equally
    well (collinear palettes such as black/grey/white: flat light grey = grey+white). With 0,
    the widest blend wins (negative: always; an anti-aliased 1 px black line on white = black+white)."""
    h, w = img.shape[:2]
    k = len(pal)
    p = img.reshape(-1, 3).astype(np.float32)
    pal = pal.astype(np.float32)
    # nearest single colour (cheap) for pixels that are already pure
    d2 = ((p[:, None, :] - pal[None]) ** 2).sum(-1)
    near = d2.argmin(1)
    dmin = np.sqrt(d2[np.arange(len(p)), near])
    cov = np.zeros((len(p), k), np.float32)
    pure = dmin < pure_dist
    cov[pure, near[pure]] = 1.0
    idx = np.nonzero(~pure)[0]
    if len(idx):
        q = p[idx]
        best_r = np.full(len(idx), np.inf, np.float32)
        best_i = np.zeros(len(idx), np.int32)
        best_j = np.zeros(len(idx), np.int32)
        best_a = np.zeros(len(idx), np.float32)
        for i in range(k):
            for j in range(i + 1, k):
                v = pal[i] - pal[j]
                vv = float(v @ v) or 1.0
                a = np.clip(((q - pal[j]) @ v) / vv, 0.0, 1.0)
                # residual + a small preference for the shortest blend (collinear palettes such
                # as black/grey/white: a light grey is grey+white, not black+white)
                r = ((q - pal[j] - a[:, None] * v) ** 2).sum(1) + short_blend * vv
                upd = r < best_r
                best_r[upd], best_i[upd], best_j[upd], best_a[upd] = r[upd], i, j, a[upd]
        cov[idx, best_i] = best_a
        cov[idx, best_j] = 1.0 - best_a
    return cov.reshape(h, w, k)


def downsample_coverage(cov: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Area-average each coverage channel to ``size_wh`` (supersampled voting)."""
    out = [np.asarray(Image.fromarray(cov[:, :, c]).resize(size_wh, Image.BOX)) for c in range(cov.shape[2])]
    return np.clip(np.stack(out, -1), 0, 1).astype(np.float32)


def ridge_rescue(cov: np.ndarray, label: np.ndarray, min_cov: float = 0.28, contrast: float = 0.2) -> np.ndarray:
    """Give a pixel colour c when its coverage of c is a 1-D local maximum across a thin line."""
    out = label.copy()
    h, w, k = cov.shape
    pad = np.pad(cov, ((1, 1), (1, 1), (0, 0)), mode="edge")
    centre = pad[1:-1, 1:-1]
    score = np.zeros((h, w, k), np.float32)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = pad[1 - dy:1 - dy + h, 1 - dx:1 - dx + w]
        b = pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        ok = (centre >= a) & (centre >= b)
        s = np.where(ok, centre - 0.5 * (a + b), 0.0)
        score = np.maximum(score, s)
    cand = (score >= contrast) & (centre >= min_cov)
    # strongest rescued colour wins; only touch pixels whose current colour is not itself a ridge
    sc = np.where(cand, score, 0.0)
    best = sc.argmax(-1)
    has = sc.max(-1) > 0
    cur_score = np.take_along_axis(sc, label[..., None], -1)[..., 0]
    upd = has & (best != label) & (cur_score <= 0)
    out[upd] = best[upd]
    return out


def line_rescue(img: np.ndarray, label: np.ndarray, pal: np.ndarray, contrast: float = 22.0,
                min_alpha: float = 0.28) -> np.ndarray:
    """Pixels that are a luminance ridge (darker or lighter than both neighbours along some
    direction by ``contrast``) are explained as a blend of the local background (mean of those
    neighbours) and one palette colour; when that colour covers at least ``min_alpha`` of the
    pixel and fits well, the pixel takes it. Recovers thin lines of any palette colour."""
    a = img.astype(np.float32)
    h, w = label.shape
    L = a @ np.array([0.299, 0.587, 0.114], np.float32)
    pa = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge")
    pL = np.pad(L, 1, mode="edge")
    best_s = np.zeros((h, w), np.float32)
    bg = np.zeros_like(a)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        La = pL[1 - dy:1 - dy + h, 1 - dx:1 - dx + w]
        Lb = pL[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        s = np.maximum(np.minimum(La - L, Lb - L), np.minimum(L - La, L - Lb))
        upd = s > best_s
        best_s[upd] = s[upd]
        mb = 0.5 * (pa[1 - dy:1 - dy + h, 1 - dx:1 - dx + w] + pa[1 + dy:1 + dy + h, 1 + dx:1 + dx + w])
        bg[upd] = mb[upd]
    cand = best_s >= contrast
    if not cand.any():
        return label
    p, b = a[cand], bg[cand]
    best_r = np.full(len(p), np.inf, np.float32)
    best_c = np.full(len(p), -1, np.int64)
    for c in range(len(pal)):
        v = pal[c] - b
        vv = np.maximum((v * v).sum(1), 1.0)
        al = ((p - b) * v).sum(1) / vv
        r = ((p - b - np.clip(al, 0, 1)[:, None] * v) ** 2).sum(1)
        ok = (al >= min_alpha) & (vv > (2 * contrast) ** 2) & (r < best_r)
        best_r[ok], best_c[ok] = r[ok], c
    good = (best_c >= 0) & (best_r < 3 * 28.0 ** 2)
    out = label.copy()
    idx = np.nonzero(cand)
    out[idx[0][good], idx[1][good]] = best_c[good]
    return out


def halo_cleanup(label: np.ndarray, img: np.ndarray, pal: np.ndarray, width_px: int = 3) -> np.ndarray:
    """Anti-alias halos. A thin (< ``width_px``) band of a colour M that lies where regions of two
    other colours A and B meet is an edge artefact when its pixels are explained better as an
    A/B blend than as M. Example: mustard next to black, the blended edge pixels are nearest to
    red; they are not red. Such pixels go to A or B, whichever is closer to the pixel."""
    k = len(pal)
    if k < 3:
        return label
    out = label.copy()
    st = np.ones((width_px, width_px), bool)
    near = [ndi.maximum_filter((label == c).astype(np.uint8), size=5).astype(bool) for c in range(k)]
    a = img.astype(np.float32)
    for m_ in range(k):
        mm = label == m_
        thin = mm & ~_open_sq(mm, st.shape[0])
        if not thin.any():
            continue
        ys, xs = np.nonzero(thin)
        px = a[ys, xs]
        d_m = ((px - pal[m_]) ** 2).sum(1)
        best = np.full(len(ys), np.inf, np.float32)
        best_c = np.full(len(ys), -1, np.int64)
        for a_ in range(k):
            for b_ in range(a_ + 1, k):
                if m_ in (a_, b_):
                    continue
                both = near[a_][ys, xs] & near[b_][ys, xs]
                if not both.any():
                    continue
                v = pal[b_] - pal[a_]
                vv = max(float(v @ v), 1.0)
                t = np.clip(((px - pal[a_]) @ v) / vv, 0, 1)
                r = ((px - pal[a_] - t[:, None] * v) ** 2).sum(1)
                ok = both & (r < best) & (r < 0.6 * d_m + 1.0)
                best[ok] = r[ok]
                best_c[ok] = np.where(t[ok] < 0.5, a_, b_)
        sel = best_c >= 0
        out[ys[sel], xs[sel]] = best_c[sel]
    return out


def absorb_islands(label: np.ndarray, k: int, min_px: int, *, keep_colour: Optional[int] = None,
                   keep_reach_px: int = 4, passes: int = 3, tone_px: int = 0,
                   lum: Optional[np.ndarray] = None, tone_tol: float = 0.35,
                   diagonal: bool = False) -> np.ndarray:
    """R3-1: every 4-connected colour island smaller than ``min_px`` takes the colour that
    dominates its surroundings (the 1-px ring around it), so no part below the minimum exists and
    the preview shows exactly what will be printed. Exception: islands of ``keep_colour`` (the
    backing colour) lying within ``keep_reach_px`` of a large region of that colour stay; the 3D
    export fuses them to the backing as ribs.

    ``tone_px`` / ``lum``: islands up to ``tone_px`` whose colour is close in lightness to the
    colour around them (a grey crumb in another grey, less than ``tone_tol`` of the palette's
    lightness range apart) are tone noise, not features, and are absorbed as well. Contrasting
    small features (an eye, a highlight) stay; the 3D export makes them "hassas" parts."""
    out = label.copy()
    lim = max(min_px, tone_px)
    rng_l = float(lum.max() - lum.min()) if lum is not None and len(lum) > 1 else 1.0
    for _ in range(passes):
        isl = np.zeros(out.shape, np.int64)
        nxt = 0
        isz, icol = [0], [0]
        st = np.ones((3, 3), bool) if diagonal else None
        for c in range(k):
            cc, n = ndi.label(out == c, st)
            if n == 0:
                continue
            sz = np.bincount(cc.ravel(), minlength=n + 1)
            sz[0] = 0
            small = (sz < lim) & (sz > 0)
            if c == keep_colour:
                big = (sz >= min_px)[cc] & (cc > 0)
                near_big = ndi.maximum_filter(big.astype(np.uint8), size=2 * keep_reach_px + 1).astype(bool)
                hit = np.zeros(n + 1, bool)
                hit[np.unique(cc[near_big & (cc > 0) & small[cc]])] = True
                small &= ~hit
            ids = np.zeros(n + 1, np.int64)
            ids[small] = np.arange(1, small.sum() + 1) + nxt
            nxt += int(small.sum())
            isz.extend(sz[small].tolist())
            icol.extend([c] * int(small.sum()))
            m = small[cc] & (cc > 0)
            isl[m] = ids[cc[m]]
        if nxt == 0:
            break
        ring = ndi.grey_dilation(isl, size=(3, 3))
        ring_ok = (isl == 0) & (ring > 0)
        votes = np.bincount(ring[ring_ok] * k + out[ring_ok], minlength=(nxt + 1) * k).reshape(nxt + 1, k)
        has = votes.sum(1) > 0
        new_col = votes.argmax(1)
        isz, icol = np.array(isz), np.array(icol)
        go = has & (isz < min_px)
        if tone_px > min_px and lum is not None:
            close = np.abs(lum[icol] - lum[new_col]) < tone_tol * rng_l
            go |= has & (isz < tone_px) & close
        if not go[1:].any():
            break
        m = isl > 0
        upd = m & go[isl]
        out[upd] = new_col[isl[upd]]
    return out


def limit_delicate(label: np.ndarray, k: int, lo_px: int, hi_px: int, backing: Optional[int],
                   max_share: float = 0.03, core_px: int = 0, min_core_px: int = 0) -> np.ndarray:
    """R3-2: islands of lo_px..hi_px (12-30 mm2) that are not in the backing colour become loose
    "hassas" parts. Keep at most ``max_share`` of all colour regions in that class: the smallest
    extra ones take the colour around them (shown in the preview, so preview = print)."""
    sizes, owners = [], []
    total = 0
    labs = []
    drop = np.zeros(label.shape, bool)
    for c in range(k):
        mc = label == c
        cc, n = ndi.label(mc)
        labs.append(cc)
        total += n
        if c == backing or n == 0:
            continue
        eff = np.bincount(cc.ravel(), minlength=n + 1)[1:]
        # printable core: what is left after the part outline is inset (~core_px all round). A
        # thin crescent of 20 mm2 can print as a 3 mm2 grain; those go too.
        core = ndi.grey_erosion(mc.astype(np.uint8), size=(2 * core_px + 1,) * 2, mode="nearest").astype(bool)             if core_px > 0 else mc
        kept = np.bincount(cc[core], minlength=n + 1)[1:]
        tiny = np.nonzero((eff < lo_px) | ((eff < hi_px) & (kept < min_core_px)))[0]
        if tiny.size:
            drop |= np.isin(cc, tiny + 1)
        sel = np.nonzero((eff >= lo_px) & (eff < hi_px) & (kept >= min_core_px))[0]
        sizes.extend(kept[sel].tolist())
        owners.extend([(c, int(i) + 1) for i in sel])
    allow = int(max_share * total)
    if len(sizes) > allow:
        for j in np.argsort(sizes)[: len(sizes) - allow]:      # drop the smallest ones
            c, i = owners[j]
            drop |= labs[c] == i
    if not drop.any():
        return label
    out = label.copy()
    isl, n = ndi.label(drop)
    ring = ndi.grey_dilation(isl, size=(3, 3))
    ok = (isl == 0) & (ring > 0)
    votes = np.bincount(ring[ok] * k + out[ok], minlength=(n + 1) * k).reshape(n + 1, k)
    new_col = votes.argmax(1)
    has = votes.sum(1) > 0
    m = (isl > 0) & has[isl]
    out[m] = new_col[isl[m]]
    return out


def remove_specks(label: np.ndarray, k: int, min_area: int, within: Optional[np.ndarray] = None,
                  diagonal: bool = True) -> np.ndarray:
    """Components (8-connected) smaller than ``min_area`` px take the colour of the nearest larger
    region. With ``within``, only components lying mostly inside that mask are removed."""
    if min_area <= 1:
        return label
    small = np.zeros(label.shape, bool)
    st = np.ones((3, 3), bool) if diagonal else None
    for c in range(k):
        m = label == c
        if not m.any():
            continue
        cc, n = ndi.label(m, st)
        sz = np.bincount(cc.ravel(), minlength=n + 1)
        sz[0] = 1 << 30
        bad = sz < min_area
        if within is not None:
            inside = np.bincount(cc.ravel(), within.ravel().astype(np.float64), n + 1) / np.maximum(sz, 1)
            bad &= inside > 0.5
        small |= bad[cc]
    if not small.any() or small.all():
        return label
    _, (iy, ix) = ndi.distance_transform_edt(small, return_indices=True)
    out = label.copy()
    out[small] = label[iy[small], ix[small]]
    return out


def thin_masks(label: np.ndarray, k: int, width_px: int) -> np.ndarray:
    """(H, W) bool: pixels that belong to a part of their colour region narrower than ``width_px``."""
    thin = np.zeros(label.shape, bool)
    if width_px <= 1:
        return thin
    st = np.ones((width_px, width_px), bool)
    for c in range(k):
        m = label == c
        if m.any():
            thin |= m & ~_open_sq(m, st.shape[0])
    return thin


def enforce_min_width(label: np.ndarray, pal: np.ndarray, width_px: int, rounds: int = 3) -> np.ndarray:
    """Grow features narrower than ``width_px`` one pixel per side per round into regions that
    can spare it (darkest colour wins conflicts), until they are wide enough."""
    if width_px <= 1:
        return label
    k = len(pal)
    out = label.copy()
    st_thin = np.ones((width_px, width_px), bool)
    st_wide = np.ones((width_px + 2, width_px + 2), bool)
    lum = pal @ np.array([0.30, 0.59, 0.11], np.float32)
    for _ in range(rounds):
        thin_by_c, donor = [], np.zeros(out.shape, bool)
        for c in range(k):
            m = out == c
            if not m.any():
                thin_by_c.append(None)
                continue
            thin_by_c.append(m & ~_open_sq(m, st_thin.shape[0]))
            donor |= _open_sq(m, st_wide.shape[0])
        if not any(t is not None and t.any() for t in thin_by_c):
            break
        cur = out.copy()
        changed = False
        # lighter colours first so that dark line work wins where two thin features compete
        for c in np.argsort(-lum):
            t = thin_by_c[c]
            if t is None or not t.any():
                continue
            grow = _dil3(t) & donor & (cur != c)
            if grow.any():
                out[grow] = c
                changed = True
        if not changed:
            break
    return out


def quantize_detail(img: np.ndarray, pal: np.ndarray, mm_per_px: float, *,
                    source: Optional[np.ndarray] = None, min_feature_mm: float = 1.2,
                    min_speck_mm2: float = 0.5, screen_mask: Optional[np.ndarray] = None,
                    screen_speck_mm2: float = 4.0, min_part_mm2: float = 12.0,
                    tone_island_mm2: float = 30.0, max_delicate_share: float = 0.015,
                    backing: Optional[int] = None) -> np.ndarray:
    """Palette index per pixel at the working resolution of ``img``.

    ``source`` (optional) is the same picture at a higher resolution; coverage is then
    computed there and area-averaged down, which keeps sub-pixel lines."""
    h, w = img.shape[:2]
    pal = np.asarray(pal, np.float32)
    src = source if (source is not None and source.shape[0] > h * 1.2) else img

    def cov_at(short):
        c = coverage(src, pal, short)
        return downsample_coverage(c, (w, h)) if src is not img else c

    cov = cov_at(0.02)
    label = cov.argmax(-1).astype(np.int32)
    label = ridge_rescue(cov, label)
    # a 1 px anti-aliased black line on white reads as "grey" when grey is in the palette:
    # judge such lines from the line-versus-background contrast instead
    label = line_rescue(img, label, pal)
    min_area = max(2, int(round(min_speck_mm2 / mm_per_px ** 2)))
    label = remove_specks(label, len(pal), min_area)
    label = halo_cleanup(label, img if img.shape[:2] == (h, w) else src, pal)
    if screen_mask is not None and screen_mask.any():
        # leftovers of a dot screen (crumbs inside descreened tone areas) are not features
        label = remove_specks(label, len(pal), int(round(screen_speck_mm2 / mm_per_px ** 2)), within=screen_mask)
    width_px = max(2, int(round(min_feature_mm / mm_per_px)))
    part_px = max(min_area, int(round(min_part_mm2 / mm_per_px ** 2)))
    reach = max(1, int(round(1.5 / mm_per_px)))
    if screen_mask is not None and screen_mask.any():
        # descreened tone areas: smooth the borders between neighbouring tones with a wide
        # majority vote, so a tone boundary is one clean line instead of a ragged crumb zone
        win = max(5, int(round(3.0 / mm_per_px)) | 1)
        votes = np.stack([ndi.uniform_filter((label == c).astype(np.float32), win) for c in range(len(pal))], -1)
        maj = votes.argmax(-1).astype(label.dtype)
        # ink (the darkest colour) is left alone: lines through a tone must not be voted away
        ink = int((pal @ np.array([0.30, 0.59, 0.11], np.float32)).argmin())
        # ... but small ink beads (dot leftovers at tone edges, < 30 mm2) may be voted away
        icc, inn = ndi.label(label == ink, np.ones((3, 3), bool))
        isz = np.bincount(icc.ravel(), minlength=inn + 1)
        isz[0] = 1 << 30
        bead = (isz < tone_island_mm2 / mm_per_px ** 2)[icc]
        sm = screen_mask & (votes.max(-1) > 0.5) & ((label != ink) | bead) & (maj != ink)
        label = np.where(sm, maj, label)
        # inside screened / hatched tone: strokes thinner than the minimum feature are texture,
        # not parts -> they melt into the tone around them instead of being widened
        thin = thin_masks(label, len(pal), width_px) & screen_mask
        # only where strokes are dense (hatching); a lone ink line through a tone stays
        dense = ndi.uniform_filter(thin.astype(np.float32), max(9, int(round(10.0 / mm_per_px)) | 1)) > 0.2
        thin &= dense
        if thin.any():
            _, (iy, ix) = ndi.distance_transform_edt(thin, return_indices=True)
            label = label.copy()
            label[thin] = label[iy[thin], ix[thin]]
    # R3-1: islands below the minimum part size go to the dominant colour around them, then
    # thin features are widened, then once more for anything the widening left behind
    lum = pal @ np.array([0.30, 0.59, 0.11], np.float32)
    tone_px = int(round(tone_island_mm2 / mm_per_px ** 2))
    kw = dict(keep_colour=backing, keep_reach_px=reach, tone_px=tone_px, lum=lum)
    # (first pass counts corner-touching pixels as connected, so a 1 px diagonal line is one
    # feature; after widening it is 4-connected and the strict pass sees it whole)
    label = absorb_islands(label, len(pal), part_px, diagonal=True, **kw)
    label = enforce_min_width(label, pal, width_px)
    label = absorb_islands(label, len(pal), part_px, **kw)
    # judged on the area the printed part keeps (see limit_delicate)
    label = limit_delicate(label, len(pal), part_px, int(round(35.0 / mm_per_px ** 2)), backing,
                           max_delicate_share, core_px=max(1, int(round(0.8 / mm_per_px))),
                           min_core_px=int(round(4.0 / mm_per_px ** 2)))
    # pieces must be 4-connected: bits that only touch their colour at a corner are specks too
    label = remove_specks(label, len(pal), min_area, diagonal=False)
    return label
