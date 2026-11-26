"""Taking the background away from a drawing or poster: what is left becomes the mosaic's outline.

The background is found on the picture's own colour areas: the colour that runs along the frame's edge, and
the areas of it that the edge reaches. A clear verdict says when this cannot be trusted; then nothing is taken
away and the user picks the areas by clicking.

zemin: {"tiklar": [[x, y, "tek"|"renk"], ...]}   clicks as fractions of the frame (the cut part of the picture);
       "tek" turns one area, "renk" every area of that colour. None = the background stays.
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

SIDE = 1000              # px, longest side the background is looked for at
NOMINAL_MM = 300.0       # the sizes below are meant at this width: the result does not change with the size
MAX_CLICKS = 200
EIGHT = np.ones((3, 3), bool)


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
    return {"tiklar": out}


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
    if info["leak"] > 1.25:
        why.append("Arka plan rengi resmin çizgilerinin arasına da akıyor.")
    if info["border"] < 0.45:
        why.append("Kenarlarda tek bir arka plan rengi yok.")
    if weak > 0.22:
        why.append("Kesilen yerin bir kısmında belirgin bir kenar yok.")
    if busy > 9:
        why.append("Kaldırılan yer düz değil, desenli.")
    if tort > 4.5:
        why.append("Dış çizgi çok girintili çıkıntılı çıkıyor.")
    if sz[0] < 0.5:
        why.append("Resim birçok parçaya bölünüyor.")
    if info["inside"] > 0.3:
        why.append("Arka planın rengi resmin içinde de var.")
    if info["pieces"] > 8:
        why.append("Arka plan birçok ayrı parçadan oluşuyor.")
    if not why:
        return "iyi", []
    sure = len(why) == 1 and share >= 0.04 and info["leak"] < 1.6 and weak < 0.35
    return ("orta" if sure else "zor"), why


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
    """(picture, colour areas, automatic keep, verdict) of one frame."""
    from . import renk, sade
    img = frame_picture(path, json.loads(kadraj))
    own = renk.find(img)
    if sade.kind(img, lambda: own)[0] == "ton":
        lab = sade.label(img, sade.find(img, NOMINAL_MM), NOMINAL_MM)
        if lab.shape != img.shape[:2]:
            lab = np.asarray(Image.fromarray(lab.astype(np.uint8)).resize(img.shape[1::-1], Image.NEAREST))
    else:
        lab = renk.label(img, own)
    lab = lab.astype(np.int32)
    keep, info = flat(img, lab)
    puan, why = judge(img, keep, info)
    if puan == "zor":                            # rather nothing than a wrong cut: the user picks
        keep = np.ones_like(keep)
    for a in (img, lab, keep):
        a.setflags(write=False)
    return img, lab, keep, puan, tuple(why)


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
    tiklar = json.loads(zemin)["tiklar"]
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


def report(keep, puan, why, picked: bool) -> dict:
    """What the window says about the background."""
    share = float(1 - keep.mean())
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
    return {"puan": puan, "soz": soz, "pay": round(share, 3)}


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
