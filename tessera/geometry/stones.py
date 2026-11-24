"""Round 6 (R6-1 .. R6-3): every stone is its own printed piece.

build_stones(m) -> StoneLayout
    stone footprint = the stone's tile polygon (shared-edge coverage of the label raster) shrunk by
    gap_mm / 2, so neighbouring stones stand gap_mm apart. Stones too small to print (< 4 mm wide
    or < 16 mm2 before the gap) join a same-colour neighbour; nothing is recoloured.
stone_arrays(st, gp)  -> (V, T)  bevelled prism, built directly with numpy (fast, watertight);
                                 the top follows the tilt plane or the freud profile
stone_solid(st, ...)  -> Manifold  same shape + carved V-grooves + engraved code underneath

Coordinates: mm, x right, y up, z out of the wall; stones stand on z = 0 (wall/board/base top).
Nothing assumes a tile shape: any polygon from the pipeline works.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import shapely
import shapely.ops
from manifold3d import CrossSection, FillRule, Manifold, Mesh, triangulate
from shapely.geometry import Polygon

from ..model import Mode, Mosaic
from .layout import (GeoParams, _clean, _pairs, _part_metrics, _polys, _tile_polygons, _UF,
                     backing_colour_index, colour_letters, panel_xy)

BEVEL_MM = 0.7               # printable top chamfer (0.6-0.8)
MIN_TOP_MM = 1.2             # lowest stone top
ENGRAVE_MIN_MM = 8.0         # stones at least this wide get their code underneath
TEMPLATE_PAGE = (190.0, 277.0)   # usable A4 area (mm) used to order plates by template page
TEMPLATE_OVERLAP = 10.0


@dataclass
class Stone:
    index: int
    code: str
    colour: int
    panel: int
    number: int
    tiles: np.ndarray
    poly: Polygon                 # printed footprint (mm), after the gap
    plane: tuple                  # (a, b, c) top = a + b*(x-cx) - c*(y-cy)
    ref: tuple                    # (cx, cy) mm
    width: float                  # inscribed diameter of the printed footprint (mm)
    area_mm2: float
    profile: int = 0              # 0 plane, 1 freud ramp, 2 quarter-round
    pdir: tuple = (1.0, 0.0)      # profile direction (unit, mm frame)
    page: int = 0                 # template page (A4 grid, reading order)
    cut: bool = False             # R8: exact outline (any shape), flat top, robust bevel
    side: bool = False            # printed on its side (freud profile, prismatic footprint)

    def top(self, x, y):
        a, b, c = self.plane
        return a + b * (np.asarray(x) - self.ref[0]) - c * (np.asarray(y) - self.ref[1])


@dataclass
class StonePanel:
    index: int
    row: int
    col: int
    region: object
    bounds: tuple
    stones: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"P{self.index + 1}"

    @property
    def size(self):
        x0, y0, x1, y1 = self.bounds
        return x1 - x0, y1 - y0


@dataclass
class StoneLayout:
    mosaic: Mosaic
    params: GeoParams
    stones: list
    panels: list
    letters: list
    grid: tuple
    backing_colour: int
    base_colour: int
    mode: str
    pages: tuple                  # (cols, rows) of the A4 template grid
    grooves: list                 # [(pts mm (N,2), width, depth)]
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def neutral_colour_index(m: Mosaic) -> int:
    """AMS base colour (shows in the gaps): settings.base_colour, else the least saturated palette
    colour (a neutral), ties -> the darker one."""
    s = m.settings
    v = getattr(s, "base_colour", None)
    if isinstance(v, int) and 0 <= v < len(s.palette):
        return v
    best, key = 0, None
    for i, f in enumerate(s.palette):
        r, g, b = f.rgb
        sat = max(r, g, b) - min(r, g, b)
        k = (sat, 0.3 * r + 0.59 * g + 0.11 * b)
        if key is None or k < key:
            best, key = i, k
    return best


def _grooves(m: Mosaic):
    """m.grooves (algorithm team): [{"pts": mm, x right / y DOWN from the top-left, "width", "depth"}].
    Returned in the 3D frame (y up): [(pts (N,2), width, depth)]."""
    out = []
    g = getattr(m, "grooves", None)
    if not g:
        return out
    Hmm = m.label.shape[0] * m.settings.mm_per_px
    for item in g:
        if isinstance(item, dict):
            pts = np.asarray(item.get("pts", []), float)
            w = float(item.get("width", item.get("w", 1.0)))
            d = float(item.get("depth", item.get("d", 0.8)))
        else:
            pts, w, d = np.asarray(item, float), 1.0, 0.8
        if pts.ndim != 2 or len(pts) < 2:
            continue
        out.append((np.column_stack([pts[:, 0], Hmm - pts[:, 1]]), w, d))
    return out


# ----------------------------------------------------------------------------- layout
def build_stones(m: Mosaic, gp: Optional[GeoParams] = None, progress=None) -> StoneLayout:
    gp = gp or GeoParams()
    s = m.settings
    mode = s.mode if s.mode in (Mode.PIECES, Mode.AMS) else Mode.PIECES
    mm = s.mm_per_px
    H, W = m.label.shape
    lab = m.label.astype(np.int64)
    n = max(int(lab.max()) + 1, m.n_tiles)
    tcol = np.zeros(n, np.int64)
    tcol[: m.n_tiles] = m.tile_colour
    gap = float(s.gap_mm)
    min_w = float(getattr(s, "min_stone_mm", 4.0))
    warnings = []

    if m.polys is not None:
        return _build_cut(m, gp, mode, progress)
    # 1. stones too small to print join a same-colour neighbour (never a colour change)
    present = np.bincount(lab.ravel(), minlength=n) > 0
    root = np.arange(n)
    root[~present] = -1
    area, width = _part_metrics(lab, root, mm)
    a, b, cnt = _pairs(lab, n)
    nb: dict[int, dict[int, int]] = {}
    for x, y, c in zip(a.tolist(), b.tolist(), cnt.tolist()):
        if tcol[x] == tcol[y]:
            nb.setdefault(x, {})[y] = c
            nb.setdefault(y, {})[x] = c
    uf = _UF(n)
    # a footprint must keep >= ~min_w after the gap is taken off both sides
    need_w = min_w + gap * 0.5
    need_a = (min_w + gap) ** 2 * 0.8
    small = [t for t in range(n) if present[t] and (area.get(t, 0) < need_a or width.get(t, 0) < need_w)]
    small.sort(key=lambda t: area.get(t, 0))
    acc = {t: area.get(t, 0.0) for t in range(n) if present[t]}
    n_join = n_tiny = 0
    for t in small:
        rt = uf.find(t)
        if rt != t and acc.get(rt, 0) >= need_a * 1.5:
            continue
        cand = {}
        for y, c in nb.get(t, {}).items():
            ry = uf.find(y)
            if ry != rt:
                cand[ry] = cand.get(ry, 0) + c
        if not cand:
            n_tiny += 1
            continue
        best = max(cand.items(), key=lambda kv: kv[1])[0]
        uf.p[rt] = best
        acc[best] = acc.get(best, 0) + acc.get(rt, 0)
        n_join += 1
    root = np.array([uf.find(i) if present[i] else -1 for i in range(n)])
    if progress:
        progress(0.1, "Taşlar çokgene çevriliyor")

    # 2. polygons: shared-edge coverage of the (joined) stones, then shrink by gap / 2
    sl = np.where(root[lab] >= 0, root[lab], 0)
    polys_px = _tile_polygons(sl, n, gp.simplify_mm / mm)
    groups: dict[int, list] = {}
    for t in np.nonzero(root >= 0)[0].tolist():
        groups.setdefault(int(root[t]), []).append(t)
    cen = np.zeros((n, 2))
    cen[: m.n_tiles] = m.tile_centroid
    plane = np.zeros((n, 3))
    plane[: m.n_tiles] = m.tile_plane
    prof = getattr(m, "tile_profile", None)
    pdir = getattr(m, "tile_dir", None)
    nx, ny = panel_xy(W * mm, H * mm, s.panel_max_mm)
    cw, ch = W * mm / nx, H * mm / ny
    pw, ph = TEMPLATE_PAGE[0] - TEMPLATE_OVERLAP, TEMPLATE_PAGE[1] - TEMPLATE_OVERLAP
    pcols = max(1, math.ceil((W * mm - TEMPLATE_OVERLAP) / pw))
    prows = max(1, math.ceil((H * mm - TEMPLATE_OVERLAP) / ph))
    raw = []
    for r, ts in groups.items():
        g = polys_px[r]
        if g is None:
            continue
        g = shapely.transform(g, lambda c: np.column_stack([c[:, 0] * mm, (H - c[:, 1]) * mm]))
        foot = g
        body = g.buffer(-gap / 2, join_style="mitre", mitre_limit=2.0)
        ps = [p for p in _polys(body) if p.area > 0.5]
        if not ps:
            body = g.buffer(-min(gap / 2, 0.2), join_style="mitre", mitre_limit=2.0)
            ps = [p for p in _polys(body) if p.area > 0.5]
            if not ps:
                continue
        p = max(ps, key=lambda q: q.area).simplify(0.05)
        if not p.exterior.is_ccw:
            p = shapely.geometry.polygon.orient(p, 1.0)
        main = max(ts, key=lambda t: m.tile_area_px[t] if t < m.n_tiles else 0)
        cx, cy = cen[main, 0] * mm, (H - cen[main, 1]) * mm
        pr = int(prof[main]) if prof is not None and main < len(prof) else 0
        dv = (1.0, 0.0)
        if pdir is not None and main < len(pdir):
            th = np.asarray(pdir[main], float)
            if th.ndim == 0:            # radians, raster x right / y down, pointing high -> low
                dv = (-math.cos(float(th)), math.sin(float(th)))          # mm frame, low -> high
            else:
                dx, dy = float(th[0]), -float(th[1])
                ln = math.hypot(dx, dy)
                if ln > 1e-9:
                    dv = (-dx / ln, -dy / ln)
        rp = p.representative_point()
        col = min(int(rp.x // cw), nx - 1)
        row = min(int((H * mm - rp.y) // ch), ny - 1)
        pc = min(int(rp.x // pw), pcols - 1)
        pg_r = min(int((H * mm - rp.y) // ph), prows - 1)
        try:
            from shapely import maximum_inscribed_circle
            wd = 2 * maximum_inscribed_circle(p, 0.05).length
        except Exception:
            wd = 2 * p.exterior.distance(shapely.ops.polylabel(p, 0.05))
        raw.append(dict(tiles=np.array(ts), colour=int(tcol[main]), poly=p, foot=foot,
                        plane=tuple(plane[main]), ref=(cx, cy), width=wd, area=float(p.area),
                        profile=pr, pdir=dv, panel=row * nx + col, page=pg_r * pcols + pc))
    if progress:
        progress(0.3, "Taş kodları veriliyor")

    # 3. panels, reading-order numbering, codes
    letters = colour_letters([f.name for f in s.palette])
    used = sorted({r_["panel"] for r_ in raw})
    remap = {p: i for i, p in enumerate(used)}
    panels = []
    stones: list[Stone] = []
    for p in used:
        rr = [x for x in raw if x["panel"] == p]
        rr.sort(key=lambda x: (-int(x["poly"].representative_point().y // 10.0), x["poly"].representative_point().x))
        k = remap[p]
        region = _clean(shapely.coverage_union_all([x["foot"] for x in rr])) if len(rr) > 1 else rr[0]["foot"]
        pn = StonePanel(k, p // nx, p % nx, region, tuple(region.bounds))
        digits = max(3, len(str(len(rr))))
        for j, x in enumerate(rr):
            st = Stone(len(stones), f"{letters[x['colour']]}{k + 1}-{j + 1:0{digits}d}", x["colour"], k, j + 1,
                       x["tiles"], x["poly"], x["plane"], x["ref"], x["width"], x["area"], x["profile"],
                       x["pdir"], x["page"])
            st.side = st.profile > 0 and _prismatic(st)
            stones.append(st)
            pn.stones.append(st.index)
        panels.append(pn)
    narrow = sum(1 for st in stones if st.width < min_w - 0.05)
    if n_join:
        warnings.append(f"{n_join} çok küçük taş aynı renkli komşusuyla tek taş yapıldı")
    if n_tiny:
        warnings.append(f"DİKKAT: {n_tiny} küçük taşın aynı renkte komşusu yok; olduğu gibi basılıyor")
    stats = {"tas": len(stones), "birlesen": n_join, "panel": len(panels), "izgara": [nx, ny],
             "dar_tas": narrow, "sablon_sayfa": [pcols, prows],
             "en_dar_mm": round(min((st.width for st in stones), default=0), 2)}
    return StoneLayout(m, gp, stones, panels, letters, (nx, ny), backing_colour_index(m),
                       neutral_colour_index(m), mode, (pcols, prows), _grooves(m), warnings, stats)


def _build_cut(m: Mosaic, gp, mode, progress=None) -> StoneLayout:
    """R8: one Stone per exact outline. Nothing is merged, moved or redrawn."""
    s = m.settings
    mm = s.mm_per_px
    H, W = m.label.shape
    wmm, hmm = W * mm, H * mm
    nx, ny = panel_xy(wmm, hmm, s.panel_max_mm)
    cw, ch = wmm / nx, hmm / ny
    pw, ph = TEMPLATE_PAGE[0] - TEMPLATE_OVERLAP, TEMPLATE_PAGE[1] - TEMPLATE_OVERLAP
    pcols = max(1, math.ceil((wmm - TEMPLATE_OVERLAP) / pw))
    prows = max(1, math.ceil((hmm - TEMPLATE_OVERLAP) / ph))
    raw = []
    for t, g in enumerate(m.polys):
        p = shapely.transform(g, lambda c: np.column_stack([c[:, 0], hmm - c[:, 1]]))
        p = shapely.geometry.polygon.orient(p, 1.0)
        rp = p.representative_point()
        col, row = min(int(max(rp.x, 0) // cw), nx - 1), min(int(max(hmm - rp.y, 0) // ch), ny - 1)
        pc, pr = min(int(max(rp.x, 0) // pw), pcols - 1), min(int(max(hmm - rp.y, 0) // ph), prows - 1)
        raw.append(dict(t=t, poly=p, rp=rp, panel=row * nx + col, page=pr * pcols + pc,
                        width=2 * shapely.maximum_inscribed_circle(p, 0.05).length))
        if progress and t % 400 == 0:
            progress(0.8 * t / len(m.polys), "Taş kodları veriliyor")
    letters = colour_letters([f.name for f in s.palette])
    used = sorted({x["panel"] for x in raw})
    panels, stones = [], []
    for k, pid in enumerate(used):
        rr = sorted((x for x in raw if x["panel"] == pid), key=lambda x: (-int(x["rp"].y // 10.0), x["rp"].x))
        region = shapely.union_all([x["poly"].buffer(s.gap_mm * 0.75, quad_segs=1) for x in rr]).buffer(-s.gap_mm * 0.5)
        region = Polygon(max(_polys(region), key=lambda q: q.area).exterior)
        pn = StonePanel(k, pid // nx, pid % nx, region, tuple(region.bounds))
        digits = max(3, len(str(len(rr))))
        for j, x in enumerate(rr):
            c = int(m.tile_colour[x["t"]])
            st = Stone(len(stones), f"{letters[c]}{k + 1}-{j + 1:0{digits}d}", c, k, j + 1, np.array([x["t"]]),
                       x["poly"], tuple(m.tile_plane[x["t"]]), (x["rp"].x, x["rp"].y), x["width"],
                       float(x["poly"].area), page=x["page"], cut=True)
            stones.append(st)
            pn.stones.append(st.index)
        panels.append(pn)
    thin = sum(st.width < 2.0 for st in stones)
    warnings = [f"{thin} taş 2 mm'den dar: basılır, cımbızla yerleştirmek gerekir"] if thin else []
    stats = {"tas": len(stones), "birlesen": 0, "panel": len(panels), "izgara": [nx, ny], "dar_tas": thin,
             "sablon_sayfa": [pcols, prows], "en_dar_mm": round(min(st.width for st in stones), 2)}
    return StoneLayout(m, gp, stones, panels, letters, (nx, ny), backing_colour_index(m), neutral_colour_index(m),
                       mode, (pcols, prows), [], warnings, stats)


CUT_BEVEL_MM = 0.4


def _cut_arrays(st: Stone, bevel: float = CUT_BEVEL_MM):
    """Flat-topped stone of any outline with a bevelled rim: outline walls + a sloped rim triangulated
    between the outline and its inset. Watertight for necks, spikes and holes."""
    h = float(st.plane[0])
    p = st.poly
    b = min(bevel, st.width * 0.22)
    rings = [np.asarray(r.coords)[:-1] for r in (p.exterior, *p.interiors)]
    ins = [q for q in _polys(p.buffer(-b, quad_segs=2)) if q.area > 0.3]
    ins = [shapely.geometry.polygon.orient(q, 1.0) for q in ins]
    for use in (ins, []):
        inner = [np.asarray(r.coords)[:-1] for q in use for r in (q.exterior, *q.interiors)]
        P = np.vstack(rings)
        n = len(P)
        Q = np.vstack(inner) if inner else np.zeros((0, 2))
        zr = h - b if inner else h
        V = np.vstack([np.column_stack([P, np.zeros(n)]), np.column_stack([P, np.full(n, zr)]),
                       np.column_stack([Q, np.full(len(Q), h)])])
        try:
            bottom = np.asarray(triangulate(rings))
            T = [bottom[:, ::-1]]
            if inner:
                rim = np.asarray(triangulate(rings + [r[::-1] for r in inner]))
                starts = n + np.cumsum([0] + [len(r) for r in inner[:-1]])
                # triangulate() saw each inset ring reversed: map those ids back
                ids = np.concatenate([np.arange(n)] + [o + np.arange(len(r))[::-1] for r, o in zip(inner, starts)])
                T.append(ids[rim] + n)
                off = 2 * n
                for q in use:
                    qr = [np.asarray(r.coords)[:-1] for r in (q.exterior, *q.interiors)]
                    T.append(np.asarray(triangulate(qr)) + off)
                    off += sum(len(r) for r in qr)
            else:
                T.append(bottom + n)
        except Exception:  # noqa: BLE001 - fall through to the plain prism
            continue
        o = 0
        for r in rings:
            i = o + np.arange(len(r))
            j = o + (np.arange(len(r)) + 1) % len(r)
            T += [np.column_stack([i, j, j + n]), np.column_stack([i, j + n, i + n])]
            o += len(r)
        T = np.vstack(T).astype(np.int64)
        e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
        a, b_, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        vol = float(np.einsum("ij,ij->i", a, np.cross(b_, c)).sum() / 6)
        ok = 0.6 * st.area_mm2 * h < vol <= 1.001 * st.area_mm2 * h
        if ok and (np.unique(e, axis=0, return_counts=True)[1] == 2).all():
            return V, T
    man = CrossSection(rings, FillRule.EvenOdd).extrude(h)
    me = man.to_mesh()
    return np.asarray(me.vert_properties, float)[:, :3], np.asarray(me.tri_verts, np.int64)


def _prismatic(st: Stone) -> bool:
    """A freud stone can lie on its side when its footprint is (nearly) a rectangle aligned with
    the profile direction, so the profile becomes the bed footprint."""
    rr = st.poly.minimum_rotated_rectangle
    return rr.area > 0 and st.poly.area / rr.area > 0.97


# ----------------------------------------------------------------------------- meshes
def _ring_inset(P: np.ndarray, d: float) -> Optional[np.ndarray]:
    """Mitre-offset every vertex of a CCW ring inwards by d (same vertex count)."""
    n = len(P)
    e_prev = P - np.roll(P, 1, 0)
    e_next = np.roll(P, -1, 0) - P
    def nrm(e):
        L = np.linalg.norm(e, axis=1, keepdims=True)
        L[L < 1e-12] = 1
        e = e / L
        return np.column_stack([-e[:, 1], e[:, 0]])          # left normal = inward for CCW
    n1, n2 = nrm(e_prev), nrm(e_next)
    bis = n1 + n2
    cos = np.clip((n1 * n2).sum(1), -0.95, 1.0)
    k = d / np.maximum((1 + cos) / 2, 0.15) ** 0.5
    bl = np.linalg.norm(bis, axis=1, keepdims=True)
    bl[bl < 1e-9] = 1
    Q = P + bis / bl * k[:, None]
    q = Polygon(Q)
    if not q.is_valid or q.area <= 0 or not Polygon(P).buffer(1e-6).contains(q):
        return None
    return Q


def _profile_z(st: Stone, x, y, a_lo, a_hi):
    """Freud profile height across the stone along pdir (t = 0..1)."""
    dx, dy = st.pdir
    proj = np.asarray(x) * dx + np.asarray(y) * dy
    t = np.clip((proj - a_lo) / max(a_hi - a_lo, 1e-6), 0, 1)
    lo = max(MIN_TOP_MM, st.plane[0] * 0.45)
    hi = max(lo + 1.0, st.plane[0] * 1.35)
    if st.profile == 2:
        f = np.sqrt(np.clip(1 - (1 - t) ** 2, 0, 1))
    else:
        f = t
    return lo + (hi - lo) * f


def _profile_block(st: Stone) -> Optional[Manifold]:
    """Freud block: the profile (along pdir, height) extruded across the stone and clipped to the
    footprint. Exact curved face with few triangles; this is also how it prints (on its side)."""
    p = st.poly
    d = np.array(st.pdir)
    e = np.array([-d[1], d[0]])
    P = np.asarray(p.exterior.coords)
    u = P @ d
    w = P @ e
    u0, u1, w0, w1 = u.min(), u.max(), w.min(), w.max()
    K = 14 if st.profile == 2 else 1
    ts = np.linspace(0, 1, K + 1)
    us = u0 + ts * (u1 - u0)
    zs = np.maximum(_profile_z(st, us * d[0], us * d[1], u0, u1), MIN_TOP_MM)
    prof = [(u0, 0.0), (u1, 0.0)] + [(float(a), float(b)) for a, b in zip(us[::-1], zs[::-1])]
    prof = np.array(prof)
    cs = CrossSection([prof], FillRule.EvenOdd)
    if cs.area() <= 0:
        cs = CrossSection([prof[::-1]], FillRule.EvenOdd)
    blk = cs.extrude(w1 - w0 + 0.2).translate((0, 0, w0 - 0.1))     # coords (u, z, w)
    M = np.array([[d[0], 0.0, e[0], 0.0], [d[1], 0.0, e[1], 0.0], [0.0, 1.0, 0.0, 0.0]])
    blk = blk.transform(M)
    if blk.volume() < 0:
        return None
    foot = CrossSection([P[:-1]], FillRule.EvenOdd).extrude(float(zs.max()) + 1.0)
    out = blk ^ foot
    return None if out.is_empty() else out


def stone_arrays(st: Stone, bevel: float = BEVEL_MM):
    """(V, T) of the stone standing on z = 0; watertight; no booleans for plain stones."""
    if st.cut:
        return _cut_arrays(st)
    p = st.poly
    if st.profile > 0 and not p.interiors:
        blk = _profile_block(st)
        if blk is not None and blk.status().name == "NoError":
            mesh = blk.to_mesh()
            return np.asarray(mesh.vert_properties, float)[:, :3], np.asarray(mesh.tri_verts, np.int64)
    if p.interiors or st.profile == 2:
        man = stone_manifold_basic(st, bevel)
        mesh = man.to_mesh()
        return np.asarray(mesh.vert_properties, float)[:, :3], np.asarray(mesh.tri_verts, np.int64)
    P = np.asarray(p.exterior.coords)[:-1]
    if st.profile == 1:
        proj = P @ np.array(st.pdir)
        f = lambda x, y: _profile_z(st, x, y, proj.min(), proj.max())     # noqa: E731
    else:
        f = st.top
    b = min(bevel, st.width * 0.25)
    Q = _ring_inset(P, b) if b > 0.05 else None
    if Q is None and b > 0.05:
        Q = _ring_inset(P, b * 0.5)
        b = b * 0.5
    n = len(P)
    zP = np.maximum(f(P[:, 0], P[:, 1]), MIN_TOP_MM)
    if Q is None:
        V = np.vstack([np.column_stack([P, np.zeros(n)]), np.column_stack([P, zP])])
        tri = triangulate([P])
        T = [tri[:, ::-1], tri + n]
        i = np.arange(n)
        j = (i + 1) % n
        T.append(np.column_stack([i, j, j + n]))
        T.append(np.column_stack([i, j + n, i + n]))
        return V, np.vstack(T).astype(np.int64)
    zQ = np.maximum(f(Q[:, 0], Q[:, 1]), MIN_TOP_MM)
    zM = np.maximum(zP - b, 0.4)
    V = np.vstack([np.column_stack([P, np.zeros(n)]), np.column_stack([P, zM]), np.column_stack([Q, zQ])])
    tri = triangulate([P])
    triq = triangulate([Q])
    i = np.arange(n)
    j = (i + 1) % n
    T = [tri[:, ::-1], triq + 2 * n,
         np.column_stack([i, j, j + n]), np.column_stack([i, j + n, i + n]),
         np.column_stack([i + n, j + n, j + 2 * n]), np.column_stack([i + n, j + 2 * n, i + 2 * n])]
    return V, np.vstack(T).astype(np.int64)


def stone_manifold_basic(st: Stone, bevel: float = BEVEL_MM) -> Manifold:
    """Manifold path (holes, quarter-round profile): extrude + warp the top."""
    cs = CrossSection([np.asarray(st.poly.exterior.coords)[:-1]] +
                      [np.asarray(r.coords)[:-1] for r in st.poly.interiors], FillRule.EvenOdd)
    if st.profile == 2:
        man = cs.extrude(1.0).refine_to_length(1.5)
        P = np.asarray(st.poly.exterior.coords)
        proj = P[:, :2] @ np.array(st.pdir)
        lo, hi = proj.min(), proj.max()

        def w(v):
            v = np.array(v, float, copy=True)
            v[:, 2] = v[:, 2] * _profile_z(st, v[:, 0], v[:, 1], lo, hi)
            return v
        return man.warp_batch(w)
    from .mesh import tile_solid
    sol = tile_solid(st.poly, lambda x, y: np.maximum(st.top(x, y), MIN_TOP_MM), MIN_TOP_MM,
                     min(bevel, st.width * 0.25))
    return sol if sol is not None else cs.extrude(MIN_TOP_MM)


def to_manifold(V, T) -> Manifold:
    return Manifold(Mesh(vert_properties=np.asarray(V, np.float32), tri_verts=np.asarray(T, np.uint32)))


def stone_solid(L: StoneLayout, st: Stone, engrave: bool = True):
    """Exact printable stone: bevelled prism + carved V-grooves + mirrored code underneath.
    Returns (Manifold, engraved_text or None)."""
    from .text import place_text
    V, T = stone_arrays(st)
    man = to_manifold(V, T)
    if man.status().name != "NoError" or man.is_empty():
        man = stone_manifold_basic(st)
    cuts = []
    for pts, w, d in L.grooves:
        line = shapely.LineString(pts)
        if not line.intersects(st.poly.buffer(w)):
            continue
        seg = line.intersection(st.poly.buffer(w))
        for part in getattr(seg, "geoms", [seg]):
            c = np.asarray(part.coords) if part.geom_type == "LineString" else None
            if c is None or len(c) < 2:
                continue
            for p0, p1 in zip(c[:-1], c[1:]):
                dv = p1 - p0
                ln = np.hypot(*dv)
                if ln < 1e-6:
                    continue
                nrm = np.array([-dv[1], dv[0]]) / ln * (w / 2)
                pts3 = []
                for q in (p0, p1):
                    zt = float(np.maximum(st.top(q[0], q[1]), MIN_TOP_MM))
                    pts3 += [[*(q + nrm), zt + 0.5], [*(q - nrm), zt + 0.5], [*q, max(0.4, zt - d)]]
                cuts.append(Manifold.hull_points(np.array(pts3)))
    if cuts:
        man = man - Manifold.compose(cuts) if len(cuts) > 1 else man - cuts[0]
    engraved = None
    if engrave:
        rr = st.poly.minimum_rotated_rectangle
        c = np.asarray(rr.exterior.coords)
        side = min(np.hypot(*(c[1] - c[0])), np.hypot(*(c[2] - c[1])))
        if side >= ENGRAVE_MIN_MM:
            from .mesh import _long_axis_angle
            short = st.code.split("-")[-1]
            tg, engraved = place_text(st.poly, [st.code, f"{st.panel + 1}-{short}", short], (2.4, 2.0),
                                      _long_axis_angle(st.poly), mirror=True, margin=0.8)
            if tg is not None:
                cut = CrossSection([np.asarray(q.exterior.coords)[:-1] for q in _polys(tg)] +
                                   [np.asarray(r.coords)[:-1] for q in _polys(tg) for r in q.interiors],
                                   FillRule.EvenOdd).extrude(0.45).translate((0, 0, -0.05))
                man = man - cut
    return man, engraved


def side_transform(st: Stone) -> np.ndarray:
    """4x4 matrix that lays a freud stone on its side: the profile plane (pdir, z) becomes the bed
    plane, the stone's width direction (perpendicular to pdir) points up."""
    dx, dy = st.pdir
    e = np.array([-dy, dx, 0.0])                 # width direction -> new z
    d = np.array([dx, dy, 0.0])                  # profile direction -> new x
    z = np.array([0.0, 0.0, 1.0])                # old height -> new y
    R = np.vstack([d, z, -e])                   # proper rotation (det +1)
    M = np.eye(4)
    M[:3, :3] = R
    return M
