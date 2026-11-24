"""Mosaic raster -> panels and parts (2D only; meshing lives in mesh.py).

Round 2 rules (docs/kararlar.md R2-1): the export NEVER recolours or deletes a feature.

Steps
1. panel grid = the UI formula ceil(w/panel_max) x ceil(h/panel_max) (`panel_xy`);
   tiles reaching further past a cut than the plate allows are split along the straight cut
2. specks (< min_tile_mm2) join a SAME-colour neighbour tile only (no colour change)
3. every tile goes to the panel holding its centroid (cuts follow tile borders); islands are
   handed to the surrounding panel so each panel is one connected piece
4. tile polygons: union of row runs per tile, then a shared-edge (coverage) simplification
5. parts: touching same-colour tiles of one panel merge (settings.merge_same_colour)
6. PIECES mode, parts thinner than 2.4 mm or smaller than 16 mm2:
   - backing colour        -> fused into the backing ("kafes") as a raised rib (kind "kafes")
   - same colour <= 1.5 mm -> joined to that part by a low hidden bridge
   - otherwise             -> loose delicate part (kind "hassas"), widened to >= 1.2 mm
   AMS mode: no special handling at all, everything is fused in its own colour.
7. printable outlines; outlines that fall apart become separate parts with their own codes
8. codes "<colour code><panel>-<nnn>" in reading order

Coordinates: polygons are in mm, 3D frame (x right, y UP, origin bottom-left of the artwork).
"""
from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import shapely
from scipy import ndimage as ndi
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import nearest_points

from ..model import Mosaic, Mode


@dataclass
class GeoParams:
    """Physical constants of the printed system (mm)."""
    rib_w: float = 0.9            # backing rib width (between parts)
    rib_h: float = 0.8            # backing rib height
    clearance: float = 0.15       # part-to-rib clearance per side
    backing_t: float = 1.2        # backing floor level (lattice members / AMS plate thickness)
    lattice: bool = True          # PIECES: open lattice backing instead of a solid plate
    frame_w: float = 4.0          # lattice: perimeter frame width
    flange_w: float = 1.0         # lattice: how far the members reach under each part edge
    brace_pitch: float = 28.0     # lattice: triangular bracing web pitch
    brace_w: float = 1.6
    hollow_min_h: float = 3.5     # tiles taller than this are hollowed (open bottom, 1.2 mm walls)
    hollow_wall: float = 1.2
    hollow_top: float = 1.0
    min_tile_mm2: float = 4.0     # specks below this join a same-colour neighbour tile
    min_part_mm2: float = 16.0    # smaller parts get the R2-1 treatment
    min_part_width: float = 2.4   # thinner parts (inscribed circle) get the R2-1 treatment
    min_loose_width: float = 1.2  # delicate parts are widened to at least this
    tiny_mm2: float = 12.0        # R3-1: no loose part below this (fused / bridged / warned)
    hassas_max_mm2: float = 30.0  # R3-1: loose parts of 12-30 mm2 are "hassas"
    tiny_bridge_dist: float = 3.0 # R3-1 safety net: bridge reach for parts below tiny_mm2
    body_min_mm2: float = 10.0    # R5-1: printed body footprint (after offsets) at least this
    hassas_inset: float = 0.1     # outline inset of delicate parts (thin ribs around them)
    bridge_dist: float = 1.5      # same-colour parts this close get a hidden bridge
    bridge_w: float = 1.2
    bridge_h: float = 0.8
    min_shared_mm: float = 2.6    # same-colour tiles merge only if they share this much border
    simplify_mm: float = 0.6      # coverage simplification tolerance (removes pixel stairs)
    groove_min: float = 0.5       # min groove depth between tiles of one part
    web_min: float = 0.8          # min web thickness of a part
    engrave_depth: float = 0.4
    text_heights: tuple = (3.0, 2.4)
    plate_spacing: float = 3.0
    plate_margin: float = 3.0
    max_overhang_mm: float = 15.0 # tiles reaching further than this past a panel cut are split

    @property
    def part_inset(self) -> float:
        return self.rib_w / 2 + self.clearance


@dataclass
class Part:
    index: int
    pid: str
    panel: int                    # 0-based panel index
    colour: int
    tiles: np.ndarray             # tile ids
    footprint: object             # shapely (Multi)Polygon, un-inset, mm
    area_mm2: float
    number: int = 0               # number within the panel (1-based)
    kind: str = "normal"          # normal | hassas (loose delicate part)
    outline: object = None        # printable z=0 outline (incl. bridges), mm
    bridges: object = None        # low bridge strips (part of outline) or None


@dataclass
class Fused:
    """A thin backing-colour feature printed as part of the backing panel."""
    panel: int
    colour: int
    tiles: np.ndarray
    footprint: object
    shape: object                 # 2D area raised on the backing


@dataclass
class Panel:
    index: int                    # 0-based
    row: int
    col: int
    region: object                # shapely geometry, mm
    bounds: tuple                 # (x0, y0, x1, y1) mm
    parts: list = field(default_factory=list)
    fused: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"P{self.index + 1}"

    @property
    def size(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bounds
        return x1 - x0, y1 - y0


@dataclass
class Layout:
    mosaic: Mosaic
    params: GeoParams
    label: np.ndarray             # tile raster (specks joined to same-colour neighbours)
    tile_colour: np.ndarray       # per tile id (never changed from the mosaic)
    tile_plane: np.ndarray        # (N, 3)
    tile_centroid_mm: np.ndarray  # (N, 2) 3D-frame mm (plane reference point)
    tile_poly: list               # per tile id: shapely Polygon or None
    tile_panel: np.ndarray
    tile_part: np.ndarray         # part index, -2 = fused into the backing, -1 = none
    panels: list
    parts: list
    letters: list                 # colour code per palette index
    grid: tuple                   # (nx, ny)
    backing_colour: int = 0
    mode: str = Mode.PIECES
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- helpers
_TR_UP = str.maketrans("ıiçğöşü", "IİÇĞÖŞÜ")


def colour_letters(names: list[str]) -> list[str]:
    """Colour codes for part ids: the first letter when it is unique in the palette, else the
    first letter + the next letter that makes it unique ("Kı" Kırmızı, "Kr" Krem)."""
    def first(n):
        n = n.strip() or "?"
        return n[0].translate(_TR_UP).upper()
    firsts = [first(n) for n in names]
    out, used = [], set()
    for n, f in zip(names, firsts):
        code = None
        if firsts.count(f) == 1 and f not in used:
            code = f
        else:
            for ch in n.strip()[1:]:
                if ch.isalpha():
                    c = f + ch.lower()
                    if c not in used:
                        code = c
                        break
        if code is None:
            k = 2
            while f"{f}{k}" in used:
                k += 1
            code = f"{f}{k}"
        used.add(code)
        out.append(code)
    return out


FINISH_TR = {"matte": "mat", "basic": "standart", "silk": "ipek", "glossy": "parlak"}


def panel_xy(w_mm: float, h_mm: float, panel_max: float) -> tuple[int, int]:
    """Panel grid shown everywhere (UI, estimate, export): ceil(w/max) x ceil(h/max)."""
    return max(1, math.ceil(w_mm / panel_max - 1e-9)), max(1, math.ceil(h_mm / panel_max - 1e-9))


def backing_colour_index(m: Mosaic) -> int:
    s = m.settings
    v = getattr(s, "backing_colour", None)
    if isinstance(v, int) and 0 <= v < len(s.palette):
        return v
    lum = [0.30 * f.rgb[0] + 0.59 * f.rgb[1] + 0.11 * f.rgb[2] for f in s.palette]
    return int(np.argmin(lum))


def mode_params(gp: GeoParams, mode: str, gap_mm: float) -> GeoParams:
    """AMS: no ribs, parts are only inset by half the hairline gap."""
    if mode == Mode.AMS:
        return replace(gp, rib_w=0.0, clearance=gap_mm / 2)
    return gp


def _polys(g):
    if g is None or g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    out = []
    for x in getattr(g, "geoms", []):
        out.extend(_polys(x))
    return out


def _inset(g, d):
    return g.buffer(-d, join_style="mitre", mitre_limit=3.0) if d else g


def _multi(ps):
    ps = [p for p in ps if p is not None and not p.is_empty]
    if not ps:
        return Polygon()
    return ps[0] if len(ps) == 1 else MultiPolygon(ps)

def ascii_name(s: str) -> str:
    s = s.replace("ı", "i").replace("İ", "I")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")
    return s or "renk"


def _pairs(lab: np.ndarray, n: int):
    """Unique adjacent tile pairs (a < b) and their shared border length in px."""
    a1, b1 = lab[:, :-1].ravel(), lab[:, 1:].ravel()
    a2, b2 = lab[:-1, :].ravel(), lab[1:, :].ravel()
    a = np.concatenate([a1, a2]).astype(np.int64)
    b = np.concatenate([b1, b2]).astype(np.int64)
    m = a != b
    a, b = a[m], b[m]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key, cnt = np.unique(lo * n + hi, return_counts=True)
    return key // n, key % n, cnt


class _UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        p = self.p
        r = x
        while p[r] != r:
            r = p[r]
        while p[x] != r:
            p[x], x = r, p[x]
        return r


def _neighbour_table(a, b, cnt):
    nb: dict[int, dict[int, int]] = {}
    for x, y, c in zip(a.tolist(), b.tolist(), cnt.tolist()):
        nb.setdefault(x, {})[y] = c
        nb.setdefault(y, {})[x] = c
    return nb



def _clean(g):
    if g is None or g.is_empty:
        return g
    if not g.is_valid:
        g = shapely.make_valid(g)
    if g.geom_type == "GeometryCollection":
        polys = [x for x in g.geoms if x.geom_type in ("Polygon", "MultiPolygon")]
        g = shapely.union_all(polys) if polys else Polygon()
    return g


def _connect_panels(lab, tpanel, npanels):
    """Make each panel one connected region: smaller islands go to the panel around them."""
    praster = tpanel[lab]
    for _ in range(3):
        changed = False
        for p in range(npanels):
            mask = praster == p
            if not mask.any():
                continue
            cc, k = ndi.label(mask)
            if k <= 1:
                continue
            sizes = np.bincount(cc.ravel())
            sizes[0] = 0
            keep = int(sizes.argmax())
            for comp in range(1, k + 1):
                if comp == keep:
                    continue
                cm = cc == comp
                ring = ndi.binary_dilation(cm) & ~cm
                nbp = praster[ring]
                nbp = nbp[(nbp != p) & (nbp >= 0)]
                if len(nbp) == 0:
                    continue
                newp = int(np.bincount(nbp).argmax())
                tiles = np.unique(lab[cm])
                tpanel[tiles] = newp
                praster[np.isin(lab, tiles)] = newp
                changed = True
        if not changed:
            break
    return tpanel


def _tile_polygons(lab, n, tol_px):
    """Polygon per tile id (pixel units, raster frame) forming a valid simplified coverage."""
    H, W = lab.shape
    d = np.ones((H, W + 1), bool)
    d[:, 1:W] = lab[:, 1:] != lab[:, :-1]
    idx = np.nonzero(d.ravel())[0]
    r, c = idx // (W + 1), idx % (W + 1)
    nxt = np.roll(c, -1)
    keep = c < W
    sr, sc, ec = r[keep], c[keep], nxt[keep]
    lr = lab[sr, sc]
    # merge identical runs in consecutive rows (vertical strips) to cut box count
    o = np.lexsort((sr, ec, sc, lr))
    sr, sc, ec, lr = sr[o], sc[o], ec[o], lr[o]
    newgrp = np.ones(len(sr), bool)
    newgrp[1:] = ~((lr[1:] == lr[:-1]) & (sc[1:] == sc[:-1]) & (ec[1:] == ec[:-1]) & (sr[1:] == sr[:-1] + 1))
    g = np.cumsum(newgrp) - 1
    starts = np.nonzero(newgrp)[0]
    r0 = sr[starts]
    r1 = np.maximum.reduceat(sr, starts) + 1
    boxes = shapely.box(sc[starts], r0, ec[starts], r1)
    bl = lr[starts]
    o = np.argsort(bl, kind="stable")
    bl, boxes = bl[o], boxes[o]
    bnd = np.searchsorted(bl, np.arange(n + 1))
    out = [None] * n
    ids, geoms = [], []
    for t in range(n):
        s0, s1 = bnd[t], bnd[t + 1]
        if s1 <= s0:
            continue
        gg = boxes[s0] if s1 - s0 == 1 else shapely.union_all(boxes[s0:s1])
        if gg.geom_type != "Polygon":
            gg = max(gg.geoms, key=lambda q: q.area) if hasattr(gg, "geoms") else gg
        ids.append(t)
        geoms.append(gg)
    arr = np.array(geoms, dtype=object)
    if tol_px > 0:
        arr = shapely.transform(shapely.segmentize(arr, 1.0), np.round)
        try:
            arr = shapely.coverage_simplify(arr, tol_px)
        except Exception:
            arr = shapely.simplify(arr, tol_px * 0.5)
    for t, gg in zip(ids, arr):
        if gg is not None and not gg.is_empty:
            out[t] = gg if gg.is_valid else shapely.make_valid(gg)
    return out




# ----------------------------------------------------------------------------- main
def build_layout(m: Mosaic, gp: Optional[GeoParams] = None, progress=None, count_only: bool = False):
    s = m.settings
    mode = s.mode if s.mode in (Mode.PIECES, Mode.AMS) else Mode.PIECES
    gp = mode_params(gp or GeoParams(), mode, s.gap_mm)
    mm = s.mm_per_px
    H, W = m.label.shape
    lab = m.label.astype(np.int64).copy()
    n0 = max(int(lab.max()) + 1, m.n_tiles)
    tile_colour = np.zeros(n0, np.int64)
    tile_colour[: m.n_tiles] = m.tile_colour
    tile_plane = np.zeros((n0, 3))
    tile_plane[: m.n_tiles] = m.tile_plane
    cen_px = np.zeros((n0, 2))
    cen_px[: m.n_tiles] = m.tile_centroid
    warnings: list[str] = []
    stats: dict = {}
    w_mm, h_mm = W * mm, H * mm
    bk = backing_colour_index(m)

    # --- 1. panel grid + split tiles that overhang a cut too far --------------------------
    nx, ny = panel_xy(w_mm, h_mm, s.panel_max_mm)
    cw, chh = W / nx, H / ny
    usable_px = (s.plate_mm - 2 * gp.plate_margin) / mm
    allow_x = max(0.0, min(gp.max_overhang_mm / mm, (usable_px - cw) / 2 - 1))
    allow_y = max(0.0, min(gp.max_overhang_mm / mm, (usable_px - chh) / 2 - 1))
    objs = ndi.find_objects(lab + 1)
    nxt = n0
    new_c, new_p, new_cp = [], [], []
    n_split = 0
    for t, sl in enumerate(objs):
        if sl is None:
            continue
        ys, xs = sl
        ccol = min(int(cen_px[t, 0] // cw), nx - 1)
        crow = min(int(cen_px[t, 1] // chh), ny - 1)
        if (xs.start < ccol * cw - allow_x or xs.stop > (ccol + 1) * cw + allow_x or
                ys.start < crow * chh - allow_y or ys.stop > (crow + 1) * chh + allow_y):
            sub = lab[sl] == t
            yy, xx = np.nonzero(sub)
            cell = (np.minimum((yy + ys.start) // chh, ny - 1).astype(int) * nx
                    + np.minimum((xx + xs.start) // cw, nx - 1).astype(int))
            cells, counts = np.unique(cell, return_counts=True)
            if len(cells) < 2:
                continue
            n_split += 1
            keep = cells[np.argmax(counts)]
            sv = lab[sl]
            for cval in cells:
                if cval == keep:
                    continue
                sel = cell == cval
                sv[yy[sel], xx[sel]] = nxt
                new_c.append(tile_colour[t]); new_p.append(tile_plane[t]); new_cp.append(cen_px[t])
                nxt += 1
    if new_c:
        tile_colour = np.concatenate([tile_colour, np.array(new_c, np.int64)])
        tile_plane = np.concatenate([tile_plane, np.array(new_p)])
        cen_px = np.concatenate([cen_px, np.array(new_cp)])
        lab = _split_components(lab, tile_colour, tile_plane, cen_px)
        tile_colour, tile_plane, cen_px = lab[1], lab[2], lab[3]
        lab = lab[0]
    if n_split:
        warnings.append(f"{n_split} büyük taş panel sınırında düz kesildi")
    n = len(tile_colour)

    # --- 2. specks join a same-colour neighbour (never a colour change) --------------------
    lab = _absorb_specks(lab, tile_colour, n, gp.min_tile_mm2 / mm ** 2)

    # --- 3. panels -------------------------------------------------------------------------
    flat = lab.ravel()
    cnt = np.maximum(np.bincount(flat, minlength=n), 1)
    yy, xx = np.divmod(np.arange(flat.size), W)
    cx = np.bincount(flat, xx, n) / cnt
    cy = np.bincount(flat, yy, n) / cnt
    present = np.bincount(flat, minlength=n) > 0
    col = np.minimum((cx // cw).astype(int), nx - 1)
    row = np.minimum((cy // chh).astype(int), ny - 1)
    tpanel = np.where(present, row * nx + col, -1)
    tpanel = _connect_panels(lab, tpanel, nx * ny)
    if progress:
        progress(0.05, "Taşlar çokgene çevriliyor")

    # --- 4. tile polygons ------------------------------------------------------------------
    def to_mm(g):
        return shapely.transform(g, lambda c: np.column_stack([c[:, 0] * mm, (H - c[:, 1]) * mm]))
    tile_poly = [None] * n
    if not count_only:
        polys_px = _tile_polygons(lab, n, gp.simplify_mm / mm)
        ids = [i for i, g in enumerate(polys_px) if g is not None]
        conv = to_mm(np.array([polys_px[i] for i in ids], dtype=object))
        if hasattr(shapely, "orient_polygons"):
            conv = shapely.orient_polygons(conv)
        for i, g in zip(ids, conv):
            tile_poly[i] = g
    cen_mm = np.column_stack([cen_px[:, 0] * mm, (H - cen_px[:, 1]) * mm])

    # --- 5. parts: merge touching same-colour tiles ----------------------------------------
    a, b, cntp = _pairs(lab, n)

    def merge_parts():
        uf = _UF(n)
        if s.merge_same_colour:
            sel = (tpanel[a] == tpanel[b]) & (tile_colour[a] == tile_colour[b]) & (cntp * mm >= gp.min_shared_mm)
            for x, y in zip(a[sel].tolist(), b[sel].tolist()):
                rx, ry = uf.find(x), uf.find(y)
                if rx != ry:
                    uf.p[ry] = rx
        r_ = np.array([uf.find(i) for i in range(n)])
        r_[~present] = -1
        return r_
    root = merge_parts()
    # a panel cut must never leave a sliver part (< tiny_mm2) of a colour region that continues
    # in the next panel: such slivers move to the panel holding the rest of the region (R3-1)
    if mode == Mode.PIECES:
        garea = np.bincount(np.where(root[lab] >= 0, root[lab], 0).ravel(), minlength=n) * mm * mm
        cross = (tpanel[a] != tpanel[b]) & (tile_colour[a] == tile_colour[b])
        moved = 0
        votes: dict[int, dict[int, int]] = {}
        for x, y, c in zip(a[cross].tolist(), b[cross].tolist(), cntp[cross].tolist()):
            for u, v in ((x, y), (y, x)):
                ru = int(root[u])
                if ru >= 0 and garea[ru] < gp.tiny_mm2 and garea[int(root[v])] > garea[ru]:
                    d = votes.setdefault(ru, {})
                    d[int(tpanel[v])] = d.get(int(tpanel[v]), 0) + c
        for ru, d in votes.items():
            newp = max(d.items(), key=lambda kv: kv[1])[0]
            tiles = np.nonzero(root == ru)[0]
            tpanel[tiles] = newp
            moved += 1
        if moved:
            root = merge_parts()
    if mode == Mode.PIECES and not count_only:
        root, _ = _split_disconnected(root, tile_poly, gp)
    if progress:
        progress(0.10, "Parçalar oluşturuluyor")

    groups: dict[int, list[int]] = {}
    for t in np.nonzero(root >= 0)[0].tolist():
        groups.setdefault(int(root[t]), []).append(t)
    fps = {}
    if count_only:
        # one (unsimplified) polygon per part straight from the raster: much cheaper
        gr = np.where(root[lab] >= 0, root[lab], 0)
        gpoly = _tile_polygons(gr, n, 0.0)
        for r in groups:
            if gpoly[r] is not None:
                fps[r] = _clean(to_mm(gpoly[r]))
    else:
        for r, ts in groups.items():
            geoms = [tile_poly[t] for t in ts if tile_poly[t] is not None]
            if geoms:
                fps[r] = _clean(shapely.coverage_union_all(geoms) if len(geoms) > 1 else geoms[0])
    groups = {r: ts for r, ts in groups.items() if r in fps and not fps[r].is_empty}
    gcol = {r: int(tile_colour[r]) for r in groups}
    gpan = {r: int(tpanel[r]) for r in groups}

    # --- 6. R2-1 / R3-1 treatment of small / thin parts (PIECES only) ----------------------
    #   backing colour, < 30 mm2 or thinner than 2.4 mm  -> fused into the backing
    #   < 12 mm2   -> bridge to a same-colour part within 3 mm, else kept + loud warning
    #   12-30 mm2  -> bridge to a same-colour part within 1.5 mm, else loose "hassas" part
    kind = {r: "normal" for r in groups}
    reach: dict[int, float] = {}
    n_kafes = n_bridge = 0
    if mode == Mode.PIECES:
        area, width = _part_metrics(lab, root, mm)
        small = {r for r in groups if area.get(r, 0) < gp.hassas_max_mm2}
        for r in groups:
            if gcol[r] == bk and (r in small or width.get(r, 0) < gp.min_part_width):
                kind[r] = "kafes"
        keys = list(groups)
        tree = shapely.STRtree([fps[r] for r in keys])
        # obstacles a bridge must not cross: printable cores of other parts + fused features
        core_cache: dict[int, object] = {}

        def core(jj):
            if jj not in core_cache:
                rr = keys[jj]
                core_cache[jj] = _inset(fps[rr], gp.part_inset) if kind[rr] != "kafes" else fps[rr]
            return core_cache[jj]
        uf2 = _UF(n)
        for r in sorted(small, key=lambda q: area.get(q, 0)):
            if kind[r] == "kafes":
                continue
            dist = gp.tiny_bridge_dist if area.get(r, 0) < gp.tiny_mm2 else gp.bridge_dist
            best, best_key = None, None
            for j in tree.query(fps[r], predicate="dwithin", distance=dist).tolist():
                q = keys[j]
                if q == r or gcol[q] != gcol[r] or gpan[q] != gpan[r] or kind[q] == "kafes":
                    continue
                if uf2.find(q) == uf2.find(r):
                    continue
                d = fps[r].distance(fps[q])
                if d > 0.05:
                    p1, p2 = nearest_points(fps[r], fps[q])
                    strip = shapely.LineString([p1, p2]).buffer(gp.bridge_w / 2 + gp.clearance)
                    hit = False
                    for jj in tree.query(strip, predicate="intersects").tolist():
                        if keys[jj] not in (r, q) and core(jj).intersection(strip).area > 0.05:
                            hit = True
                            break
                    if hit:
                        continue
                k = (q in small, d, -area.get(q, 0))
                if best_key is None or k < best_key:
                    best, best_key = q, k
            if best is not None:
                ra, rb = uf2.find(r), uf2.find(best)
                uf2.p[ra] = rb
                n_bridge += 1
                reach[rb] = max(reach.get(rb, 0.0), reach.get(ra, 0.0), dist)
        # regroup
        merged: dict[int, list[int]] = {}
        for r in groups:
            merged.setdefault(uf2.find(r), []).append(r)
        new_groups, new_kind, new_fps, new_reach = {}, {}, {}, {}
        for top, members in merged.items():
            ts = [t for r in members for t in groups[r]]
            rid = top if top in members else members[0]
            new_groups[rid] = ts
            if any(kind[r] == "kafes" for r in members):
                new_kind[rid] = "kafes"
            elif any(r not in small for r in members) or len(members) > 1:
                new_kind[rid] = "normal"
            else:
                new_kind[rid] = "hassas"
            new_fps[rid] = fps[members[0]] if len(members) == 1 else \
                _clean(shapely.union_all([fps[r] for r in members]))
            new_reach[rid] = max([reach.get(r, 0.0) for r in members] + [reach.get(top, 0.0), gp.bridge_dist])
            gcol[rid], gpan[rid] = gcol[members[0]], gpan[members[0]]
        groups, kind, fps, reach = new_groups, new_kind, new_fps, new_reach
        n_kafes = sum(1 for k in kind.values() if k == "kafes")

    if count_only:
        # fast part count for the stats bar: the groups after the R2-1/R3-1 rules are the parts
        # (outline splitting / tiny backing-colour pieces change this by ~1 %)
        loose = [r for r in groups if kind[r] != "kafes"]
        out = {"parca": len(loose), "hassas": sum(1 for r in loose if kind[r] == "hassas"),
               "panel": len({gpan[r] for r in groups}), "kafese_kaynak": n_kafes, "kopru": n_bridge}
        if count_only == "groups":
            # light layout for the fast 3D scene (tessera.export.scene)
            out.update({"groups": groups, "kind": kind, "panel_of": gpan, "colour_of": gcol, "fps": fps,
                        "grid": (nx, ny), "backing_colour": bk, "params": gp, "mode": mode,
                        "letters": colour_letters([f.name for f in s.palette])})
        return out

    # --- 7. outlines -----------------------------------------------------------------------
    outlines: dict[int, list] = {}      # group -> list of (outline, bridges, kind)
    for r in groups:
        if kind[r] != "normal":
            continue
        outlines[r] = _normal_outlines(fps[r], gp, mode, bridge=(mode == Mode.PIECES),
                                       bridge_dist=reach.get(r, gp.bridge_dist))
    # delicate parts: clipped against their neighbours; when that leaves less than 1.2 mm,
    # the neighbours are shaved instead (their colour never changes)
    if mode == Mode.PIECES:
        for p in set(gpan.values()):
            for r in sorted((r for r in groups if gpan[r] == p and kind[r] == "hassas"),
                            key=lambda q: -fps[q].area):
                near = [(r2, j) for r2, lst in outlines.items() if gpan[r2] == p
                        for j, (g, _, _) in enumerate(lst) if g.distance(fps[r]) < 1.2]
                occ = [outlines[r2][j][0] for r2, j in near]
                occ_u = shapely.union_all(occ).buffer(gp.clearance) if occ else Polygon()
                wide = _hassas_outline(fps[r], gp, mm, None)
                o = _hassas_outline(fps[r], gp, mm, occ_u)
                if _too_thin(o, gp.min_loose_width):
                    o = wide
                    cut = wide.buffer(gp.clearance, join_style="mitre", mitre_limit=3.0)
                    for r2, j in near:
                        g, br, kd = outlines[r2][j]
                        if g.intersects(cut):
                            g2 = _open(_clean(g.difference(cut)))
                            if not g2.is_empty:
                                outlines[r2][j] = (g2, br, kd)
                outlines[r] = [(q, None, "hassas") for q in _polys(o) if q.area >= 0.3]
    # R5-1: every printed body (after joint and clearance offsets) is >= 1.2 mm wide and
    # >= 10 mm2 in footprint: grow it where needed and shave the neighbours (two passes)
    need_w = gp.min_loose_width + 0.05
    need_a = gp.body_min_mm2 + 0.3
    for _pass in range(2 if mode == Mode.PIECES else 0):
        for r in list(outlines):
            for j, (o, br, kd) in enumerate(outlines[r]):
                if br is not None or (not _too_thin(o, need_w) and o.area >= need_a):
                    continue
                wide = _widen(o, need_w, need_a)
                cut = wide.buffer(gp.clearance, join_style="mitre", mitre_limit=3.0)
                for r2, lst in outlines.items():
                    if gpan[r2] != gpan[r]:
                        continue
                    for j2, (g2, br2, kd2) in enumerate(lst):
                        if (r2, j2) == (r, j) or not g2.intersects(cut):
                            continue
                        g3 = _open(_clean(g2.difference(cut)))
                        if not g3.is_empty:
                            lst[j2] = (g3, br2, kd2)
                outlines[r][j] = (wide, None, kd)
    # --- 8. Part / Fused objects + codes ---------------------------------------------------
    letters = colour_letters([f.name for f in s.palette])
    panels = [Panel(p, *divmod(p, nx), None, (0, 0, 0, 0)) for p in range(nx * ny)]
    tile_part = np.full(n, -1, np.int64)
    raw = []
    single = set()
    for r, lst in outlines.items():
        for o, br, kd in lst:
            raw.append((gpan[r], gcol[r], np.array(groups[r]), fps[r], o, br, kd))
            if len(lst) == 1:
                single.add(id(o))
    fused_raw = [(gpan[r], gcol[r], np.array(groups[r]), fps[r]) for r in groups if kind[r] == "kafes"]
    lost = [r for r in groups if kind[r] != "kafes" and not outlines.get(r)]
    if lost:
        # nothing printable survived the inset: keep it as a delicate part anyway (never delete)
        for r in lost:
            o = _hassas_outline(fps[r], gp, mm, Polygon())
            if not o.is_empty:
                raw.append((gpan[r], gcol[r], np.array(groups[r]), fps[r], o, None, "hassas"))
    parts: list[Part] = []
    for p in range(nx * ny):
        pp = [x for x in raw if x[0] == p]
        ff = [x for x in fused_raw if x[0] == p]
        geoms = [x[3] for x in pp] + [x[3] for x in ff]
        if not geoms:
            continue
        region = _clean(shapely.union_all(geoms))
        panels[p].region = region
        panels[p].bounds = tuple(region.bounds)

        # R3-1 safety net: a backing-colour piece below tiny_mm2 is printed with the backing
        keep = []
        for x in pp:
            _, colr, ts_arr, fp, o, br, kd = x
            if mode == Mode.PIECES and colr == bk and br is None:
                ins = gp.part_inset if kd == "normal" else gp.hassas_inset
                piece_fp = o.buffer(ins, join_style="mitre", mitre_limit=3.0).intersection(fp)
                if piece_fp.area < gp.tiny_mm2:
                    ff.append((p, colr, ts_arr, _clean(piece_fp)))
                    n_kafes += 1
                    continue
            keep.append(x)
        pp = keep

        def key(x):
            pt = x[4].representative_point()
            return (-int(pt.y // 15.0), pt.x)
        pp.sort(key=key)
        for k, (_, colr, ts_arr, fp, o, br, kd) in enumerate(pp):
            idx = len(parts)
            # area_mm2 = the colour area this printed body stands for: the whole footprint for a
            # one-body part, the body's share of the footprint when a region prints as several
            if id(o) in single:
                est = float(fp.area)
            else:
                ins = gp.part_inset if kd == "normal" else gp.hassas_inset
                core = o.difference(br) if br is not None else o
                est = float(core.buffer(ins + 0.05, join_style="mitre", mitre_limit=3.0).intersection(fp).area)
            if mode == Mode.PIECES and kd == "normal" and br is None and est < gp.hassas_max_mm2:
                kd = "hassas"
            parts.append(Part(idx, "", p, colr, ts_arr, fp, est, k + 1, kd, o, br))
            panels[p].parts.append(idx)
            free = ts_arr[tile_part[ts_arr] < 0]
            tile_part[free] = idx
        for (_, colr, ts_arr, fp) in ff:
            occ = [parts[i].outline for i in panels[p].parts if parts[i].outline.distance(fp) < 1.0]
            shape = fp.buffer(gp.rib_w / 2, join_style="mitre", mitre_limit=3.0)
            if occ:
                shape = shape.difference(shapely.union_all(occ).buffer(gp.clearance))
            shape = shape.intersection(region.buffer(0)) if not region.is_empty else shape
            panels[p].fused.append(Fused(p, colr, ts_arr, fp, _clean(shape)))
            tile_part[ts_arr] = -2
    panels = [pn for pn in panels if pn.region is not None]
    remap = {pn.index: i for i, pn in enumerate(panels)}
    for i, pn in enumerate(panels):
        pn.index = i
        for f in pn.fused:
            f.panel = i
    for pt in parts:
        pt.panel = remap[pt.panel]
        pt.pid = f"{letters[pt.colour]}{pt.panel + 1}-{pt.number:03d}"
    tpanel = np.array([remap.get(int(v), -1) if v >= 0 else -1 for v in tpanel])

    n_hassas = sum(1 for p in parts if p.kind == "hassas")
    tiny = [p.pid for p in parts if p.area_mm2 < gp.tiny_mm2] if mode == Mode.PIECES else []
    if n_bridge:
        warnings.append(f"{n_bridge} küçük parça aynı renkli komşusuna gizli köprüyle bağlandı")
    if n_kafes:
        warnings.append(f"{n_kafes} ince {s.palette[bk].name} çizgi kafesle birlikte basılır (kabarık çizgi)")
    if n_hassas:
        warnings.append(f"{n_hassas} hassas (küçük) parça var: kodu tabla haritasında ve kafesteki cebinde")
    if tiny:
        warnings.append(f"DİKKAT: {len(tiny)} parça {gp.tiny_mm2:.0f} mm²'den küçük, basılması ve "
                        "yerleştirilmesi zor: " + ", ".join(tiny[:8]) + (" …" if len(tiny) > 8 else ""))
    stats.update({"karo": int(present.sum()), "parca": len(parts), "panel": len(panels),
                  "izgara": [nx, ny], "hassas": n_hassas, "cok_kucuk": len(tiny),
                  "kafese_kaynak": n_kafes, "kopru": n_bridge})
    return Layout(m, gp, lab, tile_colour, tile_plane, cen_mm, tile_poly, tpanel, tile_part,
                  panels, parts, letters, (nx, ny), bk, mode, warnings, stats)


# ----------------------------------------------------------------------------- outlines
def _normal_outlines(fp, gp: GeoParams, mode: str, bridge: bool, bridge_dist: float = None):
    """Printable outline(s) of a part. Pieces close together are joined by low bridges; pieces
    that stay apart become separate parts. Returns [(outline, bridges, kind)]."""
    ps = [q for q in _polys(_open(_inset(fp, gp.part_inset))) if q.area >= 0.3]
    if mode == Mode.PIECES and len(ps) > 1:
        # the region is ONE colour area whose printable outline breaks at narrow necks: keep it
        # one part by letting the outline follow the footprint itself inside the necks
        main = _multi(ps)
        thin = _clean(fp.buffer(-0.05, join_style="mitre", mitre_limit=3.0))
        neck = thin.difference(main.buffer(gp.part_inset - 0.08, join_style="mitre", mitre_limit=3.0))
        neck = neck.buffer(gp.part_inset, join_style="mitre", mitre_limit=3.0)   # reach back to main
        joined = _clean(shapely.union_all([main, neck.intersection(thin)]))
        jp = [q for q in _polys(joined) if q.area >= 0.3]
        if len(jp) < len(ps):
            ps = jp
    if mode == Mode.AMS:
        if not ps:
            ps = [q for q in _polys(_inset(fp, 0.02)) if q.area >= 0.05]
        return [(_multi(ps), None, "normal")] if ps else []
    if not ps:
        return []
    if len(ps) == 1:
        return [(ps[0], None, "normal")]
    # join pieces with bridges (nearest first) when they are close
    ps.sort(key=lambda q: -q.area)
    comps = [[ps[0]]]
    bridges_all = [[]]
    rest = ps[1:]
    reach = (bridge_dist or gp.bridge_dist) + 2 * gp.part_inset + 0.3
    while rest:
        best = None
        for i, q in enumerate(rest):
            for c, comp in enumerate(comps):
                d = min(q.distance(x) for x in comp)
                if best is None or d < best[0]:
                    best = (d, i, c)
        d, i, c = best
        q = rest.pop(i)
        if bridge and d <= reach:
            target = min(comps[c], key=lambda x: q.distance(x))
            p1, p2 = nearest_points(q, target)
            line = shapely.LineString([p1, p2])
            if line.length < 1e-6:
                strip = p1.buffer(gp.bridge_w / 2)
            else:
                # extend into both pieces so the bridge overlaps them
                ext = shapely.affinity.scale(line, 1 + 1.6 / line.length, 1 + 1.6 / line.length)
                strip = ext.buffer(gp.bridge_w / 2, cap_style="flat")
            strip = strip.intersection(fp.buffer(gp.bridge_w))
            comps[c].append(q)
            bridges_all[c].append(strip)
        else:
            comps.append([q])
            bridges_all.append([])
    out = []
    for comp, brs in zip(comps, bridges_all):
        o = _multi(comp)
        if brs:
            br = _clean(shapely.union_all(brs))
            o = _clean(shapely.union_all([o, br]))
            out.append((o, br, "normal"))
        else:
            out.append((o, None, "normal"))
    return out


def _hassas_outline(fp, gp: GeoParams, mm: float, occupied):
    """Delicate part: small inset, widened to min_loose_width, clipped against neighbours."""
    o = _inset(fp, gp.hassas_inset)
    ps = [q for q in _polys(o) if q.area > 0.01]
    o = _multi(ps) if ps else fp
    need = gp.min_loose_width
    thin = o.buffer(-need / 2 + 1e-3).is_empty
    if thin:
        # widen until a 1.2 mm circle fits (never more than 0.5 mm past the footprint)
        for r in (0.15, 0.3, 0.45, 0.6):
            w = o.buffer(r)
            if not w.buffer(-need / 2 + 1e-3).is_empty:
                o = w
                break
        else:
            o = o.buffer(0.6)
    if occupied is not None and not occupied.is_empty:
        o2 = o.difference(occupied)
        if not o2.is_empty:
            o = o2
    o = _clean(o)
    oo = _open(o)
    return oo if not oo.is_empty else o


def _widen(o, need: float, need_area: float = 0.0):
    for r in (0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0):
        w = o.buffer(r)
        if not _too_thin(w, need) and w.area >= need_area:
            return _clean(w)
    return _clean(o.buffer(1.0))


def _too_thin(g, width: float) -> bool:
    return g is None or g.is_empty or g.buffer(-width / 2 + 0.02).is_empty


def _open(g, r: float = 0.08):
    """Morphological opening: removes zero-width necks and spikes (they split meshes)."""
    if g is None or g.is_empty:
        return g
    return _clean(g.buffer(-r, join_style="mitre", mitre_limit=3.0).buffer(r, join_style="mitre", mitre_limit=3.0))


# ----------------------------------------------------------------------------- metrics
def _part_metrics(lab, root, mm):
    """Area (mm2) and inscribed width (mm) per part root."""
    valid = root >= 0
    pr = np.where(valid[lab], root[lab], -1)
    edge = np.zeros(lab.shape, bool)
    d = pr[:, 1:] != pr[:, :-1]
    edge[:, 1:] |= d
    edge[:, :-1] |= d
    d = pr[1:, :] != pr[:-1, :]
    edge[1:, :] |= d
    edge[:-1, :] |= d
    dist = ndi.distance_transform_edt(~edge)
    H, W = lab.shape
    yy = np.minimum(np.arange(H), np.arange(H)[::-1])[:, None] + 1
    xx = np.minimum(np.arange(W), np.arange(W)[::-1])[None, :] + 1
    dist = np.minimum(dist, np.minimum(yy, xx))
    ids = np.unique(pr[pr >= 0])
    maxd = np.asarray(ndi.maximum(dist, pr, ids))
    cnt = np.bincount(pr[pr >= 0].ravel(), minlength=len(root))
    area = {int(r): float(cnt[r]) * mm * mm for r in ids.tolist()}
    width = {int(r): float(w) * 2 * mm for r, w in zip(ids.tolist(), maxd.tolist())}
    return area, width


def _absorb_specks(lab, tc, n, min_px):
    """Tiny tiles join the same-colour neighbour sharing the longest border (no colour change)."""
    area = np.bincount(lab.ravel(), minlength=n)
    small = np.nonzero((area > 0) & (area < min_px))[0]
    if len(small) == 0:
        return lab
    a, b, cnt = _pairs(lab, n)
    nb = _neighbour_table(a, b, cnt)
    uf = _UF(n)
    area = area.astype(np.int64).copy()
    for t in small[np.argsort(area[small], kind="stable")].tolist():
        rt = uf.find(t)
        if area[rt] >= min_px:
            continue
        agg: dict[int, int] = {}
        for y, c in nb.get(t, {}).items():
            ry = uf.find(y)
            if ry != rt and tc[ry] == tc[rt]:
                agg[ry] = agg.get(ry, 0) + c
        if not agg:
            continue
        best = max(agg.items(), key=lambda kv: kv[1])[0]
        uf.p[rt] = best
        area[best] += area[rt]
    root = np.array([uf.find(i) for i in range(n)])
    return root[lab]


def _split_components(lab, tc, tp, cp):
    """After straight cuts a tile may fall apart: give each 4-connected piece its own id."""
    out = lab.copy()
    nxt = int(lab.max()) + 1
    tc, tp, cp = list(tc), list(tp), list(cp)
    for t, sl in enumerate(ndi.find_objects(lab + 1)):
        if sl is None:
            continue
        sub = lab[sl] == t
        cc, k = ndi.label(sub)
        if k <= 1:
            continue
        o = out[sl]
        for j in range(2, k + 1):
            o[cc == j] = nxt
            tc.append(tc[t]); tp.append(tp[t]); cp.append(cp[t])
            nxt += 1
    return out, np.array(tc, np.int64), np.array(tp), np.array(cp)


def _split_disconnected(root, tile_poly, gp):
    """Split same-colour parts whose inset outline breaks into far-apart pieces (tiles go to the
    piece they overlap most). Pieces closer than the bridge distance stay together."""
    groups: dict[int, list[int]] = {}
    for t in np.nonzero(root >= 0)[0].tolist():
        groups.setdefault(int(root[t]), []).append(t)
    root = root.copy()
    n_split = 0
    reach = gp.bridge_dist + 2 * gp.part_inset + 0.3
    for r, ts in groups.items():
        if len(ts) < 2:
            continue
        geoms = [tile_poly[t] for t in ts if tile_poly[t] is not None]
        if len(geoms) < 2:
            continue
        fp = shapely.coverage_union_all(geoms)
        out = _inset(fp, gp.part_inset)
        pieces = [q for q in _polys(out) if q.area >= 1.0]
        if len(pieces) < 2:
            continue
        # cluster pieces that are within bridge reach
        uf = _UF(len(pieces))
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                if pieces[i].distance(pieces[j]) <= reach:
                    uf.p[uf.find(j)] = uf.find(i)
        clusters: dict[int, list] = {}
        for i in range(len(pieces)):
            clusters.setdefault(uf.find(i), []).append(i)
        if len(clusters) < 2:
            continue
        n_split += 1
        cl = list(clusters.values())
        cl_geo = [_multi([pieces[i] for i in c]) for c in cl]
        assign: dict[int, list[int]] = {}
        for t in ts:
            tp = tile_poly[t]
            if tp is None:
                continue
            ov = [tp.intersection(g).area for g in cl_geo]
            k = int(np.argmax(ov)) if max(ov) > 0 else int(np.argmin([tp.distance(g) for g in cl_geo]))
            assign.setdefault(k, []).append(t)
        for k, tl in assign.items():
            nr = r if r in tl else tl[0]
            root[np.array(tl)] = nr
    return root, n_split


def panel_grid(w_mm: float, h_mm: float, panel_max: float, mode: str = Mode.PIECES,
               gp: Optional[GeoParams] = None, plate_mm: float = 256.0, extra=(0, 0)) -> tuple[int, int]:
    """Backward-compatible alias of panel_xy (same grid in both modes since round 2)."""
    nx, ny = panel_xy(w_mm, h_mm, panel_max)
    return nx + extra[0], ny + extra[1]
