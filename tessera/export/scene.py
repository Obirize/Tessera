"""3D scene API for the three.js stage (docs/uretim_raporu.md "Tur 5").

scene_glb(m, mode=None, quality="fast"|"full") -> (glb bytes, meta dict)
    Every printed part is its own named node ("part:<code>"), lattice backings are
    "backing:<panel>", hanging cleats "cleat:1|2". Units mm, y up, origin = artwork bottom-left,
    +z towards the viewer (out of the wall). Matte PBR material in the filament colour.
    fast: coarse relief per part straight from the mosaic raster (no booleans), < 3 s.
    full: the exact export meshes (cached: changing light/explode in the UI never rebuilds).
plates_glb(m, mode=None) -> (glb bytes, meta dict)
    256 x 256 beds laid out in a grid, parts placed exactly as in the exported 3MF files.

Nothing here assumes a tile shape: everything comes from the layout (or the raster).
"""
from __future__ import annotations

import copy
import io
import math
import time

import numpy as np
import shapely
from manifold3d import CrossSection, FillRule, Manifold
from scipy import ndimage as ndi

from ..geometry.layout import GeoParams, backing_colour_index, build_layout, panel_xy
from ..model import Mode, Mosaic

SCHEMA = "tessera.scene/1"
EXPLODE_PART_MM = 30.0       # parts rise this far out of the lattice at explode = 1
EXPLODE_PANEL_MM = 12.0      # panels drift apart by this much at explode = 1
_CACHE: dict = {}


# ----------------------------------------------------------------------------- helpers
def _with_mode(m: Mosaic, mode):
    if mode is None or mode == m.settings.mode:
        return m
    m2 = copy.copy(m)
    m2.settings = copy.copy(m.settings)
    m2.settings.mode = mode
    return m2


def _key(m: Mosaic, what: str, quality: str = "") -> tuple:
    s = m.settings
    return (what, quality, id(m.label), m.label.shape, int(m.label[::97, ::89].sum()), s.mode,
            s.merge_same_colour, s.panel_max_mm, getattr(s, "backing_colour", None), s.gap_mm,
            tuple(f.rgb for f in s.palette), getattr(s, "plate_order", "panel"))


def _remember(key, value):
    while len(_CACHE) > 7:                # the oldest goes, not what was made a moment ago for the same piece
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = value
    return value


def _material(rgb, name):
    from trimesh.visual.material import PBRMaterial
    r, g, b = [int(c) for c in rgb]
    return PBRMaterial(name=name, baseColorFactor=[r, g, b, 255], metallicFactor=0.0, roughnessFactor=0.92)


def _tm(v, t, rgb, name):
    import trimesh
    from trimesh.visual import TextureVisuals
    mesh = trimesh.Trimesh(np.asarray(v, np.float64), np.asarray(t, np.int64), process=False)
    mesh.visual = TextureVisuals(material=_material(rgb, name))
    return mesh


def _arrays(man: Manifold):
    mesh = man.to_mesh()
    return (np.asarray(mesh.vert_properties, np.float64)[:, :3], np.asarray(mesh.tri_verts, np.int64))


def _cs(g) -> CrossSection:
    contours = []
    for p in getattr(g, "geoms", [g]):
        if p.geom_type != "Polygon" or p.is_empty or p.area < 1e-3:
            continue
        contours.append(np.asarray(p.exterior.coords)[:-1])
        contours.extend(np.asarray(r.coords)[:-1] for r in p.interiors)
    return CrossSection(contours, FillRule.EvenOdd)


def _bbox(v):
    lo, hi = v.min(0), v.max(0)
    return [round(float(x), 2) for x in (*lo, *hi)]


def _glb(nodes) -> bytes:
    import trimesh
    sc = trimesh.Scene()
    for name, mesh in nodes:
        sc.add_geometry(mesh, node_name=name, geom_name=name)
    return sc.export(file_type="glb")


def _palette_meta(m: Mosaic, letters):
    return [{"index": i, "name": f.name, "hex": f.hex, "rgb": list(f.rgb), "code": letters[i],
             "finish": f.finish} for i, f in enumerate(m.settings.palette)]


def _cleat_nodes(m: Mosaic, gp: GeoParams, letters):
    """Two hanging cleats behind the top of the artwork (they belong on the MDF board)."""
    from ..geometry.mesh import clean_mesh, french_cleat
    s = m.settings
    w_mm, h_mm = m.size_mm
    length = float(min(s.plate_mm - 2 * gp.plate_margin - 10, max(120.0, w_mm * 0.6)))
    v, t, _ = clean_mesh(french_cleat(length))
    bk = backing_colour_index(m)
    out = []
    for j in range(2):
        vv = v.copy()
        # stand the strip up behind the board: thickness along -z, width along y
        vv = np.column_stack([vv[:, 0] + (w_mm - length) / 2, vv[:, 2] * 0 + vv[:, 1] + h_mm - 80 - j * 32,
                              -6.0 - vv[:, 2]])
        name = f"cleat:{j + 1}"
        out.append((name, _tm(vv, t[:, ::-1], s.palette[bk].rgb, name),
                    {"name": name, "type": "cleat", "colour": bk, "hex": s.palette[bk].hex,
                     "label": f"Askı çıtası {j + 1}", "bbox": _bbox(vv),
                     "explode": {"dir": [0, 0, -1], "distance_mm": 20.0}}))
    return out


# ----------------------------------------------------------------------------- fast scene
def _fast_scene(m: Mosaic):
    s = m.settings
    mm = s.mm_per_px
    H, W = m.label.shape
    c = build_layout(m, count_only="groups")
    gp = c["params"]
    groups, kind, pan, col, fps = c["groups"], c["kind"], c["panel_of"], c["colour_of"], c["fps"]
    letters = c["letters"]
    bk = c["backing_colour"]
    nx, ny = c["grid"]
    # relief surface: mosaic heights with the hairline gaps filled
    hf = np.where(m.gap, ndi.maximum_filter(m.height, 3), m.height).astype(np.float32)
    hf = ndi.maximum_filter(hf, 2)
    inset = gp.part_inset

    def height_at(x, y):
        cc = np.clip(x / mm, 0, W - 1)
        rr = np.clip(H - y / mm, 0, H - 1)
        return ndi.map_coordinates(hf, [rr, cc], order=1, mode="nearest")

    # panel order/codes like the export: reading order (15 mm rows) inside each panel
    used = sorted({pan[r] for r in groups})
    remap = {p: i for i, p in enumerate(used)}
    loose = [r for r in groups if kind[r] != "kafes"]
    by_panel: dict[int, list] = {}
    for r in loose:
        by_panel.setdefault(pan[r], []).append(r)
    nodes, metas = [], []
    for p in used:
        lst = by_panel.get(p, [])
        rp = {r: fps[r].representative_point() for r in lst}
        lst.sort(key=lambda r: (-int(rp[r].y // 15.0), rp[r].x))
        for k, r in enumerate(lst):
            fp = fps[r]
            g = fp.buffer(-inset, join_style="mitre", mitre_limit=3.0).simplify(0.5)
            if g.is_empty:
                g = fp.buffer(-0.1).simplify(0.2)
            if g.is_empty:
                continue
            man = _cs(g).extrude(1.0).refine_to_length(2.5)
            v, t = _arrays(man)
            top = v[:, 2] > 0.5
            zt = np.maximum(height_at(v[:, 0], v[:, 1]), gp.web_min + 0.4)
            v[:, 2] = np.where(top, zt, 0.0) + gp.backing_t
            code = f"{letters[col[r]]}{remap[p] + 1}-{k + 1:03d}"
            name = f"part:{code}"
            nodes.append((name, _tm(v, t, s.palette[col[r]].rgb, name)))
            metas.append(_part_meta(name, code, col[r], s, remap[p], None, v, float(fp.area), kind[r]))
    # simple backing plates (the real lattice comes with quality="full")
    for p in used:
        reg = shapely.union_all([fps[r] for r in groups if pan[r] == p]).simplify(0.3)
        v, t = _arrays(_cs(reg).extrude(gp.backing_t))
        name = f"backing:P{remap[p] + 1}"
        nodes.append((name, _tm(v, t, [int(x * 0.85) for x in s.palette[bk].rgb], name)))
        metas.append(_backing_meta(name, f"P{remap[p] + 1}", bk, s, v))
    return nodes, metas, letters, gp


def _part_meta(name, code, colour, s, panel, plate, v, area, kind):
    bb = _bbox(v)
    return {"name": name, "type": "part", "code": code, "colour": int(colour),
            "colour_name": s.palette[colour].name, "hex": s.palette[colour].hex,
            "panel": f"P{panel + 1}", "plate": plate,
            "size_mm": [round(bb[3] - bb[0], 1), round(bb[4] - bb[1], 1), round(bb[5] - bb[2], 1)],
            "area_mm2": round(area, 1), "kind": kind, "bbox": bb,
            "explode": {"dir": [0.0, 0.0, 1.0], "distance_mm": EXPLODE_PART_MM}}


def _backing_meta(name, panel, colour, s, v):
    return {"name": name, "type": "backing", "panel": panel, "colour": int(colour),
            "colour_name": s.palette[colour].name, "hex": s.palette[colour].hex, "bbox": _bbox(v),
            "explode": {"dir": [0.0, 0.0, 0.0], "distance_mm": 0.0}}


# ----------------------------------------------------------------------------- full scene
def _full_layout(m: Mosaic):
    """Layout + solids + backings exactly as the export builds them (cached)."""
    key = _key(m, "full_layout")
    if key in _CACHE:
        return _CACHE[key]
    from ..geometry.mesh import build_backing, build_part, clean_mesh
    L = build_layout(m)
    pieces = L.mode == Mode.PIECES
    solids, meshes = {}, {}
    for pt in L.parts:
        ps = build_part(L, pt, engrave=pieces)
        if ps is None:
            continue
        solids[pt.index] = ps
        v, t, _ = clean_mesh(ps.solid)
        meshes[pt.index] = (v, t)
    backs = {}
    for pn in L.panels:
        v, t, _ = clean_mesh(build_backing(L, pn, engrave=True, label_pockets=pieces))
        backs[pn.index] = (v, t)
    return _remember(key, (L, solids, meshes, backs))


def _plate_plan(m: Mosaic, L, solids):
    """part index -> (plate label, file, plate number, angle, dx, dy) as in the export."""
    from .project import pack_colour
    from ..geometry.layout import ascii_name
    from .bambu3mf import MAX_PLATES
    s = m.settings
    gp = L.params
    order = getattr(s, "plate_order", "panel")
    plan, plates = {}, []
    names = [ascii_name(f.name) for f in s.palette]
    for ci in range(len(s.palette)):
        items, placements, n_pl, _ = pack_colour(solids, ci, s, gp, order)
        if not items:
            continue
        n_files = max(1, math.ceil(n_pl / MAX_PLATES))
        for q in placements:
            fi = q.plate // MAX_PLATES
            fname = f"{ci + 1}_{names[ci]}_tablalar{f'_{fi + 1}' if n_files > 1 else ''}.3mf"
            num = q.plate % MAX_PLATES + 1
            plan[items[q.item].part.index] = (f"{s.palette[ci].name} {num}", fname, num, q.angle, q.dx, q.dy, ci,
                                             q.plate)
        for p in range(n_pl):
            plates.append((ci, p))
    return plan, plates


def _full_scene(m: Mosaic):
    L, solids, meshes, backs = _full_layout(m)
    s = m.settings
    gp = L.params
    pieces = L.mode == Mode.PIECES
    plan = _plate_plan(m, L, solids)[0] if pieces else {}
    nodes, metas = [], []
    for pt in L.parts:
        if pt.index not in meshes:
            continue
        v, t = meshes[pt.index]
        v = v + (0.0, 0.0, gp.backing_t)
        name = f"part:{pt.pid}"
        nodes.append((name, _tm(v, t, s.palette[pt.colour].rgb, name)))
        pl = plan.get(pt.index)
        meta = _part_meta(name, pt.pid, pt.colour, s, pt.panel,
                          (pl[0] if pl else (f"Panel {L.panels[pt.panel].name}" if not pieces else None)),
                          v, pt.area_mm2, pt.kind)
        if pl:
            meta["plate_file"] = pl[1]
            meta["plate_number"] = pl[2]
        meta["engraved"] = solids[pt.index].engraved
        metas.append(meta)
    for pn in L.panels:
        v, t = backs[pn.index]
        name = f"backing:{pn.name}"
        nodes.append((name, _tm(v, t, [int(x * 0.85) for x in s.palette[L.backing_colour].rgb], name)))
        metas.append(_backing_meta(name, pn.name, L.backing_colour, s, v))
    return nodes, metas, L.letters, gp, L


# ----------------------------------------------------------------------------- public API
def scene_glb(m: Mosaic, mode=None, quality: str = "preview", progress=None):
    """GLB of the artwork as printed. Round 6 (separate stones): one node per stone,
    quality "preview" (bevelled stones, seconds) or "full" (exact export solids with grooves).
    Legacy merged parts: one node per part (the old embossed "fast" mode is gone: maps to full)."""
    m = _with_mode(m, mode)
    if not m.settings.merge_same_colour:
        q = "full" if quality == "full" else "preview"
        key = _key(m, "scene_stones", q)
        if key not in _CACHE:
            _remember(key, scene_glb_stones(m, q, progress))
        return _CACHE[key]
    quality = "full"
    key = _key(m, "scene", quality)
    if key in _CACHE:
        return _CACHE[key]
    t0 = time.time()
    s = m.settings
    if quality == "full":
        nodes, metas, letters, gp, L = _full_scene(m)
        panels = [{"name": pn.name, "row": pn.row, "col": pn.col,
                   "bounds": [round(float(x), 2) for x in pn.bounds]} for pn in L.panels]
    else:
        nodes, metas, letters, gp = _fast_scene(m)
        panels = []
        for mt in metas:
            if mt["type"] == "backing":
                b = mt["bbox"]
                panels.append({"name": mt["panel"], "bounds": [b[0], b[1], b[3], b[4]]})
    for name, mesh, meta in _cleat_nodes(m, gp, letters):
        nodes.append((name, mesh))
        metas.append(meta)
    w_mm, h_mm = m.size_mm
    cx, cy = w_mm / 2, h_mm / 2
    for p in panels:   # exploded view: panels drift away from the centre a little
        x0, y0, x1, y1 = p["bounds"]
        d = np.array([(x0 + x1) / 2 - cx, (y0 + y1) / 2 - cy])
        nd = np.linalg.norm(d)
        p["explode"] = {"dir": ([float(d[0] / nd), float(d[1] / nd), 0.0] if nd > 1e-6 else [0.0, 0.0, 0.0]),
                        "distance_mm": EXPLODE_PANEL_MM}
    glb = _glb(nodes)
    meta = {
        "schema": SCHEMA, "kind": "artwork", "quality": quality, "mode": s.mode,
        "units": "mm", "up": "+y", "facing": "+z (out of the wall)", "origin": "artwork bottom-left",
        "size_mm": [round(w_mm, 1), round(h_mm, 1)], "backing_t_mm": gp.backing_t,
        "palette": _palette_meta(m, letters), "panels": panels, "nodes": metas,
        "counts": {"parts": sum(1 for x in metas if x["type"] == "part"),
                   "backings": sum(1 for x in metas if x["type"] == "backing"),
                   "cleats": sum(1 for x in metas if x["type"] == "cleat")},
        "build_s": round(time.time() - t0, 2),
    }
    return _remember(key, (glb, meta))


def plates_glb(m: Mosaic, mode=None, gap_mm: float = 40.0, progress=None):
    """GLB of the build plates exactly as exported: 256 x 256 beds in a grid, each with its
    parts (or its backing panel / AMS panel / cleats). meta["plates"] lists them."""
    from .mass import mass_model
    m = _with_mode(m, mode)
    if not m.settings.merge_same_colour:
        key = _key(m, "plates_stones")
        if key not in _CACHE:
            _remember(key, plates_glb_stones(m, gap_mm, progress))
        return _CACHE[key]
    key = _key(m, "plates")
    if key in _CACHE:
        return _CACHE[key]
    t0 = time.time()
    s = m.settings
    L, solids, meshes, backs = _full_layout(m)
    gp = L.params
    pieces = L.mode == Mode.PIECES
    P = s.plate_mm
    beds = []          # (title, colour, list of (name, v, t, rgb, code)), file
    from ..geometry.layout import ascii_name
    kafes_file = f"0_Kafes_{ascii_name(s.palette[L.backing_colour].name)}.3mf"
    if pieces:
        plan, plates = _plate_plan(m, L, solids)
        content: dict[tuple, list] = {}
        for idx, (label, fname, num, ang, dx, dy, ci, plate) in plan.items():
            v, t = meshes[idx]
            a = math.radians(ang)
            vv = v.copy()
            vv[:, 0] = v[:, 0] * math.cos(a) - v[:, 1] * math.sin(a) + dx
            vv[:, 1] = v[:, 0] * math.sin(a) + v[:, 1] * math.cos(a) + dy
            content.setdefault((ci, plate), []).append(
                (f"part:{L.parts[idx].pid}", vv, t, s.palette[ci].rgb, L.parts[idx].pid, fname, num))
        for ci, p in plates:
            items = content.get((ci, p), [])
            if items:
                beds.append((f"{s.palette[ci].name} {items[0][6]}", ci, items, items[0][5]))
        for pn in L.panels:
            v, t = backs[pn.index]
            x0, y0, x1, y1 = pn.bounds
            off = ((P - (x1 - x0)) / 2 - x0, (P - (y1 - y0)) / 2 - y0, 0.0)
            beds.append((f"Kafes {pn.name}", L.backing_colour,
                         [(f"backing:{pn.name}", v + off, t, s.palette[L.backing_colour].rgb, f"Kafes {pn.name}",
                           kafes_file, 0)], kafes_file))
    else:
        for pn in L.panels:
            x0, y0, x1, y1 = pn.bounds
            off = np.array((P - gp.plate_margin - (x1 - x0) - x0, P - gp.plate_margin - (y1 - y0) - y0, 0.0))
            items = [(f"backing:{pn.name}", backs[pn.index][0] + off, backs[pn.index][1],
                      s.palette[L.backing_colour].rgb, f"Kafes {pn.name}", "AMS_paneller", 0)]
            for i in pn.parts:
                if i in meshes:
                    v, t = meshes[i]
                    items.append((f"part:{L.parts[i].pid}", v + off + (0, 0, gp.backing_t), t,
                                  s.palette[L.parts[i].colour].rgb, L.parts[i].pid, "AMS_paneller", 0))
            beds.append((f"Panel {pn.name}", None, items, "AMS_paneller.3mf"))
    # hanging cleats plate
    from ..geometry.mesh import clean_mesh, french_cleat
    cleat_file = kafes_file if pieces else "0_Aski_citalari.3mf"
    length = float(min(P - 2 * gp.plate_margin - 10, max(120.0, m.size_mm[0] * 0.6)))
    cv, ct, _ = clean_mesh(french_cleat(length))
    beds.append(("Askı çıtaları", L.backing_colour,
                 [(f"cleat:{j + 1}", cv + (8.0, yy, 0), ct, s.palette[L.backing_colour].rgb, f"Askı {j + 1}",
                   cleat_file, 0) for j, yy in enumerate((40.0, 120.0))], cleat_file))
    # layout beds in a grid, estimate hours by printed volume share
    mmod = mass_model(m)
    cols = max(1, math.ceil(math.sqrt(len(beds))))
    stride = P + gap_mm
    vols = [sum(abs(_signed_volume(v, t)) for _, v, t, *_ in items) for _, _, items, _ in beds]
    vtot = sum(vols) or 1.0
    nodes, plates_meta = [], []
    bed_v = np.array([[0, 0, -2], [P, 0, -2], [P, P, -2], [0, P, -2], [0, 0, 0], [P, 0, 0], [P, P, 0], [0, P, 0]],
                     float)
    bed_t = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    for k, ((title, ci, items, fname), vol) in enumerate(zip(beds, vols)):
        ox, oy = (k % cols) * stride, -(k // cols) * stride
        name = f"bed:{k + 1}"
        nodes.append((name, _tm(bed_v + (ox, oy, 0), bed_t, (58, 58, 62), name)))
        codes = []
        for nm, v, t, rgb, code, *_ in items:
            nn = f"{nm}@bed{k + 1}"
            nodes.append((nn, _tm(v + (ox, oy, 0), t, rgb, nn)))
            codes.append(code)
        share = vol / vtot
        plates_meta.append({"name": name, "title": title, "file": fname,
                            "colour": ci, "hex": s.palette[ci].hex if ci is not None else None,
                            "origin_mm": [ox, oy], "size_mm": [P, P], "codes": codes,
                            "grams": round(mmod["grams"] * share), "hours": round(mmod["hours"] * share, 1)})
    glb = _glb(nodes)
    meta = {"schema": SCHEMA, "kind": "plates", "mode": s.mode, "units": "mm", "up": "+z (bed normal)",
            "plate_mm": P, "grid_columns": cols, "stride_mm": stride, "plates": plates_meta,
            "build_s": round(time.time() - t0, 2)}
    return _remember(key, (glb, meta))


def _signed_volume(v, t):
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def clear_cache():
    _CACHE.clear()


# ============================================================================= round 6: stones
def write_glb(nodes, materials) -> bytes:
    """Minimal fast GLB writer. nodes: [(name, V (n,3), T (m,3), material index, translation or None)],
    materials: [(name, rgb)]. Matte PBR, no normals (glTF viewers then shade flat, as cut stone)."""
    import json as _json
    import struct
    bufs, views, accs, meshes, gnodes = [], [], [], [], []
    off = 0

    def add(arr, target):
        nonlocal off
        b = arr.tobytes()
        pad = (-len(b)) % 4
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(b), "target": target})
        bufs.append(b + b"\x00" * pad)
        off += len(b) + pad
        return len(views) - 1
    for name, V, T, mat, tr in nodes:
        V = np.ascontiguousarray(V, np.float32)
        T = np.ascontiguousarray(T, np.uint32)
        vi = add(V, 34962)
        ti = add(T.ravel(), 34963)
        accs.append({"bufferView": vi, "componentType": 5126, "count": int(len(V)), "type": "VEC3",
                     "min": V.min(0).tolist(), "max": V.max(0).tolist()})
        accs.append({"bufferView": ti, "componentType": 5125, "count": int(T.size), "type": "SCALAR"})
        meshes.append({"name": name, "primitives": [{"attributes": {"POSITION": len(accs) - 2},
                                                     "indices": len(accs) - 1, "material": int(mat)}]})
        nd = {"name": name, "mesh": len(meshes) - 1}
        if tr is not None:
            nd["translation"] = [float(x) for x in tr]
        gnodes.append(nd)
    mats = [{"name": nm, "pbrMetallicRoughness": {"baseColorFactor": [c / 255 for c in rgb] + [1.0],
                                                  "metallicFactor": 0.0, "roughnessFactor": 0.92}}
            for nm, rgb in materials]
    doc = {"asset": {"version": "2.0", "generator": "Tessera"}, "scene": 0,
           "scenes": [{"nodes": list(range(len(gnodes)))}], "nodes": gnodes, "meshes": meshes,
           "materials": mats, "accessors": accs, "bufferViews": views,
           "buffers": [{"byteLength": off}]}
    js = _json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    binb = b"".join(bufs)
    total = 12 + 8 + len(js) + 8 + len(binb)
    return (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(binb), 0x004E4942) + binb)


def _stone_layout(m: Mosaic):
    key = _key(m, "stone_layout")
    if key in _CACHE:
        return _CACHE[key]
    from ..geometry.stones import build_stones
    return _remember(key, build_stones(m))


def _stone_meshes(m: Mosaic, L, quality, progress=None):
    """Per stone (V, T) in the artwork frame. preview: bevelled prisms (numpy, ~0.3 ms each);
    full: exact export solids with the carved grooves."""
    key = _key(m, "stone_meshes", quality)
    if key in _CACHE:
        return _CACHE[key]
    from ..geometry.stones import stone_arrays, stone_solid
    out = []
    n = len(L.stones)
    for k, st in enumerate(L.stones):
        if quality == "full":
            man, _ = stone_solid(L, st, engrave=False)
            mesh = man.to_mesh()
            out.append((np.asarray(mesh.vert_properties, np.float64)[:, :3], np.asarray(mesh.tri_verts, np.int64)))
        else:
            out.append(stone_arrays(st))
        if progress and k % 200 == 0:
            progress(0.15 + 0.7 * k / max(n, 1), f"Taş {k + 1}/{n}")
    return _remember(key, out)


def scene_glb_stones(m: Mosaic, quality: str = "preview", progress=None):
    t0 = time.time()
    s = m.settings
    prog = progress or (lambda f, msg: None)
    prog(0.0, "Taşlar hazırlanıyor")
    L = _stone_layout(m)
    prog(0.15, "Taş modelleri")
    meshes = _stone_meshes(m, L, quality, prog)
    pieces = L.mode == Mode.PIECES
    base_t = 0.8 if not pieces else 0.0
    mats = [(f"{f.name}", f.rgb) for f in s.palette]
    nodes, metas = [], []
    for st, (V, T) in zip(L.stones, meshes):
        name = f"stone:{st.code}"
        V = V + (0.0, 0.0, base_t)
        nodes.append((name, V, T, st.colour, None))
        bb = _bbox(V)
        metas.append({"name": name, "type": "part", "code": st.code, "colour": int(st.colour),
                      "colour_name": s.palette[st.colour].name, "hex": s.palette[st.colour].hex,
                      "panel": f"P{st.panel + 1}", "template_page": int(st.page), "plate": None,
                      "size_mm": [round(bb[3] - bb[0], 1), round(bb[4] - bb[1], 1), round(bb[5] - bb[2], 1)],
                      "area_mm2": round(st.area_mm2, 1), "width_mm": round(st.width, 2),
                      "profile": int(st.profile), "side_printed": bool(st.side and pieces), "bbox": bb,
                      "explode": {"dir": [0.0, 0.0, 1.0], "distance_mm": EXPLODE_PART_MM}})
    if not pieces:     # AMS: thin base per panel in the neutral base colour
        mats.append(("taban", s.palette[L.base_colour].rgb))
        for pn in L.panels:
            v, t = _arrays(_cs(pn.region.buffer(-0.01)).extrude(base_t))
            name = f"backing:{pn.name}"
            nodes.append((name, v, t, len(mats) - 1, None))
            metas.append(_backing_meta(name, pn.name, L.base_colour, s, v))
    prog(0.9, "GLB yazılıyor")
    glb = write_glb(nodes, mats)
    w_mm, h_mm = m.size_mm
    meta = {"schema": SCHEMA, "kind": "artwork", "quality": quality, "mode": s.mode,
            "style": getattr(s, "style", ""), "units": "mm", "up": "+y", "facing": "+z (out of the wall)",
            "origin": "artwork bottom-left", "size_mm": [round(w_mm, 1), round(h_mm, 1)], "backing_t_mm": base_t,
            "palette": _palette_meta(m, L.letters),
            "panels": [{"name": pn.name, "bounds": [round(float(x), 2) for x in pn.bounds]} for pn in L.panels],
            "template_pages": list(L.pages), "nodes": metas,
            "counts": {"parts": len(L.stones), "backings": 0 if pieces else len(L.panels), "cleats": 0},
            "build_s": round(time.time() - t0, 2)}
    prog(1.0, "Hazır")
    return glb, meta


def stone_plates(m: Mosaic) -> list:
    """The single-colour plates as exported: [(colour, stones, placements, number of plates)]."""
    from ..geometry.packing import pack_outlines
    from ..geometry.stones import side_transform, stone_arrays
    s = m.settings
    key = _key(m, f"stone_plates {s.plate_mm}")
    if key in _CACHE:
        return _CACHE[key]
    L = _stone_layout(m)

    def foot(st):
        if not st.side:
            return st.poly
        return shapely.MultiPoint((stone_arrays(st)[0] @ side_transform(st)[:3, :3].T)[:, :2]).convex_hull
    out = []
    for ci in range(len(s.palette)):
        items = sorted([st for st in L.stones if st.colour == ci], key=lambda st: (st.page, st.panel, st.number))
        if items:
            groups = [st.page for st in items] if getattr(s, "plate_order", "panel") not in ("max", "renk") else None
            pl, n, _ = pack_outlines([foot(st) for st in items], s.plate_mm, L.params.plate_margin, 2.5, groups=groups)
            out.append((ci, items, pl, n))
    return _remember(key, out)


def plates_glb_stones(m: Mosaic, gap_mm: float = 40.0, progress=None):
    """Plates exactly as exported (same footprints, same packing, same order)."""
    from ..geometry.layout import ascii_name
    from ..geometry.stones import side_transform
    from .mass import mass_model
    t0 = time.time()
    s = m.settings
    L = _stone_layout(m)
    meshes = _stone_meshes(m, L, "preview", progress)
    gp = L.params
    P = s.plate_mm
    beds = []
    if L.mode == Mode.PIECES:
        printed = []
        for st, (V, T) in zip(L.stones, meshes):
            if st.side:
                M = side_transform(st)
                V = V @ M[:3, :3].T
                V[:, 2] -= V[:, 2].min()
            printed.append((V, T))
        for ci, items, pl, n_pl in stone_plates(m):
            f = s.palette[ci]
            content = [[] for _ in range(n_pl)]
            for q in pl:
                st = items[q.item]
                V, T = printed[st.index]
                a = math.radians(q.angle)
                VV = V.copy()
                VV[:, 0] = V[:, 0] * math.cos(a) - V[:, 1] * math.sin(a) + q.dx
                VV[:, 1] = V[:, 0] * math.sin(a) + V[:, 1] * math.cos(a) + q.dy
                content[q.plate].append((f"stone:{st.code}", VV, T, ci, st.code))
            for p in range(n_pl):
                beds.append((f"{f.name} {p + 1}", ci, content[p], f"{ci + 1}_{ascii_name(f.name)}_tablalar.3mf"))
    else:
        for pn in L.panels:
            x0, y0, x1, y1 = pn.bounds
            off = np.array((P - gp.plate_margin - (x1 - x0) - x0, P - gp.plate_margin - (y1 - y0) - y0, 0.0))
            v, t = _arrays(_cs(pn.region.buffer(-0.01)).extrude(0.8))
            items = [(f"backing:{pn.name}", v + off, t, L.base_colour, f"Taban {pn.name}")]
            for i in pn.stones:
                V, T = meshes[i]
                items.append((f"stone:{L.stones[i].code}", V + off + (0, 0, 0.8), T, L.stones[i].colour,
                              L.stones[i].code))
            beds.append((f"Panel {pn.name}", None, items, "AMS_paneller.3mf"))
    mmod = mass_model(m, plates=len(beds))
    cols = max(1, math.ceil(math.sqrt(len(beds))))
    stride = P + gap_mm
    mats = [(f.name, f.rgb) for f in s.palette] + [("tabla", (58, 58, 62))]
    bed_v = np.array([[0, 0, -2], [P, 0, -2], [P, P, -2], [0, P, -2], [0, 0, 0], [P, 0, 0], [P, P, 0], [0, P, 0]],
                     float)
    bed_t = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    vols = [sum(abs(_signed_volume(v, t)) for _, v, t, *_ in items) for _, _, items, _ in beds]
    vtot = sum(vols) or 1.0
    nodes, pm = [], []
    for k, ((title, ci, items, fname), vol) in enumerate(zip(beds, vols)):
        ox, oy = (k % cols) * stride, -(k // cols) * stride
        nodes.append((f"bed:{k + 1}", bed_v, bed_t, len(mats) - 1, (ox, oy, 0)))
        for nm, v, t, c, code in items:
            nodes.append((f"{nm}@bed{k + 1}", v, t, c, (ox, oy, 0)))
        share = vol / vtot
        pm.append({"name": f"bed:{k + 1}", "title": title, "file": fname, "colour": ci,
                   "hex": s.palette[ci].hex if ci is not None else None, "origin_mm": [ox, oy], "size_mm": [P, P],
                   "codes": [it[4] for it in items], "grams": round(mmod["grams"] * share),
                   "hours": round(mmod["hours"] * share, 1)})
    glb = write_glb(nodes, mats)
    return glb, {"schema": SCHEMA, "kind": "plates", "mode": s.mode, "units": "mm", "up": "+z (bed normal)",
                 "plate_mm": P, "grid_columns": cols, "stride_mm": stride, "plates": pm,
                 "build_s": round(time.time() - t0, 2)}
