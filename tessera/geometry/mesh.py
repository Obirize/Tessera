"""Layout -> watertight manifold3d solids (parts, backing panels, AMS panels).

Part  = web (part outline, inset for the backing ribs) + one prism per tile (inset by gap/2,
        top on the tile plane, edges chamfered by bevel) - mirrored id engraved in the bottom.
Backing = lattice (frame, edge members, bracing, pads) + ribs between parts + part numbers
        engraved in each pocket + panel name engraved into the top face.
All solids are in artwork mm coordinates (x right, y up), z = 0 on the backing top for parts
and z = 0 on the wall side for the backing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import shapely
import shapely.ops
from manifold3d import CrossSection, FillRule, Manifold, OpType
from shapely import affinity
from shapely.ops import nearest_points

from .layout import Layout, Part, Panel, _open
from .text import place_text, text_polygon


@dataclass
class PartSolid:
    part: Part
    solid: Manifold
    outline: object               # shapely, z=0 footprint of the printed part (mm)
    engraved: Optional[str]
    web_t: float
    height: float
    volume_mm3: float


def _polys(g):
    if g is None or g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    if hasattr(g, "geoms"):
        out = []
        for x in g.geoms:
            out.extend(_polys(x))
        return out
    return []


def cross_section(g) -> CrossSection:
    contours = []
    for p in _polys(g):
        if p.area < 1e-4:
            continue
        contours.append(np.asarray(p.exterior.coords)[:-1])
        for r in p.interiors:
            contours.append(np.asarray(r.coords)[:-1])
    return CrossSection(contours, FillRule.EvenOdd)


def extrude(g, h: float, z0: float = 0.0) -> Manifold:
    m = cross_section(g).extrude(h)
    return m.translate((0, 0, z0)) if z0 else m


def _inset(g, d):
    return g.buffer(-d, join_style="mitre", mitre_limit=3.0)


def _plane_fn(plane, cen):
    a, b, c = plane
    cx, cy = cen
    return lambda x, y: a + b * (x - cx) - c * (y - cy)


def tile_solid(poly, f, z_floor: float, bevel: float, z_base: float = 0.0) -> Optional[Manifold]:
    """Prism over `poly` from z_base up to z=f(x,y) (clamped >= z_floor) with a chamfered rim."""
    if poly.is_empty or poly.area < 0.03:
        return None
    ext = np.asarray(poly.exterior.coords)[:-1]
    inner = _inset(poly, bevel) if bevel > 0 else poly
    inner = max(_polys(inner), key=lambda q: q.area) if _polys(inner) else None
    convex = (not poly.interiors) and (poly.convex_hull.area - poly.area) <= 1e-3 * poly.area + 1e-4
    if convex:
        z_top = np.maximum(f(ext[:, 0], ext[:, 1]), z_floor)
        pts = [np.column_stack([ext, np.full(len(ext), z_base)])]
        if inner is not None and inner.area > 0.05 and bevel > 0:
            ie = np.asarray(inner.exterior.coords)[:-1]
            pts.append(np.column_stack([ext, np.maximum(z_top - bevel, z_floor - bevel * 0.5)]))
            pts.append(np.column_stack([ie, np.maximum(f(ie[:, 0], ie[:, 1]), z_floor)]))
        else:
            pts.append(np.column_stack([ext, z_top]))
        return Manifold.hull_points(np.vstack(pts))

    def warped(g, dz):
        m = cross_section(g).extrude(1.0)

        def w(v):
            v = np.array(v, dtype=np.float64, copy=True)
            top = np.maximum(f(v[:, 0], v[:, 1]) - dz, z_floor - dz)
            v[:, 2] = z_base + v[:, 2] * (top - z_base)
            return v
        return m.warp_batch(w)
    if inner is not None and inner.area > 0.05 and bevel > 0:
        return warped(poly, bevel) + warped(inner, 0.0)
    return warped(poly, 0.0)


def _long_axis_angle(g) -> float:
    try:
        rr = g.minimum_rotated_rectangle
        c = np.asarray(rr.exterior.coords)
        e1, e2 = c[1] - c[0], c[2] - c[1]
        e = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
        ang = math.degrees(math.atan2(e[1], e[0]))
    except Exception:
        ang = 0.0
    while ang > 90:
        ang -= 180
    while ang <= -90:
        ang += 180
    return ang


def part_outline(L: Layout, part: Part):
    return part.outline


def _main_tile(L: Layout, tiles):
    best, ba = None, -1.0
    for t in tiles.tolist():
        tp = L.tile_poly[t]
        if tp is not None and tp.area > ba:
            best, ba = t, tp.area
    return best


def _relief(L: Layout, tiles, outline, inset_gap: float):
    """Tile polygons of `tiles` clipped to `outline` + their plane functions, and the min top."""
    geoms, tops = [], []
    for t in tiles.tolist():
        tp = L.tile_poly[t]
        if tp is None:
            continue
        # opening removes point contacts (pieces touching at a vertex make non-manifold edges)
        g = _inset(_open(_inset(tp, inset_gap).intersection(outline), 0.03), 0.01)
        f = _plane_fn(L.tile_plane[t], L.tile_centroid_mm[t])
        for p in _polys(g):
            if p.area < 0.05:
                continue
            e = np.asarray(p.exterior.coords)
            tops.append(float(f(e[:, 0], e[:, 1]).min()))
            geoms.append((p, f))
    return geoms, (min(tops) if tops else None)


def build_part(L: Layout, part: Part, engrave: bool = True) -> Optional[PartSolid]:
    gp = L.params
    s = L.mosaic.settings
    outline = part.outline
    if outline is None or outline.is_empty:
        return None
    if part.kind == "hassas":
        # one prism on the plane of its main tile, small bevel; no engraving (number in guide)
        t = _main_tile(L, part.tiles)
        f = _plane_fn(L.tile_plane[t], L.tile_centroid_mm[t])
        solids = [tile_solid(p, f, gp.web_min + 0.4, min(s.bevel_mm, 0.15)) for p in _polys(outline)]
        solids = [m for m in solids if m is not None and not m.is_empty()]
        if not solids:
            return None
        body = Manifold.compose(solids) if len(solids) > 1 else solids[0]
        bb = body.bounding_box()
        return PartSolid(part, body, outline, None, 0.0, float(bb[5]), float(body.volume()))
    has_br = part.bridges is not None and not part.bridges.is_empty
    if has_br:
        # the pieces, overlapping the bridge strips a little so the union really fuses them
        main = outline.difference(part.bridges).buffer(0.3, join_style="mitre").intersection(outline)
    else:
        main = outline
    tile_geoms, min_top = _relief(L, part.tiles, main, s.gap_mm / 2)
    if not tile_geoms:
        # no tile survived the inset: plain slab at the main tile height
        t = _main_tile(L, part.tiles)
        f = _plane_fn(L.tile_plane[t], L.tile_centroid_mm[t])
        tile_geoms = [(p, f) for p in _polys(main)]
        min_top = min(float(f(*np.asarray(p.exterior.coords).T).min()) for p, _ in tile_geoms)
    web_t = float(np.clip(min_top - gp.groove_min, gp.web_min, s.base_thickness_mm))
    z_floor = web_t + gp.groove_min
    solids = []
    cavities = []
    for p, f in tile_geoms:
        m = tile_solid(p, f, z_floor, s.bevel_mm, z_base=min(0.1, web_t / 2))
        if m is not None and not m.is_empty() and m.volume() > 0.01:
            solids.append(m)
            cav = _cavity(p, f, z_floor, gp)
            if cav is not None:
                cavities.append(cav)
    # (walls are kept 0.01-0.02 mm apart: coplanar overlapping faces make non-manifold STL)
    if has_br:
        # bridges stay low, hidden at the bottom of the gap between the pieces
        bh = max(0.4, min(gp.bridge_h, web_t - 0.2))
        body = extrude(_inset(outline, 0.025), bh) + extrude(_inset(main, 0.04), web_t - 0.2, 0.2)
    else:
        body = extrude(_inset(outline, 0.025), web_t)
    if solids:
        body = body + Manifold.compose(solids)
    text_region = main
    if cavities:
        # tall tiles are hollow: open at the bottom, 1.2 mm walls, 1 mm roof
        body = body - Manifold.compose([c for c, _ in cavities])
        text_region = main.difference(shapely.union_all([g for _, g in cavities]).buffer(0.8))
    engraved = None
    if engrave:
        ang = _long_axis_angle(main)
        cands = [part.pid, f"{part.panel + 1}-{part.number}", f"{part.number}"]
        tg, engraved = place_text(text_region, cands, gp.text_heights, ang, mirror=True, margin=0.8)
        if tg is not None:
            body = body - extrude(tg, gp.engrave_depth + 0.05, -0.05)
    body = drop_crumbs(body)
    bb = body.bounding_box()
    return PartSolid(part, body, outline, engraved, web_t, float(bb[5]), float(body.volume()))


def drop_crumbs(man: Manifold, min_mm3: float = 2.0) -> Manifold:
    """Remove loose geometric crumbs (< 2 mm3, e.g. 0.4 mm slivers left by offsets) that would
    print as separate specks; real bodies are far larger."""
    bodies = man.decompose()
    if len(bodies) <= 1:
        return man
    keep = [b for b in bodies if b.volume() >= min_mm3]
    if not keep:
        return man
    return keep[0] if len(keep) == 1 else Manifold.compose(keep)


def _cavity(p, f, z_floor, gp):
    """Open-bottom void inside a tall tile prism (None when the tile is low or narrow)."""
    e = np.asarray(p.exterior.coords)
    top_min = float(np.maximum(f(e[:, 0], e[:, 1]), z_floor).min())
    if top_min < gp.hollow_min_h:
        return None
    inner = _inset(p, gp.hollow_wall)
    ps = [q for q in _polys(_open(inner, 0.2)) if q.area >= 4.0]
    if not ps:
        return None
    g = shapely.MultiPolygon(ps) if len(ps) > 1 else ps[0]
    h = top_min - gp.hollow_top
    return extrude(g, h + 0.2, -0.2), g


def _brace_grid(region, pitch: float, width: float):
    """Triangular bracing web (lines at 0/60/120 degrees) clipped to the region."""
    x0, y0, x1, y1 = region.bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    R = math.hypot(x1 - x0, y1 - y0) / 2 + pitch
    lines = []
    for ang in (0.0, 60.0, 120.0):
        a = math.radians(ang)
        d = np.array([math.cos(a), math.sin(a)])
        nrm = np.array([-d[1], d[0]])
        k = int(R // pitch) + 1
        for i in range(-k, k + 1):
            o = np.array([cx, cy]) + nrm * i * pitch
            lines.append(shapely.LineString([o - d * R, o + d * R]))
    web = shapely.MultiLineString(lines).buffer(width / 2, cap_style="flat", join_style="mitre")
    return web.intersection(region)


def backing_floor(L: Layout, panel: Panel, label_pockets: bool = True):
    """2D plan of the backing floor + the pocket-number placements.
    PIECES (lattice): perimeter frame + members under every part edge + triangular bracing +
    number pads + fused features; open elsewhere (the backing is glued to an MDF board).
    AMS: a full plate (the panel prints as one object)."""
    gp = L.params
    region = panel.region
    labels = []
    if label_pockets:
        for i in panel.parts:
            pt = L.parts[i]
            pocket = pt.outline if pt.bridges is None else pt.outline.difference(pt.bridges)
            tg, _ = place_text(pocket, [f"{pt.number}"], (3.0, 2.2, 1.6), _long_axis_angle(pocket), False, 0.3)
            if tg is not None:
                labels.append(tg)
    if not (gp.lattice and gp.rib_w > 0):
        return region, labels
    members = [region.difference(_inset(region, gp.frame_w))]
    insides = [_inset(L.parts[i].outline, gp.flange_w) for i in panel.parts]
    insides = [g for g in insides if not g.is_empty]
    members.append(region.difference(shapely.union_all(insides)) if insides else region)
    members.append(_brace_grid(region, gp.brace_pitch, gp.brace_w))
    members.extend(tg.envelope.buffer(1.0, join_style="mitre") for tg in labels)
    members.extend(fz.shape for fz in panel.fused if not fz.shape.is_empty)
    floor = _open(shapely.union_all(members).intersection(region), 0.3)
    return connect_pieces(floor, region, gp.brace_w), labels


def connect_pieces(g, region, width: float = 1.6):
    """Join every loose island of `g` to the main body with straight floor strips, so the
    backing prints as ONE body (no loose pads)."""
    ps = sorted(_polys(g), key=lambda q: -q.area)
    if len(ps) < 2:
        return g
    main = ps[0]
    for q in ps[1:]:
        if q.buffer(0.05).intersection(main).area > 0.3:      # really overlapping, not a corner touch
            main = _clean_union([main, q])
            continue
        a, b = nearest_points(q, main)
        line = shapely.LineString([a, b])
        if line.length > 1e-6:
            line = affinity.scale(line, 1 + 1.6 / line.length, 1 + 1.6 / line.length)
            strip = line.buffer(width / 2, cap_style="round").intersection(region)
        else:
            strip = a.buffer(width).intersection(region)
        joined = _clean_union([main, q, strip])
        if len(_polys(joined)) < len(_polys(main)) + 1 or q.area >= 150.0:
            main = joined
        # else: a small pad/island that cannot be joined inside the panel is left out (it would
        # print as a loose speck); its pocket number stays in the guide
    big = max(_polys(main), key=lambda x: x.area)
    return shapely.MultiPolygon([x for x in _polys(main) if x is big or x.area >= 150.0])         if len(_polys(main)) > 1 else main


def build_backing(L: Layout, panel: Panel, engrave: bool = True, label_pockets: bool = True) -> Manifold:
    """Backing ("kafes"): lattice floor (or AMS plate) + locating ribs + fused backing-colour
    features + pocket numbers + panel name with an up arrow engraved into the TOP face."""
    gp = L.params
    s = L.mosaic.settings
    region = panel.region
    floor, labels = backing_floor(L, panel, label_pockets)
    label_geo = None
    if engrave:
        # panel name + up arrow, engraved into the top face of its own pad
        try:
            big = max(_polys(_inset(region, 12.0)), key=lambda q: q.area)
            c = shapely.ops.polylabel(big, 1.0)
        except Exception:
            c = region.representative_point()
        tg = text_polygon(f"{panel.name}  ÜST", 6.0, False)
        arrow = shapely.Polygon([(0, 7), (-5, 0), (-1.8, 0), (-1.8, -7), (1.8, -7), (1.8, 0), (5, 0)])
        x0, y0, x1, y1 = tg.bounds
        g = affinity.translate(shapely.union_all([tg, affinity.translate(arrow, x0 - 8, 0)]), c.x, c.y)
        if _inset(region, 2.0).contains(g):
            label_geo = g
            floor = connect_pieces(_clean_union([floor, g.envelope.buffer(2.0, join_style="mitre")
                                                 .intersection(region)]), region, gp.brace_w)
    # tiny inset: lobes touching at a single vertex would give non-manifold edges
    fl = _inset(_open(floor, 0.05), 0.01)
    for _ in range(2):            # pieces that only touched along a hairline: join them for real
        if len(_polys(fl)) <= 1:
            break
        fl = _inset(_open(connect_pieces(fl, _inset(region, 0.01), gp.brace_w), 0.05), 0.005)
    body = extrude(fl, gp.backing_t)
    adds = []
    inner_region = _inset(region, 0.01)
    flat2d = []
    if gp.rib_w > 0:
        holes = [L.parts[i].outline.buffer(gp.clearance, join_style="mitre", mitre_limit=3.0)
                 for i in panel.parts]
        holes_u = shapely.union_all(holes) if holes else shapely.Polygon()
        flat2d.extend(p for p in _polys(region.difference(holes_u)) if p.area > 0.2)
    for fz in panel.fused:
        flat2d.extend(_polys(fz.shape))
    if flat2d:
        # ribs + webs of fused features: ONE 2D union, ONE extrusion (no coplanar walls);
        # kept inside the floor so nothing hangs in the air
        # opening 0.25: rib slivers under 0.5 mm cannot be printed and make the slicer report
        # "floating regions"
        rib2d = _open(shapely.union_all(flat2d).intersection(inner_region).intersection(_inset(floor, 0.03)), 0.25)
        if not rib2d.is_empty:
            adds.append(extrude(rib2d, gp.rib_h + 0.1, gp.backing_t - 0.1))
    for fz in panel.fused:
        # thin backing-colour features rise from the backing at their own relief height
        geoms, _ = _relief(L, fz.tiles, _inset(fz.shape, 0.03), s.gap_mm / 2)
        for p, f in geoms:
            m = tile_solid(p, f, gp.rib_h + 0.3, min(s.bevel_mm, 0.2), z_base=0.25)
            if m is not None and not m.is_empty() and m.volume() > 0.01:
                adds.append(m.translate((0, 0, gp.backing_t)))
    if adds:
        body = Manifold.batch_boolean([body] + adds, OpType.Add)
    cuts = [extrude(tg, 0.35, gp.backing_t - 0.3) for tg in labels]
    if label_geo is not None:
        # engraved from the TOP (a recess on the bed side would leave floating ceilings)
        cuts.append(extrude(label_geo, 0.5, gp.backing_t - 0.4))
    if cuts:
        body = body - Manifold.compose(cuts)
    return drop_crumbs(body)


def _clean_union(gs):
    g = shapely.union_all([x for x in gs if x is not None and not x.is_empty])
    return g if g.is_valid else shapely.make_valid(g)


def french_cleat(length: float) -> Manifold:
    """One printable 45-degree cleat strip (print two: one on the wall, one on the board).
    Lies on its 26 mm face; three countersunk 4.4 mm screw holes."""
    prof = shapely.Polygon([(0, 0), (26, 0), (26, 7), (7, 7)])      # 45 deg bevel: rises 7 over 7 mm
    strip = cross_section(prof).extrude(length)                        # z: length
    strip = strip.rotate((90, 0, 0)).rotate((0, 0, 90))                # length along x, lies flat
    bb = strip.bounding_box()
    strip = strip.translate((-bb[0], -bb[1], -bb[2]))
    bb = strip.bounding_box()
    holes = []
    for xc in (25.0, length / 2, length - 25.0):
        yc = (bb[1] + bb[4]) / 2
        holes.append(Manifold.cylinder(30, 2.2, 2.2, 32).translate((xc, yc, -5)))
        holes.append(Manifold.cylinder(2.3, 2.2, 4.5, 32).translate((xc, yc, bb[5] - 2.29)))
    return strip - Manifold.compose(holes)


def mesh_arrays(man: Manifold):
    mesh = man.to_mesh()
    return (np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3],
            np.asarray(mesh.tri_verts, dtype=np.int64))


def clean_mesh(man: Manifold):
    """Mesh arrays that stay watertight after float32 rounding and vertex merging (what every
    STL reader does). Returns (vertices, triangles, watertight)."""
    import trimesh
    import trimesh.repair  # noqa: F401
    best = None
    for tol in (1e-4, 1e-3, 5e-3):
        v, t = mesh_arrays(man.simplify(tol))
        v = v.astype(np.float32).astype(np.float64)
        tm = trimesh.Trimesh(v, t, process=True)
        tm.update_faces(tm.nondegenerate_faces(height=1e-7))
        tm.remove_unreferenced_vertices()
        ok = bool(tm.is_watertight and tm.is_winding_consistent)
        if not ok:
            # collapse near-coincident vertices left by sliver triangles, then close pinholes
            tm.merge_vertices(digits_vertex=4)
            tm.update_faces(tm.nondegenerate_faces(height=1e-7))
            tm.update_faces(tm.unique_faces())
            tm.remove_unreferenced_vertices()
            if not tm.is_watertight:
                try:
                    trimesh.repair.fill_holes(tm)
                except Exception:      # optional dependency (networkx) missing
                    pass
            ok = bool(tm.is_watertight and tm.is_winding_consistent and (tm.area_faces > 1e-12).all())
        if ok:
            return np.asarray(tm.vertices), np.asarray(tm.faces, dtype=np.int64), True
        if best is None:
            best = (np.asarray(tm.vertices), np.asarray(tm.faces, dtype=np.int64), False)
    return best
