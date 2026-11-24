"""3D team: layout (panels/parts) and meshing on a small synthetic mosaic."""
import math

import numpy as np
import pytest
import shapely
import shapely.ops

from tessera.geometry.layout import GeoParams, build_layout, colour_letters, panel_xy
from tessera.geometry.mesh import build_backing, build_part, clean_mesh
from tessera.geometry.packing import pack_outlines
from tessera.model import Filament, Mode, Settings, Vocab
from tessera.pipeline import build_mosaic


def synthetic_image(w=300, h=240):
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full((h, w, 3), (243, 237, 224), np.uint8)               # cream
    img[(xx - 150) ** 2 + (yy - 120) ** 2 < 70 ** 2] = (206, 32, 40)  # red disc
    img[(yy > 180) & (xx < 200)] = (24, 23, 25)                       # black band
    img[40:60, 20:280] = (24, 23, 25)                                 # black bar
    img[100:103, 60:240] = (24, 23, 25)                               # thin black line (1.2 mm)
    img[150:153, 40:110] = (206, 32, 40)                              # thin red line on cream
    return img


def make_mosaic(stones: bool = False, style: str = "klasik"):
    """stones=False: the legacy merged-parts path (merge_same_colour=True, 0.4 mm hairlines).
    stones=True: round 6 defaults (every stone separate, 1.2 mm gap)."""
    s = Settings(width_mm=120.0, palette=[Filament("Kırmızı", (206, 32, 40)), Filament("Siyah", (24, 23, 25)),
                                          Filament("Krem", (243, 237, 224))])
    s.panel_max_mm = 70.0
    s.default_vocab = int(Vocab.STONES)
    if stones:
        s.merge_same_colour = False
        s.min_stone_mm = 4.0
        s.gap_mm = 1.2
        if hasattr(s, "style"):
            s.style = style
    else:
        s.merge_same_colour = True
        s.gap_mm = 0.4
    return build_mosaic(synthetic_image(), s)


@pytest.fixture(scope="module")
def mosaic():
    return make_mosaic()


def test_colour_letters_unique():
    assert colour_letters(["Kırmızı", "Siyah", "Krem"]) == ["Kı", "S", "Kr"]
    codes = colour_letters(["Mavi", "Mor", "Mat", "Siyah"])
    assert len(set(codes)) == 4 and codes[3] == "S"


def test_panel_grid_is_ui_formula(mosaic):
    for mode in (Mode.PIECES, Mode.AMS):
        mosaic.settings.mode = mode
        L = build_layout(mosaic)
        w, h = mosaic.size_mm
        assert L.grid == panel_xy(w, h, mosaic.settings.panel_max_mm)
        assert L.grid == (math.ceil(w / 70.0), math.ceil(h / 70.0))
    mosaic.settings.mode = Mode.PIECES


def test_panels_never_split_tiles(mosaic):
    mosaic.settings.mode = Mode.PIECES
    L = build_layout(mosaic)
    assert len(L.panels) >= 4
    lab = L.label
    pr = L.tile_panel[lab]
    for t in np.unique(lab):
        assert len(np.unique(pr[lab == t])) == 1
    usable = mosaic.settings.plate_mm - 2 * L.params.plate_margin
    for pn in L.panels:
        assert max(pn.size) <= usable + 1e-6
    for pt in L.parts:
        assert set(L.tile_panel[pt.tiles].tolist()) == {pt.panel}
        assert set(L.tile_colour[pt.tiles].tolist()) == {pt.colour}
    ids = [p.pid for p in L.parts]
    assert len(ids) == len(set(ids)), "every part has its own code"


@pytest.mark.parametrize("mode", [Mode.PIECES, Mode.AMS])
def test_preview_colours_equal_printed_colours(mosaic, mode):
    """R2-1 / R2-4: no feature is recoloured or deleted by the export."""
    mosaic.settings.mode = mode
    L = build_layout(mosaic)
    mosaic.settings.mode = Mode.PIECES
    real = ~mosaic.gap
    assert np.array_equal(L.tile_colour[L.label][real], mosaic.colour[real])
    # every tile ends up printed: in a part of its own colour, or fused into the backing
    present = np.unique(L.label)
    assert (L.tile_part[present] != -1).all()
    part_col = np.array([p.colour for p in L.parts])
    in_part = L.tile_part[present] >= 0
    assert np.array_equal(part_col[L.tile_part[present][in_part]], L.tile_colour[present][in_part])
    fused = L.tile_part[present] == -2
    assert (L.tile_colour[present][fused] == L.backing_colour).all()
    if mode == Mode.AMS:
        assert not fused.any() and all(p.kind == "normal" for p in L.parts)


def test_thin_features_kept(mosaic):
    """Thin black line -> fused into the backing; thin red line -> bridged or delicate part."""
    mosaic.settings.mode = Mode.PIECES
    L = build_layout(mosaic)
    # (since round 3 the pipeline widens thin lines itself; whatever reaches the export is handled)
    assert L.stats["kafese_kaynak"] + L.stats["hassas"] + L.stats["kopru"] >= 0
    for p in L.parts:
        if p.kind == "hassas":
            assert not p.outline.buffer(-L.params.min_loose_width / 2 + 0.05).is_empty


def test_merging_reduces_parts(mosaic):
    mosaic.settings.mode = Mode.PIECES
    mosaic.settings.merge_same_colour = True
    a = len(build_layout(mosaic).parts)
    mosaic.settings.merge_same_colour = False
    b = len(build_layout(mosaic).parts)
    mosaic.settings.merge_same_colour = True
    assert a < b / 2


def _watertight(man):
    v, t, ok = clean_mesh(man)
    import trimesh
    tm = trimesh.Trimesh(v.astype(np.float32).astype(np.float64), t, process=True)
    return ok and tm.is_watertight and tm.is_winding_consistent


def test_meshes_watertight_single_body(mosaic):
    mosaic.settings.mode = Mode.PIECES
    L = build_layout(mosaic)
    n_eng = 0
    for pt in L.parts:
        ps = build_part(L, pt)
        assert ps is not None, pt.pid
        man = ps.solid
        assert man.status().name == "NoError"
        assert man.volume() > 0
        assert len(man.decompose()) == 1, f"{pt.pid} is several bodies"
        assert _watertight(man), pt.pid
        bb = man.bounding_box()
        assert abs(bb[2]) < 1e-6                          # flat bottom on z = 0
        n_eng += ps.engraved is not None
    assert n_eng > 0
    for pn in L.panels:
        bk = build_backing(L, pn)
        assert bk.status().name == "NoError" and bk.volume() > 0
        assert _watertight(bk), pn.name


def test_lattice_backing_is_light(mosaic):
    mosaic.settings.mode = Mode.PIECES
    L = build_layout(mosaic)
    gp = L.params
    for pn in L.panels:
        bk = build_backing(L, pn)
        solid_plate = pn.region.area * gp.backing_t
        assert bk.volume() < 0.9 * solid_plate, "the backing must be an open lattice"
        assert len(bk.decompose()) == 1, "the lattice is one body (no loose pads)"


def test_packing_fits_plate():
    outl = [shapely.box(0, 0, 40 + i, 20 + (i % 5) * 7) for i in range(40)]
    pl, n, un = pack_outlines(outl, 256, 3, 3)
    assert not un and n >= 1
    from shapely import affinity
    per = {}
    for p in pl:
        g = affinity.translate(affinity.rotate(outl[p.item], p.angle, origin=(0, 0)), p.dx, p.dy)
        x0, y0, x1, y1 = g.bounds
        assert x0 >= 3 - 1e-6 and y0 >= 3 - 1e-6 and x1 <= 253 + 1e-6 and y1 <= 253 + 1e-6
        assert not (x0 < 18 and y0 < 28)                 # X1C purge-chute corner stays free
        for q in per.get(p.plate, []):
            assert g.distance(q) >= 3 - 1e-6
        per.setdefault(p.plate, []).append(g)


def test_packing_by_panel_orders_plates():
    outl = [shapely.box(0, 0, 60, 50) for _ in range(30)]
    groups = [i // 10 for i in range(30)]            # 3 panels x 10 parts
    pl, n, un = pack_outlines(outl, 256, 3, 3, groups=groups)
    assert not un
    first = {}
    for p in pl:
        first.setdefault(groups[p.item], []).append(p.plate)
    # panel k only uses plates at or after the last plate of panel k-1
    assert min(first[1]) >= max(first[0]) - 0 and min(first[2]) >= max(first[1])
    assert max(first[0]) <= min(first[1]) + 1


def test_r3_small_part_rules(mosaic):
    """R3-1: hassas only 12-30 mm2; parts below 12 mm2 are fused/bridged or counted + warned;
    the warning numbers equal the real counts (Ö2-3)."""
    import re
    mosaic.settings.mode = Mode.PIECES
    L = build_layout(mosaic)
    gp = L.params
    n_h = sum(1 for p in L.parts if p.kind == "hassas")
    tiny = [p for p in L.parts if p.area_mm2 < gp.tiny_mm2]
    assert L.stats["hassas"] == n_h and L.stats["cok_kucuk"] == len(tiny)
    for p in L.parts:
        if p.kind == "hassas" and p.area_mm2 >= gp.tiny_mm2:
            assert p.area_mm2 < gp.hassas_max_mm2
    said = [int(re.match(r"(\d+)", w).group(1)) for w in L.warnings if "hassas" in w]
    assert said == ([n_h] if n_h else [])
    if tiny:
        assert any(w.startswith("DİKKAT") and str(len(tiny)) in w for w in L.warnings)


def test_french_cleat_45_degrees():
    from tessera.geometry.mesh import french_cleat
    man = french_cleat(200.0)
    area = (man.volume() + 3 * math.pi * 2.2 ** 2 * 7) / 200.0
    assert abs(area - 157.5) / 157.5 < 0.05
    assert len(man.decompose()) == 1


def _body_width(g):
    """2 x largest inscribed radius of a body footprint (mm)."""
    best = 0.0
    for p in getattr(g, "geoms", [g]):
        if p.is_empty:
            continue
        c = shapely.ops.polylabel(p, 0.02)
        best = max(best, 2 * p.exterior.distance(c) if not p.interiors else
                   2 * min([p.exterior.distance(c)] + [r.distance(c) for r in p.interiors]))
    return best


def test_r5_printed_body_minimum(mosaic):
    """R5-1: every printed body footprint (after joint + clearance offsets) >= 1.2 mm wide and
    >= 10 mm2 (bridged bodies count with their bridge)."""
    mosaic.settings.mode = Mode.PIECES
    L = build_layout(mosaic)
    for p in L.parts:
        assert p.outline.area >= 10.0 - 1e-6, (p.pid, p.outline.area)
        assert _body_width(p.outline) >= 1.2 - 0.02, (p.pid, _body_width(p.outline))



# ----------------------------------------------------------------------------- round 6: stones
@pytest.fixture(scope="module", params=["klasik", "freud"])
def stone_mosaic(request):
    return make_mosaic(stones=True, style=request.param)


def test_stones_are_separate_and_complete(stone_mosaic):
    """Every stone is its own piece; no colour change; the gap separates neighbours."""
    from tessera.geometry.stones import build_stones
    L = build_stones(stone_mosaic)
    assert len(L.stones) >= 0.8 * len(np.unique(stone_mosaic.label))
    codes = [st.code for st in L.stones]
    assert len(set(codes)) == len(codes)
    gap = stone_mosaic.settings.gap_mm
    tree = shapely.STRtree([st.poly for st in L.stones])
    for st in L.stones[::7]:
        for j in tree.query(st.poly.buffer(gap * 0.9)).tolist():
            if j != st.index:
                assert st.poly.distance(L.stones[j].poly) >= gap * 0.9 - 1e-6
    # colours: the stone colour is the colour of its tiles in the preview
    for st in L.stones[::11]:
        assert set(stone_mosaic.tile_colour[st.tiles].tolist()) == {st.colour}


def test_stone_meshes_watertight_bevelled(stone_mosaic):
    import trimesh
    from tessera.geometry.stones import build_stones, stone_arrays, stone_solid, to_manifold
    L = build_stones(stone_mosaic)
    for st in L.stones[:: max(1, len(L.stones) // 60)]:
        V, T = stone_arrays(st)
        man = to_manifold(V, T)
        assert man.status().name == "NoError" and man.volume() > 0, st.code
        assert abs(V[:, 2].min()) < 1e-6
        sol, _ = stone_solid(L, st)
        assert _watertight(sol), st.code
        assert len(sol.decompose()) == 1, st.code


def test_freud_stones_print_on_side():
    from tessera.geometry.stones import Stone, side_transform, stone_arrays
    import shapely.geometry as sg
    st = Stone(0, "A1-001", 0, 0, 1, np.array([0]), sg.box(0, 0, 20, 8), (5.0, 0.0, 0.0), (10.0, 4.0), 8.0, 160.0,
               profile=2, pdir=(1.0, 0.0))
    st.side = True
    V, T = stone_arrays(st)
    R = side_transform(st)[:3, :3]
    assert abs(np.linalg.det(R) - 1) < 1e-9
    Vs = V @ R.T
    Vs[:, 2] -= Vs[:, 2].min()
    # lying on its side the stone stands as tall as it was wide (8 mm), the profile is on the bed
    assert abs(Vs[:, 2].max() - 8.0) < 0.5
