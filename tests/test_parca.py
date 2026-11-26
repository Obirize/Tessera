"""How many stones, and the one-piece base plate they sit in."""
import json
import os

import numpy as np
import pytest
import shapely

from tessera import app2_engine as E
from tessera.export.api import estimate, export_project, plates_glb, scene_glb
from tessera.geometry.plate import CLEAR, FLOOR, build_plate, walls
from tessera.geometry.stones import build_stones, stone_arrays
from tessera.model import Mode

CAT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ornekler", "sablonlar", "Tessera kedisi.png")


@pytest.fixture(scope="module")
def cat():
    r = E.Recipe(image_path=CAT, width_mm=150.0)
    r.use_colours(E.analyse(CAT, 150)[0])
    return r


def make(cat, parca, base="Siyah", mode=Mode.PIECES, width=150.0, style="Mozaik"):
    r = cat.copy()
    r.width_mm, r.mode, r.style = width, mode, style
    r.fine.update(parca=parca, tek_taban=E.main_hex(base) if base else None)
    return E.build_mosaic_for(r)[0]


@pytest.fixture(scope="module")
def whole(cat):
    return make(cat, "sekil")


def test_every_shape_one_stone(whole):
    names = [whole.settings.palette[c].name for c in whole.tile_colour]
    assert whole.n_tiles == 36 == whole.stats["alan"] and "Siyah" not in names
    assert whole.plate is not None and whole.settings.palette[whole.plate_colour].name == "Siyah"


@pytest.mark.parametrize("n,base", [(20, "Siyah"), (100, "Siyah"), (300, "Siyah"), (60, None)])
def test_as_many_stones_as_asked_for(cat, n, base):
    m = make(cat, n, base)
    assert abs(m.n_tiles - n) <= 0.02 * n
    for i, p in enumerate(m.polys):
        assert p.is_valid and p.area >= 1.5, i
    tree = shapely.STRtree(m.polys)
    a, b = tree.query(m.polys, predicate="overlaps")
    assert all(m.polys[i].intersection(m.polys[j]).area < 0.05 for i, j in zip(a, b))      # no stone lies on another


@pytest.mark.parametrize("base", ["Siyah", None])
def test_few_stones_keep_the_nose(cat, base):
    m = make(cat, 20, base, width=100.0)
    names = [m.settings.palette[c].name for c in m.tile_colour]
    nose = [n for p, n in zip(m.polys, names) if p.contains(shapely.Point(50, 58))]
    assert nose == ["Pembe"]                                                       # not black
    assert {"Mavi", "Altın"} <= set(names) and 20 <= m.n_tiles <= 22


def test_features_cost_a_few_stones_more_at_most():
    from tessera.pipeline import kesim, parca
    from tessera.pipeline.palette import rgb_to_lab
    lab = np.zeros((200, 200), np.int16)                       # 100 x 100 mm, white with four pink dots
    for y, x in ((30, 30), (30, 160), (160, 30), (160, 160)):
        lab[y:y + 10, x:x + 10] = 1
    plab = rgb_to_lab(np.array([[240, 240, 235], [230, 140, 160]], np.uint8))
    out, over = parca.reduce(lab, 2, plab, 0.5, 4)
    assert over == 1 and (out == lab).all()                    # one stone more than asked: every dot stays
    out, over = parca.reduce(lab, 2, plab, 0.5, 2)
    assert over == 1 and kesim.count_areas(out, 2) == 3        # asked for far fewer: only one more, two dots go


def test_a_big_area_keeps_its_colour_before_a_feature():
    from tessera.pipeline import kesim, parca
    from tessera.pipeline.palette import rgb_to_lab
    lab = np.zeros((200, 200), np.int16)                       # white, a pale square (cheap to lose), black dots
    lab[120:150, 120:150] = 1
    for y, x in ((30, 30), (30, 160), (160, 30)):
        lab[y:y + 10, x:x + 10] = 2
    plab = rgb_to_lab(np.array([[240, 240, 235], [190, 175, 150], [25, 25, 28]], np.uint8))
    out, over = parca.reduce(lab, 3, plab, 0.5, 2)
    assert (out[120:150, 120:150] == 1).all() and over == 1 and kesim.count_areas(out, 3) == 3


def test_cat_at_twelve_keeps_its_background_and_ears(cat):
    m = make(cat, 12, None, width=100.0)
    names = [m.settings.palette[c].name for c in m.tile_colour]
    at = lambda x, y: [n for p, n in zip(m.polys, names) if p.contains(shapely.Point(x, y))]
    assert at(50, 8) == ["Kırmızı"] and at(23, 22) == at(77, 22) == ["Pembe"] and m.n_tiles <= 13


def test_texture_decides_when_nothing_is_asked(cat):
    m = make(cat, None, None)
    assert m.plate is None and m.n_tiles > 300 and m.stats["alan"] == 37
    assert m.settings.palette[m.stats["cizgi"]].name == "Siyah"                      # the lines: offered as a plate
    k = make(cat, None)
    assert k.n_tiles == m.n_tiles - int((m.tile_colour == m.stats["cizgi"]).sum())    # the other stones stay as they were


def test_same_wish_same_stones(cat):
    a, b = make(cat, 100), make(cat, 100)
    assert all(p.equals(q) for p, q in zip(a.polys, b.polys))


def test_stones_fit_the_printer(cat):
    m = make(cat, "sekil", None, width=600.0)
    assert max(max(p.bounds[2] - p.bounds[0], p.bounds[3] - p.bounds[1]) for p in m.polys) <= 240


def test_stones_drop_into_the_plate(whole):
    L = build_stones(whole)
    pn, = L.panels
    body = build_plate(L, pn, numbers=True)
    assert len(body.decompose()) == 1 and body.volume() > 150 * 150 * FLOOR
    w = walls(pn.region, L.stones)
    assert all(w.distance(st.poly) > CLEAR - 0.03 for st in L.stones)
    from manifold3d import Manifold, Mesh
    for st in L.stones[:6]:
        v, t = stone_arrays(st)
        s = Manifold(Mesh(np.asarray(v, np.float32), np.asarray(t, np.uint32))).translate((0, 0, FLOOR + 0.01))
        assert (s ^ body).volume() < 0.01


def test_big_plate_is_cut_into_panels(cat):
    m = make(cat, "sekil", width=300.0)
    L = build_stones(m)
    assert len(L.panels) == 4
    for i, a in enumerate(L.panels):
        assert max(a.size) <= 240 and len(build_plate(L, a).decompose()) == 1
        assert all(a.region.distance(b.region) > 0.1 for b in L.panels[i + 1:])
    assert sum(len(p.stones) for p in L.panels) == 36


def test_plate_is_in_the_scene_and_on_the_plates(whole):
    _, meta = scene_glb(whole)
    back = [n for n in meta["nodes"] if n["type"] == "backing"]
    stones = [n for n in meta["nodes"] if n["type"] == "part"]
    assert len(back) == 1 and back[0]["colour_name"] == "Siyah" and len(stones) == 36
    assert all(abs(n["bbox"][2] - FLOOR) < 0.01 for n in stones)
    _, pm = plates_glb(whole)
    e = estimate(whole)
    assert pm["plates"][0]["title"] == "Taban P1" and len(pm["plates"]) == e["tabla"]
    assert sum(p["grams"] for p in pm["plates"]) == pytest.approx(e["gram"], abs=4)
    assert pm["plates"][0]["grams"] == pytest.approx(e["kafes_gram"], abs=1)


def test_export_writes_the_plate(whole, tmp_path):
    res = export_project(whole, str(tmp_path), lambda f, m: True)
    assert os.path.exists(tmp_path / "02_tablalar" / "0_Taban_Siyah.3mf") and os.path.exists(tmp_path / "03_stl" / "taban_P1.stl")
    assert res["taban_renk"] == "Siyah" and len(res["plates"]) == estimate(whole)["tabla"] and not res["kapali_olmayan_govde"]
    assert "1 parça Siyah taban" in (tmp_path / "rapor.txt").read_text(encoding="utf-8")


def test_ams_prints_the_lines_with_the_panel(cat):
    m = make(cat, "sekil", mode=Mode.AMS)
    black = [f.name for f in m.settings.palette].index("Siyah")
    assert m.plate is None and int((m.tile_colour != black).sum()) == 36 and (m.tile_colour == black).any()
    assert E.piece_info(m)["sayi"] == 36 and m.n_tiles > 36
    L = build_stones(m)
    assert max(max(p.size) for p in L.panels) <= 245


def test_repainting_the_lines_keeps_the_stones(cat):
    r = cat.copy()
    r.fine.update(parca="sekil", tek_taban=E.main_hex("Siyah"))
    k = r.build_key(True)
    cols = [dict(c, ad="Lacivert", hex=E.main_hex("Lacivert")) if c["ad"] == "Siyah" else c for c in r.colours]
    r.fine["tek_taban"] = E.main_hex("Lacivert")
    r.use_colours(cols)
    assert r.fine["tek_taban"] == E.main_hex("Lacivert") and r.build_key(True) == k
    r.fine["parca"] = 20
    assert r.build_key(True) != k


def test_wishes_are_cleaned():
    assert [E.piece_count(v) for v in (None, "sekil", 1, 100.4, 10 ** 9, "bozuk", True)] == [None, "sekil", 2, 100, 20000, None, None]


def test_numbers_for_the_control(whole, cat):
    p = E.piece_info(whole)
    assert p["alan"] == 36 and p["en_az"] < 36 < p["en_cok"] and p["cizgi"]["ad"] == "Siyah"
    json.dumps(p)
    photo = os.path.join(os.path.dirname(CAT), "..", "poster", "kaynak.webp")
    r = E.Recipe(image_path=os.path.abspath(photo), width_mm=200.0)
    r.use_colours(E.analyse(r.image_path, 200)[0])
    r.fine["parca"] = 150
    m = E.build_mosaic_for(r)[0]
    assert E.piece_info(m)["cizgi"] is None and abs(m.n_tiles - 150) <= 3              # an ordinary picture gets no plate offered


def test_removed_lines_leave_no_knobs():
    """A window of 4 x 4 panes: where a piece of line goes, nothing of it stays stuck on the lines that cross it."""
    from tessera.pipeline import kesim, parca
    from tessera.pipeline.palette import rgb_to_lab
    t, bands = 12, [0, 114, 234, 354, 468]
    lab = np.zeros((480, 480), np.int16)
    for c in bands:
        lab[c:c + t] = 1
        lab[:, c:c + t] = 1
    plab = rgb_to_lab(np.array([[240, 238, 230], [25, 25, 28]], np.uint8)).astype(float)
    for K in (12, 5):
        out, _ = parca.reduce(lab, 2, plab, 0.25, K, base=1)
        assert kesim.count_areas(out, 2, 1) <= K
        blk, left = out == 1, 0
        for a, b in zip(bands[:-1], bands[1:]):
            for c in bands[1:-1]:
                for piece in (blk[a + t:b, c:c + t], blk[c:c + t, a + t:b]):
                    if piece.mean() < 0.9:                   # this piece was taken away
                        left += int(piece.sum())
        assert left <= 120                                   # before: 312 and 530 pixels of knobs
