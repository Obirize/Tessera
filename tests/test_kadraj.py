"""Crop, turn and outline: which part of the picture becomes the mosaic and what shape it gets."""
import json
import os
import time

import numpy as np
import pytest
import shapely
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings  # noqa: E402

from tessera import app2, app2_prep  # noqa: E402
from tessera import app2_engine as E  # noqa: E402
from tessera.export.api import estimate, export_project, plates_glb, scene_glb  # noqa: E402
from tessera.geometry.plate import panel_count  # noqa: E402
from tessera.geometry.stones import build_stones  # noqa: E402
from tessera.model import Mode  # noqa: E402
from tessera.pipeline import kadraj as K  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAT = os.path.join(ROOT, "ornekler", "sablonlar", "Tessera kedisi.png")
POSTER = os.path.join(ROOT, "ornekler", "poster", "kaynak.webp")


# ------------------------------------------------------------------ wishes and pixels
def test_nothing_cut_is_the_picture_itself():
    for k in (None, {}, {"kutu": [0, 0, 1, 1]}, {"kutu": "x"}, {"don": 360, "kutu": [0, 0, 1, 1]}, {"kutu": [.5, .5, .51, .51]}):
        assert K.clean_frame(k) is None
    assert K.clean_shape({"tur": "dikdortgen"}) is None and K.clean_shape({"tur": "yildiz"}) is None
    assert K.picture(CAT, None, None) is CAT                    # the same file is read as before
    assert K.clean_frame({"don": 450, "kutu": [.9, 0, .1, 2]}) == {"don": 90, "ayna": False, "kutu": [.1, 0.0, .9, 1.0]}
    assert K.clean_shape({"tur": "yuvarlak", "r": 9}) == {"tur": "yuvarlak", "r": .5}


@pytest.mark.parametrize("don,ayna", [(0, False), (90, False), (180, True), (270, False), (90, True)])
def test_points_go_where_the_pixels_go(tmp_path, don, ayna):
    a = np.full((60, 100, 3), 255, np.uint8)
    a[12:16, 70:74] = (255, 0, 0)                               # a red dot at (0.72, 0.233)
    p = str(tmp_path / "a.png")
    Image.fromarray(a).save(p)
    k = K.clean_frame({"don": don, "ayna": ayna, "kutu": [.02, .03, .98, .97]})
    got = K.picture(p, k, None)
    ys, xs = np.nonzero(got[..., 1] < 128)
    u, v = K.to_frame(.72, 14 / 60, k)
    assert abs((xs.mean() + .5) / got.shape[1] - u) < .02 and abs((ys.mean() + .5) / got.shape[0] - v) < .02


def test_outlines_are_exact():
    c = K.outline({"tur": "daire"}, 100, 100)
    assert c.area == pytest.approx(np.pi * 2500, rel=1e-3) and c.bounds == pytest.approx((0, 0, 100, 100), abs=.02)
    r = K.outline({"tur": "yuvarlak", "r": .2}, 100, 60)
    assert r.area == pytest.approx(6000 - (4 - np.pi) * 144, rel=1e-3)
    assert K.mask(c, (100, 100), 1.0).sum() == pytest.approx(c.area, rel=.01)


def test_outside_has_no_say_in_the_colours(tmp_path):
    a = np.full((200, 200, 3), 240, np.uint8)
    a[:30, :30] = (0, 0, 255)                                   # blue only in a corner the circle cuts off
    a[80:120, 80:120] = (200, 0, 0)
    p = str(tmp_path / "a.png")
    Image.fromarray(a).save(p)
    assert len(E.analyse(p, 100)[0]) == 3
    assert len(E.analyse(p, 100, frame=[None, {"tur": "daire"}])[0]) == 2


# ------------------------------------------------------------------ the cut
def make(path, width, sekil=None, kutu=None, mode=Mode.PIECES, **fine):
    r = E.Recipe(image_path=path, width_mm=width, mode=mode)
    z = E.sibling_zones(path)
    if z:
        r.shapes, r.zones_from_file, r.rays_center = list(z["sekiller"]), True, z.get("isin_merkezi")
        r.file_default_vocab = int(z.get("varsayilan", 0))
    r.set_frame({"kutu": kutu} if kutu else None, sekil)
    r.fine.update(fine)
    r.use_colours(E.analyse(path, width, frame=r.frame)[0])
    return r, E.build_mosaic_for(r)[0]


@pytest.fixture(scope="module")
def circle():
    return make(CAT, 150.0, {"tur": "daire"})[1]


def test_every_stone_lies_inside_the_outline(circle):
    edge = circle.outline.buffer(-circle.settings.gap_mm / 2 + .01)
    assert all(edge.covers(p) for p in circle.polys)
    assert min(p.area for p in circle.polys) >= 1.5
    filled = sum(p.area for p in circle.polys)
    assert filled > .84 * circle.outline.area                   # nothing is missing but the joints
    rim = shapely.union_all(circle.polys).buffer(.5)
    assert rim.covers(circle.outline.buffer(-1.5))


@pytest.mark.parametrize("parca", ["sekil", 30, 120])
def test_piece_count_inside_an_outline(parca):
    r, m = make(CAT, 150.0, {"tur": "oval"}, [0, .1, 1, .9], parca=parca, tek_taban=E.main_hex("Siyah"))
    if parca != "sekil":
        assert abs(m.n_tiles - parca) <= 1
    assert m.plate.equals(m.outline) and all(m.outline.covers(p) for p in m.polys)
    assert E.piece_info(m)["cizgi"]["ad"] == "Siyah"


def test_cut_picture_keeps_the_file_zones():
    r, m = make(POSTER, 200.0, None, [0, 0, 1, .45])
    assert m.size_mm[1] == pytest.approx(200 * 1.4 * .45, rel=.02)
    _, rc = r.zones((1000, 1400))
    z = E.sibling_zones(POSTER)["isin_merkezi"]
    assert rc[0] == pytest.approx(z[0]) and rc[1] == pytest.approx(z[1] / .45)
    assert len(np.unique(m.tile_zone)) > 1


# ------------------------------------------------------------------ what comes out
def test_circle_goes_through_everything(circle, tmp_path):
    L = build_stones(circle)
    assert len(L.panels) == panel_count(circle) == 1
    _, meta = scene_glb(circle)
    _, pm = plates_glb(circle)
    e = estimate(circle)
    assert len(pm["plates"]) == e["tabla"]
    assert e["kg_m2"] == pytest.approx(e["gram"] / 1000 / (circle.outline.area / 1e6), rel=.05)
    res = export_project(circle, str(tmp_path), lambda f, m: True)
    assert not res["kapali_olmayan_govde"] and len(res["plates"]) == e["tabla"]


def test_big_circle_panels():
    r, m = make(CAT, 520.0, {"tur": "daire"}, parca=200, tek_taban=E.main_hex("Siyah"))
    L = build_stones(m)
    assert len(L.panels) == panel_count(m) == 9
    assert sum(pn.region.area for pn in L.panels) == pytest.approx(m.outline.area, rel=.01)
    r.mode = Mode.AMS
    m = E.build_mosaic_for(r)[0]
    L = build_stones(m)
    for pn in L.panels:                                         # no stone stands beside its panel's base
        reg = pn.region.buffer(.3)
        assert all(reg.covers(L.stones[i].poly) for i in pn.stones)


def test_recipe_remembers_the_frame():
    r = E.Recipe(image_path=CAT, width_mm=100.0)
    r.rays_center, r.zone_mask = [.5, .5], np.zeros((4, 4), np.uint8)
    k = r.build_key()
    assert r.set_frame({"kutu": [.1, .1, .9, .9]}, {"tur": "daire"}) and not r.set_frame({"kutu": [.1, .1, .9, .9]}, {"tur": "daire"})
    assert r.build_key() != k and r.rays_center is None and r.zone_mask is None
    q = E.Recipe.from_payload(r.to_payload())
    assert q.frame == r.frame and q.settings().shape == {"tur": "daire"}


# ------------------------------------------------------------------ the window
@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or app2.make_app(["test"])


@pytest.fixture()
def bridge(qapp, tmp_path):
    b = app2.Bridge(None, QSettings(str(tmp_path / "t.ini"), QSettings.Format.IniFormat))
    yield b
    b.shutdown()


def spin(qapp, cond, timeout=90.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


class Watch:
    def __init__(self, qapp, p):
        self.qapp, self.states, self.res = qapp, [], []
        p.state.connect(lambda j: self.states.append(json.loads(j)))
        p.result.connect(lambda j: self.res.append(json.loads(j)))

    def next(self, p, patch=None):
        n = len(self.res)
        if patch is not None:
            p.set(json.dumps(patch))
        assert spin(self.qapp, lambda: any("url" in r for r in self.res[n:]))
        return next(r for r in self.res[n:][::-1] if "url" in r)


def test_square_in_the_middle():
    assert app2_prep.square([0, 0, 1, 1], 100, 200) == pytest.approx([0, .25, 1, .75])
    assert app2_prep.square([.2, .2, .4, .4], 100, 100) == pytest.approx([.2, .2, .4, .4])


def test_preparing_a_picture(qapp, bridge, tmp_path):
    p, built = bridge.prep, []
    bridge.built.connect(lambda j: built.append(json.loads(j)))
    w = Watch(qapp, p)
    bridge.setWidth(12)
    assert p.open(POSTER)
    s = w.states[-1]
    assert s["open"] and s["yeni"] and s["kutu"] == [0, 0, 1, 1] and bridge.blobs.get(s["src"].split("/data/")[1])
    assert not bridge.recipe.image_path                          # nothing is in the project yet
    w.next(p)
    assert len(p.recipe.colours) >= 3
    r = w.next(p, {"sekil": {"tur": "daire"}})
    assert r["size_mm"] == [120.0, 120.0]
    r = w.next(p, {"parca": 60})
    assert abs(r["parca"]["sayi"] - 60) <= 2
    n = len(w.states)
    p.set(json.dumps({"op": "don"}))
    p.set(json.dumps({"op": "don"}))
    p.set(json.dumps({"op": "don"}))
    p.set(json.dumps({"op": "don"}))
    assert len(w.states) == n + 4 and p.recipe.kadraj["don"] == 0    # all the way round
    p.accept()
    assert w.states[-1] == {"open": False} and bridge.recipe.image_path == POSTER
    assert bridge.recipe.sekil == {"tur": "daire"} and not bridge.project_path
    assert spin(qapp, lambda: built) and abs(built[-1]["parca"]["sayi"] - 60) <= 2
    # the frame is part of the project
    f = bridge.save(str(tmp_path / "a.tessera"))
    frame = bridge.recipe.frame
    assert p.edit() and not w.states[-1]["yeni"]
    p.set(json.dumps({"sekil": {"tur": "dikdortgen"}, "kutu": [0, 0, 1, 1]}))
    p.cancel()
    assert bridge.recipe.frame == frame                          # given up: as it was
    assert bridge.open(f) and bridge.recipe.frame == frame
    assert p.edit()
    p.accept()                                                   # nothing changed: nothing is built again
    assert not bridge.dirty


def test_turning_and_mirroring_what_is_seen(qapp, bridge):
    p = bridge.prep
    assert p.open(CAT)
    p.set(json.dumps({"kutu": [.1, .2, .5, .6]}))
    p.set(json.dumps({"op": "ayna"}))
    assert p.recipe.kadraj["kutu"] == pytest.approx([.5, .2, .9, .6])
    p.set(json.dumps({"op": "don"}))
    k = p.recipe.kadraj
    assert k["kutu"] == pytest.approx([.4, .5, .8, .9])
    src = np.asarray(Image.open(CAT).convert("RGB").resize((200, 200)))
    seen = np.asarray(K.turned(Image.fromarray(src), k))
    assert (seen == np.rot90(src[:, ::-1], -1)).all()             # mirrored first, then turned clockwise, as it was clicked
    p.cancel()
    assert not bridge.recipe.image_path


def test_bad_picture_opens_no_window(qapp, bridge, tmp_path):
    bad = tmp_path / "x.png"
    bad.write_bytes(b"no picture")
    got = []
    bridge.failed.connect(got.append)
    assert not bridge.prep.open(str(bad)) and got and bridge.prep.recipe is None
    assert not bridge.prep.edit()


def test_the_page_has_the_window():
    html = open(os.path.join(app2.WEB_DIR, "index.html"), encoding="utf-8").read()
    for s in ("Görseli hazırla", "Kadraj", "Şekil", "Daire", "Kaç parça?", 'id="p-svg"'):
        assert s in html
