"""Taking the background away: what stays becomes the mosaic's outline."""
import json
import os

import numpy as np
import pytest
import shapely
from PIL import Image, ImageDraw

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tessera import app2_engine as E  # noqa: E402
from tessera.pipeline import kadraj as K  # noqa: E402
from tessera.pipeline import zemin as Z  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAT = os.path.join(ROOT, "ornekler", "sablonlar", "Tessera kedisi.png")
CREAM, RED, BLACK, BLUE = (238, 228, 205), (200, 30, 35), (25, 25, 28), (30, 90, 180)


@pytest.fixture(scope="module")
def poster(tmp_path_factory):
    """A flat poster: a red disc and a black bar on cream; a blue square stands apart."""
    im = Image.new("RGB", (800, 600), CREAM)
    d = ImageDraw.Draw(im)
    d.ellipse((150, 100, 450, 400), fill=RED)
    d.rectangle((300, 380, 650, 460), fill=BLACK)
    d.rectangle((600, 80, 700, 180), fill=BLUE)
    p = tmp_path_factory.mktemp("zemin") / "afis.png"
    im.save(p)
    return str(p)


def test_wishes_are_cleaned():
    assert Z.clean(None) is None and Z.clean("x") is None
    assert Z.clean({}) == {"tiklar": []}
    z = Z.clean({"tiklar": [[0.123456, 2, "renk"], [-1, 0.5], ["a", 1], [0.5]] + [[0.1, 0.1]] * 300})
    assert z["tiklar"][:2] == [[0.1235, 1.0, "renk"], [0.0, 0.5, "tek"]]
    assert len(z["tiklar"]) <= Z.MAX_CLICKS
    # the background is part of the outline wish; a plain rectangle stays "no outline"
    assert K.clean_shape({"tur": "dikdortgen"}) is None
    assert K.clean_shape({"zemin": {}}) == {"tur": "dikdortgen", "zemin": {"tiklar": []}}
    assert K.clean_shape({"tur": "daire", "zemin": {"tiklar": [[.5, .5]]}}) == {"tur": "daire", "zemin": {"tiklar": [[.5, .5, "tek"]]}}


def test_flat_background_is_found(poster):
    keep, rep = Z.keep(poster, None, {})
    assert rep["puan"] == "iyi" and 0.6 < rep["pay"] < 0.8
    assert "2 ayrı parça" in " ".join(rep["soz"])            # the blue square stands apart
    h, w = keep.shape
    assert keep[h // 2, int(w * 300 / 800)] and keep[int(h * 130 / 600), int(w * 650 / 800)]   # disc, square
    assert not keep[5, 5] and not keep[h - 5, w - 5]


def test_clicks_turn_areas(poster):
    keep, _ = Z.keep(poster, None, {"tiklar": [[650 / 800, 130 / 600, "tek"]]})       # the square goes
    h, w = keep.shape
    assert not keep[int(h * 130 / 600), int(w * 650 / 800)] and keep[h // 2, int(w * 300 / 800)]
    keep, rep = Z.keep(poster, None, {"tiklar": [[0.02, 0.02, "tek"]]})                # the background comes back
    assert keep.all() and rep["pay"] == 0
    keep, _ = Z.keep(poster, None, {"tiklar": [[0.02, 0.02, "tek"], [300 / 800, 0.5, "renk"]]})   # every red goes
    assert not keep[h // 2, int(w * 300 / 800)] and keep[int(h * 420 / 600), int(w * 500 / 800)]


def test_nothing_is_taken_when_unsure(tmp_path):
    """Noise all over: no background can be told; rather nothing than a wrong cut."""
    a = np.random.default_rng(1).integers(0, 255, (300, 400, 3), np.uint8)
    p = tmp_path / "gurultu.png"
    Image.fromarray(a).save(p)
    keep, rep = Z.keep(str(p), None, {})
    assert keep.all() and rep["puan"] == "zor" and "çizgi çek" in " ".join(rep["soz"])    # noise reads as a photo


def test_silhouette_is_the_kept_area(poster):
    keep, _ = Z.keep(poster, None, {})
    P = Z.silhouette(keep, 400.0, 300.0, 0.5)
    disc = shapely.Point(150, 125).buffer(75)
    assert P.covers(disc.buffer(-1.5)) and not P.contains(shapely.Point(10, 10))
    assert P.area == pytest.approx(disc.area + 175 * 40 + 50 * 50 - disc.intersection(shapely.box(150, 190, 325, 230)).area, rel=0.06)
    assert Z.silhouette(np.ones((10, 10), bool), 100, 100, 0.5) is None


def test_background_has_no_say_in_the_colours(poster):
    sekil = K.clean_shape({"zemin": {}})
    a = K.picture(poster, None, sekil)
    assert not (np.abs(a.astype(int) - CREAM).sum(-1) < 30).any()          # not even its blended rim
    found, _ = E.analyse(poster, 400, frame=[None, sekil])
    assert len(found) == 3 and all(np.abs(np.array(c["rgb"]) - CREAM).sum() > 40 for c in found)
    assert sum(c["pay"] for c in found) == pytest.approx(1.0, abs=0.02)   # shares of the stones only
    assert max(c["pay"] for c in found) > 0.55                              # the red disc is the most


def test_stones_follow_the_silhouette(poster):
    r = E.Recipe(image_path=poster, width_mm=200)
    r.set_frame(None, {"zemin": {}})
    r.use_colours(E.analyse(poster, 200, frame=r.frame)[0])
    m = E.build_mosaic_for(r)[0]
    assert m.outline is not None and m.outline.geom_type == "MultiPolygon"
    edge = m.outline.buffer(0.05)
    assert all(edge.covers(p) for p in m.polys)
    assert sum(p.area for p in m.polys) > 0.85 * m.outline.area
    # with a base plate, the plate is the silhouette too
    r.fine["tek_taban"] = E.unique_names(r.colours)[0][1]
    m = E.build_mosaic_for(r)[0]
    assert m.plate is not None and m.plate.equals(m.outline)


def test_cat_background_by_one_click():
    """The cat's red ground is everywhere around it: one shift-click takes all of it, and the black lines
    that only parted red areas go with it."""
    keep, rep = Z.keep(CAT, None, {"tiklar": [[0.3, 0.12, "renk"]]})
    img = Z.frame_picture(CAT, None)
    red = (img[..., 0] > 150) & (img[..., 1] < 80)
    assert not (keep & red).any() and 0.35 < rep["pay"] < 0.45
    from scipy import ndimage as ndi
    assert ndi.label(keep)[1] == 1                       # one cat, no stray line


def test_the_window_takes_the_background_away(poster, tmp_path):
    from PySide6.QtCore import QSettings
    from tessera import app2
    from test_kadraj import Watch, spin
    from PySide6.QtWidgets import QApplication
    qapp = QApplication.instance() or app2.make_app(["test"])
    b = app2.Bridge(None, QSettings(str(tmp_path / "t.ini"), QSettings.Format.IniFormat))
    try:
        b.setWidth(10)                            # small: quick drafts
        p = b.prep
        w = Watch(qapp, p)
        assert p.open(poster)
        w.next(p)
        n = len(w.res)
        r = w.next(p, {"zemin": True})
        z = next(x["zemin"] for x in w.res[n:] if "zemin" in x)
        assert z["puan"] == "iyi" and b.blobs.get(z["url"].split("/data/")[1]) and z["kutu"] is None
        assert w.states[-1]["zemin"] == {"tiklar": []}
        n = len(w.res)
        r = w.next(p, {"tik": [650 / 800, 130 / 600, "tek"]})                    # the square goes
        assert p.recipe.sekil["zemin"]["tiklar"] == [[0.8125, 0.2167, "tek"]]
        assert len(K.polygons(b.engine._draft["m"].outline)) == 1
        w.next(p, {"geri": True})
        assert p.recipe.sekil["zemin"]["tiklar"] == []
        p.set(json.dumps({"tik": [0.5, 0.5, "tek"]}))
        w.next(p, {"sekil": {"tur": "daire"}})                                   # another part: the clicks go
        assert p.recipe.sekil == {"tur": "daire", "zemin": {"tiklar": []}}
        p.set(json.dumps({"tik": [0.5, 0.5, "tek"]}))
        p.set(json.dumps({"kutu": p.recipe.kadraj["kutu"]}))                      # the same box: they stay
        assert len(p.recipe.sekil["zemin"]["tiklar"]) == 1
        w.next(p, {"zemin": False, "sekil": {"tur": "dikdortgen"}})
        assert p.recipe.sekil is None
        w.next(p, {"zemin": True})
        p.accept()
        assert b.recipe.sekil == {"tur": "dikdortgen", "zemin": {"tiklar": []}}
        assert spin(qapp, lambda: b.mosaic is not None and b.mosaic.outline is not None)
        f = b.save(str(tmp_path / "a.tessera"))
        b.recipe.sekil = None
        b.open(f)
        assert b.recipe.sekil == {"tur": "dikdortgen", "zemin": {"tiklar": []}}
    finally:
        b.shutdown()


def test_silhouette_goes_through_everything(poster, tmp_path):
    """Two loose parts on one base plate: stones, plates, paper template, guide and 3MF files."""
    from tessera.export.api import estimate, export_project, plates_glb, scene_glb
    r = E.Recipe(image_path=poster, width_mm=150)
    r.set_frame(None, {"zemin": {}})
    r.use_colours(E.analyse(poster, 150, frame=r.frame)[0])
    r.fine["tek_taban"] = next(h for n, h in E.unique_names(r.colours) if n == "Siyah")
    m = E.build_mosaic_for(r, False)[0]
    assert len(K.polygons(m.plate)) == 2
    scene_glb(m)
    _, pm = plates_glb(m)
    e = estimate(m)
    res = export_project(m, str(tmp_path), lambda f, msg: True)
    assert not res["kapali_olmayan_govde"] and len(res["plates"]) == e["tabla"] == len(pm["plates"])
    assert os.path.exists(tmp_path / "02_tablalar" / "0_Taban_Siyah.3mf")
    assert any(n.endswith(".png") or n.endswith(".pdf") for n in os.listdir(tmp_path))    # guide, template


# ------------------------------------------------------------------ photographs (experimental)
@pytest.fixture(scope="module")
def photo(tmp_path_factory):
    """Photo-like: a shaded, textured figure on a soft, blotchy ground; no flat colour anywhere."""
    from scipy import ndimage as ndi
    rng = np.random.default_rng(5)
    h, w = 600, 500
    bg = ndi.gaussian_filter(rng.normal(0, 1, (h, w, 3)), (25, 25, 0))
    img = 60 + 90 * (bg - bg.min()) / (bg.max() - bg.min()) * np.array([0.6, 0.9, 0.7])
    yy, xx = np.mgrid[:h, :w]
    fig = ((xx - 250) / 140) ** 2 + ((yy - 330) / 210) ** 2 < 1
    tex = ndi.gaussian_filter(rng.normal(0, 1, (h, w)), 3) * 25
    shade = (1 - ((xx - 230) / 200) ** 2 - ((yy - 280) / 260) ** 2).clip(0, 1)
    f = np.stack([200 + 40 * shade + tex, 120 + 50 * shade + tex, 60 + 20 * shade + tex], -1)
    img = np.where(fig[..., None], f, img)
    img = (ndi.gaussian_filter(img, (1.2, 1.2, 0)) + rng.normal(0, 4, img.shape)).clip(0, 255).astype(np.uint8)
    p = tmp_path_factory.mktemp("foto") / "foto.png"
    Image.fromarray(img).save(p)
    return str(p)


def at(keep, x, y):
    h, w = keep.shape
    return bool(keep[int(y * h), int(x * w)])


def test_lines_are_cleaned():
    z = Z.clean({"cizgiler": [["kalsin", [[0.1, 0.2], [2, -1]]], ["x", [[0.5, 0.5]]], ["kalsin", []], ["gitsin", "?"]]})
    assert z == {"tiklar": [], "cizgiler": [["kalsin", [[0.1, 0.2], [1.0, 0.0]]], ["gitsin", [[0.5, 0.5]]]]}
    assert Z.clean({"cizgiler": []}) == {"tiklar": []}        # nothing drawn: the same wish as before
    m = Z.drawn(z["cizgiler"], "kalsin", (200, 100))
    assert m.shape == (100, 200) and m[20, 20] and not m[90, 20]
    assert Z.drawn(z["cizgiler"], "gitsin", (200, 100))[50, 100]


def test_photo_background(photo):
    keep, rep = Z.keep(photo, None, {})
    assert rep["yol"] == "foto" and rep["puan"] in ("iyi", "orta") and "deneme" in rep["soz"][-1]
    assert 0.6 < rep["pay"] < 0.75
    assert at(keep, 0.5, 0.55) and at(keep, 0.5, 0.25) and not at(keep, 0.05, 0.05) and not at(keep, 0.95, 0.5)
    # the edge lies on the figure's own edge, not a few pixels off
    from scipy import ndimage as ndi
    yy, xx = np.mgrid[:600, :500]
    fig = ((xx - 250) / 140) ** 2 + ((yy - 330) / 210) ** 2 < 1
    d = ndi.distance_transform_edt(~(fig & ~ndi.binary_erosion(fig)))
    cut = keep & ~ndi.binary_erosion(keep)
    assert d[cut].mean() < 2.2 and np.percentile(d[cut], 98) < 5


def test_photo_lines_lean():
    """What lies nearer the green lines (walking over the picture, edges count as a long way) leans to stay;
    only near the lines, far away the colours decide alone."""
    X = np.zeros((40, 60, 3), np.float32)
    X[:, 30:] = 50                                   # two flat halves, an edge between them
    g, r = np.zeros((40, 60), bool), np.zeros((40, 60), bool)
    g[20, 5], r[20, 55] = True, True
    lean = Z._lean(X, g, r)
    assert (lean[18:23, 2:9] > 0.3).all() and (lean[18:23, 51:58] < -0.3).all()
    assert (lean[:, 15:45] == 0).all() and lean[0, 5] == 0
    assert Z._lean(X, g, np.zeros_like(r)) is None


def test_photo_far_line_has_no_say():
    """Few lines: a red line far off on one side, a green one at the bottom. The upper half of the figure,
    pale and near the ground's colour, is far from both: it stays as it is without lines, not taken away."""
    from scipy import ndimage as ndi
    rng = np.random.default_rng(2)
    h, w = 400, 400
    g = ndi.gaussian_filter(rng.normal(0, 1, (h, w)), 8)
    img = np.stack([150 + 30 * g, 140 + 30 * g, 120 + 30 * g], -1)                  # stone-like ground
    yy, xx = np.mgrid[:h, :w]
    fig = ((xx - 230) / 120) ** 2 + ((yy - 240) / 150) ** 2 < 1
    tex = ndi.gaussian_filter(rng.normal(0, 1, (h, w)), 2) * 40
    low = np.stack([210 + tex, 120 + tex, 50 + tex], -1)                              # orange below
    high = np.stack([165 + tex, 149 + tex, 120 + tex], -1)                            # pale above
    img = np.where(fig[..., None], np.where((yy > 260)[..., None], low, high), img)
    img = (ndi.gaussian_filter(img, (1, 1, 0)) + rng.normal(0, 3, img.shape)).clip(0, 255).astype(np.uint8)
    lines = [["gitsin", [[0.03, 0.2], [0.05, 0.8]]], ["kalsin", [[0.45, 0.9], [0.7, 0.9]]]]
    auto, _ = Z.photo(img)
    keep, _ = Z.photo(img, Z.drawn(lines, "kalsin", (w, h)), Z.drawn(lines, "gitsin", (w, h)))
    top = fig & (yy < 200)
    assert (auto & top).mean() / top.mean() > 0.95 and (keep & top).mean() / top.mean() > 0.95
    assert (auto & ~keep).sum() < 0.02 * auto.sum()


def test_photo_red_line_opens_a_hole(photo):
    """A red line inside the figure: that place goes, even though the figure is all around it."""
    keep, _ = Z.keep(photo, None, {"cizgiler": [["gitsin", [[0.45, 0.55], [0.55, 0.55]]]]})
    assert not at(keep, 0.5, 0.55) and at(keep, 0.5, 0.3) and at(keep, 0.5, 0.8) and not at(keep, 0.05, 0.05)


def test_photo_lines_steer_the_cut(photo):
    z = {"cizgiler": [["kalsin", [[0.05, 0.1], [0.2, 0.1]]],          # a bit of the ground stays
                      ["gitsin", [[0.3, 0.9], [0.7, 0.9]]]]}          # the lower part of the figure goes
    keep, rep = Z.keep(photo, None, z)
    assert rep["yol"] == "foto" and "çizgilerine" in rep["soz"][0]
    assert at(keep, 0.12, 0.1) and not at(keep, 0.5, 0.9) and at(keep, 0.5, 0.5)
    # clicks mean nothing on a photograph
    assert (Z.keep(photo, None, {"tiklar": [[0.5, 0.5, "tek"]]})[0] == Z.keep(photo, None, {})[0]).all()


def test_photo_stays_a_photo(photo):
    """With its ground filled flat a photo looks drawn; it is still simplified as a photo, and only what
    stays chooses the colours."""
    sekil = K.clean_shape({"zemin": {"cizgiler": [["gitsin", [[0.3, 0.9], [0.7, 0.9]]]]}})
    found, adv = E.analyse(photo, 200, frame=[None, sekil])
    assert adv["tur"] == "ton" and all(c.get("yol") == "ton" for c in found)
    assert all(np.array(c["rgb"])[0] > np.array(c["rgb"])[1] for c in found[:2])   # the figure's warm colours lead


def test_the_window_takes_lines(photo, tmp_path):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from tessera import app2
    from test_kadraj import Watch
    qapp = QApplication.instance() or app2.make_app(["test"])
    b = app2.Bridge(None, QSettings(str(tmp_path / "t.ini"), QSettings.Format.IniFormat))
    try:
        b.setWidth(10)
        p = b.prep
        w = Watch(qapp, p)
        assert p.open(photo)
        w.next(p)
        n = len(w.res)
        w.next(p, {"zemin": True})
        assert next(x["zemin"] for x in w.res[n:] if "zemin" in x)["yol"] == "foto"
        w.next(p, {"cizgi": ["gitsin", [[0.3, 0.9], [0.7, 0.9]]]})
        w.next(p, {"cizgi": ["kalsin", [[0.05, 0.1], [0.2, 0.1]]]})
        assert [t for t, _ in p.recipe.sekil["zemin"]["cizgiler"]] == ["gitsin", "kalsin"]
        w.next(p, {"geri": True})
        assert [t for t, _ in p.recipe.sekil["zemin"]["cizgiler"]] == ["gitsin"]
        w.next(p, {"kutu": [0, 0, 1, 0.8]})                                   # another part: the lines go
        assert p.recipe.sekil == {"tur": "dikdortgen", "zemin": {"tiklar": []}}
        p.cancel()
    finally:
        b.shutdown()
