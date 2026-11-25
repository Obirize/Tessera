"""Bugs found in the review of the colour search and the simplified path; each one stays fixed."""
import json
import os
import threading
import time

import numpy as np
import pytest
from PIL import Image, ImageFilter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings  # noqa: E402

from tessera import app2  # noqa: E402
from tessera import app2_engine as E  # noqa: E402
from tessera.model import Filament  # noqa: E402
from tessera.pipeline import renk, sade  # noqa: E402
from tessera.pipeline.build import screen_free  # noqa: E402
from tessera.project import Project  # noqa: E402

BLACK, WHITE = ("Siyah", "#1B1B1D"), ("Beyaz", "#F3F1EC")


def col(rgb, paint, oz=0.0, **kw):
    return {"rgb": list(rgb), "pay": 0.3, "oz": [[oz, 0.0, 0.0, 0.0]], "ad": paint[0], "hex": paint[1], **kw}


def dots(period, cover, w=1600, h=1200):
    yy, xx = np.mgrid[0:h, 0:w]
    c = np.sqrt(.5)
    u, v = (xx * c - yy * c) / period, (xx * c + yy * c) / period
    d = np.hypot(u - np.floor(u) - .5, v - np.floor(v) - .5)
    a = np.full((h, w), 245, np.uint8)
    k = int(np.sqrt(cover * w * h))
    a[:k, :k] = np.where(d[:k, :k] < .33, 20, 245)
    a[700:1000, 900:1500] = 20
    return np.stack([np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(.7)))] * 3, -1)


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or app2.make_app(["test"])


@pytest.fixture()
def bridge(qapp, tmp_path):
    b = app2.Bridge(None, QSettings(str(tmp_path / "t.ini"), QSettings.Format.IniFormat))
    yield b
    b.shutdown()


@pytest.fixture()
def idle(bridge, monkeypatch):
    """A bridge whose engine does nothing; what it was asked for is recorded."""
    asked = []
    monkeypatch.setattr(bridge.engine, "build", lambda gen, r: asked.append(("build", gen)))
    monkeypatch.setattr(bridge.engine, "detect", lambda gen, r: asked.append(("detect", gen)))
    return bridge, asked


def picture(tmp_path, name="a.png"):
    p = str(tmp_path / name)
    Image.fromarray(np.full((60, 80, 3), (200, 40, 40), np.uint8)).save(p)
    return p


# ------------------------------------------------------------------ the queue
def engine(got):
    cb = {k: (lambda *a, k=k: got.append((k, a[0]))) for k in ("built", "scene", "thumb", "palette", "error")}
    return E.Engine(cb["built"], cb["scene"], cb["thumb"], cb["palette"], cb["error"], lambda *a: None)


def test_a_waiting_colour_search_does_not_swallow_a_newer_build(tmp_path, monkeypatch):
    got, gate = [], threading.Event()
    en = engine(got)
    monkeypatch.setattr(en, "_do_wait", lambda gen: gate.wait(), raising=False)
    monkeypatch.setattr(en, "_do_detect", lambda gen, r: got.append(("palette", gen)))
    monkeypatch.setattr(en, "_do_build", lambda gen, r: (en._check(), got.append(("built", gen))))
    try:
        en._push(("wait", 0))
        time.sleep(0.2)
        en.detect(1, E.Recipe())
        en.build(2, E.Recipe())
        gate.set()
        t = time.time()
        while ("built", 2) not in got and time.time() - t < 10:
            time.sleep(0.02)
        assert got == [("palette", 1), ("built", 2)]
        gate.clear()                                  # a project opened: its colours are there already
        en._push(("wait", 2))
        time.sleep(0.2)
        en.detect(3, E.Recipe())
        en.cancel_detect()
        en.build(4, E.Recipe())
        gate.set()
        t = time.time()
        while ("built", 4) not in got and time.time() - t < 10:
            time.sleep(0.02)
        assert got[2:] == [("built", 4)]
    finally:
        gate.set()
        en.stop()


# ------------------------------------------------------------------ a search that fails
def test_failed_search_leaves_a_way_out(idle, tmp_path):
    b, asked = idle
    states, failed = [], []
    b.state.connect(lambda j: states.append(json.loads(j)))
    b.failed.connect(failed.append)
    b.openImage(picture(tmp_path))
    assert states[-1]["finding"]
    b._ui_error(asked[-1][1], "detect", "olmadı")
    assert failed and not states[-1]["finding"] and states[-1]["colours"] == []
    n = len(asked)
    b.resetColours()                                  # "Baştan bul"
    assert asked[n][0] == "detect"
    b._ui_palette(asked[n][1], [col((0, 0, 0), BLACK), col((255, 255, 255), WHITE, 55)], {})
    n = len(asked)
    b.setColourCount(3)
    b._ui_error(asked[n][1], "detect", "olmadı")      # with colours in hand it goes on with them
    assert asked[-1][0] == "build" and not states[-1]["finding"]
    js = open(os.path.join(app2.WEB_DIR, "app.js"), encoding="utf-8").read()
    assert "Renkler bulunamadı" in js and "!cs.length ||" in js


def test_error_of_an_older_picture_is_ignored(idle, tmp_path):
    b, asked = idle
    b.openImage(picture(tmp_path, "a.png"))
    old = asked[-1][1]
    b.openImage(picture(tmp_path, "b.png"))
    b._ui_error(old, "detect", "a bozuk")
    b._ui_palette(asked[-1][1], [col((0, 0, 0), BLACK), col((255, 255, 255), WHITE, 55)], {})
    assert b.recipe.palette == ["Siyah", "Beyaz"] and asked[-1][0] == "build"


# ------------------------------------------------------------------ simplifying
def test_single_shape_picture_can_be_simplified(tmp_path):
    p = picture(tmp_path)
    c = E.settle(E.analyse(p, 200, tur="ton")[0], [])
    assert len(E.unique_names(c)) >= 2 and len({len(x["oz"][0]) for x in c}) == 1 and sade.tonal(c)
    r = E.Recipe(image_path=p, width_mm=200)
    r.use_colours(c)
    assert E.build_mosaic_for(r, True)[0].n_tiles > 0


def test_blurred_photograph_is_not_flat_art():
    yy, xx = np.mgrid[0:500, 0:500] / 500
    g = [np.stack([120 + 90 * np.sin(3 * xx + 2 * yy), 110 + 80 * np.cos(2 * xx - yy), 100 + 70 * np.sin(4 * yy)], -1),
         np.stack([250 - 120 * yy, 180 - 150 * yy, 90 + 60 * yy], -1)]
    for a in g:
        im = np.clip(a, 0, 255).astype(np.uint8)
        assert sade.kind(im, lambda: renk.find(im))[0] == "ton"


# ------------------------------------------------------------------ colours
def test_base_plate_keeps_its_colour():
    r = E.Recipe(image_path="x.png")
    c = [col((0, 0, 0), BLACK), col((9, 9, 9), BLACK, 9), col((255, 255, 255), WHITE, 55)]
    r.use_colours(c)
    r.fine["taban"] = BLACK[1]
    assert r.settings().base_colour == 0
    c = [dict(x) for x in c]
    c[0].update(ad=WHITE[0], hex=WHITE[1])            # same number of paints, other order
    r.use_colours(c)
    assert r.palette == ["Beyaz", "Siyah"] and r.settings().base_colour == 1
    c = [dict(x) for x in c]
    c[1].update(ad="Mavi", hex="#1F62B5")             # that paint is gone
    r.use_colours(c)
    assert r.fine["taban"] is None and r.settings().base_colour is None


def test_base_plate_follows_when_it_is_repainted(idle, tmp_path):
    b, asked = idle
    b.openImage(picture(tmp_path))
    b._ui_palette(asked[-1][1], [col((0, 0, 0), BLACK), col((255, 255, 255), WHITE, 55)], {})
    b.setFine(json.dumps({"taban": BLACK[1]}))
    shape = b.recipe.build_key(True)
    b.setColour(0, "Mavi")
    assert b.recipe.fine["taban"] == "#1F62B5" and b.recipe.build_key(True) == shape     # and no stone is cut anew


def test_hand_chosen_paint_goes_to_one_colour():
    before = [col((35, 35, 34), ("Kırmızı", "#C5221F"), elle=True), col((254, 254, 254), WHITE, 55)]
    found = [col((254, 254, 254), WHITE, 55), col((46, 46, 45), ("Koyu gri", "#4A4B4D"), 9), col((29, 29, 28), BLACK)]
    got = E.settle(found, before)
    assert [c["ad"] for c in got] == ["Beyaz", "Koyu gri", "Kırmızı"]


def test_dotted_print_setting_searches_the_colours_again(idle, tmp_path):
    b, asked = idle
    b.openImage(picture(tmp_path))
    b._ui_palette(asked[-1][1], [col((0, 0, 0), BLACK), col((255, 255, 255), WHITE, 55)], {})
    b.setFine(json.dumps({"tas": 1.2}))
    assert asked[-1][0] == "build"
    b.setFine(json.dumps({"tram": False}))
    assert asked[-1][0] == "detect" and b.dirty
    b.resetFine()
    assert asked[-1][0] == "detect"                   # back to "auto"


def test_older_project_keeps_its_paints(idle, tmp_path):
    b, asked = idle
    p = Project()
    p.image_path = picture(tmp_path)
    p.settings.palette = [Filament("Siyah", (27, 27, 29)), Filament("Kırmızı", (200, 30, 30)), Filament("Fildişi", (240, 235, 220))]
    f = str(tmp_path / "eski.tessera.json")
    p.save(f)
    b.open(f)
    assert asked[-1][0] == "detect" and b.recipe.fine["renk_sayisi"] == 3
    b._ui_palette(asked[-1][1], [col((250, 250, 250), WHITE, 55), col((180, 40, 40), ("Kiremit", "#A34A32"), 20),
                                 col((10, 10, 10), ("Koyu gri", "#4A4B4D"))], {})
    assert [c["hex"] for c in b.recipe.colours] == ["#F0EBDC", "#C81E1E", "#1B1B1D"] and all(c["elle"] for c in b.recipe.colours)
    assert not b.dirty                                # only opened, nothing changed


# ------------------------------------------------------------------ dotted prints
@pytest.mark.parametrize("period", [14, 18, 22])
def test_small_areas_of_coarse_dots_are_smoothed(period):
    a = dots(period, .08)
    out = screen_free(a, "auto", 420)
    assert out[40:300, 40:300, 0].std() < 0.5 * a[40:300, 40:300, 0].std()
    assert np.array_equal(out[720:980, 920:1480], a[720:980, 920:1480])
