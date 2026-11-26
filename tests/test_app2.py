"""Tessera v2 interface (tessera/app2.py + tessera/web/): bridge methods and an offscreen smoke test.

The page itself is HTML/three.js; here we check what Python promises the page (state JSON,
numbers, the GLB scene, previews, project files) and that the page boots in Qt WebEngine.
"""
from __future__ import annotations

import json
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings  # noqa: E402

from tessera import app2  # noqa: E402
from tessera import app2_engine as E  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSTER = os.path.join(ROOT, "ornekler", "poster", "kaynak.webp")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    inst = QApplication.instance()
    if inst is None:
        return app2.make_app(["test"])
    return inst


@pytest.fixture()
def store(tmp_path):
    s = QSettings(str(tmp_path / "t.ini"), QSettings.Format.IniFormat)
    return s


def spin(qapp, cond, timeout=90.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


class Rec:
    def __init__(self, bridge):
        self.got = {k: [] for k in ("state", "built", "scene", "thumb", "failed", "toast", "progress")}
        for k in self.got:
            getattr(bridge, k).connect(lambda *a, k=k: self.got[k].append(a[0] if len(a) == 1 else a))


# ------------------------------------------------------------------ pure logic
def test_style_cards_are_the_four_stone_styles():
    cards = E.style_cards()
    assert [c["name"] for c in cards] == ["Mozaik", "Kırık taş", "Blok", "İri parça"]
    assert [c["code"] for c in cards] == ["mozaik", "kirik", "blok", "iri"]
    assert E.DEFAULT_STYLE == "Mozaik"
    assert E.style_name("Afiş") == "Mozaik" and E.style_name("blok") == "Blok"   # old projects


def test_recipe_settings_r6():
    r = E.Recipe(image_path=POSTER, palette=["Siyah", "Kırmızı", "Beyaz"], style="Kırık taş")
    s = r.settings()
    assert [(f.name, f.hex) for f in s.palette] == [("Siyah", "#1B1B1D"), ("Kırmızı", "#C5221F"), ("Beyaz", "#F3F1EC")]
    assert s.style == "kirik"
    assert s.merge_same_colour is False and s.backing == "yok"        # every stone separate, no backing
    assert s.gap_mm == pytest.approx(0.4)
    r.fine.update(tas=1.2, panel=200.0, taban=2, derz=0.8, kalinlik=4.0, kabartma=1.0, oynama=0.4)
    s2 = r.settings()
    assert s2.stone_scale == 1.2 and s2.panel_max_mm == 200 and s2.base_colour == 2 and s2.gap_mm == pytest.approx(0.8)
    assert (s2.stone_h_mm, s2.relief_mm, s2.h_jitter_mm) == (4.0, 1.0, 0.4)
    assert r.build_key() != E.Recipe(image_path=POSTER, palette=r.palette, style="Kırık taş").build_key()


def test_scheme_serves_page_and_vendor_files():
    for f in ("index.html", "app.js", "stage.js", "style.css", "vendor/three/build/three.module.js",
              "vendor/three/examples/jsm/controls/OrbitControls.js",
              "vendor/three/examples/jsm/loaders/GLTFLoader.js",
              "vendor/three/examples/jsm/utils/BufferGeometryUtils.js"):
        assert os.path.isfile(os.path.join(app2.WEB_DIR, f)), f
    html = open(os.path.join(app2.WEB_DIR, "index.html"), encoding="utf-8").read()
    assert "importmap" in html and "cdn" not in html          # works offline
    for q in ("Hangi görsel?", "Hangi renkler?", "Nasıl bir doku?", "Nasıl basacaksın?", "Dışa aktar", "Gelişmiş ayarlar"):
        assert q in html


# ------------------------------------------------------------------ bridge
@pytest.fixture()
def bridge(qapp, store):
    b = app2.Bridge(None, store)
    yield b
    b.shutdown()


def test_bridge_empty_state(bridge):
    st = json.loads(bridge.getState())
    assert st["image"]["path"] == "" and st["palette"] == []
    assert st["colours"] == [] and len(st["main"]) == 28              # plain main colours, no spool list
    assert not {"spools", "library", "labels"} & set(st)
    assert not any(hasattr(bridge, n) for n in ("setOwned", "setPalette", "suggestPalette"))
    assert st["mode"] == "parca" and st["style"] == "Mozaik"
    assert "light" not in st and not hasattr(bridge, "setLight")      # no light controls any more
    assert bridge.getSceneGLB() == "" and bridge.getPreview() == ""


def test_no_spools_anywhere_on_screen():
    web = "".join(open(os.path.join(app2.WEB_DIR, f), encoding="utf-8").read() for f in ("index.html", "app.js"))
    for w in ("Makaralarım", "makara", "PLA", "spool", "Bu görsel için öner"):
        assert w not in web, w
    names = [c["name"] for c in E.main_colours()]
    assert len(set(names)) == len(names) and {"Siyah", "Beyaz", "Kırmızı", "Turkuaz"} <= set(names)


def test_advice_is_shown_and_the_kind_can_be_switched(qapp, bridge, tmp_path):
    import numpy as np
    from PIL import Image
    from tests.test_sade import photo
    p = str(tmp_path / "foto.png")
    Image.fromarray(photo()).save(p)
    rec = Rec(bridge)
    bridge.setWidth(20)
    bridge.openImage(p)
    assert spin(qapp, lambda: rec.got["built"], 120), rec.got["failed"]
    st = json.loads(bridge.getState())
    assert st["uygun"]["tur"] == "ton" and "sadeleştirildi" in " ".join(st["uygun"]["soz"])
    n = len(rec.got["built"])
    bridge.setKind("net")
    assert spin(qapp, lambda: len(rec.got["built"]) > n, 120)
    st = json.loads(bridge.getState())
    assert st["uygun"]["tur"] == "net" and st["fine"]["tur"] == "net"
    bridge.resetFine()
    assert json.loads(bridge.getState())["fine"]["tur"] == "net"      # fine settings' reset leaves it alone
    bridge.setKind("bilinmeyen")
    assert json.loads(bridge.getState())["fine"]["tur"] == "net"
    html = open(os.path.join(app2.WEB_DIR, "index.html"), encoding="utf-8").read()
    assert 'id="fit"' in html


def test_templates_are_offered_and_open(qapp, bridge):
    st = json.loads(bridge.getState())
    assert [t["name"] for t in st["templates"]] == ["Tessera kedisi", "Örnek afiş"] and "example" not in st
    for t in st["templates"]:
        assert bridge.blobs.get(t["url"].split("/data/")[1])                 # the little picture is really served
    cols, adv = E.analyse(st["templates"][0]["path"], 300)
    assert {c["ad"] for c in cols} == {"Siyah", "Beyaz", "Kırmızı", "Pembe", "Mavi", "Altın"} and adv["puan"] == "iyi"
    rec = Rec(bridge)
    bridge.openImage(st["templates"][0]["path"])
    assert bridge.display_name() == "Tessera kedisi" and not rec.got["failed"]
    html = open(os.path.join(app2.WEB_DIR, "index.html"), encoding="utf-8").read()
    assert "Hazır şablonlar" in html and "btn-example" not in html
    assert bridge.recipe.fine["parca"] == "sekil" and bridge.recipe.fine["tek_taban"] == E.main_hex("Siyah")
    assert bridge.recipe.width_mm == 200
    assert spin(qapp, lambda: rec.got["built"]) and json.loads(rec.got["built"][-1])["tiles"] == 36
    bridge.setWidth(30)
    bridge.openImage(st["templates"][1]["path"])                 # the next picture starts without them
    assert bridge.recipe.fine["parca"] is None and bridge.recipe.fine["tek_taban"] is None and bridge.recipe.width_mm == 300


def test_piece_count_and_plate_belong_to_the_project(qapp, bridge, tmp_path):
    rec = Rec(bridge)
    bridge.openImage(app2.TEMPLATES[0][1])
    assert spin(qapp, lambda: rec.got["built"])
    bridge.setFine(json.dumps({"parca": 50.0}))
    assert bridge.recipe.fine["parca"] == 50
    i = [c["ad"] for c in bridge.recipe.colours].index("Siyah")
    bridge.setColour(i, "Lacivert")
    assert bridge.recipe.fine["tek_taban"] == E.main_hex("Lacivert")        # the plate goes with its colour
    bridge.resetFine()
    assert bridge.recipe.fine["parca"] == 50 and bridge.recipe.fine["tek_taban"] == E.main_hex("Lacivert")
    p = bridge.save(str(tmp_path / "a.tessera"))
    bridge.setFine(json.dumps({"parca": None, "tek_taban": None}))
    assert bridge.open(p) and bridge.recipe.fine["parca"] == 50 and bridge.recipe.fine["tek_taban"] == E.main_hex("Lacivert")
    n = len(rec.got["built"])
    assert spin(qapp, lambda: len(rec.got["built"]) > n and json.loads(rec.got["built"][-1])["tiles"] == 50)


def test_one_preview_and_the_plates():
    html = open(os.path.join(app2.WEB_DIR, "index.html"), encoding="utf-8").read()
    import re
    nav = re.search(r'<nav class="views".*?</nav>', html, re.S).group(0)
    assert re.findall(r">([^<>]+)</button>", nav) == ["Önizleme", "Tablalar"]
    for f in ("app.js", "stage.js", "style.css"):
        src = open(os.path.join(app2.WEB_DIR, f), encoding="utf-8").read()
        assert "view-wall" not in src and "view-flat" not in src and "=== 'wall'" not in src and "'flat'" not in src, f


def test_same_paint_twice_is_one_colour():
    cols = [{"rgb": [0, 0, 0], "pay": .5, "oz": [[0, 0, 0, 0]], "ad": "Siyah", "hex": "#1B1B1D"},
            {"rgb": [9, 9, 9], "pay": .3, "oz": [[2, 0, 0, 0]], "ad": "Siyah", "hex": "#1b1b1d"},
            {"rgb": [255, 80, 0], "pay": .2, "oz": [[50, 0, 9, 9]], "ad": "Turuncu", "hex": "#FF5000"},
            {"rgb": [250, 90, 9], "pay": .2, "oz": [[51, 0, 9, 9]], "ad": "Turuncu", "hex": "#FA5A09"}]
    r = E.Recipe(image_path=POSTER)
    r.use_colours(cols)
    assert r.palette == ["Siyah", "Turuncu", "Turuncu 2"] and r.targets() == [0, 0, 1, 2]
    k, shape = r.build_key(), r.build_key(True)
    cols[2] = {**cols[2], "ad": "Mavi", "hex": "#1F62B5"}
    r.use_colours(cols)
    assert r.build_key() != k and r.build_key(True) == shape          # repainting moves no stone
    cols[2] = {**cols[2], "ad": "Siyah", "hex": "#1B1B1D"}
    r.use_colours(cols)
    assert r.build_key(True) != shape                                 # joining two colours does


def test_bridge_open_build_scene_and_controls(qapp, bridge):
    rec = Rec(bridge)
    bridge.setWidth(25)                              # small piece: quick builds (width is kept)
    assert json.loads(bridge.openImage(POSTER))["w"] > 0
    assert spin(qapp, lambda: rec.got["scene"], 240), rec.got["failed"]
    # the progress overlay: real stages 0 -> 1, the stone scene arrives at 100 %
    fr = [json.loads(x)["f"] for x in rec.got["progress"]]
    assert fr[0] == 0.0 and fr[-1] == 1.0 and len(fr) >= 5
    assert any(0.05 < f < E.BUILD_SHARE for f in fr) and any(E.BUILD_SHARE <= f < 1.0 for f in fr)
    st = json.loads(bridge.getState())
    assert st["style"] == "Mozaik"
    assert sorted(c["ad"] for c in st["colours"]) == ["Beyaz", "Kırmızı", "Siyah"] == sorted(st["palette"])
    assert sum(c["pay"] for c in st["colours"]) == pytest.approx(1, abs=0.01)
    info = json.loads(rec.got["built"][-1])
    assert info["size_mm"][0] == pytest.approx(250, abs=2)
    for mode in ("parca", "ams"):                    # both production cards get their own numbers
        assert info["estimates"][mode]["saat"] > 0
    assert info["estimates"]["parca"]["parca"] > 20
    sc = json.loads(bridge.getSceneGLB())
    blob = bridge.blobs.get(sc["url"].split("/data/")[1])
    assert blob and blob[1][:4] == b"glTF"
    assert sc["meta"]["counts"]["parts"] > 20 and sc["quality"] == "preview"
    names = [n["name"] for n in sc["meta"]["nodes"]]
    assert all(n.startswith("stone:") for n in names)             # every stone its own node, no lattice
    assert sc["meta"]["counts"]["parts"] == info["estimates"]["parca"]["parca"]
    assert spin(qapp, lambda: bridge.getPlates(), 60)
    pl = json.loads(bridge.getPlates())["meta"]["plates"]
    assert sum(len(p["codes"]) for p in pl) >= sc["meta"]["counts"]["parts"]
    assert bridge.getPreview().startswith("data:image/png;base64,")

    # the four questions each trigger a new build
    red = [c["ad"] for c in st["colours"]].index("Kırmızı")
    for call in (lambda: bridge.setWidth(30), lambda: bridge.setStyle("Blok"),
                 lambda: bridge.setMode("ams"), lambda: bridge.setColour(red, "Mavi")):
        n, k = len(rec.got["built"]), len(rec.got["progress"])
        call()
        assert spin(qapp, lambda: len(rec.got["built"]) > n, 60), rec.got["failed"]
        assert json.loads(rec.got["progress"][k])["f"] == 0.0      # every rebuild shows the overlay again
    st = json.loads(bridge.getState())
    assert st["width_cm"] == 30 and st["mode"] == "ams" and sorted(st["palette"]) == ["Beyaz", "Mavi", "Siyah"]
    assert st["colours"][red]["elle"] and st["colours"][red]["kaynak"] != st["colours"][red]["hex"]
    tiles = json.loads(rec.got["built"][-1])["tiles"]
    n = len(rec.got["built"])
    bridge.setColour(red, "#ff8800")                 # a free colour: same stones, painted anew
    assert spin(qapp, lambda: len(rec.got["built"]) > n, 60)
    assert json.loads(rec.got["built"][-1])["tiles"] == tiles
    assert {"name": "Turuncu", "hex": "#FF8800"} in json.loads(rec.got["built"][-1])["palette"]
    for i in range(3):
        if i != red:
            bridge.setColour(i, "Turuncu")           # joining is fine as long as two colours are left
    assert len(json.loads(bridge.getState())["palette"]) >= 2
    n = len(rec.got["built"])
    bridge.resetColours()
    assert spin(qapp, lambda: len(rec.got["built"]) > n, 90)
    assert sorted(json.loads(bridge.getState())["palette"]) == ["Beyaz", "Kırmızı", "Siyah"]
    assert json.loads(rec.got["built"][-1])["size_mm"][0] == pytest.approx(300, abs=2)
    bridge.setFine(json.dumps({"tas": 1.3, "bilinmeyen": 1}))
    assert json.loads(bridge.getState())["fine"]["tas"] == 1.3
    assert "bilinmeyen" not in json.loads(bridge.getState())["fine"]
    # style thumbnails arrive in the background
    assert spin(qapp, lambda: len(rec.got["thumb"]) >= 1, 60)
    assert json.loads(rec.got["thumb"][0])["url"].startswith("data:image/jpeg")


def test_bridge_save_and_open_roundtrip(qapp, bridge, store, tmp_path):
    rec = Rec(bridge)
    bridge.setWidth(25)
    bridge.openImage(POSTER)
    assert spin(qapp, lambda: rec.got["built"], 90)
    bridge.setStyle("İri parça")
    bridge.setWidth(50)
    bridge.setFine(json.dumps({"tas": 1.2}))
    path = bridge.save(str(tmp_path / "deneme"))
    assert path.endswith(".tessera.json") and os.path.exists(path)
    d = json.load(open(path, encoding="utf-8"))
    assert d["arayuz2"]["stil"] == "İri parça" and d["ayarlar"]["width_mm"] == 500

    b2 = app2.Bridge(None, store)
    try:
        assert b2.open(path) == os.path.abspath(path)
        st = json.loads(b2.getState())
        assert st["style"] == "İri parça" and st["width_cm"] == 50 and st["fine"]["tas"] == 1.2
        assert st["palette"] == json.loads(bridge.getState())["palette"] and st["palette"][:1]
        assert st["colours"] == json.loads(bridge.getState())["colours"]
        assert st["dirty"] is False
    finally:
        b2.shutdown()


def test_bridge_rejects_bad_input(qapp, bridge, tmp_path):
    rec = Rec(bridge)
    bad = tmp_path / "bozuk.png"
    bad.write_bytes(b"not an image")
    assert bridge.openImage(str(bad)) == ""
    assert rec.got["failed"] and "görsel" in json.loads(rec.got["failed"][-1])["msg"].lower()
    bridge.setMode("lazer")
    bridge.setStyle("Yok")
    st = json.loads(bridge.getState())
    assert st["mode"] == "parca" and st["style"] == "Mozaik"


def test_export_small_artwork(tmp_path):
    """The child-process export entry point, run in-process on a small piece."""
    r = E.Recipe(image_path=POSTER, palette=["Siyah", "Kırmızı", "Beyaz"], style="Mozaik", width_mm=120)
    z = E.sibling_zones(POSTER)
    r.shapes, r.zones_from_file, r.file_default_vocab = z["sekiller"], True, z["varsayilan"]
    r.rays_center = z["isin_merkezi"]
    seen = []
    res = E.run_export({"recipe": r.to_payload(), "out": str(tmp_path / "out")}, lambda f, m="": seen.append(f))
    assert res.get("parts", 0) > 0 and seen and max(seen) <= 1.0
    assert any(p.endswith(".3mf") for p in os.listdir(tmp_path / "out" / "02_tablalar")) \
        if (tmp_path / "out" / "02_tablalar").exists() else res.get("files")
    assert (tmp_path / "out" / "00_sablon_1-1_A4.pdf").exists()        # R6-2: the paper template


def test_export_card_leads_with_template(qapp, bridge, tmp_path):
    (tmp_path / "00_sablon_1-1_A4.pdf").write_bytes(b"%PDF-1.4")
    got = []
    bridge.exportDone.connect(got.append)
    bridge._export_done(str(tmp_path), {"parts": 10, "plates": [1, 2], "sablon_sayfa": [2, 3]})
    d = json.loads(got[-1])
    assert d["sablon_a4"].endswith("00_sablon_1-1_A4.pdf") and d["sablon_a3"] == "" and d["plates"] == 2
    html = open(os.path.join(app2.WEB_DIR, "index.html"), encoding="utf-8").read()
    assert "Montaj şablonu (1:1)" in html and "taşları kodlarına göre üstüne yapıştır" in html


# ------------------------------------------------------------------ the page in WebEngine
def test_splash_assets_and_switch():
    html = open(os.path.join(app2.WEB_DIR, "index.html"), encoding="utf-8").read()
    for f in ("acilis/tam.webp", "acilis/levha.webp", "acilis/taslar.json", "tessera_icon.png", "tessera_kedi.ico"):
        assert os.path.isfile(os.path.join(app2.WEB_DIR, f)), f
    assert 'id="splash"' not in html and 'id="f-acilis"' in html     # the card is its own window (acilis.py)
    js = open(os.path.join(app2.WEB_DIR, "app.js"), encoding="utf-8").read()
    assert "dial" not in html and "setLight" not in js           # no light control anywhere
    assert "Mozaiğe dönüştürülüyor" in html


def test_window_starts_maximized_and_remembers(qapp, store, monkeypatch):
    from PySide6.QtCore import Qt
    monkeypatch.setenv("TESSERA_NO_SPLASH", "1")
    win = app2.MainWindow(store)
    win.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    win.show_initial()
    assert win.windowState() & Qt.WindowState.WindowMaximized
    win.showNormal()
    win.close()
    assert str(store.value("window/maximized")).lower() == "false" and store.value("window/geometry") is not None
    win2 = app2.MainWindow(store)
    win2.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    win2.show_initial()
    try:
        assert not (win2.windowState() & Qt.WindowState.WindowMaximized)
    finally:
        win2.close()


def test_offscreen_smoke(qapp, store, monkeypatch):
    from PySide6.QtCore import Qt
    # the animated opening card upsets Chromium's GPU context under the offscreen platform only
    monkeypatch.setenv("TESSERA_NO_SPLASH", "1")
    win = app2.MainWindow(store)
    win.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    win.show()
    try:
        assert spin(qapp, lambda: win.loaded, 30), "page did not load"
        out = {}

        def probe():
            win.js("JSON.stringify({ready: !!(window.tessera && tessera.ready), canvas: !!document.querySelector"
                   "('#canvas-host canvas'), q: document.querySelectorAll('#column .q').length, btn: document.querySelector"
                   "('#btn-export').textContent})", lambda r: out.update(json.loads(r or "{}")))
        t0 = time.time()
        while time.time() - t0 < 30 and not out.get("ready"):
            probe()
            spin(qapp, lambda: False, 0.4)
        assert out.get("ready") and out.get("canvas"), out
        assert out["q"] == 4 and out["btn"] == "Dışa aktar"
    finally:
        win.close()


def test_start_does_not_load_the_mosaic_code():
    """scipy and shapely take seconds to load from a cold disk; the window must not wait for them."""
    import subprocess
    import sys
    code = ("import sys, tessera.app2 as a, tessera.app2_engine as E; E.main_colours(); E.main_hex('Siyah'); "
            "print(sorted(m for m in ('scipy', 'shapely', 'tessera.pipeline') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
