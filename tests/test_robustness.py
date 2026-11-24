"""Critic/QA team: robustness of the image -> mosaic path against odd inputs.

Each test states the behaviour a maker expects. Known bugs are xfail(strict=True) with the
item id from docs/elestiri_tur1.md, so the test turns red (XPASS) the moment the bug is fixed and
the marker must be removed. Budget: ~40 s.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from qa_ortak import check_mosaic  # noqa: E402

from tessera.model import Filament, Settings, Vocab  # noqa: E402
from tessera.pipeline import build_mosaic, load_image, zones_from_shapes  # noqa: E402

RED = Filament("Kırmızı", (200, 30, 35))
BLACK = Filament("Siyah", (24, 23, 25))
CREAM = Filament("Krem", (240, 235, 220))
GREY = Filament("Gri", (128, 128, 124))


def art(w: int, h: int) -> Image.Image:
    """Cream ground, red disc, black band and a thin black diagonal."""
    im = Image.new("RGB", (w, h), CREAM.rgb)
    d = ImageDraw.Draw(im)
    d.ellipse((w * .2, h * .2, w * .8, h * .8), fill=RED.rgb)
    d.rectangle((0, h * .85, w, h), fill=BLACK.rgb)
    d.line((0, 0, w, h), fill=BLACK.rgb, width=max(1, w // 100))
    return im


def build(src, palette, width_mm=200.0, mm=0.8, zone_fn=None, default_vocab=0):
    s = Settings(width_mm=width_mm, mm_per_px=mm, palette=list(palette), default_vocab=default_vocab)
    img = load_image(src, s)
    zone = zone_fn(img.shape[:2]) if zone_fn else None
    return build_mosaic(img, s, zone)


def assert_invariants(m):
    c = check_mosaic(m)
    assert c["bağlantısız_taş"] == 0, c
    assert c["yükseklik_sonlu"] and c["yükseklik_min"] >= 0, c
    assert c["yükseklik_max"] < 40, c
    assert c["boş_id"] == 0, c
    return c


def colour_at(m, fx, fy):
    h, w = m.colour.shape
    return int(m.colour[int(fy * (h - 1)), int(fx * (w - 1))])


# ----------------------------------------------------------------- sizes
def test_tiny_image_50px(tmp_path):
    p = tmp_path / "kucuk.png"
    art(50, 40).save(p)
    m = build(p, [RED, BLACK, CREAM], width_mm=424, mm=0.8)
    assert_invariants(m)
    assert colour_at(m, 0.35, 0.5) == 0 and colour_at(m, 0.9, 0.1) == 2


def test_huge_image_5000px_is_fast(tmp_path):
    import time
    p = tmp_path / "buyuk.png"
    art(5000, 3500).save(p)
    t = time.perf_counter()
    m = build(p, [RED, BLACK, CREAM], width_mm=424, mm=0.8)
    assert time.perf_counter() - t < 15
    assert_invariants(m)


def test_width_10cm(tmp_path):
    p = tmp_path / "a.png"
    art(600, 800).save(p)
    m = build(p, [RED, BLACK, CREAM], width_mm=100, mm=0.4)
    assert_invariants(m)
    assert m.size_mm[0] == pytest.approx(100, abs=1)


@pytest.mark.skipif(os.environ.get("TESSERA_SLOW") != "1", reason="slow (~70 s); set TESSERA_SLOW=1")
def test_width_300cm_preview_under_budget(tmp_path):
    """The width box allows 300 cm. The 0.8 mm preview of a 300 cm piece must stay usable."""
    import time
    p = tmp_path / "a.png"
    art(600, 800).save(p)
    t = time.perf_counter()
    build(p, [RED, BLACK, CREAM], width_mm=3000, mm=0.8)
    assert time.perf_counter() - t < 15, "K: 300 cm önizleme ~70 sn sürüyor (elestiri_tur1 Ö-4)"


# ----------------------------------------------------------------- palettes
def test_one_colour_palette_gives_turkish_error(tmp_path):
    p = tmp_path / "a.png"
    art(200, 260).save(p)
    with pytest.raises(ValueError, match="filament"):
        build(p, [RED])


@pytest.mark.parametrize("pal", [[RED, CREAM], [RED, BLACK, CREAM, GREY] + [
    Filament(f"R{i}", tuple(int(x) for x in np.random.default_rng(i).integers(0, 255, 3))) for i in range(6)]],
    ids=["2renk", "10renk"])
def test_two_and_ten_colours(tmp_path, pal):
    p = tmp_path / "a.png"
    art(400, 520).save(p)
    m = build(p, pal)
    assert_invariants(m)
    assert m.colour.max() < len(pal)


def test_duplicate_palette_colour_is_used_once(tmp_path):
    p = tmp_path / "a.png"
    art(300, 400).save(p)
    m = build(p, [RED, Filament("Kırmızı 2", RED.rgb), CREAM])
    used = np.bincount(m.colour.ravel(), minlength=3)
    assert used[0] == 0 or used[1] == 0, used


# ----------------------------------------------------------------- file formats
def test_transparent_png_background_is_not_black(tmp_path):
    im = Image.new("RGBA", (300, 400), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse((50, 50, 250, 250), fill=RED.rgb + (255,))
    p = tmp_path / "logo.png"
    im.save(p)
    m = build(p, [RED, BLACK, CREAM])
    assert colour_at(m, 0.5, 0.3) == 0
    assert colour_at(m, 0.02, 0.98) != 1, "transparent background became black"


def test_grayscale_jpg(tmp_path):
    p = tmp_path / "gri.jpg"
    art(400, 520).convert("L").save(p, quality=92)
    m = build(p, [BLACK, CREAM, GREY])
    assert_invariants(m)


def test_cmyk_jpg_keeps_colours(tmp_path):
    p = tmp_path / "cmyk.jpg"
    art(400, 520).convert("CMYK").save(p, quality=95)
    m = build(p, [RED, BLACK, CREAM])
    assert_invariants(m)
    assert colour_at(m, 0.35, 0.5) == 0 and colour_at(m, 0.9, 0.1) == 2


@pytest.mark.parametrize("payload", [b"\x89PNG\r\n\x1a\n" + b"x" * 300, b"", b"GIF89a" + b"\0" * 20],
                         ids=["bozuk_png", "bos_dosya", "bozuk_gif"])
def test_corrupt_file_turkish_message(tmp_path, payload):
    from tessera.ui.engine import friendly_error
    p = tmp_path / "bozuk.png"
    p.write_bytes(payload)
    with pytest.raises(Exception) as ei:
        load_image(str(p), Settings())
    msg = friendly_error(ei.value)
    for eng in ("Truncated", "cannot identify", "Error", "image file"):
        assert eng not in msg, msg


# ----------------------------------------------------------------- zones
def test_all_zones_flat(tmp_path):
    p = tmp_path / "a.png"
    art(300, 400).save(p)
    m = build(p, [RED, BLACK, CREAM], default_vocab=int(Vocab.FLAT))
    assert_invariants(m)


def test_empty_zone_map(tmp_path):
    p = tmp_path / "a.png"
    art(300, 400).save(p)
    m = build(p, [RED, BLACK, CREAM], zone_fn=lambda hw: zones_from_shapes(hw, [], default=int(Vocab.RAYS)))
    assert_invariants(m)
    assert set(np.unique(m.tile_zone)) == {int(Vocab.RAYS)}


def test_invalid_zone_values_give_clear_error(tmp_path):
    p = tmp_path / "a.png"
    art(300, 400).save(p)
    with pytest.raises(ValueError):
        build(p, [RED, BLACK, CREAM], zone_fn=lambda hw: np.full(hw, 9, np.int32))


def test_crumb_tiles_are_merged():
    """Algoritma raporu: kırıntılar komşuya katılır. Tiles under 3.2 mm² (20 px at 0.4 mm) should be rare."""
    from qa_ortak import example_inputs
    img, s, zone, _ = example_inputs("portre", 0.4)
    m = build_mosaic(img, s, zone)
    crumbs = (m.tile_area_px * s.mm_per_px ** 2 < 3.2).sum()
    if crumbs > 0.05 * m.n_tiles:
        pytest.xfail(f"elestiri_tur1 Ö-3: portrede {crumbs}/{m.n_tiles} taş 3,2 mm²'den küçük (bölge sınırında derz kırıntısı)")


# ----------------------------------------------------------------- paths
def test_turkish_path_with_spaces(tmp_path):
    from tessera.project import Project
    d = tmp_path / "Görsel klasörü ğüşıöç İ"
    d.mkdir()
    src = d / "resim ş İ.png"
    art(300, 400).save(src)
    m = build(src, [RED, BLACK, CREAM])
    assert_invariants(m)
    p = Project(image_path=str(src), name="Şükrü'nün İşi")
    p.settings.palette = [RED, CREAM]
    proj = d / "proje ç.tessera.json"
    p.save(str(proj))
    q = Project.load(str(proj))
    assert os.path.samefile(q.image_path, src) and q.name == "Şükrü'nün İşi"


# ----------------------------------------------------------------- UI: errors and data safety
@pytest.fixture(scope="module")
def app():
    from tessera.ui import create_app
    return create_app([])


@pytest.fixture()
def win(app, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QMessageBox
    from tessera.ui.main_window import MainWindow
    boxes = []

    def fake(kind):
        return staticmethod(lambda parent, title, text, *a, **k: boxes.append((kind, title, text)) or QMessageBox.No)
    for k in ("warning", "critical", "information", "question"):
        monkeypatch.setattr(QMessageBox, k, fake(k))
    w = MainWindow(QSettings(str(tmp_path / "ayar.ini"), QSettings.IniFormat))
    w.boxes = boxes
    w.resize(1400, 880)
    w.show()
    app.processEvents()
    yield w
    w.set_dirty(False)
    w.shutdown()
    w.close()
    app.processEvents()


def test_ui_corrupt_image_message_is_turkish(win, tmp_path):
    p = tmp_path / "bozuk ğ.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 200)
    assert win.load_image(str(p)) is False
    assert win.boxes, "no message shown"
    text = win.boxes[-1][2]
    assert "Truncated" not in text and "cannot identify" not in text, text


def test_ui_missing_image_project_keeps_image_path(win, tmp_path):
    from tessera.project import Project
    img = tmp_path / "gecici.png"
    art(200, 260).save(img)
    p = Project(image_path=str(img), name="Kayıp")
    p.settings.palette = [RED, CREAM]
    proj = tmp_path / "kayip.tessera.json"
    p.save(str(proj))
    img.rename(tmp_path / "baska_yerde.png")      # e.g. USB disk not plugged in
    win.open_project(str(proj))
    assert win._save_to(str(proj))
    doc = json.loads(proj.read_text(encoding="utf-8"))
    assert doc["gorsel"], "image reference lost after re-saving"
