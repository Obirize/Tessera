"""UI team tests: project round trip + offscreen smoke test of the main window.

Runs headless (QT_QPA_PLATFORM=offscreen). No pytest-qt needed. Budget: < 60 s.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from tessera.model import Filament, Mode, Settings, Vocab
from tessera.ui.engine import EXPORT_MM_PER_PX
from tessera.project import PAINT_DEFAULT, PAINT_NONE, Project, settings_from_json, settings_to_json

REPO = Path(__file__).resolve().parents[1]
POSTER = REPO / "ornekler" / "poster"


# ---------------------------------------------------------------- project (no Qt)
def test_settings_json_round_trip():
    s = Settings(width_mm=300, mode=Mode.AMS, palette=[Filament("A", (1, 2, 3)), Filament("B", (200, 100, 0), "silk")])
    s.vocab[int(Vocab.RAYS)].tile_mm = 17.5
    s.default_vocab = int(Vocab.RAYS)
    d = json.loads(json.dumps(settings_to_json(s)))
    s2 = settings_from_json(d)
    assert s2.width_mm == 300 and s2.mode == Mode.AMS and s2.default_vocab == int(Vocab.RAYS)
    assert [f.name for f in s2.palette] == ["A", "B"] and s2.palette[1].rgb == (200, 100, 0)
    assert s2.palette[1].finish == "silk"
    assert s2.vocab[int(Vocab.RAYS)].tile_mm == 17.5
    assert set(s2.vocab) == set(s.vocab)


def test_project_round_trip(tmp_path):
    img = tmp_path / "img" / "resim.png"
    img.parent.mkdir()
    from PIL import Image
    Image.new("RGB", (40, 60), (200, 30, 30)).save(img)
    p = Project(image_path=str(img), name="Deneme")
    p.settings.palette = [Filament("Kırmızı", (206, 32, 40)), Filament("Siyah", (24, 23, 25))]
    p.settings.width_mm = 250.0
    p.settings.merge_same_colour = False
    p.shapes = [{"vocab": 3, "polygon": [[0.1, 0.1], [0.5, 0.1], [0.3, 0.6]], "ad": "figür"},
                {"vocab": 2, "rect": [0.0, 0.5, 0.4, 1.0]}, {"vocab": 4, "circle": [0.5, 0.2, 0.1]}]
    p.rays_center = (0.5, 0.12)
    p.owned = ["Kırmızı"]
    p.light_az, p.light_el = 170.0, 22.0
    mask = np.full((30, 20), PAINT_NONE, np.uint8)
    mask[2:8, 2:8] = int(Vocab.FLAT)
    mask[10:12, :] = PAINT_DEFAULT
    p.paint_mask = mask
    path = tmp_path / "proje" / "deneme.tessera.json"
    path.parent.mkdir()
    p.save(str(path))
    assert path.exists() and Path(Project.mask_path_for(str(path))).exists()

    q = Project.load(str(path))
    assert os.path.samefile(q.image_path, img)
    assert q.name == "Deneme" and q.settings.width_mm == 250.0 and q.settings.merge_same_colour is False
    assert q.shapes == p.shapes and q.rays_center == (0.5, 0.12)
    assert [f.name for f in q.settings.palette] == ["Kırmızı", "Siyah"]
    assert q.owned == ["Kırmızı"] and (q.light_az, q.light_el) == (170.0, 22.0)
    assert np.array_equal(q.paint_mask, mask)
    # zone map: shapes + paint
    z = q.zone_map((60, 40))
    assert z.shape == (60, 40)
    assert (z == int(Vocab.FLAT)).any()
    # saving again without paint removes the stale mask file
    q.paint_mask = None
    q.save(str(path))
    assert not Path(Project.mask_path_for(str(path))).exists()


def test_project_relative_image_survives_move(tmp_path):
    from PIL import Image
    d = tmp_path / "a"
    d.mkdir()
    Image.new("RGB", (10, 10)).save(d / "g.png")
    p = Project(image_path=str(d / "g.png"))
    p.save(str(d / "x.tessera.json"))
    moved = tmp_path / "b"
    d.rename(moved)
    q = Project.load(str(moved / "x.tessera.json"))
    assert os.path.samefile(q.image_path, moved / "g.png")


def test_load_rejects_non_project(tmp_path):
    f = tmp_path / "bos.tessera.json"
    f.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        Project.load(str(f))


# ---------------------------------------------------------------- UI smoke (offscreen)
@pytest.fixture(scope="module")
def app():
    from tessera.ui import create_app
    return create_app([])


def _wait(app, cond, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def _idle(win):
    return (win.last_result is not None and not win._debounce.isActive()
            and win.engine._pending is None and not win.status_spin.isVisible())


@pytest.fixture()
def win(app, tmp_path):
    from PySide6.QtCore import QSettings
    from tessera.ui.main_window import MainWindow
    store = QSettings(str(tmp_path / "ayarlar.ini"), QSettings.IniFormat)
    w = MainWindow(store)
    w.resize(1400, 880)
    w.show()
    app.processEvents()
    yield w
    w.set_dirty(False)
    w.shutdown()
    w.close()
    app.processEvents()


@pytest.mark.skipif(not (POSTER / "kaynak.webp").exists(), reason="poster example missing")
def test_ui_poster_flow(app, win, tmp_path, monkeypatch):
    from tessera.ui import main_window as mw

    # empty state first
    assert win.stage_stack.currentWidget() is win.empty
    win.load_example()
    assert win.stage_stack.currentWidget() is win.stage
    doc = json.loads((POSTER / "bolgeler.json").read_text(encoding="utf-8"))
    assert len(win.zones.shapes) == len(doc["sekiller"])
    assert win.zones.default_vocab == int(Vocab.RAYS)
    assert win.zones.rays_center == tuple(doc["isin_merkezi"])
    assert len(win.project.settings.palette) == 3

    assert _wait(app, lambda: _idle(win), 40), "preview never arrived"
    r = win.last_result
    assert r.top.ndim == 3 and r.oblique.ndim == 3 and r.flat.ndim == 3
    assert r.mosaic.settings.mm_per_px == pytest.approx(mw.PREVIEW_MM_PER_PX)   # fast preview
    assert win.project.settings.mm_per_px == pytest.approx(0.4)                 # project untouched
    # stats bar shows real numbers from estimate()
    assert win.stat_items["parts"].value.text() not in ("", "—")
    assert win.stat_items["grams"].value.text().endswith("kg")
    assert "cm" in win.stat_items["size"].value.text()

    # every tab shows an image
    for i in range(5):
        win.tabs.set_current(i)
        app.processEvents()
        assert win.tab == i
        assert not win.canvas.base.pixmap().isNull()
    win.tabs.set_current(mw.TAB_ZONES)
    assert win.canvas.editing is True
    win.tabs.set_current(mw.TAB_MOSAIC)
    assert win.canvas.editing is False

    # drawing with the mouse on the Bölgeler step: rectangle drag + polygon clicks
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    win.go_step(2)
    assert win.tab == mw.TAB_ZONES
    vp = win.canvas.viewport()
    c = vp.rect().center()
    n0 = len(win.zones.shapes)
    win.set_paint_vocab(int(Vocab.BLOCKS))
    win.set_tool("rect")
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, c - QPoint(60, 60))
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, c + QPoint(40, 30))
    assert len(win.zones.shapes) == n0 + 1 and "rect" in win.zones.shapes[-1]
    assert win.zones.shapes[-1]["vocab"] == int(Vocab.BLOCKS)
    win.set_tool("polygon")
    for d in (QPoint(-50, 0), QPoint(0, -50), QPoint(50, 10)):
        QTest.mouseClick(vp, Qt.LeftButton, Qt.NoModifier, c + d)
    QTest.keyClick(win.canvas, Qt.Key_Return)
    assert len(win.zones.shapes) == n0 + 2 and len(win.zones.shapes[-1]["polygon"]) == 3
    win.set_tool("center")
    QTest.mouseClick(vp, Qt.LeftButton, Qt.NoModifier, c)
    rc = win.zones.rays_center
    assert rc is not None and 0 < rc[0] < 1 and 0 < rc[1] < 1
    assert win.canvas.tool == "pan"          # one-shot tool returns to pan
    win.zones.undo(); win.zones.undo(); win.zones.undo()
    assert len(win.zones.shapes) == n0

    # zones: add a rectangle and paint a stroke -> rebuild with new zones, undo works
    from PySide6.QtCore import QPointF
    n = len(win.zones.shapes)
    win.zones.add_shape({"vocab": int(Vocab.FLAT), "rect": [0.0, 0.9, 0.3, 1.0]})
    assert len(win.zones.shapes) == n + 1
    win.zones.begin_stroke()
    win.zones.paint_segment(QPointF(100, 100), QPointF(200, 120), 12, int(Vocab.BLOCKS))
    win.zones.end_stroke()
    assert win.zones.has_paint()
    win.zones.undo()
    assert not win.zones.has_paint()
    win.zones.redo()
    assert win.zones.has_paint()

    # light change re-renders without rebuilding
    assert _wait(app, lambda: _idle(win) and win.last_result.gen == win.gen, 30)
    win.set_light_live(10.0, 30.0)
    assert _wait(app, lambda: win._debounce.isActive() is False and _idle(win)
                 and win.last_result.gen == win.gen, 30)
    assert win.last_result.relit_only

    # width change rebuilds; height follows the image
    win.go_step(0)
    win.image_step.width_spin.setValue(30.0)
    assert win.project.settings.width_mm == pytest.approx(300.0)
    assert _wait(app, lambda: _idle(win) and win.last_result.gen == win.gen, 30)
    assert win.last_result.mosaic.size_mm[0] == pytest.approx(300, abs=1)
    assert win.stat_items["size"].value.text().startswith("30,0 ×")

    # mode toggle
    win.set_mode(Mode.AMS)
    assert win.project.settings.mode == Mode.AMS
    assert _wait(app, lambda: _idle(win) and win.last_result.gen == win.gen, 30)

    # save + reopen
    proj = tmp_path / "poster.tessera.json"
    assert win._save_to(str(proj))
    assert proj.exists() and Path(Project.mask_path_for(str(proj))).exists()
    assert win.dirty is False
    assert win.open_project(str(proj))
    assert len(win.zones.shapes) == n + 1 and win.zones.has_paint()
    assert win.project.settings.mode == Mode.AMS
    assert _wait(app, lambda: _idle(win), 30)

    # export: must call the export API with a FULL resolution mosaic
    calls = {}

    def fake_export(m, out_dir, progress=None):
        calls["mm"] = m.settings.mm_per_px
        calls["out"] = out_dir
        if progress:
            progress(0.5, "yarı yol")
            progress(1.0, "bitti")
        os.makedirs(out_dir, exist_ok=True)
        return {"files": [os.path.join(out_dir, "rapor.json")], "plates": [1, 2], "parts": 12,
                "panels": 2, "grams": 900, "hours": 30, "warnings": ["deneme uyarısı"]}

    import tessera.export.api as api
    monkeypatch.setattr(api, "export_project", fake_export)
    out = tmp_path / "cikti"
    win.export_in_process = False          # thread runner, so the monkeypatch is seen
    win.go_step(5)
    win.export_step.folder.setText(str(out))
    win.export()
    assert _wait(app, lambda: win._export_task is not None and win._export_task.isFinished()
                 and win.last_export is not None, 40), "export did not finish"
    app.processEvents()
    assert calls["mm"] == pytest.approx(EXPORT_MM_PER_PX)
    assert os.path.normpath(calls["out"]) == os.path.normpath(str(out))
    assert win.export_step.result.isVisibleTo(win.export_step)
    assert win.last_export["parts"] == 12


def test_ui_empty_state_and_theme(app, win):
    assert win.rail.enabled[0] and not any(win.rail.enabled[1:])
    win.go_step(3)                       # no image -> stays on step 1
    assert win.panels.currentIndex() == 0
    before = win.theme.name
    win.toggle_theme()
    assert win.theme.name != before
    win.toggle_theme()
    win.set_advanced(True)
    assert all(not w.isHidden() for w in win.zones_step.advanced)
    win.set_advanced(False)


def test_ui_new_image_gets_palette_and_preview(app, win, tmp_path):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (300, 400), (240, 235, 220))
    d = ImageDraw.Draw(im)
    d.ellipse((60, 60, 240, 240), fill=(200, 30, 35))
    d.rectangle((0, 300, 300, 400), fill=(20, 20, 22))
    path = tmp_path / "deneme.png"
    im.save(path)
    assert win.load_image(str(path))
    assert win.stage_stack.currentWidget() is win.stage
    assert _wait(app, lambda: _idle(win) and (win._suggest_task is None or win._suggest_task.isFinished()), 40)
    assert len(win.project.settings.palette) >= 2
    assert win.last_result is not None
    assert win.image_step.height_lbl.text().endswith("cm")
    # automatic zone suggestion (Algorithm team) when it exists
    if win.suggest_zones_available():
        win.go_step(2)
        assert win.zones_step.auto_btn.isVisibleTo(win.zones_step)
        task_before = win._suggest_task
        win.suggest_zones()
        assert _wait(app, lambda: win._suggest_task is not task_before and win._suggest_task.isFinished(), 40)
        app.processEvents()
        assert win.zones.has_paint() or win.zones.shapes
    # a preset from the Art team changes relief parameters
    if win.relief_step.presets:
        pr = win.relief_step.presets[-1]
        win.relief_step._apply_preset(len(win.relief_step.presets) - 1)
        assert win.relief_step.preset_btns[-1].isChecked()
        if "figure_bump_mm" in pr:
            assert win.project.settings.figure_bump_mm == pytest.approx(pr["figure_bump_mm"])


def test_export_process_runner_small(tmp_path):
    """The real child-process export path end to end on a tiny artwork (keeps the window free)."""
    from PIL import Image, ImageDraw
    from tessera.model import Settings
    from tessera.ui import create_app
    from tessera.ui.engine import ExportJob, export_payload
    app = create_app([])
    im = Image.new("RGB", (120, 90), (240, 235, 220))
    ImageDraw.Draw(im).ellipse((30, 15, 90, 75), fill=(200, 30, 35))
    src = tmp_path / "kucuk.png"
    im.save(src)
    s = Settings(width_mm=60, palette=[Filament("Krem", (240, 235, 220)), Filament("Kırmızı", (200, 30, 35))])
    out = tmp_path / "cikti"
    job = ExportJob(export_payload(s, str(src), [], None, None, str(out)))
    got = {}
    job.done.connect(lambda r: got.setdefault("res", r))
    job.error.connect(lambda m: got.setdefault("err", m))
    job.start()
    assert _wait(app, lambda: job.isFinished(), 50), "export process did not finish"
    assert "err" not in got, got.get("err")
    assert isinstance(got["res"], dict) and out.exists()


# ---------------------------------------------------------------- Round 2 (docs/elestiri_tur1.md, docs/kararlar.md)
def _visible_texts(win):
    from PySide6.QtWidgets import QAbstractButton, QLabel
    out = []
    for w in win.findChildren(QLabel) + win.findChildren(QAbstractButton):
        if w.isVisibleTo(win) and w.text():
            out.append(w.text())
    return out


def test_r2_missing_image_keeps_path_and_relinks(app, win, tmp_path, monkeypatch):
    """K-3: a missing image never erases the stored path; 'Görseli bul…' relinks it."""
    from PIL import Image
    from PySide6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: shown.append(a[2]) or QMessageBox.Ok))
    img = tmp_path / "resim.png"
    Image.new("RGB", (200, 300), (200, 30, 30)).save(img)
    p = Project(image_path=str(img), name="Kayıp")
    p.settings.palette = [Filament("Kırmızı", (200, 30, 35)), Filament("Krem", (240, 235, 220))]
    p.shapes = [{"vocab": 3, "rect": [0.1, 0.1, 0.5, 0.5]}]
    mask = np.full((1500, 1000), PAINT_NONE, np.uint8)
    mask[100:200, 100:200] = int(Vocab.BLOCKS)
    p.paint_mask = mask
    proj = tmp_path / "kayip.tessera.json"
    p.save(str(proj))
    moved = tmp_path / "baska" / "resim.png"
    moved.parent.mkdir()
    img.rename(moved)
    assert win.open_project(str(proj))
    assert win.image_missing and not win.has_image()
    assert shown and "Errno" not in shown[-1] and "Görseli bul" in shown[-1]
    assert win.empty.pick.text() == "Görseli bul…"
    assert win._save_to(str(proj))
    doc = json.loads(proj.read_text(encoding="utf-8"))
    assert os.path.normpath(doc["gorsel"]) == os.path.normpath(str(img))
    q = Project.load(str(proj))
    assert np.array_equal(q.paint_mask, mask) and q.shapes == p.shapes     # nothing lost
    assert win.load_image(str(moved))                                       # relink
    assert not win.image_missing and win.has_image()
    assert len(win.zones.shapes) == 1 and win.zones.has_paint()
    assert os.path.samefile(win.project.image_path, moved)


def test_r2_turkish_errors():
    from tessera.ui.engine import friendly_error
    from PIL import UnidentifiedImageError
    for e in (OSError("Truncated File Read"), UnidentifiedImageError("cannot identify image file 'x.png'"),
              FileNotFoundError(2, "No such file or directory", "C:\a\b.png"), KeyError(9),
              RuntimeError("boom")):
        msg = friendly_error(e)
        for eng in ("Truncated", "cannot identify", "Errno", "Error", "No such", "boom"):
            assert eng not in msg, msg


@pytest.mark.skipif(not (POSTER / "kaynak.webp").exists(), reason="poster example missing")
def test_r2_example_preset_budget_and_honest_modes(app, win):
    from tessera.model import Mode
    win.load_example()
    assert win.project.preset == "Sovyet Afiş" or not win.relief_step.presets
    assert _wait(app, lambda: _idle(win), 40)
    r = win.last_result
    assert r.budget.get("kg_m2") is not None or not r.estimate
    assert len(r.budget.get("dagilim", [])) == 3
    if r.estimate:
        # both production cards show their own estimate() numbers (no home-made panel formula)
        assert set(r.est_by_mode) == {Mode.PIECES, Mode.AMS}
        assert f"{r.est_by_mode[Mode.AMS]['panel']} panel" in win.prod_step.on.stat.text()
        assert "0 g" in win.prod_step.off.stat.text()
    texts = " ".join(_visible_texts(win))
    win.go_step(4)
    app.processEvents()
    texts += " ".join(_visible_texts(win))
    for bad in ("purge", "Basic", "Monumental", "AMS kapalı", "AMS açık"):
        assert bad not in texts, bad
    assert "AMS = Bambu'nun çok renkli besleyicisi" in texts
    # moving a slider marks the preset as changed and offers to go back (G-3)
    win.go_step(3)
    win.set_vocab_param(int(Vocab.STONES), "base_height_mm", 2.2)
    assert win.project.preset_modified and win.relief_step.preset_reset.isVisibleTo(win.relief_step)


def test_r2_duplicate_colour_blocked(app, win, tmp_path):
    from PIL import Image
    p = tmp_path / "g.png"
    Image.new("RGB", (100, 140), (240, 235, 220)).save(p)
    win.load_image(str(p))
    _wait(app, lambda: win._suggest_task is None or win._suggest_task.isFinished(), 30)
    pal = win.project.settings.palette
    n = len(pal)
    twin = Filament(pal[0].name + " ikiz", pal[0].rgb)
    win.toggle_palette(twin)
    assert len(win.project.settings.palette) == n
    assert "aynı renk" in win.fil_step.note.text()


def test_r2_new_image_default_preset_and_big_size(app, win, tmp_path, monkeypatch):
    from PIL import Image
    from tessera.ui.filaments import default_preset, load_presets
    p = tmp_path / "g.png"
    Image.new("RGB", (100, 140), (240, 235, 220)).save(p)
    win.load_image(str(p))
    pr = default_preset(load_presets())
    if pr:
        assert win.project.preset == pr["ad"]
    monkeypatch.setattr(type(win), "confirm_big", lambda self, w: True)
    win.go_step(0)
    win.image_step.width_spin.setValue(160)
    assert win.project.settings.width_mm == pytest.approx(1600)
    assert win.image_step.big_warn.isVisibleTo(win.image_step)
    from tessera.ui.engine import export_mm_per_px, preview_mm_per_px
    assert preview_mm_per_px(1600) > 0.8 + 1e-9 or preview_mm_per_px(1600) >= 0.8
    assert export_mm_per_px(3000) >= 3000 / 2600 - 1e-9
    monkeypatch.setattr(type(win), "confirm_big", lambda self, w: False)
    win.image_step.width_spin.setValue(100)
    win.image_step.width_spin.setValue(200)          # refused -> stays at 100
    assert win.project.settings.width_mm == pytest.approx(1000)


def test_r2_plates_and_panel3d_tabs(app, win, tmp_path, monkeypatch):
    from PIL import Image
    from tessera.ui import main_window as mw
    from tessera.ui import views
    p = tmp_path / "g.png"
    Image.new("RGB", (100, 140), (200, 30, 30)).save(p)
    win.load_image(str(p))
    _wait(app, lambda: _idle(win), 30)
    assert not win.tabs.shown[mw.TAB_PLATES]
    pngs = []
    for i in range(3):
        f = tmp_path / f"tabla_{i + 1}.png"
        Image.new("RGB", (64, 64), (i * 60, 100, 100)).save(f)
        pngs.append(str(f))
    win._export_done({"files": [], "plate_images": pngs, "parts": 3})
    assert win.tabs.shown[mw.TAB_PLATES]
    win.tabs.set_current(mw.TAB_PLATES)
    assert win.stage_stack.currentWidget() is win.gallery and len(win.gallery.thumbs) == 3
    # Panel 3D appears only when the 3D team exposes preview_parts
    cube_v = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [0, 0, 5], [10, 0, 5], [10, 10, 5], [0, 10, 5]], float)
    cube_f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                       [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    monkeypatch.setattr(views, "preview_parts_fn",
                        lambda: (lambda m, panel=0: [{"vertices": cube_v, "faces": cube_f, "rgb": (200, 30, 30)}]))
    win._refresh_3d_tab()
    assert win.tabs.shown[mw.TAB_3D]
    win.tabs.set_current(mw.TAB_3D)
    assert _wait(app, lambda: win.panel3d.has_content(), 20)
    # the 3D team's real form: a ready-rendered RGB image
    monkeypatch.setattr(views, "preview_parts_fn",
                        lambda: (lambda m, panel_id=0, size=900, az_deg=-30.0, el_deg=50.0, as_png=False:
                                 np.full((64, 64, 3), 120, np.uint8)))
    win._load_panel3d(0)
    assert _wait(app, lambda: win.panel3d.image is not None, 20)
    assert not win.panel3d.grab().isNull()
    win.tabs.set_current(mw.TAB_MOSAIC)
    assert win.stage_stack.currentWidget() is win.stage


# ---------------------------------------------------------------- Round 3 (docs/elestiri_tur2.md, kararlar R3-*)
def test_r3_budget_colour_follows_weight_only():
    from tessera.ui.engine import budget_level
    assert budget_level("kg", 2.9) == "ok" and budget_level("kg", 3.9) == "warn" and budget_level("kg", 4.1) == "bad"
    from tessera.ui import create_app
    create_app([])
    from tessera.ui.views import BudgetCard
    c = BudgetCard()
    c.update_budget({"kg_m2": 2.5, "parca_m2": 3000, "dagilim": [0.2, 0.3, 0.5], "alan_m2": 0.25})
    assert c.worst() == "ok" and c.verdict.text() == "Ağırlık uygun"   # many parts do not paint it red
    assert "çok parça" in c.rows["parts"][2].text()
    c.update_budget({"kg_m2": 3.8, "parca_m2": 100, "dagilim": [1, 0, 0]})
    assert c.worst() == "warn"


def _new_image(app, win, tmp_path, w=300, h=400):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (w, h), (240, 235, 220))
    d = ImageDraw.Draw(im)
    d.ellipse((w * .2, h * .2, w * .8, h * .6), fill=(200, 30, 35))
    d.rectangle((0, h * .8, w, h), fill=(20, 20, 22))
    p = tmp_path / "yeni.png"
    im.save(p)
    assert win.load_image(str(p))
    assert _wait(app, lambda: _idle(win) and (win._suggest_task is None or win._suggest_task.isFinished()), 40)
    return p


def test_r3_exact_parts_replace_estimate(app, win, tmp_path):
    from tessera.model import Mode
    _new_image(app, win, tmp_path)
    assert win.project.settings.mode == Mode.PIECES
    assert _wait(app, lambda: win.exact_parts_current() is not None, 60), "exact count never arrived"
    ex = win.exact_parts_current()
    assert win.stat_items["parts"].value.text() == tr_num_(ex["parca"])
    assert "Dışa aktarımla aynı" in win.stat_items["parts"].toolTip()


def tr_num_(v):
    from tessera.ui.widgets import tr_num
    return tr_num(v)


def test_r3_export_warnings(app, win, tmp_path):
    _new_image(app, win, tmp_path)
    win.go_step(0)
    win.image_step.width_spin.setValue(18)
    assert win.image_step.small_warn.isVisibleTo(win.image_step)
    assert any("küçük" in w for w in win.export_warnings())
    # delicate share above 5 % -> warning with a plain explanation
    win.exact, win.exact_gen = {"parca": 100, "hassas": 12}, win.last_result.gen
    assert any("hassas parça" in w and "cımbız" in w for w in win.export_warnings())
    win.exact = {"parca": 100, "hassas": 3}
    win.image_step.width_spin.setValue(40)
    win.exact, win.exact_gen = {"parca": 100, "hassas": 3}, win.last_result.gen
    assert not win.export_warnings()


@pytest.mark.skipif(not (POSTER / "kaynak.webp").exists(), reason="poster example missing")
def test_r3_names_from_model_and_no_clipping(app, win):
    from PySide6.QtWidgets import QAbstractButton, QLabel, QWidget
    from tessera.model import VOCAB_NAMES_TR
    win.load_example()
    _wait(app, lambda: _idle(win), 40)
    old = ("Piksel blok", "piksel blok", "Işın şerit", "ışın şerit", "Figür (kabartma)", "Kesik kâğıt", "Taşlar")
    texts = []
    for i in range(6):
        win.go_step(i)
        app.processEvents()
        for w in win.findChildren(QWidget):
            if isinstance(w, (QLabel, QAbstractButton)) and w.text():
                texts.append(w.text())
            if w.toolTip():
                texts.append(w.toolTip())
    import re
    blob = "\n".join(texts)
    for o in old:
        assert not re.search(r"(?<!\w)" + re.escape(o) + r"(?!\w)", blob), o
    for name in VOCAB_NAMES_TR.values():
        assert name in blob
    win.go_step(0)
    app.processEvents()
    ch = next(b for b in win.image_step.findChildren(QAbstractButton) if b.text() == "Değiştir…")
    assert ch.width() >= ch.fontMetrics().horizontalAdvance(ch.text()) + 8


def test_r3_panel3d_follows_mode(app, win, tmp_path, monkeypatch):
    from tessera.model import Mode
    from tessera.ui import main_window as mw
    from tessera.ui import views
    calls = []

    def fake(m, panel_id=0, size=900, az_deg=-30.0, el_deg=50.0, quality="full"):
        quick = quality == "fast"
        calls.append((m.settings.mode, quick))
        return np.full((32, 32, 3), 200 if quick else 100, np.uint8)
    monkeypatch.setattr(views, "preview_parts_fn", lambda: fake)
    _new_image(app, win, tmp_path)
    win._refresh_3d_tab()
    win.tabs.set_current(mw.TAB_3D)
    assert _wait(app, lambda: any(c == (Mode.PIECES, False) for c in calls), 20)
    assert (Mode.PIECES, True) in calls                          # rough view first
    win.set_mode(Mode.AMS)
    assert _wait(app, lambda: any(c[0] == Mode.AMS and not c[1] for c in calls), 30), calls
