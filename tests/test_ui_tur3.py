"""UI team tests, round 3 (docs/elestiri_tur2.md, kararlar R3-*). Split from test_ui.py so that these
window tests run beside the others; the checks are the same."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from test_ui import POSTER, _idle, _wait, app, win  # noqa: F401,E402


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


def _new_image(app, win, tmp_path, w=300, h=400, width_mm=200.0):
    """A new picture, made at 20 cm (the size does not matter to the tests that use it; a small
    piece builds several times faster than the 42 cm default)."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (w, h), (240, 235, 220))
    d = ImageDraw.Draw(im)
    d.ellipse((w * .2, h * .2, w * .8, h * .6), fill=(200, 30, 35))
    d.rectangle((0, h * .8, w, h), fill=(20, 20, 22))
    p = tmp_path / "yeni.png"
    im.save(p)
    win.set_width_mm(width_mm)
    assert win.load_image(str(p))
    assert win.project.settings.width_mm == pytest.approx(width_mm)
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
