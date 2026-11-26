"""Critic/QA team: end-to-end matrix from docs/test_matrisi.md.

Fast part (always): every example at 0.4 mm/px -> invariants, preview speed, estimate sanity.
Slow part (TESSERA_E2E=1): full export of the poster in both modes (+ TESSERA_SLICE=1: slice
every 3MF with the Bambu Studio CLI). Known bugs are xfail(strict=True) with the item id from
docs/elestiri_tur1.md / docs/elestiri_tur2.md. With TESSERA_SLICE=1 the slow test also checks
R2-4 (estimate within 15 % of the slicer).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from qa_ortak import check_mosaic, example_inputs  # noqa: E402

from tessera.export.api import estimate  # noqa: E402
from tessera.model import Mode  # noqa: E402
from tessera.pipeline import build_mosaic  # noqa: E402
from tessera.render import render_top  # noqa: E402

SLOW = os.environ.get("TESSERA_E2E") == "1"
SLICE = os.environ.get("TESSERA_SLICE") == "1"
# the poster and the ink print are checked here; the other examples in test_e2e_ornekler.py, so the
# two halves of this slow matrix run side by side
HERE = ["poster", "murekkep_tram"]


@pytest.fixture(scope="module")
def built():
    cache = {}

    def get(name):
        if name not in cache:
            img, s, z, _ = example_inputs(name, 0.4)
            t = time.perf_counter()
            m = build_mosaic(img, s, z)
            cache[name] = (m, time.perf_counter() - t)
        return cache[name]
    return get


# ----------------------------------------------------------------- pipeline invariants
@pytest.mark.parametrize("name", HERE)
def test_example_invariants(built, name):
    check_example(built, name)


def check_example(built, name):
    m, secs = built(name)
    c = check_mosaic(m)
    assert c["bağlantısız_taş"] == 0, c
    assert c["boş_id"] == 0, c
    assert c["yükseklik_sonlu"] and c["yükseklik_min"] >= 0 and c["yükseklik_max"] < 30, c
    if os.environ.get("TESSERA_PERF") == "1":  # R5-2: timing is advisory
        assert secs < 12, f"{name}: 0,4 mm yapım {secs:.1f} sn"
    # the palette is actually used: no filament is chosen without a single tile
    used = np.bincount(m.tile_colour, minlength=len(m.settings.palette))
    assert (used > 0).all(), used


@pytest.mark.skipif(os.environ.get("TESSERA_PERF") != "1", reason="R5-2: süre testleri TESSERA_PERF=1 ile")
def test_poster_preview_speed():
    """test_matrisi: önizleme < 1,5 sn (0,8 mm/px build + render)."""
    img, s, z, _ = example_inputs("poster", 0.8)
    t = time.perf_counter()
    m = build_mosaic(img, s, z)
    render_top(m)
    assert time.perf_counter() - t < 1.5 * 1.5   # 50 % slack for a busy test machine


@pytest.mark.parametrize("name", ["poster"])           # portre, demir_maske: test_e2e_ornekler.py
def test_min_feature_width(built, name):
    check_min_feature(built, name)


def check_min_feature(built, name):
    """No real tile (>= 3.2 mm²) narrower than settings.min_feature_mm (pipeline promise)."""
    from qa_ortak import tile_widths_mm
    m, _ = built(name)
    w = tile_widths_mm(m.label, m.settings.mm_per_px)
    real = m.tile_area_px * m.settings.mm_per_px ** 2 >= 3.2
    narrow = int((w[real] < m.settings.min_feature_mm - 1e-6).sum())
    assert narrow <= 0.01 * real.sum(), f"{narrow} gerçek taş {m.settings.min_feature_mm} mm'den dar"


# ----------------------------------------------------------------- estimate sanity
@pytest.mark.parametrize("name", ["poster"])           # orman_masali: test_e2e_ornekler.py
def test_estimate_both_modes_fast_and_sane(built, name):
    check_estimates(built, name)


def check_estimates(built, name):
    m, _ = built(name)
    out = {}
    for mode in (Mode.PIECES, Mode.AMS):
        m.settings.mode = mode
        t = time.perf_counter()
        out[mode] = estimate(m)
        if os.environ.get("TESSERA_PERF") == "1":  # R5-2
            assert time.perf_counter() - t < 2.0
        e = out[mode]
        assert e["parca"] > 0 and e["tabla"] > 0 and e["gram"] > 0 and e["saat"] > 0
    m.settings.mode = Mode.PIECES
    assert out[Mode.PIECES]["purge_gram"] == 0 and out[Mode.AMS]["purge_gram"] > 0
    assert out[Mode.AMS]["gram"] > out[Mode.PIECES]["gram"]


# (3D team, round 2: fixed; xfail removed)
def test_estimate_ams_has_fewer_parts_and_same_panels(built):
    m, _ = built("poster")
    m.settings.mode = Mode.PIECES
    ep = estimate(m)
    m.settings.mode = Mode.AMS
    ea = estimate(m)
    m.settings.mode = Mode.PIECES
    # the UI promises "Daha az parça ve daha kolay montaj" for AMS
    assert ea["parca"] < ep["parca"], (ea, ep)
    # Üretim step shows ceil(w/panel) x ceil(h/panel); the stats bar shows estimate()['panel']
    w, h = m.size_mm
    local = math.ceil(w / m.settings.panel_max_mm) * math.ceil(h / m.settings.panel_max_mm)
    assert ea["panel"] == local, (ea["panel"], local)


def test_estimate_preview_vs_full_resolution(built):
    """The stats bar uses the 0,8 mm preview; numbers must stay close to the 0,4 mm export build."""
    img, s, z, _ = example_inputs("poster", 0.8)
    m = built("poster")[0]                       # the same 0,4 mm build, made once for the module
    m.settings.mode = Mode.PIECES
    res = {0.8: estimate(build_mosaic(img, s, z)), 0.4: estimate(m)}
    for k in ("parca", "tabla", "gram"):
        a, b = res[0.8][k], res[0.4][k]
        assert abs(a - b) <= 0.25 * max(a, b), (k, res)


# ----------------------------------------------------------------- preview vs printed colours
# (3D team, round 2: fixed; xfail removed)
def test_printed_colours_match_preview_ams(built):
    """In AMS mode a panel is one multi-colour object: no area should change colour vs the preview."""
    from tessera.geometry.layout import build_layout
    m, _ = built("poster")
    m.settings.mode = Mode.AMS
    try:
        lay = build_layout(m)
    finally:
        m.settings.mode = Mode.PIECES
    diff = (lay.tile_colour[lay.label] != m.colour) & ~m.gap
    assert diff.mean() < 0.001, f"%{100 * diff.mean():.2f} alan başka renkte basılıyor"


# ----------------------------------------------------------------- full export (slow)
def _export(m, out, mode):
    from tessera.export.api import export_project
    m.settings.mode = mode
    prog = []
    t = time.perf_counter()
    res = export_project(m, str(out), progress=lambda f, msg="": prog.append((float(f), msg)))
    return res, time.perf_counter() - t, prog


@pytest.mark.skipif(not SLOW, reason="slow: set TESSERA_E2E=1")
@pytest.mark.parametrize("mode", [Mode.PIECES, Mode.AMS])
def test_poster_full_export(built, tmp_path, mode):
    m, _ = built("poster")
    out = tmp_path / "Çıktı klasörü ş"           # Turkish characters + spaces on purpose
    res, secs, prog = _export(m, out, mode)
    m.settings.mode = Mode.PIECES
    assert secs < 180, f"tam dışa aktarım {secs:.0f} sn (hedef < 3 dk)"
    fr = [f for f, _ in prog]
    assert fr and fr[-1] == pytest.approx(1.0) and all(b >= a - 1e-9 for a, b in zip(fr, fr[1:]))
    assert all(isinstance(msg, str) for _, msg in prog)
    for f in res["files"]:
        assert os.path.exists(f), f
    assert (out / "01_montaj_kilavuzu.pdf").exists()
    assert (out / "rapor.json").exists() and (out / "rapor.txt").exists()
    mfs = [f for f in res["files"] if f.endswith(".3mf")]
    assert mfs
    rap = json.loads((out / "rapor.json").read_text(encoding="utf-8"))
    assert rap
    # estimate shown in the UI vs the real export
    m.settings.mode = mode
    e = estimate(m)
    m.settings.mode = Mode.PIECES
    assert 0.6 < e["parca"] / max(1, res["parts"]) < 1.6, (e, res["parts"])
    assert 0.7 < e["gram"] / max(1, res["grams"]) < 1.4, (e, res["grams"])
    if SLICE:
        from tessera.export.validate import bambu_available, slice_3mf
        if not bambu_available():
            pytest.skip("Bambu Studio yok")
        grams = hours = 0.0
        for f in mfs:
            r = slice_3mf(f)
            assert r["ok"], (f, r)
            grams += sum(p["grams"] for p in r["plates"])
            hours += sum(p["minutes"] for p in r["plates"]) / 60
        # R2-4: estimate within 15 % of the slicer
        assert abs(e["gram"] - grams) <= 0.15 * grams, (e["gram"], grams)
        assert abs(e["saat"] - hours) <= 0.15 * hours, (e["saat"], hours)


# ================================================================= QA round 2 (docs/elestiri_tur2.md)
@pytest.fixture(scope="module")
def poster_layout(built):
    from tessera.geometry.layout import build_layout
    m, _ = built("poster")
    m.settings.mode = Mode.PIECES
    return m, build_layout(m)


def test_preview_equals_print_colours_pieces(poster_layout):
    """R2-4 'önizleme = baskı': no area changes colour in PIECES mode either (K-1 of round 1)."""
    m, lay = poster_layout
    diff = (lay.tile_colour[lay.label] != m.colour) & ~m.gap
    assert diff.mean() < 0.001
    assert not ((lay.tile_part[lay.label] == -1) & ~m.gap).any(), "some area belongs to no part"


# (3D team, round 3: fixed; xfail removed)
def test_hassas_warning_matches_count(poster_layout):
    import re
    _, lay = poster_layout
    n = sum(1 for p in lay.parts if p.kind == "hassas")
    said = [int(re.match(r"(\d+)", w).group(1)) for w in lay.warnings if "hassas" in w]
    assert said == [n], (said, n)


# (3D team, round 3: fixed; xfail removed)
def test_no_rice_grain_parts(poster_layout):
    _, lay = poster_layout
    tiny = [(p.pid, round(p.outline.area, 1)) for p in lay.parts
            if p.kind == "hassas" and p.outline is not None and p.outline.area < 4.0]
    assert not tiny, tiny


# (3D team, round 3: fixed; xfail removed)
def test_lattice_backing_is_one_piece(poster_layout):
    import trimesh
    from tessera.geometry.mesh import build_backing, mesh_arrays
    _, lay = poster_layout
    v, t = mesh_arrays(build_backing(lay, lay.panels[0]))
    m = trimesh.Trimesh(v, t, process=True)
    assert m.body_count == 1, m.body_count


# (3D team, round 3: fixed; xfail removed)
def test_french_cleat_is_45_degrees():
    from tessera.geometry.mesh import french_cleat
    L = 200.0
    man = french_cleat(L)
    # 26 x 7 mm profile with a 45 degree bevel has area (26 + 19) / 2 * 7 = 157.5 mm2
    area = (man.volume() + 3 * 3.1416 * 2.2 ** 2 * 7) / L
    assert area == pytest.approx(157.5, rel=0.08), area


# (3D team, round 3: fixed; xfail removed)
def test_estimate_parts_close_to_layout(built):
    from tessera.geometry.stones import build_stones
    for name in ("poster", "murekkep_tram"):
        m, _ = built(name)
        m.settings.mode = Mode.PIECES
        assert estimate(m)["parca"] == len(build_stones(m).stones)


def test_zone_reasons_use_new_names():
    from tessera.pipeline import load_image, suggest_zones
    from tessera.model import Settings
    s = Settings(mm_per_px=0.8)
    _, info = suggest_zones(load_image(str(ROOT / "ornekler" / "poster" / "kaynak.webp"), s), s)
    text = " ".join(info.get("nedenler", []))
    for old in ("Piksel blok", "Işın şeritleri", "Figür (kabartma)", "Odak:"):
        assert old not in text, old


def test_ui_budget_colour_follows_r2_8():
    from tessera.ui.engine import budget_level
    assert budget_level("kg", 3.9) == "warn"
    assert budget_level("kg", 4.1) == "bad"
