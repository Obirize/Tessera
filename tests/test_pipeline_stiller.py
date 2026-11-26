"""Algorithm team tests, round 6 (separate stones in every style) on the poster. Split from
test_pipeline.py so that its four poster builds run beside the rest; the checks are the same."""
import json
import os
import time

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage as ndi

from tessera.model import Vocab
from tessera.pipeline import build_mosaic, load_image, zones_from_shapes

from test_pipeline import POSTER, ZONES, _line_art, _settings


# ------------------------------------------------------------------ round 6: separate stones (R6-1..R6-3)
STYLES_ALL = ("klasik", "smalti", "freud", "orman")


@pytest.fixture(scope="module")
def poster_styles():
    """The 42 x 59 cm poster built once in every style."""
    doc = json.load(open(ZONES, encoding="utf-8"))
    out = {}
    for st in STYLES_ALL:
        s = _settings()
        s.style = st
        img = load_image(POSTER, s)
        h, w = img.shape[:2]
        s.default_vocab = doc["varsayilan"]
        s.rays_center = (doc["isin_merkezi"][0] * w, doc["isin_merkezi"][1] * h)
        z = zones_from_shapes((h, w), doc["sekiller"], default=doc["varsayilan"])
        t = time.perf_counter()
        m = build_mosaic(img, s, z)
        out[st] = (m, time.perf_counter() - t, (img, s, z))
    return out


@pytest.mark.parametrize("st", STYLES_ALL)
def test_style_builds_separate_stones(poster_styles, st):
    from tessera.pipeline.build import stone_widths_mm
    m, secs, _ = poster_styles[st]
    assert m.n_tiles > 500
    # R6-3: no printed stone narrower than 4 mm (widest inscribed circle of its non-gap pixels)
    wd = stone_widths_mm(m.label, m.gap, m.settings.mm_per_px)
    assert wd.min() >= m.settings.min_stone_mm - 1e-6, sorted(wd)[:5]
    # real gaps: ~gap_mm band on every joint, the wall shows (height 0)
    a, b = m.label[:, :-1], m.label[:, 1:]
    joint = a != b
    assert (m.gap[:, :-1] | m.gap[:, 1:])[joint].all()
    assert 0.08 < m.gap.mean() < 0.4
    assert (m.height[m.gap] == 0).all() and (m.height[~m.gap] > 0).mean() > 0.99
    # every stone is one connected piece
    from tessera.pipeline.tiling import split_components
    assert int(split_components(m.label).max()) + 1 == m.n_tiles
    # per-stone extras
    assert m.tile_profile.shape == (m.n_tiles,) and m.tile_dir.shape == (m.n_tiles,)
    if st == "freud":
        assert set(np.unique(m.tile_profile)) <= {1, 2} and (m.tile_profile == 2).mean() > 0.5
    else:
        assert (m.tile_profile == 0).all()
    for g in m.grooves:
        assert len(g["pts"]) >= 2 and g["width"] > 0 and g["depth"] > 0


def test_style_heights_and_tilts(poster_styles):
    deg = lambda m: np.degrees(np.arctan(np.hypot(m.tile_plane[:, 1], m.tile_plane[:, 2])))
    k = poster_styles["klasik"][0]
    nonfig = k.tile_zone != Vocab.FIGURE
    assert np.percentile(deg(k)[nonfig], 95) <= 3.2                  # klasik: +-3 deg
    sm = poster_styles["smalti"][0]
    assert 3.0 < np.percentile(deg(sm), 95) <= 5.2                   # smalti: +-5 deg
    o = poster_styles["orman"][0]
    top = o.tile_plane[:, 0]
    assert np.median(top[o.tile_zone == Vocab.FIGURE]) > np.median(top[o.tile_zone == Vocab.RAYS]) + 4   # levels
    f = poster_styles["freud"][0]
    hi, lo = f.tile_hilo[:, 0], f.tile_hilo[:, 1]
    assert (hi > lo).all() and hi.max() <= 10.0 + 1e-6


def test_face_details_survive_every_style(poster_styles):
    """Eyes, nostril and lips of the man (thin dark features of the source) are still there as
    dark stones or carved grooves."""
    src = np.asarray(Image.open(POSTER).convert("RGB")).astype(float) @ np.array([0.299, 0.587, 0.114])
    y0, y1, x0, x1 = 400, 600, 380, 520
    dark = src[y0:y1, x0:x1] < 90
    feat = dark & ~ndi.binary_opening(dark, np.ones((11, 11), bool))
    assert feat.sum() > 500
    for st in STYLES_ALL:
        m = poster_styles[st][0]
        kept = (m.colour[y0:y1, x0:x1] == 1) | (m.groove_depth[y0:y1, x0:x1] > 0.05)
        kept = ndi.binary_dilation(kept, iterations=1)
        assert kept[feat].mean() > 0.88, (st, kept[feat].mean())


def test_line_modes_row_and_groove():
    img, feats = _line_art()
    out = {}
    for lm in ("row", "groove", "auto"):
        s = _settings()
        s.line_mode = lm
        out[lm] = build_mosaic(img, s)
    assert out["row"].grooves == [] and out["groove"].grooves
    # row: the 1 px line is now a row of stones >= 4 mm wide, in its own colour
    from tessera.pipeline.build import stone_widths_mm
    r = out["row"]
    assert (r.colour[28:33, 60:250] == 1).mean() > 0.5
    assert stone_widths_mm(r.label, r.gap, 0.4).min() >= 4.0 - 1e-6
    # groove: the line is carved into the cream stones instead
    g = out["groove"]
    assert (g.groove_depth[28:33, 60:250] > 0).mean() > 0.3
    assert all(pt[0] >= 0 and pt[1] >= 0 for gr in g.grooves for pt in gr["pts"])


@pytest.mark.parametrize("st", STYLES_ALL)
def test_style_deterministic(st):
    img, _ = _line_art()
    s = _settings()
    s.style = st
    a, b = build_mosaic(img, s), build_mosaic(img, s)
    assert np.array_equal(a.label, b.label) and np.allclose(a.height, b.height)


def test_style_speed(poster_styles):
    if os.environ.get("TESSERA_PERF") != "1":
        pytest.skip("timing budgets only with TESSERA_PERF=1 (R5-2)")
    for st in STYLES_ALL:
        assert poster_styles[st][1] < 10.0, (st, poster_styles[st][1])
