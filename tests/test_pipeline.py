"""Algorithm team tests: tessera.pipeline (build, colour, descreen, tiling, zones, palette)."""
import json
import math
import os
import time

import numpy as np
import pytest
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from tessera.model import Filament, Settings, Vocab
from tessera.pipeline import (auto_palette, build_mosaic, descreen, load_filaments, load_image,
                              suggest_palette, suggest_zones, zones_from_shapes)
from tessera.pipeline.colour import quantize_detail

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSTER = os.path.join(ROOT, "ornekler", "poster", "kaynak.webp")
ZONES = os.path.join(ROOT, "ornekler", "poster", "bolgeler.json")

BLACK, WHITE, RED, GREY = (24, 23, 25), (243, 237, 224), (206, 32, 40), (140, 140, 140)


def _settings(mm=0.4, width_mm=None, palette=(RED, BLACK, WHITE), legacy=False):
    s = Settings(palette=[Filament(f"c{i}", c) for i, c in enumerate(palette)], mm_per_px=mm)
    s.style, s.gap_mm, s.min_stone_mm = "klasik", 1.2, 4.0      # this file covers the pre-R8 tiling; R8: test_kesim.py
    if width_mm:
        s.width_mm = width_mm
    if legacy:
        # the pre-R6 raster path (hairline gaps, merged same-colour parts): kept and tested
        s.style = None
        s.gap_mm = 0.4
        s.min_feature_mm = 1.6
    return s


def _line_art(w=320, h=240, ss=4):
    """Anti-aliased thin features on cream: 1 px and 2 px black lines at several angles, a
    3x3 px dot, a thin red line and a solid black disc. Returns (img, feature masks)."""
    W, H = w * ss, h * ss
    im = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(im)
    feats = {}
    lines = {"yatay_1px": ((20, 30), (300, 30), 1), "egik_1px": ((20, 60), (300, 110), 1),
             "dik_2px": ((160, 120), (160, 220), 2), "capraz_1px": ((30, 220), (130, 130), 1)}
    for k, (a, b, wd) in lines.items():
        d.line((a[0] * ss, a[1] * ss, b[0] * ss, b[1] * ss), fill=BLACK, width=wd * ss)
        feats[k] = (a, b)
    d.line((200 * ss, 150 * ss, 300 * ss, 210 * ss), fill=RED, width=1 * ss)
    feats["kirmizi_1px"] = ((200, 150), (300, 210))
    d.ellipse((60 * ss, 160 * ss, 63 * ss, 163 * ss), fill=BLACK)            # 3 px "nostril"
    d.ellipse((230 * ss, 40 * ss, 290 * ss, 100 * ss), fill=BLACK)
    return np.asarray(im.resize((w, h), Image.BOX)), feats


def _segment_pixels(a, b, n=200, margin=0.12):
    t = np.linspace(margin, 1 - margin, n)
    return (np.rint(a[1] + (b[1] - a[1]) * t).astype(int), np.rint(a[0] + (b[0] - a[0]) * t).astype(int))


def _near(colour, ys, xs, c, r=1):
    """Fraction of the sample points with colour ``c`` within ``r`` px."""
    hits = 0
    for y, x in zip(ys, xs):
        hits += bool((colour[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] == c).any())
    return hits / len(ys)


@pytest.fixture(scope="module")
def poster():
    s = _settings()
    doc = json.load(open(ZONES, encoding="utf-8"))
    img = load_image(POSTER, s)
    h, w = img.shape[:2]
    s.default_vocab = doc["varsayilan"]
    s.rays_center = (doc["isin_merkezi"][0] * w, doc["isin_merkezi"][1] * h)
    z = zones_from_shapes((h, w), doc["sekiller"], default=doc["varsayilan"])
    return img, s, z, doc


@pytest.fixture(scope="module")
def poster_mosaic(poster):
    """The poster built once (klasik, 0.4 mm/px); the tests that only read it share it."""
    img, s, z, _ = poster
    t = time.perf_counter()
    m = build_mosaic(img, s, z)
    return m, time.perf_counter() - t


# ------------------------------------------------------------------ contract
def test_shapes_and_types():
    img, _ = _line_art()
    s = _settings()
    m = build_mosaic(img, s)
    h, w = img.shape[:2]
    for a in (m.colour, m.zone, m.label, m.gap, m.height):
        assert a.shape == (h, w)
    n = m.n_tiles
    assert m.tile_colour.shape == (n,) and m.tile_zone.shape == (n,) and m.tile_area_px.shape == (n,)
    assert m.tile_centroid.shape == (n, 2) and m.tile_plane.shape == (n, 3)
    assert m.label.min() == 0 and m.label.max() == n - 1
    assert m.height.dtype == np.float32 and (m.height >= 0).all()
    assert not m.height[m.gap].any()
    # every tile is one colour and one zone
    assert (m.tile_colour[m.label] == m.colour).all()
    assert (m.tile_zone[m.label] == m.zone).all()
    assert "taş_toplam" in m.stats


def test_deterministic(poster, poster_mosaic):
    img, s, z, _ = poster
    a = poster_mosaic[0]
    b = build_mosaic(img, s, z)                                # built again, after other builds ran
    assert np.array_equal(a.label, b.label) and np.array_equal(a.colour, b.colour)
    assert np.allclose(a.height, b.height)


def _assert_connected(m):
    from tessera.pipeline.tiling import split_components
    assert len(np.unique(m.label)) == m.n_tiles, "empty tile ids"
    # splitting every tile into its 4-connected parts must not create new tiles
    assert int(split_components(m.label).max()) + 1 == m.n_tiles, "a tile has several parts"


def test_every_tile_connected_poster(poster_mosaic):
    _assert_connected(poster_mosaic[0])


def test_every_tile_connected_all_vocabs():
    img, _ = _line_art()
    h, w = img.shape[:2]
    s = _settings()
    s.rays_center = (w * 0.5, h * 0.2)
    z = np.zeros((h, w), np.int32)
    for i, v in enumerate(Vocab):
        z[:, i * w // len(Vocab):(i + 1) * w // len(Vocab)] = int(v)
    m = build_mosaic(img, s, z)
    _assert_connected(m)
    assert set(np.unique(m.tile_zone)) == {int(v) for v in Vocab}


# ------------------------------------------------------------------ detail preservation
def test_thin_features_survive_quantisation():
    img, feats = _line_art()
    pal = np.array([RED, BLACK, WHITE], np.float32)
    col = quantize_detail(img, pal, 0.4)
    for k, (a, b) in feats.items():
        c = 0 if k.startswith("kirmizi") else 1
        ys, xs = _segment_pixels(a, b)
        assert _near(col, ys, xs, c) > 0.95, f"{k} lost"
    # R3-1: a 3 px (1.4 mm²) dot is below the 12 mm² minimum part: absorbed, visible in preview
    assert (col[160:164, 60:64] == 1).sum() == 0


def test_thin_features_survive_as_tiles_with_visible_width():
    img, feats = _line_art()
    m = build_mosaic(img, _settings(legacy=True))
    visible = ~m.gap
    for k, (a, b) in feats.items():
        c = 0 if k.startswith("kirmizi") else 1
        ys, xs = _segment_pixels(a, b, n=60, margin=0.2)
        # the line is still there after the hairline gaps are cut, and it is its own colour
        ok = [((m.colour[y - 2:y + 3, x - 2:x + 3] == c) & visible[y - 2:y + 3, x - 2:x + 3]).sum() >= 2
              for y, x in zip(ys, xs)]
        assert np.mean(ok) > 0.9, f"{k} eaten by gaps"
    # thin line tiles are printable: at least ~1.2 mm (3 px) across somewhere
    t = m.label[30, 160]
    assert m.tile_colour[t] == 1
    assert (m.label[:, 160] == t).sum() >= 3


def test_old_quantiser_was_worse():
    """Guard: the legacy path (detail=False) loses the 1 px lines; the new one keeps them."""
    img, feats = _line_art()
    old = build_mosaic(img, _settings(legacy=True), detail=False).colour
    new = build_mosaic(img, _settings(legacy=True)).colour
    ys, xs = _segment_pixels(*feats["egik_1px"])
    assert _near(new, ys, xs, 1, r=0) > _near(old, ys, xs, 1, r=0)


# ------------------------------------------------------------------ descreen
def halftone_image(w=600, h=420, period=6.0, ss=2):
    """Synthetic print screen: 45 deg dots; gradient band, a 40 % disc crossed by an ink line,
    a 25 % rectangle; 3 px ink outlines."""
    W, H = w * ss, h * ss
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32) / ss
    tone = np.zeros((H, W), np.float32)
    band = yy < 150
    tone[band] = np.clip(xx[band] / w, 0.08, 0.92)
    disc = (xx - 170) ** 2 + (yy - 290) ** 2 < 90 ** 2
    tone[disc] = 0.40
    tone[(xx > 330) & (xx < 540) & (yy > 210) & (yy < 380)] = 0.25
    c = s = math.sqrt(0.5)
    u, v = (xx * c - yy * s) / period, (xx * s + yy * c) / period
    dist = np.hypot(u - np.floor(u) - 0.5, v - np.floor(v) - 0.5)
    im = Image.fromarray(np.where(dist < np.sqrt(tone / np.pi), 20, 245).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.ellipse((80 * ss, 200 * ss, 260 * ss, 380 * ss), outline=20, width=3 * ss)
    d.rectangle((330 * ss, 210 * ss, 540 * ss, 380 * ss), outline=20, width=3 * ss)
    d.line((90 * ss, 230 * ss, 260 * ss, 360 * ss), fill=20, width=2 * ss)
    return np.asarray(im.resize((w, h), Image.BOX).convert("RGB"))


def test_descreen_turns_dots_into_clean_grey():
    a = halftone_image()
    pal = np.array([(20, 20, 20), GREY, (245, 245, 245)], np.float32)
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    inner = (xx - 170) ** 2 + (yy - 290) ** 2 < 70 ** 2
    on_line = np.abs((yy - 230) - (xx - 90) * 130 / 170) < 7   # the ink line and its fringe
    raw = quantize_detail(a, pal, 0.4)
    ds, mask = descreen(a, "auto", max_period_px=15, return_mask=True)
    clean = quantize_detail(ds, pal, 0.4, screen_mask=mask > 0.5)
    assert (raw[inner & ~on_line] == 1).mean() < 0.5           # before: speckle, not grey
    assert (clean[inner & ~on_line] == 1).mean() > 0.95        # after: a clean grey zone
    rect = (xx > 345) & (xx < 525) & (yy > 225) & (yy < 365)
    assert (clean[rect] == 0).mean() < 0.01                    # no black salt-and-pepper
    # the ink line crossing the tone stays black
    ys, xs = _segment_pixels((100, 238), (245, 348))
    assert _near(clean, ys, xs, 0) > 0.9
    # few components in the grey zone (no salt and pepper)
    _, n = ndi.label((clean == 0) & inner & ~on_line)
    assert n < 10


def test_descreen_end_to_end_and_off_switch():
    a = halftone_image(1200, 840, period=7)
    s = _settings(legacy=True, width_mm=240, palette=[(20, 20, 20), GREY, (245, 245, 245)])
    on = build_mosaic(load_image(Image.fromarray(a), s), s)
    off = build_mosaic(load_image(Image.fromarray(a), s, descreen=False), s, descreen=False)
    disc = np.zeros(on.colour.shape, bool)
    yy, xx = np.mgrid[0:disc.shape[0], 0:disc.shape[1]]
    disc = (xx - 85) ** 2 + (yy - 145) ** 2 < 30 ** 2
    assert (on.colour[disc] == 1).mean() > 0.9
    assert (off.colour[disc] == 1).mean() < 0.5
    assert on.n_tiles < off.n_tiles


def test_descreen_leaves_flat_art_alone():
    img = np.asarray(Image.open(POSTER).convert("RGB"))
    assert np.array_equal(descreen(img, "auto", max_period_px=17), img)
    assert np.array_equal(descreen(img, False), img)


# ------------------------------------------------------------------ zones
def _sunburst(w=500, h=400, centre=(260, 90)):
    yy, xx = np.mgrid[0:h, 0:w]
    th = np.degrees(np.arctan2(yy - centre[1], xx - centre[0])) % 360
    img = np.where(((th // 10) % 2 == 0)[..., None], np.array(RED), np.array(WHITE)).astype(np.uint8)
    r = np.hypot(xx - centre[0], yy - centre[1])
    img[r < 40] = (250, 210, 60)
    return img


def test_suggest_zones_finds_sunburst_and_sun():
    img = _sunburst()
    z, info = suggest_zones(img, _settings())
    assert z.shape == img.shape[:2] and z.dtype == np.int32
    assert info["isin_merkezi"] is not None
    cx, cy = info["isin_merkezi"]
    assert abs(cx * 500 - 260) < 15 and abs(cy * 400 - 90) < 15
    assert (z == Vocab.RAYS).mean() > 0.3
    assert z[90, 260] == Vocab.FOCAL
    assert info["nedenler"] and all(isinstance(r, str) for r in info["nedenler"])


def test_suggest_zones_poster():
    img = np.asarray(Image.open(POSTER).convert("RGB"))
    t = time.perf_counter()
    z, info = suggest_zones(img, _settings())
    assert time.perf_counter() - t < 6.0     # ~0.8 s on an idle machine
    cx, cy = info["isin_merkezi"]
    assert abs(cx - 0.5) < 0.05 and abs(cy - 0.118) < 0.05
    h, w = z.shape
    assert z[int(0.118 * h), int(0.5 * w)] == Vocab.FOCAL
    assert (z == Vocab.BLOCKS).any() and (z == Vocab.FIGURE).any()
    assert z[int(0.30 * h), int(0.40 * w)] == Vocab.FIGURE      # the man's face


# ------------------------------------------------------------------ palette
def test_filament_library():
    lib = load_filaments()
    assert len(lib) >= 50
    ids = [e["id"] for e in lib]
    assert len(set(ids)) == len(ids)
    names = [e["ad"] for e in lib]
    assert len(set(names)) == len(names)
    for e in lib:
        assert e["seri"] in ("PLA Basic", "PLA Matte")
        assert len(e["id"]) == 5 and e["id"].isdigit()
        assert len(e["hex"]) == 7 and e["hex"].startswith("#")
    by = {e["id"]: e for e in lib}
    assert by["10101"]["ad"] == "Siyah" and by["10100"]["ad_en"] == "Jade White"


def test_suggest_palette_poster_roles():
    img = np.asarray(Image.open(POSTER).convert("RGB"))
    owned = ["10101", "10100", "10200", "10601", "10501", "11100", "10103"]
    res = suggest_palette(img, load_filaments(owned_ids=owned), 3)
    assert len(res) == 3
    ids = {r["id"] for r in res}
    assert "10101" in ids and "10200" in ids                    # black + red must be there
    assert ids <= set(owned)
    lum = [0.3 * r["filament"].rgb[0] + 0.59 * r["filament"].rgb[1] + 0.11 * r["filament"].rgb[2] for r in res]
    assert lum == sorted(lum)                                   # dark -> light
    roles = [r["rol"] for r in res]
    assert roles[0] == "yapı" and roles[-1] == "ışık" and "vurgu" in roles
    assert abs(sum(r["pay"] for r in res) - 1) < 0.02


def test_auto_palette_ignores_antialias_greys():
    img, _ = _line_art()
    img = img.copy()
    img[150:230, 200:310] = RED
    pal = auto_palette(img, 3)
    got = np.array([f.rgb for f in pal], float)
    for want in (BLACK, WHITE, RED):
        assert np.min(np.abs(got - np.array(want)).sum(1)) < 45, f"{want} missing: {got}"
    assert pal == auto_palette(img, 3)                          # deterministic


# ------------------------------------------------------------------ speed
def test_speed_budget(poster):
    if os.environ.get("TESSERA_PERF") != "1":       # R5-2: without the budgets this test checks nothing
        pytest.skip("timing budgets only with TESSERA_PERF=1 (R5-2)")
    img, s, z, doc = poster
    build_mosaic(img, s, z)                                    # warm-up (imports, caches)
    import gc
    gc.collect()
    t = time.perf_counter()
    build_mosaic(img, s, z)
    full = time.perf_counter() - t
    assert full < 6.0, f"full build {full:.2f} s"

    sp = _settings(mm=0.8)
    imgp = load_image(POSTER, sp)
    h, w = imgp.shape[:2]
    sp.default_vocab = doc["varsayilan"]
    sp.rays_center = (doc["isin_merkezi"][0] * w, doc["isin_merkezi"][1] * h)
    zp = zones_from_shapes((h, w), doc["sekiller"], default=doc["varsayilan"])
    import gc
    prev = 1e9
    for _ in range(3):                                         # best of three: a busy test machine
        gc.collect()        # don't bill the build for collecting the earlier tests' garbage
        t = time.perf_counter()
        build_mosaic(imgp, sp, zp)
        prev = min(prev, time.perf_counter() - t)
    assert prev < 1.5, f"preview build {prev:.2f} s"


# ------------------------------------------------------------------ round 2
def test_transparent_png_on_white(tmp_path):
    im = Image.new("RGBA", (200, 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse((40, 30, 160, 130), fill=RED + (255,))
    p = tmp_path / "logo.png"
    im.save(p)
    s = _settings(width_mm=80)
    img = load_image(str(p), s)
    assert (img[2, 2] > 240).all()                        # background white, not black
    m = build_mosaic(img, s)
    assert m.colour[2, 2] == 2 and m.colour[img.shape[0] // 2, img.shape[1] // 2] == 0


@pytest.mark.parametrize("payload", [b"", b"\x89PNG\r\n\x1a\n" + b"x" * 50, b"hello"])
def test_corrupt_image_turkish_error(tmp_path, payload):
    p = tmp_path / "bozuk.png"
    p.write_bytes(payload)
    with pytest.raises(ValueError, match="Görsel açılamadı"):
        load_image(str(p), _settings())


def test_invalid_input_errors_are_turkish():
    img, _ = _line_art(80, 60)
    with pytest.raises(ValueError, match="taş tipi"):
        build_mosaic(img, _settings(), np.full(img.shape[:2], 9, np.int32))
    with pytest.raises(ValueError, match="En az 2"):
        build_mosaic(img, _settings(palette=[RED]))
    s = _settings()
    s.palette = []
    with pytest.raises(ValueError, match="En az 2"):
        build_mosaic(img, s)


def test_duplicate_palette_colour_used_once():
    img, _ = _line_art()
    m = build_mosaic(img, _settings(palette=[RED, BLACK, WHITE, BLACK]))
    used = np.bincount(m.colour.ravel(), minlength=4)
    assert used[3] == 0 and used[1] > 0


def test_working_resolution_rule():
    from tessera.pipeline import working_mm_per_px
    assert working_mm_per_px(424) == 0.4 and working_mm_per_px(424, preview=True) == 0.8
    assert working_mm_per_px(3000) == pytest.approx(3000 / 2600)
    assert working_mm_per_px(3000, preview=True) == pytest.approx(3000 / 1300)
    s = _settings(mm=0.4, width_mm=3000)
    img = load_image(Image.new("RGB", (300, 400), WHITE), s)
    assert s.mm_per_px == pytest.approx(3000 / 2600, abs=1e-3) and img.shape[1] == 2600


def test_relief_budget_poster(poster):
    import copy
    from tessera.pipeline import relief_budget
    img, s, z, _ = poster
    s = copy.deepcopy(s)
    s.style, s.gap_mm = None, 0.4              # relief bands belong to the pre-R6 raster path
    m = build_mosaic(img, s, z)
    rb = relief_budget(m)
    # grams come from the 3D team's single formula (R2-7); here only check the shape
    assert rb["durum"] in ("yeşil", "sarı", "kırmızı", "bilinmiyor")
    assert rb["kg_m2"] is None or rb["kg_m2"] > 0
    assert rb["bantlar"]["0–3 mm"] >= 0.65 and rb["bantlar"]["8 mm üstü"] <= 0.06
    assert rb["bantlar"]["8 mm üstü"] > 0.005                  # relief stays visible (R2-9)
    assert rb["odak_en_yuksek"]
    top = m.tile_plane[:, 0]
    foc = m.tile_zone == Vocab.FOCAL
    assert top[foc].min() >= top[~foc].max()
    # round 5 (first version's look): the whole FIGURE zone is one dome, every colour is a tilted
    # facet of it; dark clothing is not a separate level
    fig = m.tile_zone == Vocab.FIGURE
    dark = fig & (m.tile_colour == 1)
    light = fig & (m.tile_colour == 2)
    assert np.abs(m.tile_plane[dark, 1:]).sum(1).mean() > 0.01       # facets tilt
    assert np.abs(m.tile_plane[light, 1:]).sum(1).mean() > 0.01
    # no flattening: the hierarchy is kept, steps of 1.5 mm remain
    assert top.max() > 8.0
    # plateau steps on a 1.5 mm grid (jitter at most 0.25 mm)
    plain = ~fig & ~foc
    r = np.abs(top[plain] / 1.5 - np.rint(top[plain] / 1.5)) * 1.5
    assert (r <= 0.26).mean() > 0.99


def test_flat_zone_cut_paper_layers():
    img, _ = _line_art()
    s = _settings(legacy=True)
    s.default_vocab = int(Vocab.FLAT)
    m = build_mosaic(img, s)
    tops = {c: np.median(m.tile_plane[m.tile_colour == c, 0]) for c in np.unique(m.tile_colour)}
    vals = sorted(tops.values())
    assert all(b - a >= 1.4 for a, b in zip(vals, vals[1:]))   # one paper layer per colour
    assert tops[2] == min(vals)                                  # cream (largest area) at the bottom


def test_gap_mm_is_honoured():
    img, feats = _line_art()
    s1, s2 = _settings(legacy=True), _settings(legacy=True)
    s2.gap_mm = 1.2
    g1 = build_mosaic(img, s1).gap.mean()
    m2 = build_mosaic(img, s2)
    assert m2.gap.mean() > 1.8 * g1
    ys, xs = _segment_pixels(*feats["egik_1px"], n=40, margin=0.2)
    ok = [((m2.colour[y - 2:y + 3, x - 2:x + 3] == 1) & ~m2.gap[y - 2:y + 3, x - 2:x + 3]).any() for y, x in zip(ys, xs)]
    assert np.mean(ok) > 0.9                                     # thin lines are not eaten


def test_zone_border_crumbs_are_merged():
    """A zone border cutting through colour regions must not leave sliver tiles."""
    img, _ = _line_art()
    h, w = img.shape[:2]
    z = np.zeros((h, w), np.int32)
    yy, xx = np.mgrid[0:h, 0:w]
    z[(xx + 0.37 * yy) % 97 < 48] = int(Vocab.FLAT)               # slanted stripes of two zones
    m = build_mosaic(img, _settings(), z)
    crumbs = (m.tile_area_px * 0.16 < 3.2)
    only_gap = np.bincount(m.label.ravel(), (~m.gap).ravel(), m.n_tiles) == 0
    assert only_gap.sum() == 0
    assert crumbs.sum() <= 0.05 * m.n_tiles + 3


def test_rays_logpolar_and_focal_grid():
    """Round 5: ray planks have a constant angular width and grow with the distance from the
    sun (log-polar); the focal disc is the first version's fine square grid."""
    img = _sunburst(600, 600, centre=(300, 300))
    s = _settings(legacy=True, palette=[RED, WHITE, (250, 210, 60)])
    s.rays_center = (300, 300)
    s.default_vocab = int(Vocab.RAYS)
    m = build_mosaic(img, s)
    d = np.hypot(m.tile_centroid[:, 0] - 300, m.tile_centroid[:, 1] - 300)
    a = m.tile_area_px.astype(float)
    inner, outer = (d > 60) & (d < 110), (d > 230) & (d < 290)
    assert np.median(a[outer]) > 3 * np.median(a[inner])        # perspective: bigger outside
    z = zones_from_shapes((600, 600), [{"vocab": int(Vocab.FOCAL), "circle": [0.5, 0.5, 0.12]}],
                          default=int(Vocab.RAYS))
    mf = build_mosaic(img, s, z)
    foc = (mf.tile_zone == Vocab.FOCAL) & (mf.tile_area_px > 100)
    side = 5.2 / 0.4 if s.vocab[int(Vocab.FOCAL)].tile_mm == 5.2 else s.vocab[int(Vocab.FOCAL)].tile_mm / 0.4
    assert foc.sum() > 10
    assert np.median(mf.tile_area_px[foc]) <= 1.2 * side * side    # grid squares, not rings

def test_figure_tone_change_keeps_one_level():
    """A face in two tones (lit + grey shadow half) is one rounded body, not two levels."""
    h, w = 240, 240
    img = np.full((h, w, 3), 245, np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    face = (xx - 120) ** 2 / 70 ** 2 + (yy - 120) ** 2 / 95 ** 2 < 1
    img[face] = (235, 235, 235)
    img[face & (xx < 120)] = (130, 130, 130)                    # shadow half
    img[(np.abs(xx - 120) < 3) & face] = (10, 10, 10)          # ink line down the middle
    s = _settings(legacy=True, palette=[(10, 10, 10), (130, 130, 130), (235, 235, 235)])
    z = np.full((h, w), int(Vocab.STONES), np.int32)
    z[face] = int(Vocab.FIGURE)
    m = build_mosaic(img, s, z, andamento=True)
    # measure locally at the tone edge (strips right beside the ink line), not half against half:
    # farther out the dome and the 70/25/5 height bands legitimately change the level
    hl = m.height[100:140, 106:115][~m.gap[100:140, 106:115]]
    hr = m.height[100:140, 125:134][~m.gap[100:140, 125:134]]
    assert abs(np.median(hl) - np.median(hr)) < 0.75               # no full 1.5 mm step at the tone edge


def test_ray_planks_not_needle_thin():
    img = _sunburst(500, 400, centre=(250, 30))
    s = _settings(palette=[RED, WHITE, (250, 210, 60)])
    s.rays_center = (250, 30)
    s.default_vocab = int(Vocab.RAYS)
    m = build_mosaic(img, s)
    pad = np.pad(m.label, 1, constant_values=-1)
    c = pad[1:-1, 1:-1]
    bnd = (pad[:-2, 1:-1] != c) | (pad[2:, 1:-1] != c) | (pad[1:-1, :-2] != c) | (pad[1:-1, 2:] != c)
    d = ndi.distance_transform_edt(~bnd)
    widest = np.asarray(ndi.maximum(d, m.label, np.arange(m.n_tiles)))
    real = m.tile_area_px * 0.16 >= 3.2
    width_mm = (2 * np.floor(widest) + 1) * 0.4
    assert (width_mm[real] < 1.6).mean() < 0.05                  # no needle planks (min 2 mm strips)


# ------------------------------------------------------------------ round 3 (R3-1)
def test_min_part_rule_islands_and_halo():
    """R3-1: no colour island under 12 mm² (except backing-colour ribs next to a dark line), and
    the anti-alias edge between mustard and black must not turn red (K2-1)."""
    h, w = 300, 400
    W, H = w * 4, h * 4
    im = Image.new("RGB", (W, H), (200, 160, 60))                 # mustard ground
    d = ImageDraw.Draw(im)
    d.polygon([(0, 700), (1600, 300), (1600, 420), (0, 820)], fill=(20, 20, 20))   # black diagonal band
    d.ellipse((1100, 800, 1300, 1000), fill=(200, 30, 35))         # a real red disc
    for i in range(12):                                            # 1.2 mm cream specks (noise)
        d.ellipse((100 + i * 100, 1050, 112 + i * 100, 1062), fill=(240, 235, 220))
    img = np.asarray(im.resize((w, h), Image.BOX))
    s = _settings(palette=[(20, 20, 20), (200, 30, 35), (200, 160, 60), (240, 235, 220)])
    m = build_mosaic(img, s)
    mm2 = s.mm_per_px ** 2
    for c in range(4):
        lab, n = ndi.label(m.colour == c)
        a = np.bincount(lab.ravel())[1:] * mm2
        assert (a < 12).sum() == 0 or c == 0, (c, sorted(a)[:5])
    red, n_red = ndi.label(m.colour == 1)
    assert n_red == 1                                              # only the real disc, no halo
    assert (m.colour[255:265, 25:300] == 3).sum() == 0             # specks absorbed (preview shows it)


def test_min_feature_is_1_6_mm():
    img, feats = _line_art()
    s = _settings(legacy=True)
    m = build_mosaic(img, s)
    t = m.label[30, 160]
    assert m.tile_colour[t] == 1 and (m.label[:, 160] == t).sum() >= 4   # 1 px line -> >= 1.6 mm


def test_progress_is_monotonic_and_completes(tmp_path):
    p = tmp_path / "a.png"
    img, _ = _line_art()
    Image.fromarray(img).save(p)
    s = _settings(width_mm=128)
    z = np.zeros(img.shape[:2], np.int32)
    z[:, 160:] = int(Vocab.RAYS)
    calls = []
    build_mosaic(str(p), s, None, progress=lambda f, msg: calls.append((f, msg)))
    fr = [f for f, _ in calls]
    assert fr[0] == 0.0 and fr[-1] == 1.0
    assert all(b >= a for a, b in zip(fr, fr[1:]))
    assert all(isinstance(m, str) and m for _, m in calls) and len(calls) >= 6
    calls2 = []
    build_mosaic(img, _settings(), z, progress=lambda f, msg: calls2.append(f))
    assert calls2[-1] == 1.0 and all(b >= a for a, b in zip(calls2, calls2[1:]))
