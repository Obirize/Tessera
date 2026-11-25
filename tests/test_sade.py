"""Tonal pictures (pipeline/sade.py): which pictures are simplified, and what simplifying does."""
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi

from tessera import app2_engine as E
from tessera.pipeline import descreen, kesim, open_rgb, renk, sade
from tests.test_renk import dark_mask

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def photo(w=640, h=520, seed=2):
    """A lit head with soft shading, a small red mouth and dark eyes on a grainy, uneven ground."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    g = ndi.gaussian_filter(rng.normal(size=(h, w)), 2.2)
    g = g / g.std()
    ground = np.stack([58 + 26 * xx / w + 7 * g, 60 + 22 * xx / w + 7 * g, 52 + 18 * yy / h + 7 * g], -1)
    d = ((xx - w / 2) / 150) ** 2 + ((yy - h / 2) / 190) ** 2
    light = np.clip(1.15 - 0.75 * ((xx - 0.4 * w) / 230) ** 2 - 0.35 * ((yy - 0.42 * h) / 260) ** 2, 0.25, 1.1)
    grain = ndi.gaussian_filter(rng.normal(size=(h, w)), 1.0) * 22
    skin = np.stack([215 * light, 168 * light, 134 * light], -1) + grain[..., None]
    a = np.where((d < 1)[..., None], skin, ground)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    for cx in (0.41, 0.56):
        dr.ellipse((cx * w - 17, 0.42 * h - 9, cx * w + 17, 0.42 * h + 9), fill=(30, 24, 22))
    dr.ellipse((0.42 * w, 0.66 * h, 0.56 * w, 0.71 * h), fill=(150, 40, 40))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(1.6)))


def own(im):
    return lambda: renk.find(im)


def test_flat_art_is_never_simplified():
    assert sade.kind(dark_mask(), own(dark_mask()))[0] == "net"
    for f in ("poster/kaynak.webp", "portre/kaynak.png", "logo_grafik/kaynak.png", "orman_masali/kaynak.png",
              "demir_maske/kaynak.png", "ankara_kedisi/kaynak.png"):
        p = os.path.join(ROOT, "ornekler", f)
        if os.path.exists(p):
            im = renk._shrink(np.asarray(open_rgb(p)), 1200)
            assert sade.kind(im, own(im))[0] == "net", f


def test_photo_is_recognised_and_becomes_few_clean_shapes():
    im = photo()
    assert sade.kind(im, own(im))[0] == "ton"
    c = sade.find(im, 300)
    assert 4 <= len(c) <= sade.MAX_PAINTS and all(x["yol"] == "ton" for x in c)
    assert len({x["hex"] for x in c}) == len(c) and sum(x["pay"] for x in c) == pytest.approx(1, abs=0.01)
    L = sade.label(im, c, 300)
    mm = 300 / L.shape[1]
    tiny = sum(int((np.bincount(ndi.label(L == i)[0].ravel())[1:] * mm * mm < 6).sum()) for i in range(len(c)))
    assert tiny == 0                                                  # no crumbs to cut
    h, w = L.shape
    skin = L[int(.5 * h), int(.45 * w)]
    for x in (.41, .56):                                              # the eyes stay
        assert L[int(.42 * h), int(x * w)] != skin
    assert L[int(.685 * h), int(.49 * w)] != skin                     # so does the small mouth


def test_simplified_picture_cuts_into_sound_stones():
    im = photo()
    c = sade.find(im, 200)
    pal = np.array([renk.hex_rgb(x["hex"]) for x in c])
    r = kesim.build(im, 200, pal, colours=c, to=np.arange(len(c)))
    assert len(r["polys"]) > 60 and all(p.is_valid and p.area >= 1.5 for p in r["polys"])
    n = np.array([2 * p.area / p.length for p in r["polys"]])
    assert (n < 1.2).mean() < 0.15


def test_number_of_paints_can_be_chosen():
    for n in (3, 6):
        assert len(sade.find(photo(), 300, n)) <= n


def test_brush_strokes_are_not_a_printed_screen():
    rng = np.random.default_rng(1)
    g = ndi.gaussian_filter(rng.normal(size=(500, 500)), (1.2, 3.0))
    a = np.clip(120 + 60 * g / g.std(), 0, 255).astype(np.uint8)
    a = np.stack([a] * 3, -1)
    assert np.array_equal(descreen(a, "auto", max_period_px=20), a)
    yy, xx = np.mgrid[0:500, 0:500]
    dots = np.hypot((xx + yy) % 9 - 4, (xx - yy) % 9 - 4) < 3.2       # a real one still is
    b = np.stack([np.where(dots, 20, 245).astype(np.uint8)] * 3, -1)
    assert not np.array_equal(descreen(b, "auto", max_period_px=20), b)


def test_advice_is_plain_and_honest(tmp_path):
    p = str(tmp_path / "f.png")
    Image.fromarray(photo()).save(p)
    c, a = E.analyse(p, 300)
    assert a["tur"] == "ton" and a["puan"] in ("orta", "zor") and c[0]["yol"] == "ton"
    c, a = E.analyse(p, 300, tur="net")
    assert a["tur"] == "net" and "yol" not in c[0] and "lekeli" in " ".join(a["soz"])
    assert any("küçük" in x for x in E.analyse(p, 900)[1]["soz"])      # 640 px cannot be 90 cm
    m = str(tmp_path / "m.png")
    Image.fromarray(dark_mask()).save(m)
    a = E.analyse(m, 300)[1]
    assert a["puan"] == "iyi" and a["soz"] == ["Net renkli bir görsel, mozaiğe uygun."]


def test_nearly_black_stays_black_but_a_dark_teal_does_not():
    assert renk.match([{"rgb": (36, 31, 19)}])[0][0] == renk.match([{"rgb": (31, 26, 21)}])[0][0] == "Siyah"
    assert renk.match([{"rgb": (3, 36, 41)}])[0][0] == "Turkuaz"
