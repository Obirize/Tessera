"""The picture's own main colours (pipeline/renk.py)."""
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from tessera import app2_engine as E
from tessera.pipeline import kesim, renk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dark_mask(side=900):
    """A dark head on white: a soft gloss on it, thin bronze ornaments and dark teal eyes that are no
    lighter than the head itself (only their hue differs)."""
    k = 2
    n = side * k
    yy, xx = np.mgrid[0:n, 0:n] / n
    gloss = 22 * np.exp(-(((xx - .6) / .12) ** 2 + ((yy - .28) / .12) ** 2))
    body = np.stack([24 + gloss, 24 + gloss, 23 + gloss], -1)
    im = Image.fromarray(body.astype(np.uint8))
    head = Image.new("L", (n, n), 0)
    ImageDraw.Draw(head).ellipse((.18 * n, .05 * n, .82 * n, .95 * n), fill=255)
    d = ImageDraw.Draw(im)
    for cx in (.38, .62):
        d.ellipse(((cx - .07) * n, .45 * n, (cx + .07) * n, .59 * n), fill=(3, 42, 48))
    for x in (.46, .54):
        d.line([(x * n, .1 * n), (x * n, .4 * n)], fill=(120, 103, 84), width=int(.008 * n))
        d.line([(x * n, .62 * n), (x * n, .88 * n)], fill=(78, 67, 55), width=int(.008 * n))      # in the shade
    d.polygon([(.5 * n, .2 * n), (.53 * n, .25 * n), (.5 * n, .3 * n), (.47 * n, .25 * n)], fill=(140, 120, 98))
    out = Image.composite(im, Image.new("RGB", (n, n), (254, 254, 254)), head).filter(ImageFilter.GaussianBlur(1.2))
    a = np.asarray(out.resize((side, side), Image.LANCZOS)).astype(int)
    a = a + np.random.default_rng(3).integers(-3, 4, a.shape)
    return np.clip(a, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def mask():
    im = dark_mask()
    return im, renk.find(im)


def test_dark_picture_keeps_eyes_and_ornaments(mask):
    im, c = mask
    assert len(c) == 4, [x["rgb"] for x in c]
    names = [n for n, _ in renk.match(c)]
    assert names[:2] == ["Beyaz", "Siyah"] and "Turkuaz" in names
    assert set(names) & {"Kahverengi", "Açık kahve", "Bej"}
    L = renk.label(im, c)
    n = im.shape[0]
    at = lambda x, y: names[L[int(y * n), int(x * n)]]  # noqa: E731
    for x in (.38, .62):                             # both eyes, up to near their rim
        assert at(x, .52) == at(x - .05, .52) == at(x + .05, .52) == "Turkuaz"
    assert at(.6, .28) == at(.7, .3) == at(.5, .55) == "Siyah"                  # the gloss is not a colour
    assert at(.5, .25) not in ("Siyah", "Beyaz", "Turkuaz") and at(.02, .02) == "Beyaz"
    for x in (.46, .54):                             # thin lines stay whole, in the shade too
        for y0, y1 in ((.12, .38), (.64, .86)):
            col = L[int(y0 * n):int(y1 * n), int(x * n) - 4:int(x * n) + 5]
            assert np.mean([(r == L[int(.25 * n), int(.5 * n)]).any() for r in col]) > 0.9
    # the rim of the head is black against white: no third colour along it
    rim = L[int(.5 * n), int(.16 * n):int(.2 * n)]
    assert {names[v] for v in rim} <= {"Siyah", "Beyaz"}


def test_detection_does_not_depend_on_the_seed(mask):
    im, _ = mask
    assert {len(renk.find(im, seed=s)) for s in (1, 2, 5)} == {4}


@pytest.mark.parametrize("f,names", [
    ("poster/kaynak.webp", {"Kırmızı", "Beyaz", "Siyah"}),
    ("portre/kaynak.png", {"Altın", "Siyah", "Kırmızı", "Beyaz"}),
    ("logo_grafik/kaynak.png", {"Beyaz", "Siyah", "Kırmızı", "Gri"}),
    ("demir_maske/kaynak.png", {"Kırmızı", "Siyah", "Beyaz", "Altın"}),
    ("orman_masali/kaynak.png", {"Koyu yeşil", "Yeşil", "Açık yeşil", "Krem", "Açık mavi", "Kahverengi", "Sarı"}),
])
def test_example_pictures(f, names):
    p = os.path.join(ROOT, "ornekler", f)
    if not os.path.exists(p):
        pytest.skip("örnek yok")
    c = E.analyse(p, 420)[0]
    assert {x["ad"] for x in c} == names
    assert len({x["hex"] for x in c}) == len(c)


def test_exact_number_of_colours(mask):
    im, _ = mask
    for n in (2, 3, 6):
        assert len(renk.find(im, n)) == n


def test_greys_stay_grey():
    g = [{"rgb": [v, v, v]} for v in (0, 30, 60, 100, 140, 190, 250)]
    assert {n for n, _ in renk.match(g)} <= {"Siyah", "Koyu gri", "Gri", "Açık gri", "Beyaz"}


def test_labels_grow_smoothly():
    lab = np.zeros((60, 60), np.int16)
    yy, xx = np.mgrid[0:60, 0:60]
    lab[(yy - 30) ** 2 + (xx - 30) ** 2 < 400] = 1
    big = renk.enlarge(lab, 2, (240, 240))
    r = np.hypot(*(np.argwhere(big == 1) - 121.5).T)
    assert big.shape == (240, 240) and 76 < r.max() < 86


def test_repainting_moves_no_stone(mask):
    im, c = mask
    to = np.arange(len(c))
    a = kesim.build(im, 160, np.array([(243, 241, 236), (27, 27, 29), (119, 75, 43), (13, 143, 151)]), colours=c, to=to)
    b = kesim.build(im, 160, np.array([(20, 20, 20), (240, 220, 0), (10, 90, 200), (220, 0, 0)]), colours=c, to=to)
    assert len(a["polys"]) == len(b["polys"]) and all(p.equals(q) for p, q in zip(a["polys"], b["polys"]))
    assert set(a["colour"].tolist()) == {0, 1, 2, 3}                  # eyes and ornaments became stones
