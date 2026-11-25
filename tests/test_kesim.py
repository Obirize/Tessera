"""Cut-first stones (R8): one colour per stone, details stay, every stone is its own printable body."""
import io
import json
import os
import threading

import numpy as np
import pytest
import trimesh
from PIL import Image, ImageDraw

from tessera import app2_engine as E
from tessera.export.api import estimate, scene_glb
from tessera.geometry.stones import build_stones, stone_arrays
from tessera.model import Filament, Settings
from tessera.pipeline import build_mosaic
from tessera.pipeline.kesim import LAYOUTS
from tessera.render import render_cut

PAL = [Filament("Siyah", (20, 20, 22)), Filament("Kırmızı", (200, 40, 34)), Filament("Beyaz", (245, 245, 243))]


def face(w=500, h=600):
    """White head on red, with a black eye, a 1.6 mm lip line and a ring (hole) at 200 mm wide."""
    im = Image.new("RGB", (w * 2, h * 2), (196, 30, 28))
    d = ImageDraw.Draw(im)
    d.ellipse((200, 200, 800, 1000), fill=(248, 246, 242))
    d.ellipse((330, 420, 400, 460), fill=(15, 15, 15))
    d.line([(420, 800), (600, 800)], fill=(15, 15, 15), width=8)
    d.ellipse((560, 400, 680, 520), outline=(15, 15, 15), width=16)
    return np.asarray(im.resize((w, h), Image.LANCZOS))


def make(style="mozaik", mode="parca", **kw):
    s = Settings(width_mm=200.0, palette=list(PAL), mode=mode, style=style, mm_per_px=0.4, **kw)
    return build_mosaic(face(), s)


@pytest.fixture(scope="module")
def mosaics():
    return {st: make(st) for st in LAYOUTS}


@pytest.mark.parametrize("style", list(LAYOUTS))
def test_every_stone_is_one_separate_watertight_body(mosaics, style):
    m = mosaics[style]
    L = build_stones(m)
    assert len(L.stones) == len(m.polys) == m.n_tiles                  # nothing merged, nothing dropped
    assert len({s.code for s in L.stones}) == len(L.stones)
    for st in L.stones:
        t = trimesh.Trimesh(*stone_arrays(st), process=False)
        assert t.is_watertight and t.is_winding_consistent and t.body_count == 1, st.code
        assert 0.6 * st.area_mm2 * st.plane[0] < t.volume <= 1.001 * st.area_mm2 * st.plane[0], st.code


def test_stones_never_overlap_and_keep_a_joint(mosaics):
    import shapely
    m = mosaics["mozaik"]
    tree = shapely.STRtree(m.polys)
    a, b = tree.query(m.polys, predicate="dwithin", distance=0.05)
    assert (a != b).sum() == 0


def test_details_survive_in_their_own_colour(mosaics):
    import shapely
    for style, m in mosaics.items():
        black = shapely.union_all([p for p, c in zip(m.polys, m.tile_colour) if c == 0])
        white = shapely.union_all([p for p, c in zip(m.polys, m.tile_colour) if c == 2])
        f = 200 / 1000
        assert black.contains(shapely.Point(365 * f, 440 * f)), style             # eye
        assert black.buffer(0.3).contains(shapely.Point(510 * f, 800 * f)), style   # lip line
        assert black.buffer(0.3).contains(shapely.Point(568 * f, 460 * f)), style   # ring
        assert white.buffer(0.3).contains(shapely.Point(620 * f, 460 * f)), style   # inside the ring


def test_no_colour_in_between(mosaics):
    """Blended edge pixels must not become stones of a third colour."""
    import shapely
    m = mosaics["mozaik"]
    head = shapely.Point(100, 120).buffer(1)
    head = shapely.affinity.scale(head, 60, 80).boundary.buffer(1.0)
    assert not any(c == 0 and p.intersects(head) for p, c in zip(m.polys, m.tile_colour))


def test_ams_prints_small_shapes_into_the_stones():
    import shapely

    def touching(m):              # neighbouring pieces of different colours without a joint between them
        a, b = shapely.STRtree(m.polys).query(m.polys, predicate="dwithin", distance=0.05)
        return int((m.tile_colour[a] != m.tile_colour[b]).sum())
    assert touching(make()) == 0                       # single colour: a colour change is always a cut
    assert touching(make(mode="ams")) > 0              # AMS: the eye is printed into its stone


def test_height_settings():
    flat = make()
    assert np.ptp(flat.tile_plane[:, 0]) == 0                          # even surface by default
    rough = make(h_jitter_mm=0.8, stone_h_mm=4.0)
    z = rough.tile_plane[:, 0]
    assert z.min() >= 4.0 and 0.3 < np.ptp(z) <= 0.8 + 1e-6
    assert np.allclose(np.round(z / 0.2) * 0.2, z)                     # whole printing layers


def test_deterministic_and_seed(mosaics):
    a, b = mosaics["mozaik"], make()
    assert a.n_tiles == b.n_tiles and np.array_equal(a.tile_colour, b.tile_colour)
    assert make(seed=99).n_tiles > 0


def test_scene_estimate_and_preview(mosaics):
    m = mosaics["mozaik"]
    glb, meta = scene_glb(m, None, "preview")
    sc = trimesh.load(io.BytesIO(glb), file_type="glb")
    assert sum(n.startswith("stone:") for n in sc.graph.nodes_geometry) == m.n_tiles == meta["counts"]["parts"]
    assert estimate(m)["parca"] == m.n_tiles
    assert render_cut(m, 400).shape[1] == 400


def test_preview_is_export_stone_for_stone():
    r = E.Recipe(image_path=os.path.join("ornekler", "portre", "kaynak.png"), width_mm=200,
                 palette=["Siyah", "Kırmızı", "Beyaz"])
    assert r.settings(True).mm_per_px == r.settings(False).mm_per_px
    s = r.settings()
    assert (s.style, s.gap_mm, s.backing, s.merge_same_colour) == ("mozaik", 0.4, "yok", False)


def test_opening_card_comes_before_the_window():
    """The window must not exist on screen until the animation has ended."""
    src = open(os.path.join("tessera", "app2.py"), encoding="utf-8").read()
    body = src[src.index("def reveal("):]
    assert body.index("card.done.is_set()") < body.index("win.show_initial()")
    assert "-m tessera.baslat" in open("Tessera.bat", encoding="utf-8").read()
    from tessera import acilis
    with open(os.path.join(acilis.DIR, "taslar.json"), encoding="utf-8") as fh:
        order = acilis.schedule(json.load(fh)["stones"])
    eyes, rest = [s for s in order if s["eye"]], [s for s in order if not s["eye"]]
    assert eyes and max(s["t0"] for s in eyes) < min(s["t0"] for s in rest)            # eyes first
    assert min(s["t0"] for s in rest) < min(s["t0"] + s["dur"] for s in eyes)          # and no pause after them
    turn = [s["turn"] for s in rest]
    assert turn == sorted(turn) and turn[0] < 20 and turn[-1] > 330                    # one round, clockwise from the top
    gaps = {round(b["t0"] - a["t0"]) for a, b in zip(rest, rest[1:])}
    assert len(gaps) == 1                                                              # at an even pace
    assert acilis.total_ms(order) < 3500
    h = acilis.Handle.__new__(acilis.Handle)                                          # a dead card never blocks
    h.done = threading.Event()
    h.p = type("P", (), {"stdout": iter([])})()
    h._read()
    assert h.done.is_set()
    html = open(os.path.join("tessera", "web", "index.html"), encoding="utf-8").read()
    assert 'id="splash"' not in html and "Gelişmiş ayarlar" in html and "İnce ayar" not in html
    assert json.dumps([c["name"] for c in E.style_cards()], ensure_ascii=False) == '["Mozaik", "Kırık taş", "Blok", "İri parça"]'


def test_plate_count_on_screen_is_the_real_one(tmp_path):
    from tessera.export.api import export_project, plates_glb
    m = make()
    for order in ("panel", "renk"):
        m.settings.plate_order = order
        e = estimate(m)
        real = plates_glb(m)[1]["plates"]
        assert e["tabla"] == len(real) and e["tabla_kesin"]
        assert sum(p["hours"] for p in real) == pytest.approx(e["saat"], abs=0.4)
    res = export_project(m, str(tmp_path / "o"))
    assert len(res["plates"]) == e["tabla"] and res["hours"] == pytest.approx(e["saat"], abs=0.06)
    rough = estimate(m, exact=False)                                    # the card that is not selected: marked as rough
    assert not rough["tabla_kesin"] and abs(rough["tabla"] - e["tabla"]) <= 1


def test_remembered_colour_areas_change_nothing():
    """Another texture starts from the remembered areas; the stones must be the ones a fresh run gives."""
    from tessera.pipeline import kesim
    im, pal = face(), np.array([f.rgb for f in PAL])
    kesim.build(im, 150, pal, layout="mozaik")
    again = kesim.build(im, 150, pal, layout="blok")
    kesim._AREAS.clear()
    fresh = kesim.build(im, 150, pal, layout="blok")
    assert [p.wkb for p in again["polys"]] == [p.wkb for p in fresh["polys"]]
    other = kesim.build(im[:, ::-1].copy(), 150, pal, layout="blok")             # another picture is not mixed up
    assert [p.wkb for p in other["polys"]] != [p.wkb for p in fresh["polys"]]
