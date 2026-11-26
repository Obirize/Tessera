"""Crop and outline: the stones of a circle, and everything that comes out of it (scene, plates, export,
panels). Split from test_kadraj.py so that these builds run beside the others; the checks are the same."""
import os

import pytest
import shapely

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tessera import app2_engine as E  # noqa: E402
from tessera.export.api import estimate, export_project, plates_glb, scene_glb  # noqa: E402
from tessera.geometry.plate import panel_count  # noqa: E402
from tessera.geometry.stones import build_stones  # noqa: E402
from tessera.model import Mode  # noqa: E402

from test_kadraj import CAT, make  # noqa: E402


@pytest.fixture(scope="module")
def circle():
    return make(CAT, 150.0, {"tur": "daire"})[1]


def test_every_stone_lies_inside_the_outline(circle):
    edge = circle.outline.buffer(-circle.settings.gap_mm / 2 + .01)
    assert all(edge.covers(p) for p in circle.polys)
    assert min(p.area for p in circle.polys) >= 1.5
    filled = sum(p.area for p in circle.polys)
    assert filled > .84 * circle.outline.area                   # nothing is missing but the joints
    rim = shapely.union_all(circle.polys).buffer(.5)
    assert rim.covers(circle.outline.buffer(-1.5))


# ------------------------------------------------------------------ what comes out
def test_circle_goes_through_everything(circle, tmp_path):
    L = build_stones(circle)
    assert len(L.panels) == panel_count(circle) == 1
    _, meta = scene_glb(circle)
    _, pm = plates_glb(circle)
    e = estimate(circle)
    assert len(pm["plates"]) == e["tabla"]
    assert e["kg_m2"] == pytest.approx(e["gram"] / 1000 / (circle.outline.area / 1e6), rel=.05)
    res = export_project(circle, str(tmp_path), lambda f, m: True)
    assert not res["kapali_olmayan_govde"] and len(res["plates"]) == e["tabla"]


def test_big_circle_panels():
    r, m = make(CAT, 520.0, {"tur": "daire"}, parca=200, tek_taban=E.main_hex("Siyah"))
    L = build_stones(m)
    assert len(L.panels) == panel_count(m) == 9
    assert sum(pn.region.area for pn in L.panels) == pytest.approx(m.outline.area, rel=.01)
    r.mode = Mode.AMS
    m = E.build_mosaic_for(r)[0]
    L = build_stones(m)
    for pn in L.panels:                                         # no stone stands beside its panel's base
        reg = pn.region.buffer(.3)
        assert all(reg.covers(L.stones[i].poly) for i in pn.stones)
