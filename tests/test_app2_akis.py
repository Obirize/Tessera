"""Tessera v2 interface: the bridge with real pictures (advice, templates, piece count) and the export entry
point. Split from test_app2.py so that these builds run beside the others; the checks are the same."""
from __future__ import annotations

import json
import os

from tessera import app2  # noqa: E402
from tessera import app2_engine as E  # noqa: E402

from test_app2 import POSTER, Rec, bridge, qapp, spin, store  # noqa: F401


def test_advice_is_shown_and_the_kind_can_be_switched(qapp, bridge, tmp_path):
    import numpy as np
    from PIL import Image
    from tests.test_sade import photo
    p = str(tmp_path / "foto.png")
    Image.fromarray(photo()).save(p)
    rec = Rec(bridge)
    bridge.setWidth(20)
    bridge.openImage(p)
    assert spin(qapp, lambda: rec.got["built"], 120), rec.got["failed"]
    st = json.loads(bridge.getState())
    assert st["uygun"]["tur"] == "ton" and "sadeleştirildi" in " ".join(st["uygun"]["soz"])
    n = len(rec.got["built"])
    bridge.setKind("net")
    assert spin(qapp, lambda: len(rec.got["built"]) > n, 120)
    st = json.loads(bridge.getState())
    assert st["uygun"]["tur"] == "net" and st["fine"]["tur"] == "net"
    bridge.resetFine()
    assert json.loads(bridge.getState())["fine"]["tur"] == "net"      # fine settings' reset leaves it alone
    bridge.setKind("bilinmeyen")
    assert json.loads(bridge.getState())["fine"]["tur"] == "net"
    html = open(os.path.join(app2.WEB_DIR, "index.html"), encoding="utf-8").read()
    assert 'id="fit"' in html


def test_templates_are_offered_and_open(qapp, bridge):
    st = json.loads(bridge.getState())
    assert [t["name"] for t in st["templates"]] == ["Tessera kedisi", "Örnek afiş"] and "example" not in st
    for t in st["templates"]:
        assert bridge.blobs.get(t["url"].split("/data/")[1])                 # the little picture is really served
    cols, adv = E.analyse(st["templates"][0]["path"], 300)
    assert {c["ad"] for c in cols} == {"Siyah", "Beyaz", "Kırmızı", "Pembe", "Mavi", "Altın"} and adv["puan"] == "iyi"
    rec = Rec(bridge)
    bridge.openImage(st["templates"][0]["path"])
    assert bridge.display_name() == "Tessera kedisi" and not rec.got["failed"]
    html = open(os.path.join(app2.WEB_DIR, "index.html"), encoding="utf-8").read()
    assert "Hazır şablonlar" in html and "btn-example" not in html
    assert bridge.recipe.fine["parca"] == "sekil" and bridge.recipe.fine["tek_taban"] == E.main_hex("Siyah")
    assert bridge.recipe.width_mm == 200
    assert spin(qapp, lambda: rec.got["built"]) and json.loads(rec.got["built"][-1])["tiles"] == 36
    bridge.setWidth(30)
    bridge.openImage(st["templates"][1]["path"])                 # the next picture starts without them
    assert bridge.recipe.fine["parca"] is None and bridge.recipe.fine["tek_taban"] is None and bridge.recipe.width_mm == 300


def test_piece_count_and_plate_belong_to_the_project(qapp, bridge, tmp_path):
    rec = Rec(bridge)
    bridge.openImage(app2.TEMPLATES[0][1])
    assert spin(qapp, lambda: rec.got["built"])
    bridge.setFine(json.dumps({"parca": 50.0}))
    assert bridge.recipe.fine["parca"] == 50
    i = [c["ad"] for c in bridge.recipe.colours].index("Siyah")
    bridge.setColour(i, "Lacivert")
    assert bridge.recipe.fine["tek_taban"] == E.main_hex("Lacivert")        # the plate goes with its colour
    bridge.resetFine()
    assert bridge.recipe.fine["parca"] == 50 and bridge.recipe.fine["tek_taban"] == E.main_hex("Lacivert")
    p = bridge.save(str(tmp_path / "a.tessera"))
    bridge.setFine(json.dumps({"parca": None, "tek_taban": None}))
    assert bridge.open(p) and bridge.recipe.fine["parca"] == 50 and bridge.recipe.fine["tek_taban"] == E.main_hex("Lacivert")
    n = len(rec.got["built"])
    assert spin(qapp, lambda: len(rec.got["built"]) > n and json.loads(rec.got["built"][-1])["tiles"] == 50)


def test_export_small_artwork(tmp_path):
    """The child-process export entry point, run in-process on a small piece."""
    r = E.Recipe(image_path=POSTER, palette=["Siyah", "Kırmızı", "Beyaz"], style="Mozaik", width_mm=80)
    z = E.sibling_zones(POSTER)
    r.shapes, r.zones_from_file, r.file_default_vocab = z["sekiller"], True, z["varsayilan"]
    r.rays_center = z["isin_merkezi"]
    seen = []
    res = E.run_export({"recipe": r.to_payload(), "out": str(tmp_path / "out")}, lambda f, m="": seen.append(f))
    assert res.get("parts", 0) > 0 and seen and max(seen) <= 1.0
    assert any(p.endswith(".3mf") for p in os.listdir(tmp_path / "out" / "02_tablalar")) \
        if (tmp_path / "out" / "02_tablalar").exists() else res.get("files")
    assert (tmp_path / "out" / "00_sablon_1-1_A4.pdf").exists()        # R6-2: the paper template
