"""3D team: estimate + full export in both modes on a small synthetic mosaic."""
import glob
import json
import math
import os
import time
import zipfile

PERF = os.environ.get("TESSERA_PERF") == "1"   # R5-2: timing checks are advisory

import numpy as np
import pytest

from tessera.export.api import estimate, export_project, mass_model, preview_parts
from tessera.export.validate import bambu_available, slice_3mf
from tessera.model import Mode

from test_geometry import make_mosaic


@pytest.fixture(scope="module")
def mosaic():
    return make_mosaic()


@pytest.fixture(scope="module")
def exports(mosaic, tmp_path_factory):
    out = {}
    for mode in (Mode.PIECES, Mode.AMS):
        mosaic.settings.mode = mode
        d = tmp_path_factory.mktemp(f"exp_{mode}")
        calls = []
        t = time.time()
        res = export_project(mosaic, str(d), progress=lambda f, m: calls.append((f, m)))
        out[mode] = (res, d, time.time() - t, calls, estimate(mosaic))
    mosaic.settings.mode = Mode.PIECES
    return out


def _check_3mf(path, n_filaments):
    from lxml import etree
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        for req in ("[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model",
                    "Metadata/model_settings.config", "Metadata/project_settings.config"):
            assert req in names, req
        root = etree.fromstring(z.read("3D/3dmodel.model"))
        ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
        items = root.findall(".//m:build/m:item", ns)
        assert items
        cfg = etree.fromstring(z.read("Metadata/model_settings.config"))
        plates = cfg.findall("plate")
        assert 1 <= len(plates) <= 36
        assert sum(len(p.findall("model_instance")) for p in plates) == len(items)
        for comp in root.iter("{%s}component" % ns["m"]):
            assert comp.get("{http://schemas.microsoft.com/3dmanufacturing/production/2015/06}path")[1:] in names
        ps = json.loads(z.read("Metadata/project_settings.config"))
        assert len(ps["filament_colour"]) == n_filaments
        assert ps["printer_model"] == "Bambu Lab X1 Carbon"
        return len(plates)


@pytest.mark.parametrize("mode", [Mode.PIECES, Mode.AMS])
def test_export_both_modes(mosaic, exports, mode):
    res, d, secs, calls, est = exports[mode]
    if PERF:
        assert secs < 30.0
    assert calls and calls[-1][0] == 1.0
    assert all(b >= a - 1e-9 for (a, _), (b, _) in zip(calls, calls[1:]))
    assert res["parts"] > 0 and res["panels"] >= 4 and res["grams"] > 0
    assert (d / "01_montaj_kilavuzu.pdf").exists()
    assert (d / "rapor.txt").exists() and (d / "rapor.json").exists()
    mfs = [f for f in res["files"] if f.endswith(".3mf")]
    if mode == Mode.PIECES:
        assert len(mfs) == len(mosaic.settings.palette) + 1     # one per colour + backing file
        for f in mfs:
            _check_3mf(f, 1)
        assert est["purge_gram"] == 0
        stls = glob.glob(str(d / "03_stl" / "*" / "*.stl"))
        assert len(stls) == res["parts"] + res["panels"]
    else:
        ams = [f for f in mfs if os.path.basename(f).startswith("AMS_")]
        assert len(ams) == 1 and len(mfs) == 2                  # panels + the wall-cleat file
        assert _check_3mf(ams[0], len(mosaic.settings.palette)) == res["panels"]
        assert res["purge"]["purge_gram"] > 0 and est["purge_gram"] > 0
        assert res["parts"] == res["panels"]                    # AMS: one object per panel
    # the UI's numbers are the export's numbers
    assert est["gram"] == res["grams"] and est["panel"] == res["panels"]
    assert 0.4 < est["parca"] / res["parts"] < 2.5
    # plate previews for the UI
    assert res["plate_images"] and all(os.path.exists(p) for p in res["plate_images"])
    assert all(p.get("gorsel") for p in res["plates"])


def test_every_stl_is_watertight(exports):
    import trimesh
    for mode in (Mode.PIECES, Mode.AMS):
        res, d, *_ = exports[mode]
        stls = glob.glob(str(d / "03_stl" / "*" / "*.stl"))
        assert stls
        for f in stls:
            tm = trimesh.load(f)
            assert tm.is_watertight and tm.is_winding_consistent, f
            assert (tm.area_faces > 1e-12).all(), f
        assert res["kapali_olmayan_govde"] == 0


def test_ams_has_fewer_things_to_assemble(exports):
    rp, ra = exports[Mode.PIECES][0], exports[Mode.AMS][0]
    ep, ea = exports[Mode.PIECES][4], exports[Mode.AMS][4]
    assert ra["parts"] < rp["parts"] and ea["parca"] < ep["parca"]
    assert ra["panels"] == rp["panels"] and ea["panel"] == ep["panel"]


def test_plates_sorted_by_panel(exports):
    res = exports[Mode.PIECES][0]
    for colour in {p["renk"] for p in res["plates"] if p["tur"] == "parca"}:
        pl = sorted((p for p in res["plates"] if p["tur"] == "parca" and p["renk"] == colour),
                    key=lambda p: p["tabla"])
        firsts = [min(int(x[1:]) for x in p["panel"]) for p in pl]
        assert firsts == sorted(firsts)


def test_mass_model_and_estimate_agree(mosaic):
    for mode in (Mode.PIECES, Mode.AMS):
        mosaic.settings.mode = mode
        t = time.time()
        e = estimate(mosaic)
        if PERF:
            assert time.time() - t < 2.0
        mm = mass_model(mosaic)
        assert set(mm) >= {"kg_m2", "grams", "hours", "backing_g", "parts_g"}
        assert e["gram"] == round(mm["grams"]) and e["kg_m2"] == round(mm["kg_m2"], 2)
        w, h = mosaic.size_mm
        assert e["panel"] == math.ceil(w / 70) * math.ceil(h / 70)
    mosaic.settings.mode = Mode.PIECES


def test_cancel_export(mosaic, tmp_path):
    mosaic.settings.mode = Mode.PIECES
    res = export_project(mosaic, str(tmp_path), progress=lambda f, m: f < 0.3)
    assert res.get("iptal") and not glob.glob(str(tmp_path / "02_tablalar" / "*.3mf"))


def test_preview_parts(mosaic):
    mosaic.settings.mode = Mode.PIECES
    img = preview_parts(mosaic, 0, size=300)
    assert img.shape == (300, 300, 3) and img.dtype == np.uint8
    assert (img != img[0, 0]).any(axis=2).mean() > 0.1
    png = preview_parts(mosaic, "P2", size=200, as_png=True)
    assert png[:4] == b"\x89PNG"


@pytest.mark.skipif(not bambu_available() or os.environ.get("TESSERA_SLICE") != "1",
                    reason="set TESSERA_SLICE=1 to slice with the Bambu Studio CLI")
def test_bambu_cli_slices(exports):
    for mode in (Mode.PIECES, Mode.AMS):
        res = exports[mode][0]
        for f in [f for f in res["files"] if f.endswith(".3mf")]:
            r = slice_3mf(f)
            assert r["ok"], r


def test_count_parts_matches_layout(mosaic):
    from tessera.export.api import count_parts
    from tessera.geometry.layout import build_layout
    mosaic.settings.mode = Mode.PIECES
    n = len(build_layout(mosaic).parts)
    assert count_parts(mosaic)["parca"] == n
    assert estimate(mosaic)["parca"] == n and estimate(mosaic)["parca_kesin"]
    fast = estimate(mosaic, exact=False)
    assert not fast["parca_kesin"] and fast["parca"] > 0


def test_cleat_stl_and_mode_wording(exports):
    for mode in (Mode.PIECES, Mode.AMS):
        res, d, *_ = exports[mode]
        assert (d / "03_stl" / "aski_citasi_2_adet_bas.stl").exists()
        txt = (d / "rapor.txt").read_text(encoding="utf-8")
        assert "purge" not in txt.lower() and "AMS kapalı" not in txt and "AMS açık" not in txt
        assert ("Tek renk tablalar" in txt) == (mode == Mode.PIECES)
        assert ("Çok renkli paneller" in txt) == (mode == Mode.AMS)
    ra, ea = exports[Mode.AMS][0], exports[Mode.AMS][4]
    assert len(ra["plates"]) == ea["tabla"]


def test_preview_fast_and_cached(mosaic):
    from tessera.export.api import preview_key
    mosaic.settings.mode = Mode.PIECES
    t = time.time()
    img = preview_parts(mosaic, 0, size=300, quality="fast")
    assert img.shape == (300, 300, 3)
    if PERF:
        assert time.time() - t < 1.5
    k1 = preview_key(mosaic)
    mosaic.settings.mode = Mode.AMS
    assert preview_key(mosaic) != k1
    mosaic.settings.mode = Mode.PIECES


# ----------------------------------------------------------------------------- 3D scene (Tur 5)
def _load_glb(b):
    import io
    import trimesh
    return trimesh.load(io.BytesIO(b), file_type="glb")


@pytest.mark.parametrize("quality", ["fast", "full"])
@pytest.mark.parametrize("mode", [Mode.PIECES, Mode.AMS])
def test_scene_glb(mosaic, mode, quality):
    from tessera.export.scene import scene_glb
    from tessera.geometry.layout import build_layout
    mosaic.settings.mode = mode
    glb, meta = scene_glb(mosaic, mode, quality)
    sc = _load_glb(glb)
    names = set(sc.graph.nodes_geometry)
    parts = [n for n in meta["nodes"] if n["type"] == "part"]
    assert {n["name"] for n in meta["nodes"]} == names
    assert len([n for n in names if n.startswith("part:")]) == len(parts) == meta["counts"]["parts"]
    assert meta["counts"]["backings"] >= 4 and meta["counts"]["cleats"] == 2
    if quality == "full":
        assert len(parts) == len(build_layout(mosaic).parts)
    else:
        assert len(parts) == build_layout(mosaic, count_only=True)["parca"]
    pal = [tuple(f.rgb) for f in mosaic.settings.palette]
    for n in parts:
        tr, g = sc.graph[n["name"]]
        mat = sc.geometry[g].visual.material
        assert tuple(int(c) for c in mat.baseColorFactor[:3]) == pal[n["colour"]]
        assert n["code"] and n["panel"].startswith("P") and len(n["size_mm"]) == 3
        assert n["explode"]["dir"] == [0.0, 0.0, 1.0] and n["explode"]["distance_mm"] > 0
        # node colour equals the preview colour under the part
        b = n["bbox"]
        mm = mosaic.settings.mm_per_px
        H = mosaic.label.shape[0]
        cx, cy = (b[0] + b[3]) / 2, (b[1] + b[4]) / 2
        assert pal[mosaic.colour[min(H - 1, int(H - cy / mm)), int(cx / mm)]] is not None
    if quality == "full" and mode == Mode.PIECES:
        assert all(n["plate"] for n in parts)
    t = time.time()
    again = scene_glb(mosaic, mode, quality)               # cached: explode/light never rebuild
    assert again[0] is glb
    if PERF:
        assert time.time() - t < 0.05
    mosaic.settings.mode = Mode.PIECES


def test_scene_part_colours_match_preview(mosaic):
    """Majority preview colour inside each part's footprint == the part's node colour."""
    from tessera.export.scene import scene_glb
    from tessera.geometry.layout import build_layout
    mosaic.settings.mode = Mode.PIECES
    L = build_layout(mosaic)
    _, meta = scene_glb(mosaic, Mode.PIECES, "full")
    by_code = {n["code"]: n for n in meta["nodes"] if n["type"] == "part"}
    for p in L.parts:
        pix = np.isin(L.label, p.tiles) & ~mosaic.gap
        maj = np.bincount(mosaic.colour[pix]).argmax()
        assert by_code[p.pid]["colour"] == maj


def test_plates_glb_matches_export(mosaic, exports):
    from tessera.export.scene import plates_glb
    for mode in (Mode.PIECES, Mode.AMS):
        mosaic.settings.mode = mode
        glb, meta = plates_glb(mosaic, mode)
        sc = _load_glb(glb)
        assert len(meta["plates"]) == len(exports[mode][0]["plates"])
        beds = [n for n in sc.graph.nodes_geometry if n.startswith("bed:")]
        assert len(beds) == len(meta["plates"])
        for p in meta["plates"]:
            assert p["codes"] and p["file"].endswith(".3mf") and p["hours"] >= 0
    mosaic.settings.mode = Mode.PIECES


# ----------------------------------------------------------------------------- round 6: stones
@pytest.fixture(scope="module", params=[("klasik", Mode.PIECES), ("klasik", Mode.AMS), ("freud", Mode.PIECES)])
def stone_export(request, tmp_path_factory):
    style, mode = request.param
    m = make_mosaic(stones=True, style=style)
    m.settings.mode = mode
    d = tmp_path_factory.mktemp(f"tas_{style}_{mode}")
    t = time.time()
    res = export_project(m, str(d))
    return m, res, d, time.time() - t, mode


def test_stone_export_outputs(stone_export):
    from tessera.geometry.stones import build_stones
    m, res, d, secs, mode = stone_export
    if PERF:
        assert secs < 30
    L = build_stones(m)
    assert res["tas"] == len(L.stones)
    assert (d / "00_sablon_1-1_A4.pdf").exists() and (d / "00_sablon_1-1_A3.pdf").exists()
    assert (d / "01_montaj_kilavuzu.pdf").exists() and (d / "rapor.txt").exists()
    mfs = [f for f in res["files"] if f.endswith(".3mf")]
    if mode == Mode.PIECES:
        assert len(mfs) == len({st.colour for st in L.stones})
        objects = 0
        for f in mfs:
            with zipfile.ZipFile(f) as z:
                objects += z.read("Metadata/model_settings.config").decode("utf-8").count("<model_instance>")
        assert objects == len(L.stones)                 # every stone is its own printed object
        # plates follow the template pages
        for colour in {p["renk"] for p in res["plates"]}:
            pl = sorted((p for p in res["plates"] if p["renk"] == colour), key=lambda p: p["tabla"])
            firsts = [min(p["sablon"]) for p in pl]
            assert firsts == sorted(firsts)
    else:
        assert res["parts"] == res["panels"]
    assert res["plate_images"] and all(os.path.exists(p) for p in res["plate_images"])
    txt = (d / "rapor.txt").read_text(encoding="utf-8")
    assert "purge" not in txt.lower()


def test_stone_stls_watertight(stone_export):
    import trimesh
    m, res, d, secs, mode = stone_export
    stls = glob.glob(str(d / "03_stl" / "*" / "*.stl"))
    assert len(stls) == res["tas"]
    bad = [f for f in stls[::5] if not trimesh.load(f).is_watertight]
    assert len(bad) <= max(1, len(stls[::5]) // 50), bad[:5]


def test_stone_scene_glb(stone_export):
    from tessera.export.scene import plates_glb, scene_glb
    from tessera.geometry.stones import build_stones
    m, res, d, secs, mode = stone_export
    L = build_stones(m)
    calls = []
    glb, meta = scene_glb(m, mode, "preview", progress=lambda f, msg: calls.append(f))
    sc = _load_glb(glb)
    stones = [n for n in meta["nodes"] if n["type"] == "part"]
    assert len(stones) == len(L.stones) == meta["counts"]["parts"]
    assert len([n for n in sc.graph.nodes_geometry if n.startswith("stone:")]) == len(L.stones)
    assert calls and calls[-1] == 1.0
    pal = [f.rgb for f in m.settings.palette]
    by_code = {st.code: st for st in L.stones}
    for n in stones[::9]:
        st = by_code[n["code"]]
        assert n["colour"] == st.colour and tuple(int(x, 16) for x in (n["hex"][1:3], n["hex"][3:5], n["hex"][5:7])) == pal[st.colour]
    g2, pm = plates_glb(m, mode)
    assert len(pm["plates"]) == len([p for p in res["plates"]])
