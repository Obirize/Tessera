"""Round 6 export: every stone a separate printed piece (R6-1/R6-2).

Called by project.export when settings.merge_same_colour is False (the default).
Output:
  00_sablon_1-1_A4.pdf, 00_sablon_1-1_A3.pdf   1:1 paper template (assembly)
  01_montaj_kilavuzu.pdf (+ PNG)               overview + how to glue / hang
  02_tablalar/<n>_<renk>_tablalar.3mf          single-colour plates, ordered by template page
  02_tablalar/AMS_paneller.3mf                 AMS: one panel per plate (thin base + stones)
  02_tablalar/0_Kafes_<renk>.3mf               only with backing="kafes"
  02_tablalar/0_Taban_<renk>.3mf               only with a one-piece base plate
  02_tablalar/onizleme/*.png                   plate maps with stone codes
  03_stl/<panel>/<code>.stl                    fallback
  rapor.txt / rapor.json
"""
from __future__ import annotations

import math
import os
import time
from types import SimpleNamespace

import numpy as np
import shapely
from manifold3d import Manifold
from PIL import Image, ImageDraw
from shapely import affinity

from ..geometry.layout import FINISH_TR, GeoParams, ascii_name
from ..geometry.mesh import build_backing, clean_mesh, extrude, french_cleat
from ..geometry.packing import pack_outlines
from ..geometry.plate import build_plate
from ..geometry.stones import StoneLayout, build_stones, side_transform, stone_arrays, stone_solid
from ..geometry.text import ui_font
from ..model import Mode, Mosaic
from .bambu3mf import MAX_PLATES, FilamentSpec, Obj, Volume, write_bambu_3mf
from .mass import mass_model, split_over_plates
from .project import write_stl

AMS_BASE_T = 0.8          # thin base under the stones in AMS mode


def _rt(v, ang, dx, dy):
    a = math.radians(ang)
    c, s = math.cos(a), math.sin(a)
    out = v.copy()
    out[:, 0] = v[:, 0] * c - v[:, 1] * s + dx
    out[:, 1] = v[:, 0] * s + v[:, 1] * c + dy
    return out


def printed_stone(L: StoneLayout, st, engrave=True):
    """Mesh as it lies on the bed + its bed footprint outline + engraved text.
    Freud blocks with a prismatic footprint lie on their side (curved face = smooth wall)."""
    man, eng = stone_solid(L, st, engrave=engrave and not st.side)
    v, t, ok = clean_mesh(man)
    if st.side:
        M = side_transform(st)
        v = v @ M[:3, :3].T
        v[:, 2] -= v[:, 2].min()
        foot = shapely.MultiPoint(v[:, :2]).convex_hull
    else:
        foot = st.poly
    return v, t, ok, foot, eng


def stone_plate_image(items, rgb, title, plate_mm=256.0, px=900):
    head = 44
    im = Image.new("RGB", (px, px + head), (255, 255, 255))
    d = ImageDraw.Draw(im)
    k = px / plate_mm
    d.rectangle((0, head, px - 1, head + px - 1), fill=(46, 46, 50))
    d.rectangle((0, head + px - 28 * k, 18 * k, head + px - 1), fill=(90, 60, 60))
    d.text((10, 8), title, font=ui_font(24), fill=(20, 20, 20))
    lum = 0.30 * rgb[0] + 0.59 * rgb[1] + 0.11 * rgb[2]
    ink = (20, 20, 20) if lum > 110 else (245, 245, 245)

    def P(x, y):
        return (x * k, head + px - y * k)
    for g, code, *own in items:
        for p in getattr(g, "geoms", [g]):
            if p.geom_type == "Polygon" and not p.is_empty:
                d.polygon([P(x, y) for x, y in p.exterior.coords], fill=tuple(own[0] if own else rgb), outline=(120, 120, 120))
    f = ui_font(11)
    for g, code, *_ in items:
        c = g.representative_point()
        lab = code.split("-")[-1]
        d.text(P(c.x, c.y), lab, font=f, fill=ink, anchor="mm")
    return im


def export_stones(m: Mosaic, out_dir: str, prog, created: list) -> dict:
    t0 = time.time()
    s = m.settings
    mode = s.mode if s.mode in (Mode.PIECES, Mode.AMS) else Mode.PIECES
    pieces = mode == Mode.PIECES
    kafes = pieces and getattr(s, "backing", "yok") == "kafes"
    taban = pieces and m.plate is not None
    plates_dir = os.path.join(out_dir, "02_tablalar")
    img_dir = os.path.join(plates_dir, "onizleme")
    stl_dir = os.path.join(out_dir, "03_stl")
    import shutil
    for d in (plates_dir, stl_dir):
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        created.append(d)
    os.makedirs(img_dir, exist_ok=True)
    prog(0.01, "Taşlar hesaplanıyor")
    L = build_stones(m, progress=lambda f, msg: prog(0.01 + 0.09 * f, msg))
    gp = L.params
    pal = s.palette
    names = [ascii_name(f.name) for f in pal]
    fil = [FilamentSpec(f.name, f.hex, f.finish) for f in pal]
    mm_ = mass_model(m, mode)
    warnings = list(L.warnings)
    files, plates_info, plate_images = [], [], []
    w_mm, h_mm = m.size_mm

    # --- meshes -------------------------------------------------------------------------
    prog(0.1, "Taşların 3B modelleri oluşturuluyor")
    meshes = {}
    not_tight = 0
    n = len(L.stones)
    for k, st in enumerate(L.stones):
        if pieces:
            v, t, ok, foot, eng = printed_stone(L, st)
        else:
            man, eng = stone_solid(L, st, engrave=False)
            v, t, ok = clean_mesh(man)
            foot = st.poly
        meshes[st.index] = (v, t, foot, eng)
        not_tight += not ok
        if k % 50 == 0:
            prog(0.1 + 0.55 * (k + 1) / max(n, 1), f"Taş {k + 1}/{n}")
    vol = {i: abs(_vol(v, t)) for i, (v, t, _, _) in meshes.items()}

    order = getattr(s, "plate_order", "panel")
    plate_of = {}
    base_tight = 0
    if pieces:
        if taban:
            prog(0.64, "Taban hazırlanıyor")
            base_tight = _base_plates(m, L, s, names, fil, plates_dir, img_dir, stl_dir, files, plates_info, plate_images)
        prog(0.66, "Tablalara diziliyor")
        for ci in range(len(pal)):
            items = [st for st in L.stones if st.colour == ci]
            if not items:
                continue
            items.sort(key=lambda st: (st.page, st.panel, st.number))
            outl = [meshes[st.index][2] for st in items]
            groups = [st.page for st in items] if order not in ("max", "renk") else None
            placements, n_pl, unplaced = pack_outlines(outl, s.plate_mm, gp.plate_margin, 2.5, groups=groups)
            for i in unplaced:
                warnings.append(f"{items[i].code} tablaya sığmadı")
            n_files = max(1, math.ceil(n_pl / MAX_PLATES))
            fname = [f"{ci + 1}_{names[ci]}_tablalar{f'_{fi + 1}' if n_files > 1 else ''}.3mf" for fi in range(n_files)]
            objs, geo = [], [[] for _ in range(n_pl)]
            pvol = [0.0] * n_pl
            ppages = [set() for _ in range(n_pl)]
            for q in placements:
                st = items[q.item]
                v, t, foot, eng = meshes[st.index]
                vv = _rt(v, q.angle, q.dx, q.dy)
                objs.append((q.plate, Obj(st.code, [Volume(st.code, vv, t, 1)], q.plate)))
                geo[q.plate].append((affinity.translate(affinity.rotate(foot, q.angle, origin=(0, 0)), q.dx, q.dy),
                                     st.code))
                pvol[q.plate] += vol[st.index]
                ppages[q.plate].add(st.page)
                plate_of[st.index] = (fname[q.plate // MAX_PLATES], q.plate % MAX_PLATES + 1)
            for fi in range(n_files):
                lo, hi = fi * MAX_PLATES, min(n_pl, (fi + 1) * MAX_PLATES)
                sub = []
                for plate, ob in objs:
                    if lo <= plate < hi:
                        ob.plate = plate - lo
                        sub.append(ob)
                path = os.path.join(plates_dir, fname[fi])
                pn = [f"{pal[ci].name} {p - lo + 1} (şablon {_pages(ppages[p], L)})" for p in range(lo, hi)]
                write_bambu_3mf(path, sub, hi - lo, [fil[ci]], plate_names=pn, prime_tower=False,
                                title=f"Tessera {pal[ci].name}")
                files.append(path)
            for p in range(n_pl):
                fi = p // MAX_PLATES
                title = f"{pal[ci].name} · tabla {p - fi * MAX_PLATES + 1} · şablon {_pages(ppages[p], L)}"
                img = os.path.join(img_dir, f"{fname[fi][:-4]}_{p - fi * MAX_PLATES + 1:02d}.png")
                stone_plate_image(geo[p], pal[ci].rgb, title, s.plate_mm).save(img, optimize=True)
                plate_images.append(img)
                plates_info.append({"dosya": fname[fi], "renk": pal[ci].name, "tabla": p - fi * MAX_PLATES + 1,
                                    "parca": len(geo[p]), "sablon": sorted(ppages[p]), "panel": [],
                                    "tur": "parca", "_vol": pvol[p], "gorsel": img})
        if kafes:
            _kafes_plates(m, L, s, gp, pal, names, fil, plates_dir, img_dir, files, plates_info, plate_images)
    else:
        prog(0.66, "Çok renkli paneller hazırlanıyor")
        base_ci = L.base_colour
        objs, wt, pnames, no_tower = [], [], [], 0
        for k, pn in enumerate(L.panels):
            x0, y0, x1, y1 = pn.bounds
            w, h = x1 - x0, y1 - y0
            dx = s.plate_mm - gp.plate_margin - w - x0
            dy = s.plate_mm - gp.plate_margin - h - y0
            room = max(s.plate_mm - 2 * gp.plate_margin - h, s.plate_mm - 2 * gp.plate_margin - w)
            no_tower += room < 45
            wt.append((max(30.0, (s.plate_mm - 35) / 2), gp.plate_margin + 2.0)
                      if s.plate_mm - h >= s.plate_mm - w else (gp.plate_margin + 2.0, (s.plate_mm - 35) / 2))
            vols = []
            base = extrude(pn.region.buffer(-0.01), AMS_BASE_T).translate((dx, dy, 0))
            v, t, _ = clean_mesh(base)
            vols.append(Volume(f"Taban {pn.name}", v, t, base_ci + 1))
            per = {}
            for i in pn.stones:
                per.setdefault(L.stones[i].colour, []).append(i)
            for ci, lst in sorted(per.items()):
                V, T, off = [], [], 0
                for i in lst:
                    v, t, _, _ = meshes[i]
                    V.append(v + (dx, dy, AMS_BASE_T - 0.01))
                    T.append(t + off)
                    off += len(v)
                vols.append(Volume(f"{pal[ci].name} {pn.name}", np.vstack(V), np.vstack(T), ci + 1))
            objs.append(Obj(f"Panel {pn.name}", vols, k))
            pnames.append(f"Panel {pn.name}")
            img = os.path.join(img_dir, f"AMS_{pn.name}.png")
            items = [(affinity.translate(L.stones[i].poly, dx, dy), L.stones[i].code) for i in pn.stones]
            stone_plate_image(items, (200, 200, 200), f"Çok renkli panel {pn.name}", s.plate_mm).save(img)
            plate_images.append(img)
            plates_info.append({"dosya": "AMS_paneller.3mf", "renk": "çok renkli", "tabla": k + 1,
                                "panel": [pn.name], "parca": 1, "renk_bolgesi": len(pn.stones), "tur": "ams",
                                "_vol": sum(vol[i] for i in pn.stones) + pn.region.area * AMS_BASE_T,
                                "gorsel": img})
        if no_tower:
            warnings.append("Paneller tablayı doldurduğu için temizleme kulesi kapalı: renk değişiminde atık "
                            "filament arkadaki oluktan atılır. Renkler birbirine biraz bulaşabilir; ilk paneli "
                            "deneme olarak bas")
        for fi in range(max(1, math.ceil(len(objs) / MAX_PLATES))):
            lo, hi = fi * MAX_PLATES, min(len(objs), (fi + 1) * MAX_PLATES)
            sub = objs[lo:hi]
            for o in sub:
                o.plate -= lo
            path = os.path.join(plates_dir, f"AMS_paneller{f'_{fi + 1}' if len(objs) > MAX_PLATES else ''}.3mf")
            write_bambu_3mf(path, sub, hi - lo, fil, plate_names=pnames[lo:hi], wipe_tower_xy=wt[lo:hi],
                            prime_tower=not no_tower, title="Tessera çok renkli")
            files.append(path)
    if pieces and not kafes:
        mm_ = mass_model(m, mode, len(plates_info))      # now the plates are counted, not guessed
    # split model totals over plates
    shares = split_over_plates(mm_, [p["_vol"] for p in plates_info], [p["tur"] == "taban" for p in plates_info])
    for p, (g, hh) in zip(plates_info, shares):
        p["gram"], p["saat"] = round(g), round(hh, 1)
        p.pop("_vol", None)

    # --- STL fallback ---------------------------------------------------------------------
    prog(0.86, "Yedek STL dosyaları yazılıyor")
    for st in L.stones:
        v, t, _, _ = meshes[st.index]
        write_stl(os.path.join(stl_dir, f"P{st.panel + 1}", f"{st.code}.stl"), v, t)

    # --- template + guide + report ----------------------------------------------------------
    prog(0.9, "1:1 şablon çiziliyor")
    from .template import write_template
    for paper in ("A4", "A3"):
        files.append(write_template(L, out_dir, paper))
    prog(0.95, "Montaj kılavuzu çiziliyor")
    from .guide import write_guide_stones
    files.extend(write_guide_stones(L, out_dir, mode, plates_info))
    if base_tight:
        warnings.append(f"{base_tight} taban parçasının modeli tam kapalı değil (Bambu Studio onarır)")
    if not_tight:
        warnings.append(f"{not_tight} taşın STL'i tam kapalı değil (Bambu Studio onarır)")
    n_eng = sum(1 for v in meshes.values() if v[3])
    res = {
        "mod": mode, "files": files, "plates": plates_info, "plate_images": plate_images,
        "parts": len(L.stones) if pieces else len(L.panels), "montaj_adedi": len(L.stones),
        "tas": len(L.stones), "parca_tanimi": "Ayrı basılan taş sayısı" if pieces else
        "Çok renkli panel sayısı (taşlar panelin üstünde basılır)",
        "renk_bolgesi": len(L.stones), "hassas_parca": 0, "panels": len(L.panels), "panel_xy": list(L.grid),
        "grams": round(mm_["grams"]), "hours": round(mm_["hours"], 1), "kg_m2": round(mm_["kg_m2"], 2),
        "parca_gram": round(mm_["parts_g"]), "kafes_gram": round(mm_["backing_g"]), "warnings": warnings,
        "renk_gram": {pal[i].name: round(g + (mm_["backing_g"] if taban and i == m.plate_colour else 0))
                      for i, g in enumerate(mm_["colour_g"])},
        "renk_parca": {f.name: sum(1 for st in L.stones if st.colour == i) for i, f in enumerate(pal)},
        "kafes_renk": pal[L.backing_colour].name if kafes else None,
        "taban_renk": pal[L.base_colour].name if not pieces else pal[m.plate_colour].name if taban else None,
        "numarasiz_parca": sum(1 for v in meshes.values() if not v[3]) if pieces else 0,
        "kodlu_tas": n_eng, "yan_basilan": sum(1 for st in L.stones if st.side and pieces),
        "purge": ({"degisim": mm_["swaps"], "purge_gram": round(mm_["purge_g"]),
                   "saat": round(mm_["swaps"] * 78.6 / 3600, 1), "paneller": []} if not pieces else None),
        "tabla_sirasi": "sablon" if order not in ("max", "renk") else "renk",
        "sablon_sayfa": list(L.pages),
        "panel_boyutlari_mm": [[round(v) for v in pn.size] for pn in L.panels],
        "kapali_olmayan_govde": not_tight, "istatistik": L.stats,
        "sure_sn": round(time.time() - t0, 1),
    }
    from .report import write_report_stones
    files.extend(write_report_stones(L, out_dir, res))
    prog(1.0, "Dışa aktarım tamamlandı")
    return res


def _pages(pages, L):
    cols = L.pages[0]
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return ", ".join(f"{letters[(p // cols) % 26]}{p % cols + 1}" for p in sorted(pages))


def _vol(v, t):
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _base_plates(m, L, s, names, fil, plates_dir, img_dir, stl_dir, files, plates_info, plate_images) -> int:
    """The one-piece base plate, one panel per build plate. Returns how many are not watertight."""
    ci = m.plate_colour
    f = s.palette[ci]
    objs, title, bad = [], [], 0
    fname = f"0_Taban_{names[ci]}"
    n_files = max(1, math.ceil(len(L.panels) / MAX_PLATES))
    for k, pn in enumerate(L.panels):
        v, t, ok = clean_mesh(build_plate(L, pn, numbers=True))
        bad += not ok
        x0, y0, x1, y1 = pn.bounds
        dx, dy = (s.plate_mm - x1 - x0) / 2, (s.plate_mm - y1 - y0) / 2
        nm = f"Taban {pn.name}"
        objs.append(Obj(nm, [Volume(nm, v + (dx, dy, 0.0), t, 1)], k % MAX_PLATES))
        title.append(nm)
        write_stl(os.path.join(stl_dir, f"taban_{pn.name}.stl"), v, t)
        img = os.path.join(img_dir, f"{fname}_{pn.name}.png")
        here = [st for st in L.stones if pn.region.intersects(st.poly)]
        items = [(pn.region, "", f.rgb)] + [(st.poly.intersection(pn.region), f"{st.number}" if st.panel == pn.index else "",
                                             (236, 232, 224)) for st in here]
        stone_plate_image([(affinity.translate(g, dx, dy), c, rgb) for g, c, rgb in items], (255, 255, 255), nm,
                          s.plate_mm).save(img, optimize=True)
        plate_images.append(img)
        plates_info.append({"dosya": f"{fname}{f'_{k // MAX_PLATES + 1}' if n_files > 1 else ''}.3mf", "renk": f.name,
                            "tabla": k % MAX_PLATES + 1, "parca": 1, "tur": "taban", "panel": [pn.name],
                            "_vol": abs(_vol(v, t)), "gorsel": img})
    for i in range(n_files):
        lo, hi = i * MAX_PLATES, min(len(objs), (i + 1) * MAX_PLATES)
        path = os.path.join(plates_dir, f"{fname}{f'_{i + 1}' if n_files > 1 else ''}.3mf")
        write_bambu_3mf(path, objs[lo:hi], hi - lo, [fil[ci]], plate_names=title[lo:hi], prime_tower=False,
                        title="Tessera taban")
        files.append(path)
    return bad


def _kafes_plates(m, L, s, gp, pal, names, fil, plates_dir, img_dir, files, plates_info, plate_images):
    """Optional lattice backing (backing="kafes"): the stones' footprints are the pockets."""
    parts = [SimpleNamespace(outline=st.poly, bridges=None, number=st.number) for st in L.stones]
    adapter = SimpleNamespace(params=gp, mosaic=m, parts=parts, tile_poly=[], tile_plane=None)
    objs, pn_names = [], []
    bci = L.backing_colour
    for k, pn in enumerate(L.panels):
        panel = SimpleNamespace(region=pn.region, parts=pn.stones, fused=[], name=pn.name)
        bk = build_backing(adapter, panel, engrave=True, label_pockets=True)
        v, t, _ = clean_mesh(bk)
        x0, y0, x1, y1 = pn.bounds
        off = ((s.plate_mm - (x1 - x0)) / 2 - x0, (s.plate_mm - (y1 - y0)) / 2 - y0, 0.0)
        objs.append(Obj(f"Kafes {pn.name}", [Volume(f"Kafes {pn.name}", v + off, t, 1)], k))
        pn_names.append(f"Kafes {pn.name}")
        plates_info.append({"dosya": f"0_Kafes_{names[bci]}.3mf", "renk": pal[bci].name, "tabla": k + 1,
                            "parca": 1, "tur": "kafes", "panel": [pn.name], "_vol": bk.volume(), "gorsel": ""})
    cl = 200.0
    cv, ct, _ = clean_mesh(french_cleat(min(cl, s.plate_mm - 16)))
    kc = len(objs)
    for j, yy in enumerate((40.0, 120.0)):
        objs.append(Obj(f"Askı çıtası {j + 1}", [Volume("Askı", cv + (8.0, yy, 0), ct, 1)], kc))
    pn_names.append("Duvar askısı")
    plates_info.append({"dosya": f"0_Kafes_{names[bci]}.3mf", "renk": pal[bci].name, "tabla": kc + 1, "parca": 2,
                        "tur": "aski", "panel": [], "_vol": 2 * 157.5 * 200 * 0.55, "gorsel": ""})
    path = os.path.join(plates_dir, f"0_Kafes_{names[bci]}.3mf")
    write_bambu_3mf(path, objs, kc + 1, [fil[bci]], plate_names=pn_names, prime_tower=False, title="Tessera kafes")
    files.append(path)
