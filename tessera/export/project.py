"""Full production export: layout -> solids -> plates -> Bambu 3MF + STL + guide + report.

Totals (grams, hours, kg/m2) come from mass.mass_model (the same numbers the UI shows);
per-plate numbers split those totals by printed volume.
"""
from __future__ import annotations

import io
import math
import os
import shutil
import struct
import time
from typing import Callable, Optional

import numpy as np
import shapely
import shapely.ops
from manifold3d import Manifold
from PIL import Image, ImageDraw
from shapely import affinity

from ..geometry.layout import (FINISH_TR, GeoParams, Layout, ascii_name, backing_colour_index,  # noqa: F401
                               build_layout)
from ..geometry.mesh import build_backing, build_part, clean_mesh, french_cleat
from ..geometry.packing import pack_outlines
from ..geometry.text import ui_font
from ..model import Mode, Mosaic
from .bambu3mf import MAX_PLATES, FilamentSpec, Obj, Volume, write_bambu_3mf
from .mass import PLA_DENSITY, PURGE_G_PER_SWAP, SWAP_S, mass_model  # noqa: F401

PLATE_IMG_PX = 768


class ExportCancelled(Exception):
    pass


def mass_model_export(m: Mosaic) -> dict:     # backwards friendly name
    return mass_model(m)


def _filaments(m: Mosaic):
    return [FilamentSpec(f.name, f.hex, f.finish) for f in m.settings.palette]


def write_stl(path: str, v: np.ndarray, t: np.ndarray):
    tri = v[t].astype(np.float32)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = n
    rec["v"] = tri
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"Tessera STL".ljust(80, b" "))
        fh.write(struct.pack("<I", len(t)))
        fh.write(rec.tobytes())


def _plate_thumb(outlines, rgb, plate_mm=256.0, px=256) -> bytes:
    im = Image.new("RGB", (px, px), (40, 40, 44))
    d = ImageDraw.Draw(im)
    k = px / plate_mm
    for g in outlines:
        for p in getattr(g, "geoms", [g]):
            if p.geom_type != "Polygon" or p.is_empty:
                continue
            d.polygon([(x * k, px - y * k) for x, y in p.exterior.coords], fill=tuple(rgb))
            for r in p.interiors:
                d.polygon([(x * k, px - y * k) for x, y in r.coords], fill=(40, 40, 44))
    bio = io.BytesIO()
    im.save(bio, "PNG")
    return bio.getvalue()


def plate_image(items, rgb, title: str, plate_mm=256.0, px=PLATE_IMG_PX) -> Image.Image:
    """Top view of one plate: parts in filament colour with their codes ("tabla haritası")."""
    head = 44
    im = Image.new("RGB", (px, px + head), (255, 255, 255))
    d = ImageDraw.Draw(im)
    k = px / plate_mm
    d.rectangle((0, head, px - 1, head + px - 1), fill=(46, 46, 50))
    d.rectangle((0, head + px - 28 * k, 18 * k, head + px - 1), fill=(90, 60, 60))   # purge chute corner
    d.text((10, 8), title, font=ui_font(24), fill=(20, 20, 20))
    lum = 0.30 * rgb[0] + 0.59 * rgb[1] + 0.11 * rgb[2]
    ink = (20, 20, 20) if lum > 110 else (245, 245, 245)
    edge = tuple(int(c * 0.6) for c in rgb) if lum > 60 else (120, 120, 120)

    def P(x, y):
        return (x * k, head + px - y * k)
    for g, code in items:
        for p in getattr(g, "geoms", [g]):
            if p.geom_type != "Polygon" or p.is_empty:
                continue
            d.polygon([P(x, y) for x, y in p.exterior.coords], fill=tuple(rgb), outline=edge)
            for r in p.interiors:
                d.polygon([P(x, y) for x, y in r.coords], fill=(46, 46, 50))
    for g, code in items:
        try:
            big = max(getattr(g, "geoms", [g]), key=lambda q: q.area)
            c = shapely.ops.polylabel(big, 0.3)
        except Exception:
            c = g.representative_point()
        fs = int(np.clip(math.sqrt(max(g.area, 1.0)) * k * 0.28, 10, 20))
        d.text(P(c.x, c.y), code, font=ui_font(fs), fill=ink, anchor="mm",
               stroke_width=1, stroke_fill=tuple(rgb))
    return im


def pack_colour(solids: dict, ci: int, s, gp, order: str):
    """Plate packing of one colour's parts (shared by the export and the 3D plate scene).
    Returns (items sorted by panel/number, placements, n_plates, unplaced)."""
    items = [p for p in solids.values() if p.part.colour == ci]
    if not items:
        return [], [], 0, []
    items.sort(key=lambda p: (p.part.panel, p.part.number))
    outlines = [p.outline for p in items]
    groups = [p.part.panel for p in items] if order not in ("max", "renk") else None
    placements, n_pl, unplaced = pack_outlines(outlines, s.plate_mm, gp.plate_margin,
                                               gp.plate_spacing, groups=groups)
    return items, placements, n_pl, unplaced


# ----------------------------------------------------------------------------- main
def export(m: Mosaic, out_dir: str, progress: Optional[Callable[[float, str], None]] = None,
           gp: Optional[GeoParams] = None, write_stl_files: bool = True) -> dict:
    created = []
    try:
        return _export(m, out_dir, progress, gp, write_stl_files, created)
    except ExportCancelled:
        for p in created:
            shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else None
        try:
            with open(os.path.join(out_dir, "rapor.txt"), "w", encoding="utf-8") as fh:
                fh.write("Dışa aktarım yarıda kesildi (kullanıcı vazgeçti). Dosyalar silindi.\n")
        except OSError:
            pass
        return {"files": [], "plates": [], "plate_images": [], "parts": 0, "panels": 0, "grams": 0,
                "hours": 0, "warnings": ["Dışa aktarım iptal edildi"], "iptal": True}


def _export(m, out_dir, progress, gp, write_stl_files, created):
    t0 = time.time()

    def prog(f, msg):
        if progress is not None and progress(f, msg) is False:
            raise ExportCancelled()
    s = m.settings
    mode = s.mode if s.mode in (Mode.PIECES, Mode.AMS) else Mode.PIECES
    os.makedirs(out_dir, exist_ok=True)
    if not s.merge_same_colour:
        # R6-1: every stone is its own printed piece (default since round 6)
        from .stones_export import export_stones
        return export_stones(m, out_dir, prog, created)
    plates_dir = os.path.join(out_dir, "02_tablalar")
    img_dir = os.path.join(plates_dir, "onizleme")
    stl_dir = os.path.join(out_dir, "03_stl")
    for d in (plates_dir, stl_dir):
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        created.append(d)
    os.makedirs(img_dir, exist_ok=True)
    prog(0.01, "Paneller ve parçalar hesaplanıyor")
    L = build_layout(m, gp, progress=lambda f, msg: prog(0.01 + f * 0.9, msg))
    gp = L.params
    warnings = list(L.warnings)
    bci = L.backing_colour
    pal = s.palette
    names = [ascii_name(f.name) for f in pal]
    fil = _filaments(m)
    mm_ = mass_model(m, mode)

    # --- solids ---------------------------------------------------------------------------
    prog(0.15, "Parçaların 3B modelleri oluşturuluyor")
    solids, meshes = {}, {}
    n_parts = len(L.parts)
    not_tight = 0
    for k, pt in enumerate(L.parts):
        ps = build_part(L, pt, engrave=(mode == Mode.PIECES))
        if ps is None:
            warnings.append(f"{pt.pid} için katı model oluşturulamadı")
            continue
        solids[pt.index] = ps
        if mode == Mode.PIECES:
            v, t, ok = clean_mesh(ps.solid)
            meshes[pt.index] = (v, t)
            not_tight += not ok
        if k % 10 == 0:
            prog(0.15 + 0.5 * (k + 1) / max(n_parts, 1), f"Parça {k + 1}/{n_parts}")
    prog(0.66, "Kafes panelleri oluşturuluyor")
    backings, bmesh = {}, {}
    for pn in L.panels:
        bk = build_backing(L, pn, engrave=True, label_pockets=(mode == Mode.PIECES))
        backings[pn.index] = bk
        if mode == Mode.PIECES:
            v, t, ok = clean_mesh(bk)
            bmesh[pn.index] = (v, t)
            not_tight += not ok
    no_id = sum(1 for p in solids.values() if p.engraved is None and p.part.kind == "normal") \
        if mode == Mode.PIECES else 0

    files, plates_info, plate_images = [], [], []
    plate_of_part: dict[int, str] = {}
    vol_by_colour = {i: 0.0 for i in range(len(pal))}
    parts_by_colour = {i: 0 for i in range(len(pal))}
    order = getattr(s, "plate_order", "panel")
    purge = None
    w_mm, h_mm = m.size_mm

    if mode == Mode.PIECES:
        prog(0.72, "Tablalara diziliyor")
        for ci in range(len(pal)):
            items, placements, n_pl, unplaced = pack_colour(solids, ci, s, gp, order)
            if not items:
                continue
            parts_by_colour[ci] = len(items)
            for i in unplaced:
                warnings.append(f"{items[i].part.pid} tablaya sığmadı")
            n_files = max(1, math.ceil(n_pl / MAX_PLATES))
            fname = [f"{ci + 1}_{names[ci]}_tablalar{f'_{fi + 1}' if n_files > 1 else ''}.3mf"
                     for fi in range(n_files)]
            objs = []
            plate_geo = [[] for _ in range(n_pl)]
            plate_vol = [0.0] * n_pl
            plate_panels = [set() for _ in range(n_pl)]
            for pl in placements:
                ps = items[pl.item]
                v, t = meshes[ps.part.index]
                v = _rot_translate(v, pl.angle, pl.dx, pl.dy)
                objs.append((pl.plate, Obj(ps.part.pid, [Volume(ps.part.pid, v, t, 1)], pl.plate)))
                g = affinity.translate(affinity.rotate(ps.outline, pl.angle, origin=(0, 0)), pl.dx, pl.dy)
                plate_geo[pl.plate].append((g, ps.part.pid))
                plate_vol[pl.plate] += ps.volume_mm3
                plate_panels[pl.plate].add(ps.part.panel)
                vol_by_colour[ci] += ps.volume_mm3
                plate_of_part[ps.part.index] = f"{pal[ci].name} {pl.plate % MAX_PLATES + 1}"
            for fi in range(n_files):
                lo, hi = fi * MAX_PLATES, min(n_pl, (fi + 1) * MAX_PLATES)
                fobjs = []
                for plate, ob in objs:
                    if lo <= plate < hi:
                        ob.plate = plate - lo
                        fobjs.append(ob)
                path = os.path.join(plates_dir, fname[fi])
                pnames = [f"{pal[ci].name} {p - lo + 1} ({', '.join(f'P{q + 1}' for q in sorted(plate_panels[p]))})"
                          for p in range(lo, hi)]
                thumbs = [_plate_thumb([g for g, _ in plate_geo[p]], pal[ci].rgb, s.plate_mm) for p in range(lo, hi)]
                write_bambu_3mf(path, fobjs, hi - lo, [fil[ci]], plate_names=pnames, thumbnails=thumbs,
                                prime_tower=False, title=f"Tessera {pal[ci].name}")
                files.append(path)
            for p in range(n_pl):
                fi = p // MAX_PLATES
                title = (f"{pal[ci].name} · tabla {p - fi * MAX_PLATES + 1} · panel "
                         f"{', '.join(f'P{q + 1}' for q in sorted(plate_panels[p]))}")
                img = os.path.join(img_dir, f"{fname[fi][:-4]}_{p - fi * MAX_PLATES + 1:02d}.png")
                plate_image(plate_geo[p], pal[ci].rgb, title, s.plate_mm).save(img, optimize=True)
                plate_images.append(img)
                plates_info.append({"dosya": fname[fi], "renk": pal[ci].name, "tabla": p - fi * MAX_PLATES + 1,
                                    "parca": len(plate_geo[p]), "panel": sorted(f"P{q + 1}" for q in plate_panels[p]),
                                    "tur": "parca", "_vol": plate_vol[p], "gorsel": img})
        # backing plates: one panel per plate (+ one plate with the wall cleats), own file
        prog(0.82, "Kafes tablaları yazılıyor")
        objs, thumbs, pnames, bvol = [], [], [], []
        for k, pn in enumerate(L.panels):
            v, t = bmesh[pn.index]
            x0, y0, x1, y1 = pn.bounds
            w, h = x1 - x0, y1 - y0
            dx = (s.plate_mm - w) / 2 - x0
            dy = (s.plate_mm - h) / 2 - y0
            if (s.plate_mm - w) / 2 < 18 + 1 and (s.plate_mm - h) / 2 < 28 + 1:
                dx = s.plate_mm - gp.plate_margin - w - x0
                dy = s.plate_mm - gp.plate_margin - h - y0
            objs.append(Obj(f"Kafes {pn.name}", [Volume(f"Kafes {pn.name}", v + (dx, dy, 0), t, 1)], k))
            g = affinity.translate(pn.region, dx, dy)
            thumbs.append(_plate_thumb([g], pal[bci].rgb, s.plate_mm))
            pnames.append(f"Kafes {pn.name}")
            bvol.append(backings[pn.index].volume())
            img = os.path.join(img_dir, f"0_Kafes_{pn.name}.png")
            plate_image([(g, f"Kafes {pn.name}")], pal[bci].rgb, f"Kafes {pn.name} ({pal[bci].name})",
                        s.plate_mm).save(img, optimize=True)
            plate_images.append(img)
        # French cleat pair for hanging the MDF board
        cl_len = float(min(s.plate_mm - 2 * gp.plate_margin - 10, max(120.0, w_mm * 0.6)))
        cl = french_cleat(cl_len)
        cv, ct, _ = clean_mesh(cl)
        kc = len(objs)
        for j, yy in enumerate((40.0, 120.0)):
            objs.append(Obj(f"Askı çıtası {j + 1}", [Volume("Askı", cv + (8.0, yy, 0), ct, 1)], kc))
        thumbs.append(_plate_thumb([shapely.box(8, 40, 8 + cl_len, 70), shapely.box(8, 120, 8 + cl_len, 150)],
                                   pal[bci].rgb, s.plate_mm))
        pnames.append("Duvar askısı (fransız çıta)")
        bvol.append(2 * cl.volume())
        img = os.path.join(img_dir, "0_Kafes_aski.png")
        plate_image([(shapely.box(8, 40, 8 + cl_len, 70), "Askı 1"), (shapely.box(8, 120, 8 + cl_len, 150), "Askı 2")],
                    pal[bci].rgb, "Duvar askısı: 2 fransız çıta", s.plate_mm).save(img, optimize=True)
        plate_images.append(img)
        n_b = len(pnames)
        for fi in range(max(1, math.ceil(n_b / MAX_PLATES))):
            lo, hi = fi * MAX_PLATES, min(n_b, (fi + 1) * MAX_PLATES)
            sub = [o for o in objs if lo <= o.plate < hi]
            for o in sub:
                o.plate -= lo
            suffix = f"_{fi + 1}" if n_b > MAX_PLATES else ""
            fname = f"0_Kafes_{names[bci]}{suffix}.3mf"
            path = os.path.join(plates_dir, fname)
            write_bambu_3mf(path, sub, hi - lo, [fil[bci]], plate_names=pnames[lo:hi],
                            thumbnails=thumbs[lo:hi], prime_tower=False, title="Tessera kafes")
            files.append(path)
            for k in range(lo, hi):
                plates_info.append({"dosya": fname, "renk": pal[bci].name, "tabla": k - lo + 1, "parca": 1,
                                    "tur": "kafes" if k < len(L.panels) else "aski",
                                    "panel": [L.panels[k].name] if k < len(L.panels) else [],
                                    "_vol": bvol[k], "gorsel": plate_images[-(n_b - k)]})
        # split model grams over plates by printed volume
        tot_part_vol = sum(p["_vol"] for p in plates_info if p["tur"] == "parca") or 1.0
        tot_back_vol = sum(p["_vol"] for p in plates_info if p["tur"] != "parca") or 1.0
        for p in plates_info:
            g = (mm_["parts_g"] * p["_vol"] / tot_part_vol if p["tur"] == "parca"
                 else mm_["backing_g"] * p["_vol"] / tot_back_vol)
            p["gram"] = round(g)
    else:
        prog(0.72, "AMS panelleri hazırlanıyor")
        objs, thumbs, pnames, wt = [], [], [], []
        no_tower = 0
        for k, pn in enumerate(L.panels):
            x0, y0, x1, y1 = pn.bounds
            w, h = x1 - x0, y1 - y0
            room_y = s.plate_mm - 2 * gp.plate_margin - h
            room_x = s.plate_mm - 2 * gp.plate_margin - w
            dx = s.plate_mm - gp.plate_margin - w - x0
            dy = s.plate_mm - gp.plate_margin - h - y0
            if room_y >= room_x:
                wt.append((max(30.0, (s.plate_mm - 35) / 2), gp.plate_margin + 2.0))
                room = room_y
            else:
                wt.append((gp.plate_margin + 2.0, max(35.0, (s.plate_mm - 35) / 2)))
                room = room_x
            no_tower += room < 45
            vols = []
            bk = backings[pn.index].translate((dx, dy, 0))
            v, t, ok = clean_mesh(bk)
            not_tight += not ok
            vols.append(Volume(f"Kafes {pn.name}", v, t, bci + 1))
            per_col: dict[int, list] = {}
            for i in pn.parts:
                if i in solids:
                    per_col.setdefault(L.parts[i].colour, []).append(solids[i])
            for ci, lst in sorted(per_col.items()):
                man = Manifold.compose([p.solid for p in lst]).translate((dx, dy, gp.backing_t))
                v, t, ok = clean_mesh(man)
                not_tight += not ok
                vols.append(Volume(f"{pal[ci].name} {pn.name}", v, t, ci + 1))
                vol_by_colour[ci] += sum(p.volume_mm3 for p in lst)
                parts_by_colour[ci] += len(lst)
            objs.append(Obj(f"Panel {pn.name}", vols, k))
            g = affinity.translate(pn.region, dx, dy)
            thumbs.append(_plate_thumb([g], pal[bci].rgb, s.plate_mm))
            pnames.append(f"Panel {pn.name}")
            img = os.path.join(img_dir, f"AMS_{pn.name}.png")
            items = [(affinity.translate(L.parts[i].outline, dx, dy), "") for i in pn.parts]
            pim = Image.new("RGB", (PLATE_IMG_PX, PLATE_IMG_PX + 44), "white")
            _ams_plate_image(pim, L, pn, dx, dy, s.plate_mm, wt[-1] if room >= 45 else None)
            pim.save(img, optimize=True)
            plate_images.append(img)
            plates_info.append({"dosya": "", "renk": "çok renkli", "tabla": k + 1, "panel": [pn.name],
                                "parca": 1, "renk_bolgesi": len(pn.parts), "tur": "ams",
                                "_vol": backings[pn.index].volume() + sum(solids[i].volume_mm3 for i in pn.parts
                                                                       if i in solids), "gorsel": img})
        tower = not no_tower
        if no_tower:
            warnings.append("Paneller tablayı doldurduğu için temizleme kulesi kapalı: renk değişiminde "
                            "atık filament arkadaki oluktan atılır. Renkler birbirine biraz bulaşabilir; "
                            "ilk paneli deneme olarak bas")
        for fi in range(max(1, math.ceil(len(objs) / MAX_PLATES))):
            lo, hi = fi * MAX_PLATES, min(len(objs), (fi + 1) * MAX_PLATES)
            sub = objs[lo:hi]
            for o in sub:
                o.plate -= lo
            suffix = f"_{fi + 1}" if len(objs) > MAX_PLATES else ""
            path = os.path.join(plates_dir, f"AMS_paneller{suffix}.3mf")
            write_bambu_3mf(path, sub, hi - lo, fil, plate_names=pnames[lo:hi], thumbnails=thumbs[lo:hi],
                            wipe_tower_xy=wt[lo:hi], prime_tower=tower, title="Tessera AMS")
            files.append(path)
            for k in range(lo, hi):
                plates_info[k]["dosya"] = os.path.basename(path)
                plates_info[k]["tabla"] = k - lo + 1
        # wall cleats in their own single-colour file
        cl_len = float(min(s.plate_mm - 2 * gp.plate_margin - 10, max(120.0, w_mm * 0.6)))
        cv, ct, _ = clean_mesh(french_cleat(cl_len))
        cobjs = [Obj(f"Askı çıtası {j + 1}", [Volume("Askı", cv + (8.0, yy, 0), ct, 1)], 0)
                 for j, yy in enumerate((40.0, 120.0))]
        geo = [shapely.box(8, 40, 8 + cl_len, 66), shapely.box(8, 120, 8 + cl_len, 146)]
        path = os.path.join(plates_dir, "0_Aski_citalari.3mf")
        write_bambu_3mf(path, cobjs, 1, [fil[bci]], plate_names=["Duvar askısı (fransız çıta)"],
                        thumbnails=[_plate_thumb(geo, pal[bci].rgb, s.plate_mm)], prime_tower=False,
                        title="Tessera askı")
        files.append(path)
        img = os.path.join(img_dir, "0_Aski_citalari.png")
        plate_image([(geo[0], "Askı 1"), (geo[1], "Askı 2")], pal[bci].rgb, "Duvar askısı: 2 fransız çıta",
                    s.plate_mm).save(img, optimize=True)
        plate_images.append(img)
        swaps_by_panel = _ams_swaps_by_panel(L)
        tot_sw = sum(swaps_by_panel) or 1
        purge = {"degisim": mm_["swaps"], "purge_gram": round(mm_["purge_g"]),
                 "saat": round(mm_["swaps"] * SWAP_S / 3600, 1), "paneller": []}
        totv = sum(p["_vol"] for p in plates_info) or 1.0
        for k, p in enumerate(plates_info):
            sw = int(round(mm_["swaps"] * swaps_by_panel[k] / tot_sw))
            p["degisim"] = sw
            p["purge_gram"] = round(mm_["purge_g"] * swaps_by_panel[k] / tot_sw, 1)
            p["gram"] = round((mm_["parts_g"] + mm_["backing_g"]) * p["_vol"] / totv + p["purge_gram"])
            purge["paneller"].append({"panel": p["panel"][0], "degisim": sw, "purge_gram": p["purge_gram"]})
    if mode == Mode.AMS:
        # the wall-cleat plate is a plate too (estimate()['tabla'] counts it)
        cg = 2 * french_cleat(cl_len).volume() / 1000 * PLA_DENSITY * 0.55
        plates_info.append({"dosya": "0_Aski_citalari.3mf", "renk": pal[bci].name, "tabla": 1, "parca": 2,
                            "tur": "aski", "panel": [], "gram": round(cg), "gorsel": plate_images[-1]})
    # hours per plate: share of the model total by grams
    gsum = sum(p["gram"] for p in plates_info) or 1.0
    for p in plates_info:
        p["saat"] = round(mm_["hours"] * p["gram"] / gsum, 1)
        p.pop("_vol", None)

    # --- STL fallback -------------------------------------------------------------------------
    if write_stl_files:
        prog(0.88, "Yedek STL dosyaları yazılıyor")
        cv, ct, _ = clean_mesh(french_cleat(float(min(s.plate_mm - 2 * gp.plate_margin - 10,
                                                      max(120.0, w_mm * 0.6)))))
        write_stl(os.path.join(stl_dir, "aski_citasi_2_adet_bas.stl"), cv, ct)
        for pn in L.panels:
            pdir = os.path.join(stl_dir, pn.name)
            if mode == Mode.PIECES:
                write_stl(os.path.join(pdir, f"kafes_{pn.name}.stl"), *bmesh[pn.index])
                for i in pn.parts:
                    if i in meshes:
                        write_stl(os.path.join(pdir, f"{L.parts[i].pid}.stl"), *meshes[i])
            else:
                v, t, _ = clean_mesh(backings[pn.index])
                write_stl(os.path.join(pdir, f"kafes_{pn.name}.stl"), v, t)
                per_col: dict[int, list] = {}
                for i in pn.parts:
                    if i in solids:
                        per_col.setdefault(L.parts[i].colour, []).append(solids[i].solid)
                for ci, lst in per_col.items():
                    v, t, _ = clean_mesh(Manifold.compose(lst).translate((0, 0, gp.backing_t)))
                    write_stl(os.path.join(pdir, f"{pn.name}_{names[ci]}.stl"), v, t)
    if not_tight:
        warnings.append(f"{not_tight} gövde STL'de tam kapalı değil (Bambu Studio onarır)")

    # --- guide + report ---------------------------------------------------------------------
    prog(0.92, "Montaj kılavuzu çiziliyor")
    from .guide import write_guide
    guide_files = write_guide(L, out_dir, mode, bci, plate_of_part, plates_info)
    files.extend(guide_files)
    vol_tot = sum(vol_by_colour.values()) or 1.0
    renk_gram = {pal[i].name: round(mm_["parts_g"] * v / vol_tot) for i, v in vol_by_colour.items()}
    renk_gram[pal[bci].name] = renk_gram.get(pal[bci].name, 0) + round(mm_["backing_g"])
    n_assemble = len(solids) if mode == Mode.PIECES else len(L.panels)
    result = {
        "mod": mode,
        "files": files,
        "plates": plates_info,
        "plate_images": plate_images,
        "parts": n_assemble,
        "montaj_adedi": n_assemble,
        "parca_tanimi": ("Elle yerleştirilecek tek renk parça sayısı" if mode == Mode.PIECES else
                         "AMS'de her panel tek nesne: yerleştirilecek parça sayısı = panel sayısı"),
        "renk_bolgesi": len(solids),
        "hassas_parca": sum(1 for p in solids.values() if p.part.kind == "hassas"),
        "panels": len(L.panels),
        "panel_xy": list(L.grid),
        "grams": round(mm_["grams"]),
        "hours": round(mm_["hours"], 1),
        "kg_m2": round(mm_["kg_m2"], 2),
        "parca_gram": round(mm_["parts_g"]),
        "kafes_gram": round(mm_["backing_g"]),
        "warnings": warnings,
        "renk_gram": renk_gram,
        "renk_parca": {pal[i].name: n for i, n in parts_by_colour.items()},
        "kafes_renk": pal[bci].name,
        "numarasiz_parca": no_id,
        "purge": purge,
        "tabla_sirasi": "panel" if order not in ("max", "renk") else "renk",
        "panel_boyutlari_mm": [[round(v) for v in pn.size] for pn in L.panels],
        "kapali_olmayan_govde": not_tight,
        "sure_sn": round(time.time() - t0, 1),
    }
    from .report import write_report
    files.extend(write_report(L, out_dir, result))
    prog(1.0, "Dışa aktarım tamamlandı")
    return result


def _rot_translate(v, angle, dx, dy):
    a = math.radians(angle)
    c, s_ = math.cos(a), math.sin(a)
    out = v.copy()
    out[:, 0] = v[:, 0] * c - v[:, 1] * s_ + dx
    out[:, 1] = v[:, 0] * s_ + v[:, 1] * c + dy
    return out


def _ams_swaps_by_panel(L: Layout) -> list:
    """Relative AMS cost per panel (layers x (colours-1)) from the layout tiles."""
    gp = L.params
    out = []
    for pn in L.panels:
        tops: dict[int, float] = {}
        for i in pn.parts:
            pt = L.parts[i]
            h = float(np.max(L.tile_plane[pt.tiles, 0])) + 0.3
            tops[pt.colour] = max(tops.get(pt.colour, 0.0), h)
        zmax = gp.backing_t + max(tops.values() or [0.0])
        sw = 0
        for k in range(int(math.ceil(zmax / 0.2))):
            z = (k + 0.5) * 0.2
            pres = {L.backing_colour} if z < gp.backing_t else {c for c, t in tops.items() if z < gp.backing_t + t}
            sw += max(0, len(pres) - 1)
        out.append(sw)
    return out


def _ams_plate_image(im, L, pn, dx, dy, plate_mm, tower):
    pal = L.mosaic.settings.palette
    px = PLATE_IMG_PX
    head = 44
    d = ImageDraw.Draw(im)
    k = px / plate_mm
    d.rectangle((0, head, px - 1, head + px - 1), fill=(46, 46, 50))
    d.text((10, 8), f"AMS · Panel {pn.name} (tek nesne, çok renkli)", font=ui_font(24), fill=(20, 20, 20))

    def P(x, y):
        return (x * k, head + px - y * k)
    reg = affinity.translate(pn.region, dx, dy)
    for p in getattr(reg, "geoms", [reg]):
        d.polygon([P(x, y) for x, y in p.exterior.coords], fill=tuple(pal[L.backing_colour].rgb))
    for i in pn.parts:
        pt = L.parts[i]
        g = affinity.translate(pt.outline, dx, dy)
        for p in getattr(g, "geoms", [g]):
            if p.geom_type == "Polygon" and not p.is_empty:
                d.polygon([P(x, y) for x, y in p.exterior.coords], fill=tuple(pal[pt.colour].rgb))
    if tower is None:
        return
    tx, ty = tower
    d.rectangle((*P(tx, ty + 35), *P(tx + 35, ty)), outline=(255, 200, 0), width=2)
    d.text(P(tx + 17.5, ty + 17.5), "kule", font=ui_font(14), fill=(255, 200, 0), anchor="mm")
