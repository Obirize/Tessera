"""Mass / time model: the single source of truth for grams, hours and kg/m2 (R2-7).

mass_model(m) works on the Mosaic raster only (no meshing, < 0.5 s) and mirrors what the
export builds and what Bambu Studio slices (0.20 mm Standard, 2 walls, 15 % infill):

parts   printed volume = walls (tile perimeter x height x 0.84 mm) + tops (1.0 mm) + bottoms
        (0.6 mm) + 15 % of the rest; tall tiles are hollow (open bottom) so they lose their
        bottom and infill and gain inner walls.
backing PIECES: lattice = frame + members along every part edge + triangular bracing + ribs.
        AMS: full plate.
purge   AMS: per panel and layer, colours present - 1 changes, PURGE_G_PER_SWAP each.

The K_* factors are fitted to Bambu Studio CLI slices of the six examples (docs/uretim_raporu.md).
"""
from __future__ import annotations

import math

import numpy as np

from ..geometry.layout import GeoParams, backing_colour_index, panel_xy
from ..model import Mode, Mosaic

PLA_DENSITY = 1.24          # g/cm3
WALL_MM = 0.84              # 2 walls x 0.42 mm
TOP_MM = 1.0                # 5 top layers
BOTTOM_MM = 0.6             # 3 bottom layers
INFILL = 0.15
# fitted factors (sliced / model); see fit in docs/uretim_raporu.md
K_PARTS = 0.77
K_PARTS_AMS = 0.735
K_BACKING = 0.70              # R5: lattice cleaned (rib slivers removed, pads joined); CLI poster 225 g
K_BACKING_AMS = 0.735
# time model: hours = grams / G_PER_H + wall_m * H_PER_WALL_M + plates * PLATE_H + swaps * SWAP_S
G_PER_H = 45.9              # fitted on 8 complete CLI slices (6 examples), error <= 5 %
H_PER_WALL_M = 10.69
PLATE_H = 0.10
PURGE_G_PER_SWAP = 0.48     # CLI: 0.47-0.50 g per change with the prime tower
PURGE_G_PER_SWAP_NO_TOWER = 0.39
SWAP_S = 78.6
SWAP_SCALE = 0.90           # our layer model over-counts changes by ~10 % (CLI)
PART_PLATE_FILL = 0.42      # share of a plate covered by packed parts (OBB packing)


def _tile_features(m: Mosaic, gp: GeoParams):
    mm = m.settings.mm_per_px
    lab = m.label
    n = int(lab.max()) + 1
    area = np.bincount(lab.ravel(), minlength=n).astype(np.float64) * mm * mm
    h = np.zeros(n)
    h[: m.n_tiles] = np.clip(m.tile_plane[:, 0], 0.0, None)
    col = np.zeros(n, np.int64)
    col[: m.n_tiles] = m.tile_colour
    # perimeter per tile (pixel edges against other tiles or the image border)
    per = np.zeros(n)
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
        d = a != b
        per += np.bincount(a[d], minlength=n) + np.bincount(b[d], minlength=n)
    per += np.bincount(lab[0, :], minlength=n) + np.bincount(lab[-1, :], minlength=n)
    per += np.bincount(lab[:, 0], minlength=n) + np.bincount(lab[:, -1], minlength=n)
    per *= mm
    # colour-boundary length (part edges ~ colour edges): pixel edges between different colours
    c = m.colour
    cb = (np.count_nonzero(c[:, 1:] != c[:, :-1]) + np.count_nonzero(c[1:, :] != c[:-1, :])) * mm
    return area, h, per, col, cb


def _parts_volume(area, h, per, gp: GeoParams, hollow: bool):
    """Printed (sliced) volume of the relief, mm3, per tile."""
    a = np.maximum(area, 1e-6)
    hh = np.maximum(h, 0.0)
    solid = a * hh
    walls = np.minimum(per * WALL_MM * hh, solid)
    tops = np.minimum(a * TOP_MM, np.maximum(solid - walls, 0))
    bottoms = np.minimum(a * BOTTOM_MM, np.maximum(solid - walls - tops, 0))
    rest = np.maximum(solid - walls - tops - bottoms, 0)
    vol = walls + tops + bottoms + INFILL * rest
    if hollow:
        # inner area of a tile with 1.2 mm walls (approx. a - per * wall)
        inner = np.maximum(a - per * gp.hollow_wall, 0)
        tall = (hh >= gp.hollow_min_h) & (inner >= 4.0)
        inner_per = np.sqrt(inner) * 4
        hol = (np.minimum(per * WALL_MM * hh + inner_per * WALL_MM * (hh - gp.hollow_top), solid)
               + np.minimum(a - inner, a) * BOTTOM_MM + a * TOP_MM)
        vol = np.where(tall, np.minimum(hol, vol + inner_per * WALL_MM * hh), vol)
    return vol


# --- round 6: separate stones (fitted to Bambu CLI slices of the poster, klasik + freud)
# CLI (poster 42x59 cm): klasik 915 g / 54.4 h single colour, 1262 g / 66.6 h AMS;
#                       freud 802 g / 42.6 h single colour, 1076 g / 48.7 h AMS
K_STONES = {"freud": 1.0, "mozaik": 0.84, "kirik": 0.84, "blok": 0.84, "iri": 0.84}  # R8: fitted to one CLI slice (poster, 658 g / 36.0 h) #              # sliced / model grams per style (default below)
K_STONES_DEFAULT = 0.88                 # bevelled stones: the chamfer takes material off
K_STONES_AMS = 0.92                     # stones fused to the base lose their bottom shells
H_WALL_STYLE = {"freud": 1.0, "mozaik": 1.13, "kirik": 1.13, "blok": 1.13, "iri": 1.13} #          # wall-time factor per style
H_WALL_DEFAULT = 0.82
H_PER_STONE = 0.0           # per-object overhead (travel, retraction, seams), hours
STONE_PLATE_FILL = 0.62
CUT_PLATE_FILL = 0.74        # R8 stones: 147 guessed for 148 packed plates (13 pieces, 25-70 cm)
AMS_BASE_T = 0.8


def mass_model_stones(m: Mosaic, mode: str, n_plates=None) -> dict:
    gp = GeoParams()
    s = m.settings
    mm = s.mm_per_px
    H, W = m.label.shape
    w_mm, h_mm = W * mm, H * mm
    art_mm2 = w_mm * h_mm
    nx, ny = panel_xy(w_mm, h_mm, s.panel_max_mm)
    if m.polys is not None:              # R8: measured on the real outlines (already without the joint)
        area = np.array([p.area for p in m.polys])
        per = np.array([p.length for p in m.polys])
        h = np.asarray(m.tile_plane[:, 0], float)
        col = np.asarray(m.tile_colour)
        a_eff, per_eff = area, per
    else:
        area, h, per, col, cb = _tile_features(m, gp)
        g = float(s.gap_mm)
        a_eff = np.maximum(area - per * g / 2, 0.0)
        per_eff = np.maximum(per - 4 * g, 0.0)
    present = area > 0
    vol_t = _parts_volume(a_eff, h, per_eff, gp, hollow=False) * present
    style = str(getattr(s, "style", "klasik"))
    k = K_STONES.get(style, K_STONES_DEFAULT) * (1.0 if mode == Mode.PIECES else K_STONES_AMS)
    hw = H_WALL_STYLE.get(style, H_WALL_DEFAULT)
    parts_g = vol_t.sum() / 1000 * PLA_DENSITY * k
    by_col = np.bincount(col, vol_t, minlength=len(s.palette)) / 1000 * PLA_DENSITY * k
    n_st = int(present.sum())
    backing_g = 0.0
    swaps = 0
    if mode == Mode.AMS:
        backing_g = art_mm2 * AMS_BASE_T / 1000 * PLA_DENSITY * K_BACKING_AMS
        plates = nx * ny
        swaps = int(round(ams_swaps(m, gp, nx, ny) * SWAP_SCALE))
    else:
        usable = (s.plate_mm - 2 * gp.plate_margin) ** 2 * (STONE_PLATE_FILL if m.polys is None else CUT_PLATE_FILL)
        plates = sum(max(1, math.ceil(a / usable)) for a in
                     np.bincount(col, a_eff + per * 1.25, minlength=len(s.palette)) if a > 0)
        if getattr(s, "backing", "yok") == "kafes":
            perim = 2 * (w_mm + h_mm) + 2 * (nx - 1) * h_mm + 2 * (ny - 1) * w_mm
            floor = min(art_mm2, perim * gp.frame_w + (per.sum() / 2) * 2 * gp.flange_w
                        + art_mm2 * 3 * gp.brace_w / gp.brace_pitch * 0.8)
            backing_g = (floor * gp.backing_t + per.sum() / 2 * gp.rib_w * gp.rib_h) / 1000 * PLA_DENSITY * K_BACKING
            plates += nx * ny + 1
        elif n_plates is not None:
            plates = n_plates
    usable_l = s.plate_mm - 2 * gp.plate_margin
    tower = usable_l - (min(w_mm / nx, h_mm / ny) + 10.0) >= 45.0
    purge_g = swaps * (PURGE_G_PER_SWAP if tower else PURGE_G_PER_SWAP_NO_TOWER)
    wall_m = float((per_eff * np.maximum(h, 0.5)).sum()) / 1000.0 / 0.2 / 1000.0
    grams = parts_g + backing_g + purge_g
    hours = ((parts_g + backing_g) / G_PER_H + wall_m * H_PER_WALL_M * hw + plates * PLATE_H
             + n_st * H_PER_STONE + swaps * SWAP_S / 3600.0)
    return {"grams": float(grams), "hours": float(hours), "kg_m2": float(grams / 1000.0 / (art_mm2 / 1e6)),
            "parts_g": float(parts_g), "backing_g": float(backing_g), "purge_g": float(purge_g),
            "swaps": int(swaps), "prime_tower": bool(tower) if mode == Mode.AMS else None,
            "plates": int(plates), "panel_xy": [nx, ny], "colour_g": [float(x) for x in by_col],
            "backing_colour": backing_colour_index(m), "stones": n_st}


def mass_model(m: Mosaic, mode: str | None = None, plates: int | None = None) -> dict:
    """grams, hours, kg/m2 for the whole artwork (fast, raster based). plates: how many single-colour
    plates the stones really fill, when they have been packed (otherwise it is guessed from their area)."""
    gp = GeoParams()
    s = m.settings
    mode = mode or (s.mode if s.mode in (Mode.PIECES, Mode.AMS) else Mode.PIECES)
    if not s.merge_same_colour:
        return mass_model_stones(m, mode, plates)
    mm = s.mm_per_px
    H, W = m.label.shape
    w_mm, h_mm = W * mm, H * mm
    art_mm2 = w_mm * h_mm
    nx, ny = panel_xy(w_mm, h_mm, s.panel_max_mm)
    area, h, per, col, cb = _tile_features(m, gp)
    vol_t = _parts_volume(area, h, per, gp, hollow=True)
    # web under each part (web_min..base) is mostly bottom shell + walls along part edges
    web_vol = art_mm2 * 0.0 + cb * WALL_MM * gp.web_min
    kp = K_PARTS if mode == Mode.PIECES else K_PARTS_AMS
    parts_g = (vol_t.sum() + web_vol) / 1000 * PLA_DENSITY * kp
    parts_by_colour = np.bincount(col, vol_t, minlength=len(s.palette)) / 1000 * PLA_DENSITY * kp
    # backing
    perim = 2 * (w_mm + h_mm) + 2 * (nx - 1) * h_mm + 2 * (ny - 1) * w_mm   # panel outlines
    cleat_len = min(s.plate_mm - 2 * gp.plate_margin - 10, max(120.0, w_mm * 0.6))
    cleats = 2 * (26 + 19) / 2 * 7 * cleat_len * 0.55          # two hanging cleats (sliced)
    if mode == Mode.PIECES:
        frame = perim * gp.frame_w
        members = (cb + perim) * 2 * (gp.flange_w + gp.part_inset)
        brace = art_mm2 * 3 * gp.brace_w / gp.brace_pitch
        floor = min(art_mm2, frame + members + brace * 0.8)
        ribs = (cb + perim * 0.5) * gp.rib_w * gp.rib_h
        back_vol = floor * gp.backing_t + ribs + cleats
        backing_g = back_vol / 1000 * PLA_DENSITY * K_BACKING
    else:
        back_vol = art_mm2 * gp.backing_t
        backing_g = (back_vol * K_BACKING_AMS + cleats * K_BACKING) / 1000 * PLA_DENSITY
    # plates
    n_pan = nx * ny
    wall_m = float((per * np.maximum(h, 0.5)).sum()) / 1000.0 / 0.2 / 1000.0   # wall path km-ish
    swaps = 0
    if mode == Mode.PIECES:
        usable_a = (s.plate_mm - 2 * gp.plate_margin) ** 2 * PART_PLATE_FILL
        part_plates = sum(max(1, math.ceil(a / usable_a)) for a in
                          np.bincount(col, area, minlength=len(s.palette)) if a > 0)
        plates = part_plates + n_pan + 1          # + the hanging-cleat plate
    else:
        plates = n_pan + 1                        # + the hanging-cleat plate
        swaps = int(round(ams_swaps(m, gp, nx, ny) * SWAP_SCALE))
    usable = s.plate_mm - 2 * gp.plate_margin
    tower = usable - (min(w_mm / nx, h_mm / ny) + 10.0) >= 45.0
    purge_g = swaps * (PURGE_G_PER_SWAP if tower else PURGE_G_PER_SWAP_NO_TOWER)
    grams = parts_g + backing_g + purge_g
    hours = ((parts_g + backing_g) / G_PER_H + wall_m * H_PER_WALL_M + plates * PLATE_H
             + swaps * SWAP_S / 3600.0)
    return {
        "grams": float(grams),
        "hours": float(hours),
        "kg_m2": float(grams / 1000.0 / (art_mm2 / 1e6)),
        "parts_g": float(parts_g),
        "backing_g": float(backing_g),
        "purge_g": float(purge_g),
        "swaps": int(swaps),
        "prime_tower": bool(tower) if mode == Mode.AMS else None,
        "plates": int(plates),
        "panel_xy": [nx, ny],
        "colour_g": [float(g) for g in parts_by_colour],
        "backing_colour": backing_colour_index(m),
    }


def ams_swaps(m: Mosaic, gp: GeoParams, nx: int, ny: int, layer_h: float = 0.2) -> int:
    """Filament changes in AMS mode: per panel cell and layer, (colours present - 1)."""
    s = m.settings
    H, W = m.label.shape
    bk = backing_colour_index(m)
    col = m.colour
    hgt = m.height
    rows = np.minimum((np.arange(H) * ny) // H, ny - 1)
    cols = np.minimum((np.arange(W) * nx) // W, nx - 1)
    cell = rows[:, None] * nx + cols[None, :]
    keep = ~m.gap
    swaps = 0
    for c in range(nx * ny):
        sel = (cell == c) & keep
        if not sel.any():
            continue
        tops = {}
        cc, hh = col[sel], hgt[sel]
        for ci in np.unique(cc).tolist():
            tops[ci] = float(np.percentile(hh[cc == ci], 99.5)) + 0.3
        zmax = gp.backing_t + max(tops.values())
        for k in range(int(math.ceil(zmax / layer_h))):
            z = (k + 0.5) * layer_h
            present = set()
            if z < gp.backing_t:
                present.add(bk)
            else:
                present.update(ci for ci, t in tops.items() if z < gp.backing_t + t)
            swaps += max(0, len(present) - 1)
    return swaps
