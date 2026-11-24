"""Export API used by the UI (signatures fixed by docs/MIMARI.md).

estimate(m, exact=True)              stats-bar numbers; exact=False is the fast "≈" variant
count_parts(m)                       exact part count with the export layout (run in background)
export_project(m, out_dir, progress) full production export (both modes); progress(f, msg)
                                     returning False cancels the export
mass_model(m)                        grams / hours / kg/m2 (single source of truth, R2-7)
preview_parts(m, panel_id)           rendered 3D view (numpy RGB) of one panel's real meshes
panel_xy(w_mm, h_mm, panel_max)      the panel grid used everywhere

Keep this module importable without Qt and without heavy work at import time.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
from scipy import ndimage as ndi

from ..geometry.layout import GeoParams, backing_colour_index, panel_xy  # noqa: F401
from ..model import Mode, Mosaic
from .mass import mass_model  # noqa: F401


def _part_count(m: Mosaic, gp: GeoParams, nx: int, ny: int) -> int:
    """Approximate loose part count in PIECES mode: same-colour regions per panel cell."""
    s = m.settings
    mm = s.mm_per_px
    st = 2 if mm < 0.6 else 1
    col = m.colour[::st, ::st]
    pmm = mm * st
    h, w = col.shape
    cell_r = np.minimum((np.arange(h) * ny) // h, ny - 1)
    cell_c = np.minimum((np.arange(w) * nx) // w, nx - 1)
    cell = cell_r[:, None] * nx + cell_c[None, :]
    bk = backing_colour_index(m)
    big_px = gp.min_part_mm2 / pmm ** 2
    tiny_px = 1.5 / pmm ** 2
    n = 0
    for ci in range(len(s.palette)):
        mask = col == ci
        if not mask.any():
            continue
        if not s.merge_same_colour:
            tiles = np.unique(m.label[::st, ::st][mask])
            n += len(tiles) if ci != bk else int((m.tile_area_px[tiles] * mm * mm >= gp.min_part_mm2).sum())
            continue
        lbl, k = ndi.label(mask)
        key = np.where(mask, lbl.astype(np.int64) * (nx * ny + 1) + cell, -1)
        u, cnt = np.unique(key[key >= 0], return_counts=True)
        big = cnt >= big_px
        n += int(big.sum())
        if ci != bk:
            # small regions: about half are bridged to a same-colour neighbour, the rest are loose
            n += int(0.5 * ((cnt >= tiny_px) & ~big).sum())
    return n


def count_parts(m: Mosaic) -> dict:
    """Exact part count with the export's own layout logic (R3-4). ~1-3 s at preview
    resolution: run it in the background and replace the fast "≈" number with it.
    Returns dict(parca, hassas, cok_kucuk, panel). AMS: parca = panels to assemble."""
    from ..geometry.layout import build_layout
    s = m.settings
    mode = s.mode if s.mode in (Mode.PIECES, Mode.AMS) else Mode.PIECES
    if mode == Mode.AMS:
        H, W = m.label.shape
        nx, ny = panel_xy(W * s.mm_per_px, H * s.mm_per_px, s.panel_max_mm)
        return {"parca": nx * ny, "hassas": 0, "cok_kucuk": 0, "panel": nx * ny}
    if not s.merge_same_colour:
        from ..geometry.stones import build_stones
        Ls = build_stones(m)
        return {"parca": len(Ls.stones), "hassas": 0, "cok_kucuk": int(Ls.stats.get("dar_tas", 0)),
                "panel": len(Ls.panels), "tas": len(Ls.stones)}
    key = (id(m.label), m.label.shape, int(m.label[::97, ::89].sum()), mode, s.merge_same_colour,
           s.panel_max_mm, getattr(s, "backing_colour", None), s.gap_mm)
    if key in _COUNT_CACHE:
        return dict(_COUNT_CACHE[key])
    L = _LAYOUT_CACHE.get(preview_key(m))
    if L is None:
        L = build_layout(m)
        _LAYOUT_CACHE.clear()
        _MESH_CACHE.clear()
        _LAYOUT_CACHE[preview_key(m)] = L      # Panel 3D reuses this layout
    out = {"parca": len(L.parts), "hassas": int(L.stats.get("hassas", 0)),
           "cok_kucuk": int(L.stats.get("cok_kucuk", 0)), "panel": len(L.panels)}
    if len(_COUNT_CACHE) > 8:
        _COUNT_CACHE.clear()
    _COUNT_CACHE[key] = out
    return dict(out)


_COUNT_CACHE: dict = {}


def _quick_count(m: Mosaic) -> dict:
    """Same rules as the export layout, without outlines/polygon simplification (±2 %, ~0.3 s
    at preview resolution)."""
    from ..geometry.layout import build_layout
    s = m.settings
    key = ("quick", id(m.label), m.label.shape, int(m.label[::97, ::89].sum()), s.mode,
           s.merge_same_colour, s.panel_max_mm, getattr(s, "backing_colour", None))
    if key not in _COUNT_CACHE:
        if len(_COUNT_CACHE) > 8:
            _COUNT_CACHE.clear()
        _COUNT_CACHE[key] = build_layout(m, count_only=True)
    return dict(_COUNT_CACHE[key])


def estimate(m: Mosaic, exact: bool = True) -> dict:
    """Numbers for the stats bar. Same panel grid and mass model as the export.

    exact=True  (default): the parts count uses the export's layout rules (±2 %, ~0.3 s at
                 preview resolution); count_parts(m) gives the exact export number.
    exact=False: fast raster approximation (< 0.1 s); `parca_kesin` is False, show it with "≈"
                 and call count_parts(m) in the background."""
    gp = GeoParams()
    s = m.settings
    mode = s.mode if s.mode in (Mode.PIECES, Mode.AMS) else Mode.PIECES
    H, W = m.label.shape
    mm = s.mm_per_px
    nx, ny = panel_xy(W * mm, H * mm, s.panel_max_mm)
    mm_ = mass_model(m, mode)
    hassas = None
    if mode == Mode.AMS:
        parca = nx * ny          # AMS: each panel is one printed object to assemble
    elif not s.merge_same_colour:
        parca = int(mm_.get("stones", m.n_tiles))      # every stone is its own piece (R6-1)
    elif exact:
        c = _quick_count(m)
        parca, hassas = c["parca"], c["hassas"]
    else:
        parca = _part_count(m, gp, nx, ny)
    return {
        "parca": int(parca),
        "tabla": int(mm_["plates"]),
        "gram": int(round(mm_["grams"])),
        "saat": round(mm_["hours"], 1),
        "purge_gram": 0 if mode == Mode.PIECES else int(round(mm_["purge_g"])),
        "panel": int(nx * ny),
        "panel_xy": [nx, ny],
        "kg_m2": round(mm_["kg_m2"], 2),
        "kafes_gram": int(round(mm_["backing_g"])),
        "parca_gram": int(round(mm_["parts_g"])),
        "degisim": int(mm_["swaps"]),
        "parca_kesin": bool(exact or mode == Mode.AMS),
        "hassas": hassas,
        "parca_tanimi": ("elle yerleştirilecek parça" if mode == Mode.PIECES
                         else "panel (her panel tek nesne olarak basılır)"),
    }


def export_project(m: Mosaic, out_dir: str, progress: Optional[Callable[[float, str], None]] = None) -> dict:
    """Write the full production package to out_dir (see docs/MIMARI.md for the layout).

    Returns dict(files, plates, plate_images, parts, panels, grams, hours, warnings, ...)."""
    from . import project as P
    return P.export(m, out_dir, progress)


_LAYOUT_CACHE: dict = {}
_MESH_CACHE: dict = {}


def preview_key(m: Mosaic) -> tuple:
    """Identity of what preview_parts shows. When this changes (new mosaic, production mode,
    merge setting, panel size, backing colour) the UI must request a new Panel 3D image."""
    s = m.settings
    return (id(m), id(m.label), s.mode, s.merge_same_colour, s.panel_max_mm,
            getattr(s, "backing_colour", None))


def clear_preview_cache() -> None:
    _LAYOUT_CACHE.clear()
    _MESH_CACHE.clear()


def preview_parts(m: Mosaic, panel_id=0, size: int = 900, az_deg: float = -30.0,
                  el_deg: float = 50.0, as_png: bool = False, quality: str = "full"):
    """Rendered 3D view of panel `panel_id` (0-based int or "P1"). Returns uint8 RGB
    (size, size, 3), or PNG bytes when as_png=True. Run it off the UI thread.

    quality="fast": the mosaic height raster of the panel's grid cell as a coloured surface,
                    no meshing (< 1 s) - show it first;
    quality="full": the real meshes (lattice backing + parts); meshes are cached per panel and
                    preview_key(m), so turning the view only re-renders (~0.5-1 s).
    The cache follows preview_key(m): a production-mode switch gives a fresh image."""
    from ..geometry.layout import build_layout
    from ..geometry.mesh import build_backing, build_part, mesh_arrays
    from .preview3d import heightfield_mesh, render_meshes
    s = m.settings
    if isinstance(panel_id, str):
        panel_id = int(panel_id.strip().upper().lstrip("P")) - 1
    key = preview_key(m)
    if quality == "fast":
        H, W = m.label.shape
        mm = s.mm_per_px
        nx, ny = panel_xy(W * mm, H * mm, s.panel_max_mm)
        k = max(0, min(int(panel_id), nx * ny - 1))
        r, c = divmod(k, nx)
        cw, ch = W * mm / nx, H * mm / ny
        x0, x1 = c * cw, (c + 1) * cw
        y1 = H * mm - r * ch
        y0 = y1 - ch
        v, t, rgb = heightfield_mesh(m, x0, y0, x1, y1, GeoParams().backing_t)
        meshes = [(v, t, rgb)]
    else:
        L = _LAYOUT_CACHE.get(key)
        if L is None:
            _LAYOUT_CACHE.clear()
            _MESH_CACHE.clear()
            L = build_layout(m)
            _LAYOUT_CACHE[key] = L
        pn = L.panels[max(0, min(int(panel_id), len(L.panels) - 1))]
        mk = (key, pn.index)
        meshes = _MESH_CACHE.get(mk)
        if meshes is None:
            pal = s.palette
            meshes = []
            bk = build_backing(L, pn, engrave=False, label_pockets=(L.mode == Mode.PIECES))
            v, t = mesh_arrays(bk)
            meshes.append((v, t, tuple(int(c * 0.8) for c in pal[L.backing_colour].rgb)))
            for i in pn.parts:
                ps = build_part(L, L.parts[i], engrave=False)
                if ps is None:
                    continue
                v, t = mesh_arrays(ps.solid)
                meshes.append((v + (0.0, 0.0, L.params.backing_t), t, pal[L.parts[i].colour].rgb))
            _MESH_CACHE[mk] = meshes
    img = render_meshes(meshes, size, az_deg, el_deg)
    if as_png:
        import io
        from PIL import Image
        bio = io.BytesIO()
        Image.fromarray(img).save(bio, "PNG")
        return bio.getvalue()
    return img


def scene_glb(m: Mosaic, mode=None, quality: str = "preview", progress=None):
    """GLB + meta of the artwork as printed (one node per stone). See tessera.export.scene."""
    from .scene import scene_glb as _f
    return _f(m, mode, quality, progress)


def plates_glb(m: Mosaic, mode=None, progress=None):
    """GLB + meta of the build plates exactly as exported. See tessera.export.scene."""
    from .scene import plates_glb as _f
    return _f(m, mode, progress=progress)
