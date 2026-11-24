"""Algorithm team tests: tessera.render (top and oblique previews)."""
import time

import numpy as np
import pytest

from tessera.model import Filament, Mosaic, Settings
from tessera.render import render_oblique, render_top


def _block_mosaic(h=120, w=160, block_h=6.0, flat_h=1.0):
    """Synthetic mosaic: flat cream floor (height ``flat_h``) with one raised red block."""
    s = Settings(palette=[Filament("a", (230, 230, 230)), Filament("b", (200, 40, 40))])
    colour = np.zeros((h, w), np.int64)
    height = np.full((h, w), flat_h, np.float32)
    colour[40:80, 50:90] = 1
    height[40:80, 50:90] = block_h
    label = colour.copy()
    gap = np.zeros((h, w), bool)
    n = 2
    return Mosaic(settings=s, image_rgb=np.zeros((h, w, 3), np.uint8), colour=colour,
                  zone=np.zeros((h, w), np.int64), label=label, gap=gap, height=height,
                  tile_colour=np.arange(n), tile_zone=np.zeros(n, int), tile_area_px=np.bincount(label.ravel()),
                  tile_centroid=np.zeros((n, 2)), tile_plane=np.zeros((n, 3)))


def test_shape_and_type():
    m = _block_mosaic()
    top = render_top(m)
    assert top.shape == (120, 160, 3) and top.dtype == np.uint8
    ob = render_oblique(top, m)
    assert ob.ndim == 3 and ob.shape[1] == 160 and ob.dtype == np.uint8


def test_no_wraparound_band_on_top_edge():
    """np.roll used to wrap the bottom rows' heights to the top: a shadow band appeared."""
    m = _block_mosaic()
    m.height[100:, :] = 12.0           # tall wall at the bottom edge
    top = render_top(m, az_deg=-90, el_deg=30).astype(int)   # light from the top (towards -y)
    # first rows (flat floor, nothing above them) must be as bright as the rows just below
    assert abs(top[0:3, :, 0].mean() - top[5:10, :, 0].mean()) < 3
    m2 = _block_mosaic()
    top2 = render_top(m2, az_deg=180, el_deg=30).astype(int)   # light from the left
    assert abs(top2[:, -3:, 0].mean() - top2[:, -10:-5, 0].mean()) < 3


def test_shadow_is_cast_away_from_light_and_soft():
    m = _block_mosaic()
    hard = render_top(m, az_deg=180, el_deg=25, light_size_deg=0.0).astype(float)
    soft = render_top(m, az_deg=180, el_deg=25, light_size_deg=6.0).astype(float)
    floor_lit = hard[60, 20, 0]                 # floor left of the block (towards the light)
    floor_shadow = hard[60, 95, 0]              # floor just right of the block
    assert floor_shadow < floor_lit - 20
    # soft light: a wider band of intermediate values (penumbra) along the shadow's far end
    row_h, row_s = hard[60, 90:150, 0], soft[60, 90:150, 0]
    mid = lambda r: int(((r > floor_shadow + 5) & (r < floor_lit - 5)).sum())
    assert mid(row_s) > mid(row_h)


def test_preview_scale_is_smaller_and_faster():
    m = _block_mosaic(480, 640)
    t = time.perf_counter()
    full = render_top(m)
    t_full = time.perf_counter() - t
    t = time.perf_counter()
    half = render_top(m, preview_scale=0.5)
    t_half = time.perf_counter() - t
    assert half.shape == (240, 320, 3)
    assert t_half < t_full
    # similar overall look
    assert abs(full.mean() - half.mean()) < 8
    ob = render_oblique(half, m)                # oblique follows the reduced top view
    assert ob.shape[1] == 320


def test_deterministic():
    m = _block_mosaic()
    assert np.array_equal(render_top(m), render_top(m))
