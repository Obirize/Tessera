"""Core data model shared by every Tessera module.

CONTRACT: other teams build against these classes. Change only with care and
update docs/MIMARI.md when you do.

Coordinate conventions
- Raster arrays are (rows, cols) = (y, x), y grows downward, 1 px = Settings.mm_per_px mm.
- 3D/mm space: x right, y UP (flipped from raster), z out of the wall. Origin = bottom-left of artwork.
- Heights are in mm above the backing surface (z = 0).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Optional

import numpy as np


class Vocab(IntEnum):
    """Piece vocabularies ("diller"). The int value is stored in zone maps."""
    STONES = 0   # irregular Voronoi stones (default)
    RAYS = 1     # long planks radiating from a centre point
    BLOCKS = 2   # square pixel blocks on a grid
    FIGURE = 3   # small stones, raised, tilted facets of a rounded body
    FOCAL = 4    # small blocks, highest relief, focal accent
    FLAT = 5     # flat cut-paper sheet: whole colour region is one piece


VOCAB_NAMES_TR = {
    Vocab.STONES: "Akışlı taş",
    Vocab.RAYS: "Işın tahtası",
    Vocab.BLOCKS: "Kare blok",
    Vocab.FIGURE: "Kabartma figür",
    Vocab.FOCAL: "Odak taşı",
    Vocab.FLAT: "Kesik levha",
}


class Mode:
    """Production modes."""
    PIECES = "parca"   # AMS KAPALI: every part single colour, printed one colour per plate, assembled
    AMS = "ams"        # AMS AÇIK: each panel printed as one multi-colour object (filament swaps)


def rgb_hex(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*[int(v) for v in rgb])


@dataclass
class Filament:
    name: str
    rgb: tuple[int, int, int]
    finish: str = "matte"      # matte | basic | silk
    brand: str = "Bambu"
    owned: bool = True

    @property
    def hex(self) -> str:
        return rgb_hex(self.rgb)


@dataclass
class VocabParams:
    tile_mm: float = 9.6          # nominal tile size
    base_height_mm: float = 3.0   # relief height of this vocabulary
    jitter_mm: float = 0.35       # random per-tile height jitter
    lloyd: int = 2


def default_vocab_params() -> dict[int, VocabParams]:
    # Mirrors the "Klasik mozaik" preset (tessera/data/presets.json, varsayilan=true): the first
    # prototype's look the user chose (R5-3). The relief-band solver keeps weight in budget.
    return {
        int(Vocab.STONES): VocabParams(tile_mm=9.6, base_height_mm=3.0, jitter_mm=0.35),
        int(Vocab.RAYS): VocabParams(tile_mm=23.0, base_height_mm=2.0, jitter_mm=0.35),
        int(Vocab.BLOCKS): VocabParams(tile_mm=8.4, base_height_mm=4.0, jitter_mm=0.35),
        int(Vocab.FIGURE): VocabParams(tile_mm=6.0, base_height_mm=9.0, jitter_mm=0.35),
        int(Vocab.FOCAL): VocabParams(tile_mm=5.2, base_height_mm=6.5, jitter_mm=0.35),
        int(Vocab.FLAT): VocabParams(tile_mm=0.0, base_height_mm=3.0, jitter_mm=0.0),
    }


@dataclass
class Settings:
    width_mm: float = 424.0               # physical width of the artwork; height follows the image
    mm_per_px: float = 0.4                # working raster resolution
    palette: list[Filament] = field(default_factory=list)
    source_colours: list = field(default_factory=list)   # the picture's colours (renk.find) + "hedef": palette index
    mode: str = Mode.PIECES
    vocab: dict[int, VocabParams] = field(default_factory=default_vocab_params)
    default_vocab: int = int(Vocab.STONES)
    style: str = "mozaik"                 # R8: "mozaik" | "kirik" | "blok" | "iri" (cut-first textures)
    stone_h_mm: float = 3.0               # stone thickness (flat tops)
    relief_mm: float = 0.0                # how far figure stones stand forward (0 = even surface)
    h_jitter_mm: float = 0.0              # random height difference between stones
    stone_scale: float = 1.0              # stone size factor
    pieces: object = None                 # how many stones: a number, "sekil" (every shape one stone), None (by texture)
    shape: Optional[dict] = None          # outline of the artwork (pipeline.kadraj), None = rectangle
    plate_colour: Optional[int] = None    # palette index of the one-piece base plate the stones sit in (None: no plate)
    detail_adapt: float = 0.7             # R7: 0 = uniform stones, 1 = small stones only where detail needs them
    min_stone_mm: float = 1.2            # R6-3: no printed stone narrower than this (nozzle limit)
    line_mode: str = "auto"               # R6-3: how sub-4 mm lines are kept: "auto"|"row"|"groove"|"step"
    backing: str = "yok"                  # R6-2: "yok" (glue to wall/board, paper template) | "kafes" (optional)
    gap_mm: float = 0.4                   # real gap between separate stones (no grout, no backing needed)
    bevel_mm: float = 0.45                # pillowed edge drop
    figure_bump_mm: float = 4.0         # extra rounding on FIGURE zones
    rays_center: Optional[tuple[float, float]] = None  # (x, y) px; None = auto/centre-top
    base_thickness_mm: float = 2.0        # backing / web under tiles in printed parts
    panel_max_mm: float = 240.0           # max panel size (X1C plate 256 mm)
    plate_mm: float = 256.0
    merge_same_colour: bool = False       # R6-1: every stone is its own printed piece (merging is legacy)
    plate_order: str = "panel"            # "panel": plates grouped by panel (assemble early) | "renk": max packing per colour
    base_colour: Optional[int] = None     # AMS-mode base plate colour (palette index); None = most neutral
    backing_colour: Optional[int] = None# palette index of the backing ("kafes"); None = darkest
    descreen: object = "auto"            # "auto" | True | False: smooth halftone dot screens before quantizing
    min_feature_mm: float = 1.6           # thinner colour features are widened to this (detail is kept as tiles)
    min_part_mm2: float = 12.0            # colour islands smaller than this are absorbed (visible in preview)
    seed: int = 7

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class Mosaic:
    """Result of the 2D/2.5D pipeline. Pure arrays; no meshes."""
    settings: Settings
    image_rgb: np.ndarray          # (H, W, 3) uint8, resized working image
    colour: np.ndarray             # (H, W) int, palette index per pixel
    zone: np.ndarray               # (H, W) int, Vocab per pixel
    label: np.ndarray              # (H, W) int, tile id per pixel (gap pixels keep their tile id)
    gap: np.ndarray                # (H, W) bool, hairline pixels
    height: np.ndarray             # (H, W) float mm, rendered surface (gap = 0)
    tile_colour: np.ndarray        # (N,) int
    tile_zone: np.ndarray          # (N,) int
    tile_area_px: np.ndarray       # (N,) int
    tile_centroid: np.ndarray      # (N, 2) float px (x, y)
    tile_plane: np.ndarray         # (N, 3) float: top z = a + b*(x-cx)*mm + c*(y-cy)*mm  -> (a, b, c), mm units
    stats: dict = field(default_factory=dict)
    # R6: separate-stone styles
    polys: Optional[list] = None                    # R8: exact stone outlines (shapely, mm, y down), one per tile
    style: Optional[str] = None                     # "klasik" | "smalti" | "freud" | "orman"
    tile_profile: Optional[np.ndarray] = None       # (N,) int: 0 flat, 1 ramp, 2 quarter-round (freud)
    tile_dir: Optional[np.ndarray] = None           # (N,) float radians: slope direction
    tile_hilo: Optional[np.ndarray] = None          # (N, 2) float mm: high/low heights of a profiled stone
    outline: Optional[object] = None                # the artwork's own outline (shapely, mm, y down), None = the rectangle
    plate: Optional[object] = None                  # outline of the one-piece base plate (shapely, mm, y down) or None
    grooves: list = field(default_factory=list)     # [{"pts": [[x, y] mm, y down], "width": mm, "depth": mm}]
    groove_depth: float = 0.0

    @property
    def n_tiles(self) -> int:
        return int(self.tile_colour.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return self.label.shape

    @property
    def size_mm(self) -> tuple[float, float]:
        h, w = self.label.shape
        return w * self.settings.mm_per_px, h * self.settings.mm_per_px

    @property
    def area_mm2(self) -> float:
        w, h = self.size_mm
        return w * h if self.outline is None else float(self.outline.area)

    @property
    def plate_colour(self) -> Optional[int]:
        return None if self.plate is None else self.settings.plate_colour

    def albedo(self, lattice_rgb=(178, 174, 166)) -> np.ndarray:
        """Per-pixel colour. Gaps show a neutral wall/board colour (R6-2: no dark backing assumed)."""
        pal = np.array([f.rgb for f in self.settings.palette], np.float32)
        a = pal[self.colour]
        a[self.gap] = lattice_rgb if self.plate is None else pal[self.plate_colour]
        return a
