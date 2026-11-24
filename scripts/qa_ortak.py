"""QA (critic team) shared helpers: build examples and check pipeline invariants.

Used by scripts/qa_*.py and tests/test_e2e.py / tests/test_robustness.py. Not product code.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tessera.model import Filament, Settings, VocabParams, default_vocab_params  # noqa: E402

POSTER_PALETTE = [("Kırmızı", (206, 32, 40)), ("Siyah", (24, 23, 25)), ("Krem", (243, 237, 224))]
EXAMPLES = ["poster", "murekkep_tram", "portre", "orman_masali", "logo_grafik", "demir_maske"]
VARIANTS = {   # file -> (zone source example or None for suggest_zones, k colours)
    "poster_2renk.png": ("poster", 2),
    "poster_3000px.png": ("poster", 3),
    "poster_5renk.png": ("poster", 5),
    "poster_600px.png": ("poster", 3),
    "poster_yuzler.png": (None, 3),
    "murekkep_90derece.png": (None, 3),
    "murekkep_kaba_tram.png": ("murekkep_tram", 3),
    "murekkep_tramsiz.png": ("murekkep_tram", 3),
}


def load_presets() -> dict:
    p = ROOT / "tessera" / "data" / "presets.json"
    if not p.exists():
        return {}
    return {d["ad"]: d for d in json.loads(p.read_text(encoding="utf-8"))["ayarlar"]}


def apply_preset(s: Settings, preset: dict | None) -> Settings:
    """Same mapping as the UI (vocab, figure_bump_mm, bevel_mm, gap_mm)."""
    if not preset:
        return s
    vocab = default_vocab_params()
    for k, v in preset.get("vocab", {}).items():
        base = vocab.get(int(k), VocabParams())
        vocab[int(k)] = VocabParams(**{**base.__dict__, **{kk: vv for kk, vv in v.items() if kk in base.__dict__}})
    s.vocab = vocab
    for key in ("figure_bump_mm", "bevel_mm", "gap_mm"):
        if key in preset:
            setattr(s, key, float(preset[key]))
    return s


def example_source(name: str) -> Path:
    d = ROOT / "ornekler" / name
    return next(d / f for f in ("kaynak.png", "kaynak.webp", "kaynak.jpg") if (d / f).exists())


def example_inputs(name: str, mm_per_px: float = 0.4, preset: bool = True):
    """(image, settings, zone, zone_doc) for ornekler/<name>, the way the UI's example loader would."""
    from tessera.pipeline import load_image, zones_from_shapes
    d = ROOT / "ornekler" / name
    zdoc = json.loads((d / "bolgeler.json").read_text(encoding="utf-8"))
    pal = zdoc.get("palet")
    pal = [(c["ad"], tuple(c["rgb"])) for c in pal] if pal else POSTER_PALETTE
    s = Settings(palette=[Filament(n, tuple(rgb)) for n, rgb in pal], mm_per_px=mm_per_px)
    s.width_mm = float(zdoc.get("genislik_mm", 424.0))
    pr_name = zdoc.get("onerilen_ayar") or ("Sovyet Afiş" if name == "poster" else None)
    if preset and pr_name:
        apply_preset(s, load_presets().get(pr_name))
    img = load_image(str(example_source(name)), s)
    h, w = img.shape[:2]
    s.default_vocab = int(zdoc.get("varsayilan", 0))
    if zdoc.get("isin_merkezi"):
        s.rays_center = (zdoc["isin_merkezi"][0] * w, zdoc["isin_merkezi"][1] * h)
    zone = zones_from_shapes((h, w), zdoc.get("sekiller", []), default=s.default_vocab)
    return img, s, zone, zdoc


def variant_inputs(fname: str, mm_per_px: float = 0.4):
    from tessera.pipeline import auto_palette, load_image, suggest_zones, zones_from_shapes
    zsrc, k = VARIANTS[fname]
    path = ROOT / "ornekler" / "varyantlar" / fname
    s = Settings(mm_per_px=mm_per_px, width_mm=424.0 if fname.startswith("poster") else 400.0)
    img = load_image(str(path), s)
    s.palette = auto_palette(img, k)
    h, w = img.shape[:2]
    if zsrc:
        zdoc = json.loads((ROOT / "ornekler" / zsrc / "bolgeler.json").read_text(encoding="utf-8"))
        s.default_vocab = int(zdoc.get("varsayilan", 0))
        if zdoc.get("isin_merkezi"):
            s.rays_center = (zdoc["isin_merkezi"][0] * w, zdoc["isin_merkezi"][1] * h)
        zone = zones_from_shapes((h, w), zdoc.get("sekiller", []), default=s.default_vocab)
    else:
        zone, info = suggest_zones(img, s)
        c = info.get("isin_merkezi")
        if c:
            s.rays_center = (c[0] * w, c[1] * h)
    return img, s, zone, path


# ------------------------------------------------------------------ invariants
def tile_components(label: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Number of 4-connected pieces per tile id (independent of the pipeline's own helper).
    ``mask`` limits the check to some pixels (e.g. ~gap)."""
    h, w = label.shape
    idx = np.arange(h * w).reshape(h, w)
    m = np.ones((h, w), bool) if mask is None else mask
    rows, cols = [], []
    for a, b in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                 ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        same = (label[a] == label[b]) & m[a] & m[b]
        rows.append(idx[a][same])
        cols.append(idx[b][same])
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    g = coo_matrix((np.ones(len(r), np.int8), (r, c)), shape=(h * w, h * w))
    _, comp = connected_components(g, directed=False)
    comp = comp.reshape(h, w)
    lab = label[m]
    cm = comp[m]
    pairs = np.unique(np.stack([lab, cm], 1), axis=0)
    n = int(label.max()) + 1
    return np.bincount(pairs[:, 0], minlength=n)


def tile_widths_mm(label: np.ndarray, mm: float) -> np.ndarray:
    """Approximate largest inscribed width per tile (mm), conservative by <= 1 px."""
    pad = np.pad(label, 1, constant_values=-1)
    c = pad[1:-1, 1:-1]
    bnd = ((pad[:-2, 1:-1] != c) | (pad[2:, 1:-1] != c) | (pad[1:-1, :-2] != c) | (pad[1:-1, 2:] != c))
    d = ndi.distance_transform_edt(~bnd)
    n = int(label.max()) + 1
    mx = ndi.maximum(d, label, index=np.arange(n))
    return (2 * np.floor(np.asarray(mx)) + 1) * mm


def check_mosaic(m) -> dict:
    """Pipeline invariants and printability numbers for one Mosaic."""
    s = m.settings
    mm = s.mm_per_px
    n = m.n_tiles
    comps = tile_components(m.label)
    present = np.bincount(m.label.ravel(), minlength=n) > 0
    comps_nogap = tile_components(m.label, ~m.gap)
    if m.polys is not None:      # R8: the outlines are the stones; the raster is only a coarse picture of them
        comps = np.array([1 if p.is_valid and p.geom_type == "Polygon" and not p.is_empty else 2 for p in m.polys])
        comps_nogap = comps
        present = np.ones(n, bool)
    widths = tile_widths_mm(m.label, mm)
    area_mm2 = m.tile_area_px * mm * mm
    real = area_mm2 >= 3.2
    hfin = bool(np.isfinite(m.height).all())
    pfin = bool(np.isfinite(m.tile_plane).all())
    min_feat = float(getattr(s, "min_feature_mm", 1.2))
    out = {
        "taş": n,
        "taş_gerçek(>=3.2mm2)": int(real.sum()),
        "bağlantısız_taş": int((comps[present] > 1).sum()),
        "bağlantısız_taş_derzsiz": int((comps_nogap[present] > 1).sum()),
        "boş_id": int((~present).sum()),
        "yükseklik_sonlu": hfin and pfin,
        "yükseklik_min": float(np.nanmin(m.height)),
        "yükseklik_max": float(np.nanmax(m.height)),
        "dar_taş_<min_feature": int((widths[present] < min_feat - 1e-6).sum()),
        "genişlik<3mm_%": round(100 * float((widths[present & real] < 3).mean()), 1) if (present & real).any() else 0,
        "genişlik<6mm_%": round(100 * float((widths[present & real] < 6).mean()), 1) if (present & real).any() else 0,
        "kırıntı_<3.2mm2": int((present & ~real).sum()),
        "boyut_mm": [round(v) for v in m.size_mm],
    }
    return out


def timed(fn, *a, **k):
    t = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t
