"""Filament library for the UI.

Reads tessera/data/filaments.json (Algorithm team) when it exists; otherwise uses a small
built-in list. The JSON loader is tolerant: it accepts a list of filaments or {"filamentler": [...]}
/ {"filaments": [...]}, with English or Turkish keys and rgb as [r,g,b] or "#RRGGBB".
"""
from __future__ import annotations

import json
import os

from ..model import Filament

FINISH_TR = {"matte": "Mat", "basic": "Standart", "silk": "İpek", "glow": "Fosforlu", "metal": "Metalik",
             "translucent": "Yarı saydam", "petg": "PETG"}

# Approximate colours of common Bambu Lab PLA lines (eyeballed, for preview only).
BUILTIN: list[Filament] = [
    Filament("Fildişi Beyazı", (243, 237, 224), "matte"),
    Filament("Kemik Beyazı", (228, 220, 200), "matte"),
    Filament("Beyaz", (246, 246, 244), "basic"),
    Filament("Limon Sarısı", (240, 205, 50), "matte"),
    Filament("Mandalina", (233, 120, 40), "matte"),
    Filament("Kırmızı", (206, 32, 40), "matte"),
    Filament("Koyu Kırmızı", (140, 30, 36), "matte"),
    Filament("Bordo", (110, 32, 44), "basic"),
    Filament("Terakota", (176, 84, 58), "matte"),
    Filament("Aşı Boyası", (150, 70, 50), "matte"),
    Filament("Kum", (200, 175, 135), "matte"),
    Filament("Kahve", (90, 60, 42), "matte"),
    Filament("Çimen Yeşili", (95, 140, 70), "matte"),
    Filament("Koyu Yeşil", (40, 80, 55), "matte"),
    Filament("Buz Mavisi", (150, 190, 210), "matte"),
    Filament("Gök Mavisi", (60, 120, 190), "matte"),
    Filament("Lacivert", (30, 40, 80), "matte"),
    Filament("Beton Grisi", (150, 148, 144), "matte"),
    Filament("Kül Grisi", (95, 95, 98), "matte"),
    Filament("Kömür", (50, 50, 52), "matte"),
    Filament("Siyah", (24, 23, 25), "matte"),
    Filament("Altın", (200, 160, 70), "silk"),
    Filament("Bakır", (180, 105, 70), "silk"),
    Filament("Gümüş", (190, 192, 196), "silk"),
]

# The palette of the poster example (same colours as scripts/smoke.py).
POSTER_PALETTE = ["Kırmızı", "Siyah", "Fildişi Beyazı"]


def _data_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "filaments.json")


def _parse_rgb(v) -> tuple[int, int, int] | None:
    if isinstance(v, str):
        v = v.strip().lstrip("#")
        if len(v) >= 6:
            try:
                return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
            except ValueError:
                return None
        return None
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return tuple(int(c) for c in v[:3])  # type: ignore[return-value]
    return None


def _parse(item: dict) -> Filament | None:
    if not isinstance(item, dict):
        return None
    name = item.get("name") or item.get("ad") or item.get("isim")
    rgb = _parse_rgb(item.get("rgb") or item.get("onizleme_hex") or item.get("hex") or item.get("renk"))
    if not name or rgb is None:
        return None
    finish = str(item.get("finish") or item.get("yuzey") or item.get("seri") or "matte").lower()
    brand = item.get("brand") or item.get("marka") or "Bambu"
    return Filament(str(name), rgb, finish, str(brand), owned=bool(item.get("owned", False)))


def load_library() -> tuple[list[Filament], bool]:
    """-> (filaments, from_file)."""
    p = _data_path()
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
            items = d if isinstance(d, list) else (d.get("filamentler") or d.get("filaments") or [])
            out = [f for f in (_parse(i) for i in items) if f is not None]
            if out:
                return out, True
        except (OSError, ValueError, AttributeError, TypeError):
            pass
    return [Filament(f.name, f.rgb, f.finish, f.brand, False) for f in BUILTIN], False


def find(lib: list[Filament], name: str) -> Filament | None:
    for f in lib:
        if f.name == name:
            return f
    return None


def nearest(lib: list[Filament], rgb) -> Filament:
    r, g, b = rgb
    return min(lib, key=lambda f: (f.rgb[0] - r) ** 2 * 0.3 + (f.rgb[1] - g) ** 2 * 0.59 + (f.rgb[2] - b) ** 2 * 0.11)


def luminance(rgb) -> float:
    return 0.30 * rgb[0] + 0.59 * rgb[1] + 0.11 * rgb[2]


# ------------------------------------------------------------------ presets (Art team)
def load_presets() -> list[dict]:
    """tessera/data/presets.json -> list of {"ad", "aciklama", "vocab", ...}; [] if missing."""
    p = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "presets.json")
    try:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        items = d.get("ayarlar", []) if isinstance(d, dict) else d
        return [i for i in items if isinstance(i, dict) and i.get("ad")]
    except (OSError, ValueError):
        return []


def preset_label(pr: dict) -> str:
    """Name shown to the user (R2-5: plain Turkish)."""
    return str(pr.get("ad", "")).replace("Monumental", "Anıtsal")


def default_preset(presets: list[dict]) -> dict | None:
    """The Art team's recommended preset for a new image: flagged with varsayilan/default,
    otherwise the first one."""
    for pr in presets:
        if pr.get("varsayilan") is True or pr.get("default") is True or pr.get("varsayilan_mi") is True:
            return pr
    return presets[0] if presets else None


def find_preset(presets: list[dict], name: str) -> dict | None:
    for pr in presets:
        if pr.get("ad") == name or preset_label(pr) == name:
            return pr
    return None
