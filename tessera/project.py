"""Tessera project file (``*.tessera.json``) — owned by the UI team.

A project stores everything needed to rebuild a mosaic:
- the source image path (absolute + relative to the project file),
- Settings (Settings.to_dict, palette included),
- zone shapes in fractional 0..1 coordinates (same format as ornekler/poster/bolgeler.json),
- the rays centre as a fraction of width/height (Settings.rays_center is in px and depends on
  the working resolution, so it is not stored there),
- optional painted zone mask, saved as a PNG next to the project file.

Painted mask PNG (mode "L"): value 0..5 = Vocab painted there, PAINT_DEFAULT (254) = "back to the
default vocabulary" (eraser), PAINT_NONE (255) = no paint (shapes / default show through).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .model import Filament, Mode, Settings, VocabParams, Vocab, default_vocab_params

FORMAT_VERSION = 1
PROJECT_SUFFIX = ".tessera.json"
MASK_SUFFIX = ".bolgeler.png"
PAINT_DEFAULT = 254
PAINT_NONE = 255
GENERIC_NAMES = {"kaynak", "gorsel", "görsel", "resim", "image", "img", "source", "original", "orijinal"}


# ------------------------------------------------------------------ settings <-> json
def settings_to_json(s: Settings) -> dict:
    d = s.to_dict()
    d["vocab"] = {str(k): v for k, v in d["vocab"].items()}
    d["palette"] = [{"name": f.name, "rgb": list(f.rgb), "finish": f.finish, "brand": f.brand,
                     "owned": f.owned} for f in s.palette]
    d.pop("rays_center", None)  # stored fractionally in Project.rays_center
    return d


def settings_from_json(d: dict) -> Settings:
    s = Settings()
    known = set(Settings.__dataclass_fields__)
    for k, v in (d or {}).items():
        if k not in known or k in ("palette", "vocab", "rays_center"):
            continue
        setattr(s, k, v)
    s.palette = [Filament(name=f["name"], rgb=tuple(int(c) for c in f["rgb"]),
                          finish=f.get("finish", "matte"), brand=f.get("brand", "Bambu"),
                          owned=bool(f.get("owned", True)))
                 for f in (d or {}).get("palette", [])]
    vocab = default_vocab_params()
    vp_fields = set(VocabParams.__dataclass_fields__)
    for k, v in (d or {}).get("vocab", {}).items():
        vocab[int(k)] = VocabParams(**{kk: vv for kk, vv in v.items() if kk in vp_fields})
    s.vocab = vocab
    if s.mode not in (Mode.PIECES, Mode.AMS):
        s.mode = Mode.PIECES
    return s


# ------------------------------------------------------------------ project
@dataclass
class Project:
    image_path: str = ""
    settings: Settings = field(default_factory=Settings)
    shapes: list[dict] = field(default_factory=list)           # fractional zone shapes
    rays_center: Optional[tuple[float, float]] = None          # fractional (x, y)
    owned: list[str] = field(default_factory=list)             # filament names the user owns
    light_az: float = -130.0
    light_el: float = 40.0
    export_dir: str = ""
    name: str = ""
    preset: str = ""                                           # Art-team preset name ("" = none)
    image_size: Optional[tuple[int, int]] = None               # (w, h) px, kept even if the image goes missing
    preset_modified: bool = False                              # user moved a slider after applying it
    paint_mask: Optional[np.ndarray] = None                    # uint8 (h, w) or None
    path: str = ""                                             # where it was saved/loaded (not stored)

    # -------------------------------------------------- helpers
    @property
    def has_paint(self) -> bool:
        return self.paint_mask is not None and bool((self.paint_mask != PAINT_NONE).any())

    def display_name(self) -> str:
        if self.name:
            return self.name
        if self.path:
            return os.path.basename(self.path)[: -len(PROJECT_SUFFIX)] if self.path.endswith(
                PROJECT_SUFFIX) else os.path.splitext(os.path.basename(self.path))[0]
        if self.image_path:
            stem = os.path.splitext(os.path.basename(self.image_path))[0]
            if stem.lower() in GENERIC_NAMES:      # "kaynak.png" -> its folder ("portre")
                parent = os.path.basename(os.path.dirname(os.path.abspath(self.image_path)))
                if parent:
                    return parent
            return stem
        return "Adsız proje"

    @staticmethod
    def mask_path_for(project_path: str) -> str:
        base = project_path[: -len(PROJECT_SUFFIX)] if project_path.endswith(PROJECT_SUFFIX) \
            else os.path.splitext(project_path)[0]
        return base + MASK_SUFFIX

    # -------------------------------------------------- json
    def to_json(self, project_path: str = "") -> dict:
        rel = ""
        if self.image_path and project_path:
            try:
                rel = os.path.relpath(self.image_path, os.path.dirname(os.path.abspath(project_path)))
            except ValueError:  # different drive on Windows
                rel = ""
        mask_file = ""
        if self.has_paint and project_path:
            mask_file = os.path.basename(self.mask_path_for(project_path))
        return {
            "tessera": FORMAT_VERSION,
            "ad": self.name,
            "gorsel": os.path.abspath(self.image_path) if self.image_path else "",
            "gorsel_goreli": rel.replace("\\", "/"),
            "gorsel_boyut": list(self.image_size) if self.image_size else None,
            "ayarlar": settings_to_json(self.settings),
            "bolgeler": {
                "varsayilan": int(self.settings.default_vocab),
                "isin_merkezi": list(self.rays_center) if self.rays_center else None,
                "sekiller": self.shapes,
                "boyali_maske": mask_file,
            },
            "sahip_olduklarim": list(self.owned),
            "isik": {"az": self.light_az, "el": self.light_el},
            "disa_aktarim_klasoru": self.export_dir,
            "hazir_ayar": {"ad": self.preset, "degistirildi": self.preset_modified},
        }

    @classmethod
    def from_json(cls, d: dict, project_path: str = "") -> "Project":
        if not isinstance(d, dict) or "tessera" not in d:
            raise ValueError("Bu dosya bir Tessera projesi değil.")
        p = cls()
        p.name = d.get("ad", "") or ""
        p.settings = settings_from_json(d.get("ayarlar", {}))
        z = d.get("bolgeler", {}) or {}
        p.settings.default_vocab = int(z.get("varsayilan", p.settings.default_vocab))
        rc = z.get("isin_merkezi")
        p.rays_center = (float(rc[0]), float(rc[1])) if rc else None
        p.shapes = list(z.get("sekiller", []))
        p.owned = list(d.get("sahip_olduklarim", []))
        light = d.get("isik", {}) or {}
        p.light_az = float(light.get("az", -130.0))
        p.light_el = float(light.get("el", 40.0))
        p.export_dir = d.get("disa_aktarim_klasoru", "") or ""
        ha = d.get("hazir_ayar") or {}
        p.preset = str(ha.get("ad", "") or "")
        gb = d.get("gorsel_boyut")
        p.image_size = (int(gb[0]), int(gb[1])) if gb else None
        p.preset_modified = bool(ha.get("degistirildi", False))
        img = d.get("gorsel", "") or ""
        rel = d.get("gorsel_goreli", "") or ""
        if project_path and rel:
            cand = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(project_path)), rel))
            if os.path.exists(cand) or not img:
                img = cand  # project folder moved together with its image
        p.image_path = img
        mask_file = z.get("boyali_maske", "")
        if mask_file and project_path:
            mp = os.path.join(os.path.dirname(os.path.abspath(project_path)), mask_file)
            if os.path.exists(mp):
                from PIL import Image
                p.paint_mask = np.asarray(Image.open(mp).convert("L")).copy()
        p.path = project_path
        return p

    # -------------------------------------------------- files
    def save(self, path: str) -> None:
        path = os.path.abspath(path)
        data = self.to_json(path)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
        mp = self.mask_path_for(path)
        if self.has_paint:
            from PIL import Image
            Image.fromarray(self.paint_mask.astype(np.uint8), "L").save(mp)
        elif os.path.exists(mp):
            os.remove(mp)
        self.path = path

    @classmethod
    def load(cls, path: str) -> "Project":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return cls.from_json(d, os.path.abspath(path))

    # -------------------------------------------------- zones
    def zone_map(self, shape_hw) -> np.ndarray:
        """Final Vocab map at (h, w): shapes, then the painted mask on top."""
        from .pipeline import zones_from_shapes
        h, w = shape_hw
        default = int(self.settings.default_vocab)
        zone = zones_from_shapes((h, w), self.shapes, default=default)
        return apply_paint(zone, self.paint_mask, default)


def apply_paint(zone: np.ndarray, mask: Optional[np.ndarray], default: int) -> np.ndarray:
    if mask is None:
        return zone
    h, w = zone.shape
    if mask.shape != (h, w):
        from PIL import Image
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8), "L").resize((w, h), Image.NEAREST))
    out = zone.copy()
    painted = mask < len(Vocab)
    out[painted] = mask[painted]
    out[mask == PAINT_DEFAULT] = default
    return out
