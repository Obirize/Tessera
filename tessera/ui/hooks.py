"""Optional Algorithm-team functions, called defensively (they may not exist yet, and their exact
return types are not fixed in the contract). See docs/istekler.md for the signatures we ask for.
"""
from __future__ import annotations

import inspect
from typing import Callable, Optional

import numpy as np

from ..model import Filament, Settings


def suggest_zones_fn() -> Optional[Callable]:
    try:
        from .. import pipeline
        fn = getattr(pipeline, "suggest_zones", None)
        if fn is None:
            from ..pipeline import zones as _z  # type: ignore[attr-defined]
            fn = getattr(_z, "suggest_zones", None)
        return fn if callable(fn) else None
    except Exception:  # noqa: BLE001
        return None


def suggest_palette_fn() -> Optional[Callable]:
    try:
        from ..pipeline import palette as _p  # type: ignore[attr-defined]
        fn = getattr(_p, "suggest_palette", None)
        return fn if callable(fn) else None
    except Exception:  # noqa: BLE001
        return None


def _call_flexible(fn: Callable, candidates: list[tuple[tuple, dict]]):
    last = None
    for args, kw in candidates:
        try:
            inspect.signature(fn).bind(*args, **kw)
        except TypeError as e:
            last = e
            continue
        except ValueError:
            pass
        return fn(*args, **kw)
    raise last or TypeError("uyumsuz fonksiyon")


def run_suggest_zones(fn: Callable, img: np.ndarray, settings: Settings) -> dict:
    """-> dict(shapes=list|None, mask=ndarray|None, rays_center=(fx,fy)|None, default=int|None)."""
    out = _call_flexible(fn, [((img, settings), {}), ((img,), {})])
    h, w = img.shape[:2]
    res = {"shapes": None, "mask": None, "rays_center": None, "default": None, "reasons": []}
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], np.ndarray):
        out = {"zone": out[0], **out[1]} if isinstance(out[1], dict) else {"zone": out[0], "rays_center": out[1]}
    if isinstance(out, np.ndarray):
        out = {"zone": out}
    if isinstance(out, list):
        out = {"shapes": out}
    if not isinstance(out, dict):
        raise ValueError("Otomatik bölge önerisi anlaşılamadı.")
    shapes = out.get("sekiller", out.get("shapes"))
    if shapes is not None:
        res["shapes"] = list(shapes)
    zone = out.get("zone", out.get("bolge"))
    if isinstance(zone, np.ndarray):
        res["mask"] = zone.astype(np.uint8)
        if res["shapes"] is None:
            res["shapes"] = []
    d = out.get("varsayilan", out.get("default"))
    if d is not None:
        res["default"] = int(d)
    reasons = out.get("nedenler", out.get("reasons")) or []
    res["reasons"] = [str(r) for r in reasons] if isinstance(reasons, (list, tuple)) else [str(reasons)]
    rc = out.get("isin_merkezi", out.get("rays_center"))
    if rc is not None:
        fx, fy = float(rc[0]), float(rc[1])
        if fx > 1.5 or fy > 1.5:          # pixels -> fractions
            fx, fy = fx / w, fy / h
        res["rays_center"] = (fx, fy)
    return res


def palette_roles(out) -> dict[str, str]:
    """Role names per filament name from a suggest_palette result (if it has them)."""
    roles = {}
    for item in out if isinstance(out, list) else []:
        if isinstance(item, dict) and item.get("rol"):
            f = item.get("filament")
            nm = f.name if isinstance(f, Filament) else (item.get("name") or item.get("ad"))
            if nm:
                roles[nm] = str(item["rol"])
    return roles


def run_suggest_palette(fn: Optional[Callable], img: np.ndarray, library: list[Filament],
                        n: int, owned_only: bool) -> tuple[list[Filament], dict[str, str]]:
    cands = [f for f in library if f.owned] if owned_only else list(library)
    if len(cands) < 2:
        cands = list(library)
    if fn is not None:
        out = _call_flexible(fn, [((img, cands), {"k": n}), ((img, cands), {"n": n}), ((img, cands, n), {}),
                                  ((img, cands), {}), ((img,), {"n": n}), ((img,), {})])
        return _to_filaments(out, cands, library)[:max(2, n)], palette_roles(out)
    return fallback_palette(img, cands, n), {}


def _to_filaments(out, cands: list[Filament], library: list[Filament]) -> list[Filament]:
    from .filaments import nearest
    if isinstance(out, dict):
        out = out.get("palette") or out.get("palet") or out.get("filaments") or []
    res: list[Filament] = []
    for item in out or []:
        f = None
        if isinstance(item, dict) and isinstance(item.get("filament"), Filament):
            item = item["filament"]
        if isinstance(item, Filament):
            f = next((c for c in library if c.name == item.name), item)
        elif isinstance(item, (int, np.integer)):
            f = cands[int(item)] if 0 <= int(item) < len(cands) else None
        elif isinstance(item, str):
            f = next((c for c in library if c.name == item), None)
        elif isinstance(item, dict) and ("name" in item or "ad" in item):
            nm = item.get("name") or item.get("ad")
            f = next((c for c in library if c.name == nm), None)
        elif isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 3:
            f = nearest(cands, tuple(int(c) for c in item[:3]))
        if f is not None and f not in res:
            res.append(f)
    if len(res) < 2:
        raise ValueError("Palet önerisi boş döndü.")
    return res


def fallback_palette(img: np.ndarray, cands: list[Filament], n: int) -> list[Filament]:
    """Simple local suggestion: dominant colours (median cut) -> nearest distinct filaments."""
    from PIL import Image
    from .filaments import luminance, nearest
    small = Image.fromarray(img).convert("RGB")
    small.thumbnail((160, 160))
    q = small.quantize(colors=max(n * 3, 8), method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette()[: 3 * max(n * 3, 8)]
    counts = sorted(q.getcolors(), reverse=True)
    out: list[Filament] = []
    for _, idx in counts:
        rgb = tuple(pal[3 * idx: 3 * idx + 3])
        f = nearest([c for c in cands if c not in out] or cands, rgb)
        if f not in out:
            out.append(f)
        if len(out) >= n:
            break
    # always keep a dark and a light anchor if the image has contrast
    out.sort(key=lambda f: luminance(f.rgb))
    return out
