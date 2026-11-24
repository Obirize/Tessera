"""Background work for the Tessera v2 interface (tessera/app2.py). No widgets here.

- ``Recipe``: everything the user decided (image, width, spools, style, mode, fine settings).
  It is plain data so it can travel to a child process and into a project file.
- ``Engine``: one long-lived worker thread. Only the newest build matters (a slider drag never
  queues work). After a build it renders the style thumbnails, at a lower priority.
- ``ChildJob``: runs heavy work (export, exact part meshes) in a separate process, so the
  window and the 3D stage never wait for the GIL.
"""
from __future__ import annotations

import base64
import copy
import io
import json
import os
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .model import Mode, Settings, VocabParams, Vocab
from .project import PAINT_NONE, Project, apply_paint

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# The four textures (R8): card name, Settings.style, one honest sentence. The picture is always cut the
# same way (one colour per stone, eyes and lips whole); the texture only decides how large areas divide.
STYLE_CARDS = [
    ("Mozaik", "mozaik", "Taş sıraları şekillerin kenarını izler. En dengeli görünüm."),
    ("Kırık taş", "kirik", "Gelişigüzel kırılmış küçük taşlar. En çok taş bu dokuda çıkar."),
    ("Blok", "blok", "Tuğla gibi düzgün, sıralı bloklar."),
    ("İri parça", "iri", "Az sayıda büyük parça. En hızlı basılan ve dizilen doku."),
]
DEFAULT_STYLE = "Mozaik"
STYLE_CODES = {name: code for name, code, _ in STYLE_CARDS}
OLD_STYLES = {"klasik": "Mozaik", "Klasik taş": "Mozaik", "smalti": "Mozaik", "Smalti": "Mozaik",
              "freud": "Blok", "Freud blok": "Blok", "orman": "İri parça", "Orman katmanı": "İri parça"}


def style_name(value: str) -> str:
    """Card name for a stored style (card name, style code or an old pre-R6 preset name)."""
    if value in STYLE_CODES:
        return value
    for name, code, _ in STYLE_CARDS:
        if value == code:
            return name
    return OLD_STYLES.get(value, DEFAULT_STYLE)


STARTER_SPOOLS = ["Siyah", "Kırmızı", "Fildişi Beyazı", "Yeşim Beyazı", "Gri", "Altın", "Mavi",
                  "Kakao Kahve"]


# ------------------------------------------------------------------ friendly errors
_TR = set("çğıöşüÇĞİÖŞÜ")


def friendly_error(e: BaseException) -> str:
    """Plain Turkish message; never raw English exception text."""
    fn = getattr(e, "filename", None)
    if isinstance(e, FileNotFoundError):
        return f"Dosya bulunamadı: {os.path.normpath(str(fn))}" if fn else "Dosya bulunamadı."
    if isinstance(e, PermissionError):
        return "Bu dosyaya ya da klasöre yazma izni yok. Başka bir klasör seç."
    if isinstance(e, MemoryError):
        return "Bellek yetmedi. Eseri küçültmeyi ya da daha az renk kullanmayı dene."
    msg = str(e).strip()
    if msg and any(c in _TR for c in msg):
        return msg
    if isinstance(e, OSError):
        return "Dosya okunurken ya da yazılırken bir sorun oluştu."
    return "Beklenmeyen bir sorun oluştu. Ayarı biraz değiştirip yeniden dene."


# ------------------------------------------------------------------ library & presets
_LIB: Optional[list] = None
_PRESETS: Optional[list] = None


def library() -> list[dict]:
    global _LIB
    if _LIB is None:
        from .pipeline.palette import load_filaments
        _LIB = load_filaments()
    return _LIB


def lib_by_name() -> dict[str, dict]:
    return {e["ad"]: e for e in library()}


def presets() -> list[dict]:
    global _PRESETS
    if _PRESETS is None:
        with open(os.path.join(DATA_DIR, "presets.json"), encoding="utf-8") as fh:
            d = json.load(fh)
        _PRESETS = d.get("ayarlar", []) if isinstance(d, dict) else list(d)
    return _PRESETS


def style_cards() -> list[dict]:
    return [{"name": n, "code": c, "aciklama": d} for n, c, d in STYLE_CARDS]


# Plain colour words for the interface (the product name appears only in tooltips).
PLAIN_NAMES = {
    "Yeşim Beyazı": "Beyaz", "Fildişi Beyazı": "Beyaz", "Kemik Beyazı": "Bej", "Bej": "Bej", "Çöl Beji": "Bej",
    "Siyah": "Siyah", "Kömür Siyahı": "Siyah",
    "Gümüş": "Gümüş", "Gri": "Gri", "Kül Gri": "Gri", "Nardo Gri": "Gri", "Mavi Gri": "Gri",
    "Açık Gri": "Açık gri", "Koyu Gri": "Koyu gri",
    "Kırmızı": "Kırmızı", "Al Kırmızı": "Kırmızı", "Koyu Kırmızı": "Kırmızı", "Bordo": "Bordo", "Kiremit": "Kiremit",
    "Macenta": "Pembe", "Pembe": "Pembe", "Fuşya Pembe": "Pembe", "Sakura Pembe": "Açık pembe",
    "Turuncu": "Turuncu", "Balkabağı Turuncu": "Turuncu", "Mandalina Turuncu": "Turuncu",
    "Sarı": "Sarı", "Ayçiçeği Sarısı": "Sarı", "Limon Sarısı": "Sarı", "Altın": "Altın",
    "Bambu Yeşili": "Yeşil", "Ökseotu Yeşili": "Yeşil", "Çimen Yeşili": "Yeşil", "Parlak Yeşil": "Açık yeşil",
    "Elma Yeşili": "Açık yeşil", "Koyu Yeşil": "Koyu yeşil", "Bronz": "Haki",
    "Mavi": "Mavi", "Kobalt Mavi": "Mavi", "Deniz Mavisi": "Mavi", "Camgöbeği": "Açık mavi",
    "Gök Mavisi": "Açık mavi", "Buz Mavisi": "Açık mavi", "Turkuaz": "Turkuaz", "Lacivert": "Lacivert",
    "Mor": "Mor", "Çivit Moru": "Mor", "Erik Moru": "Mor", "Leylak Moru": "Lila",
    "Kahverengi": "Kahverengi", "Kakao Kahve": "Kahverengi", "Koyu Kahve": "Kahverengi",
    "Bitter Çikolata": "Koyu kahve", "Sütlü Kahve": "Açık kahve", "Karamel": "Açık kahve",
}


def plain_from_rgb(rgb) -> str:
    """Fallback for spools that are not in the table: a word from hue and lightness."""
    import colorsys
    r, g, b = [c / 255 for c in rgb]
    h, l, sat = colorsys.rgb_to_hls(r, g, b)
    if sat < 0.15 or l < 0.08 or l > 0.93:
        if l > 0.85:
            return "Beyaz"
        if l < 0.18:
            return "Siyah"
        return "Açık gri" if l > 0.62 else ("Koyu gri" if l < 0.38 else "Gri")
    deg = h * 360
    for lim, word in ((15, "Kırmızı"), (40, "Turuncu"), (65, "Sarı"), (165, "Yeşil"), (200, "Turkuaz"),
                      (250, "Mavi"), (290, "Mor"), (340, "Pembe"), (361, "Kırmızı")):
        if deg < lim:
            if word in ("Turuncu", "Kırmızı") and l < 0.35:
                return "Kahverengi"
            return word
    return "Renk"


def plain_name(name: str) -> str:
    if name in PLAIN_NAMES:
        return PLAIN_NAMES[name]
    e = lib_by_name().get(name)
    if e is not None:
        h = (e.get("onizleme_hex") or e["hex"]).lstrip("#")
        return plain_from_rgb((int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)))
    return name


def plain_labels(names) -> dict:
    """{product name: plain label}, unique within ``names``: when two spools share a word, the
    matte one says so ("Mat beyaz"); any remaining tie gets a number."""
    names = list(dict.fromkeys(names))
    lab = {n: plain_name(n) for n in names}
    seen: dict = {}
    for n in names:
        seen.setdefault(lab[n], []).append(n)
    for word, group in seen.items():
        if len(group) < 2:
            continue
        for n in group:
            e = lib_by_name().get(n) or {}
            if e.get("seri") == "PLA Matte":
                lab[n] = "Mat " + word[0].lower() + word[1:]
        again: dict = {}
        for n in group:
            again.setdefault(lab[n], []).append(n)
        for w, g2 in again.items():
            for i, n in enumerate(g2[1:], start=2):
                lab[n] = f"{w} {i}"
    return lab


def filament_for(name: str):
    from .model import Filament
    from .pipeline.palette import to_filament
    e = lib_by_name().get(name)
    if e is None:
        return Filament(name, (128, 128, 128))
    return to_filament(e)


# ------------------------------------------------------------------ recipe
FINE_DEFAULTS = {
    "tas": 1.0,                # stone size factor
    "derz": 0.4,               # joint between stones, mm
    "kalinlik": 3.0,           # stone thickness, mm
    "kabartma": 0.0,           # figures stand forward by this much, mm
    "oynama": 0.0,             # random height difference between stones, mm
    "panel": 240.0,            # AMS panel size, mm
    "taban": None,             # AMS base plate colour (palette index; None = most neutral)
    "tram": "auto",            # dotted (printed) pictures: auto | True | False
    "otomatik_bolge": True,    # find rays/figures on pictures without zones
    "tabla_sirasi": "panel",   # plates in template order | fewest plates
    "renk_sayisi": 4,          # spools used by "Bu görsel için öner"
    "tohum": 7,                # another arrangement of the same stones
}
LIGHT_AZ, LIGHT_EL = -125.0, 36.0  # the one fixed light (upper left, raking); no light controls


@dataclass
class Recipe:
    image_path: str = ""
    width_mm: float = 424.0
    palette: list = field(default_factory=list)       # filament names
    style: str = DEFAULT_STYLE
    mode: str = Mode.PIECES
    fine: dict = field(default_factory=lambda: dict(FINE_DEFAULTS))
    shapes: list = field(default_factory=list)        # fractional zone shapes (bolgeler.json)
    zones_from_file: bool = False
    file_default_vocab: Optional[int] = None
    rays_center: Optional[list] = None                # fractional (x, y)
    zone_mask: Optional[np.ndarray] = None            # uint8 paint mask (auto zones) or None
    light_az: float = -130.0
    light_el: float = 40.0
    name: str = ""

    def copy(self) -> "Recipe":
        r = copy.copy(self)
        r.palette = list(self.palette)
        r.fine = dict(self.fine)
        r.shapes = copy.deepcopy(self.shapes)
        r.zone_mask = None if self.zone_mask is None else self.zone_mask.copy()
        return r

    def build_key(self) -> str:
        import hashlib
        h = hashlib.sha1()
        d = {k: v for k, v in asdict(self).items() if k not in ("zone_mask", "light_az", "light_el", "name")}
        h.update(json.dumps(d, sort_keys=True, default=str).encode())
        if self.zone_mask is not None:
            h.update(self.zone_mask.tobytes())
        return h.hexdigest()

    # -------------------------------------------------- settings
    def settings(self, preview: bool = True) -> Settings:
        from .pipeline import working_mm_per_px
        s = Settings()
        s.width_mm = float(self.width_mm)
        s.palette = [filament_for(n) for n in self.palette]
        s.mode = self.mode if self.mode in (Mode.PIECES, Mode.AMS) else Mode.PIECES
        s.style = STYLE_CODES.get(style_name(self.style), "mozaik")
        if self.zones_from_file and self.file_default_vocab is not None:
            s.default_vocab = int(self.file_default_vocab)
        f = {**FINE_DEFAULTS, **(self.fine or {})}
        s.gap_mm = float(f.get("derz") or 0.4)
        s.stone_scale = float(f.get("tas", 1.0))
        s.stone_h_mm = float(f.get("kalinlik", 3.0))
        s.relief_mm = float(f.get("kabartma", 0.0))
        s.h_jitter_mm = float(f.get("oynama", 0.0))
        s.panel_max_mm = float(f.get("panel", 240.0))
        s.merge_same_colour = False                   # every stone is its own piece
        s.backing = "yok"

        def idx(v):
            return int(v) if v is not None and 0 <= int(v) < len(s.palette) else None
        s.base_colour = idx(f.get("taban"))
        s.descreen = f.get("tram", "auto")
        s.plate_order = "renk" if f.get("tabla_sirasi") == "renk" else "panel"
        s.seed = int(f.get("tohum", 7))
        s.mm_per_px = working_mm_per_px(s.width_mm, preview=False)   # preview = export, stone for stone
        return s

    # -------------------------------------------------- project file
    def to_project(self) -> Project:
        p = Project()
        p.image_path = self.image_path
        p.settings = self.settings(preview=False)
        p.shapes = copy.deepcopy(self.shapes)
        p.rays_center = tuple(self.rays_center) if self.rays_center else None
        p.owned = []
        p.light_az, p.light_el = self.light_az, self.light_el
        p.preset = self.style
        p.name = self.name
        p.paint_mask = None if self.zone_mask is None else self.zone_mask.copy()
        return p

    def to_payload(self) -> dict:
        d = asdict(self)
        d["zone_mask"] = None if self.zone_mask is None else self.zone_mask.copy()
        return d

    @classmethod
    def from_payload(cls, d: dict) -> "Recipe":
        r = cls()
        for k, v in d.items():
            if hasattr(r, k):
                setattr(r, k, v)
        return r


def recipe_extra(r: Recipe) -> dict:
    """What a project file needs beyond Project's own fields (stored in its 'ad' block free keys)."""
    return {"stil": style_name(r.style), "ince_ayar": r.fine, "bolge_dosyadan": r.zones_from_file,
            "dosya_varsayilan": r.file_default_vocab}


def sibling_zones(image_path: str) -> Optional[dict]:
    """bolgeler.json next to the image (the example projects ship one)."""
    p = os.path.join(os.path.dirname(os.path.abspath(image_path)), "bolgeler.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) and "sekiller" in d else None
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ building
BUILD_SHARE = 0.72      # share of the progress bar for the mosaic itself; the stone meshes take the rest


def _build_accepts_progress() -> bool:
    import inspect
    from .pipeline import build_mosaic
    try:
        return "progress" in inspect.signature(build_mosaic).parameters
    except (TypeError, ValueError):
        return False


def build_mosaic_for(r: Recipe, preview: bool = True, image_cache: Optional[dict] = None,
                     auto_zone_cache: Optional[dict] = None, s: Optional[Settings] = None,
                     progress: Optional[Callable[[float, str], Any]] = None):
    """Recipe -> (Mosaic, auto zone mask or None). Thread/process safe. ``progress(f, msg)`` gets
    0..1 over the whole build: image load ~10 %, then the pipeline's own stages when
    build_mosaic takes a progress callback (feature-detected), else stage markers from here."""
    from .pipeline import build_mosaic, load_image, zones_from_shapes
    prog = progress or (lambda f, msg: None)
    s = s or r.settings(preview)
    key = (r.image_path, round(s.width_mm, 3), s.mm_per_px, str(s.descreen))
    if image_cache is not None and image_cache.get("key") == key:
        img = image_cache["img"]
    else:
        prog(0.0, "Görsel okunuyor")
        img = load_image(r.image_path, s)
        if image_cache is not None:
            image_cache.clear()
            image_cache.update(key=key, img=img)
    prog(0.1, "Bölgeler hazırlanıyor")
    h, w = img.shape[:2]
    mask = r.zone_mask
    found = None
    if (mask is None and not r.zones_from_file and not r.shapes
            and (r.fine or {}).get("otomatik_bolge", True)):
        ak = (r.image_path, h, w)
        if auto_zone_cache is not None and ak in auto_zone_cache:
            found = auto_zone_cache[ak]
        else:
            found = auto_zones(img, s)
            if auto_zone_cache is not None:
                auto_zone_cache.clear()
                auto_zone_cache[ak] = found
        mask = found["mask"]
        if r.rays_center is None and found.get("rays_center"):
            rc = found["rays_center"]
            s.rays_center = (rc[0] * w, rc[1] * h)
    if r.rays_center:
        s.rays_center = (r.rays_center[0] * w, r.rays_center[1] * h)
    zone = zones_from_shapes((h, w), r.shapes, default=int(s.default_vocab))
    zone = apply_paint(zone, mask, int(s.default_vocab))
    if _build_accepts_progress():
        m = build_mosaic(img, s, zone, source=r.image_path, progress=lambda f, msg="": prog(0.15 + 0.85 * float(f), str(msg)))
    else:
        prog(0.2, "Renkler ayrılıyor")
        m = build_mosaic(img, s, zone)
    prog(1.0, "Mozaik hazır")
    return m, found


def auto_zones(img: np.ndarray, s: Settings) -> dict:
    """Algorithm team's zone suggestion as a paint mask: only the zones it is sure about are
    painted; everything else follows the style's background stone type."""
    try:
        from .pipeline import suggest_zones
        zone, info = suggest_zones(img, s)
    except Exception:  # noqa: BLE001 - suggestions are optional
        traceback.print_exc()
        return {"mask": None, "rays_center": None, "reasons": []}
    default = int(info.get("varsayilan", int(Vocab.STONES)))
    mask = np.where(zone != default, zone, PAINT_NONE).astype(np.uint8)
    if not (mask != PAINT_NONE).any():
        mask = None
    rc = info.get("isin_merkezi")
    return {"mask": mask, "rays_center": list(rc) if rc else None,
            "reasons": list(info.get("nedenler", []))}


def other_mode(mode: str) -> str:
    return Mode.AMS if mode == Mode.PIECES else Mode.PIECES


def estimates(m) -> dict:
    """{mode: estimate} for both production modes (the two cards show their own numbers)."""
    from .export.api import estimate
    out = {}
    for mode in (Mode.PIECES, Mode.AMS):
        m2 = copy.copy(m)
        m2.settings = copy.copy(m.settings)
        m2.settings.mode = mode
        try:
            out[mode] = _plain(estimate(m2, exact=(mode == m.settings.mode or mode == Mode.AMS)))
        except Exception:  # noqa: BLE001 - another team's code; the preview must still show
            traceback.print_exc()
            out[mode] = {}
    return out


def png_bytes(a: np.ndarray, fmt: str = "PNG", quality: int = 88) -> bytes:
    from PIL import Image
    bio = io.BytesIO()
    im = Image.fromarray(np.asarray(a, np.uint8))
    if fmt == "JPEG":
        im.save(bio, "JPEG", quality=quality)
    else:
        im.save(bio, "PNG", optimize=False, compress_level=3)
    return bio.getvalue()


def data_url(b: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(b).decode("ascii")


def preview_png(m, az: float, el: float) -> bytes:
    from .render import render_cut, render_top
    return png_bytes(render_cut(m) if m.polys is not None else render_top(m, az_deg=az, el_deg=el))


def style_thumbnail(img: np.ndarray, r: "Recipe", s: Settings, found: Optional[dict], size: int = 300) -> bytes:
    """A close detail in the given texture: only that part of the picture is cut, so it is quick."""
    from PIL import Image
    from scipy import ndimage as ndi
    from .pipeline import build_mosaic, zones_from_shapes
    from .render import render_cut
    h, w = img.shape[:2]
    side = int(min(h, w) * 0.34)
    g = img.mean(2)
    dens = ndi.uniform_filter(ndi.gaussian_gradient_magnitude(g, 2.0), side)
    dens[: side // 2] = dens[-(side // 2):] = 0
    dens[:, : side // 2] = dens[:, -(side // 2):] = 0
    y, x = np.unravel_index(int(np.argmax(dens)), dens.shape)
    y0, x0 = int(y - side // 2), int(x - side // 2)
    s = copy.copy(s)
    zone = zones_from_shapes((h, w), r.shapes, default=int(s.default_vocab))
    zone = apply_paint(zone, r.zone_mask if r.zone_mask is not None else (found or {}).get("mask"), int(s.default_vocab))
    rc = r.rays_center or (found or {}).get("rays_center")
    s.rays_center = (rc[0] * w - x0, rc[1] * h - y0) if rc else None
    s.width_mm = s.width_mm * side / w
    m = build_mosaic(img[y0:y0 + side, x0:x0 + side], s, zone[y0:y0 + side, x0:x0 + side])
    return png_bytes(np.asarray(Image.fromarray(render_cut(m, 2 * size)).resize((size, size), Image.LANCZOS)), "JPEG", 88)


def _plain(obj):
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


plain = _plain


# ------------------------------------------------------------------ the worker thread
class Engine:
    """One worker thread. Callbacks are invoked from the worker thread; the Qt host turns
    them into queued signals.

    on_built(gen, info dict, mosaic)       after every build (numbers + preview png bytes)
    on_scene(gen, glb bytes, meta dict)    real part meshes (fast quality), right after
    on_thumb(gen, style, jpeg bytes)       style cards
    on_palette(gen, names, roles)          suggested spools
    on_error(gen, kind, message_tr)
    on_busy(bool, message_tr)
    """

    def __init__(self, on_built, on_scene, on_thumb, on_palette, on_error, on_busy, on_progress=None,
                 on_plates=None):
        self.cb = dict(built=on_built, scene=on_scene, thumb=on_thumb, palette=on_palette,
                       error=on_error, busy=on_busy, progress=on_progress or (lambda gen, f, msg: None),
                       plates=on_plates or (lambda gen, glb, meta: None))
        self._cv = threading.Condition()
        self._build: Optional[tuple] = None      # (gen, recipe)
        self._jobs: list = []                    # low priority: ("thumbs", gen, recipe) ...
        self._stop = False
        self._image_cache: dict = {}
        self._zones: dict = {}
        self._thumb_cache: dict = {}
        self.mosaic = None
        self.mosaic_key = ""
        self.mosaic_gen = -1
        self._t = threading.Thread(target=self._run, name="tessera-engine", daemon=True)
        self._t.start()

    # -------------------------------------------------- api
    def build(self, gen: int, r: Recipe) -> None:
        with self._cv:
            self._build = (gen, r.copy())
            self._jobs = [j for j in self._jobs if j[0] != "thumbs"]
            self._cv.notify()

    def thumbs(self, gen: int, r: Recipe) -> None:
        self._push(("thumbs", gen, r.copy()))

    def suggest(self, gen: int, r: Recipe, owned: list) -> None:
        self._push(("suggest", gen, r.copy(), list(owned)))

    def _push(self, job) -> None:
        with self._cv:
            self._jobs = [j for j in self._jobs if j[0] != job[0]] + [job]
            self._cv.notify()

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._t.join(3)

    def superseded(self) -> bool:
        with self._cv:
            return self._build is not None

    # -------------------------------------------------- thread
    def _run(self) -> None:
        while True:
            with self._cv:
                while self._build is None and not self._jobs and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                if self._build is not None:
                    job, self._build = ("build",) + self._build, None
                else:
                    job = self._jobs.pop(0)
            try:
                getattr(self, "_do_" + job[0])(*job[1:])
            except Exception as e:  # noqa: BLE001 - shown to the user
                traceback.print_exc()
                self.cb["error"](job[1], job[0], friendly_error(e))
            finally:
                with self._cv:
                    idle = self._build is None
                if idle and job[0] == "build":
                    self.cb["busy"](False, "")

    def _do_build(self, gen: int, r: Recipe) -> None:
        """Mosaic + numbers. Progress covers 0..BUILD_SHARE of the whole conversion; the real
        per-stone scene (child process) reports the rest."""
        if not r.image_path:
            return

        def P(f, msg):
            self.cb["progress"](gen, BUILD_SHARE * f, msg)
        self.cb["busy"](True, "Mozaiğe dönüştürülüyor")
        P(0.0, "Görsel okunuyor")
        t0 = time.perf_counter()
        key = r.build_key()
        if key != self.mosaic_key or self.mosaic is None:
            m, found = build_mosaic_for(r, True, self._image_cache, self._zones,
                                        progress=lambda f, msg: P(0.02 + 0.86 * max(0.0, min(1.0, f)), msg))
            if self.superseded():
                return
            self.mosaic, self.mosaic_key = m, key
        else:
            found = None
        self.mosaic_gen = gen
        m = self.mosaic
        P(0.9, "Parça ve süre hesaplanıyor")
        est = estimates(m)
        if self.superseded():
            return
        w_mm, h_mm = m.size_mm
        info = {
            "size_mm": [round(w_mm, 1), round(h_mm, 1)],
            "estimates": est,
            "mode": m.settings.mode,
            "palette": [{"name": f.name, "hex": f.hex} for f in m.settings.palette],
            "tiles": int(m.n_tiles),
            "seconds": round(time.perf_counter() - t0, 2),
            "zones": None if found is None else {"rays_center": found.get("rays_center"),
                                                 "reasons": found.get("reasons", []),
                                                 "mask": found.get("mask")},
        }
        self.cb["built"](gen, info, m)
        # every stone as its own node, with real progress (R6: 2-5 s at preview resolution)
        from .export.api import scene_glb

        def S(f, msg=""):
            self.cb["progress"](gen, BUILD_SHARE + (1 - BUILD_SHARE) * 0.98 * max(0.0, min(1.0, float(f))),
                                msg or "Taşlar tek tek kuruluyor")
        S(0.0, "Taşlar tek tek kuruluyor")
        glb, meta = scene_glb(m, None, "preview", S)
        if self.superseded():
            return
        self.cb["scene"](gen, glb, _plain(meta))
        self._push(("plates", gen, m))

    def _do_plates(self, gen: int, m) -> None:
        from .export.api import plates_glb
        if self.superseded() or m is not self.mosaic:
            return
        pglb, pmeta = plates_glb(m, None)
        if not self.superseded() and m is self.mosaic:
            self.cb["plates"](gen, pglb, _plain(pmeta))

    def _do_thumbs(self, gen: int, r: Recipe) -> None:
        if not r.image_path or not r.palette:
            return
        for name, _code, _d in STYLE_CARDS:
            if self.superseded():
                return
            rr = r.copy()
            rr.style = name
            s = rr.settings(True)
            k = (rr.build_key(), s.mm_per_px)
            jpg = self._thumb_cache.get(k)
            if jpg is None:
                img = self._image_cache.get("img")
                if img is None:
                    from .pipeline import load_image
                    img = load_image(rr.image_path, s)
                found = next(iter(self._zones.values()), None)
                jpg = style_thumbnail(img, rr, s, found)
                if self.superseded():
                    return
                if len(self._thumb_cache) > 24:
                    self._thumb_cache.clear()
                self._thumb_cache[k] = jpg
            self.cb["thumb"](gen, name, jpg)

    def _do_suggest(self, gen: int, r: Recipe, owned: list) -> None:
        from PIL import Image
        from .pipeline import open_rgb, suggest_palette
        im = open_rgb(r.image_path)
        im.thumbnail((300, 300))
        lib = [dict(e, owned=(e["ad"] in owned)) for e in library()]
        k = int((r.fine or {}).get("renk_sayisi", 4))
        res = suggest_palette(np.asarray(im), lib, k, owned_only=True)
        names = [x["filament"].name for x in res]
        roles = {x["filament"].name: x.get("rol", "") for x in res}
        self.cb["palette"](gen, names, roles)


# ------------------------------------------------------------------ child processes
def _below_normal_priority() -> None:
    if os.name == "nt":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
        except Exception:  # noqa: BLE001
            pass


def run_export(payload: dict, progress: Callable[[float, str], Any]) -> dict:
    """Full-resolution build + export_project (runs in the child process)."""
    from .export.api import export_project
    r = Recipe.from_payload(payload["recipe"])
    progress(0.02, "Mozaik tam çözünürlükte hesaplanıyor")
    m, _ = build_mosaic_for(r, preview=False)
    progress(0.25, "Parçalar ve tablalar hazırlanıyor")
    res = export_project(m, payload["out"], progress=lambda f, msg="": progress(0.25 + 0.75 * float(f), str(msg)))
    return _plain(res if isinstance(res, dict) else {})


def run_full_scene(payload: dict, progress: Callable[[float, str], Any], part=None) -> dict:
    """Every stone as its own mesh (+ the build plates) for the 3D stage. Child process, low
    priority. The scene is reported through ``part`` as soon as it exists; plates follow."""
    _below_normal_priority()
    import pickle
    from .export.api import plates_glb, scene_glb
    progress(0.02, "Taşlar tek tek kuruluyor")
    if payload.get("mosaic"):
        with open(payload["mosaic"], "rb") as fh:
            m = pickle.load(fh)
    else:
        m, _ = build_mosaic_for(Recipe.from_payload(payload["recipe"]), preview=True)
    progress(0.08, "Taşlar tek tek kuruluyor")
    import inspect
    try:
        takes = "progress" in inspect.signature(scene_glb).parameters
    except (TypeError, ValueError):
        takes = False
    if takes:     # requested from the 3D team (docs/istekler.md); used as soon as it exists
        glb, meta = scene_glb(m, None, "full",
                              progress=lambda f, msg="": progress(0.08 + 0.8 * float(f), msg or "Taşlar tek tek kuruluyor"))
    else:
        glb, meta = scene_glb(m, None, "full")
    sp = os.path.join(payload["out"], "scene_full.glb")
    with open(sp, "wb") as fh:
        fh.write(glb)
    res = {"scene": sp, "scene_meta": _plain(meta)}
    if part is not None:
        part(dict(res))
    progress(0.9, "Tablalar diziliyor")
    pglb, pmeta = plates_glb(m, None)
    pp = os.path.join(payload["out"], "plates.glb")
    with open(pp, "wb") as fh:
        fh.write(pglb)
    res.update(plates=pp, plates_meta=_plain(pmeta))
    return res


JOBS = {"export": run_export, "full": run_full_scene}


def child_main(kind: str, payload: dict, q) -> None:   # runs in the child process
    try:
        kw = {"part": lambda d: q.put(("part", d))} if kind == "full" else {}
        res = JOBS[kind](payload, lambda f, msg="": q.put(("p", float(f), str(msg))), **kw)
        q.put(("done", res))
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        q.put(("error", friendly_error(e)))
