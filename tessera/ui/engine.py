"""Background work. The UI thread never builds or renders a mosaic.

- PreviewEngine: one long-lived thread. `submit()` replaces any waiting request (only the newest
  matters), so fast slider drags never queue up work. It caches the loaded image and the last
  Mosaic: a light-direction change only re-renders (~0.2 s), it does not rebuild.
- TaskThread: one-shot jobs (export, suggestions) with progress.
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Signal

from ..model import Settings
from ..project import apply_paint, settings_to_json

PREVIEW_MM_PER_PX = 0.8
EXPORT_MM_PER_PX = 0.4


_TR_CHARS = set("çğıöşüÇĞİÖŞÜâÂ")


def _is_turkish(msg: str) -> bool:
    return any(c in _TR_CHARS for c in msg)


def friendly_error(e: BaseException) -> str:
    """Plain Turkish message for the user. Never shows raw English exception text."""
    import os as _os
    fn = getattr(e, "filename", None)
    name = _os.path.basename(str(fn)) if fn else ""
    try:
        from PIL import UnidentifiedImageError
    except ImportError:  # pragma: no cover
        UnidentifiedImageError = ()  # type: ignore[assignment]
    if isinstance(e, FileNotFoundError):
        return f"Dosya bulunamadı: {_os.path.normpath(str(fn))}" if fn else "Dosya bulunamadı."
    if isinstance(e, PermissionError):
        return (f"Bu dosyaya ya da klasöre erişim izni yok: {_os.path.normpath(str(fn))}" if fn
                else "Dosyaya erişim izni yok. Başka bir program açık tutuyor olabilir.")
    if isinstance(e, MemoryError):
        return "Bellek yetmedi. Eseri küçültmeyi ya da daha az renk kullanmayı dene."
    if UnidentifiedImageError and isinstance(e, UnidentifiedImageError):
        return "Bu dosya bir resim değil ya da bozuk. JPG, PNG ya da WEBP bir görsel seç."
    msg = str(e).strip()
    low = msg.lower()
    if isinstance(e, (OSError, SyntaxError)) and any(k in low for k in ("truncated", "image", "decod", "broken",
                                                                         "png", "jpeg", "unrecognized")):
        return "Bu görsel dosyası bozuk ya da yarım kalmış. Dosyayı yeniden kaydedip tekrar dene."
    if isinstance(e, OSError):
        if getattr(e, "errno", None) == 28:
            return "Diskte yer kalmadı. Başka bir klasör seç ya da yer aç."
        return "Dosya okunurken ya da yazılırken bir sorun oluştu." + (f" ({name})" if name else "")
    if msg and _is_turkish(msg):
        return msg                     # our own pipeline messages are already Turkish
    return "Beklenmeyen bir sorun oluştu. Ayarları biraz değiştirip yeniden dene; sürerse projeyi kaydedip programı yeniden başlat."


# ---------------------------------------------------------------------- working resolution (R2-3)
def _resolution(width_mm: float, preview: bool) -> float:
    """mm per pixel for builds. Uses the Algorithm team's working_mm_per_px when it exists."""
    try:
        from .. import pipeline as _pl
        fn = getattr(_pl, "working_mm_per_px", None)
        if fn is None:
            from ..pipeline import build as _b
            fn = getattr(_b, "working_mm_per_px", None)
        if callable(fn):
            for args, kw in (((width_mm,), {"preview": preview}), ((width_mm, preview), {})):
                try:
                    v = float(fn(*args, **kw))
                    if v > 0:
                        return v
                except TypeError:
                    continue
    except Exception:  # noqa: BLE001
        pass
    return max(PREVIEW_MM_PER_PX, width_mm / 1300) if preview else max(EXPORT_MM_PER_PX, width_mm / 2600)


def preview_mm_per_px(width_mm: float) -> float:
    return _resolution(width_mm, True)


def export_mm_per_px(width_mm: float) -> float:
    return _resolution(width_mm, False)


# ---------------------------------------------------------------------- budget (PLAN §9, R2-2)
BUDGET_KG_M2 = 3.0            # R3-5: green <= 3, yellow <= 4, red > 4
BUDGET_KG_M2_MAX = 4.0
BUDGET_PARTS_M2 = 600.0       # easy to lay out by hand
BUDGET_PARTS_M2_MAX = 1200.0  # beyond this: a lot of tweezers work
HASSAS_WARN_SHARE = 0.05      # R3-1: warn before export when delicate parts exceed 5 %
SMALL_WIDTH_MM = 250.0        # Ö2-10: below this, parts get tiny
RELIEF_TARGET = (0.70, 0.25, 0.05)      # share of area <=3 mm, <=8 mm, higher


def compute_budget(m, est: dict) -> dict:
    """kg/m², relief distribution and parts/m². Uses the Algorithm team's relief_budget(m) when
    present; anything it does not provide is computed here."""
    out: dict = {}
    try:
        from .. import pipeline as _pl
        fn = getattr(_pl, "relief_budget", None)
        if callable(fn):
            rb = fn(m)
            if isinstance(rb, dict):
                out.update(rb)
    except Exception:  # noqa: BLE001
        pass
    w_mm, h_mm = m.size_mm
    area_m2 = max(w_mm * h_mm / 1e6, 1e-6)
    out["alan_m2"] = area_m2

    def pick(*keys):
        for k in keys:
            if k in out and out[k] is not None:
                return out[k]
        return None

    kg = est.get("kg_m2")                     # 3D team's figure includes the backing: prefer it
    if kg is None:
        kg = pick("kg_m2", "kg_per_m2", "kg_m²")
    if kg is None and est.get("gram") is not None:
        kg = float(est["gram"]) / 1000 / area_m2
    parts = pick("parca_m2", "parts_per_m2")
    if parts is None and est.get("parca") is not None:
        parts = float(est["parca"]) / area_m2
    dist = pick("bantlar", "dagilim", "dağılım", "distribution")
    if isinstance(dist, dict):
        dist = list(dist.values())[:3]          # low, medium, high in the producer's order
    if not (isinstance(dist, (list, tuple)) and len(dist) >= 3):
        hgt = m.height[~m.gap]
        n = max(1, hgt.size)
        dist = [float((hgt <= 3.0).sum()) / n, float(((hgt > 3.0) & (hgt <= 8.0)).sum()) / n,
                float((hgt > 8.0).sum()) / n]
    out.update(kg_m2=kg, parca_m2=parts, dagilim=[float(d) for d in dist[:3]])
    return out


def budget_level(kind: str, value) -> str:
    """'ok' | 'warn' | 'bad' against the targets."""
    if value is None:
        return "ok"
    if kind == "kg":
        return "ok" if value <= BUDGET_KG_M2 else ("warn" if value <= BUDGET_KG_M2_MAX else "bad")
    if kind == "parts":
        return "ok" if value <= BUDGET_PARTS_M2 else ("warn" if value <= BUDGET_PARTS_M2_MAX else "bad")
    if kind == "relief":
        low, mid, high = value
        if high <= 0.08 and low >= 0.6:
            return "ok"
        return "warn" if high <= 0.18 else "bad"
    return "ok"


@dataclass
class PreviewRequest:
    image_path: str
    settings: Settings
    shapes: list
    paint_mask: Optional[np.ndarray]
    rays_center: Optional[tuple]          # fractional
    az: float = -130.0
    el: float = 40.0
    oblique_elev: float = 42.0
    bg: tuple = (236, 234, 230)           # oblique view background (matches the stage)
    gen: int = 0


@dataclass
class PreviewResult:
    gen: int
    mosaic: Any
    top: np.ndarray
    oblique: np.ndarray
    flat: np.ndarray
    estimate: dict
    seconds: float
    relit_only: bool
    warnings: list = field(default_factory=list)
    budget: dict = field(default_factory=dict)
    est_by_mode: dict = field(default_factory=dict)   # {"parca": estimate, "ams": estimate}


def build_for(req_settings: Settings, image_path: str, shapes, paint_mask, rays_center_frac,
              image_cache: Optional[dict] = None):
    """Load image at settings.mm_per_px, rasterise zones and build the Mosaic. Thread-safe."""
    from ..pipeline import build_mosaic, load_image, zones_from_shapes
    s = req_settings
    key = (image_path, round(s.width_mm, 3), s.mm_per_px)
    if image_cache is not None and image_cache.get("key") == key:
        img = image_cache["img"]
    else:
        img = load_image(image_path, s)
        if image_cache is not None:
            image_cache.update(key=key, img=img)
    h, w = img.shape[:2]
    if rays_center_frac:
        s.rays_center = (rays_center_frac[0] * w, rays_center_frac[1] * h)
    else:
        s.rays_center = None
    zone = zones_from_shapes((h, w), shapes, default=int(s.default_vocab))
    zone = apply_paint(zone, paint_mask, int(s.default_vocab))
    return build_mosaic(img, s, zone)


def _build_key(req: PreviewRequest) -> str:
    h = hashlib.sha1()
    h.update(req.image_path.encode())
    h.update(json.dumps(settings_to_json(req.settings), sort_keys=True, default=str).encode())
    h.update(json.dumps(req.shapes, sort_keys=True).encode())
    h.update(json.dumps(req.rays_center).encode())
    if req.paint_mask is not None:
        h.update(req.paint_mask.tobytes())
    return h.hexdigest()


def flat_colours(m) -> np.ndarray:
    return np.clip(m.albedo((34, 32, 30)), 0, 255).astype(np.uint8)


class PreviewEngine(QThread):
    ready = Signal(object)          # PreviewResult
    failed = Signal(int, str)       # gen, message
    busy_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cv = threading.Condition()
        self._pending: Optional[PreviewRequest] = None
        self._stop = False
        self._image_cache: dict = {}
        self._mosaic = None
        self._mosaic_key = ""

    # ---------------------------------------------- api (UI thread)
    def submit(self, req: PreviewRequest) -> None:
        req = copy.copy(req)
        req.settings = copy.deepcopy(req.settings)
        req.shapes = copy.deepcopy(req.shapes)
        req.paint_mask = None if req.paint_mask is None else req.paint_mask.copy()
        with self._cv:
            self._pending = req
            self._cv.notify()
        if not self.isRunning():
            self.start()

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        self.wait(5000)

    def invalidate(self) -> None:
        self._mosaic_key = ""

    # ---------------------------------------------- thread
    def run(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                req, self._pending = self._pending, None
            self.busy_changed.emit(True)
            try:
                res = self._process(req)
                if res is not None:
                    self.ready.emit(res)
            except Exception as e:  # noqa: BLE001 - shown to the user
                traceback.print_exc()
                self.failed.emit(req.gen, friendly_error(e))
            finally:
                with self._cv:
                    idle = self._pending is None
                if idle:
                    self.busy_changed.emit(False)

    def _superseded(self) -> bool:
        with self._cv:
            return self._pending is not None

    def _process(self, req: PreviewRequest) -> Optional[PreviewResult]:
        from ..render import render_oblique, render_top
        t0 = time.perf_counter()
        key = _build_key(req)
        relit_only = key == self._mosaic_key and self._mosaic is not None
        warnings: list[str] = []
        if not relit_only:
            m = build_for(req.settings, req.image_path, req.shapes, req.paint_mask, req.rays_center,
                          self._image_cache)
            if self._superseded():
                return None
            self._mosaic, self._mosaic_key = m, key
        m = self._mosaic
        top = render_top(m, az_deg=req.az, el_deg=req.el)
        ob = render_oblique(top, m, elev_deg=req.oblique_elev, bg=req.bg)
        try:
            from ..export.api import estimate
            est = estimate(m, exact=False)       # fast "≈"; count_parts follows in the background
        except Exception as e:  # noqa: BLE001 - estimate is another team's code; keep the preview
            traceback.print_exc()
            est = {}
            warnings.append("Tahmin hesaplanamadı: " + friendly_error(e))
        try:
            budget = compute_budget(m, est)
        except Exception:  # noqa: BLE001
            budget = {}
        by_mode = {m.settings.mode: est}
        try:   # the other production mode too, so both cards show honest numbers (Ö-1, Ö-15)
            from ..model import Mode
            other = Mode.AMS if m.settings.mode == Mode.PIECES else Mode.PIECES
            m2 = copy.copy(m)
            m2.settings = copy.copy(m.settings)
            m2.settings.mode = other
            from ..export.api import estimate
            by_mode[other] = estimate(m2, exact=False)
        except Exception:  # noqa: BLE001
            pass
        return PreviewResult(req.gen, m, top, ob, flat_colours(m), est,
                             time.perf_counter() - t0, relit_only, warnings, budget, by_mode)


class TaskThread(QThread):
    """Run fn(progress) in a thread. progress(fraction, message_tr)."""
    progress = Signal(float, str)
    done = Signal(object)
    error = Signal(str)

    def __init__(self, fn: Callable[[Callable[[float, str], None]], Any], parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            res = self._fn(lambda f, msg="": self.progress.emit(float(f), str(msg)))
            self.done.emit(res)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.error.emit(friendly_error(e))


# ---------------------------------------------------------------------- export in a child process
def _plain(obj):
    """Make an export result safe to send between processes (plain Python types only)."""
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


def export_payload(settings: Settings, image_path: str, shapes, paint_mask, rays_center, out_dir: str) -> dict:
    return {"settings": settings_to_json(settings), "image": image_path, "shapes": copy.deepcopy(shapes),
            "mask": None if paint_mask is None else paint_mask.copy(), "rays": rays_center, "out": out_dir}


def run_export(payload: dict, progress: Callable[[float, str], None]) -> dict:
    """Full-resolution build + export_project. Used by both the process and the thread runner."""
    from ..export.api import export_project
    from ..project import settings_from_json
    s = settings_from_json(payload["settings"])
    s.mm_per_px = export_mm_per_px(s.width_mm)
    progress(0.02, "Mozaik tam çözünürlükte hesaplanıyor…")
    m = build_for(s, payload["image"], payload["shapes"], payload["mask"], payload["rays"])
    progress(0.30, "Parçalar ve tablalar hazırlanıyor…")
    res = export_project(m, payload["out"], progress=lambda f, msg="": progress(0.30 + 0.70 * float(f), str(msg)))
    return _plain(res if isinstance(res, dict) else {})


def _export_worker(payload: dict, q) -> None:  # runs in the child process
    try:
        res = run_export(payload, lambda f, msg="": q.put(("p", float(f), str(msg))))
        q.put(("done", res))
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        q.put(("error", friendly_error(e)))


class ExportJob(QObject):
    """Runs the export in a separate process so heavy geometry code (which may hold the GIL for
    seconds) can never freeze the window. Same signals as TaskThread; can be cancelled."""
    progress = Signal(float, str)
    done = Signal(object)
    error = Signal(str)
    cancelled = Signal()

    def __init__(self, payload: dict, parent=None):
        super().__init__(parent)
        self.payload = payload
        self._proc = None
        self._q = None
        self._finished = False
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._poll)

    def start(self) -> None:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self._q = ctx.Queue()
        self._proc = ctx.Process(target=_export_worker, args=(self.payload, self._q), daemon=True,
                                 name="tessera-export")
        self._proc.start()
        self._timer.start()

    def isRunning(self) -> bool:
        return self._proc is not None and not self._finished

    def isFinished(self) -> bool:
        return self._finished

    def wait(self, ms: int = 0) -> None:
        if self._proc is not None:
            self._proc.join(ms / 1000 if ms else None)

    def cancel(self) -> None:
        if self._proc is not None and not self._finished:
            self._finish()
            self._proc.terminate()
            self.cancelled.emit()

    def _finish(self) -> None:
        self._finished = True
        self._timer.stop()

    def _poll(self) -> None:
        import queue
        latest = None
        while True:
            try:
                kind, *rest = self._q.get_nowait()
            except queue.Empty:
                break
            if kind == "p":
                latest = rest
            elif kind == "done":
                self._finish()
                self.done.emit(rest[0])
                return
            elif kind == "error":
                self._finish()
                self.error.emit(rest[0])
                return
        if latest is not None:
            self.progress.emit(latest[0], latest[1])
        if not self._proc.is_alive() and not self._finished:
            self._finish()
            self.error.emit("Dışa aktarım beklenmedik biçimde durdu (çıkış kodu "
                            f"{self._proc.exitcode}). Tekrar dene; sürerse genişliği küçült.")


# ---------------------------------------------------------------------- exact part count (R3-4)
def exact_parts(m) -> dict:
    """The export's own part count (3D team's count_parts; ~1 s at preview resolution):
    {"parca", "hassas", "cok_kucuk", "panel"}."""
    from ..export.api import count_parts
    return dict(count_parts(m))
