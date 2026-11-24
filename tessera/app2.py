"""Tessera v2 desktop app: one window hosting the HTML/three.js interface in tessera/web/.

Run:  .venv\\Scripts\\pythonw.exe -m tessera.app2  [image or project file]

Python <-> JS through QWebChannel (object "py", class Bridge). Static files and generated
blobs (GLB scenes, previews) are served by the custom scheme ``tessera://``:
    tessera://app/<file>          files from tessera/web/
    tessera://app/data/<name>     in-memory blobs (BlobStore; same origin, so fetch() works)
    tessera://app/qt/qwebchannel.js
Heavy work never runs on the UI thread: builds in Engine's worker thread, export and exact
part meshes in a child process (ChildJob).
"""
from __future__ import annotations

import gc
import json
import mimetypes
import os
import sys
import tempfile
import threading
import traceback
from typing import Optional

import numpy as np
from PySide6.QtCore import (QBuffer, QByteArray, QFile, QIODevice, QObject, QRect, QSettings, QStandardPaths, Qt,
                            QTimer, QUrl, Signal, Slot)
from PySide6.QtGui import QColor, QDesktopServices, QIcon
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import (QWebEnginePage, QWebEngineProfile, QWebEngineSettings, QWebEngineUrlRequestJob,
                                     QWebEngineUrlScheme, QWebEngineUrlSchemeHandler)
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow

from . import app2_engine as E
from .model import Mode
from .project import PROJECT_SUFFIX, Project

SCHEME = b"tessera"
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_IMAGE = os.path.join(ROOT_DIR, "ornekler", "poster", "kaynak.webp")
IMAGE_FILTER = "Görseller (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)"


def register_scheme() -> None:
    """Must run before QApplication is created."""
    if QWebEngineUrlScheme.schemeByName(SCHEME).name() == SCHEME:
        return
    sc = QWebEngineUrlScheme(SCHEME)
    sc.setSyntax(QWebEngineUrlScheme.Syntax.Host)
    F = QWebEngineUrlScheme.Flag
    sc.setFlags(F.SecureScheme | F.LocalAccessAllowed | F.CorsEnabled | F.FetchApiAllowed
                | F.ContentSecurityPolicyIgnored)
    QWebEngineUrlScheme.registerScheme(sc)


# ------------------------------------------------------------------ blobs + scheme
class BlobStore:
    def __init__(self):
        self._d: dict[str, tuple[str, bytes]] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()

    def put(self, name: str, data: bytes, mime: str, keep: int = 24) -> str:
        with self._lock:
            if name in self._d:
                self._order.remove(name)
            self._d[name] = (mime, data)
            self._order.append(name)
            while len(self._order) > keep:
                self._d.pop(self._order.pop(0), None)
        return f"tessera://app/data/{name}"

    def get(self, name: str):
        with self._lock:
            return self._d.get(name)


class SchemeHandler(QWebEngineUrlSchemeHandler):
    def __init__(self, blobs: BlobStore, parent=None):
        super().__init__(parent)
        self.blobs = blobs
        self._qwc: Optional[bytes] = None

    def requestStarted(self, job: QWebEngineUrlRequestJob) -> None:  # noqa: N802 (Qt API)
        url = job.requestUrl()
        host, path = url.host(), url.path().lstrip("/")
        data = mime = None
        if host == "app" and path.startswith("data/"):
            b = self.blobs.get(path[5:])
            if b is not None:
                mime, data = b
        elif host == "app" and path == "qt/qwebchannel.js":
            if self._qwc is None:
                f = QFile(":/qtwebchannel/qwebchannel.js")
                f.open(QIODevice.OpenModeFlag.ReadOnly)
                self._qwc = bytes(f.readAll())
                f.close()
            data, mime = self._qwc, "text/javascript"
        elif host == "app":
            full = os.path.normpath(os.path.join(WEB_DIR, path or "index.html"))
            if full.startswith(os.path.normpath(WEB_DIR)) and os.path.isfile(full):
                with open(full, "rb") as fh:
                    data = fh.read()
                mime = mimetypes.guess_type(full)[0] or "application/octet-stream"
                if full.endswith(".js"):
                    mime = "text/javascript"
        if data is None:
            job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
            return
        buf = QBuffer(parent=job)
        buf.setData(QByteArray(data))
        buf.open(QIODevice.OpenModeFlag.ReadOnly)
        job.reply(mime.encode(), buf)


# ------------------------------------------------------------------ child process job
class ChildJob(QObject):
    """Run app2_engine.JOBS[kind] in a spawned process; poll its queue from the UI thread."""
    progress = Signal(float, str)
    partial = Signal(object)       # an early result (the stone scene, before the plates)
    done = Signal(object)
    error = Signal(str)

    def __init__(self, kind: str, payload: dict, parent=None):
        super().__init__(parent)
        self.kind, self.payload = kind, payload
        self._proc = self._q = None
        self.finished = False
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll)

    def start(self) -> None:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self._q = ctx.Queue()
        self._proc = ctx.Process(target=E.child_main, args=(self.kind, self.payload, self._q), daemon=True,
                                 name=f"tessera-{self.kind}")
        self._proc.start()
        self._timer.start()

    def cancel(self) -> None:
        if self._proc is not None and not self.finished:
            self.finished = True
            self._timer.stop()
            self._proc.terminate()

    def _poll(self) -> None:
        import queue
        latest = None
        while True:
            try:
                kind, *rest = self._q.get_nowait()
            except queue.Empty:
                break
            except (EOFError, OSError):
                break
            if kind == "p":
                latest = rest
            elif kind == "part":
                if latest is not None:
                    self.progress.emit(latest[0], latest[1])
                    latest = None
                self.partial.emit(rest[0])
            elif kind in ("done", "error"):
                self.finished = True
                self._timer.stop()
                (self.done if kind == "done" else self.error).emit(rest[0])
                return
        if latest is not None:
            self.progress.emit(latest[0], latest[1])
        if self._proc is not None and not self._proc.is_alive() and not self.finished:
            self.finished = True
            self._timer.stop()
            self.error.emit("İşlem beklenmedik biçimde durdu. Tekrar dene; sürerse genişliği küçült.")


# ------------------------------------------------------------------ bridge
class Bridge(QObject):
    """Everything the page can ask for. Slots return JSON strings; long work answers later
    through signals. All signals carry JSON text (one format, easy to test)."""
    state = Signal(str)            # full UI state after any change
    busy = Signal(str)             # {"on": bool, "msg": str}
    progress = Signal(str)         # {"f": 0..1, "msg": str}: converting an image into stones
    built = Signal(str)            # numbers for the summary line and the two production cards
    scene = Signal(str)            # {"url", "meta", "quality"}
    plates = Signal(str)           # {"url", "meta"}
    thumb = Signal(str)            # {"style", "url"}
    exportProgress = Signal(float, str)
    exportDone = Signal(str)
    failed = Signal(str)           # {"kind", "msg"}
    toast = Signal(str)

    def __init__(self, window: Optional["MainWindow"] = None, store: Optional[QSettings] = None):
        super().__init__()
        self.window = window
        self.store = store or QSettings("Tessera", "Tessera")
        self.blobs = BlobStore()
        self.recipe = E.Recipe()
        self.recipe.palette = []
        self.gen = 0
        self.mosaic = None
        self.mosaic_gen = -1
        self.last_info: dict = {}
        self.scene_info: Optional[dict] = None
        self.plates_info: Optional[dict] = None
        self.thumbs: dict = {}
        self.project_path = ""
        self.export_job: Optional[ChildJob] = None
        self.full_job: Optional[ChildJob] = None
        self.full_gen = -1
        self.last_export: Optional[dict] = None
        self.dirty = False
        self._tmp = tempfile.mkdtemp(prefix="tessera2_")
        self.engine = E.Engine(self._on_built, self._on_scene, self._on_thumb, self._on_palette,
                               self._on_error, self._on_busy, self._on_progress, self._on_plates)
        self._scene_cache: dict = {}            # recipe key -> (scene_info, plates_info)
        self._conv_gen = -1                     # generation whose conversion is on screen
        self._await_palette: Optional[int] = None
        self._thumb_gens: dict = {}
        self._sig_internal.connect(self._dispatch, Qt.ConnectionType.QueuedConnection)

    # -------------------------------------------------- engine callbacks (worker thread)
    _sig_internal = Signal(str, object)

    def _on_built(self, gen, info, m):
        self._sig_internal.emit("built", (gen, info, m))

    def _on_scene(self, gen, glb, meta):
        self._sig_internal.emit("scene", (gen, glb, meta))

    def _on_thumb(self, gen, style, jpg):
        self._sig_internal.emit("thumb", (gen, style, jpg))

    def _on_palette(self, gen, names, roles):
        self._sig_internal.emit("palette", (gen, names, roles))

    def _on_error(self, gen, kind, msg):
        self._sig_internal.emit("error", (gen, kind, msg))

    def _on_busy(self, on, msg):
        self._sig_internal.emit("busy", (on, msg))

    def _on_plates(self, gen, glb, meta):
        self._sig_internal.emit("plates", (gen, glb, meta))

    def _on_progress(self, gen, f, msg):
        self._sig_internal.emit("progress", (gen, f, msg))

    def _ui_progress(self, gen, f, msg):
        if gen == self.gen:
            self._emit_progress(f, msg)

    def _emit_progress(self, f, msg):
        self.progress.emit(json.dumps({"f": round(max(0.0, min(1.0, float(f))), 4), "msg": msg},
                                      ensure_ascii=False))

    @Slot(str, object)
    def _dispatch(self, what: str, args) -> None:
        getattr(self, "_ui_" + what)(*args)

    # -------------------------------------------------- engine results (UI thread)
    def _ui_built(self, gen, info, m):
        if gen != self.gen:
            return
        self.mosaic, self.mosaic_gen = m, gen
        z = info.pop("zones", None)
        if z and z.get("mask") is not None and self.recipe.zone_mask is None:
            self.recipe.zone_mask = z["mask"]
            if z.get("rays_center") and not self.recipe.rays_center:
                self.recipe.rays_center = z["rays_center"]
            # the mask is now part of the recipe: the next build key includes it (same result)
            self.engine.mosaic_key = self.recipe.build_key()
        try:
            png = E.preview_png(m, E.LIGHT_AZ, E.LIGHT_EL)
            info["preview"] = self.blobs.put(f"preview_{gen}.png", png, "image/png")
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        self.last_info = info
        self.built.emit(json.dumps(info, ensure_ascii=False))
        if not self.thumbs.get("_key") == self._thumb_key():
            self.thumbs = {"_key": self._thumb_key()}
        if len([k for k in self.thumbs if not k.startswith("_")]) < len(E.STYLE_CARDS):
            self._thumb_gens[gen] = self.thumbs["_key"]
            self.engine.thumbs(gen, self.recipe)     # (re)queued: a newer build may have cut it short

    def _ui_scene(self, gen, glb, meta):
        """Every stone as its own node (R6). The progress overlay closes when this arrives."""
        if gen != self.gen:
            return
        url = self.blobs.put(f"scene_{gen}.glb", glb, "model/gltf-binary")
        self.scene_info = {"url": url, "meta": meta, "quality": meta.get("quality", "preview"), "gen": gen}
        self._emit_progress(1.0, "Hazır")
        self.scene.emit(json.dumps(self.scene_info, ensure_ascii=False))

    def _ui_plates(self, gen, glb, meta):
        if gen != self.gen:
            return
        url = self.blobs.put(f"plates_{gen}.glb", glb, "model/gltf-binary")
        self.plates_info = {"url": url, "meta": meta, "gen": gen}
        self.full_gen = gen
        self.plates.emit(json.dumps(self.plates_info, ensure_ascii=False))

    def _ui_thumb(self, gen, style, jpg):
        if self._thumb_gens.get(gen) != self.thumbs.get("_key"):
            return                                   # made for an older picture/palette
        url = E.data_url(jpg, "image/jpeg")
        self.thumbs[style] = url
        self.thumb.emit(json.dumps({"style": style, "url": url}, ensure_ascii=False))

    def _ui_palette(self, gen, names, roles):
        first = self._await_palette is not None
        self._await_palette = None
        if names:
            self.recipe.palette = list(names)
        self._changed(rebuild=True)
        if not first and names:
            self.toast.emit("Renkler bu görsele göre seçildi.")

    def _ui_error(self, gen, kind, msg):
        if kind == "suggest" and self._await_palette is not None:
            self._await_palette = None
            self._changed(rebuild=True)
            return
        self.failed.emit(json.dumps({"kind": kind, "msg": msg}, ensure_ascii=False))

    def _ui_busy(self, on, msg):
        self.busy.emit(json.dumps({"on": bool(on), "msg": msg}, ensure_ascii=False))

    def _thumb_key(self):
        r = self.recipe
        return (r.image_path, round(r.width_mm), tuple(r.palette), json.dumps(r.fine, sort_keys=True, default=str),
                r.zone_mask is None)

    # -------------------------------------------------- state
    def owned(self) -> list[str]:
        v = self.store.value("owned", [], type=list) or []
        names = E.lib_by_name()
        own = [n for n in v if n in names]
        return own

    def state_dict(self) -> dict:
        r = self.recipe
        names = E.lib_by_name()
        own = self.owned()
        shown = own if own else [n for n in E.STARTER_SPOOLS if n in names]
        for n in r.palette:       # a project may use spools that are not marked as owned
            if n not in shown:
                shown.append(n)
        lib = [{"name": e["ad"], "label": E.plain_name(e["ad"]), "hex": e.get("onizleme_hex") or e["hex"],
                "series": e.get("seri", ""), "code": e.get("renk_kodu", ""), "owned": e["ad"] in own}
               for e in E.library()]
        labels = E.plain_labels(list(shown) + list(r.palette))
        img = r.image_path
        return {
            "image": {"path": img, "name": os.path.basename(img) if img else "",
                      "url": self._image_thumb_url() if img else ""},
            "name": self.display_name(),
            "width_cm": round(r.width_mm / 10, 1),
            "palette": list(r.palette),
            "spools": [{"name": n, "label": labels.get(n, n), "hex": names[n].get("onizleme_hex") or names[n]["hex"],
                        "owned": n in own} for n in shown if n in names],
            "labels": labels,
            "library": lib,
            "style": r.style,
            "styles": E.style_cards(),
            "acilis": splash_enabled(self.store),
            "thumbs": {k: v for k, v in self.thumbs.items() if not k.startswith("_")},
            "mode": r.mode,
            "fine": {**E.FINE_DEFAULTS, **r.fine},
            "project": self.project_path,
            "dirty": self.dirty,
            "example": EXAMPLE_IMAGE if os.path.exists(EXAMPLE_IMAGE) else "",
            "has_full": self.plates_info is not None and self.full_gen == self.gen,
        }

    def display_name(self) -> str:
        if self.project_path:
            b = os.path.basename(self.project_path)
            return b[: -len(PROJECT_SUFFIX)] if b.endswith(PROJECT_SUFFIX) else os.path.splitext(b)[0]
        if self.recipe.image_path:
            p = Project(image_path=self.recipe.image_path)
            return p.display_name()
        return ""

    def _image_thumb_url(self) -> str:
        key = f"src_{abs(hash(self.recipe.image_path)) % 10**8}.jpg"
        if self.blobs.get(key) is None:
            try:
                from .pipeline import open_rgb
                im = open_rgb(self.recipe.image_path)
                im.thumbnail((240, 240))
                self.blobs.put(key, E.png_bytes(np.asarray(im), "JPEG", 85), "image/jpeg", keep=64)
            except Exception:  # noqa: BLE001
                return ""
        return f"tessera://app/data/{key}"

    def _emit_state(self) -> None:
        self.state.emit(json.dumps(self.state_dict(), ensure_ascii=False))
        if self.window is not None:
            self.window.update_title(self.display_name(), self.dirty)

    def _changed(self, rebuild: bool = True) -> None:
        self.dirty = bool(self.recipe.image_path)
        if rebuild and self.recipe.image_path and self.recipe.palette:
            self.gen += 1
            if self.full_job is not None and not self.full_job.finished:
                self.full_job.cancel()
            self.plates_info = None
            self._conv_gen = self.gen
            self._emit_progress(0.0, "Görsel okunuyor")
            self.engine.build(self.gen, self.recipe)
        self._emit_state()

    # -------------------------------------------------- slots: the four questions
    @Slot(result=str)
    def getState(self) -> str:  # noqa: N802
        return json.dumps(self.state_dict(), ensure_ascii=False)

    @Slot(str, result=str)
    def openImage(self, path: str = "") -> str:  # noqa: N802
        if not path:
            start = os.path.dirname(self.recipe.image_path) if self.recipe.image_path else \
                QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation)
            path, _ = QFileDialog.getOpenFileName(self.window, "Görsel seç", start, IMAGE_FILTER)
            if not path:
                return ""
        path = os.path.abspath(path)
        if path.endswith(PROJECT_SUFFIX):
            return self.open(path)
        if not os.path.exists(path):
            self.failed.emit(json.dumps({"kind": "image", "msg": f"Dosya bulunamadı: {path}"}, ensure_ascii=False))
            return ""
        try:
            from .pipeline import open_rgb
            im = open_rgb(path)
            iw, ih = im.size
        except Exception as e:  # noqa: BLE001
            self.failed.emit(json.dumps({"kind": "image", "msg": E.friendly_error(e)}, ensure_ascii=False))
            return ""
        r = E.Recipe()
        keep_palette = [n for n in self.recipe.palette]
        r.width_mm = self.recipe.width_mm         # a width typed before the picture is kept
        r.style, r.mode, r.fine = self.recipe.style, self.recipe.mode, dict(self.recipe.fine)
        r.fine.pop("kafes_rengi", None)
        r.fine.pop("taban", None)
        r.image_path = path
        z = E.sibling_zones(path)
        if z:
            r.shapes = list(z.get("sekiller", []))
            r.zones_from_file = True
            r.file_default_vocab = int(z.get("varsayilan", 0))
            rc = z.get("isin_merkezi")
            r.rays_center = list(rc) if rc else None
        self.recipe = r
        self.project_path = ""
        self.thumbs = {}
        self.plates_info = None
        self.scene_info = None
        self.last_export = None
        # first look: spools picked for this picture from what you own; the build starts when
        # the suggestion arrives (or with the first spools if it fails)
        r.palette = keep_palette or self._quick_palette(path)
        self.gen += 1
        self._await_palette = self.gen
        self.engine.suggest(self.gen, r, self.owned() or [s["name"] for s in self.state_dict()["spools"]])
        self.busy.emit(json.dumps({"on": True, "msg": "Renkler seçiliyor"}, ensure_ascii=False))
        self._changed(rebuild=False)
        return json.dumps({"w": iw, "h": ih})

    def _quick_palette(self, path) -> list[str]:
        own = self.owned() or [n for n in E.STARTER_SPOOLS if n in E.lib_by_name()]
        return own[:4]

    @Slot(float)
    def setWidth(self, cm: float) -> None:  # noqa: N802
        cm = max(10.0, min(300.0, float(cm)))
        if abs(cm * 10 - self.recipe.width_mm) < 0.5:
            return
        self.recipe.width_mm = cm * 10
        self._changed()

    @Slot(str)
    def setPalette(self, names_json: str) -> None:  # noqa: N802
        names = [n for n in json.loads(names_json or "[]") if n in E.lib_by_name()]
        if not names or names == self.recipe.palette:
            self._emit_state()
            return
        for key in ("kafes_rengi", "taban"):        # colour choices follow their spool
            bk = self.recipe.fine.get(key)
            if bk is not None:
                old = self.recipe.palette[bk] if bk < len(self.recipe.palette) else None
                self.recipe.fine[key] = names.index(old) if old in names else None
        self.recipe.palette = names
        self._changed()

    @Slot()
    def suggestPalette(self) -> None:  # noqa: N802
        if not self.recipe.image_path:
            return
        own = self.owned() or [s["name"] for s in self.state_dict()["spools"]]
        self.engine.suggest(self.gen, self.recipe, own)
        self.busy.emit(json.dumps({"on": True, "msg": "Renkler seçiliyor"}, ensure_ascii=False))

    @Slot(str)
    def setOwned(self, names_json: str) -> None:  # noqa: N802
        names = [n for n in json.loads(names_json or "[]") if n in E.lib_by_name()]
        self.store.setValue("owned", names)
        self._emit_state()

    @Slot(str)
    def setStyle(self, name: str) -> None:  # noqa: N802
        if name == self.recipe.style or name not in [c[0] for c in E.STYLE_CARDS]:
            return
        self.recipe.style = name
        self._changed()

    @Slot(str)
    def setMode(self, mode: str) -> None:  # noqa: N802
        if mode not in (Mode.PIECES, Mode.AMS) or mode == self.recipe.mode:
            return
        self.recipe.mode = mode
        self._changed()

    @Slot(str)
    def setFine(self, patch_json: str) -> None:  # noqa: N802
        patch = json.loads(patch_json or "{}")
        changed = False
        for k, v in patch.items():
            if k in E.FINE_DEFAULTS and self.recipe.fine.get(k, E.FINE_DEFAULTS[k]) != v:
                self.recipe.fine[k] = v
                changed = True
                if k == "otomatik_bolge":
                    self.recipe.zone_mask = None
                    if not self.recipe.zones_from_file:
                        self.recipe.rays_center = None
        if changed:
            self._changed(rebuild=not set(patch) <= {"renk_sayisi"})

    @Slot()
    def resetFine(self) -> None:  # noqa: N802
        self.recipe.fine = dict(E.FINE_DEFAULTS)
        self._changed()

    # -------------------------------------------------- slots: stage data
    @Slot(result=str)
    def getPreview(self) -> str:  # noqa: N802
        """Flat relit render (PNG data URL) of the current mosaic for the 2D 'Önizleme' (the one
        fixed light)."""
        if self.mosaic is None:
            return ""
        return E.data_url(E.preview_png(self.mosaic, E.LIGHT_AZ, E.LIGHT_EL))

    @Slot(result=str)
    def getSceneGLB(self) -> str:  # noqa: N802
        """{"url", "meta", "quality"} of the newest part scene ('' while none is ready)."""
        return json.dumps(self.scene_info, ensure_ascii=False) if self.scene_info else ""

    @Slot(result=str)
    def getPlates(self) -> str:  # noqa: N802
        if self.plates_info and self.plates_info.get("gen") == self.gen:
            return json.dumps(self.plates_info, ensure_ascii=False)
        return ""            # arrives through the plates signal a moment after the stones

    # -------------------------------------------------- export
    def default_export_dir(self) -> str:
        docs = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
        return os.path.join(docs, "Tessera", self.display_name() or "mozaik")

    @Slot(str, result=str)
    def export(self, out_dir: str = "") -> str:
        if not self.recipe.image_path or not self.recipe.palette:
            return ""
        if self.export_job is not None and not self.export_job.finished:
            return ""
        if not out_dir:
            base = self.default_export_dir()
            os.makedirs(os.path.dirname(base), exist_ok=True)
            out_dir = QFileDialog.getExistingDirectory(self.window, "Dosyalar nereye yazılsın?",
                                                       os.path.dirname(base))
            if not out_dir:
                return ""
            if os.listdir(out_dir) if os.path.isdir(out_dir) else False:
                out_dir = os.path.join(out_dir, self.display_name() or "mozaik")
        out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        job = ChildJob("export", {"recipe": self.recipe.to_payload(), "out": out_dir}, self)
        job.progress.connect(self.exportProgress.emit)
        job.done.connect(lambda res, d=out_dir: self._export_done(d, res))
        job.error.connect(lambda msg: self.failed.emit(json.dumps({"kind": "export", "msg": msg},
                                                                  ensure_ascii=False)))
        self.export_job = job
        job.start()
        self.exportProgress.emit(0.0, "Başlıyor")
        return out_dir

    def _export_done(self, out_dir, res) -> None:
        res = dict(res or {})
        res["out_dir"] = out_dir
        self.last_export = res
        keep = {k: res.get(k) for k in ("out_dir", "parts", "panels", "grams", "hours", "warnings", "iptal")}
        keep["plates"] = len(res.get("plates") or []) if isinstance(res.get("plates"), list) else res.get("plates")
        keep["files"] = len(res.get("files") or [])
        # R6-2: the 1:1 paper template is the main assembly aid
        for paper in ("A4", "A3"):
            f = os.path.join(out_dir, f"00_sablon_1-1_{paper}.pdf")
            keep[f"sablon_{paper.lower()}"] = f if os.path.exists(f) else ""
        keep["sablon_sayfa"] = res.get("sablon_sayfa")
        keep["plates_dir"] = os.path.join(out_dir, "02_tablalar") if os.path.isdir(os.path.join(out_dir, "02_tablalar")) \
            else out_dir
        self.exportDone.emit(json.dumps(E.plain(keep), ensure_ascii=False))

    @Slot()
    def cancelExport(self) -> None:  # noqa: N802
        if self.export_job is not None and not self.export_job.finished:
            self.export_job.cancel()
            self.failed.emit(json.dumps({"kind": "export", "msg": "Dışa aktarma durduruldu."}, ensure_ascii=False))

    @Slot(str)
    def reveal(self, path: str) -> None:
        if path and os.path.exists(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    # -------------------------------------------------- project files
    @Slot(str, result=str)
    def save(self, path: str = "") -> str:
        if not self.recipe.image_path:
            return ""
        if not path:
            path = self.project_path
        if not path:
            start = os.path.join(os.path.dirname(self.recipe.image_path), (self.display_name() or "mozaik")
                                 + PROJECT_SUFFIX)
            path, _ = QFileDialog.getSaveFileName(self.window, "Projeyi kaydet", start,
                                                  f"Tessera projesi (*{PROJECT_SUFFIX})")
            if not path:
                return ""
        if not path.endswith(PROJECT_SUFFIX):
            path = os.path.splitext(path)[0] + PROJECT_SUFFIX
        try:
            p = self.recipe.to_project()
            p.owned = self.owned()
            p.save(path)
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
            d["arayuz2"] = E.plain(E.recipe_extra(self.recipe))
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(d, fh, ensure_ascii=False, indent=1)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(json.dumps({"kind": "save", "msg": E.friendly_error(e)}, ensure_ascii=False))
            return ""
        self.project_path = path
        self.dirty = False
        self._emit_state()
        self.toast.emit("Kaydedildi.")
        return path

    @Slot(str, result=str)
    def open(self, path: str = "") -> str:
        if not path:
            path, _ = QFileDialog.getOpenFileName(self.window, "Proje aç", "",
                                                  f"Tessera projesi (*{PROJECT_SUFFIX});;{IMAGE_FILTER}")
            if not path:
                return ""
        if not path.endswith(PROJECT_SUFFIX):
            return self.openImage(path)
        try:
            p = Project.load(path)
            with open(path, encoding="utf-8") as fh:
                extra = json.load(fh).get("arayuz2") or {}
        except Exception as e:  # noqa: BLE001
            self.failed.emit(json.dumps({"kind": "open", "msg": E.friendly_error(e)}, ensure_ascii=False))
            return ""
        if not os.path.exists(p.image_path):
            self.failed.emit(json.dumps({"kind": "open", "msg": f"Projenin görseli bulunamadı: {p.image_path}"},
                                        ensure_ascii=False))
            return ""
        r = E.Recipe()
        r.image_path = p.image_path
        s = p.settings
        r.width_mm = s.width_mm
        r.palette = [f.name for f in s.palette]
        r.mode = s.mode
        r.style = E.style_name(extra.get("stil") or p.preset or E.DEFAULT_STYLE)
        r.fine = {**E.FINE_DEFAULTS, **{k: v for k, v in (extra.get("ince_ayar") or {}).items()
                                         if k in E.FINE_DEFAULTS and not (k == "kafes" and not isinstance(v, bool))}}
        r.shapes = p.shapes
        r.rays_center = list(p.rays_center) if p.rays_center else None
        r.zones_from_file = bool(extra.get("bolge_dosyadan", bool(p.shapes)))
        r.file_default_vocab = extra.get("dosya_varsayilan", s.default_vocab if p.shapes else None)
        r.zone_mask = p.paint_mask
        r.light_az, r.light_el = p.light_az, p.light_el
        r.name = p.name
        self.recipe = r
        self.project_path = os.path.abspath(path)
        self.thumbs = {}
        self.plates_info = None
        self.scene_info = None
        self._changed(rebuild=True)
        self.dirty = False
        self._emit_state()
        return self.project_path

    # -------------------------------------------------- window bar (the window has no system frame)
    @Slot()
    def winDrag(self) -> None:  # noqa: N802
        if self.window is not None and self.window.windowHandle() is not None:
            self.window.windowHandle().startSystemMove()

    @Slot(int)
    def winResize(self, edges: int) -> None:  # noqa: N802
        if self.window is not None and self.window.windowHandle() is not None:
            E_ = Qt.Edge
            v = Qt.Edges()
            for bit, e in ((1, E_.TopEdge), (2, E_.LeftEdge), (4, E_.RightEdge), (8, E_.BottomEdge)):
                if edges & bit:
                    v |= e
            self.window.windowHandle().startSystemResize(v)

    @Slot()
    def winMinimize(self) -> None:  # noqa: N802
        if self.window is not None:
            self.window.showMinimized()

    @Slot()
    def winToggle(self) -> None:  # noqa: N802
        if self.window is not None:
            self.window.showNormal() if self.window.isMaximized() else self.window.showMaximized()

    @Slot()
    def winClose(self) -> None:  # noqa: N802
        if self.window is not None:
            QTimer.singleShot(0, self.window.close)

    @Slot(bool)
    def setSplash(self, on: bool) -> None:  # noqa: N802
        self.store.setValue("acilis", bool(on))

    @Slot(str)
    def log(self, msg: str) -> None:
        print("[js]", msg, flush=True)

    def shutdown(self) -> None:
        self.engine.stop()
        for j in (self.export_job, self.full_job):
            if j is not None and not j.finished:
                j.cancel()


# ------------------------------------------------------------------ window
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):  # noqa: N802
        if os.environ.get("TESSERA_JS_LOG"):
            print(f"[js:{line}] {message}", flush=True)


class DropView(QWebEngineView):
    """Files dropped anywhere on the window open as the image (or project)."""
    def __init__(self, bridge: Bridge, parent=None):
        super().__init__(parent)
        self.bridge = bridge

    def childEvent(self, ev):  # noqa: N802 - install the filter on Chromium's render widget
        super().childEvent(ev)
        ch = ev.child()
        if ev.added() and ch is not None and ch.isWidgetType():
            ch.installEventFilter(self)

    def eventFilter(self, obj, ev):  # noqa: N802
        from PySide6.QtCore import QEvent
        t = ev.type()
        if t in (QEvent.Type.DragEnter, QEvent.Type.DragMove) and ev.mimeData().hasUrls():
            ev.acceptProposedAction()
            self.page().runJavaScript("window.tessera && tessera.dragHint(true)")
            return True
        if t == QEvent.Type.DragLeave:
            self.page().runJavaScript("window.tessera && tessera.dragHint(false)")
        if t == QEvent.Type.Drop and ev.mimeData().hasUrls():
            urls = [u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()]
            self.page().runJavaScript("window.tessera && tessera.dragHint(false)")
            if urls:
                ev.acceptProposedAction()
                QTimer.singleShot(0, lambda p=urls[0]: self.bridge.openImage(p))
            return True
        return super().eventFilter(obj, ev)


_PROFILE: Optional[QWebEngineProfile] = None


def web_profile() -> QWebEngineProfile:
    """A disk-backed profile, so page settings in localStorage (the opening animation switch)
    survive a restart. Owned by the application so it outlives every page."""
    global _PROFILE
    if _PROFILE is None:
        _PROFILE = QWebEngineProfile("tessera2", QApplication.instance())
        _PROFILE.setHttpCacheType(QWebEngineProfile.HttpCacheType.NoCache)
    return _PROFILE


class MainWindow(QMainWindow):
    def __init__(self, store: Optional[QSettings] = None):
        super().__init__()
        self.setWindowTitle("Tessera")
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.WindowMinMaxButtonsHint | Qt.WindowType.WindowSystemMenuHint)
        self.resize(1480, 920)
        self.setMinimumSize(1100, 720)
        self.bridge = Bridge(self, store)
        self.blobs = self.bridge.blobs
        prof = web_profile()
        self._handler = SchemeHandler(self.blobs, self)
        prof.removeUrlSchemeHandler(prof.urlSchemeHandler(SCHEME)) if prof.urlSchemeHandler(SCHEME) else None
        prof.installUrlSchemeHandler(SCHEME, self._handler)
        self.view = DropView(self.bridge, self)
        self.page = Page(prof, self.view)
        self.view.setPage(self.page)
        st = self.page.settings()
        st.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        st.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
        self.channel = QWebChannel(self.page)
        self.channel.registerObject("py", self.bridge)
        self.page.setWebChannel(self.channel)
        self.page.setBackgroundColor(QColor("#f3eee5"))
        self.setCentralWidget(self.view)
        self.setStyleSheet("QMainWindow{background:#f3eee5}")   # never a white flash before the first paint
        self.view.load(QUrl("tessera://app/index.html"))
        self.loaded = False
        self.page.loadFinished.connect(self._loaded)
        ico = os.path.join(WEB_DIR, "tessera_kedi.ico")
        if os.path.exists(ico):
            self.setWindowIcon(QIcon(ico))

    def _loaded(self, ok: bool) -> None:
        self.loaded = ok
        self.js(f"window.tessera_window && tessera_window({str(self.isMaximized()).lower()})")

    def update_title(self, name: str, dirty: bool) -> None:
        self.setWindowTitle(("Tessera" + (f" · {name}" if name else "")) + (" •" if dirty else ""))

    def js(self, code: str, cb=None) -> None:
        if cb is None:
            self.page.runJavaScript(code)
        else:
            self.page.runJavaScript(code, 0, cb)

    def show_initial(self) -> None:
        """Maximized on first start; afterwards the way the user left it."""
        st = self.bridge.store
        geo = st.value("window/geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        maxed = st.value("window/maximized", True)
        maxed = maxed if isinstance(maxed, bool) else str(maxed).lower() == "true"
        if maxed or geo is None:
            self.showMaximized()
        else:
            self.show()

    def changeEvent(self, ev):  # noqa: N802
        if ev.type() == ev.Type.WindowStateChange and getattr(self, "loaded", False):
            self.js(f"window.tessera_window && tessera_window({str(self.isMaximized()).lower()})")
        super().changeEvent(ev)

    def closeEvent(self, ev):  # noqa: N802
        st = self.bridge.store
        st.setValue("window/maximized", self.isMaximized())
        st.setValue("window/geometry", self.saveGeometry())
        self.bridge.shutdown()
        super().closeEvent(ev)


def splash_enabled(store: QSettings) -> bool:
    v = store.value("acilis", True)
    return v if isinstance(v, bool) else str(v).lower() != "false"


def set_app_id() -> None:
    """Windows groups taskbar buttons by AppUserModelID; without our own ID the taskbar shows
    python's icon instead of the cat."""
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Tessera.App")
        except Exception:  # noqa: BLE001
            pass


def make_app(argv=None) -> QApplication:
    set_app_id()
    register_scheme()
    # a lost GPU context (driver hiccup, sleep/resume) must not get the 3D stage blocked for
    # the rest of the session
    flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
    for f in ("--ignore-gpu-blocklist", "--disable-domain-blocking-for-3d-apis"):
        if f not in flags:
            flags = (flags + " " + f).strip()
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = flags
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setApplicationName("Tessera")
    app.setOrganizationName("Tessera")
    ico = os.path.join(WEB_DIR, "tessera_kedi.ico")
    if os.path.exists(ico):
        app.setWindowIcon(QIcon(ico))
    collect_on_gui_thread(app)
    return app


def collect_on_gui_thread(app) -> None:
    """Python's cycle collector runs on whichever thread happens to allocate; on the engine thread it
    may destroy Qt objects there, which aborts the program. So it only runs here, on the GUI thread."""
    if getattr(app, "_gc_timer", None) is not None:
        return
    gc.disable()
    limit = gc.get_threshold()[0] or 2000
    n = {"i": 0}

    def step():
        if gc.get_count()[0] < limit:
            return
        n["i"] += 1
        gc.collect(1 if n["i"] % 20 else 2)

    app._gc_timer = QTimer(app)
    app._gc_timer.timeout.connect(step)
    app._gc_timer.start(500)


def to_front(win) -> None:
    """Above other windows even if Windows refuses to hand over the focus; it does not stay on top."""
    if sys.platform != "win32":
        return
    import ctypes
    f = ctypes.windll.user32.SetWindowPos
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 4 + [ctypes.c_uint]
    for after in (-1, -2):                        # HWND_TOPMOST, then HWND_NOTOPMOST
        f(int(win.winId()), ctypes.c_void_p(after), 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0010)


def main(argv=None, card=None) -> int:
    """``card``: the opening card's handle (tessera.baslat); the window waits until it has finished."""
    argv = list(argv if argv is not None else sys.argv)
    app = make_app(argv)
    target = argv[1] if len(argv) > 1 and os.path.exists(argv[1]) else None
    win = MainWindow()
    t = {"n": 0}

    def open_target():
        if not win.loaded:
            QTimer.singleShot(200, open_target)
        elif target.endswith(PROJECT_SUFFIX):
            win.bridge.open(target)
        else:
            win.bridge.openImage(target)

    def painted() -> bool:
        """The dark title bar is really there (until the first paint the view is plain paper)."""
        im = win.view.grab(QRect(0, 0, 320, 24)).toImage()
        return any(im.pixelColor(x, im.height() // 2).lightness() < 90 for x in range(0, im.width(), 8))

    def lift(n):
        """The window is up but still see-through; swap it for the card in one go once it is painted."""
        if n < 80 and not painted():
            QTimer.singleShot(25, lambda: lift(n + 1))
            return
        win.setWindowOpacity(1.0)
        to_front(win)
        QTimer.singleShot(60, card.close)

    def reveal():
        t["n"] += 1
        if card is not None and t["n"] < 400 and not (card.done.is_set() and (win.loaded or t["n"] > 300)):
            QTimer.singleShot(25, reveal)         # at most 10 s, whatever happens to the card
            return
        if card is not None:
            win.setWindowOpacity(0.0)
        win.show_initial()
        win.raise_()
        win.activateWindow()
        if card is not None:
            lift(0)
        if target:
            QTimer.singleShot(300, open_target)

    reveal()
    try:
        return app.exec()
    finally:
        if card is not None:
            card.close()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
