"""Round 2 views: budget bars, the "Tablalar" gallery and the optional "Panel 3D" view."""
from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                               QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

from ..model import VOCAB_NAMES_TR, Vocab
from .engine import BUDGET_KG_M2, BUDGET_KG_M2_MAX, BUDGET_PARTS_M2, RELIEF_TARGET, budget_level
from .theme import body_font, current, head_font
from .widgets import label, section, tr_num


def level_colour(level: str) -> str:
    t = current()
    return {"ok": t.ok, "warn": t.warn, "bad": t.red}.get(level, t.muted)


# ====================================================================== budget
class MeterBar(QWidget):
    """One value against a target: filled bar, target tick, colour by level."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frac = 0.0
        self.target_frac = 0.5
        self.level = "ok"
        self.setFixedHeight(8)

    def set(self, value: Optional[float], target: float, level: str) -> None:
        scale = target * 2
        self.frac = 0.0 if value is None else min(1.0, value / scale)
        self.target_frac = target / scale
        self.level = level
        self.update()

    def paintEvent(self, _):
        t = current()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(0, 1, self.width(), 6)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(t.line))
        p.drawRoundedRect(r, 3, 3)
        p.setBrush(QColor(level_colour(self.level)))
        p.drawRoundedRect(QRectF(0, 1, max(6, self.width() * self.frac), 6), 3, 3)
        x = self.width() * self.target_frac
        p.setPen(QPen(QColor(t.text), 1.5))
        p.drawLine(QPointF(x, 0), QPointF(x, 8))
        p.end()


class ReliefBar(QWidget):
    """Share of area that is low (≤3 mm) / medium (≤8 mm) / high, with the 70/25/5 target."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dist = (1.0, 0.0, 0.0)
        self.setFixedHeight(14)

    def set(self, dist) -> None:
        self.dist = tuple(float(d) for d in dist[:3])
        self.update()

    def paintEvent(self, _):
        t = current()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cols = [QColor(t.faint), QColor(t.gold), QColor(t.red)]
        x = 0.0
        tot = max(sum(self.dist), 1e-6)
        p.setPen(Qt.NoPen)
        for d, c in zip(self.dist, cols):
            w = self.width() * d / tot
            p.setBrush(c)
            p.drawRect(QRectF(x, 2, w, 8))
            x += w
        p.setPen(QPen(QColor(t.text), 1.5))
        acc = 0.0
        for tgt in RELIEF_TARGET[:2]:
            acc += tgt
            xx = self.width() * acc
            p.drawLine(QPointF(xx, 0), QPointF(xx, 12))
        p.end()


class BudgetCard(QFrame):
    """kg/m², parts/m² and relief distribution against the targets (PLAN §9, R2-2)."""
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(6)
        self.verdict = label("", "modeTitle")
        v.addWidget(self.verdict)
        self.rows = {}
        for key, title in (("kg", "Ağırlık"), ("parts", "Parça yoğunluğu"), ("relief", "Kabartma yüksekliği")):
            h = QHBoxLayout()
            h.addWidget(label(title))
            h.addStretch(1)
            val = label("—", "value")
            h.addWidget(val)
            v.addLayout(h)
            bar = ReliefBar() if key == "relief" else MeterBar()
            v.addWidget(bar)
            hint = label("", "hint", wrap=True)
            v.addWidget(hint)
            self.rows[key] = (val, bar, hint)
        self.advice = label("", "hint", wrap=True)
        v.addWidget(self.advice)
        self.budget: dict = {}

    def update_budget(self, b: dict) -> None:
        self.budget = dict(b or {})
        kg, parts, dist = b.get("kg_m2"), b.get("parca_m2"), b.get("dagilim") or (1, 0, 0)
        lv = {"kg": budget_level("kg", kg), "parts": budget_level("parts", parts),
              "relief": budget_level("relief", dist)}
        val, bar, hint = self.rows["kg"]
        val.setText(f"{tr_num(kg, 1)} kg/m²" if kg is not None else "—")
        val.setStyleSheet(f"color: {level_colour(lv['kg'])};")
        bar.set(kg, BUDGET_KG_M2, lv["kg"])
        hint.setText(f"metrekare başına ağırlık · {tr_num(BUDGET_KG_M2, 0)}'e kadar yeşil, "
                     f"{tr_num(BUDGET_KG_M2_MAX, 0)}'e kadar sarı")
        val, bar, hint = self.rows["parts"]
        exact = bool(b.get("parca_kesin"))
        val.setText(("" if exact else "≈ ") + f"{tr_num(parts)} parça/m²" if parts is not None else "—")
        val.setStyleSheet(f"color: {level_colour(lv['parts'])};")
        bar.set(parts, BUDGET_PARTS_M2, lv["parts"])
        words = {"ok": "elle dizmesi kolay", "warn": "dizmesi biraz uğraştırır",
                 "bad": "çok parça: dizmek uzun sürer"}[lv["parts"]]
        extra = ""
        if b.get("hassas") is not None:
            extra = f" · {tr_num(b['hassas'])} hassas parça (%{tr_num(b.get('hassas_oran', 0) * 100)})"
        hint.setText(words + extra + ("" if exact else " · kesin sayı hesaplanıyor"))
        val, bar, hint = self.rows["relief"]
        lo, mid, hi = (list(dist) + [0, 0, 0])[:3]
        val.setText(f"%{tr_num(hi * 100)} yüksek")
        val.setStyleSheet(f"color: {level_colour(lv['relief'])};")
        bar.set((lo, mid, hi))
        hint.setText(f"alçak (≤3 mm) %{tr_num(lo * 100)} · orta (≤8 mm) %{tr_num(mid * 100)} · "
                     f"yüksek %{tr_num(hi * 100)}  —  hedef %70 · %25 · %5")
        # R3-5: the card's verdict and colour follow the weight only
        worst = lv["kg"] if kg is not None else "ok"
        self.verdict.setText({"ok": "Ağırlık uygun", "warn": "Ağırlık sınırda", "bad": "Çok ağır"}[worst])
        self.verdict.setStyleSheet(f"color: {level_colour(worst)};")
        self.setStyleSheet(f"#card {{ border-color: {level_colour(worst) if worst != 'ok' else current().line}; }}")
        tips = [str(b["mesaj"])] if b.get("mesaj") and worst != "ok" else []
        if lv["kg"] != "ok" or lv["relief"] != "ok":
            fig, foc = (VOCAB_NAMES_TR[Vocab.FIGURE].lower(), VOCAB_NAMES_TR[Vocab.FOCAL].lower())
            tips.append(f"Yükseklikleri alçalt (özellikle {fig} ve {foc}) ya da bir hazır ayar seç.")
        if lv["parts"] != "ok":
            tips.append("Taş boylarını büyüt ya da “aynı renk komşuları birleştir”i aç.")
        self.advice.setText(" ".join(tips))
        self.advice.setVisible(bool(tips))

    def worst(self) -> str:
        """The card's colour: weight only (R3-5)."""
        return budget_level("kg", self.budget.get("kg_m2"))

    def mousePressEvent(self, e) -> None:
        self.clicked.emit()


# ====================================================================== Tablalar gallery
def collect_export_images(res: dict) -> list[tuple[str, str, str]]:
    """-> [(group, caption, path)] from an export result: plate previews (plate_images, 3D team)
    and the assembly guide PNGs. Tolerant of list-of-paths or list-of-dicts."""
    out: list[tuple[str, str, str]] = []
    for it in res.get("plate_images") or []:
        if isinstance(it, str):
            path, cap = it, os.path.splitext(os.path.basename(it))[0].replace("_", " ")
        elif isinstance(it, dict):
            path = it.get("path") or it.get("png") or it.get("dosya") or ""
            cap = it.get("ad") or it.get("baslik") or it.get("caption") or \
                os.path.splitext(os.path.basename(path))[0].replace("_", " ")
        else:
            continue
        if path and os.path.exists(path):
            out.append(("Tablalar", cap, path))
    for f in res.get("files") or []:
        if isinstance(f, str) and f.lower().endswith(".png") and "montaj" in os.path.basename(f).lower() \
                and os.path.exists(f):
            out.append(("Montaj kılavuzu", os.path.splitext(os.path.basename(f))[0].replace("_", " "), f))
    return out


class Thumb(QLabel):
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Büyük açmak için tıkla")
        self.setAlignment(Qt.AlignCenter)
        self._pm = QPixmap(path)

    def fit(self, w: int) -> None:
        if not self._pm.isNull():
            self.setPixmap(self._pm.scaledToWidth(max(80, w), Qt.SmoothTransformation))

    def mousePressEvent(self, e) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.path))


class PlateGallery(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("stage")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.inner = QWidget()
        self.inner.setObjectName("stage")
        self.setWidget(self.inner)
        self.lay = QVBoxLayout(self.inner)
        self.lay.setContentsMargins(28, 22, 28, 28)
        self.lay.setSpacing(12)
        self.thumbs: list[Thumb] = []
        self.items: list = []

    def set_items(self, items: list[tuple[str, str, str]]) -> None:
        self.items = items
        while self.lay.count():
            it = self.lay.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
            elif it.layout():
                _clear(it.layout())
        self.thumbs = []
        group = None
        grid = None
        k = 0
        for g, cap, path in items:
            if g != group:
                group = g
                self.lay.addWidget(section(g))
                gw = QWidget()
                grid = QGridLayout(gw)
                grid.setContentsMargins(0, 0, 0, 8)
                grid.setHorizontalSpacing(16)
                grid.setVerticalSpacing(14)
                self.lay.addWidget(gw)
                k = 0
            box = QVBoxLayout()
            box.setSpacing(4)
            th = Thumb(path)
            self.thumbs.append(th)
            box.addWidget(th)
            box.addWidget(label(cap, "hint"))
            w = QWidget()
            w.setLayout(box)
            grid.addWidget(w, k, 0)
            k += 1
        self.lay.addStretch(1)
        self._fit()

    def _fit(self) -> None:
        w = min(1100, self.viewport().width() - 56 - 14)
        for th in self.thumbs:
            th.fit(w)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._fit()


def _clear(layout) -> None:
    while layout.count():
        it = layout.takeAt(0)
        if it.widget():
            it.widget().deleteLater()
        elif it.layout():
            _clear(it.layout())


# ====================================================================== Panel 3D (optional)
def preview_parts_fn():
    """3D team's preview_parts, if it exists (export.api, geometry, geometry.mesh)."""
    import importlib
    for mod in ("tessera.export.api", "tessera.geometry", "tessera.geometry.mesh", "tessera.geometry.layout"):
        try:
            fn = getattr(importlib.import_module(mod), "preview_parts", None)
        except Exception:  # noqa: BLE001
            fn = None
        if callable(fn):
            return fn
    return None


def normalise_meshes(out) -> list[tuple[np.ndarray, np.ndarray, tuple]]:
    """Accept many plausible shapes of preview_parts output -> [(V (n,3), F (m,3), rgb)]."""
    if isinstance(out, dict):
        out = out.get("parts") or out.get("parcalar") or out.get("meshes") or [out]
    res = []
    for it in out or []:
        V = F = None
        rgb = (180, 175, 165)
        if isinstance(it, (tuple, list)) and len(it) >= 2 and not isinstance(it[0], (int, float)):
            V, F = it[0], it[1]
            if len(it) >= 3 and it[2] is not None:
                rgb = it[2]
        elif isinstance(it, dict):
            V = it.get("vertices") if it.get("vertices") is not None else it.get("V")
            F = it.get("faces") if it.get("faces") is not None else it.get("F")
            rgb = it.get("rgb") or it.get("colour") or it.get("color") or it.get("renk") or rgb
            mesh = it.get("mesh") or it.get("manifold")
            if V is None and mesh is not None:
                V, F = _arrays(mesh)
        else:
            V, F = _arrays(it)
        if V is None or F is None:
            continue
        if isinstance(rgb, str):
            c = QColor(rgb)
            rgb = (c.red(), c.green(), c.blue())
        res.append((np.asarray(V, np.float64)[:, :3], np.asarray(F, np.int64)[:, :3], tuple(int(c) for c in rgb[:3])))
    return res


def _arrays(obj):
    if hasattr(obj, "vertices") and hasattr(obj, "faces"):
        return obj.vertices, obj.faces
    try:
        from ..geometry.mesh import mesh_arrays
        return mesh_arrays(obj)[:2]
    except Exception:  # noqa: BLE001
        return None, None


def quick_kw(fn) -> dict:
    """preview_parts(quality="fast") (~0.2 s height-map view) to show before quality="full"."""
    import inspect
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    return {"quality": "fast"} if "quality" in params else {}


def call_preview_parts(fn, mosaic, panel: int, az: float = -30.0, el: float = 50.0, size: int = 900,
                       extra: dict | None = None):
    import inspect
    try:
        params = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        params = set()
    if "panel_id" in params:
        kw = {"panel_id": panel}
        if "size" in params:
            kw["size"] = size
        if "az_deg" in params:
            kw.update(az_deg=az, el_deg=el)
        kw.update(extra or {})
        return fn(mosaic, **kw)
    if "panel" in params:
        return fn(mosaic, panel=panel)
    try:
        return fn(mosaic, panel)
    except TypeError:
        return fn(mosaic)


def preview_result(out):
    """-> ("image", uint8 RGB array) for a ready-rendered view, or ("meshes", [...])."""
    if isinstance(out, (bytes, bytearray)):
        from io import BytesIO
        from PIL import Image
        return "image", np.asarray(Image.open(BytesIO(out)).convert("RGB"))
    if isinstance(out, np.ndarray) and out.ndim == 3:
        return "image", out
    return "meshes", normalise_meshes(out)


class Panel3DView(QWidget):
    """Shows the 3D team's rendered panel (an image), or renders meshes itself (flat-shaded
    triangles, painter's algorithm). Drag to orbit; an image is re-requested on release."""
    panel_changed = Signal(int)
    view_changed = Signal(float, float)     # az, el after a drag (image mode)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("stage")
        self.setAutoFillBackground(True)
        self.meshes: list = []
        self.busy = False
        self.image: Optional[QImage] = None
        self.yaw, self.pitch = -30.0, 50.0
        self._drag: Optional[QPointF] = None
        self.message = "Panel hazırlanıyor…"
        self.setCursor(Qt.OpenHandCursor)
        bar = QHBoxLayout(self)
        bar.setContentsMargins(16, 12, 16, 12)
        top = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(label("Panel", "hint"))
        self.spin = QSpinBox()
        self.spin.setRange(1, 1)
        self.spin.setFixedWidth(64)
        self.spin.valueChanged.connect(lambda v: self.panel_changed.emit(v - 1))
        row.addWidget(self.spin)
        row.addWidget(label("· sürükleyerek döndür", "hint"))
        row.addStretch(1)
        top.addLayout(row)
        top.addStretch(1)
        bar.addLayout(top)

    def set_panel_count(self, n: int) -> None:
        self.spin.blockSignals(True)
        self.spin.setRange(1, max(1, n))
        self.spin.blockSignals(False)

    def set_meshes(self, meshes, message: str = "") -> None:
        self.meshes = meshes
        self.image = None
        self.message = message
        self.update()

    def set_image(self, arr) -> None:
        from .zones import rgb_to_qimage
        self.image = rgb_to_qimage(arr)
        self.meshes = []
        self.message = ""
        self.update()

    def has_content(self) -> bool:
        return bool(self.meshes) or self.image is not None

    def set_busy(self, on: bool) -> None:
        self.busy = on
        self.update()

    def mousePressEvent(self, e) -> None:
        self._drag = e.position()
        self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, e) -> None:
        if self._drag is not None:
            d = e.position() - self._drag
            self._drag = e.position()
            self.yaw += d.x() * 0.4
            self.pitch = max(-85.0, min(85.0, self.pitch + d.y() * 0.4))
            self.update()

    def mouseReleaseEvent(self, e) -> None:
        self._drag = None
        self.setCursor(Qt.OpenHandCursor)
        if self.image is not None:
            self.pitch = max(10.0, min(85.0, self.pitch))
            self.view_changed.emit(self.yaw, self.pitch)
        self.update()

    def paintEvent(self, _):
        t = current()
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(t.stage))
        if self.image is not None:
            s = min((self.width() - 40) / self.image.width(), (self.height() - 70) / self.image.height())
            w, h = self.image.width() * s, self.image.height() * s
            p.setRenderHint(QPainter.SmoothPixmapTransform)
            p.drawImage(QRectF((self.width() - w) / 2, 50 + (self.height() - 70 - h) / 2, w, h), self.image)
            if self.busy:
                p.setPen(QColor(t.gold))
                p.setFont(body_font(12))
                p.drawText(QRectF(0, 14, self.width() - 20, 24), Qt.AlignRight | Qt.AlignVCenter,
                           "Güncelleniyor…")
            if self._drag is not None:
                p.setPen(QColor(t.muted))
                p.setFont(body_font(12))
                p.drawText(QRectF(0, self.height() - 30, self.width(), 24), Qt.AlignCenter,
                           f"Bırakınca yeni açıdan çizilir · yön {int(self.yaw)}°, eğim {int(self.pitch)}°")
            p.end()
            return
        if not self.meshes:
            p.setPen(QColor(t.muted))
            p.setFont(body_font(14))
            p.drawText(self.rect(), Qt.AlignCenter, self.message or "")
            p.end()
            return
        V_all = np.concatenate([m[0] for m in self.meshes])
        c = (V_all.min(0) + V_all.max(0)) / 2
        span = float(np.ptp(V_all[:, :2], axis=0).max()) or 1.0
        ya, pa = math.radians(self.yaw), math.radians(self.pitch)
        Rz = np.array([[math.cos(ya), -math.sin(ya), 0], [math.sin(ya), math.cos(ya), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, math.cos(pa), -math.sin(pa)], [0, math.sin(pa), math.cos(pa)]])
        R = Rx @ Rz
        scale = 0.8 * min(self.width(), self.height()) / span
        light = np.array([-0.4, -0.5, 0.75])
        light /= np.linalg.norm(light)
        tris, depth, cols = [], [], []
        budget = 12000 if self._drag is not None else 60000
        total = sum(len(m[1]) for m in self.meshes)
        step = max(1, total // budget)
        for V, F, rgb in self.meshes:
            P = (V - c) @ R.T
            F = F[::step]
            A, B, C = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
            n = np.cross(B - A, C - A)
            nn = np.linalg.norm(n, axis=1) + 1e-9
            facing = n[:, 2] > 0
            shade = np.clip(0.35 + 0.65 * (n @ light) / nn, 0.2, 1.0)
            for k in np.nonzero(facing)[0]:
                tris.append((A[k], B[k], C[k]))
                depth.append(float(A[k][2] + B[k][2] + C[k][2]))
                cols.append(tuple(int(ch * shade[k]) for ch in rgb))
        order = np.argsort(depth)
        p.setRenderHint(QPainter.Antialiasing, self._drag is None)
        cx, cy = self.width() / 2, self.height() / 2 + 20
        for i in order:
            a, b, cc = tris[i]
            poly = QPolygonF([QPointF(cx + a[0] * scale, cy - a[1] * scale),
                              QPointF(cx + b[0] * scale, cy - b[1] * scale),
                              QPointF(cx + cc[0] * scale, cy - cc[1] * scale)])
            col = QColor(*cols[i])
            p.setPen(QPen(col, 0.6))
            p.setBrush(col)
            p.drawPolygon(poly)
        p.end()
