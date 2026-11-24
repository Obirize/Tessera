"""The big zoomable / pannable preview, also the zone drawing surface.

Scene coordinates = pixels of the (capped) original image, so every flat view (Orijinal,
Mozaik, Bölgeler, Parça haritası) shares the same zoom and position when you switch tabs.
"""
from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap,
                           QTransform)
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem,
                               QGraphicsPixmapItem, QGraphicsScene, QGraphicsView)

from ..project import PAINT_DEFAULT
from .zones import ZoneDoc

TOOLS = ("pan", "brush", "eraser", "polygon", "rect", "circle", "center")


class SunMarker(QGraphicsItem):
    """Rays centre: a small gold sun with spokes. Ignores zoom (constant screen size)."""

    def __init__(self, gold: QColor, ink: QColor):
        super().__init__()
        self.gold, self.ink = gold, ink
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        self.setZValue(30)

    def boundingRect(self) -> QRectF:
        return QRectF(-26, -26, 52, 52)

    def paint(self, p: QPainter, *_):
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(self.ink, 4, Qt.SolidLine, Qt.RoundCap))
        for k in range(12):
            a = k * math.pi / 6
            p.drawLine(QPointF(math.cos(a) * 12, math.sin(a) * 12), QPointF(math.cos(a) * 21, math.sin(a) * 21))
        p.setPen(QPen(self.gold, 2, Qt.SolidLine, Qt.RoundCap))
        for k in range(12):
            a = k * math.pi / 6
            p.drawLine(QPointF(math.cos(a) * 12, math.sin(a) * 12), QPointF(math.cos(a) * 21, math.sin(a) * 21))
        p.setPen(QPen(self.ink, 2))
        p.setBrush(self.gold)
        p.drawEllipse(QPointF(0, 0), 8, 8)


class Canvas(QGraphicsView):
    zoom_changed = Signal(float)
    tool_used = Signal(str)            # emitted after a shape/centre was placed
    brush_radius_changed = Signal(float)
    files_dropped = Signal(list)

    def __init__(self, zones: ZoneDoc, parent=None):
        super().__init__(parent)
        self.zones = zones
        self.setObjectName("stage")
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)

        self.scene_w, self.scene_h = 1000.0, 1000.0
        self.base = QGraphicsPixmapItem()
        self.base.setTransformationMode(Qt.SmoothTransformation)
        self.base.setShapeMode(QGraphicsPixmapItem.BoundingRectShape)
        self.scene().addItem(self.base)
        self.overlay = QGraphicsPixmapItem()
        self.overlay.setTransformationMode(Qt.FastTransformation)
        self.overlay.setOpacity(0.55)
        self.overlay.setZValue(5)
        self.overlay.setVisible(False)
        self.scene().addItem(self.overlay)
        self.preview_path = QGraphicsPathItem()
        self.preview_path.setZValue(20)
        self.scene().addItem(self.preview_path)
        self.select_path = QGraphicsPathItem()
        self.select_path.setZValue(19)
        self.scene().addItem(self.select_path)
        self.cursor_ring = QGraphicsEllipseItem()
        self.cursor_ring.setZValue(25)
        self.cursor_ring.setVisible(False)
        self.scene().addItem(self.cursor_ring)
        self.sun: Optional[SunMarker] = None

        self.tool = "pan"
        self.editing = False          # zone tools active only on the Bölgeler tab
        self.paint_vocab = 0
        self.brush_mm = 20.0
        self.width_mm = 424.0
        self._fit_mode = True
        self.reserve_left = 0
        self.reserve_right = 0
        self._pan_from: Optional[QPointF] = None
        self._space = False
        self._stroke_last: Optional[QPointF] = None
        self._poly: list[QPointF] = []
        self._drag_from: Optional[QPointF] = None
        self._mouse_scene = QPointF()
        self._bg = QColor("#111")
        self._grid = QColor("#1a1a1a")
        self._accent = QColor("#D9A441")
        self._ink = QColor("#111")
        self.zones.overlay_changed.connect(self._refresh_overlay)

    # ---------------------------------------------- theming
    def set_colours(self, stage: str, grid: str, accent: str, ink: str) -> None:
        self._bg, self._grid, self._accent, self._ink = QColor(stage), QColor(grid), QColor(accent), QColor(ink)
        self.setBackgroundBrush(self._bg)
        if self.sun:
            self.sun.gold, self.sun.ink = self._accent, self._ink
        pen = QPen(self._accent, 2, Qt.DashLine)
        pen.setCosmetic(True)
        self.select_path.setPen(pen)
        pen2 = QPen(QColor("#FFFFFF"), 1.6)
        pen2.setCosmetic(True)
        self.preview_path.setPen(pen2)
        ring = QPen(QColor("#FFFFFF"), 1.4)
        ring.setCosmetic(True)
        self.cursor_ring.setPen(ring)
        self.viewport().update()

    def drawBackground(self, p: QPainter, rect: QRectF) -> None:
        p.fillRect(rect, self._bg)
        # faint constructivist grid: hints at tiles without competing with the image
        step = 48 / max(self.transform().m11(), 1e-3)
        step = max(step, 8)
        pen = QPen(self._grid, 1)
        pen.setCosmetic(True)
        p.setPen(pen)
        x = math.floor(rect.left() / step) * step
        while x < rect.right():
            p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += step
        y = math.floor(rect.top() / step) * step
        while y < rect.bottom():
            p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += step

    # ---------------------------------------------- content
    def set_scene_size(self, w: float, h: float) -> None:
        self.scene_w, self.scene_h = float(w), float(h)
        self.scene().setSceneRect(QRectF(-w * 2, -h * 2, w * 5, h * 5))
        self._refresh_overlay()
        self._refresh_sun()
        self._fit_mode = True
        self.fit()

    def set_image(self, img: Optional[QImage], keep_view: bool = True) -> None:
        """Show an image stretched to scene width (height follows its own aspect)."""
        if img is None or img.isNull():
            self.base.setPixmap(QPixmap())
            return
        pm = QPixmap.fromImage(img)
        self.base.setPixmap(pm)
        s = self.scene_w / max(1, pm.width())
        self.base.setTransform(QTransform.fromScale(s, s))
        if self._fit_mode or not keep_view:
            self.fit()

    def content_rect(self) -> QRectF:
        r = self.base.sceneBoundingRect()
        if r.isEmpty():
            r = QRectF(0, 0, self.scene_w, self.scene_h)
        return r

    def fit(self) -> None:
        r = self.content_rect()
        if r.isEmpty() or self.viewport().width() < 10:
            return
        m = 28
        left, right = self.reserve_left, self.reserve_right
        vw = self.viewport().width() - 2 * m - left - right
        vh = self.viewport().height() - 2 * m
        if vw < 200:                      # too narrow: ignore the reserved strips
            left = right = 0
            vw = self.viewport().width() - 2 * m
        s = min(vw / r.width(), vh / r.height())
        self.setTransform(QTransform.fromScale(s, s))
        # centre the artwork in the free area between the floating boxes
        self.centerOn(r.center() + QPointF((right - left) / 2 / s, 0))
        self._fit_mode = True
        self.zoom_changed.emit(s)

    def set_reserved(self, left: int, right: int) -> None:
        if (left, right) != (self.reserve_left, self.reserve_right):
            self.reserve_left, self.reserve_right = left, right
            if self._fit_mode:
                self.fit()

    def zoom_by(self, f: float, anchor_view: Optional[QPointF] = None) -> None:
        cur = self.transform().m11()
        nf = max(0.05, min(40.0, cur * f)) / cur
        if anchor_view is None:
            self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.scale(nf, nf)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._fit_mode = False
        self.zoom_changed.emit(self.transform().m11())
        self._update_cursor_ring()

    def zoom_actual(self) -> None:
        """1 image pixel = 1 screen pixel."""
        self.setTransform(QTransform())
        self._fit_mode = False
        self.zoom_changed.emit(1.0)

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if self._fit_mode:
            self.fit()

    def show_zones(self, on: bool) -> None:
        self.overlay.setVisible(on)
        self.select_path.setVisible(on)
        self._refresh_sun()

    def _refresh_overlay(self) -> None:
        ov = self.zones.overlay()
        if ov.isNull():
            return
        self.overlay.setPixmap(QPixmap.fromImage(ov))
        self.overlay.setTransform(QTransform.fromScale(self.scene_w / max(1, ov.width()),
                                                       self.scene_h / max(1, ov.height())))
        self._refresh_sun()

    def _refresh_sun(self) -> None:
        rc = self.zones.rays_center
        show = self.overlay.isVisible() and rc is not None
        if show and self.sun is None:
            self.sun = SunMarker(self._accent, self._ink)
            self.scene().addItem(self.sun)
        if self.sun:
            self.sun.setVisible(show)
            if rc:
                self.sun.setPos(rc[0] * self.scene_w, rc[1] * self.scene_h)

    def highlight_shape(self, shape: Optional[dict]) -> None:
        self.select_path.setPath(self._shape_path(shape) if shape else QPainterPath())

    def _shape_path(self, s: dict) -> QPainterPath:
        w, h = self.scene_w, self.scene_h
        path = QPainterPath()
        if "polygon" in s:
            pts = [QPointF(x * w, y * h) for x, y in s["polygon"]]
            if pts:
                path.moveTo(pts[0])
                for q in pts[1:]:
                    path.lineTo(q)
                path.closeSubpath()
        elif "rect" in s:
            x0, y0, x1, y1 = s["rect"]
            path.addRect(QRectF(QPointF(x0 * w, y0 * h), QPointF(x1 * w, y1 * h)).normalized())
        elif "circle" in s:
            cx, cy, r = s["circle"]
            path.addEllipse(QPointF(cx * w, cy * h), r * w, r * w)
        return path

    # ---------------------------------------------- tools
    def set_tool(self, tool: str) -> None:
        self.tool = tool
        self._poly.clear()
        self._drag_from = None
        self.preview_path.setPath(QPainterPath())
        self._update_cursor()

    def set_editing(self, on: bool) -> None:
        self.editing = on
        self.show_zones(on)
        self._update_cursor()

    def set_brush_mm(self, mm: float) -> None:
        self.brush_mm = max(1.0, mm)
        self._update_cursor_ring()

    def _active_tool(self) -> str:
        if self._space or not self.editing:
            return "pan"
        return self.tool

    def _update_cursor(self) -> None:
        t = self._active_tool()
        self.viewport().setCursor({"pan": Qt.OpenHandCursor, "brush": Qt.BlankCursor,
                                   "eraser": Qt.BlankCursor}.get(t, Qt.CrossCursor))
        self.cursor_ring.setVisible(t in ("brush", "eraser") and self.underMouse())
        self._update_cursor_ring()

    def _brush_scene_radius(self) -> float:
        return self.brush_mm / max(self.width_mm, 1) * self.scene_w / 2

    def _update_cursor_ring(self) -> None:
        r = self._brush_scene_radius()
        c = self._mouse_scene
        self.cursor_ring.setRect(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r))

    def _to_mask(self, sp: QPointF) -> QPointF:
        mw, mh = self.zones.size
        return QPointF(sp.x() / self.scene_w * mw, sp.y() / self.scene_h * mh)

    def _frac(self, sp: QPointF) -> list[float]:
        return [round(min(1.0, max(0.0, sp.x() / self.scene_w)), 4),
                round(min(1.0, max(0.0, sp.y() / self.scene_h)), 4)]

    # ---------------------------------------------- events
    def wheelEvent(self, e) -> None:
        d = e.angleDelta().y()
        if e.modifiers() & Qt.AltModifier and self._active_tool() in ("brush", "eraser"):
            self.set_brush_mm(self.brush_mm * (1.12 if d > 0 else 1 / 1.12))
            self.brush_radius_changed.emit(self.brush_mm)
            return
        if d:
            self.zoom_by(1.15 if d > 0 else 1 / 1.15, e.position())

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = True
            self._update_cursor()
            return
        if self.tool == "polygon" and self._poly:
            if e.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._finish_polygon()
                return
            if e.key() == Qt.Key_Escape:
                self.set_tool("polygon")
                return
            if e.key() == Qt.Key_Backspace:
                self._poly.pop()
                self._update_poly_preview()
                return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e) -> None:
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = False
            self._update_cursor()
            return
        super().keyReleaseEvent(e)

    def mousePressEvent(self, e) -> None:
        sp = self.mapToScene(e.position().toPoint())
        t = self._active_tool()
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and t == "pan"):
            self._pan_from = e.position()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            return
        if e.button() == Qt.RightButton and t == "polygon" and self._poly:
            self._finish_polygon()
            return
        if e.button() != Qt.LeftButton:
            return
        if t in ("brush", "eraser"):
            self.zones.begin_stroke()
            mp = self._to_mask(sp)
            self._stroke_last = mp
            self._paint_to(mp)
        elif t == "polygon":
            if len(self._poly) >= 3:
                first = self.mapFromScene(self._poly[0])
                if (QPointF(first) - e.position()).manhattanLength() < 12:
                    self._finish_polygon()
                    return
            self._poly.append(sp)
            self._update_poly_preview()
        elif t in ("rect", "circle"):
            self._drag_from = sp
        elif t == "center":
            self.zones.set_rays_center(tuple(self._frac(sp)))
            self.tool_used.emit("center")

    def mouseDoubleClickEvent(self, e) -> None:
        if self._active_tool() == "polygon" and len(self._poly) >= 3:
            self._finish_polygon()
            return
        if self._active_tool() == "pan":
            self.fit()
            return
        super().mouseDoubleClickEvent(e)

    def mouseMoveEvent(self, e) -> None:
        sp = self.mapToScene(e.position().toPoint())
        self._mouse_scene = sp
        if self._pan_from is not None:
            d = e.position() - self._pan_from
            self._pan_from = e.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - d.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - d.y()))
            self._fit_mode = False
            return
        t = self._active_tool()
        if t in ("brush", "eraser"):
            self.cursor_ring.setVisible(True)
            self._update_cursor_ring()
            if self._stroke_last is not None and e.buttons() & Qt.LeftButton:
                self._paint_to(self._to_mask(sp))
        elif t == "polygon" and self._poly:
            self._update_poly_preview(sp)
        elif t in ("rect", "circle") and self._drag_from is not None:
            path = QPainterPath()
            a = self._drag_from
            if t == "rect":
                path.addRect(QRectF(a, sp).normalized())
            else:
                r = math.hypot(sp.x() - a.x(), sp.y() - a.y())
                path.addEllipse(a, r, r)
            self.preview_path.setPath(path)

    def mouseReleaseEvent(self, e) -> None:
        sp = self.mapToScene(e.position().toPoint())
        if self._pan_from is not None:
            self._pan_from = None
            self._update_cursor()
            return
        t = self._active_tool()
        if t in ("brush", "eraser") and self._stroke_last is not None:
            self._stroke_last = None
            self.zones.end_stroke()
        elif t in ("rect", "circle") and self._drag_from is not None:
            a, self._drag_from = self._drag_from, None
            self.preview_path.setPath(QPainterPath())
            if t == "rect":
                r = QRectF(a, sp).normalized()
                if r.width() > 4 and r.height() > 4:
                    x0, y0 = self._frac(r.topLeft())
                    x1, y1 = self._frac(r.bottomRight())
                    self.zones.add_shape({"vocab": self.paint_vocab, "rect": [x0, y0, x1, y1]})
                    self.tool_used.emit("rect")
            else:
                rad = math.hypot(sp.x() - a.x(), sp.y() - a.y())
                if rad > 3:
                    cx, cy = self._frac(a)
                    self.zones.add_shape({"vocab": self.paint_vocab,
                                          "circle": [cx, cy, round(rad / self.scene_w, 4)]})
                    self.tool_used.emit("circle")

    def leaveEvent(self, e) -> None:
        self.cursor_ring.setVisible(False)
        super().leaveEvent(e)

    def enterEvent(self, e) -> None:
        self._update_cursor()
        super().enterEvent(e)

    def _paint_to(self, mp: QPointF) -> None:
        mw, _ = self.zones.size
        r = self.brush_mm / max(self.width_mm, 1) * mw / 2
        value = PAINT_DEFAULT if self._active_tool() == "eraser" else self.paint_vocab
        self.zones.paint_segment(self._stroke_last, mp, r, value)
        self._stroke_last = mp

    def _update_poly_preview(self, hover: Optional[QPointF] = None) -> None:
        path = QPainterPath()
        if self._poly:
            path.moveTo(self._poly[0])
            for q in self._poly[1:]:
                path.lineTo(q)
            if hover is not None:
                path.lineTo(hover)
            for q in self._poly:
                rr = 4 / max(self.transform().m11(), 1e-3)
                path.addEllipse(q, rr, rr)
        self.preview_path.setPath(path)

    def _finish_polygon(self) -> None:
        if len(self._poly) >= 3:
            pts = [self._frac(q) for q in self._poly]
            self.zones.add_shape({"vocab": self.paint_vocab, "polygon": pts})
            self.tool_used.emit("polygon")
        self._poly.clear()
        self.preview_path.setPath(QPainterPath())

    # ---------------------------------------------- drag & drop
    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.files_dropped.emit(paths)
