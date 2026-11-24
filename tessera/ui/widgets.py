"""Small custom widgets, painted by hand so they follow the theme exactly."""
from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QSizePolicy, QSlider, QVBoxLayout,
                               QWidget)

from . import icons
from .theme import body_font, current, head_font


def tr_num(v: float, decimals: int = 0) -> str:
    """Turkish number format: 1.234,5"""
    s = f"{v:,.{decimals}f}"
    return s.replace(",", "\0").replace(".", ",").replace("\0", ".")


# Bahnschrift's ascent is tight: labels set in it get a little top room so capital diacritics
# (İ Ö Ü Ç Ş) are not clipped. (A style-sheet padding would also shift the text sideways.)
_HEAD_LABELS = {"section", "stepEyebrow", "floatLabel", "stepTitle", "modeTitle", "value", "bigValue",
                "statValue", "emptyTitle", "brand"}


def label(text: str, obj: str = "", wrap: bool = False) -> QLabel:
    lb = QLabel(text)
    if obj:
        lb.setObjectName(obj)
        if obj in _HEAD_LABELS:
            lb.setContentsMargins(0, 3, 0, 0)
    lb.setWordWrap(wrap)
    return lb


def tr_upper(s: str) -> str:
    """Turkish-aware upper case (i -> İ, ı -> I)."""
    return s.replace("i", "İ").replace("ı", "I").upper()


def tr_lower(s: str) -> str:
    """Turkish-aware lower case (I -> ı, İ -> i)."""
    return s.replace("I", "ı").replace("İ", "i").lower()


def section(text: str) -> QLabel:
    return label(tr_upper(text), "section")


def hline() -> QFrame:
    f = QFrame()
    f.setFixedHeight(1)
    f.setStyleSheet(f"background: {current().line};")
    return f


# ====================================================================== step rail
class StepRail(QWidget):
    """Vertical list of the six steps. Constructivist number squares; the active step is red."""
    current_changed = Signal(int)

    ROW_H = 66

    def __init__(self, steps: list[tuple[str, str]], parent=None):
        super().__init__(parent)
        self.steps = steps               # (title, icon)
        self.subtitles = [""] * len(steps)
        self.done = [False] * len(steps)
        self.enabled = [True] * len(steps)
        self.index = 0
        self._hover = -1
        self.setMouseTracking(True)
        self.setMinimumWidth(208)
        self.setFixedHeight(self.ROW_H * len(steps) + 8)
        self.setCursor(Qt.PointingHandCursor)

    def set_current(self, i: int) -> None:
        if 0 <= i < len(self.steps) and self.enabled[i]:
            self.index = i
            self.update()

    def set_subtitle(self, i: int, text: str) -> None:
        self.subtitles[i] = text
        self.update()

    def set_done(self, i: int, done: bool) -> None:
        self.done[i] = done
        self.update()

    def set_enabled_steps(self, flags: list[bool]) -> None:
        self.enabled = flags
        self.update()

    def _row_at(self, y: float) -> int:
        i = int((y - 4) // self.ROW_H)
        return i if 0 <= i < len(self.steps) else -1

    def mouseMoveEvent(self, e) -> None:
        i = self._row_at(e.position().y())
        if i != self._hover:
            self._hover = i
            self.update()

    def leaveEvent(self, e) -> None:
        self._hover = -1
        self.update()

    def mousePressEvent(self, e) -> None:
        i = self._row_at(e.position().y())
        if i >= 0 and self.enabled[i] and i != self.index:
            self.index = i
            self.update()
            self.current_changed.emit(i)

    def paintEvent(self, _):
        t = current()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        for i, (title, ic) in enumerate(self.steps):
            y = 4 + i * self.ROW_H
            r = QRectF(10, y + 3, w - 20, self.ROW_H - 6)
            active = i == self.index
            en = self.enabled[i]
            if active:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(t.card))
                p.drawRoundedRect(r, 9, 9)
                p.setBrush(QColor(t.red))
                p.drawRoundedRect(QRectF(10, y + 15, 3, self.ROW_H - 30), 1.5, 1.5)
            elif i == self._hover and en:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(t.card))
                p.setOpacity(0.55)
                p.drawRoundedRect(r, 9, 9)
                p.setOpacity(1)
            # number square
            sq = QRectF(26, y + self.ROW_H / 2 - 15, 30, 30)
            if active:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(t.red))
                p.drawRect(sq)
                numc = QColor(t.on_red)
            elif self.done[i]:
                p.setPen(QPen(QColor(t.gold), 1.5))
                p.setBrush(Qt.NoBrush)
                p.drawRect(sq.adjusted(0.75, 0.75, -0.75, -0.75))
                numc = QColor(t.gold)
            else:
                p.setPen(QPen(QColor(t.line if en else t.card2), 1.5))
                p.setBrush(Qt.NoBrush)
                p.drawRect(sq.adjusted(0.75, 0.75, -0.75, -0.75))
                numc = QColor(t.muted if en else t.faint)
            if self.done[i] and not active:
                p.drawPixmap(int(sq.x() + 5), int(sq.y() + 5), icons.pixmap("check", t.gold, 20))
            else:
                p.setPen(numc)
                p.setFont(head_font(16, QFont.Bold))
                p.drawText(sq, Qt.AlignCenter, str(i + 1))
            # texts
            tx = 70
            p.setPen(QColor(t.text if en else t.faint))
            p.setFont(head_font(15, QFont.DemiBold if active else QFont.Normal))
            sub = self.subtitles[i]
            if sub:
                p.drawText(QRectF(tx, y + 13, w - tx - 14, 22), Qt.AlignLeft | Qt.AlignVCenter, title)
                p.setPen(QColor(t.muted if en else t.faint))
                p.setFont(body_font(11))
                fm = QFontMetrics(body_font(11))
                p.drawText(QRectF(tx, y + 35, w - tx - 14, 18), Qt.AlignLeft | Qt.AlignVCenter,
                           fm.elidedText(sub, Qt.ElideRight, int(w - tx - 16)))
            else:
                p.drawText(QRectF(tx, y, w - tx - 14, self.ROW_H), Qt.AlignLeft | Qt.AlignVCenter, title)
        p.end()


# ====================================================================== tabs
class TabStrip(QWidget):
    """Text tabs with a red underline. Keyboard: Alt+1..5 handled by the window."""
    current_changed = Signal(int)

    def __init__(self, items: list[str], parent=None):
        super().__init__(parent)
        self.items = items
        self.enabled = [True] * len(items)
        self.shown = [True] * len(items)
        self.index = 0
        self._hover = -1
        self.setMouseTracking(True)
        self.setFixedHeight(46)
        self.setCursor(Qt.PointingHandCursor)
        self._rects: list[QRectF] = []

    def sizeHint(self) -> QSize:
        return QSize(560, 46)

    def set_shown(self, i: int, on: bool) -> None:
        self.shown[i] = on
        self.updateGeometry()
        self.update()

    def set_current(self, i: int, emit: bool = True) -> None:
        if 0 <= i < len(self.items) and self.enabled[i] and self.shown[i] and i != self.index:
            self.index = i
            self.update()
            if emit:
                self.current_changed.emit(i)

    def set_enabled(self, i: int, on: bool) -> None:
        self.enabled[i] = on
        self.update()

    def _widths(self, compact: bool) -> list[float]:
        f = QFontMetrics(head_font(13 if compact else 14, QFont.DemiBold))
        pad = 14 if compact else 26
        return [(f.horizontalAdvance(it) + pad) if sh else 0.0 for it, sh in zip(self.items, self.shown)]

    def _compact(self) -> bool:
        return sum(self._widths(False)) + 2 * sum(self.shown) + 18 > self.width()

    def minimumSizeHint(self) -> QSize:
        return QSize(int(sum(self._widths(True)) + 2 * sum(self.shown) + 10), 46)

    def _layout(self) -> None:
        compact = self._compact()
        x = 6.0 if compact else 18.0
        self._rects = []
        for wd, sh in zip(self._widths(compact), self.shown):
            self._rects.append(QRectF(x, 0, wd, self.height()) if sh else QRectF())
            x += (wd + 2) if sh else 0

    def mouseMoveEvent(self, e) -> None:
        self._layout()
        h = next((i for i, r in enumerate(self._rects) if r.contains(e.position())), -1)
        if h != self._hover:
            self._hover = h
            self.update()

    def leaveEvent(self, e) -> None:
        self._hover = -1
        self.update()

    def mousePressEvent(self, e) -> None:
        self._layout()
        for i, r in enumerate(self._rects):
            if r.contains(e.position()):
                self.set_current(i)

    def paintEvent(self, _):
        t = current()
        self._layout()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        for i, (it, r) in enumerate(zip(self.items, self._rects)):
            if not self.shown[i]:
                continue
            active = i == self.index
            col = t.text if active else (t.muted if self.enabled[i] else t.faint)
            if i == self._hover and not active and self.enabled[i]:
                col = t.text
            p.setPen(QColor(col))
            compact = self._compact()
            p.setFont(head_font(13 if compact else 14, QFont.DemiBold if active else QFont.Normal))
            p.drawText(r, Qt.AlignCenter, it)
            if active:
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(t.red))
                inset = 6 if compact else 12
                p.drawRect(QRectF(r.x() + inset, r.bottom() - 3, r.width() - 2 * inset, 3))
        p.end()


# ====================================================================== light dial
class LightDial(QWidget):
    """Drag the sun around the artwork. Angle = light direction, distance = height of the sun
    (centre = straight above, rim = grazing light)."""
    changed = Signal(float, float)     # az, el (while dragging)
    released = Signal(float, float)

    EL_MIN, EL_MAX = 12.0, 80.0

    def __init__(self, size: int = 104, parent=None):
        super().__init__(parent)
        self.az, self.el = -130.0, 40.0
        self.setFixedSize(size, size)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Işık yönü: güneşi sürükle. Ortaya yaklaştıkça ışık tepeden gelir.")
        self._drag = False

    def set_light(self, az: float, el: float) -> None:
        self.az, self.el = az, el
        self.update()

    def _geom(self):
        c = QPointF(self.width() / 2, self.height() / 2)
        R = self.width() / 2 - 9
        return c, R

    def _sun_pos(self) -> QPointF:
        c, R = self._geom()
        f = (self.EL_MAX - self.el) / (self.EL_MAX - self.EL_MIN)
        f = min(1.0, max(0.0, f)) * 0.82 + 0.18
        a = math.radians(self.az)
        return QPointF(c.x() + math.cos(a) * R * f, c.y() + math.sin(a) * R * f)

    def _set_from(self, pos: QPointF) -> None:
        c, R = self._geom()
        dx, dy = pos.x() - c.x(), pos.y() - c.y()
        self.az = math.degrees(math.atan2(dy, dx))
        f = min(1.0, math.hypot(dx, dy) / R)
        f = max(0.0, (f - 0.18) / 0.82)
        self.el = round(self.EL_MAX - f * (self.EL_MAX - self.EL_MIN), 1)
        self.update()
        self.changed.emit(self.az, self.el)

    def mousePressEvent(self, e) -> None:
        self._drag = True
        self._set_from(e.position())

    def mouseMoveEvent(self, e) -> None:
        if self._drag:
            self._set_from(e.position())

    def mouseReleaseEvent(self, e) -> None:
        self._drag = False
        self.released.emit(self.az, self.el)

    def paintEvent(self, _):
        t = current()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c, R = self._geom()
        p.setPen(QPen(QColor(t.line), 1.2))
        p.setBrush(QColor(t.card))
        p.drawEllipse(c, R, R)
        p.setPen(QPen(QColor(t.line), 1, Qt.DotLine))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(c, R * 0.5, R * 0.5)
        # ticks
        p.setPen(QPen(QColor(t.faint), 1.2))
        for k in range(24):
            a = k * math.pi / 12
            r0 = R - (5 if k % 6 == 0 else 2.5)
            p.drawLine(QPointF(c.x() + math.cos(a) * r0, c.y() + math.sin(a) * r0),
                       QPointF(c.x() + math.cos(a) * R, c.y() + math.sin(a) * R))
        # the artwork: a little tile, shaded by the light
        s = R * 0.42
        tile = QRectF(c.x() - s / 2, c.y() - s * 0.7, s, s * 1.4)
        a = math.radians(self.az)
        g = QLinearGradient(QPointF(tile.center().x() + math.cos(a) * s, tile.center().y() + math.sin(a) * s),
                            QPointF(tile.center().x() - math.cos(a) * s, tile.center().y() - math.sin(a) * s))
        g.setColorAt(0, QColor(t.text))
        g.setColorAt(1, QColor(t.faint))
        p.setPen(Qt.NoPen)
        p.setBrush(g)
        p.drawRect(tile)
        # beam
        sp = self._sun_pos()
        p.setPen(QPen(QColor(t.gold), 1.2, Qt.DashLine))
        p.drawLine(sp, c)
        p.setPen(QPen(QColor(t.panel), 2))
        p.setBrush(QColor(t.gold))
        p.drawEllipse(sp, 7.5, 7.5)
        p.end()


# ====================================================================== slider row
class SliderRow(QWidget):
    value_changed = Signal(float)

    def __init__(self, title: str, lo: float, hi: float, step: float, value: float, unit: str = "",
                 decimals: int = 1, hint: str = "", parent=None):
        super().__init__(parent)
        self.lo, self.step, self.decimals, self.unit = lo, step, decimals, unit
        self.default = value
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        top = QHBoxLayout()
        top.setSpacing(6)
        self.title = QLabel(title)
        self.val = label("", "value")
        top.addWidget(self.title)
        top.addStretch(1)
        top.addWidget(self.val)
        v.addLayout(top)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, int(round((hi - lo) / step)))
        self.slider.setCursor(Qt.PointingHandCursor)
        v.addWidget(self.slider)
        if hint:
            h = label(hint, "hint", wrap=True)
            v.addWidget(h)
        self.slider.valueChanged.connect(self._changed)
        self.set_value(value, emit=False)

    def value(self) -> float:
        return round(self.lo + self.slider.value() * self.step, 4)

    def set_value(self, v: float, emit: bool = False) -> None:
        self.slider.blockSignals(not emit)
        self.slider.setValue(int(round((v - self.lo) / self.step)))
        self.slider.blockSignals(False)
        self._show()

    def _show(self) -> None:
        self.val.setText(f"{tr_num(self.value(), self.decimals)} {self.unit}".strip())

    def _changed(self, _):
        self._show()
        self.value_changed.emit(self.value())

    def mouseDoubleClickEvent(self, e) -> None:
        self.set_value(self.default, emit=True)


# ====================================================================== swatches
def paint_tile(p: QPainter, r: QRectF, rgb, radius: float = 6, relief: bool = True) -> None:
    """A filament colour drawn as a small pillowed mosaic tile."""
    base = QColor(*rgb)
    p.setPen(Qt.NoPen)
    if relief:
        g = QLinearGradient(r.topLeft(), r.bottomRight())
        g.setColorAt(0, base.lighter(118))
        g.setColorAt(0.5, base)
        g.setColorAt(1, base.darker(130))
        p.setBrush(g)
    else:
        p.setBrush(base)
    p.drawRoundedRect(r, radius, radius)
    p.setBrush(Qt.NoBrush)
    lum = 0.3 * rgb[0] + 0.59 * rgb[1] + 0.11 * rgb[2]
    p.setPen(QPen(QColor(0, 0, 0, 60) if lum > 150 else QColor(255, 255, 255, 30), 1))
    p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)


class FilamentChip(QWidget):
    """Library item: click = add/remove from palette, star = I own this."""
    toggled = Signal(object)        # Filament
    owned_toggled = Signal(object)

    W, H = 74, 92

    def __init__(self, fil, parent=None):
        super().__init__(parent)
        self.fil = fil
        self.in_palette = False
        self._hover = False
        self.setFixedSize(self.W, self.H)
        self.setCursor(Qt.PointingHandCursor)
        from .filaments import FINISH_TR
        self.setToolTip(f"{fil.name} · {FINISH_TR.get(fil.finish, fil.finish)} PLA · {fil.brand}\n"
                        "Tıkla: palete ekle/çıkar · Yıldız: bende var")

    def _star_rect(self) -> QRectF:
        return QRectF(self.W - 27, 3, 24, 24)

    def enterEvent(self, e) -> None:
        self._hover = True
        self.update()

    def leaveEvent(self, e) -> None:
        self._hover = False
        self.update()

    def mousePressEvent(self, e) -> None:
        if self._star_rect().contains(e.position()):
            self.fil.owned = not self.fil.owned
            self.owned_toggled.emit(self.fil)
        else:
            self.toggled.emit(self.fil)
        self.update()

    def paintEvent(self, _):
        t = current()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        sw = QRectF(7, 7, self.W - 14, self.W - 14)
        if self.in_palette:
            p.setPen(QPen(QColor(t.red), 2.2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(sw.adjusted(-4, -4, 4, 4), 10, 10)
        elif self._hover:
            p.setPen(QPen(QColor(t.faint), 1.2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(sw.adjusted(-4, -4, 4, 4), 10, 10)
        paint_tile(p, sw, self.fil.rgb, 7)
        if self.fil.finish == "silk":
            p.setPen(Qt.NoPen)
            g = QLinearGradient(sw.topLeft(), sw.bottomRight())
            g.setColorAt(0.35, QColor(255, 255, 255, 0))
            g.setColorAt(0.5, QColor(255, 255, 255, 90))
            g.setColorAt(0.65, QColor(255, 255, 255, 0))
            p.setBrush(g)
            p.drawRoundedRect(sw, 7, 7)
        lum = 0.3 * self.fil.rgb[0] + 0.59 * self.fil.rgb[1] + 0.11 * self.fil.rgb[2]
        if self.fil.owned or self._hover:
            col = "#2A2520" if lum > 140 else "#F7EEDD"
            name = "starfill" if self.fil.owned else "star"
            sr = self._star_rect()
            p.drawPixmap(int(sr.x() + 2), int(sr.y() + 2), icons.pixmap(name, col if not self.fil.owned else t.gold, 18))
            if self.fil.owned:
                p.drawPixmap(int(sr.x() + 2), int(sr.y() + 2), icons.pixmap("star", col, 18))
        if self.in_palette:
            c = QPointF(sw.x() + 13, sw.bottom() - 13)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(t.red))
            p.drawEllipse(c, 9, 9)
            p.drawPixmap(int(c.x() - 7), int(c.y() - 7), icons.pixmap("check", t.on_red, 14))
        p.setPen(QColor(t.text if self.in_palette else t.muted))
        p.setFont(body_font(11, QFont.DemiBold if self.in_palette else QFont.Normal))
        fm = QFontMetrics(p.font())
        p.drawText(QRectF(0, self.W - 2, self.W, 20), Qt.AlignHCenter | Qt.AlignVCenter,
                   fm.elidedText(self.fil.name, Qt.ElideRight, self.W - 2))
        p.end()


class PaletteStrip(QWidget):
    """The chosen palette as one row of big tiles (like a smalt sample board)."""
    remove_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.palette = []
        self.roles: dict[str, str] = {}
        self.setMinimumHeight(96)
        self.setMouseTracking(True)
        self._hover = -1

    def set_palette(self, pal, roles: dict | None = None) -> None:
        self.palette = list(pal)
        self.roles = dict(roles or {})
        self.update()

    def _rects(self) -> list[QRectF]:
        n = max(1, len(self.palette))
        gap = 6
        w = (self.width() - gap * (n - 1)) / n
        w = min(w, 76)
        return [QRectF(i * (w + gap), 0, w, 62) for i in range(len(self.palette))]

    def mouseMoveEvent(self, e) -> None:
        h = next((i for i, r in enumerate(self._rects()) if r.contains(e.position())), -1)
        if h != self._hover:
            self._hover = h
            self.setCursor(Qt.PointingHandCursor if h >= 0 else Qt.ArrowCursor)
            self.update()

    def leaveEvent(self, e) -> None:
        self._hover = -1
        self.update()

    def mousePressEvent(self, e) -> None:
        for i, r in enumerate(self._rects()):
            if r.contains(e.position()):
                self.remove_requested.emit(i)

    def paintEvent(self, _):
        t = current()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if not self.palette:
            p.setPen(QPen(QColor(t.line), 1.2, Qt.DashLine))
            p.drawRoundedRect(QRectF(0.5, 0.5, self.width() - 1, 62), 8, 8)
            p.setPen(QColor(t.faint))
            p.drawText(QRectF(0, 0, self.width(), 62), Qt.AlignCenter, "Aşağıdan en az 2 renk seç")
        for i, (f, r) in enumerate(zip(self.palette, self._rects())):
            paint_tile(p, r, f.rgb, 8)
            if i == self._hover:
                lum = 0.3 * f.rgb[0] + 0.59 * f.rgb[1] + 0.11 * f.rgb[2]
                col = "#2A2520" if lum > 140 else "#F7EEDD"
                p.drawPixmap(int(r.center().x() - 9), int(r.center().y() - 9), icons.pixmap("close", col, 18))
            p.setPen(QColor(t.text))
            p.setFont(body_font(11))
            fm = QFontMetrics(p.font())
            p.drawText(QRectF(r.x() - 4, 66, r.width() + 8, 16), Qt.AlignHCenter,
                       fm.elidedText(f.name, Qt.ElideRight, int(r.width() + 8)))
            role = self.roles.get(f.name)
            p.setPen(QColor(t.gold if role else t.faint))
            p.setFont(body_font(10))
            p.drawText(QRectF(r.x() - 4, 81, r.width() + 8, 14), Qt.AlignHCenter, role or f.hex)
        p.end()


# ====================================================================== misc
class Spinner(QWidget):
    def __init__(self, size: int = 16, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._a = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def start(self) -> None:
        self._timer.start(40)
        self.show()

    def stop(self) -> None:
        self._timer.stop()
        self.hide()

    def _tick(self) -> None:
        self._a = (self._a + 24) % 360
        self.update()

    def paintEvent(self, _):
        t = current()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(2, 2, self.width() - 4, self.height() - 4)
        p.setPen(QPen(QColor(t.line), 2))
        p.drawEllipse(r)
        p.setPen(QPen(QColor(t.gold), 2, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(r, -self._a * 16, 100 * 16)
        p.end()


class Dot(QWidget):
    def __init__(self, colour: str, size: int = 12, square: bool = True, parent=None):
        super().__init__(parent)
        self.colour = colour
        self.square = square
        self.setFixedSize(size, size)

    def set_colour(self, c: str) -> None:
        self.colour = c
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self.colour))
        if self.square:
            p.drawRoundedRect(QRectF(0, 0, self.width(), self.height()), 2.5, 2.5)
        else:
            p.drawEllipse(QRectF(0, 0, self.width(), self.height()))
        p.end()


class ClickFrame(QFrame):
    clicked = Signal()

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(e)


class StatItem(QWidget):
    def __init__(self, caption: str, icon_name: str = "", parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(9)
        self.icon = QLabel()
        self.icon_name = icon_name
        if icon_name:
            h.addWidget(self.icon)
        v = QVBoxLayout()
        v.setSpacing(0)
        self.value = label("—", "statValue")
        self.caption = label(caption, "statLabel")
        v.addWidget(self.value)
        v.addWidget(self.caption)
        h.addLayout(v)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    def retheme(self) -> None:
        if self.icon_name:
            self.icon.setPixmap(icons.pixmap(self.icon_name, current().faint, 20))

    def set(self, value: str, tip: str = "") -> None:
        self.value.setText(value)
        self.setToolTip(tip)
