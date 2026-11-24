"""Opening card. It is its own small window: the cat assembles first, and only when the animation has
finished does the program's window appear. The page is loaded out of sight meanwhile, so nothing waits."""
from __future__ import annotations

import os
from typing import Callable

from PySide6.QtCore import QElapsedTimer, QPropertyAnimation, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImageReader, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import QWidget

SIZE, TEXT_H, MARGIN = 440, 86, 26
PAPER, INK, INK2 = QColor(226, 216, 198), QColor(30, 28, 25), QColor(93, 87, 78)


class Splash(QWidget):
    def __init__(self, path: str, on_done: Callable[[], None]):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.SplashScreen)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("Tessera")
        self.resize(SIZE + 2 * MARGIN, SIZE + TEXT_H + 2 * MARGIN)
        self._done, self._finished = on_done, False
        self.frames: list[QPixmap] = []
        self.ends: list[int] = []
        dpr = self.devicePixelRatioF()
        rd = QImageReader(path)
        t = 0
        while rd.canRead():
            im = rd.read()
            if im.isNull():
                break
            t += max(10, rd.nextImageDelay() or 33)
            px = QPixmap.fromImage(im.scaled(int(SIZE * dpr), int(SIZE * dpr), Qt.AspectRatioMode.IgnoreAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation))
            px.setDevicePixelRatio(dpr)
            self.frames.append(px)
            self.ends.append(t)
        self.i = 0
        self.clock = QElapsedTimer()
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        scr = self.screen().availableGeometry()
        self.move(scr.center() - self.rect().center())

    def play(self) -> None:
        if not self.frames:
            return self._end()
        self.clock.start()
        self.timer.start(8)

    def _tick(self) -> None:
        ms = self.clock.elapsed()
        while self.i < len(self.ends) - 1 and ms >= self.ends[self.i]:
            self.i += 1
        self.update()
        if ms >= self.ends[-1]:
            self._end()

    def _end(self) -> None:
        if not self._finished:
            self._finished = True
            self.timer.stop()
            self._done()

    def mousePressEvent(self, ev):  # noqa: N802 - a click skips the animation
        self._end()

    def keyPressEvent(self, ev):  # noqa: N802
        self._end()

    def fade_out(self) -> None:
        self._fx = QPropertyAnimation(self, b"windowOpacity")
        self._fx.setDuration(220)
        self._fx.setStartValue(1.0)
        self._fx.setEndValue(0.0)
        self._fx.finished.connect(self.close)
        self._fx.start()

    def paintEvent(self, ev):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        card = QRectF(MARGIN, MARGIN, SIZE, SIZE + TEXT_H)
        for k in range(MARGIN, 0, -2):           # soft shadow
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(30, 22, 14, 3))
            p.drawRoundedRect(card.adjusted(-k, -k + 6, k, k + 2), 14 + k, 14 + k)
        clip = QPainterPath()
        clip.addRoundedRect(card, 14, 14)
        p.setClipPath(clip)
        p.fillRect(card, PAPER)
        if self.frames:
            p.drawPixmap(MARGIN, MARGIN, self.frames[self.i])
        f = QFont("Bahnschrift", 17, QFont.Weight.DemiBold)
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 7)
        p.setFont(f)
        p.setPen(INK)
        p.drawText(QRectF(MARGIN + 4, MARGIN + SIZE + 2, SIZE, 38), Qt.AlignmentFlag.AlignCenter, "TESSERA")
        p.setFont(QFont("Segoe UI", 9))
        p.setPen(INK2)
        p.drawText(QRectF(MARGIN, MARGIN + SIZE + 38, SIZE, 24), Qt.AlignmentFlag.AlignCenter, "taş taş mozaik")


def start(path: str, make_window: Callable[[], QWidget], show: Callable[[QWidget], None],
          is_ready: Callable[[QWidget], bool], enabled: bool = True) -> dict:
    """Show the card, build the window out of sight, play, then reveal the window. Returns the state dict
    (keep it alive)."""
    st: dict = {"win": None, "splash": None, "played": not enabled}

    def reveal() -> None:
        w = st["win"]
        if w is None or not st["played"] or st.get("shown"):
            return
        if not is_ready(w) and st["tries"] < 60:      # at most 6 s; the last frame stays on screen
            st["tries"] += 1
            QTimer.singleShot(100, reveal)
            return
        st["shown"] = True
        show(w)
        if st["splash"] is not None:
            QTimer.singleShot(60, st["splash"].fade_out)

    def played() -> None:
        st["played"] = True
        reveal()

    st["tries"] = 0
    if enabled and os.path.exists(path):
        st["splash"] = sp = Splash(path, played)
        sp.show()
        sp.repaint()

        def play_when_loaded(n: int = 0) -> None:
            # loading blocks the event loop now and then; playing afterwards keeps the motion smooth
            if not is_ready(st["win"]) and n < 60:
                QTimer.singleShot(50, lambda: play_when_loaded(n + 1))
            else:
                QTimer.singleShot(120, sp.play)

        def build() -> None:
            st["win"] = make_window()
            play_when_loaded()
        QTimer.singleShot(30, build)
    else:
        st["played"] = True
        st["win"] = make_window()
        reveal()
    return st

