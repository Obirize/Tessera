"""Opening card: the cat is laid stone by stone, eyes first, seen straight from the front.

It runs in its OWN process (``python -m tessera.acilis``), so loading the program can never make it
stutter, and it draws the stones live at the screen's refresh rate. The program's window appears only
after the card says it has finished.

    program -> card   stdin   "kapan" (or end of input): fade away and quit
    card -> program   stdout  "bitti": the cat is complete (or the user clicked to skip)
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "web", "acilis")

SIZE, TEXT_H, MARGIN = 440, 86, 30        # card, px
# timeline, ms
FADE_IN = 200
EYES_AT, EYE_GAP, EYE_FLIGHT = 240, 80, 480
REST_AT, GAP_FIRST, GAP_LAST, FLIGHT = 900, 54, 30, 380
SETTLE, FINISH, HOLD = 140, 180, 260
RANK = {"pupil": 0, "eye": 0, "nose": 1, "face": 2, "ear_in": 3, "ear": 4, "ruff": 5, "bg": 6}


def schedule(stones: list) -> list:
    """Start time and flight of every stone. Eyes together; then one stone after another, always
    alternating left and right, so the face grows evenly from the middle."""
    for s in stones:
        xs, ys = [p[0] for p in s["pts"]], [p[1] for p in s["pts"]]
        s["cx"], s["cy"] = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        s["rank"] = RANK.get(s["region"], 5)
    eyes = sorted((s for s in stones if s["rank"] == 0), key=lambda s: -abs(s["cx"] - 0.5))
    for i, s in enumerate(eyes):                  # outer halves of both eyes, then the inner ones
        s["t0"], s["dur"] = EYES_AT + (i // 2) * EYE_GAP, EYE_FLIGHT
    rest = []
    for rank in sorted({s["rank"] for s in stones} - {0}):
        grp = [s for s in stones if s["rank"] == rank]
        sign = -1 if rank in (3, 4, 6) else 1     # ears and ground from the bottom up, the rest downwards
        left = sorted((s for s in grp if s["cx"] < 0.495), key=lambda s: sign * s["cy"])
        right = sorted((s for s in grp if s["cx"] > 0.505), key=lambda s: sign * s["cy"])
        mid = [s for s in grp if 0.495 <= s["cx"] <= 0.505]
        if rank != 5:
            rest += mid
        for a in range(max(len(left), len(right))):
            rest += left[a:a + 1] + right[a:a + 1]
        if rank == 5:                             # the collar closes at the bottom middle
            rest += mid
    t = float(REST_AT)
    for i, s in enumerate(rest):
        s["t0"], s["dur"] = t, FLIGHT
        t += GAP_FIRST + (GAP_LAST - GAP_FIRST) * i / max(len(rest) - 1, 1)
    return eyes + rest


def total_ms(order: list) -> float:
    return max(s["t0"] + s["dur"] for s in order) + FINISH + HOLD


# ------------------------------------------------------------------ the program's side
class Handle:
    """What the program holds: is the card done, and tell it to go."""

    def __init__(self):
        self.done = threading.Event()
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.p = subprocess.Popen([sys.executable, "-m", "tessera.acilis"], cwd=os.path.dirname(HERE),
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  creationflags=flags)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        try:
            for line in self.p.stdout:
                if line.strip() == b"bitti":
                    break
        except OSError:
            pass
        self.done.set()                           # finished, skipped or gone: never keep the program waiting

    def close(self):
        try:
            self.p.stdin.write(b"kapan\n")
            self.p.stdin.close()
        except (OSError, ValueError):
            pass


def spawn():
    try:
        return Handle() if os.path.exists(os.path.join(DIR, "taslar.json")) else None
    except OSError:
        return None


# ------------------------------------------------------------------ the card
def main() -> int:
    from PySide6.QtCore import QElapsedTimer, QObject, QPointF, QRectF, Qt, QTimer, Signal
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPixmap, QPolygonF
    from PySide6.QtWidgets import QApplication, QWidget

    PAPER, INK, INK2 = QColor(228, 219, 203), QColor(30, 28, 25), QColor(93, 87, 78)
    app = QApplication(["tessera"])
    with open(os.path.join(DIR, "taslar.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    order = schedule(d["stones"])
    END = total_ms(order)
    FMT = QImage.Format.Format_ARGB32_Premultiplied
    HINT = QPainter.RenderHint
    SMOOTH, STRETCH = Qt.TransformationMode.SmoothTransformation, Qt.AspectRatioMode.IgnoreAspectRatio

    def blurred(path, w: int, h: int, pad: int, r: int, alpha: int):
        """Soft shadow of a path given in w x h px."""
        im = QImage(w + 2 * pad, h + 2 * pad, FMT)
        im.fill(0)
        p = QPainter(im)
        p.setRenderHint(HINT.Antialiasing)
        p.translate(pad, pad)
        p.fillPath(path, QColor(12, 8, 4, alpha))
        p.end()
        for _ in range(2):
            small = im.scaled(max(1, im.width() // r), max(1, im.height() // r), STRETCH, SMOOTH)
            im = small.scaled(im.width(), im.height(), STRETCH, SMOOTH)
        return im

    class Card(QWidget):
        def __init__(self):
            super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
                             | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowDoesNotAcceptFocus)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
            self.resize(SIZE + 2 * MARGIN, SIZE + TEXT_H + 2 * MARGIN)
            self.move(self.screen().availableGeometry().center() - self.rect().center())
            self.k = k = self.devicePixelRatioF()
            self.N = N = int(round(SIZE * k))

            def load(name):
                im = QImage(os.path.join(DIR, name)).convertToFormat(FMT)
                return im.scaled(N, N, STRETCH, SMOOTH)

            def path_of(pts, ox=0.0, oy=0.0):
                q = QPainterPath()
                q.addPolygon(QPolygonF([QPointF(x * N - ox, y * N - oy) for x, y in pts]))
                q.closeSubpath()
                return q

            full = load("tam.webp")
            plate = load("levha.webp")
            # background: card, its shadow, the plate and its shadow (drawn once)
            bg = QImage(int(self.width() * k), int(self.height() * k), FMT)
            bg.fill(0)
            card = QPainterPath()
            card.addRoundedRect(QRectF(0, 0, SIZE * k, (SIZE + TEXT_H) * k), 14 * k, 14 * k)
            m = int(MARGIN * k)
            p = QPainter(bg)
            p.setRenderHint(HINT.Antialiasing)
            p.drawImage(0, int(6 * k), blurred(card, int(SIZE * k), int((SIZE + TEXT_H) * k), m, int(9 * k), 95))
            p.translate(m, m)
            p.fillPath(card, PAPER)
            p.drawImage(int(5 * k), int(8 * k), blurred(path_of(d["plate"]), N, N, 0, int(7 * k), 120))
            p.drawImage(0, 0, plate)
            p.end()
            self.bg = QPixmap.fromImage(bg)
            self.full = QPixmap.fromImage(full)
            # board: finished stones are baked here, so a frame only draws the ones in the air
            self.board = QPixmap(N, N)
            self.board.fill(Qt.GlobalColor.transparent)
            for s in order:
                r = path_of(s["pts"]).boundingRect().adjusted(-1, -1, 1, 1).toAlignedRect()
                s["r"] = QRectF(r)
                im = QImage(r.size(), FMT)
                im.fill(0)
                p = QPainter(im)
                p.setRenderHint(HINT.Antialiasing)
                p.fillPath(path_of(s["pts"], r.x(), r.y()), Qt.GlobalColor.white)
                p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
                p.drawImage(-r.x(), -r.y(), full)
                p.end()
                s["px"] = QPixmap.fromImage(im)
                s["pad"] = pad = int(14 * k)
                s["sh"] = QPixmap.fromImage(blurred(path_of(s["pts"], r.x(), r.y()), r.width(), r.height(), pad,
                                                    int(5 * k), 150))
                s["spin"] = (7.0 if s["cx"] < 0.5 else -7.0) * (0.6 if s["rank"] == 0 else 1.0)
                s["baked"] = False
            self.clock = QElapsedTimer()
            self.said = self.leaving = False
            self.left_at = 0.0
            hz = max(50.0, min(240.0, self.screen().refreshRate() or 60.0))
            self.timer = QTimer(self)
            self.timer.setTimerType(Qt.TimerType.PreciseTimer)
            self.timer.timeout.connect(self.tick)
            self.timer.start(max(4, int(1000 / hz)))
            self.setWindowOpacity(0.0)

        def now(self) -> float:
            return self.clock.nsecsElapsed() / 1e6 if self.clock.isValid() else 0.0

        def tick(self):
            if not self.clock.isValid():
                return
            t = self.now()
            if self.leaving:
                u = min(1.0, (t - self.left_at) / 240)
                self.setWindowOpacity(1 - u * u * (3 - 2 * u))
                if u >= 1:
                    app.quit()
                return
            if t < FADE_IN + 60:
                self.setWindowOpacity(min(1.0, t / FADE_IN))
            if t >= END:
                self.say()
                if t > END + 200:
                    return                        # nothing moves any more
            self.update()

        def say(self):
            if not self.said:
                self.said = True
                try:
                    sys.stdout.write("bitti\n")
                    sys.stdout.flush()
                except (OSError, ValueError, AttributeError):
                    pass

        def leave(self):
            if not self.leaving:
                self.leaving, self.left_at = True, self.now()

        def mousePressEvent(self, ev):  # noqa: N802 - a click skips
            self.say()

        def bake(self, s):
            k = self.k
            q = QPainter(self.board)
            q.setOpacity(0.5)
            q.drawPixmap(QPointF(s["r"].x() - s["pad"] + 1.2 * k, s["r"].y() - s["pad"] + 1.6 * k), s["sh"])
            q.setOpacity(1)
            q.drawPixmap(s["r"].topLeft(), s["px"])
            q.end()
            s["baked"] = True

        def paintEvent(self, ev):  # noqa: N802
            k, t = self.k, self.now()
            p = QPainter(self)
            p.setRenderHint(HINT.SmoothPixmapTransform)
            p.scale(1 / k, 1 / k)
            p.drawPixmap(0, 0, self.bg)
            p.translate(int(MARGIN * k), int(MARGIN * k))
            air = []
            for s in order:
                if s["baked"] or t < s["t0"]:
                    continue
                if t > s["t0"] + s["dur"] + SETTLE:
                    self.bake(s)
                else:
                    air.append((s, (t - s["t0"]) / s["dur"]))
            p.drawPixmap(0, 0, self.board)
            air.sort(key=lambda a: -a[1])         # the newest stone is the one nearest to us
            for s, u in air:
                c = s["r"].center()
                if u < 1:
                    h = (1 - u) ** 3              # height above the plate, 1 -> 0
                    sc = 1 + 0.30 * h
                else:                             # pressed into place: a tiny dip that springs back
                    h = 0.0
                    v = (t - s["t0"] - s["dur"]) / SETTLE
                    sc = 1 - 0.02 * math.sin(math.pi * min(1.0, v))
                dx, dy = (c.x() - self.N / 2) * 0.16 * h, (c.y() - self.N / 2) * 0.16 * h
                alpha = min(1.0, max(0.0, (t - s["t0"]) / 55))
                for what in (0, 1):
                    off = (1.2 + 13 * h) * k if what == 0 else 0.0
                    p.save()
                    p.translate(c.x() + dx + off, c.y() + dy + off * 1.35)
                    p.rotate(s["spin"] * h)
                    p.scale(sc, sc)
                    if what == 0:
                        p.setOpacity(alpha * (0.5 - 0.28 * h))
                        p.drawPixmap(QPointF(-s["r"].width() / 2 - s["pad"], -s["r"].height() / 2 - s["pad"]), s["sh"])
                    else:
                        p.setOpacity(alpha)
                        p.drawPixmap(QPointF(-s["r"].width() / 2, -s["r"].height() / 2), s["px"])
                    p.restore()
            land = END - HOLD - FINISH            # the real contact shadows settle in once everything lies
            if t > land:
                p.setOpacity(min(1.0, (t - land) / FINISH))
                p.drawPixmap(0, 0, self.full)
            p.setOpacity(min(1.0, max(0.0, (t - 500) / 500)))
            f = QFont("Bahnschrift")
            f.setPixelSize(int(22 * k))
            f.setWeight(QFont.Weight.DemiBold)
            f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 7 * k)
            p.setFont(f)
            p.setPen(INK)
            mid = Qt.AlignmentFlag.AlignCenter
            p.drawText(QRectF(3.5 * k, (SIZE - 2) * k, SIZE * k, 38 * k), mid, "TESSERA")
            g = QFont("Segoe UI")
            g.setPixelSize(int(13 * k))
            p.setFont(g)
            p.setPen(INK2)
            p.drawText(QRectF(0, (SIZE + 34) * k, SIZE * k, 24 * k), mid, "taş taş mozaik")
            p.end()
            hook = globals().get("_frame_hook")   # tests measure the real pacing here
            if hook:
                hook(self, t)

    class Wire(QObject):
        go = Signal()

    wire = Wire()
    card = Card()
    wire.go.connect(card.leave)

    def listen():
        try:
            sys.stdin.readline()                  # "kapan", or end of input when the program is gone
        except (OSError, ValueError, AttributeError):
            pass
        wire.go.emit()

    if sys.stdin is not None:
        threading.Thread(target=listen, daemon=True).start()
    card.show()
    app.processEvents()
    card.clock.start()
    QTimer.singleShot(30000, app.quit)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
