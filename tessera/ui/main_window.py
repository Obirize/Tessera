"""Tessera main window: header, step rail, preview stage, step panels, stats bar."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import QPointF, QRectF, QSettings, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import (QAction, QColor, QDesktopServices, QFont, QIcon, QImage, QKeySequence,
                           QPainter, QPen, QShortcut)
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QMainWindow, QMenu, QMessageBox, QPushButton,
                               QSizePolicy, QStackedWidget, QToolButton, QVBoxLayout, QWidget)

from ..model import VOCAB_NAMES_TR, Filament, Mode, Settings, Vocab
from ..project import PROJECT_SUFFIX, Project
from . import icons, theme
from .canvas import Canvas
from .engine import (PREVIEW_MM_PER_PX, ExportJob, friendly_error, preview_mm_per_px, PreviewEngine, PreviewRequest, TaskThread,
                     export_payload, run_export)
from .filaments import POSTER_PALETTE, find, load_library, nearest
from .views import quick_kw
from .hooks import run_suggest_palette, run_suggest_zones, suggest_palette_fn, suggest_zones_fn
from .steps import (VOCAB_ORDER, ExportStep, FilamentStep, ImageStep, ProductionStep, ReliefStep,
                    ZonesStep, light_text)
from .theme import current, head_font
from .widgets import LightDial, Spinner, StatItem, StepRail, TabStrip, label, tr_num, tr_upper
from .zones import ZoneDoc, gray_to_qimage, rgb_to_qimage

REPO = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO / "ornekler" / "poster"
IMAGE_FILTER = "Görseller (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)"
PROJECT_FILTER = "Tessera projesi (*.tessera.json)"
DISPLAY_MAX = 1800   # longest side of the on-screen original

STEPS = [("Görsel", "image"), ("Filamentler", "palette"), ("Bölgeler", "zones"),
         ("Kabartma", "relief"), ("Üretim", "print"), ("Dışa aktar", "export")]
TABS = ["Orijinal", "Mozaik", "Açılı görünüm", "Bölgeler", "Parça haritası", "Tablalar", "Panel 3D"]
TAB_ORIG, TAB_MOSAIC, TAB_OBLIQUE, TAB_ZONES, TAB_FLAT, TAB_PLATES, TAB_3D = range(7)


def _project_image_exists(project_path: str) -> bool:
    try:
        p = Project.load(project_path)
        return bool(p.image_path) and os.path.exists(p.image_path)
    except Exception:  # noqa: BLE001
        return False


# ====================================================================== hero art (empty state)
class HeroArt(QWidget):
    """A small constructivist mosaic: a red sun with gold/cream ray tiles and charcoal blocks."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(540, 230)

    def paintEvent(self, _):
        t = current()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy = 200.0, 214.0
        ring_cols = [t.gold, t.text, t.gold, t.muted]
        for k, r in enumerate((78, 108, 138, 168)):
            n = 9 + k * 4
            for i in range(n):
                a0 = 180 + i * 180 / n + 1.2
                span = 180 / n - 2.4
                if a0 + span > 360:
                    continue
                col = QColor(ring_cols[(i + k) % len(ring_cols)])
                col.setAlphaF(0.95 if (i + k) % 3 else 0.55)
                p.setPen(QPen(col, 19 - k * 1.5, Qt.SolidLine, Qt.FlatCap))
                rr = QRectF(cx - r, cy - r, 2 * r, 2 * r)
                p.drawArc(rr, int(-a0 * 16), int(-span * 16))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(t.red))
        p.drawPie(QRectF(cx - 60, cy - 60, 120, 120), 0, 180 * 16)
        # blocks (a city skyline in pixel blocks)
        p.setBrush(QColor(t.faint))
        for i, h in enumerate((60, 92, 48, 120, 76, 100)):
            x = 388 + i * 24
            for j in range(int(h // 22)):
                p.drawRoundedRect(QRectF(x, cy - 20 - j * 22, 20, 18), 2, 2)
        # a constructivist diagonal
        p.setBrush(QColor(t.red))
        p.translate(450, 46)
        p.rotate(-24)
        p.drawRect(QRectF(-70, -5, 140, 10))
        p.end()


class EmptyState(QWidget):
    def __init__(self, win: "MainWindow"):
        super().__init__()
        self.win = win
        self.setObjectName("stage")
        self.setAutoFillBackground(True)
        self.setAcceptDrops(True)
        v = QVBoxLayout(self)
        v.addStretch(2)
        col = QVBoxLayout()
        col.setSpacing(14)
        col.setAlignment(Qt.AlignHCenter)
        self.hero = HeroArt()
        col.addWidget(self.hero, 0, Qt.AlignHCenter)
        col.addSpacing(10)
        t = label("Görselden kabartma mozaiğe", "emptyTitle")
        t.setAlignment(Qt.AlignCenter)
        self.title = t
        col.addWidget(t)
        ld = label("Bir görseli buraya sürükle ya da seç. Tessera onu tek renk parçalara böler, "
                   "kabartmasını hesaplar ve Bambu Studio'ya hazır tablalar çıkarır.", "emptyLead", wrap=True)
        ld.setAlignment(Qt.AlignCenter)
        ld.setFixedWidth(520)
        ld.setMinimumHeight(48)
        self.lead = ld
        self._lead_text = ld.text()
        col.addWidget(ld, 0, Qt.AlignHCenter)
        col.addSpacing(8)
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addStretch(1)
        self.pick = QPushButton("Görsel seç…")
        self.pick.setObjectName("primary")
        self.pick.setCursor(Qt.PointingHandCursor)
        self.pick.clicked.connect(win.choose_image)
        row.addWidget(self.pick)
        self.example = QPushButton("  Örnek proje: Sovyet afişi")
        self.example.setCursor(Qt.PointingHandCursor)
        self.example.setMinimumHeight(40)
        self.example.clicked.connect(win.load_example)
        row.addWidget(self.example)
        row.addStretch(1)
        col.addLayout(row)
        col.addSpacing(18)
        self.recent_title = label("SON PROJELER", "section")
        self.recent_title.setAlignment(Qt.AlignCenter)
        col.addWidget(self.recent_title)
        self.recent_box = QVBoxLayout()
        self.recent_box.setSpacing(0)
        col.addLayout(self.recent_box)
        v.addLayout(col)
        v.addStretch(3)

    def set_missing(self, path: str) -> None:
        """K-3: the project's image is missing. Offer to relink instead of starting over."""
        if path:
            self.title.setText("Görsel bulunamadı")
            self.lead.setText(f"Bu proje şu görseli kullanıyor:\n{os.path.normpath(path)}\n"
                              "Bölgeler, renkler ve ayarlar duruyor; görselin yeni yerini göstermen yeterli.")
            self.lead.setMinimumHeight(72)
            self.pick.setText("Görseli bul…")
        else:
            self.title.setText("Görselden kabartma mozaiğe")
            self.lead.setText(self._lead_text)
            self.lead.setMinimumHeight(48)
            self.pick.setText("Görsel seç…")
        self.example.setVisible(not path)
        self.hero.setVisible(not path)

    def retheme(self) -> None:
        self.example.setIcon(icons.icon("magic", current().gold, 18))
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(current().stage))
        self.setPalette(pal)
        self.hero.update()

    def refresh_recent(self, files: list[str]) -> None:
        while self.recent_box.count():
            it = self.recent_box.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        files = [f for f in files if os.path.exists(f)][:4]
        self.recent_title.setVisible(bool(files))
        for f in files:
            b = QPushButton(os.path.basename(f).replace(PROJECT_SUFFIX, ""))
            b.setObjectName("link")
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(f)
            if not _project_image_exists(f):          # G-5: image missing -> dimmed
                b.setStyleSheet(f"color: {current().faint};")
                b.setText(b.text() + "  (görsel bulunamıyor)")
            b.clicked.connect(lambda _=False, f=f: self.win.open_project(f))
            self.recent_box.addWidget(b, 0, Qt.AlignHCenter)

    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.win.open_any(paths[0])


# ====================================================================== stage
class Stage(QWidget):
    """Canvas plus floating boxes (light dial, busy chip, error banner, zone legend)."""

    def __init__(self, win: "MainWindow", canvas: Canvas):
        super().__init__()
        self.win = win
        self.canvas = canvas
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(canvas)

        self.light_box = QFrame(self)
        self.light_box.setObjectName("floatBox")
        lv = QVBoxLayout(self.light_box)
        lv.setContentsMargins(12, 10, 12, 10)
        lv.setSpacing(6)
        lv.addWidget(label("IŞIK YÖNÜ", "floatLabel"), 0, Qt.AlignHCenter)
        self.dial = LightDial(96)
        lv.addWidget(self.dial, 0, Qt.AlignHCenter)

        self.busy = QFrame(self)
        self.busy.setObjectName("floatBox")
        bh = QHBoxLayout(self.busy)
        bh.setContentsMargins(12, 8, 14, 8)
        bh.setSpacing(8)
        self.spinner = Spinner(16)
        bh.addWidget(self.spinner)
        self.busy_lbl = label("Mozaik hesaplanıyor…")
        bh.addWidget(self.busy_lbl)
        self.busy.hide()

        self.error = QFrame(self)
        self.error.setObjectName("floatBox")
        eh = QHBoxLayout(self.error)
        eh.setContentsMargins(12, 10, 10, 10)
        eh.setSpacing(10)
        self.err_icon = QLabel()
        eh.addWidget(self.err_icon)
        self.err_lbl = label("", "", wrap=True)
        self.err_lbl.setMinimumWidth(240)
        self.err_lbl.setMaximumWidth(460)
        eh.addWidget(self.err_lbl, 1)
        self.err_btn = QPushButton("Tekrar dene")
        self.err_btn.clicked.connect(win.error_action)
        eh.addWidget(self.err_btn)
        self.error.hide()

        self.legend = QFrame(self)
        self.legend.setObjectName("floatBox")
        lg = QVBoxLayout(self.legend)
        lg.setContentsMargins(12, 10, 14, 10)
        lg.setSpacing(5)
        lg.addWidget(label("TAŞ TİPLERİ", "floatLabel"))
        from .widgets import Dot
        from .theme import VOCAB_COLOURS
        self.legend_rows = {}
        for v in VOCAB_ORDER:
            h = QHBoxLayout()
            h.setContentsMargins(0, 1, 0, 1)
            h.setSpacing(7)
            h.addWidget(Dot(VOCAB_COLOURS[v], 10))
            lb = label(VOCAB_NAMES_TR[Vocab(v)], "hint")
            h.addWidget(lb)
            h.addStretch(1)
            w = QWidget()
            w.setLayout(h)
            lg.addWidget(w)
            self.legend_rows[v] = w
        self.legend.hide()

    def retheme(self) -> None:
        self.err_icon.setPixmap(icons.pixmap("warn", current().red, 22))
        self.dial.update()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self.place()

    def place(self) -> None:
        m = 16
        for w in (self.light_box, self.busy, self.error, self.legend):
            w.adjustSize()
        # Ö-10: the fitted artwork leaves room for the floating boxes
        self.canvas.set_reserved(self.legend.width() + m if self.legend.isVisible() else 0,
                                 self.light_box.width() + m if self.light_box.isVisible() else 0)
        self.light_box.move(self.width() - self.light_box.width() - m, self.height() - self.light_box.height() - m)
        self.busy.move(m, m)
        self.error.move((self.width() - self.error.width()) // 2, m)
        self.legend.move(m, self.height() - self.legend.height() - m)
        for w in (self.light_box, self.busy, self.error, self.legend):
            w.raise_()


# ====================================================================== main window
class MainWindow(QMainWindow):
    def __init__(self, store: Optional[QSettings] = None):
        super().__init__()
        self.store = store or QSettings("Tessera", "Tessera")
        self.setWindowTitle("Tessera")
        self.resize(1480, 920)
        self.setMinimumSize(1120, 720)
        self.setAcceptDrops(True)

        self.advanced_on = self.store.value("advanced", False, type=bool)
        self.theme = theme.LIGHT if self.store.value("theme", "dark") == "light" else theme.DARK
        theme.apply(QApplication.instance(), self.theme)

        self.project = Project()
        self.library, self.library_from_file = load_library()
        owned = set(self.store.value("owned", [], type=list) or [])
        for f in self.library:
            f.owned = f.name in owned
        self.zones = ZoneDoc(self)
        self.original_qimage: Optional[QImage] = None
        self.zones_qimage: Optional[QImage] = None
        self.image_px = (0, 0)
        self.last_result = None
        self.last_estimate: Optional[dict] = None
        self.dirty = False
        self.tab = TAB_ORIG
        self._tab_before_zones = TAB_MOSAIC
        self._geom_kind = "flat"
        self._error = ""
        self._export_task: Optional[TaskThread] = None
        self._suggest_task: Optional[TaskThread] = None
        self.last_export: Optional[dict] = None
        self.palette_roles: dict[str, str] = {}
        self.image_missing = False
        self._missing_aspect = 1.0
        self.image_note = ""
        self.export_in_process = True       # tests switch this off to monkeypatch export_project
        self.exact: Optional[dict] = None     # R3-4: exact part count for self.exact_gen
        self.exact_gen = -1
        self._count_job = None
        self._count_timer = QTimer(self)
        self._count_timer.setSingleShot(True)
        self._count_timer.timeout.connect(self._start_exact_count)

        self.engine = PreviewEngine(self)
        self.engine.ready.connect(self._on_preview)
        self.engine.failed.connect(self._on_preview_failed)
        self.engine.busy_changed.connect(self._on_busy)
        self.gen = 0
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._submit_preview)

        self._build_ui()
        self._build_shortcuts()
        self.zones.changed.connect(self._zones_changed)
        self.zones.overlay_changed.connect(self._zones_overlay_changed)
        self.retheme()
        self.new_project(confirm=False)

    # ------------------------------------------------------------------ ui
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---------------- header
        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(58)
        hh = QHBoxLayout(header)
        hh.setContentsMargins(18, 0, 14, 0)
        hh.setSpacing(6)
        self.logo = QLabel()
        hh.addWidget(self.logo)
        hh.addSpacing(6)
        brand = label("TESSERA", "brand")
        hh.addWidget(brand)
        hh.addSpacing(18)
        sep = QFrame()
        sep.setFixedSize(1, 24)
        sep.setObjectName("statSep")
        hh.addWidget(sep)
        hh.addSpacing(12)
        self.proj_lbl = label("Adsız proje", "projectName")
        hh.addWidget(self.proj_lbl)
        self.dirty_lbl = label("●", "dirtyDot")
        self.dirty_lbl.setToolTip("Kaydedilmemiş değişiklikler var")
        hh.addWidget(self.dirty_lbl)
        hh.addStretch(1)

        def hbtn(text: str, tip: str) -> QToolButton:
            b = QToolButton()
            b.setObjectName("ghost")
            b.setText(text)
            b.setToolTip(tip)
            b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon if text else Qt.ToolButtonIconOnly)
            b.setIconSize(QSize(18, 18))
            b.setCursor(Qt.PointingHandCursor)
            hh.addWidget(b)
            return b

        self.btn_new = hbtn("Yeni", "Yeni proje (Ctrl+N)")
        self.btn_new.clicked.connect(lambda: self.new_project())
        self.btn_open = hbtn("Aç", "Proje ya da görsel aç (Ctrl+O)")
        self.btn_open.clicked.connect(self.open_dialog)
        self.btn_recent = hbtn("", "Son projeler")
        self.recent_menu = QMenu(self)
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        self.btn_recent.setMenu(self.recent_menu)
        self.btn_recent.setPopupMode(QToolButton.InstantPopup)
        self.btn_save = hbtn("Kaydet", "Projeyi kaydet (Ctrl+S)")
        self.btn_save.clicked.connect(self.save)
        hh.addSpacing(8)
        sep2 = QFrame()
        sep2.setFixedSize(1, 24)
        sep2.setObjectName("statSep")
        hh.addWidget(sep2)
        hh.addSpacing(8)
        self.btn_adv = hbtn("Gelişmiş", "Gelişmiş ayarları göster/gizle")
        self.btn_adv.setCheckable(True)
        self.btn_adv.setChecked(self.advanced_on)
        self.btn_adv.toggled.connect(self.set_advanced)
        self.btn_theme = hbtn("", "Açık / koyu tema (Ctrl+T)")
        self.btn_theme.clicked.connect(self.toggle_theme)
        self.btn_help = hbtn("", "Kısayollar (F1)")
        self.btn_help.clicked.connect(self.show_shortcuts)
        hh.addSpacing(8)
        self.btn_export = QPushButton("Dışa aktar")
        self.btn_export.setObjectName("primary")
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.setToolTip("Dışa aktarım adımına git (Ctrl+E)")
        self.btn_export.clicked.connect(lambda: self.go_step(5))
        hh.addWidget(self.btn_export)
        outer.addWidget(header)

        # ---------------- body
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        outer.addLayout(body, 1)

        rail = QFrame()
        rail.setObjectName("rail")
        rail.setFixedWidth(224)
        rv = QVBoxLayout(rail)
        rv.setContentsMargins(0, 16, 0, 14)
        rv.setSpacing(0)
        self.rail = StepRail([(t, i) for t, i in STEPS])
        self.rail.current_changed.connect(self.go_step)
        rv.addWidget(self.rail)
        rv.addStretch(1)
        self.rail_example = QPushButton("  Örnek proje")
        self.rail_example.setObjectName("ghost")
        self.rail_example.setToolTip("Örnek proje: Sovyet afişi")
        self.rail_example.setCursor(Qt.PointingHandCursor)
        self.rail_example.clicked.connect(self.load_example)
        ex = QHBoxLayout()
        ex.setContentsMargins(14, 0, 14, 6)
        ex.addWidget(self.rail_example)
        ex.addStretch(1)
        rv.addLayout(ex)
        foot = label("Bambu Lab X1C · v0.1", "railFoot")
        foot.setContentsMargins(24, 0, 0, 0)
        rv.addWidget(foot)
        body.addWidget(rail)

        centre = QWidget()
        cv = QVBoxLayout(centre)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        bar = QFrame()
        self.stage_bar = bar
        bar.setObjectName("stageBar")
        bar.setFixedHeight(46)
        bh = QHBoxLayout(bar)
        bh.setContentsMargins(0, 0, 12, 0)
        bh.setSpacing(4)
        self.tabs = TabStrip(TABS)
        self.tabs.current_changed.connect(self.show_tab)
        bh.addWidget(self.tabs, 1)

        def zbtn(ic: str, tip: str, fn) -> QToolButton:
            b = QToolButton()
            b.setObjectName("ghost")
            b.setToolTip(tip)
            b.setIconSize(QSize(18, 18))
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(fn)
            b.setProperty("icon_name", ic)
            bh.addWidget(b)
            return b

        self.canvas = Canvas(self.zones)
        self.z_out = zbtn("zoomout", "Uzaklaş (Ctrl+-)", lambda: self.canvas.zoom_by(1 / 1.25))
        self.zoom_lbl = label("100%", "hint")
        self.zoom_lbl.setFixedWidth(44)
        self.zoom_lbl.setAlignment(Qt.AlignCenter)
        bh.addWidget(self.zoom_lbl)
        self.z_in = zbtn("zoomin", "Yakınlaş (Ctrl++)", lambda: self.canvas.zoom_by(1.25))
        self.z_fit = zbtn("fit", "Ekrana sığdır (F)", self.canvas.fit)
        cv.addWidget(bar)
        self.canvas.zoom_changed.connect(lambda s: self.zoom_lbl.setText(f"{int(round(s * 100))}%"))
        self.canvas.files_dropped.connect(lambda ps: self.open_any(ps[0]))
        self.canvas.tool_used.connect(self._tool_used)
        self.canvas.brush_radius_changed.connect(lambda mm: self.zones_step.brush.set_value(mm / 10))
        self.stage = Stage(self, self.canvas)
        self.stage.dial.changed.connect(self.set_light_live)
        self.stage.dial.released.connect(self.set_light_live)
        self.empty = EmptyState(self)
        from .views import Panel3DView, PlateGallery
        self.gallery = PlateGallery()
        self.panel3d = Panel3DView()
        self.panel3d.panel_changed.connect(self._load_panel3d)
        self.panel3d.view_changed.connect(lambda az, el: self._load_panel3d(self.panel3d.spin.value() - 1))
        self.stage_stack = QStackedWidget()
        self.stage_stack.addWidget(self.empty)
        self.stage_stack.addWidget(self.stage)
        self.stage_stack.addWidget(self.gallery)
        self.stage_stack.addWidget(self.panel3d)
        self.tabs.set_shown(TAB_PLATES, False)
        self.tabs.set_shown(TAB_3D, False)
        cv.addWidget(self.stage_stack, 1)
        body.addWidget(centre, 1)

        side = QFrame()
        side.setObjectName("sidePanel")
        side.setFixedWidth(392)
        sv = QVBoxLayout(side)
        sv.setContentsMargins(0, 0, 0, 0)
        self.panels = QStackedWidget()
        self.image_step = ImageStep(self)
        self.fil_step = FilamentStep(self)
        self.zones_step = ZonesStep(self)
        self.relief_step = ReliefStep(self)
        self.prod_step = ProductionStep(self)
        self.export_step = ExportStep(self)
        self.steps = [self.image_step, self.fil_step, self.zones_step, self.relief_step,
                      self.prod_step, self.export_step]
        for p in self.steps:
            self.panels.addWidget(p)
        sv.addWidget(self.panels)
        body.addWidget(side)

        # ---------------- stats bar
        stats = QFrame()
        stats.setObjectName("statsBar")
        stats.setFixedHeight(60)
        st = QHBoxLayout(stats)
        st.setContentsMargins(22, 0, 18, 0)
        st.setSpacing(22)
        self.stat_items = {
            "size": StatItem("boyut", "fit"),
            "parts": StatItem("parça", "relief"),
            "plates": StatItem("tabla", "print"),
            "grams": StatItem("filament", "spool"),
            "hours": StatItem("baskı süresi", "recent"),
            "purge": StatItem("atılan filament", "spools"),
            "budget": StatItem("bütçe", "relief"),
        }
        for i, w in enumerate(self.stat_items.values()):
            if i:
                s = QFrame()
                s.setObjectName("statSep")
                s.setFixedSize(1, 28)
                st.addWidget(s)
            st.addWidget(w)
        self.stat_items["budget"].setCursor(Qt.PointingHandCursor)
        self.stat_items["budget"].mousePressEvent = lambda e: self.go_step(3)
        st.addStretch(1)
        self.status_spin = Spinner(14)
        self.status_spin.hide()
        st.addWidget(self.status_spin)
        self.status = label("", "status")
        st.addWidget(self.status)
        outer.addWidget(stats)

    def _build_shortcuts(self) -> None:
        def sc(seq, fn, ctx=Qt.WindowShortcut):
            s = QShortcut(QKeySequence(seq), self)
            s.setContext(ctx)
            s.activated.connect(fn)
            return s

        sc("Ctrl+N", lambda: self.new_project())
        sc("Ctrl+O", self.open_dialog)
        sc("Ctrl+S", self.save)
        sc("Ctrl+Shift+S", self.save_as)
        sc("Ctrl+E", lambda: self.go_step(5))
        sc("Ctrl+T", self.toggle_theme)
        sc("F1", self.show_shortcuts)
        sc("Ctrl+Z", self.zones.undo)
        sc("Ctrl+Y", self.zones.redo)
        sc("Ctrl+Shift+Z", self.zones.redo)
        sc("Ctrl+0", self.canvas.fit)
        sc("F", self.canvas.fit)
        sc("Ctrl++", lambda: self.canvas.zoom_by(1.25))
        sc("Ctrl+=", lambda: self.canvas.zoom_by(1.25))
        sc("Ctrl+-", lambda: self.canvas.zoom_by(1 / 1.25))
        for i in range(6):
            sc(f"Ctrl+{i + 1}", lambda i=i: self.go_step(i))
        for i in range(len(TABS)):
            sc(f"Alt+{i + 1}", lambda i=i: self.tabs.set_current(i))
        for key, tool in (("H", "pan"), ("B", "brush"), ("E", "eraser"), ("P", "polygon"),
                          ("R", "rect"), ("C", "circle"), ("M", "center")):
            sc(key, lambda t=tool: self._zone_key(lambda: self.set_tool(t)))
        for i, v in enumerate(VOCAB_ORDER):
            sc(str(i + 1), lambda v=v: self._zone_key(lambda: self.set_paint_vocab(v)))
        sc("[", lambda: self._zone_key(lambda: self._brush_step(1 / 1.2)))
        sc("]", lambda: self._zone_key(lambda: self._brush_step(1.2)))
        sc("Delete", lambda: self._zone_key(self.zones_step.delete_selected))

    def _zone_key(self, fn) -> None:
        if self.tab == TAB_ZONES and self.stage_stack.currentWidget() is self.stage:
            fn()

    def _brush_step(self, f: float) -> None:
        v = max(0.5, min(12, self.zones_step.brush.value() * f))
        self.zones_step.brush.set_value(v, emit=True)

    # ------------------------------------------------------------------ theme / modes
    def retheme(self) -> None:
        t = current()
        self.logo.setPixmap(icons.logo(28, t.text, t.bg, t.red, t.gold))
        self.btn_new.setIcon(icons.icon("new", t.text, 18))
        self.btn_open.setIcon(icons.icon("open", t.text, 18))
        self.btn_recent.setIcon(icons.icon("recent", t.text, 18))
        self.btn_save.setIcon(icons.icon("save", t.text, 18))
        self.btn_adv.setIcon(icons.icon("sliders", t.text, 18))
        self.btn_theme.setIcon(icons.icon("sun" if t.name == "dark" else "moon", t.text, 18))
        self.btn_help.setIcon(icons.icon("info", t.text, 18))
        self.rail_example.setIcon(icons.icon("magic", t.gold, 18))
        for b in (self.z_out, self.z_in, self.z_fit):
            b.setIcon(icons.icon(b.property("icon_name"), t.text, 18))
        for w in self.stat_items.values():
            w.retheme()
        self.canvas.set_colours(t.stage, t.stage_grid, t.gold, "#141210")
        self.empty.retheme()
        self.stage.retheme()
        for p in self.steps:
            p.retheme()
        # hairline separators drawn with inline styles
        for f in self.findChildren(QFrame, "advLine"):
            f.setStyleSheet(f"background: {t.line};")
        self.rail.update()
        self.tabs.update()
        self.zones_step.set_paint_vocab(self.canvas.paint_vocab)
        self.prod_step.refresh() if self.has_image() else None

    def toggle_theme(self) -> None:
        self.theme = theme.LIGHT if self.theme is theme.DARK else theme.DARK
        self.store.setValue("theme", self.theme.name)
        theme.apply(QApplication.instance(), self.theme)
        self.retheme()
        self.request_preview(immediate=True)   # oblique background follows the theme

    def set_advanced(self, on: bool) -> None:
        self.advanced_on = bool(on)
        self.store.setValue("advanced", self.advanced_on)
        for p in self.steps:
            p.set_advanced(self.advanced_on)
        if self.btn_adv.isChecked() != self.advanced_on:
            self.btn_adv.setChecked(self.advanced_on)

    # ------------------------------------------------------------------ project lifecycle
    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        box = QMessageBox(self)
        box.setWindowTitle("Kaydedilmemiş değişiklikler")
        box.setText("Bu projede kaydedilmemiş değişiklikler var.")
        box.setInformativeText("Önce kaydetmek ister misin?")
        save = box.addButton("Kaydet", QMessageBox.AcceptRole)
        box.addButton("Kaydetme", QMessageBox.DestructiveRole)
        cancel = box.addButton("Vazgeç", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is cancel:
            return False
        if box.clickedButton() is save:
            return self.save()
        return True

    def new_project(self, confirm: bool = True) -> None:
        if confirm and not self._confirm_discard():
            return
        self.project = Project()
        self.project.settings.palette = []
        self.original_qimage = None
        self.zones_qimage = None
        self.image_px = (0, 0)
        self.image_missing = False
        self.last_result = None
        self.last_estimate = None
        self.last_export = None
        self.engine.invalidate()
        self.zones.reset(1.0, default_vocab=int(Vocab.STONES))
        self.canvas.set_image(None)
        self._apply_project_to_ui()
        self.go_step(0)
        self.set_dirty(False)

    def _load_image_files(self, path: str) -> None:
        """Load the original for display; raises on unreadable files."""
        src = ImageOps.exif_transpose(Image.open(path))
        src.load()
        note = ""
        if src.mode in ("RGBA", "LA", "PA") or (src.mode == "P" and "transparency" in src.info):
            rgba = src.convert("RGBA")
            if rgba.getextrema()[3][0] < 250:
                bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                src = Image.alpha_composite(bg, rgba)
                note = "Saydam alanlar beyaz sayıldı."
        im = src.convert("RGB")
        self.image_note = note
        self.image_px = im.size
        disp = im.copy()
        disp.thumbnail((DISPLAY_MAX, DISPLAY_MAX), Image.LANCZOS)
        arr = np.asarray(disp)
        self.original_qimage = rgb_to_qimage(arr)
        g = np.asarray(disp.convert("L"), np.float32)
        t = current()
        # dim, desaturated base under the zone colours
        base = (g * 0.55 + 40).clip(0, 255).astype(np.uint8)
        self.zones_qimage = rgb_to_qimage(np.repeat(base[..., None], 3, 2))
        self.canvas.set_scene_size(disp.width, disp.height)
        self.canvas.width_mm = self.project.settings.width_mm

    def aspect(self) -> float:
        w, h = self.image_px
        if w:
            return h / w
        return self._missing_aspect if self.image_missing else 0.0

    def has_image(self) -> bool:
        """An image is loaded and on screen (not merely referenced by the project)."""
        return self.original_qimage is not None

    def open_any(self, path: str) -> None:
        if path.lower().endswith(".json"):
            self.open_project(path)
        else:
            self.load_image(path)

    def choose_image(self) -> None:
        start = os.path.expanduser("~")
        for cand in (os.path.dirname(self.project.image_path or ""), os.path.dirname(self.project.path or "")):
            if cand and os.path.isdir(cand):
                start = cand
                break
        title = "Görseli bul" if self.image_missing else "Görsel seç"
        if self.image_missing:
            start = os.path.join(start, os.path.basename(self.project.image_path))
        path, _ = QFileDialog.getOpenFileName(self, title, start, IMAGE_FILTER)
        if path:
            self.load_image(path)

    def load_image(self, path: str, relink: bool = False) -> bool:
        """Open an image. relink=True (or a missing project image): this is the new location of
        the project's image, so zones, palette and settings are kept as they are."""
        relink = relink or self.image_missing
        had_image = self.has_image()
        try:
            old_aspect = self.aspect()
            self._load_image_files(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Görsel açılamadı",
                                f"“{os.path.basename(path)}” açılamadı.\n\n{friendly_error(e)}")
            return False
        keep_zones = relink or (had_image and abs(old_aspect - self.aspect()) < 0.01)
        if relink and abs(old_aspect - self.aspect()) > 0.02:
            QTimer.singleShot(0, lambda: self._flash("Dikkat: bu görselin oranı farklı; bölgeleri kontrol et."))
        self.image_missing = False
        self.project.image_path = os.path.abspath(path)
        if not keep_zones:
            self.zones.reset(self.aspect(), default_vocab=int(self.project.settings.default_vocab))
        if not had_image and not relink:
            self._apply_default_preset()
        self.last_result = None
        self.engine.invalidate()
        auto = len(self.project.settings.palette) < 2
        if auto:
            self._auto_palette()          # instant local guess, refined below in the background
        self._apply_project_to_ui()
        self.set_dirty(True)
        self.show_tab(TAB_MOSAIC)
        if auto and suggest_palette_fn() is not None:
            self.suggest_palette(n=4, auto=True)
        else:
            self.request_preview(immediate=True)
        return True

    def _auto_palette(self) -> None:
        """First image: pick a starting palette locally (fast) so the preview appears at once."""
        try:
            from .hooks import fallback_palette
            im = Image.open(self.project.image_path).convert("RGB")
            im.thumbnail((200, 200))
            owned = [f for f in self.library if f.owned]
            pal = fallback_palette(np.asarray(im), owned if len(owned) >= 3 else self.library, 4)
            self.project.settings.palette = [self._fil_copy(f) for f in pal]
        except Exception:  # noqa: BLE001
            self.project.settings.palette = [self._fil_copy(f) for f in self.library[:3]]

    @staticmethod
    def _fil_copy(f: Filament) -> Filament:
        return Filament(f.name, tuple(f.rgb), f.finish, f.brand, f.owned)

    def load_example(self) -> None:
        if not self._confirm_discard():
            return
        img = EXAMPLE_DIR / "kaynak.webp"
        zfile = EXAMPLE_DIR / "bolgeler.json"
        if not img.exists():
            QMessageBox.warning(self, "Örnek bulunamadı", f"Örnek görsel bulunamadı:\n{img}")
            return
        p = Project()
        p.name = "Sovyet afişi"
        p.image_path = str(img)
        p.settings = Settings()
        pal = []
        for nm, rgb in zip(POSTER_PALETTE, [(206, 32, 40), (24, 23, 25), (243, 237, 224)]):
            f = find(self.library, nm) or nearest(self.library, rgb)
            f.owned = True
            pal.append(self._fil_copy(f))
        p.settings.palette = pal
        if zfile.exists():
            doc = json.loads(zfile.read_text(encoding="utf-8"))
            p.shapes = doc.get("sekiller", [])
            p.settings.default_vocab = int(doc.get("varsayilan", int(Vocab.RAYS)))
            rc = doc.get("isin_merkezi")
            p.rays_center = (float(rc[0]), float(rc[1])) if rc else None
        else:
            p.settings.default_vocab = int(Vocab.RAYS)
            p.rays_center = (0.5, 0.118)
        from .filaments import find_preset, load_presets
        pr = find_preset(load_presets(), "Sovyet Afiş")
        self._save_owned()
        self.apply_project(p)
        if pr:
            self.apply_preset(pr, rebuild=False)
            self.request_preview(immediate=True)
        self.set_dirty(False)
        self.go_step(0)

    def apply_project(self, p: Project) -> None:
        self.project = p
        self.last_result = None
        self.last_estimate = None
        self.last_export = None
        self.image_missing = False
        self.original_qimage = None
        self.zones_qimage = None
        self.image_px = (0, 0)
        self.engine.invalidate()
        if p.image_path:
            try:
                self._load_image_files(p.image_path)
            except Exception as e:  # noqa: BLE001
                # K-3: never forget the stored path. Keep everything and offer to relink.
                self.original_qimage = None
                self.zones_qimage = None
                self.image_px = (0, 0)
                self.image_missing = True
                m = p.paint_mask
                if p.image_size and p.image_size[0]:
                    self._missing_aspect = p.image_size[1] / p.image_size[0]
                elif m is not None and m.shape[1]:
                    self._missing_aspect = m.shape[0] / m.shape[1]
                else:
                    self._missing_aspect = 1.0
                why = ("Dosya taşınmış, adı değişmiş ya da disk takılı değil olabilir."
                       if isinstance(e, FileNotFoundError) else friendly_error(e))
                QMessageBox.warning(self, "Görsel bulunamadı",
                                    f"Projenin görseli açılamadı:\n{os.path.normpath(p.image_path)}\n\n{why}\n\n"
                                    "Bölgeler, renkler ve ayarlar korunuyor. “Görseli bul…” ile "
                                    "görselin yeni yerini gösterebilirsin.")
        # merge project's owned list into the user's owned list
        if p.owned:
            for f in self.library:
                if f.name in p.owned:
                    f.owned = True
            self._save_owned()
        self.zones.reset(self.aspect() or 1.0, p.shapes, p.paint_mask, p.rays_center,
                         int(p.settings.default_vocab))
        self._apply_project_to_ui()
        if self.has_image():
            self.show_tab(TAB_MOSAIC)
            self.request_preview(immediate=True)

    def _apply_project_to_ui(self) -> None:
        has = self.has_image()
        self.empty.set_missing(self.project.image_path if self.image_missing else "")
        self.stage_stack.setCurrentWidget(self.stage if has else self.empty)
        self.stage_bar.setVisible(has)
        self.empty.refresh_recent(self.recent_files())
        self.canvas.width_mm = self.project.settings.width_mm
        self.stage.dial.set_light(self.project.light_az, self.project.light_el)
        for p in self.steps:
            p.refresh()
            p.set_advanced(self.advanced_on)
        self.fil_step.build_library()
        self.fil_step.refresh()
        self.rail.set_enabled_steps([True] + [has] * 5)
        self.btn_export.setEnabled(has)
        self.btn_save.setEnabled(True)
        self._update_rail()
        self._update_stats()
        self._update_title()
        self.set_tool(self.canvas.tool if has else "pan")
        self._refresh_plates_tab()
        self._refresh_3d_tab()
        self._update_budget_chip({})
        self.relief_step.update_budget({})
        if self.tab in (TAB_PLATES, TAB_3D) and not self.tabs.shown[self.tab]:
            self.show_tab(TAB_MOSAIC)

    def open_dialog(self) -> None:
        start = os.path.dirname(self.project.path) if self.project.path else os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Aç", start,
                                              f"{PROJECT_FILTER};;{IMAGE_FILTER};;Tüm dosyalar (*.*)")
        if path:
            self.open_any(path)

    def open_project(self, path: str) -> bool:
        if not self._confirm_discard():
            return False
        try:
            p = Project.load(path)
        except Exception as e:  # noqa: BLE001
            why = (friendly_error(e) if isinstance(e, OSError) else
                   "Bu dosya bir Tessera projesi değil ya da bozulmuş.")
            QMessageBox.warning(self, "Proje açılamadı", f"“{os.path.basename(path)}” açılamadı.\n\n{why}")
            return False
        self.apply_project(p)
        self._add_recent(path)
        self.set_dirty(False)
        self.go_step(0)
        return True

    def _collect(self) -> None:
        p = self.project
        p.shapes = [dict(s) for s in self.zones.shapes]
        p.rays_center = self.zones.rays_center
        p.settings.default_vocab = self.zones.default_vocab
        p.paint_mask = self.zones.mask_array()
        p.owned = [f.name for f in self.library if f.owned]
        if self.image_px[0]:
            p.image_size = tuple(self.image_px)

    def save(self) -> bool:
        if not self.project.path:
            return self.save_as()
        return self._save_to(self.project.path)

    def save_as(self) -> bool:
        base = self.project.display_name()
        start = os.path.dirname(self.project.path) if self.project.path else (
            os.path.dirname(self.project.image_path) if self.project.image_path else os.path.expanduser("~"))
        path, _ = QFileDialog.getSaveFileName(self, "Projeyi kaydet", os.path.join(start, base + PROJECT_SUFFIX),
                                              PROJECT_FILTER)
        if not path:
            return False
        if not path.endswith(PROJECT_SUFFIX):
            path = os.path.splitext(path)[0].removesuffix(".tessera") + PROJECT_SUFFIX
        return self._save_to(path)

    def _save_to(self, path: str) -> bool:
        self._collect()
        try:
            self.project.save(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Kaydedilemedi",
                                f"Proje kaydedilemedi:\n{os.path.normpath(path)}\n\n{friendly_error(e)}")
            return False
        self._add_recent(path)
        self.set_dirty(False)
        self._flash(f"Kaydedildi: {os.path.basename(path)}" +
                    (" (görsel hâlâ bulunamıyor; yolu korundu)" if self.image_missing else ""))
        return True

    # ------------------------------------------------------------------ recent files
    def recent_files(self) -> list[str]:
        return [f for f in (self.store.value("recent", [], type=list) or []) if f]

    def _add_recent(self, path: str) -> None:
        path = os.path.abspath(path)
        lst = [path] + [f for f in self.recent_files() if os.path.abspath(f) != path]
        self.store.setValue("recent", lst[:8])

    def _fill_recent_menu(self) -> None:
        self.recent_menu.clear()
        files = [f for f in self.recent_files() if os.path.exists(f)]
        for f in files:
            a = self.recent_menu.addAction(os.path.basename(f).replace(PROJECT_SUFFIX, ""))
            a.setToolTip(f)
            a.triggered.connect(lambda _=False, f=f: self.open_project(f))
        if not files:
            a = self.recent_menu.addAction("Henüz kayıtlı proje yok")
            a.setEnabled(False)
        self.recent_menu.addSeparator()
        self.recent_menu.addAction("Örnek proje: Sovyet afişi", self.load_example)

    # ------------------------------------------------------------------ dirty / title
    def set_dirty(self, on: bool = True) -> None:
        self.dirty = on
        self.dirty_lbl.setVisible(on)
        self._update_title()

    def _update_title(self) -> None:
        name = self.project.display_name()
        self.proj_lbl.setText(name)
        self.setWindowTitle(f"{'• ' if self.dirty else ''}{name} — Tessera")

    def _flash(self, msg: str) -> None:
        self.status.setStyleSheet("")
        self.status.setText(msg)
        QTimer.singleShot(2500, self._update_status)

    # ------------------------------------------------------------------ steps & tabs
    def go_step(self, i: int) -> None:
        if i > 0 and not self.has_image():
            i = 0
        self.rail.set_current(i)
        self.panels.setCurrentIndex(i)
        if i == 2:
            if self.tab != TAB_ZONES:
                self._tab_before_zones = self.tab
            self.show_tab(TAB_ZONES)
        elif self.tab == TAB_ZONES and self.panels.currentIndex() != 2:
            self.show_tab(self._tab_before_zones if self._tab_before_zones != TAB_ZONES else TAB_MOSAIC)
        if i == 4:
            self.prod_step.refresh()
        if i == 5:
            self.export_step.refresh()

    def show_tab(self, i: int) -> None:
        self.tab = i
        self.tabs.set_current(i, emit=False)
        if self.has_image():
            page = {TAB_PLATES: self.gallery, TAB_3D: self.panel3d}.get(i, self.stage)
            self.stage_stack.setCurrentWidget(page)
            for b in (self.z_out, self.z_in, self.z_fit, self.zoom_lbl):
                b.setEnabled(page is self.stage)
            if i == TAB_3D:
                self._load_panel3d(self.panel3d.spin.value() - 1)
            if page is not self.stage:
                return
        self.canvas.set_editing(i == TAB_ZONES)
        geom = "oblique" if i == TAB_OBLIQUE else "flat"
        kind_changed = geom != self._geom_kind
        self._geom_kind = geom
        self._show_tab_image(fit=kind_changed)
        self.stage.light_box.setVisible(i in (TAB_MOSAIC, TAB_OBLIQUE) and self.last_result is not None)
        self.stage.legend.setVisible(i == TAB_ZONES)
        self._update_legend()
        self.stage.place()

    def _show_tab_image(self, fit: bool = False) -> None:
        r = self.last_result
        img = None
        if self.tab == TAB_ORIG:
            img = self.original_qimage
        elif self.tab == TAB_ZONES:
            img = self.zones_qimage
        elif r is not None:
            arr = {TAB_MOSAIC: r.top, TAB_OBLIQUE: r.oblique, TAB_FLAT: r.flat}[self.tab]
            img = rgb_to_qimage(arr)
        else:
            img = self.original_qimage   # placeholder until the first preview arrives
        self.canvas.set_image(img, keep_view=not fit)
        if fit:
            self.canvas.fit()

    def _update_legend(self) -> None:
        used = self.zones.used_vocabs()
        for v, w in self.stage.legend_rows.items():
            w.setVisible(v in used)
        self.stage.place()

    def _update_rail(self) -> None:
        p = self.project
        s = p.settings
        has = self.has_image()
        a = self.aspect()
        self.rail.set_subtitle(0, f"{tr_num(s.width_mm / 10, 1)} × {tr_num(s.width_mm * a / 10, 1)} cm"
                               if has else "Görsel seç")
        self.rail.set_done(0, has)
        n = len(s.palette)
        self.rail.set_subtitle(1, f"{n} renk" if n else "Renk seç")
        self.rail.set_done(1, has and n >= 2)
        ns = len(self.zones.shapes)
        dv = VOCAB_NAMES_TR[Vocab(self.zones.default_vocab)]
        self.rail.set_subtitle(2, (f"{ns} şekil · " if ns else "") + f"zemin: {dv}")
        self.rail.set_done(2, has and (ns > 0 or self.zones.has_paint()))
        self.rail.set_subtitle(3, light_text(p.light_az, p.light_el).split(" · ")[0])
        self.rail.set_subtitle(4, "Tek renk tablalar" if s.mode == Mode.PIECES else "Çok renkli paneller (AMS)")
        self.rail.set_subtitle(5, "Hazır" if self.last_export else "3MF · kılavuz · rapor")
        self.rail.set_done(5, bool(self.last_export))

    # ------------------------------------------------------------------ edits from panels
    def set_width_mm(self, mm: float) -> None:
        if abs(mm - self.project.settings.width_mm) < 1e-6:
            return
        self.project.settings.width_mm = float(mm)
        self.canvas.width_mm = float(mm)
        self.changed()

    def set_setting(self, key: str, value, rebuild: bool = True) -> None:
        if getattr(self.project.settings, key) == value:
            return
        setattr(self.project.settings, key, value)
        if key in ("figure_bump_mm", "bevel_mm", "base_thickness_mm"):
            self._preset_touched()
        if key == "panel_max_mm":
            self.image_step._update_height()
            self.prod_step.refresh()
        self.changed(rebuild=rebuild)

    def set_vocab_param(self, v: int, key: str, value: float) -> None:
        vp = self.project.settings.vocab[int(v)]
        if abs(getattr(vp, key) - value) < 1e-9:
            return
        setattr(vp, key, float(value))
        self._preset_touched()
        self.changed()

    def _preset_touched(self) -> None:
        if self.project.preset and not self.project.preset_modified:
            self.project.preset_modified = True
            self.relief_step.show_preset()

    def apply_preset(self, pr: dict, set_default_vocab: bool = False, rebuild: bool = True) -> None:
        """Art-team preset: stone sizes/heights and shape settings. Zones stay as drawn; only a
        brand-new image (set_default_vocab) also takes the preset's background stone type."""
        from ..model import VocabParams
        s = self.project.settings
        self.project.preset = str(pr.get("ad", ""))
        self.project.preset_modified = False
        if set_default_vocab and "default_vocab" in pr:
            s.default_vocab = int(pr["default_vocab"])
            self.zones.default_vocab = int(pr["default_vocab"])
            self.zones.recompose()
        fields = set(VocabParams.__dataclass_fields__)
        for k, vals in (pr.get("vocab") or {}).items():
            vp = s.vocab.get(int(k))
            if vp is None:
                continue
            for f, v in vals.items():
                if f in fields:
                    setattr(vp, f, type(getattr(vp, f))(v))
        for key in ("figure_bump_mm", "bevel_mm", "gap_mm", "base_thickness_mm"):
            if key in pr:
                setattr(s, key, float(pr[key]))
        self.relief_step.refresh()
        if rebuild:
            self.changed()

    def _apply_default_preset(self) -> None:
        from .filaments import default_preset, load_presets
        pr = default_preset(load_presets())
        if pr:
            self.apply_preset(pr, set_default_vocab=True, rebuild=False)

    def set_mode(self, mode: str) -> None:
        if self.project.settings.mode != mode:
            self.project.settings.mode = mode
            self.prod_step.refresh()
            self.changed()
            self._update_stats()

    def reseed(self) -> None:
        self.project.settings.seed = int(np.random.default_rng().integers(1, 9999))
        self.relief_step.refresh()
        self.changed()

    def changed(self, rebuild: bool = True) -> None:
        self.set_dirty(True)
        self._update_rail()
        self._update_stats(local_only=True)
        if rebuild:
            self.request_preview()

    def set_light_live(self, az: float, el: float) -> None:
        self.project.light_az, self.project.light_el = float(az), float(el)
        self.stage.dial.set_light(az, el)
        self.relief_step.set_light(az, el)
        self.rail.set_subtitle(3, light_text(az, el).split(" · ")[0])
        self.request_preview(delay=40)

    # palette
    def toggle_palette(self, fil: Filament) -> None:
        pal = self.project.settings.palette
        idx = next((i for i, f in enumerate(pal) if f.name == fil.name), -1)
        self.fil_step.show_note("")
        if idx >= 0:
            del pal[idx]
        else:
            if len(pal) >= 8:
                self._flash("En fazla 8 renk seçilebilir.")
                return
            same = next((f for f in pal if sum((a - b) ** 2 for a, b in zip(f.rgb, fil.rgb)) < 64), None)
            if same is not None:
                self._flash(f"“{fil.name}” paletteki “{same.name}” ile aynı renk; aynı rengi iki kez seçemezsin.")
                self.fil_step.show_note(f"“{fil.name}”, “{same.name}” ile aynı renk. "
                                        "Aynı renk iki makara gibi bölünürdü; bu yüzden eklenmedi.")
                return
            pal.append(self._fil_copy(fil))
        self.fil_step.refresh()
        self.changed()

    def owned_changed(self, fil: Filament) -> None:
        self._save_owned()
        self.fil_step._filter()

    def _save_owned(self) -> None:
        self.store.setValue("owned", [f.name for f in self.library if f.owned])

    # zones
    def set_tool(self, tool: str) -> None:
        self.canvas.set_tool(tool)
        self.zones_step.set_tool(tool)
        if tool != "pan" and self.tab != TAB_ZONES and self.has_image():
            self.go_step(2)

    def set_paint_vocab(self, v: int) -> None:
        self.canvas.paint_vocab = int(v)
        self.zones_step.set_paint_vocab(int(v))
        if self.canvas.tool in ("pan", "eraser", "center"):
            self.set_tool("brush")

    def _tool_used(self, tool: str) -> None:
        if tool == "center":
            self.set_tool("pan")

    def _zones_changed(self) -> None:
        self.zones_step.refresh()
        self.relief_step.refresh()
        self._update_legend()
        self.changed()

    def _zones_overlay_changed(self) -> None:
        pass  # canvas listens directly

    def suggest_zones_available(self) -> bool:
        return suggest_zones_fn() is not None

    # ------------------------------------------------------------------ preview
    def request_preview(self, immediate: bool = False, delay: int = 300) -> None:
        if not self.has_image():
            return
        if len(self.project.settings.palette) < 2:
            self._error = "Önizleme için en az 2 filament seç."
            self._show_error()
            return
        self._debounce.start(0 if immediate else delay)

    def _submit_preview(self) -> None:
        if not self.has_image() or len(self.project.settings.palette) < 2:
            return
        self.gen += 1
        s = copy.deepcopy(self.project.settings)
        s.mm_per_px = preview_mm_per_px(s.width_mm)
        st = QColor(current().stage)
        req = PreviewRequest(self.project.image_path, s, self.zones.shapes, self.zones.mask_array(),
                             self.zones.rays_center, self.project.light_az, self.project.light_el,
                             bg=(st.red(), st.green(), st.blue()), gen=self.gen)
        self.engine.submit(req)

    def _on_busy(self, on: bool) -> None:
        self.stage.busy.setVisible(on and not (self.last_result and self.last_result.relit_only))
        if on:
            self.stage.spinner.start()
            self.status_spin.start()
            self.status.setText("Önizleme hesaplanıyor…")
        else:
            self.stage.spinner.stop()
            self.status_spin.stop()
            self.stage.busy.hide()
            self._update_status()
        self.stage.place()

    def _on_preview(self, res) -> None:
        first = self.last_result is None
        self.last_result = res
        self.last_estimate = res.estimate
        self._error = ""
        self._show_error()
        if self.tab in (TAB_MOSAIC, TAB_OBLIQUE, TAB_FLAT):
            self._show_tab_image(fit=first and self.tab == TAB_OBLIQUE)
        self.stage.light_box.setVisible(self.tab in (TAB_MOSAIC, TAB_OBLIQUE))
        self._update_stats()
        self.prod_step.update_estimate(res.estimate)
        self.image_step._update_height()
        if not res.relit_only:
            self.exact = None
            self._count_timer.start(600)        # exact count once editing settles
        try:
            from ..export.api import preview_key
            key = preview_key(res.mosaic)
        except Exception:  # noqa: BLE001
            key = res.gen
        if key != getattr(self, "_panel3d_key", None):
            self._panel3d_stale = True
        self.relief_step.update_budget(self._budget_with_exact(res.budget))
        self._update_budget_chip(res.budget)
        self._refresh_3d_tab()
        if self.tab == TAB_3D and getattr(self, "_panel3d_stale", False):
            self._load_panel3d(self.panel3d.spin.value() - 1)
        self._update_status()
        self.stage.place()

    def _on_preview_failed(self, gen: int, msg: str) -> None:
        self._error = msg
        self._show_error()
        self._update_status()

    def error_action(self) -> None:
        if len(self.project.settings.palette) < 2:
            self.go_step(1)
        else:
            self.request_preview(immediate=True)

    def _show_error(self) -> None:
        from PySide6.QtWidgets import QGraphicsOpacityEffect
        self.stage.error.setVisible(bool(self._error))
        self.stage.err_lbl.setText(self._error)
        self.stage.err_btn.setText("Filamentlere git" if len(self.project.settings.palette) < 2 else "Tekrar dene")
        for w in self.stat_items.values():   # old numbers would mislead while the preview is broken
            if self._error:
                eff = QGraphicsOpacityEffect(w)
                eff.setOpacity(0.35)
                w.setGraphicsEffect(eff)
            else:
                w.setGraphicsEffect(None)
        self.stage.place()
        self._update_status()

    def _update_status(self) -> None:
        if self.engine and self.status_spin.isVisible():
            return
        if self._error:
            self.status.setText("Önizleme yapılamadı")
            self.status.setStyleSheet(f"color: {current().red};")
            return
        self.status.setStyleSheet("")
        r = self.last_result
        if r is None:
            self.status.setText("" if not self.has_image() else "Önizleme bekleniyor")
        else:
            mm = r.mosaic.settings.mm_per_px
            from .engine import export_mm_per_px
            self.status.setText(f"Hızlı önizleme güncel · {tr_num(r.seconds, 1)} sn")
            self.status.setToolTip(f"Önizleme {tr_num(mm, 1)} mm/piksel çözünürlükte hesaplanır; sayılar (≈) "
                                   f"tahmindir. Dışa aktarım {tr_num(export_mm_per_px(self.project.settings.width_mm), 1)}"
                                   " mm/piksel ile daha ince hesaplanır.")

    def estimates_by_mode(self) -> dict:
        r = self.last_result
        return dict(getattr(r, "est_by_mode", {}) or {}) if r is not None else {}

    def current_estimate(self) -> Optional[dict]:
        return self.estimates_by_mode().get(self.project.settings.mode) or (
            self.last_estimate if self.estimate_is_current() else None)

    def confirm_big(self, width_cm: float) -> bool:
        a = self.aspect() or 1.0
        r = QMessageBox.question(
            self, "Çok büyük eser",
            f"{tr_num(width_cm, 0)} × {tr_num(width_cm * a, 0)} cm çok büyük bir eser.\n\n"
            "Yüzlerce tabla ve haftalarca baskı demek olabilir; önizleme yavaşlar, dışa aktarım "
            "dakikalar sürer ve çok bellek ister. Bu boyuta geçilsin mi?",
            QMessageBox.Yes | QMessageBox.No)
        return r == QMessageBox.Yes

    def estimate_is_current(self) -> bool:
        """The estimate depends on the production mode: is it from a mosaic built in this mode?"""
        r = self.last_result
        return r is not None and r.mosaic.settings.mode == self.project.settings.mode

    def _update_stats(self, local_only: bool = False) -> None:
        s = self.project.settings
        si = self.stat_items
        a = self.aspect()
        if a:
            si["size"].set(f"{tr_num(s.width_mm / 10, 1)} × {tr_num(s.width_mm * a / 10, 1)} cm")
        else:
            si["size"].set("—")
        if local_only:
            return
        e = self.last_estimate or {}

        def num(k):
            v = e.get(k)
            return None if v is None else float(v)

        parca, tabla, gram, saat, purge, panel = (num(k) for k in ("parca", "tabla", "gram", "saat", "purge_gram", "panel"))
        if s.mode == Mode.AMS and panel is not None and self.estimate_is_current():
            si["parts"].caption.setText("panel (tek parça)")
            si["parts"].set(tr_num(panel), "Çok renkli modda her panel tek parça basılır.")
        else:
            si["parts"].caption.setText("parça")
            ex = self.exact_parts_current()
            if ex is not None:
                h = ex.get("hassas")
                si["parts"].set(tr_num(ex["parca"]),
                                "Dışa aktarımla aynı hesap." + (f" {tr_num(h)} tanesi hassas (küçük) parça." if h else ""))
            else:
                si["parts"].set(f"≈ {tr_num(parca)}" if parca is not None else "—",
                                "Hızlı tahmin. Kesin sayı arka planda hesaplanıyor…")
        si["plates"].set(tr_num(tabla) if tabla is not None else "—")
        si["grams"].set(f"≈ {tr_num(gram / 1000, 2)} kg" if gram is not None else "—")
        si["hours"].set(f"~{tr_num(saat)} saat" if saat is not None else "—")
        if not self.project.image_path:
            si["purge"].set("—")
        elif s.mode == Mode.PIECES:
            si["purge"].set("0 g", "Tek renk tablalar: filament hiç değişmez, renk değişiminde atılan filament yok.")
        elif not self.estimate_is_current():
            si["purge"].set("…", "Yeni moda göre hesaplanıyor.")
        else:
            si["purge"].set(f"≈ {tr_num(purge)} g" if purge is not None else "hesaplanmadı",
                            "Çok renkli panellerde her renk değişiminde atılan filament (tahmin).")

    # ------------------------------------------------------------------ suggestions
    def _start_suggest(self, fn, on_done, label_text: str, on_error=None) -> None:
        if self._suggest_task and self._suggest_task.isRunning():
            return
        self.status.setText(label_text)
        self.status_spin.start()
        t = TaskThread(fn, self)
        t.done.connect(on_done)
        t.error.connect(on_error or (lambda msg: QMessageBox.warning(self, "Öneri yapılamadı", msg)))
        t.finished.connect(lambda: (self.status_spin.stop(), self._update_status()))
        self._suggest_task = t
        t.start()

    def _preview_settings(self) -> Settings:
        s = copy.deepcopy(self.project.settings)
        s.mm_per_px = preview_mm_per_px(s.width_mm)
        return s

    def suggest_palette(self, n: Optional[int] = None, auto: bool = False) -> None:
        """auto=True: first palette for a new image (thumbnail, quiet on failure)."""
        if not self.has_image():
            return
        n = n or self.fil_step.n_spin.value()
        fn = suggest_palette_fn()
        lib = [self._fil_copy(f) for f in self.library]
        path, s = self.project.image_path, self._preview_settings()

        def job(progress):
            if auto:
                im = Image.open(path).convert("RGB")
                im.thumbnail((240, 240))
                img = np.asarray(im)
            else:
                from ..pipeline import load_image
                img = load_image(path, s)
            return run_suggest_palette(fn, img, lib, n, owned_only=True)

        def done(res):
            pal, roles = res
            if path != self.project.image_path:
                return
            self.palette_roles = roles
            uniq = []
            for f in pal:
                if all(sum((a - b) ** 2 for a, b in zip(f.rgb, g.rgb)) >= 64 for g in uniq):
                    uniq.append(f)
            pal = uniq
            self.project.settings.palette = [self._fil_copy(f) for f in pal]
            self.fil_step.refresh()
            self.changed()
            if auto:
                self.request_preview(immediate=True)
            self._flash(f"Önerilen palet: {', '.join(f.name for f in pal)}")

        def failed(msg):
            if auto:
                self.request_preview(immediate=True)
            else:
                QMessageBox.warning(self, "Öneri yapılamadı", msg)

        self._start_suggest(job, done, "Renkler öneriliyor…", failed)

    def suggest_zones(self) -> None:
        fn = suggest_zones_fn()
        if fn is None or not self.has_image():
            return

        path, s = self.project.image_path, self._preview_settings()

        def job(progress):
            from ..pipeline import load_image
            return run_suggest_zones(fn, load_image(path, s), s)

        def done(res):
            self.zones.replace(res["shapes"], res["mask"], res["rays_center"], res["default"])
            self.zones_step.show_reasons(res.get("reasons") or [])
            self._flash("Bölgeler önerildi. Fırçayla düzeltebilirsin.")

        self._start_suggest(job, done, "Bölgeler öneriliyor…")

    # ------------------------------------------------------------------ export
    def default_export_dir(self) -> str:
        name = self.project.display_name().replace(" ", "_")
        if self.project.path:
            return os.path.join(os.path.dirname(self.project.path), f"{name}_cikti")
        return os.path.join(os.path.expanduser("~"), "Documents", "Tessera", name)

    def set_export_dir(self, d: str) -> None:
        self.project.export_dir = d
        self.set_dirty(True)

    def export(self) -> None:
        if self._export_task and self._export_task.isRunning():
            return
        if not self.has_image() or len(self.project.settings.palette) < 2:
            self.export_step.show_error("Önce bir görsel ve en az 2 filament seç.")
            return
        out = (self.export_step.folder.text().strip() or self.default_export_dir())
        if os.path.isdir(out) and os.listdir(out) and not getattr(self, "_skip_overwrite_prompt", False):
            r = QMessageBox.question(self, "Klasör dolu",
                                     f"“{os.path.basename(out)}” klasöründe dosyalar var. "
                                     "Aynı adlı dosyaların üzerine yazılsın mı?",
                                     QMessageBox.Yes | QMessageBox.No)
            if r != QMessageBox.Yes:
                return
        warn = self.export_warnings()
        if warn and not getattr(self, "_skip_export_warnings", False):
            r = QMessageBox.question(self, "Dışa aktarmadan önce", "\n\n".join(warn) + "\n\nYine de dışa aktarılsın mı?",
                                     QMessageBox.Yes | QMessageBox.No)
            if r != QMessageBox.Yes:
                return
        payload = export_payload(self.project.settings, self.project.image_path, self.zones.shapes,
                                 self.zones.mask_array(), self.zones.rays_center, out)
        self.project.export_dir = out
        if self.export_in_process:
            t = ExportJob(payload, self)       # separate process: the window can never freeze
            t.cancelled.connect(self._export_cancelled)
        else:
            t = TaskThread(lambda progress: run_export(payload, progress), self)
        self.export_step.set_running(True, can_cancel=self.export_in_process)
        t.progress.connect(self._export_progress)
        t.done.connect(self._export_done)
        t.error.connect(self._export_error)
        self._export_task = t
        t.start()

    def cancel_export(self) -> None:
        if isinstance(self._export_task, ExportJob) and self._export_task.isRunning():
            self._export_task.cancel()
        if self._count_job is not None and self._count_job.isRunning():
            self._count_job.wait(60000)

    def _export_cancelled(self) -> None:
        self.export_step.set_running(False)
        self.export_step.show_error("Dışa aktarım iptal edildi. Klasörde yarım dosyalar kalmış olabilir.")
        self._update_status()

    def _export_progress(self, f: float, msg: str) -> None:
        self.export_step.set_progress(f, msg)
        self.status.setText(f"Dışa aktarılıyor · %{int(f * 100)}")

    # ------------------------------------------------------------------ exact parts (R3-4)
    def exact_parts_current(self) -> Optional[dict]:
        r = self.last_result
        if self.exact and r is not None and self.exact_gen == r.gen and self.project.settings.mode == Mode.PIECES:
            return self.exact
        return None

    def _start_exact_count(self) -> None:
        r = self.last_result
        if r is None or not self.has_image() or self.project.settings.mode != Mode.PIECES:
            return
        if self._debounce.isActive() or self.engine._pending is not None:
            self._count_timer.start(800)
            return
        if self._count_job is not None and self._count_job.isRunning():
            self._count_timer.start(500)          # one at a time; try again shortly
            return
        from .engine import exact_parts
        gen, m = r.gen, r.mosaic
        job = TaskThread(lambda progress: exact_parts(m), self)
        job.done.connect(lambda res, g=gen: self._exact_done(g, res))
        job.error.connect(lambda msg: None)
        self._count_job = job
        job.start()

    def _exact_done(self, gen: int, res: dict) -> None:
        if self.last_result is None or gen != self.last_result.gen or not isinstance(res, dict):
            return
        self.exact, self.exact_gen = res, gen
        self._update_stats()
        self.relief_step.update_budget(self._budget_with_exact(self.last_result.budget))
        self.prod_step.update_estimate()

    def _budget_with_exact(self, b: dict) -> dict:
        b = dict(b or {})
        ex = self.exact_parts_current()
        if ex is not None and b.get("alan_m2"):
            b["parca_m2"] = ex["parca"] / b["alan_m2"]
            b["parca_kesin"] = True
            if ex.get("hassas") is not None:
                b["hassas"] = ex["hassas"]
                b["hassas_oran"] = ex["hassas"] / max(1, ex["parca"])
        return b

    def export_warnings(self) -> list[str]:
        """Plain-language warnings before an export (R3-1, Ö2-10)."""
        from .engine import HASSAS_WARN_SHARE, SMALL_WIDTH_MM
        s = self.project.settings
        out = []
        if s.mode == Mode.PIECES:
            ex = self.exact_parts_current()
            if ex and ex.get("hassas") is not None and ex["parca"]:
                share = ex["hassas"] / ex["parca"]
                if share > HASSAS_WARN_SHARE:
                    out.append(f"Bu görselde {tr_num(ex['hassas'])} hassas parça çıkacak (parçaların %{tr_num(share * 100)}'i). "
                               "Hassas parça, üstüne numara sığmayacak kadar küçük (12–30 mm²) bir parçadır; "
                               "cımbızla, kılavuza bakarak yerleştirilir. Azaltmak için eseri büyüt, taşları büyüt "
                               "ya da Çok renkli paneller (AMS) modunu seç.")
            if s.width_mm < SMALL_WIDTH_MM:
                out.append(f"Eser küçük ({tr_num(s.width_mm / 10, 1)} cm). Bu boyutta parçalar çok küçük olur ve "
                           "birçoğuna numara sığmaz. 25 cm'den küçük eserlerde Çok renkli paneller (AMS) daha uygun.")
        return out

    def _update_budget_chip(self, b: dict) -> None:
        from .engine import budget_level
        from .views import level_colour
        w = self.stat_items["budget"]
        kg = (b or {}).get("kg_m2")
        if kg is None:
            w.set("—")
            return
        lv = budget_level("kg", kg)
        w.set(f"{tr_num(kg, 1)} kg/m²", "Metrekare başına ağırlık: 3'e kadar yeşil, 4'e kadar sarı. "
                                        "Ayrıntı için tıkla.")
        w.value.setStyleSheet(f"color: {level_colour(lv)};")
        w.caption.setText({"ok": "ağırlık · uygun", "warn": "ağırlık · sınırda", "bad": "ağırlık · çok ağır"}[lv])

    def _refresh_plates_tab(self) -> None:
        from .views import collect_export_images
        items = collect_export_images(self.last_export or {})
        self.gallery.set_items(items)
        self.tabs.set_shown(TAB_PLATES, bool(items))

    def _refresh_3d_tab(self) -> None:
        from .views import preview_parts_fn
        self.tabs.set_shown(TAB_3D, self.has_image() and preview_parts_fn() is not None)

    def _load_panel3d(self, panel: int) -> None:
        from .views import call_preview_parts, preview_parts_fn, preview_result
        fn = preview_parts_fn()
        r = self.last_result
        if fn is None or r is None:
            self.panel3d.set_meshes([], "Önizleme hazır olunca panel burada görünecek.")
            return
        e = self.current_estimate() or {}
        self.panel3d.set_panel_count(int(e.get("panel") or 1))
        prev = getattr(self, "_panel3d_task", None)
        if prev is not None and prev.isRunning():
            self._panel3d_pending = panel          # one request at a time; newest wins
            return
        self._panel3d_pending = None
        self._panel3d_stale = False
        try:
            from ..export.api import preview_key
            self._panel3d_key = preview_key(r.mosaic)
        except Exception:  # noqa: BLE001
            self._panel3d_key = r.gen
        if not self.panel3d.has_content():
            self.panel3d.set_meshes([], "Panel hazırlanıyor… Parçalar gerçek geometriyle kuruluyor; "
                                        "ilk açılış birkaç saniye sürebilir.")
        self.panel3d.set_busy(True)
        m = r.mosaic
        az, el = self.panel3d.yaw, self.panel3d.pitch
        quick = quick_kw(fn)

        def job(progress):
            if quick:     # Ö2-8: a fast rough view first, then the full one
                progress(0.5, "")
                self._panel3d_quick = preview_result(call_preview_parts(fn, m, panel, az, el, extra=quick))
                t.progress.emit(0.6, "kaba")
            return preview_result(call_preview_parts(fn, m, panel, az, el))

        t = TaskThread(job, self)

        def show(res):
            kind, data = res
            if kind == "image":
                self.panel3d.set_image(data)
            else:
                self.panel3d.set_meshes(data, "Bu panelde gösterilecek parça yok.")

        def step(f, msg):
            if msg == "kaba" and getattr(self, "_panel3d_quick", None):
                show(self._panel3d_quick)

        def done(res):
            self.panel3d.set_busy(False)
            show(res)
            if self._panel3d_pending is not None:
                self._load_panel3d(self._panel3d_pending)

        def failed(msg):
            self.panel3d.set_busy(False)
            self.panel3d.set_meshes([], msg)

        t.progress.connect(step)
        t.done.connect(done)
        t.error.connect(failed)
        self._panel3d_task = t
        t.start()

    def _export_done(self, res: dict) -> None:
        self.last_export = res if isinstance(res, dict) else {}
        self._refresh_plates_tab()
        self.export_step.set_running(False)
        self.export_step.show_result(self.last_export)
        self._update_rail()
        self._flash("Dışa aktarım bitti.")

    def _export_error(self, msg: str) -> None:
        self.export_step.set_running(False)
        self.export_step.show_error(msg)
        self._update_status()

    BAMBU_EXE = r"C:\Program Files\Bambu Studio\bambu-studio.exe"

    def bambu_target(self) -> str:
        """First 3MF of the last export, if Bambu Studio is installed (G-12)."""
        if not os.path.exists(self.BAMBU_EXE):
            return ""
        for f in (self.last_export or {}).get("files") or []:
            if isinstance(f, str) and f.lower().endswith(".3mf") and os.path.exists(f):
                return f
        return ""

    def open_in_bambu(self) -> None:
        from PySide6.QtCore import QProcess
        f = self.bambu_target()
        if f:
            QProcess.startDetached(self.BAMBU_EXE, [f])

    def open_export_folder(self) -> None:
        d = self.project.export_dir or self.default_export_dir()
        if os.path.isdir(d):
            QDesktopServices.openUrl(QUrl.fromLocalFile(d))

    # ------------------------------------------------------------------ help
    def show_shortcuts(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Kısayollar")
        v = QVBoxLayout(dlg)
        v.setContentsMargins(24, 20, 24, 20)
        v.addWidget(label("Kısayollar", "stepTitle"))
        g = QGridLayout()
        g.setHorizontalSpacing(24)
        g.setVerticalSpacing(6)
        rows = [("Ctrl+N / O / S", "Yeni / Aç / Kaydet"), ("Ctrl+Shift+S", "Farklı kaydet"),
                ("Ctrl+1 … 6", "Adımlar"), ("Alt+1 … 5", "Önizleme sekmeleri"),
                ("Ctrl+E", "Dışa aktar"), ("Ctrl+T", "Açık / koyu tema"),
                ("Tekerlek · F", "Yakınlaştır · sığdır"), ("Boşluk + sürükle", "Kaydır"),
                ("H B E P R C M", "Kaydır, fırça, silgi, çokgen, dikdörtgen, daire, ışın merkezi"),
                ("1 … 6", "Boyanacak taş tipi (Bölgeler sekmesinde)"), ("[  ]", "Fırça boyu"),
                ("Ctrl+Z / Ctrl+Y", "Geri al / yinele"), ("Enter · Esc", "Çokgeni kapat · vazgeç"),
                ("Delete", "Seçili şekli sil")]
        for i, (k, d) in enumerate(rows):
            kl = label(k, "value")
            g.addWidget(kl, i, 0)
            g.addWidget(label(d, "hint"), i, 1)
        v.addLayout(g)
        b = QPushButton("Tamam")
        b.clicked.connect(dlg.accept)
        v.addWidget(b, 0, Qt.AlignRight)
        dlg.exec()

    # ------------------------------------------------------------------ window events
    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.open_any(paths[0])

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        narrow = self.width() < 1320           # keep all five tabs readable on small screens
        for w in (self.z_out, self.z_in, self.zoom_lbl):
            w.setVisible(not narrow)

    def closeEvent(self, e) -> None:
        if not self._confirm_discard():
            e.ignore()
            return
        self.shutdown()
        e.accept()

    def shutdown(self) -> None:
        self.engine.stop()
        if isinstance(self._export_task, ExportJob) and self._export_task.isRunning():
            self._export_task.cancel()
        if self._count_job is not None and self._count_job.isRunning():
            self._count_job.wait(60000)
        for t in (self._export_task, self._suggest_task):
            if t is not None and t.isRunning():
                t.wait(60000)
