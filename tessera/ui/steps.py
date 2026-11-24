"""The six step panels shown on the right side of the window."""
from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QSpinBox, QToolButton, QVBoxLayout,
                               QWidget)

from ..model import VOCAB_NAMES_TR, Mode, Vocab
from . import icons
from .theme import VOCAB_COLOURS, VOCAB_HINTS_TR, current, head_font
from .widgets import (ClickFrame, Dot, FilamentChip, LightDial, PaletteStrip, SliderRow, label,
                      section, tr_num, tr_upper)
from .filaments import find_preset, load_presets, preset_label
from .widgets import tr_lower


def current_names(text: str) -> str:
    """Old stone-type names (from older algorithm messages) -> the names in VOCAB_NAMES_TR."""
    old = {"Piksel bloklar": Vocab.BLOCKS, "Piksel blok": Vocab.BLOCKS, "Kare piksel bloklar": Vocab.BLOCKS,
           "Işın şeritleri": Vocab.RAYS, "Figür (kabartma)": Vocab.FIGURE, "Kesik kâğıt": Vocab.FLAT,
           "Taşlar:": Vocab.STONES, "Odak:": Vocab.FOCAL}
    for k in sorted(old, key=len, reverse=True):
        new = VOCAB_NAMES_TR[old[k]] + (":" if k.endswith(":") else "")
        text = text.replace(k, new)
    # lower-case phrases used inside sentences (e.g. Art-team preset descriptions)
    for k, v in (("ışın şeritli", "ışın tahtalı"), ("ışın şeritleri", "ışın tahtaları"),
                 ("piksel bloklar", "kare bloklar"), ("piksel blok", "kare blok"), ("piksel gökyüzü", "kare blok gökyüzü"),
                 ("kesik kâğıt", "kesik levha"), ("Monumental", "Anıtsal")):
        text = text.replace(k, v)
    return text
from .zones import shape_kind_tr

if TYPE_CHECKING:
    from .main_window import MainWindow

VOCAB_ORDER = [int(Vocab.STONES), int(Vocab.RAYS), int(Vocab.BLOCKS), int(Vocab.FIGURE),
               int(Vocab.FOCAL), int(Vocab.FLAT)]


def light_text(az: float, el: float) -> str:
    dirs = ["sağdan", "sağ alttan", "alttan", "sol alttan", "soldan", "sol üstten", "üstten", "sağ üstten"]
    k = int(round(((az % 360) / 45))) % 8
    height = "tepeden" if el >= 65 else ("yüksekten" if el >= 45 else ("alçaktan" if el >= 25 else "yandan, sıyırarak"))
    return f"Işık {dirs[k]} · {height} ({int(round(el))}°)"


class StepPanel(QScrollArea):
    """Scrollable panel with an eyebrow / title / lead header."""

    def __init__(self, win: "MainWindow", index: int, title: str, lead: str):
        super().__init__()
        self.win = win
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        inner.setObjectName("panelInner")
        self.setWidget(inner)
        self.body = QVBoxLayout(inner)
        self.body.setContentsMargins(26, 24, 24, 28)
        self.body.setSpacing(14)
        self.body.addWidget(label(f"ADIM {index} / 6", "stepEyebrow"))
        t = label(title, "stepTitle")
        self.body.addWidget(t)
        ld = label(lead, "stepLead", wrap=True)
        self.body.addWidget(ld)
        self.body.addSpacing(6)
        self.advanced: list[QWidget] = []

    def finish(self) -> None:
        self.body.addStretch(1)

    def add_advanced(self, w: QWidget) -> QWidget:
        self.advanced.append(w)
        self.body.addWidget(w)
        return w

    def set_advanced(self, on: bool) -> None:
        for w in self.advanced:
            w.setVisible(on)

    def refresh(self) -> None:  # sync from project
        pass

    def retheme(self) -> None:
        pass


def card(parent_layout=None, flat: bool = False, margins=(16, 14, 16, 14), spacing=10):
    f = QFrame()
    f.setObjectName("cardFlat" if flat else "card")
    v = QVBoxLayout(f)
    v.setContentsMargins(*margins)
    v.setSpacing(spacing)
    if parent_layout is not None:
        parent_layout.addWidget(f)
    return f, v


def advanced_box(title: str = "Gelişmiş") -> tuple[QWidget, QVBoxLayout]:
    w = QWidget()
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 8, 0, 0)
    v.setSpacing(12)
    row = QHBoxLayout()
    row.setSpacing(8)
    s = section(title)
    row.addWidget(s)
    line = QFrame()
    line.setFixedHeight(1)
    line.setObjectName("advLine")
    line.setStyleSheet(f"background: {current().line};")
    row.addWidget(line, 1)
    v.addLayout(row)
    return w, v


# ====================================================================== 1 Görsel
class ImageStep(StepPanel):
    def __init__(self, win):
        super().__init__(win, 1, "Görsel",
                         "Mozaiğe dönüşecek resmi seç ve duvardaki gerçek boyutunu belirle.")
        # drop / file card
        self.drop, dv = card(self.body, flat=True, margins=(18, 22, 18, 20), spacing=8)
        self.drop.setMinimumHeight(150)
        self.drop_icon = QLabel()
        self.drop_icon.setAlignment(Qt.AlignCenter)
        dv.addWidget(self.drop_icon)
        t = label("Görseli buraya sürükle", "modeTitle")
        t.setAlignment(Qt.AlignCenter)
        dv.addWidget(t)
        h = label("JPG, PNG ya da WEBP · net, iri şekilli görseller en iyi sonucu verir", "hint", wrap=True)
        h.setAlignment(Qt.AlignCenter)
        dv.addWidget(h)
        row = QHBoxLayout()
        row.addStretch(1)
        self.pick = QPushButton("Görsel seç…")
        self.pick.clicked.connect(win.choose_image)
        row.addWidget(self.pick)
        row.addStretch(1)
        dv.addLayout(row)

        # file info card
        self.info, iv = card(self.body, margins=(12, 12, 14, 12))
        ih = QHBoxLayout()
        ih.setSpacing(12)
        self.thumb = QLabel()
        self.thumb.setFixedSize(64, 64)
        self.thumb.setAlignment(Qt.AlignCenter)
        ih.addWidget(self.thumb)
        tv = QVBoxLayout()
        tv.setSpacing(2)
        self.fname = label("", "modeTitle")
        self.fmeta = label("", "hint")
        tv.addWidget(self.fname)
        tv.addWidget(self.fmeta)
        ch = QPushButton("Değiştir…")
        ch.setObjectName("link")
        ch.setCursor(Qt.PointingHandCursor)
        ch.setMinimumWidth(ch.fontMetrics().horizontalAdvance(ch.text()) + 16)   # G2-5: never clipped
        ch.clicked.connect(win.choose_image)
        tv.addWidget(ch, 0, Qt.AlignLeft)
        ih.addLayout(tv, 1)
        iv.addLayout(ih)

        # K-3: project image missing -> relink card
        self.missing, mv = card(self.body, margins=(14, 12, 14, 12))
        self.missing.setObjectName("warnBox")
        mv.addWidget(label("Görsel bulunamadı", "modeTitle"))
        self.missing_path = label("", "hint", wrap=True)
        mv.addWidget(self.missing_path)
        mv.addWidget(label("Bölgeler, renkler ve ayarlar korunuyor. Kaydedersen eski yol da korunur.",
                           "hint", wrap=True))
        self.relink = QPushButton("Görseli bul…")
        self.relink.setObjectName("primary")
        self.relink.setCursor(Qt.PointingHandCursor)
        self.relink.clicked.connect(win.choose_image)
        mv.addWidget(self.relink, 0, Qt.AlignLeft)
        self.missing.hide()

        self.example = QPushButton("  Örnek proje: Sovyet afişi")
        self.example.clicked.connect(win.load_example)
        self.body.addWidget(self.example)

        # size
        self.body.addSpacing(8)
        self.body.addWidget(section("Duvardaki boyut"))
        sc, sv = card(self.body)
        g = QGridLayout()
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(4)
        g.addWidget(label("Genişlik", "hint"), 0, 0)
        g.addWidget(label("Yükseklik", "hint"), 0, 2)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setObjectName("bigSpin")
        self.width_spin.setRange(10, 300)
        self.width_spin.setDecimals(1)
        self.width_spin.setSingleStep(1)
        self.width_spin.setSuffix(" cm")
        self.width_spin.setKeyboardTracking(False)
        self.width_spin.valueChanged.connect(self._width_changed)
        g.addWidget(self.width_spin, 1, 0)
        g.addWidget(label("×", "bigValue"), 1, 1)
        self.height_lbl = label("—", "bigValue")
        g.addWidget(self.height_lbl, 1, 2)
        g.setColumnStretch(0, 1)
        g.setColumnStretch(2, 1)
        sv.addLayout(g)
        chips = QHBoxLayout()
        chips.setSpacing(6)
        self.presets = []
        for cm in (30, 42.4, 60, 90):
            b = QPushButton(f"{tr_num(cm, 0 if cm == int(cm) else 1)} cm")
            b.setObjectName("chip")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, v=cm: self.width_spin.setValue(v))
            chips.addWidget(b)
            self.presets.append(b)
        chips.addStretch(1)
        sv.addLayout(chips)
        self.size_hint = label("", "hint", wrap=True)
        sv.addWidget(self.size_hint)
        self.big_warn, bv = card(self.body, margins=(14, 10, 14, 10), spacing=4)
        self.big_warn.setObjectName("warnBox")
        bv.addWidget(label("Çok büyük bir eser", "modeTitle"))
        self.big_text = label("", "hint", wrap=True)
        bv.addWidget(self.big_text)
        self.big_warn.hide()
        self.small_warn, smv = card(self.body, margins=(14, 10, 14, 10), spacing=4)
        self.small_warn.setObjectName("warnBox")
        smv.addWidget(label("Küçük bir eser", "modeTitle"))
        smv.addWidget(label("25 cm'den küçük eserlerde taşlar çok küçülür; birçok parçaya numara sığmaz ve "
                            "cımbızla yerleştirilmesi gerekir. Bu boyutta Çok renkli paneller (AMS) daha uygun; "
                            "ya da eseri büyüt.", "hint", wrap=True))
        self.small_warn.hide()
        self.finish()

    def retheme(self) -> None:
        t = current()
        self.drop_icon.setPixmap(icons.pixmap("upload", t.faint, 34))
        self.example.setIcon(icons.icon("magic", t.gold, 18))

    BIG_CM = 150.0

    def _width_changed(self, v: float) -> None:
        old = self.win.project.settings.width_mm / 10
        if v >= self.BIG_CM > old and not self.win.confirm_big(v):
            self.width_spin.blockSignals(True)
            self.width_spin.setValue(old)
            self.width_spin.blockSignals(False)
            return
        self.win.set_width_mm(v * 10)
        self._update_height()

    def _update_height(self) -> None:
        w_cm = self.width_spin.value()
        a = self.win.aspect()
        self.height_lbl.setText(f"{tr_num(w_cm * a, 1)} cm" if a else "—")
        parts = ["Yükseklik görselin oranından gelir."]
        e = self.win.current_estimate()
        if e and e.get("panel"):
            parts.append(f"Bu boyutta eser {tr_num(e['panel'])} panele bölünür.")
        img_w = self.win.image_px[0]
        if img_w and w_cm * 10 / img_w > 0.8:
            parts.append(f"Görsel bu boyut için küçük ({tr_num(w_cm * 10 / img_w, 1)} mm/piksel); "
                         "kenarlar yumuşak çıkar.")
        if self.win.image_note:
            parts.append(self.win.image_note)
        self.size_hint.setText(" ".join(parts))
        big = w_cm >= self.BIG_CM
        self.big_warn.setVisible(big)
        self.small_warn.setVisible(bool(a) and w_cm < 25 and self.win.project.settings.mode == Mode.PIECES)
        if big:
            from .engine import preview_mm_per_px
            txt = (f"{tr_num(w_cm, 0)} cm genişlik. Önizleme daha kaba hesaplanır "
                   f"({tr_num(preview_mm_per_px(w_cm * 10), 1)} mm/piksel) ve her değişiklik daha uzun sürer. "
                   "Dışa aktarım dakikalar sürer ve çok bellek ister.")
            if e and e.get("gram") is not None:
                txt += (f" Tahmin: ≈ {tr_num(e['gram'] / 1000, 1)} kg, ~{tr_num(e.get('saat') or 0)} saat baskı, "
                        f"{tr_num(e.get('tabla') or 0)} tabla.")
            self.big_text.setText(txt)

    def refresh(self) -> None:
        p = self.win.project
        has = self.win.has_image()
        missing = self.win.image_missing
        self.missing.setVisible(missing)
        if missing:
            self.missing_path.setText(os.path.normpath(p.image_path))
        self.drop.setVisible(not has and not missing)
        self.info.setVisible(has)
        self.example.setVisible(not has and not missing)
        if has:
            self.fname.setText(os.path.basename(p.image_path))
            w, h = self.win.image_px
            self.fmeta.setText(f"{w} × {h} piksel")
            if self.win.original_qimage is not None:
                pm = QPixmap.fromImage(self.win.original_qimage).scaled(
                    64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.thumb.setPixmap(pm)
        self.width_spin.blockSignals(True)
        self.width_spin.setValue(p.settings.width_mm / 10)
        self.width_spin.blockSignals(False)
        self._update_height()


# ====================================================================== 2 Filamentler
class FilamentStep(StepPanel):
    COLS = 4

    def __init__(self, win):
        super().__init__(win, 2, "Filamentler",
                         "Mozaik yalnızca bu renklerden oluşur. Her renk bir filament makarası demek.")
        top = QHBoxLayout()
        top.addWidget(section("Paletin"))
        top.addStretch(1)
        self.count = label("", "hint")
        top.addWidget(self.count)
        self.body.addLayout(top)
        self.strip = PaletteStrip()
        self.strip.remove_requested.connect(self._remove_at)
        self.body.addWidget(self.strip)
        self.roles_hint = label("Renklerin rolü: yapı = çizgiler ve gölgeler (en koyu) · ışık = en açık yerler · "
                                "vurgu = en canlı renk · alan = geniş yüzeyler. Bir renge tıklarsan paletten çıkar.",
                                "hint", wrap=True)
        self.body.addWidget(self.roles_hint)

        sug = QHBoxLayout()
        sug.setSpacing(8)
        self.suggest = QPushButton("  Bu görsel için öner")
        self.suggest.setToolTip("Görselin renklerine en yakın filamentleri seçer (öncelik: sahip oldukların).")
        self.suggest.clicked.connect(win.suggest_palette)
        sug.addWidget(self.suggest, 1)
        self.n_spin = QSpinBox()
        self.n_spin.setRange(2, 8)
        self.n_spin.setValue(4)
        self.n_spin.setSuffix(" renk")
        self.n_spin.setFixedWidth(78)
        self.n_spin.setAlignment(Qt.AlignCenter)
        self.n_spin.setToolTip("Önerilecek renk sayısı")
        sug.addWidget(self.n_spin)
        self.body.addLayout(sug)

        self.body.addSpacing(8)
        lib_head = QHBoxLayout()
        lib_head.addWidget(section("Filament kütüphanesi"))
        lib_head.addStretch(1)
        self.body.addLayout(lib_head)
        filt = QHBoxLayout()
        filt.setSpacing(8)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Renk ara…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        filt.addWidget(self.search, 1)
        self.only_owned = QCheckBox("Sahip olduklarım")
        self.only_owned.toggled.connect(self._filter)
        filt.addWidget(self.only_owned)
        self.body.addLayout(filt)
        filt = QHBoxLayout()
        filt.addWidget(label("Seri:", "hint"))
        self.series = QComboBox()
        self.series.currentIndexChanged.connect(self._filter)
        filt.addWidget(self.series, 1)
        self.body.addLayout(filt)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 2, 0, 0)
        self.grid.setHorizontalSpacing(4)
        self.grid.setVerticalSpacing(2)
        self.body.addWidget(self.grid_host)
        self.src = label("", "hint", wrap=True)
        self.body.addWidget(self.src)
        self.note = label("", "hint", wrap=True)
        self.note.setStyleSheet(f"color: {current().warn};")
        self.note.hide()
        self.body.insertWidget(self.body.indexOf(self.strip) + 1, self.note)
        self.chips: list[FilamentChip] = []
        self.finish()

    def retheme(self) -> None:
        self.suggest.setIcon(icons.icon("magic", current().gold, 18))
        for c in self.chips:
            c.update()
        self.strip.update()

    def build_library(self) -> None:
        for c in self.chips:
            c.setParent(None)
            c.deleteLater()
        self.chips = []
        for f in self.win.library:
            c = FilamentChip(f)
            c.toggled.connect(self.win.toggle_palette)
            c.owned_toggled.connect(self.win.owned_changed)
            self.chips.append(c)
        from .filaments import FINISH_TR
        self.series.blockSignals(True)
        self.series.clear()
        self.series.addItem("Tüm seriler", "")
        for fin in sorted({f.finish for f in self.win.library}):
            self.series.addItem(f"{FINISH_TR.get(fin, fin)} PLA", fin)
        self.series.blockSignals(False)
        self.src.setText("Bambu Lab PLA renkleri (Standart ve Mat seri). Yıldız = bende var." if self.win.library_from_file else
                         "Şimdilik yerleşik kısa liste kullanılıyor; renkler yaklaşık. "
                         "Yıldız = bende var.")
        self._filter()

    def _filter(self) -> None:
        q = self.search.text().strip().lower()
        only = self.only_owned.isChecked()
        fin = self.series.currentData() if hasattr(self, "series") else ""
        for c in self.chips:
            self.grid.removeWidget(c)
            c.hide()
        k = 0
        for c in self.chips:
            if q and q not in c.fil.name.lower():
                continue
            if only and not c.fil.owned:
                continue
            if fin and c.fil.finish != fin:
                continue
            self.grid.addWidget(c, k // self.COLS, k % self.COLS)
            c.show()
            k += 1

    def show_note(self, text: str) -> None:
        self.note.setStyleSheet(f"color: {current().warn};")
        self.note.setText(text)
        self.note.setVisible(bool(text))

    def _remove_at(self, i: int) -> None:
        pal = self.win.project.settings.palette
        if 0 <= i < len(pal):
            self.win.toggle_palette(pal[i])

    def refresh(self) -> None:
        pal = self.win.project.settings.palette
        names = {f.name for f in pal}
        for c in self.chips:
            c.in_palette = c.fil.name in names
            c.update()
        roles = {k: v for k, v in self.win.palette_roles.items() if k in {f.name for f in pal}}
        self.strip.set_palette(pal, roles)
        self.roles_hint.setVisible(bool(roles))
        n = len(pal)
        self.count.setText(f"{n} renk" + (" · en az 2 gerekli" if n < 2 else ""))
        self.n_spin.setValue(max(2, min(8, n or 4)))


# ====================================================================== 3 Bölgeler
class VocabRow(ClickFrame):
    def __init__(self, v: int):
        super().__init__()
        self.v = v
        self.setObjectName("vocabRow")
        self.setCursor(Qt.PointingHandCursor)
        h = QHBoxLayout(self)
        h.setContentsMargins(10, 7, 10, 7)
        h.setSpacing(10)
        self.dot = Dot(VOCAB_COLOURS[v], 14)
        h.addWidget(self.dot)
        tv = QVBoxLayout()
        tv.setSpacing(0)
        self.name = label(VOCAB_NAMES_TR[Vocab(v)])
        self.hint = label(VOCAB_HINTS_TR[v], "hint")
        self.hint.setWordWrap(True)
        tv.addWidget(self.name)
        tv.addWidget(self.hint)
        h.addLayout(tv, 1)
        self.key = label(str(VOCAB_ORDER.index(v) + 1), "hint")
        h.addWidget(self.key)
        self.set_selected(False)

    def set_selected(self, on: bool) -> None:
        t = current()
        self.setStyleSheet(
            f"#vocabRow {{ background: {t.card if on else 'transparent'}; border: 1px solid "
            f"{t.red if on else 'transparent'}; border-radius: 8px; }}"
            f"#vocabRow:hover {{ background: {t.card}; }}")
        f = self.name.font()
        f.setBold(on)
        self.name.setFont(f)


class ZonesStep(StepPanel):
    TOOLS = [("pan", "hand", "Kaydır (H) — boşlukla da kaydırabilirsin"),
             ("brush", "brush", "Fırça (B)"),
             ("eraser", "eraser", "Silgi (E) — boyanan yeri “geri kalan her yer” taş tipine döndürür"),
             ("polygon", "polygon", "Çokgen (P) — tıkla tıkla, ilk noktaya tıklayınca ya da Enter ile kapanır"),
             ("rect", "rect", "Dikdörtgen (R) — sürükle"),
             ("circle", "circle", "Daire (C) — merkezden dışa sürükle"),
             ("center", "target", f"Işın merkezi (M) — {tr_lower(VOCAB_NAMES_TR[Vocab.RAYS])} taşlarının "
                                  "yayıldığı noktaya tıkla")]

    def __init__(self, win):
        super().__init__(win, 3, "Bölgeler",
                         "Resmin her bölümüne bir taş tipi ver: gökyüzüne "
                         f"{tr_lower(VOCAB_NAMES_TR[Vocab.RAYS])}, binalara {tr_lower(VOCAB_NAMES_TR[Vocab.BLOCKS])}, "
                         f"yüzlere {tr_lower(VOCAB_NAMES_TR[Vocab.FIGURE])}.")
        self.body.addWidget(section("Araçlar"))
        tools = QHBoxLayout()
        tools.setSpacing(5)
        self.tool_btns: dict[str, QToolButton] = {}
        for key, ic, tip in self.TOOLS:
            b = QToolButton()
            b.setObjectName("tool")
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setIconSize(QSize(20, 20))
            b.setFixedSize(38, 38)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, k=key: win.set_tool(k))
            tools.addWidget(b)
            self.tool_btns[key] = b
        tools.addStretch(1)
        self.body.addLayout(tools)
        self.brush = SliderRow("Fırça boyu", 0.5, 12, 0.5, 2.0, "cm", 1)
        self.brush.value_changed.connect(lambda v: win.canvas.set_brush_mm(v * 10))
        self.body.addWidget(self.brush)
        self.tool_hint = label("", "hint", wrap=True)
        self.body.addWidget(self.tool_hint)

        self.auto_btn = QPushButton("  Otomatik öner")
        self.auto_btn.setToolTip("Program görsele bakıp bölgeleri önerir; sonra fırçayla düzeltirsin.")
        self.auto_btn.clicked.connect(win.suggest_zones)
        self.body.addWidget(self.auto_btn)
        self.reasons = label("", "hint", wrap=True)
        self.reasons.hide()
        self.body.addWidget(self.reasons)
        self.optional = label("", "hint", wrap=True)
        self.optional.setStyleSheet(f"color: {current().gold};")
        self.body.insertWidget(self.body.indexOf(self.auto_btn), self.optional)

        self.body.addSpacing(4)
        self.body.addWidget(section("Boyanacak taş tipi"))
        self.rows: dict[int, VocabRow] = {}
        lst = QVBoxLayout()
        lst.setSpacing(2)
        for v in VOCAB_ORDER:
            r = VocabRow(v)
            r.clicked.connect(lambda v=v: win.set_paint_vocab(v))
            lst.addWidget(r)
            self.rows[v] = r
        self.body.addLayout(lst)

        dr = QHBoxLayout()
        dr.addWidget(label("Geri kalan her yer:"))
        self.default_combo = QComboBox()
        for v in VOCAB_ORDER:
            self.default_combo.addItem(VOCAB_NAMES_TR[Vocab(v)], v)
        self.default_combo.currentIndexChanged.connect(
            lambda i: win.zones.set_default_vocab(self.default_combo.itemData(i)))
        dr.addWidget(self.default_combo, 1)
        self.body.addLayout(dr)

        self.body.addSpacing(4)
        sh = QHBoxLayout()
        sh.addWidget(section("Çizilen şekiller"))
        sh.addStretch(1)
        self.undo_btn = QToolButton()
        self.undo_btn.setObjectName("ghost")
        self.undo_btn.setToolTip("Geri al (Ctrl+Z)")
        self.undo_btn.clicked.connect(win.zones.undo)
        self.redo_btn = QToolButton()
        self.redo_btn.setObjectName("ghost")
        self.redo_btn.setToolTip("Yinele (Ctrl+Y)")
        self.redo_btn.clicked.connect(win.zones.redo)
        sh.addWidget(self.undo_btn)
        sh.addWidget(self.redo_btn)
        self.body.addLayout(sh)
        self.shape_list = QListWidget()
        self.shape_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.shape_list.setMinimumHeight(120)
        self.shape_list.currentRowChanged.connect(self._select_shape)
        self.body.addWidget(self.shape_list)
        br = QHBoxLayout()
        br.setSpacing(6)
        self.del_btn = QPushButton("Seçileni sil")
        self.del_btn.clicked.connect(self.delete_selected)
        self.clear_paint_btn = QPushButton("Fırça izlerini temizle")
        self.clear_paint_btn.clicked.connect(win.zones.clear_paint)
        br.addWidget(self.del_btn)
        br.addWidget(self.clear_paint_btn)
        self.body.addLayout(br)

        adv, av = advanced_box()
        self.opacity = SliderRow("Bölge renklerinin görünürlüğü", 10, 90, 5, 55, "%", 0)
        self.opacity.value_changed.connect(lambda v: win.canvas.overlay.setOpacity(v / 100))
        av.addWidget(self.opacity)
        clr = QPushButton("Tüm bölgeleri sil")
        clr.clicked.connect(win.zones.clear_all)
        av.addWidget(clr)
        self.add_advanced(adv)
        self.finish()

    def retheme(self) -> None:
        t = current()
        for key, ic, _ in self.TOOLS:
            self.tool_btns[key].setIcon(icons.icon(ic, t.text, 20, t.on_red))
        self.undo_btn.setIcon(icons.icon("undo", t.text, 18))
        self.redo_btn.setIcon(icons.icon("redo", t.text, 18))
        self.auto_btn.setIcon(icons.icon("magic", t.gold, 18))
        for v, r in self.rows.items():
            r.set_selected(v == self.win.canvas.paint_vocab)

    def set_tool(self, tool: str) -> None:
        for k, b in self.tool_btns.items():
            b.setChecked(k == tool)
        self.brush.setVisible(tool in ("brush", "eraser"))
        self.tool_hint.setText({
            "pan": "Sürükleyerek kaydır, tekerlekle yakınlaştır. Çift tık: ekrana sığdır.",
            "brush": "Seçili dille boya. Alt + tekerlek: fırça boyu. [ ve ] tuşları da olur.",
            "eraser": "Boyanan yeri “geri kalan her yer” taş tipine döndürür.",
            "polygon": "Köşelere tıkla. İlk noktaya tıkla, çift tıkla ya da Enter: kapat. Esc: vazgeç.",
            "rect": "Bir köşeden karşı köşeye sürükle.",
            "circle": "Merkezden dışarı doğru sürükle.",
            "center": "Işın şeritlerinin yayılacağı noktaya tıkla (genelde güneş).",
        }.get(tool, ""))

    def show_reasons(self, reasons: list[str]) -> None:
        self.reasons.setText("\n".join("• " + current_names(r) for r in reasons[:6]))
        self.reasons.setVisible(bool(reasons))

    def set_paint_vocab(self, v: int) -> None:
        for k, r in self.rows.items():
            r.set_selected(k == v)

    def refresh(self) -> None:
        z = self.win.zones
        self.default_combo.blockSignals(True)
        self.default_combo.setCurrentIndex(VOCAB_ORDER.index(z.default_vocab))
        self.default_combo.blockSignals(False)
        row = self.shape_list.currentRow()
        self.shape_list.blockSignals(True)
        self.shape_list.clear()
        for i, s in enumerate(z.shapes):
            v = int(s.get("vocab", 0))
            name = s.get("ad", "")
            it = QListWidgetItem(f"{VOCAB_NAMES_TR[Vocab(v)]} · {shape_kind_tr(s)}" + (f" · {name}" if name else ""))
            pm = QPixmap(12, 12)
            pm.fill(Qt.transparent)
            from PySide6.QtGui import QColor, QPainter
            p = QPainter(pm)
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(VOCAB_COLOURS[v]))
            p.drawRoundedRect(0, 0, 12, 12, 2.5, 2.5)
            p.end()
            it.setIcon(pm)
            self.shape_list.addItem(it)
        if not z.shapes:
            it = QListWidgetItem("Henüz şekil yok. Bir araç seçip resmin üstüne çiz.")
            it.setFlags(Qt.NoItemFlags)
            self.shape_list.addItem(it)
        self.shape_list.blockSignals(False)
        if 0 <= row < len(z.shapes):
            self.shape_list.setCurrentRow(row)
        else:
            self.win.canvas.highlight_shape(None)
        self.undo_btn.setEnabled(z.can_undo())
        self.redo_btn.setEnabled(z.can_redo())
        self.del_btn.setEnabled(bool(z.shapes))
        self.clear_paint_btn.setEnabled(z.has_paint())
        self.auto_btn.setVisible(self.win.suggest_zones_available())
        empty = not z.shapes and not z.has_paint()
        name = VOCAB_NAMES_TR[Vocab(z.default_vocab)]
        self.optional.setText(
            f"Şimdilik her yer “{name}”. Bu adım isteğe bağlı: istersen "
            + ("“Otomatik öner”e bas ya da " if self.win.suggest_zones_available() else "")
            + "bir taş tipi seçip resmin üstüne boya. Hiçbir şey yapmadan da devam edebilirsin.")
        self.optional.setVisible(empty)

    def _select_shape(self, row: int) -> None:
        z = self.win.zones
        self.win.canvas.highlight_shape(z.shapes[row] if 0 <= row < len(z.shapes) else None)

    def delete_selected(self) -> None:
        row = self.shape_list.currentRow()
        if 0 <= row < len(self.win.zones.shapes):
            self.win.zones.remove_shape(row)


# ====================================================================== 4 Kabartma
class ReliefStep(StepPanel):
    def __init__(self, win):
        super().__init__(win, 4, "Kabartma",
                         "Taşların boyunu ve ne kadar öne çıkacaklarını ayarla. "
                         "Ton ve canlılık renkten değil, ışığın kabartmaya vurmasından gelir.")
        self.presets = load_presets()
        self.preset_btns: list[QPushButton] = []
        if self.presets:
            self.body.addWidget(section("Hazır ayar"))
            pg = QGridLayout()
            pg.setHorizontalSpacing(6)
            pg.setVerticalSpacing(6)
            for i, pr in enumerate(self.presets):
                b = QPushButton(preset_label(pr))
                b.setObjectName("chip")
                b.setCheckable(True)
                b.setCursor(Qt.PointingHandCursor)
                b.setToolTip(current_names(pr.get("aciklama", "")))
                b.clicked.connect(lambda _=False, i=i: self._apply_preset(i))
                pg.addWidget(b, i // 3, i % 3)
                self.preset_btns.append(b)
            self.body.addLayout(pg)
            self.preset_hint = label("Bir hazır ayar seç ya da aşağıdan kendin ayarla.", "hint", wrap=True)
            self.body.addWidget(self.preset_hint)
            self.preset_reset = QPushButton("Hazır ayara dön")
            self.preset_reset.setObjectName("link")
            self.preset_reset.setCursor(Qt.PointingHandCursor)
            self.preset_reset.clicked.connect(self._reset_preset)
            self.preset_reset.hide()
            self.body.addWidget(self.preset_reset, 0, Qt.AlignLeft)
            self.body.addSpacing(4)
        self.body.addWidget(section("Bütçe"))
        from .views import BudgetCard
        self.budget = BudgetCard()
        self.body.addWidget(self.budget)
        self.budget.update_budget({})
        self.body.addSpacing(4)
        self.body.addWidget(section("Işık"))
        lc, lv = card(self.body)
        lh = QHBoxLayout()
        lh.setSpacing(14)
        self.dial = LightDial(104)
        self.dial.changed.connect(win.set_light_live)
        self.dial.released.connect(win.set_light_live)
        lh.addWidget(self.dial)
        rv = QVBoxLayout()
        rv.setSpacing(6)
        self.light_lbl = label("", "", wrap=True)
        rv.addWidget(self.light_lbl)
        rv.addWidget(label("Yalnızca önizlemeyi değiştirir. Gerçekte eser her ışıkta okunur.", "hint", wrap=True))
        pr = QHBoxLayout()
        pr.setSpacing(4)
        for txt, az, el in (("Sol üst", -130, 40), ("Tepe", -90, 75), ("Yandan", 180, 18)):
            b = QPushButton(txt)
            b.setObjectName("chip")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, a=az, e=el: win.set_light_live(a, e))
            pr.addWidget(b)
        pr.addStretch(1)
        rv.addLayout(pr)
        lh.addLayout(rv, 1)
        lv.addLayout(lh)

        self.body.addSpacing(4)
        self.body.addWidget(section("Taş tiplerine göre"))
        self.vcards: dict[int, tuple[QFrame, SliderRow | None, SliderRow, SliderRow, QLabel]] = {}
        for v in VOCAB_ORDER:
            f, fv = card(self.body, margins=(14, 12, 14, 12), spacing=8)
            hh = QHBoxLayout()
            hh.setSpacing(8)
            hh.addWidget(Dot(VOCAB_COLOURS[v], 12))
            nm = label(VOCAB_NAMES_TR[Vocab(v)], "modeTitle")
            hh.addWidget(nm)
            hh.addStretch(1)
            unused = label("bu görselde yok", "hint")
            hh.addWidget(unused)
            fv.addLayout(hh)
            size = None
            if v != int(Vocab.FLAT):
                lo, hi = (8.0, 40.0) if v == int(Vocab.RAYS) else (3.0, 20.0)
                size = SliderRow("Taş boyu", lo, hi, 0.2, 9.6, "mm", 1)
                size.value_changed.connect(lambda val, v=v: win.set_vocab_param(v, "tile_mm", val))
                fv.addWidget(size)
            height = SliderRow("Yükseklik", 0.4, 14, 0.2, 3, "mm", 1)
            height.value_changed.connect(lambda val, v=v: win.set_vocab_param(v, "base_height_mm", val))
            fv.addWidget(height)
            jit = SliderRow("Yükseklik oynaması", 0, 2, 0.05, 0.35, "mm", 2,
                            hint="Her taşın yüksekliği biraz oynar: el işi hissi.")
            jit.value_changed.connect(lambda val, v=v: win.set_vocab_param(v, "jitter_mm", val))
            fv.addWidget(jit)
            self.advanced.append(jit)
            self.vcards[v] = (f, size, height, jit, unused)

        self.body.addSpacing(4)
        self.body.addWidget(section("Biçim"))
        fig = VOCAB_NAMES_TR[Vocab.FIGURE]
        self.bump = SliderRow(f"{fig} yuvarlaklığı", 0, 10, 0.5, 4, "mm", 1,
                              hint=f"{fig} bölgelerinin ortası kabarık bir gövde gibi yuvarlanır.")
        self.bump.value_changed.connect(lambda val: win.set_setting("figure_bump_mm", val))
        self.body.addWidget(self.bump)

        adv, av = advanced_box()
        self.bevel = SliderRow("Kenar yumuşatma", 0, 1.5, 0.05, 0.45, "mm", 2,
                               hint="Taş kenarlarının yastık gibi aşağı inmesi.")
        self.bevel.value_changed.connect(lambda val: win.set_setting("bevel_mm", val))
        av.addWidget(self.bevel)
        self.base = SliderRow("Taban kalınlığı", 1, 4, 0.2, 2, "mm", 1,
                              hint="Parçaların altındaki taşıyıcı katman.")
        self.base.value_changed.connect(lambda val: win.set_setting("base_thickness_mm", val))
        av.addWidget(self.base)
        sr = QHBoxLayout()
        self.seed_lbl = label("", "hint")
        sr.addWidget(label("Rastgele desen"))
        sr.addStretch(1)
        sr.addWidget(self.seed_lbl)
        self.dice = QPushButton("  Karıştır")
        self.dice.setToolTip("Taşların rastgele dizilişini değiştir")
        self.dice.clicked.connect(win.reseed)
        sr.addWidget(self.dice)
        av.addLayout(sr)
        self.add_advanced(adv)
        self.finish()

    def _apply_preset(self, i: int) -> None:
        self.win.apply_preset(self.presets[i])

    def update_budget(self, b: dict) -> None:
        self.budget.update_budget(b or {})

    def _reset_preset(self) -> None:
        pr = find_preset(self.presets, self.win.project.preset)
        if pr:
            self.win.apply_preset(pr)

    def show_preset(self) -> None:
        """Chip state + hint from project.preset / preset_modified (G-3)."""
        if not self.presets:
            return
        name, mod = self.win.project.preset, self.win.project.preset_modified
        pr = find_preset(self.presets, name)
        for b, p in zip(self.preset_btns, self.presets):
            b.setChecked(p is pr)
        if pr is None:
            self.preset_hint.setText("Bir hazır ayar seç ya da aşağıdan kendin ayarla.")
        elif mod:
            self.preset_hint.setText(f"{preset_label(pr)} · değiştirildi. Aşağıdaki ayarlar artık senin.")
        else:
            self.preset_hint.setText(current_names(str(pr.get("aciklama", ""))))
        self.preset_reset.setVisible(pr is not None and mod)

    def clear_preset(self) -> None:
        self.show_preset()

    def retheme(self) -> None:
        self.dice.setIcon(icons.icon("dice", current().text, 18))
        self.dial.update()

    def set_light(self, az: float, el: float) -> None:
        self.dial.set_light(az, el)
        self.light_lbl.setText(light_text(az, el))

    def refresh(self) -> None:
        s = self.win.project.settings
        used = self.win.zones.used_vocabs()
        for v, (f, size, height, jit, unused) in self.vcards.items():
            p = s.vocab[v]
            if size:
                size.set_value(p.tile_mm)
            height.set_value(p.base_height_mm)
            jit.set_value(p.jitter_mm)
            unused.setVisible(v not in used)
            # unused vocabularies collapse to their title (less noise)
            for w in (size, height):
                if w:
                    w.setVisible(v in used)
            jit.setVisible(v in used and self.win.advanced_on)
        self.bump.set_value(s.figure_bump_mm)
        self.bevel.set_value(s.bevel_mm)
        self.base.set_value(s.base_thickness_mm)
        self.seed_lbl.setText(f"#{s.seed}")
        self.set_light(self.win.project.light_az, self.win.project.light_el)
        self.show_preset()

    def set_advanced(self, on: bool) -> None:
        super().set_advanced(on)
        used = self.win.zones.used_vocabs()
        for v, (_, _, _, jit, _) in self.vcards.items():
            jit.setVisible(on and v in used)


# ====================================================================== 5 Üretim
class ModeCard(ClickFrame):
    def __init__(self, title: str, sub: str, text: str, icon_name: str):
        super().__init__()
        self.setObjectName("modeCard")
        self.setCursor(Qt.PointingHandCursor)
        self.icon_name = icon_name
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(6)
        h = QHBoxLayout()
        h.setSpacing(10)
        self.ic = QLabel()
        h.addWidget(self.ic)
        tv = QVBoxLayout()
        tv.setSpacing(0)
        tv.addWidget(label(title, "modeTitle"))
        tv.addWidget(label(sub, "hint"))
        h.addLayout(tv, 1)
        self.radio = QLabel()
        h.addWidget(self.radio, 0, Qt.AlignTop)
        v.addLayout(h)
        v.addWidget(label(text, "", wrap=True))
        self.stat = label("", "value")
        v.addWidget(self.stat)
        self.selected = False

    def set_selected(self, on: bool) -> None:
        self.selected = on
        self.setProperty("selected", "true" if on else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        self.retheme()

    def retheme(self) -> None:
        t = current()
        self.ic.setPixmap(icons.pixmap(self.icon_name, t.red if self.selected else t.muted, 28))
        self.radio.setPixmap(icons.pixmap("check", t.red, 20) if self.selected else QPixmap())


class ProductionStep(StepPanel):
    def __init__(self, win):
        super().__init__(win, 5, "Üretim",
                         "Yazıcın nasıl basacak? İki yol da aynı mozaiği verir; "
                         "fark montajda ve atılan filamentte.")
        ams, av = card(self.body, flat=True, margins=(14, 10, 14, 10), spacing=2)
        av.addWidget(label("AMS = Bambu'nun çok renkli besleyicisi.", "modeTitle"))
        av.addWidget(label("Yazıcıya 4 makara takılır, baskı sırasında rengi kendisi değiştirir. "
                           "AMS'in yoksa ilk yolu seç.", "hint", wrap=True))
        self.off = ModeCard("Tek renk tablalar", "AMS gerekmez · parça parça",
                            "Her parça tek renk basılır; filament hiç değişmez, hiç plastik atılmaz. "
                            "Parçaları koyu arka panele (kafes) sen dizersin; her parça yalnızca "
                            "kendi yerine oturur.", "spool")
        self.on = ModeCard("Çok renkli paneller", "AMS gerekir · panel panel",
                           "Her panel tek parça olarak, renk renk basılır. Montaj kolay: panelleri "
                           "yan yana koyarsın. Ama her renk değişiminde yazıcı bir miktar filament atar.",
                           "spools")
        self.off.clicked.connect(lambda: win.set_mode(Mode.PIECES))
        self.on.clicked.connect(lambda: win.set_mode(Mode.AMS))
        self.body.addWidget(self.off)
        self.body.addWidget(self.on)

        self.body.addSpacing(6)
        self.body.addWidget(section("Paneller"))
        self.panel = SliderRow("Panel boyu", 100, 240, 5, 240, "mm", 0,
                               hint="Eser bu boydan büyük olmayan panellere bölünür. X1C tablası 256 mm.")
        self.panel.value_changed.connect(lambda v: win.set_setting("panel_max_mm", v))
        self.body.addWidget(self.panel)
        self.panel_info = label("", "hint", wrap=True)
        self.body.addWidget(self.panel_info)
        self.merge = QCheckBox("Aynı renk komşuları birleştir")
        self.merge.toggled.connect(lambda on: win.set_setting("merge_same_colour", on))
        self.body.addWidget(self.merge)
        self.body.addWidget(label("Yan yana duran aynı renk taşlar tek parça basılır; aradaki çizgiler "
                                  "oluk olarak kalır. Çok daha az parça demek.", "hint", wrap=True))
        self.finish()

    def retheme(self) -> None:
        self.off.retheme()
        self.on.retheme()

    def refresh(self) -> None:
        s = self.win.project.settings
        self.off.set_selected(s.mode == Mode.PIECES)
        self.on.set_selected(s.mode == Mode.AMS)
        self.panel.set_value(s.panel_max_mm)
        self.merge.blockSignals(True)
        self.merge.setChecked(s.merge_same_colour)
        self.merge.blockSignals(False)
        self.update_estimate(self.win.last_estimate)

    def update_estimate(self, est: dict | None = None) -> None:
        """Both cards show the numbers of *their* mode, straight from estimate() (Ö-1, R2-4)."""
        by = self.win.estimates_by_mode()
        cur = self.win.project.settings.mode
        e_off, e_on = by.get(Mode.PIECES), by.get(Mode.AMS)

        def kg(e):
            return f"≈ {tr_num((e.get('gram') or 0) / 1000, 2)} kg toplam" if e.get("gram") is not None else ""

        ex = self.win.exact_parts_current()
        if e_off:
            n_off = tr_num(ex["parca"]) if ex else f"≈ {tr_num(e_off.get('parca') or 0)}"
            self.off.stat.setText(f"{n_off} parça · {tr_num(e_off.get('tabla') or 0)} tabla · "
                                  f"{kg(e_off)}\nRenk değişiminde atılan filament: 0 g")
        else:
            self.off.stat.setText("Sayılar hesaplanıyor…")
        if e_on:
            pg = e_on.get("purge_gram")
            self.on.stat.setText(f"{tr_num(e_on.get('panel') or 0)} panel, her biri tek parça · "
                                 f"{tr_num(e_on.get('tabla') or 0)} tabla · {kg(e_on)}\n"
                                 "Renk değişiminde atılan filament: "
                                 + (f"≈ {tr_num(pg)} g" if pg is not None else "hesaplanamadı"))
        else:
            self.on.stat.setText("Sayılar hesaplanıyor…")
        e = by.get(cur)
        if e and e.get("panel"):
            xy = e.get("panel_xy")
            grid = f" ({xy[0]} × {xy[1]})" if isinstance(xy, (list, tuple)) and len(xy) == 2 else ""
            self.panel_info.setText(f"Bu modda eser {tr_num(e['panel'])} panele{grid} bölünür.")
        else:
            self.panel_info.setText("")


# ====================================================================== 6 Dışa aktar
class ExportStep(StepPanel):
    def __init__(self, win):
        super().__init__(win, 6, "Dışa aktar",
                         "Bambu Studio'da açılacak tablalar, montaj kılavuzu ve rapor tek klasöre yazılır.")
        self.body.addWidget(section("Klasör"))
        fr = QHBoxLayout()
        fr.setSpacing(6)
        self.folder = QLineEdit()
        self.folder.setPlaceholderText("Bir klasör seç…")
        self.folder.textChanged.connect(win.set_export_dir)
        fr.addWidget(self.folder, 1)
        self.pick = QToolButton()
        self.pick.setToolTip("Klasör seç")
        self.pick.setFixedSize(36, 34)
        self.pick.setIconSize(QSize(18, 18))
        self.pick.clicked.connect(self._pick)
        fr.addWidget(self.pick)
        self.body.addLayout(fr)

        self.body.addSpacing(4)
        self.body.addWidget(section("Klasörde neler olacak"))
        lc, lv = card(self.body, margins=(14, 12, 14, 12), spacing=9)
        self.what_icons = []
        for ic, title, sub in (("print", "Tablalar (3MF)", "Bambu Studio'da doğrudan açılır"),
                               ("zones", "Montaj kılavuzu", "Panel panel renkli harita, PDF ve PNG"),
                               ("relief", "STL dosyaları", "Yedek, her parça ayrı"),
                               ("info", "Rapor", "Gram, saat, renk renk parça sayısı")):
            h = QHBoxLayout()
            h.setSpacing(10)
            il = QLabel()
            self.what_icons.append((il, ic))
            h.addWidget(il)
            tv = QVBoxLayout()
            tv.setSpacing(0)
            tv.addWidget(label(title))
            tv.addWidget(label(sub, "hint"))
            h.addLayout(tv, 1)
            lv.addLayout(h)

        self.go = QPushButton("Dışa aktar")
        self.go.setObjectName("primary")
        self.go.setMinimumHeight(44)
        self.go.setCursor(Qt.PointingHandCursor)
        self.go.clicked.connect(win.export)
        self.body.addWidget(self.go)
        self.note = label("Tam çözünürlükte (0,4 mm) yeniden hesaplanır; birkaç dakika sürebilir. "
                          "Bu sırada programı kullanmaya devam edebilirsin.", "hint", wrap=True)
        self.body.addWidget(self.note)

        # progress
        self.prog_box = QWidget()
        pv = QVBoxLayout(self.prog_box)
        pv.setContentsMargins(0, 4, 0, 0)
        pv.setSpacing(6)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        pv.addWidget(self.bar)
        pr = QHBoxLayout()
        self.prog_msg = label("", "hint", wrap=True)
        pr.addWidget(self.prog_msg, 1)
        self.cancel_btn = QPushButton("Vazgeç")
        self.cancel_btn.setObjectName("chip")
        self.cancel_btn.setCursor(Qt.PointingHandCursor)
        self.cancel_btn.clicked.connect(win.cancel_export)
        pr.addWidget(self.cancel_btn, 0, Qt.AlignTop)
        pv.addLayout(pr)
        self.body.addWidget(self.prog_box)
        self.prog_box.hide()

        # result
        self.result, rv = card(self.body, margins=(16, 14, 16, 16), spacing=10)
        rh = QHBoxLayout()
        self.res_icon = QLabel()
        rh.addWidget(self.res_icon)
        self.res_title = label("Hazır", "modeTitle")
        rh.addWidget(self.res_title, 1)
        rv.addLayout(rh)
        self.res_grid = QGridLayout()
        self.res_grid.setHorizontalSpacing(16)
        self.res_grid.setVerticalSpacing(8)
        rv.addLayout(self.res_grid)
        self.res_colours = QGridLayout()
        self.res_colours.setHorizontalSpacing(10)
        self.res_colours.setVerticalSpacing(4)
        rv.addLayout(self.res_colours)
        self.res_hassas = label("", "hint", wrap=True)
        rv.addWidget(self.res_hassas)
        self.res_warn = label("", "hint", wrap=True)
        rv.addWidget(self.res_warn)
        ob = QHBoxLayout()
        ob.setSpacing(6)
        self.open_btn = QPushButton("  Klasörü aç")
        self.open_btn.clicked.connect(win.open_export_folder)
        ob.addWidget(self.open_btn, 1)
        self.bambu_btn = QPushButton("Bambu Studio'da aç")
        self.bambu_btn.setToolTip("İlk tablayı Bambu Studio'da açar")
        self.bambu_btn.clicked.connect(win.open_in_bambu)
        ob.addWidget(self.bambu_btn, 1)
        rv.addLayout(ob)
        self.result.hide()

        self.err, ev = card(self.body, margins=(14, 12, 14, 12))
        self.err.setObjectName("errorBox")
        self.err_lbl = label("", "", wrap=True)
        ev.addWidget(self.err_lbl)
        self.err.hide()
        self.finish()

    def retheme(self) -> None:
        t = current()
        self.pick.setIcon(icons.icon("folder", t.text, 18))
        for il, ic in self.what_icons:
            il.setPixmap(icons.pixmap(ic, t.muted, 22))
        self.open_btn.setIcon(icons.icon("open", t.text, 18))
        self.res_icon.setPixmap(icons.pixmap("check", t.ok, 24))

    def _pick(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Dışa aktarım klasörü", self.folder.text() or os.path.expanduser("~"))
        if d:
            self.folder.setText(os.path.normpath(d))

    def refresh(self) -> None:
        self.folder.blockSignals(True)
        self.folder.setText(self.win.project.export_dir or self.win.default_export_dir())
        self.folder.blockSignals(False)

    def set_running(self, on: bool, can_cancel: bool = False) -> None:
        self.cancel_btn.setVisible(on and can_cancel)
        self.go.setEnabled(not on)
        self.go.setText("Dışa aktarılıyor…" if on else "Dışa aktar")
        self.prog_box.setVisible(on)
        if on:
            self.result.hide()
            self.err.hide()
            self.bar.setValue(0)

    def set_progress(self, f: float, msg: str) -> None:
        self.bar.setValue(int(max(0, min(1, f)) * 1000))
        if msg:
            self.prog_msg.setText(msg)

    def show_result(self, res: dict) -> None:
        while self.res_grid.count():
            it = self.res_grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        files = res.get("files", []) or []
        items = [("Dosya", tr_num(len(files))), ("Tabla", tr_num(len(res.get("plates", []) or []))),
                 ("Parça", tr_num(res.get("parts", 0) or 0)), ("Panel", tr_num(res.get("panels", 0) or 0)),
                 ("Filament", f"{tr_num((res.get('grams', 0) or 0) / 1000, 2)} kg"),
                 ("Süre", f"~{tr_num(res.get('hours', 0) or 0)} saat")]
        for i, (k, v) in enumerate(items):
            box = QVBoxLayout()
            box.setSpacing(0)
            box.addWidget(label(v, "bigValue"))
            box.addWidget(label(k, "hint"))
            w = QWidget()
            w.setLayout(box)
            self.res_grid.addWidget(w, i // 3, i % 3)
        # colour by colour (when the 3D team's result carries it)
        while self.res_colours.count():
            it = self.res_colours.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        rg, rp = res.get("renk_gram") or {}, res.get("renk_parca") or {}
        pal = {f.name: f for f in self.win.project.settings.palette}
        names = [n for n in pal if n in rg or n in rp] + [n for n in rg if n not in pal]
        if names:
            self.res_colours.addWidget(label("RENK RENK", "section"), 0, 0, 1, 4)
        for i, nm in enumerate(names, start=1):
            col = pal[nm].hex if nm in pal else current().faint
            self.res_colours.addWidget(Dot(col, 12), i, 0)
            extra = "  · kafes" if res.get("kafes_renk") == nm else ""
            self.res_colours.addWidget(label(nm + extra), i, 1)
            self.res_colours.addWidget(label(f"{tr_num(rp.get(nm, 0))} parça", "hint"), i, 2)
            self.res_colours.addWidget(label(f"{tr_num(rg.get(nm, 0))} g", "value"), i, 3)
        self.res_colours.setColumnStretch(1, 1)
        hs = res.get("hassas_parca", res.get("hassas"))
        parts = res.get("parts") or 0
        if hs:
            share = hs / max(1, parts)
            self.res_hassas.setText(
                f"{tr_num(hs)} hassas parça (%{tr_num(share * 100)}). Hassas parça, üstüne numara sığmayacak kadar "
                "küçük bir parçadır: numarası kılavuzda ve kafesteki cebinde yazar, cımbızla yerleştirilir.")
            self.res_hassas.setStyleSheet(f"color: {current().warn if share > 0.05 else current().muted};")
        self.res_hassas.setVisible(bool(hs))
        warns = res.get("warnings", []) or []
        self.res_warn.setVisible(bool(warns))
        self.res_warn.setText("\n".join("• " + w for w in warns))
        self.res_title.setText("Hazır" if not warns else "Hazır (uyarılarla)")
        self.bambu_btn.setVisible(bool(self.win.bambu_target()))
        self.result.show()
        self.err.hide()
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, lambda: self.ensureWidgetVisible(self.open_btn, 0, 24))

    def show_error(self, msg: str) -> None:
        self.err_lbl.setText("Dışa aktarım yapılamadı.\n" + msg)
        self.err.show()
        self.result.hide()
