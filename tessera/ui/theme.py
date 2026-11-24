"""Visual identity: colours, fonts and the Qt style sheet.

Palette idea: a Soviet mosaic wall seen in a calm modern room. Warm cream plaster, charcoal
lattice, one deep red for meaning/action and a small amount of smalt gold for focus.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

from ..model import Vocab

HEAD_FONT = "Bahnschrift"
BODY_FONT = "Segoe UI"


@dataclass(frozen=True)
class Theme:
    name: str
    bg: str          # window background
    panel: str       # side panels
    card: str        # raised cards / inputs
    card2: str       # hover / pressed
    line: str        # hairlines
    text: str
    muted: str
    faint: str
    red: str
    red_hover: str
    on_red: str
    gold: str
    stage: str       # preview background
    stage_grid: str
    ok: str
    warn: str

    def c(self, key: str) -> QColor:
        return QColor(getattr(self, key))


DARK = Theme(
    name="dark",
    bg="#171513", panel="#1F1C19", card="#2A2622", card2="#35302B", line="#3B352F",
    text="#F1E9DA", muted="#B3A893", faint="#7A715F",
    red="#C9372C", red_hover="#DD4A3E", on_red="#FFF6EA", gold="#D9A441",
    stage="#110F0E", stage_grid="#1A1715", ok="#7FA36B", warn="#E0A33A",
)

LIGHT = Theme(
    name="light",
    bg="#EFE7D8", panel="#F7F1E6", card="#FFFBF4", card2="#EADFCC", line="#D8CBB4",
    text="#2A2520", muted="#6B6152", faint="#A0937D",
    red="#B3261E", red_hover="#C93A2F", on_red="#FFF8EE", gold="#B5832A",
    stage="#E4DAC8", stage_grid="#DDD2BE", ok="#4F7A3E", warn="#B9781A",
)

# zone colours (overlay + legend). Chosen to sit with the palette, distinct at a glance.
VOCAB_COLOURS = {
    int(Vocab.STONES): "#C9772F",   # terracotta
    int(Vocab.RAYS): "#E2B33C",     # smalt gold
    int(Vocab.BLOCKS): "#3E8FA8",   # cold blue
    int(Vocab.FIGURE): "#D0413A",   # red
    int(Vocab.FOCAL): "#F2E06B",    # bright yellow
    int(Vocab.FLAT): "#8AA37A",     # sage
}

VOCAB_HINTS_TR = {
    int(Vocab.STONES): "Biçime uyan düzensiz küçük taşlar. Her yere uyar.",
    int(Vocab.RAYS): "Bir merkezden yayılan uzun tahtalar. Gökyüzü, güneş, hale.",
    int(Vocab.BLOCKS): "Izgaraya dizili kare taşlar. Binalar, pencereler, dokular.",
    int(Vocab.FIGURE): "Yükseltilmiş, yuvarlak kabartma. Yüzler, eller, insanlar.",
    int(Vocab.FOCAL): "En yüksek ve en ince işçilik. Eserde tek bir yer için.",
    int(Vocab.FLAT): "Tek parça düz levha. Sakin, geniş alanlar.",
}


_current: list[Theme] = [DARK]


def current() -> Theme:
    return _current[0]


def load_fonts() -> None:
    """Make Bahnschrift / Segoe UI available even on platforms without the system font
    database (the offscreen platform used by tests and screenshots)."""
    if HEAD_FONT in QFontDatabase.families() and BODY_FONT in QFontDatabase.families():
        return
    fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    for f in ("bahnschrift.ttf", "segoeui.ttf", "segoeuib.ttf", "seguisb.ttf", "segoeuisl.ttf",
              "segoeuil.ttf", "seguisym.ttf"):
        p = os.path.join(fonts, f)
        if os.path.exists(p):
            QFontDatabase.addApplicationFont(p)


def head_font(px: int, weight: QFont.Weight = QFont.DemiBold, condensed: bool = False) -> QFont:
    f = QFont(HEAD_FONT)
    f.setPixelSize(px)
    f.setWeight(weight)
    if condensed:
        f.setStretch(QFont.SemiCondensed)
    return f


def body_font(px: int = 13, weight: QFont.Weight = QFont.Normal) -> QFont:
    f = QFont(BODY_FONT)
    f.setPixelSize(px)
    f.setWeight(weight)
    return f


def apply(app: QApplication, t: Theme) -> None:
    _current[0] = t
    pal = QPalette()
    pal.setColor(QPalette.Window, t.c("bg"))
    pal.setColor(QPalette.WindowText, t.c("text"))
    pal.setColor(QPalette.Base, t.c("card"))
    pal.setColor(QPalette.AlternateBase, t.c("panel"))
    pal.setColor(QPalette.Text, t.c("text"))
    pal.setColor(QPalette.Button, t.c("card"))
    pal.setColor(QPalette.ButtonText, t.c("text"))
    pal.setColor(QPalette.Highlight, t.c("red"))
    pal.setColor(QPalette.HighlightedText, t.c("on_red"))
    pal.setColor(QPalette.ToolTipBase, t.c("card"))
    pal.setColor(QPalette.ToolTipText, t.c("text"))
    pal.setColor(QPalette.PlaceholderText, t.c("faint"))
    pal.setColor(QPalette.Mid, t.c("line"))
    app.setPalette(pal)
    app.setFont(body_font(13))
    app.setStyleSheet(stylesheet(t))


def stylesheet(t: Theme) -> str:
    from .icons import check_svg_path
    check = check_svg_path()
    return f"""
* {{ font-family: "{BODY_FONT}"; font-size: 13px; color: {t.text}; outline: 0; }}
QMainWindow, #root {{ background: {t.bg}; }}
QToolTip {{ background: {t.card}; color: {t.text}; border: 1px solid {t.line}; padding: 6px 8px; border-radius: 6px; }}

/* ---------- header ---------- */
#header {{ background: {t.panel}; border-bottom: 1px solid {t.line}; }}
#brand {{ font-family: "{HEAD_FONT}"; font-size: 19px; font-weight: 700; letter-spacing: 4px; color: {t.text}; }}
#projectName {{ color: {t.muted}; font-size: 13px; }}
#dirtyDot {{ color: {t.gold}; font-size: 13px; }}

/* ---------- buttons ---------- */
QPushButton, QToolButton {{
  background: {t.card}; border: 1px solid {t.line}; border-radius: 7px;
  padding: 7px 12px; color: {t.text};
}}
QPushButton:hover, QToolButton:hover {{ background: {t.card2}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {t.line}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {t.faint}; background: {t.panel}; }}
QPushButton:checked, QToolButton:checked {{ background: {t.text}; color: {t.bg}; border-color: {t.text}; }}
QPushButton#primary {{
  background: {t.red}; border: 1px solid {t.red}; color: {t.on_red};
  font-family: "{HEAD_FONT}"; font-weight: 600; font-size: 14px; letter-spacing: 0.5px; padding: 9px 18px;
}}
QPushButton#primary:hover {{ background: {t.red_hover}; border-color: {t.red_hover}; }}
QPushButton#primary:disabled {{ background: {t.card2}; border-color: {t.line}; color: {t.faint}; }}
QPushButton#ghost, QToolButton#ghost {{ background: transparent; border: 1px solid transparent; }}
QPushButton#ghost:hover, QToolButton#ghost:hover {{ background: {t.card}; border-color: {t.line}; }}
QToolButton#ghost:checked {{ background: {t.card2}; color: {t.text}; border-color: {t.line}; }}
QPushButton#chip {{ background: transparent; border: 1px solid {t.line}; border-radius: 13px; padding: 4px 11px; color: {t.muted}; font-size: 12px; }}
QPushButton#chip:hover {{ border-color: {t.faint}; color: {t.text}; }}
QPushButton#chip:checked {{ background: {t.card2}; border-color: {t.gold}; color: {t.text}; }}
QPushButton#link {{ background: transparent; border: none; color: {t.red}; padding: 2px 0; text-align: left; }}
QPushButton#link:hover {{ text-decoration: underline; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}
QToolButton#tool {{ padding: 6px; border-radius: 7px; }}
QToolButton#tool:checked {{ background: {t.red}; border-color: {t.red}; color: {t.on_red}; }}

/* ---------- step rail ---------- */
#rail {{ background: {t.panel}; border-right: 1px solid {t.line}; }}
#railFoot {{ color: {t.faint}; font-size: 11px; }}

/* ---------- side panel ---------- */
#sidePanel {{ background: {t.panel}; border-left: 1px solid {t.line}; }}
#stepEyebrow {{ font-family: "{HEAD_FONT}"; font-size: 11px; font-weight: 600; letter-spacing: 2px; color: {t.red}; }}
#stepTitle {{ font-family: "{HEAD_FONT}"; font-size: 24px; font-weight: 600; color: {t.text}; }}
#stepLead {{ color: {t.muted}; font-size: 13px; }}
#section {{ font-family: "{HEAD_FONT}"; font-size: 11px; font-weight: 600; letter-spacing: 1.6px; color: {t.muted}; }}
#hint, QLabel#hint {{ color: {t.muted}; font-size: 12px; }}
#value {{ font-family: "{HEAD_FONT}"; font-size: 13px; color: {t.text}; }}
#bigValue {{ font-family: "{HEAD_FONT}"; font-size: 22px; font-weight: 600; color: {t.text}; }}
#card {{ background: {t.card}; border: 1px solid {t.line}; border-radius: 10px; }}
#cardFlat {{ background: transparent; border: 1px solid {t.line}; border-radius: 10px; }}
#modeCard {{ background: {t.card}; border: 1px solid {t.line}; border-radius: 12px; }}
#modeCard[selected="true"] {{ border: 2px solid {t.red}; }}
#modeTitle {{ font-family: "{HEAD_FONT}"; font-size: 15px; font-weight: 600; }}
#warnBox {{ background: transparent; border: 1px solid {t.warn}; border-radius: 8px; color: {t.text}; }}
#errorBox {{ background: transparent; border: 1px solid {t.red}; border-radius: 8px; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}

/* ---------- inputs ---------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
  background: {t.card}; border: 1px solid {t.line}; border-radius: 7px; padding: 6px 8px;
  selection-background-color: {t.red}; selection-color: {t.on_red};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {t.gold}; }}
QDoubleSpinBox#bigSpin {{ font-family: "{HEAD_FONT}"; font-size: 20px; font-weight: 600; padding: 4px 8px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ width: 0px; border: none; image: none; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {t.card}; border: 1px solid {t.line}; selection-background-color: {t.card2}; selection-color: {t.text}; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {t.faint}; background: {t.card}; }}
QCheckBox::indicator:checked {{ background: {t.red}; border-color: {t.red}; image: url({check}); }}
QRadioButton::indicator {{ width: 16px; height: 16px; border-radius: 8px; border: 1px solid {t.faint}; background: {t.card}; }}
QRadioButton::indicator:checked {{ border: 5px solid {t.red}; background: {t.on_red}; }}

QSlider::groove:horizontal {{ height: 4px; background: {t.line}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {t.red}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {t.text}; width: 14px; height: 14px; margin: -6px 0; border-radius: 7px; border: 2px solid {t.panel}; }}
QSlider::handle:horizontal:hover {{ background: {t.gold}; }}

QProgressBar {{ background: {t.card}; border: 1px solid {t.line}; border-radius: 6px; height: 12px; text-align: center; font-size: 11px; color: transparent; }}
QProgressBar::chunk {{ background: {t.red}; border-radius: 5px; }}

QListWidget {{ background: {t.card}; border: 1px solid {t.line}; border-radius: 8px; padding: 4px; }}
QListWidget::item {{ padding: 6px 6px; border-radius: 6px; }}
QListWidget::item:selected {{ background: {t.card2}; color: {t.text}; }}
QListWidget::item:hover {{ background: {t.card2}; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {t.line}; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {t.faint}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {t.line}; border-radius: 4px; min-width: 30px; }}

QMenu {{ background: {t.card}; border: 1px solid {t.line}; border-radius: 8px; padding: 6px; }}
QMenu::item {{ padding: 7px 26px 7px 12px; border-radius: 5px; }}
QMenu::item:selected {{ background: {t.card2}; }}
QMenu::item:disabled {{ color: {t.faint}; }}
QMenu::separator {{ height: 1px; background: {t.line}; margin: 5px 8px; }}

/* ---------- centre ---------- */
#stage {{ background: {t.stage}; }}
#stageBar {{ background: {t.bg}; border-bottom: 1px solid {t.line}; }}
#floatBox {{ background: {t.panel}; border: 1px solid {t.line}; border-radius: 10px; }}
#floatLabel {{ font-family: "{HEAD_FONT}"; font-size: 10px; font-weight: 600; letter-spacing: 1.4px; color: {t.muted}; }}

/* ---------- stats bar ---------- */
#statsBar {{ background: {t.panel}; border-top: 1px solid {t.line}; }}
#statValue {{ font-family: "{HEAD_FONT}"; font-size: 15px; font-weight: 600; color: {t.text}; }}
#statLabel {{ font-size: 11px; color: {t.muted}; }}
#statSep {{ background: {t.line}; }}
#status {{ color: {t.muted}; font-size: 12px; }}

/* ---------- empty state ---------- */
#emptyTitle {{ font-family: "{HEAD_FONT}"; font-size: 30px; font-weight: 600; color: {t.text}; }}
#emptyLead {{ color: {t.muted}; font-size: 14px; }}
"""
