"""Hand-drawn line icons (24x24, stroke 1.7). Drawn for Tessera; no third-party assets.

`icon(name, colour)` renders an SVG to a crisp QIcon. `C` is replaced by the colour.
"""
from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

_S = 'fill="none" stroke="C" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"'

PATHS = {
    # --- steps
    "image": f'<rect x="3.5" y="4.5" width="17" height="15" rx="1.5" {_S}/><circle cx="9" cy="10" r="1.8" {_S}/>'
             f'<path d="M4 17.5l5-5 3.5 3.5 2.5-2.5 5 5" {_S}/>',
    "palette": f'<circle cx="7.5" cy="7.5" r="3.5" {_S}/><circle cx="16.5" cy="7.5" r="3.5" {_S}/>'
               f'<circle cx="7.5" cy="16.5" r="3.5" {_S}/><rect x="13" y="13" width="7" height="7" rx="1" fill="C"/>',
    "zones": f'<path d="M3.5 20.5V3.5h17v17z" {_S}/><path d="M3.5 13.5c4-1 6-4 7-10" {_S}/>'
             f'<path d="M10 20.5c0-5 3-8 10.5-9" {_S}/><circle cx="16.5" cy="7" r="1.4" fill="C"/>',
    "relief": f'<path d="M3 19.5h18" {_S}/><path d="M4 19.5v-4h4v4M10 19.5v-9h4v9M16 19.5v-6h4v6" {_S}/>'
              f'<path d="M10 7.5l2-3 2 3" {_S}/>',
    "print": f'<rect x="4" y="3.5" width="16" height="17" rx="1.5" {_S}/><path d="M4 8h16" {_S}/>'
             f'<path d="M9 8v3l3 2 3-2V8" {_S}/><path d="M7.5 17.5h9" {_S}/>',
    "export": f'<path d="M12 3.5v11M7.5 8L12 3.5 16.5 8" {_S}/><path d="M4.5 13v6.5h15V13" {_S}/>',
    # --- header
    "new": f'<path d="M6 3.5h8l4 4v13H6z" {_S}/><path d="M14 3.5v4h4M12 11v6M9 14h6" {_S}/>',
    "open": f'<path d="M3.5 6.5v12h15l2.5-8H8L5.5 18" {_S}/><path d="M3.5 6.5V5h6l2 2h6v3.5" {_S}/>',
    "save": f'<path d="M4.5 4.5h12l3 3v12h-15z" {_S}/><path d="M8 4.5v4.5h7V4.5M8 19.5v-6h8v6" {_S}/>',
    "recent": f'<circle cx="12" cy="12" r="8.5" {_S}/><path d="M12 7.5V12l3 2" {_S}/>',
    "sun": f'<circle cx="12" cy="12" r="4" {_S}/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4" {_S}/>',
    "moon": f'<path d="M19.5 14.5A8 8 0 0 1 9.5 4.5a8 8 0 1 0 10 10z" {_S}/>',
    "sliders": f'<path d="M4 7h9M17 7h3M4 17h3M11 17h9" {_S}/><circle cx="15" cy="7" r="2" {_S}/><circle cx="9" cy="17" r="2" {_S}/>',
    "more": f'<circle cx="6" cy="12" r="1.3" fill="C"/><circle cx="12" cy="12" r="1.3" fill="C"/><circle cx="18" cy="12" r="1.3" fill="C"/>',
    # --- zone tools
    "brush": f'<path d="M14.5 4.5l5 5-7.5 7.5-5-5z" {_S}/><path d="M7 12l-2.5 2.5c-1 1-1 3.5-1 6 2.5 0 5 0 6-1L12 17" {_S}/>',
    "eraser": f'<path d="M9 19.5l-5-5 9.5-9.5 6.5 6.5-8 8z" {_S}/><path d="M9.5 9l6 6M9 19.5h11" {_S}/>',
    "polygon": f'<path d="M5 18l2-12 9-2 4 9-7 6z" {_S}/><circle cx="5" cy="18" r="1.5" fill="C"/><circle cx="7" cy="6" r="1.5" fill="C"/>'
               f'<circle cx="16" cy="4" r="1.5" fill="C"/><circle cx="20" cy="13" r="1.5" fill="C"/><circle cx="13" cy="19" r="1.5" fill="C"/>',
    "rect": f'<rect x="4" y="6" width="16" height="12" rx="1" {_S}/>',
    "circle": f'<circle cx="12" cy="12" r="8" {_S}/>',
    "target": f'<circle cx="12" cy="12" r="3" {_S}/><path d="M12 2.5v5M12 16.5v5M2.5 12h5M16.5 12h5" {_S}/>'
              f'<path d="M5.5 5.5l2 2M16.5 16.5l2 2M18.5 5.5l-2 2M7.5 16.5l-2 2" {_S}/>',
    "hand": f'<path d="M8 12V6.5a1.5 1.5 0 0 1 3 0V11M11 10.5V5a1.5 1.5 0 0 1 3 0v6M14 6.5a1.5 1.5 0 0 1 3 0V13" {_S}/>'
            f'<path d="M17 9.5a1.5 1.5 0 0 1 3 0v4c0 4-2.5 7-6.5 7-3 0-4.5-1.5-6-4l-2.5-4a1.5 1.5 0 0 1 2.5-1.5L8 12" {_S}/>',
    "magic": f'<path d="M4 20L15 9M13.5 7.5l3 3" {_S}/><path d="M18 3v3M16.5 4.5h3M20 10v2M19 11h2M10 3v2M9 4h2" {_S}/>',
    "undo": f'<path d="M9 7.5L4.5 12 9 16.5" {_S}/><path d="M4.5 12H15a4.5 4.5 0 0 1 0 9h-2" {_S}/>',
    "redo": f'<path d="M15 7.5l4.5 4.5-4.5 4.5" {_S}/><path d="M19.5 12H9a4.5 4.5 0 0 0 0 9h2" {_S}/>',
    "trash": f'<path d="M4.5 7h15M9.5 7V4.5h5V7M6.5 7l1 13h9l1-13M10.5 11v5.5M13.5 11v5.5" {_S}/>',
    # --- view
    "zoomin": f'<circle cx="10.5" cy="10.5" r="6.5" {_S}/><path d="M15.5 15.5l5 5M8 10.5h5M10.5 8v5" {_S}/>',
    "zoomout": f'<circle cx="10.5" cy="10.5" r="6.5" {_S}/><path d="M15.5 15.5l5 5M8 10.5h5" {_S}/>',
    "fit": f'<path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5" {_S}/>',
    # --- misc
    "check": f'<path d="M5 12.5l4.5 4.5L19 7.5" {_S}/>',
    "warn": f'<path d="M12 3.5l9.5 16.5h-19z" {_S}/><path d="M12 10v4.5" {_S}/><circle cx="12" cy="17.2" r="1" fill="C"/>',
    "info": f'<circle cx="12" cy="12" r="8.5" {_S}/><path d="M12 11v5.5" {_S}/><circle cx="12" cy="7.8" r="1" fill="C"/>',
    "star": f'<path d="M12 3.5l2.6 5.4 5.9.8-4.3 4.1 1 5.9L12 16.9l-5.2 2.8 1-5.9-4.3-4.1 5.9-.8z" {_S}/>',
    "starfill": '<path d="M12 3.5l2.6 5.4 5.9.8-4.3 4.1 1 5.9L12 16.9l-5.2 2.8 1-5.9-4.3-4.1 5.9-.8z" fill="C" stroke="C" stroke-width="1.2" stroke-linejoin="round"/>',
    "plus": f'<path d="M12 5v14M5 12h14" {_S}/>',
    "close": f'<path d="M6 6l12 12M18 6L6 18" {_S}/>',
    "folder": f'<path d="M3.5 6.5V19h17V8.5h-8.5l-2-2.5h-6.5z" {_S}/>',
    "upload": f'<path d="M12 15.5V4M7.5 8.5L12 4l4.5 4.5" {_S}/><path d="M4 14v5.5h16V14" {_S}/>',
    "spool": f'<circle cx="12" cy="12" r="8.5" {_S}/><circle cx="12" cy="12" r="2.5" {_S}/><path d="M12 3.5v6M12 14.5v6" {_S}/>',
    "spools": f'<circle cx="7" cy="7" r="3.5" {_S}/><circle cx="17" cy="7" r="3.5" {_S}/><circle cx="7" cy="17" r="3.5" {_S}/>'
              f'<circle cx="17" cy="17" r="3.5" {_S}/><circle cx="7" cy="7" r="0.8" fill="C"/><circle cx="17" cy="7" r="0.8" fill="C"/>'
              f'<circle cx="7" cy="17" r="0.8" fill="C"/><circle cx="17" cy="17" r="0.8" fill="C"/>',
    "dice": f'<rect x="4" y="4" width="16" height="16" rx="3" {_S}/><circle cx="9" cy="9" r="1.2" fill="C"/>'
            f'<circle cx="15" cy="15" r="1.2" fill="C"/><circle cx="15" cy="9" r="1.2" fill="C"/><circle cx="9" cy="15" r="1.2" fill="C"/>',
    "light": f'<circle cx="12" cy="12" r="8.5" {_S}/><path d="M12 3.5a8.5 8.5 0 0 1 0 17z" fill="C"/>',
}

# the brand mark: a 3x3 mosaic with a quarter sun in red and one gold tile
LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
<rect x="1" y="1" width="30" height="30" rx="4" fill="{bg}"/>
<path d="M4 4h12a12 12 0 0 1-12 12z" fill="{red}"/>
<rect x="18" y="4" width="10" height="5" rx="1" fill="{fg}"/>
<rect x="18" y="11" width="10" height="5" rx="1" fill="{fg}" opacity=".55"/>
<rect x="4" y="18" width="5" height="10" rx="1" fill="{fg}" opacity=".55"/>
<rect x="11" y="18" width="5" height="10" rx="1" fill="{fg}"/>
<rect x="18" y="18" width="10" height="10" rx="1" fill="{gold}"/>
</svg>"""


def svg(name: str, colour: str) -> str:
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
            + PATHS[name].replace('"C"', f'"{colour}"') + "</svg>")


def _render(svg_text: str, size: int, dpr: float = 2.0) -> QPixmap:
    r = QSvgRenderer(QByteArray(svg_text.encode()))
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    r.render(p, QRectF(0, 0, size * dpr, size * dpr))
    p.end()
    pm.setDevicePixelRatio(dpr)
    return pm


@lru_cache(maxsize=512)
def pixmap(name: str, colour: str, size: int = 20) -> QPixmap:
    return _render(svg(name, colour), size)


@lru_cache(maxsize=512)
def icon(name: str, colour: str, size: int = 20, checked_colour: str | None = None) -> QIcon:
    ic = QIcon()
    ic.addPixmap(pixmap(name, colour, size), QIcon.Normal, QIcon.Off)
    if checked_colour:
        ic.addPixmap(pixmap(name, checked_colour, size), QIcon.Normal, QIcon.On)
    return ic


@lru_cache(maxsize=16)
def logo(size: int, bg: str, fg: str, red: str, gold: str) -> QPixmap:
    return _render(LOGO_SVG.format(bg=bg, fg=fg, red=red, gold=gold), size)


def check_svg_path() -> str:
    """Path to the white check mark used by QCheckBox (style sheets need a file)."""
    import os
    import tempfile
    p = os.path.join(tempfile.gettempdir(), "tessera_check.svg")
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(svg("check", "#FFFFFF").replace('stroke-width="1.7"', 'stroke-width="3"'))
    return p.replace("\\", "/")
