"""Tessera desktop app (PySide6). Entry point: ``python -m tessera`` or ``run()``."""
from __future__ import annotations

import sys


def create_app(argv=None):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    from . import theme
    app = QApplication.instance()
    if app is None:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        app = QApplication(list(argv or sys.argv))
    app.setApplicationName("Tessera")
    app.setOrganizationName("Tessera")
    theme.load_fonts()
    return app


def run(argv=None) -> int:
    argv = list(argv or sys.argv)
    app = create_app(argv)
    from . import icons, theme
    from .main_window import MainWindow
    t = theme.current()
    app.setWindowIcon(__import__("PySide6.QtGui", fromlist=["QIcon"]).QIcon(
        icons.logo(64, t.text, t.bg, t.red, t.gold)))
    win = MainWindow()
    win.show()
    if len(argv) > 1:
        win.open_any(argv[1])
    return app.exec()
