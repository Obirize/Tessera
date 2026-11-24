"""Start here: the opening card first (it is light), then the program itself."""
import os
import sys


def main() -> int:
    from PySide6.QtCore import QSettings
    v = QSettings("Tessera", "Tessera").value("acilis", True)
    on = (v if isinstance(v, bool) else str(v).lower() != "false") and not os.environ.get("TESSERA_NO_SPLASH")
    card = None
    if on:
        from . import acilis
        card = acilis.spawn()
    from . import app2                            # the heavy part loads while the card plays
    return app2.main(sys.argv, card=card)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
