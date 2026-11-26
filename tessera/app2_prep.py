"""The "Görseli hazırla" window: crop, outline, size and piece count of a picture, with the real stones beside it.

``Prep`` holds a draft recipe. Nothing of it reaches the project before accept(). The page talks to it over
the web channel as ``prep``; the stones come from the engine's draft job.
"""
from __future__ import annotations

import json
from typing import Optional

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from . import app2_engine as E

WHOLE = (0.0, 0.0, 1.0, 1.0)  # kadraj.WHOLE; the pipeline is not loaded at start (it takes seconds)
SIDE = 1400              # longest side of the picture the page shows, px
WAIT_MS = 250            # a drag is over when nothing has come for this long


class Prep(QObject):
    state = Signal(str)          # everything the window shows; {"open": false} closes it
    result = Signal(str)         # the stones of the newest draft (picture url + numbers)
    progress = Signal(str)       # {"f": 0..1, "msg"}; {"err": ...} when it failed

    def __init__(self, bridge):
        super().__init__(bridge)
        self.bridge = bridge
        self.recipe: Optional[E.Recipe] = None
        self.new = False                          # a picture that was just opened, not the one in the project
        self.gen = 0
        self._find = False                        # the colours have to be looked for (again)
        self._size = (1, 1)
        self._im = None                           # the picture as shown, not turned
        self._src: dict = {}
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._start)

    # -------------------------------------------------- page
    @Slot(str, result=bool)
    def open(self, path: str = "") -> bool:
        """A new picture ('' asks for one); a project file opens as a project."""
        r = self.bridge.new_recipe(path)
        if r is None:
            return False
        return self._begin(r, True)

    @Slot(result=bool)
    def edit(self) -> bool:
        """The picture that is in the project."""
        return bool(self.bridge.recipe.image_path) and self._begin(self.bridge.recipe.copy(), False)

    def _begin(self, r: E.Recipe, new: bool) -> bool:
        from PIL import Image
        from .resim import open_rgb
        im = open_rgb(r.image_path)
        self._size = im.size
        im.thumbnail((SIDE, SIDE), Image.BOX)
        self.recipe, self.new, self._im, self._src = r, new, im, {}
        self._find = not r.colours
        self._emit()
        self._timer.start(0)
        return True

    @Slot(str)
    def set(self, patch: str) -> None:
        """{"kutu": [...]} | {"sekil": {...}} | {"op": "don"|"ayna"} | {"width_cm"} | {"parca"} | {"tek_taban"}
        | {"zemin": true|false} | {"tik": [x, y, "tek"|"renk"]} (a click on the frame)
        | {"cizgi": ["kalsin"|"gitsin", [[x, y], ...]]} (a line drawn on a photograph) | {"geri": true} (undo the last)"""
        r = self.recipe
        if r is None:
            return
        d = json.loads(patch or "{}")
        k = dict(r.kadraj or {"don": 0, "ayna": False, "kutu": list(WHOLE)})
        z = (r.sekil or {}).get("zemin")
        sekil = dict(d["sekil"] if "sekil" in d else r.sekil or {"tur": "dikdortgen"})
        if "zemin" in d:
            z = {"tiklar": []} if d["zemin"] else None
        if z is not None:
            z = {"tiklar": list(z["tiklar"]), "cizgiler": list(z.get("cizgiler") or [])}
            if d.get("tik"):
                z["tiklar"].append(d["tik"])
            if d.get("cizgi"):
                z["cizgiler"].append(d["cizgi"])
            if d.get("geri"):
                (z["cizgiler"] or z["tiklar"] or [None]).pop()
        was = dict(k)
        if d.get("kutu"):
            k["kutu"] = d["kutu"]
        x0, y0, x1, y1 = k["kutu"]
        if d.get("op") == "don":                  # a quarter turn of what is seen, clockwise
            k["don"] = (k["don"] + (270 if k["ayna"] else 90)) % 360
            k["kutu"] = [1 - y1, x0, 1 - y0, x1]
        elif d.get("op") == "ayna":
            k["ayna"] = not k["ayna"]
            k["kutu"] = [1 - x1, y0, 1 - x0, y1]
        if (sekil or {}).get("tur") == "daire":
            k["kutu"] = square(k["kutu"], *self.turned_size(k))
        turned = (k["don"], k["ayna"]) != (was["don"], was["ayna"])
        moved = turned or max(abs(float(u) - v) for u, v in zip(k["kutu"], was["kutu"])) > 1e-4
        if z is not None and moved:
            z = {"tiklar": []}                    # the clicks and lines were on another part of the picture
        sekil.pop("zemin", None)
        if z is not None:
            sekil["zemin"] = z
        before = r.build_key()
        self._find |= r.set_frame(k, sekil)
        if "width_cm" in d:
            try:
                r.width_mm = 10 * max(10.0, min(300.0, float(d["width_cm"])))
            except (TypeError, ValueError):
                pass
        if "parca" in d:
            r.fine["parca"] = E.piece_count(d["parca"])
        if "tek_taban" in d:
            r.fine["tek_taban"] = d["tek_taban"] if d["tek_taban"] in [h for _, h in E.unique_names(r.colours)] else None
        self._emit()
        if r.build_key() != before:
            self._timer.start(WAIT_MS)

    @Slot()
    def accept(self) -> None:
        r, find, new = self.recipe, self._find, self.new
        if r is None:
            return
        same = not new and not find and r.build_key() == self.bridge.recipe.build_key()
        self._close(keep=True)
        if not same:
            self.bridge.adopt(r, new, find)

    @Slot()
    def cancel(self) -> None:
        if self.recipe is not None:
            self._close()
            self.bridge.ask_thumbs()              # a draft may have cut them short

    def _close(self, keep: bool = False) -> None:
        self._timer.stop()
        self.gen += 1
        self.bridge.engine.cancel_draft(keep)
        self.recipe = self._im = None
        self._src = {}
        self.state.emit(json.dumps({"open": False}))

    # -------------------------------------------------- what the window shows
    def turned_size(self, k) -> tuple:
        w, h = self._size
        return (h, w) if (k or {}).get("don", 0) % 180 else (w, h)

    def state_dict(self) -> dict:
        r = self.recipe
        if r is None:
            return {"open": False}
        k = r.kadraj or {"don": 0, "ayna": False, "kutu": list(WHOLE)}
        w, h = self.turned_size(k)
        return {"open": True, "yeni": self.new, "ad": r.name or E.Project(image_path=r.image_path).display_name(),
                "src": self._picture(k), "w": w, "h": h, "kutu": k["kutu"], "sekil": r.sekil or {"tur": "dikdortgen"},
                "zemin": (r.sekil or {}).get("zemin"),
                "width_cm": round(r.width_mm / 10, 1), "parca": r.fine.get("parca"), "tek_taban": r.fine.get("tek_taban"),
                "hexes": dict(E.unique_names(r.colours)), "mode": r.mode, "gen": self.gen}

    def _emit(self) -> None:
        self.state.emit(json.dumps(self.state_dict(), ensure_ascii=False))

    def _picture(self, k) -> str:
        key = (k["don"], k["ayna"])
        if key not in self._src:
            import io
            from .pipeline.kadraj import turned
            b = io.BytesIO()
            turned(self._im, k).save(b, "JPEG", quality=90)
            self.bridge.serial += 1
            self._src[key] = self.bridge.blobs.put(f"prep_{self.bridge.serial}.jpg", b.getvalue(), "image/jpeg", keep=64)
        return self._src[key]

    # -------------------------------------------------- the stones
    def _start(self) -> None:
        if self.recipe is not None:
            self.gen += 1
            self.progress.emit(json.dumps({"f": 0.0, "msg": "Taşlar hesaplanıyor"}))
            self.bridge.engine.draft(self.gen, self.recipe, self._find)

    def got(self, gen: int, d: dict) -> None:
        """From the engine (on the UI thread)."""
        if gen != self.gen or self.recipe is None:
            return
        if "f" in d:
            self.progress.emit(json.dumps({"f": d["f"], "msg": d["msg"]}, ensure_ascii=False))
            return
        if "colours" in d:
            self._find = False
            self.recipe.use_colours(d.pop("colours"))
            self.recipe.uygun = d.pop("uygun")
            self._emit()
        if "zemin" in d:                          # the background as found: it comes before the stones
            z = d.pop("zemin")
            self.bridge.serial += 1
            z["url"] = self.bridge.blobs.put(f"prep_{self.bridge.serial}.png", z.pop("png"), "image/png", keep=64)
            self.result.emit(json.dumps({"zemin": z}, ensure_ascii=False))
            if not d:
                return
        jpg = d.pop("jpg", None)
        if jpg:
            self.bridge.serial += 1
            d["url"] = self.bridge.blobs.put(f"prep_{self.bridge.serial}.jpg", jpg, "image/jpeg", keep=64)
        d["soz"] = (self.recipe.uygun or {}).get("soz", [])
        d["puan"] = (self.recipe.uygun or {}).get("puan", "iyi")
        self.result.emit(json.dumps(E.plain(d), ensure_ascii=False))

    def failed(self, gen: int, msg: str) -> None:
        if gen == self.gen and self.recipe is not None:
            self.progress.emit(json.dumps({"err": msg}, ensure_ascii=False))


def square(kutu, w: int, h: int) -> list:
    """The largest square in the middle of a box (fractions of a w x h picture)."""
    x0, y0, x1, y1 = kutu
    bw, bh = (x1 - x0) * w, (y1 - y0) * h
    s = min(bw, bh)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return [cx - s / w / 2, cy - s / h / 2, cx + s / w / 2, cy + s / h / 2]
