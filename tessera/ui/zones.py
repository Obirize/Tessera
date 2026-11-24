"""Zone document: fractional shapes + a painted mask, with undo/redo.

The painted mask is a Grayscale8 QImage (so QPainter can paint into it directly) at a fixed
width of MASK_W pixels. Values follow tessera.project: 0..5 Vocab, 254 = eraser (back to the
default vocabulary), 255 = nothing painted.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen

from ..model import Vocab
from ..project import PAINT_DEFAULT, PAINT_NONE, apply_paint
from .theme import VOCAB_COLOURS

MASK_W = 1000
UNDO_LIMIT = 30


def qimage_to_gray(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format_Grayscale8)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    arr = np.frombuffer(img.constBits(), np.uint8, count=bpl * h).reshape(h, bpl)
    return arr[:, :w].copy()


def gray_to_qimage(a: np.ndarray) -> QImage:
    a = np.ascontiguousarray(a.astype(np.uint8))
    h, w = a.shape
    return QImage(a.data, w, h, w, QImage.Format_Grayscale8).copy()


def rgb_to_qimage(a: np.ndarray) -> QImage:
    a = np.ascontiguousarray(a[..., :3].astype(np.uint8))
    h, w = a.shape[:2]
    return QImage(a.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def rgba_to_qimage(a: np.ndarray) -> QImage:
    a = np.ascontiguousarray(a.astype(np.uint8))
    h, w = a.shape[:2]
    return QImage(a.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()


_LUT = np.zeros((256, 3), np.uint8)
for _v, _hex in VOCAB_COLOURS.items():
    _c = QColor(_hex)
    _LUT[_v] = (_c.red(), _c.green(), _c.blue())


def vocab_rgb(v: int) -> tuple[int, int, int]:
    return tuple(int(c) for c in _LUT[int(v)])  # type: ignore[return-value]


class ZoneDoc(QObject):
    changed = Signal()          # zones changed (rebuild mosaic)
    overlay_changed = Signal()  # overlay image must be redrawn

    def __init__(self, parent=None):
        super().__init__(parent)
        self.shapes: list[dict] = []
        self.rays_center: Optional[tuple[float, float]] = None
        self.default_vocab: int = int(Vocab.STONES)
        self.aspect = 1.0                    # h / w of the image
        self.mask = QImage()
        self._overlay = QImage()
        self._used: Optional[set[int]] = None
        self._undo: list = []
        self._redo: list = []
        self.reset(1.0)

    # ---------------------------------------------- setup
    def reset(self, aspect: float, shapes=None, mask: Optional[np.ndarray] = None,
              rays_center=None, default_vocab: Optional[int] = None) -> None:
        self.aspect = max(0.05, float(aspect))
        h = max(8, int(round(MASK_W * self.aspect)))
        self.mask = QImage(MASK_W, h, QImage.Format_Grayscale8)
        self.mask.fill(PAINT_NONE)
        if mask is not None:
            from PIL import Image
            m = mask if mask.shape == (h, MASK_W) else np.asarray(
                Image.fromarray(mask.astype(np.uint8), "L").resize((MASK_W, h), Image.NEAREST))
            self.mask = gray_to_qimage(m)
        self.shapes = copy.deepcopy(shapes or [])
        self.rays_center = tuple(rays_center) if rays_center else None
        if default_vocab is not None:
            self.default_vocab = int(default_vocab)
        self._undo.clear()
        self._redo.clear()
        self.recompose()

    @property
    def size(self) -> tuple[int, int]:
        return self.mask.width(), self.mask.height()

    def mask_array(self) -> Optional[np.ndarray]:
        a = qimage_to_gray(self.mask)
        return a if (a != PAINT_NONE).any() else None

    def has_paint(self) -> bool:
        return self.mask_array() is not None

    def zone_map(self) -> np.ndarray:
        from ..pipeline import zones_from_shapes
        w, h = self.size
        z = zones_from_shapes((h, w), self.shapes, default=self.default_vocab)
        return apply_paint(z, qimage_to_gray(self.mask), self.default_vocab)

    def used_vocabs(self) -> set[int]:
        if self._used is None:
            z = self.zone_map()
            self._used = set(int(v) for v in np.flatnonzero(np.bincount(z.ravel(), minlength=len(Vocab))))
        return self._used

    # ---------------------------------------------- overlay
    def recompose(self) -> None:
        z = self.zone_map()
        self._used = set(int(v) for v in np.flatnonzero(np.bincount(z.ravel(), minlength=len(Vocab))))
        rgba = np.empty(z.shape + (4,), np.uint8)
        rgba[..., :3] = _LUT[z]
        rgba[..., 3] = 255
        self._overlay = rgba_to_qimage(rgba)
        self.overlay_changed.emit()

    def overlay(self) -> QImage:
        return self._overlay

    # ---------------------------------------------- undo
    def _snapshot(self):
        return (copy.deepcopy(self.shapes), self.mask.copy(), self.rays_center, self.default_vocab)

    def _restore(self, snap) -> None:
        self.shapes, self.mask, self.rays_center, self.default_vocab = \
            copy.deepcopy(snap[0]), snap[1].copy(), snap[2], snap[3]
        self.recompose()
        self.changed.emit()

    def push_undo(self) -> None:
        self._undo.append(self._snapshot())
        del self._undo[:-UNDO_LIMIT]
        self._redo.clear()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> None:
        if self._undo:
            self._redo.append(self._snapshot())
            self._restore(self._undo.pop())

    def redo(self) -> None:
        if self._redo:
            self._undo.append(self._snapshot())
            self._restore(self._redo.pop())

    # ---------------------------------------------- edits
    def add_shape(self, shape: dict) -> None:
        self.push_undo()
        self.shapes.append(shape)
        self.recompose()
        self.changed.emit()

    def remove_shape(self, idx: int) -> None:
        if 0 <= idx < len(self.shapes):
            self.push_undo()
            del self.shapes[idx]
            self.recompose()
            self.changed.emit()

    def set_shape_vocab(self, idx: int, vocab: int) -> None:
        if 0 <= idx < len(self.shapes) and self.shapes[idx].get("vocab") != vocab:
            self.push_undo()
            self.shapes[idx]["vocab"] = int(vocab)
            self.recompose()
            self.changed.emit()

    def set_default_vocab(self, v: int) -> None:
        if int(v) != self.default_vocab:
            self.push_undo()
            self.default_vocab = int(v)
            self.recompose()
            self.changed.emit()

    def set_rays_center(self, frac: Optional[tuple[float, float]]) -> None:
        self.push_undo()
        self.rays_center = frac
        self.overlay_changed.emit()
        self.changed.emit()

    def clear_paint(self) -> None:
        if self.has_paint():
            self.push_undo()
            self.mask.fill(PAINT_NONE)
            self.recompose()
            self.changed.emit()

    def clear_all(self) -> None:
        self.push_undo()
        self.shapes = []
        self.mask.fill(PAINT_NONE)
        self.recompose()
        self.changed.emit()

    def replace(self, shapes=None, mask: Optional[np.ndarray] = None, rays_center=None,
                default_vocab: Optional[int] = None) -> None:
        """Apply an automatic suggestion as one undoable step."""
        self.push_undo()
        if shapes is not None:
            self.shapes = copy.deepcopy(shapes)
        if mask is not None:
            from PIL import Image
            w, h = self.size
            m = mask if mask.shape == (h, w) else np.asarray(
                Image.fromarray(mask.astype(np.uint8), "L").resize((w, h), Image.NEAREST))
            self.mask = gray_to_qimage(m)
        elif shapes is not None:
            self.mask.fill(PAINT_NONE)
        if rays_center is not None:
            self.rays_center = tuple(rays_center)
        if default_vocab is not None:
            self.default_vocab = int(default_vocab)
        self.recompose()
        self.changed.emit()

    # ---------------------------------------------- painting (mask pixel coords)
    def begin_stroke(self) -> None:
        self.push_undo()

    def paint_segment(self, a: QPointF, b: QPointF, radius_px: float, value: int) -> None:
        """value: Vocab int, or PAINT_DEFAULT for the eraser."""
        pen = QPen(QColor(value, value, value), max(1.0, radius_px * 2), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p = QPainter(self.mask)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.setPen(pen)
        if a == b:
            p.drawPoint(a)
        else:
            p.drawLine(a, b)
        p.end()
        shown = self.default_vocab if value == PAINT_DEFAULT else value
        r, g, bl = vocab_rgb(shown)
        p = QPainter(self._overlay)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.setPen(QPen(QColor(r, g, bl), max(1.0, radius_px * 2), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if a == b:
            p.drawPoint(a)
        else:
            p.drawLine(a, b)
        p.end()
        self.overlay_changed.emit()

    def end_stroke(self) -> None:
        self.recompose()
        self.changed.emit()


def shape_kind_tr(s: dict) -> str:
    if "polygon" in s:
        return "çokgen"
    if "rect" in s:
        return "dikdörtgen"
    if "circle" in s:
        return "daire"
    return "şekil"
