"""1:1 paper assembly template (R6-2): the artwork at true size, tiled on A4 or A3 pages with
overlap, registration marks, page grid labels, a scale check bar, every stone outline in its
colour and its code. PDF pages are exactly A4/A3 at 1:1 when printed at 100 % ("gerçek boyut")."""
from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw

from ..geometry.text import ui_font

PAPER = {"A4": (210.0, 297.0), "A3": (297.0, 420.0)}
MARGIN = 10.0          # unprintable border on most printers (mm)
OVERLAP = 10.0         # neighbouring pages share this strip (mm)
DPI = 150


def page_grid(w_mm, h_mm, paper="A4"):
    pw, ph = PAPER[paper]
    cw, ch = pw - 2 * MARGIN, ph - 2 * MARGIN
    cols = max(1, math.ceil((w_mm - OVERLAP) / (cw - OVERLAP)))
    rows = max(1, math.ceil((h_mm - OVERLAP) / (ch - OVERLAP)))
    return cols, rows, cw, ch


def write_template(L, out_dir: str, paper: str = "A4") -> str:
    """L: StoneLayout. Returns the PDF path."""
    m = L.mosaic
    pal = m.settings.palette
    w_mm, h_mm = m.size_mm
    cols, rows, cw, ch = page_grid(w_mm, h_mm, paper)
    pw, ph = PAPER[paper]
    k = DPI / 25.4                      # px per mm
    W, H = int(round(pw * k)), int(round(ph * k))
    step_x, step_y = cw - OVERLAP, ch - OVERLAP
    pages = []
    font_small = ui_font(max(9, int(1.8 * k)))
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for r in range(rows):
        for c in range(cols):
            x0 = c * step_x                       # artwork mm at the left content edge
            ytop = h_mm - r * step_y              # artwork mm (y up) at the top content edge
            im = Image.new("RGB", (W, H), "white")
            d = ImageDraw.Draw(im)

            def P(x, y):
                return ((MARGIN + x - x0) * k, (MARGIN + ytop - y) * k)
            box = (x0 - 1, ytop - ch - 1, x0 + cw + 1, ytop + 1)
            for st in L.stones:
                bx0, by0, bx1, by1 = st.poly.bounds
                if bx1 < box[0] or bx0 > box[2] or by1 < box[1] or by0 > box[3]:
                    continue
                rgb = pal[st.colour].rgb
                tint = tuple(int(255 - (255 - v) * 0.35) for v in rgb)
                edge = tuple(int(v * 0.75) for v in rgb) if sum(rgb) > 200 else (60, 60, 60)
                pts = [P(x, y) for x, y in st.poly.exterior.coords]
                d.polygon(pts, fill=tint, outline=edge, width=max(1, int(0.25 * k)))
                cpt = st.poly.representative_point()
                if st.width >= 4.5:
                    lab = st.code if st.width >= 9 else st.code.split("-")[-1]
                    d.text(P(cpt.x, cpt.y), lab, font=font_small, fill=(20, 20, 20), anchor="mm")
            # white margins, overlap shading, registration marks
            d.rectangle((0, 0, W, MARGIN * k), fill="white")
            d.rectangle((0, H - MARGIN * k, W, H), fill="white")
            d.rectangle((0, 0, MARGIN * k, H), fill="white")
            d.rectangle((W - MARGIN * k, 0, W, H), fill="white")
            cx0, cy0, cx1, cy1 = MARGIN * k, MARGIN * k, W - MARGIN * k, H - MARGIN * k
            d.rectangle((cx0, cy0, cx1, cy1), outline=(0, 0, 0), width=1)
            ov = OVERLAP * k
            for (ax, ay) in ((cx0, cy0), (cx1, cy0), (cx0, cy1), (cx1, cy1),
                             (cx1 - ov, cy0), (cx1 - ov, cy1), (cx0, cy1 - ov), (cx1, cy1 - ov)):
                d.line((ax - 12, ay, ax + 12, ay), fill=(200, 0, 0), width=2)
                d.line((ax, ay - 12, ax, ay + 12), fill=(200, 0, 0), width=2)
                d.ellipse((ax - 6, ay - 6, ax + 6, ay + 6), outline=(200, 0, 0), width=2)
            if c < cols - 1:
                d.line((cx1 - ov, cy0, cx1 - ov, cy1), fill=(200, 0, 0), width=1)
            if r < rows - 1:
                d.line((cx0, cy1 - ov, cx1, cy1 - ov), fill=(200, 0, 0), width=1)
            name = f"{letters[r % 26]}{c + 1}"
            f = ui_font(int(4.0 * k))
            d.text((MARGIN * k, MARGIN * k * 0.45), f"Sayfa {name}  ({r + 1}. satır, {c + 1}. sütun / {rows}×{cols})",
                   font=f, fill=(0, 0, 0), anchor="lm")
            # scale check bar: 100 mm
            bx, by = MARGIN * k, H - MARGIN * k * 0.5
            d.line((bx, by, bx + 100 * k, by), fill=(0, 0, 0), width=3)
            for t in range(0, 101, 10):
                d.line((bx + t * k, by - 6, bx + t * k, by + (8 if t % 50 == 0 else 4)), fill=(0, 0, 0), width=1)
            d.text((bx + 104 * k, by), "100 mm: %100 (gerçek boyut) yazdırın; cetvelle kontrol edin",
                   font=ui_font(int(2.6 * k)), fill=(0, 0, 0), anchor="lm")
            nbrs = []
            if c < cols - 1:
                nbrs.append(f"sağ: {letters[r % 26]}{c + 2}")
            if r < rows - 1:
                nbrs.append(f"alt: {letters[(r + 1) % 26]}{c + 1}")
            if nbrs:
                d.text((W - MARGIN * k, MARGIN * k * 0.45), "  ".join(nbrs) + " (kırmızı şerit üst üste)",
                       font=ui_font(int(2.6 * k)), fill=(150, 0, 0), anchor="rm")
            pages.append(im)
    path = os.path.join(out_dir, f"00_sablon_1-1_{paper}.pdf")
    pages[0].save(path, save_all=True, append_images=pages[1:], resolution=float(DPI))
    if paper == "A4":           # small page previews for the UI / quick checks
        pdir = os.path.join(out_dir, "00_sablon_onizleme")
        os.makedirs(pdir, exist_ok=True)
        for i, pg in enumerate(pages):
            r, c = divmod(i, cols)
            pg.resize((pg.width // 2, pg.height // 2)).save(os.path.join(pdir, f"A4_{letters[r % 26]}{c + 1}.png"))
    return path
