"""Assembly guide (Turkish): overview + mounting page + one page per panel. PNG pages + PDF."""
from __future__ import annotations

import os

import numpy as np
import shapely
import shapely.ops
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from ..geometry.layout import FINISH_TR, Layout
from ..geometry.text import ui_font
from ..model import Mode

PAGE_W, PAGE_H = 1240, 1754          # A4 @ 150 dpi
INK = (28, 28, 30)
LATTICE = (40, 38, 36)


def _colour_raster(L: Layout, lattice=LATTICE, numbered=True):
    """Colours exactly as printed (tile colours are never changed); part borders in the
    lattice colour, grooves slightly darker."""
    pal = np.array([f.rgb for f in L.mosaic.settings.palette], np.float32)
    lab = L.label
    img = pal[L.tile_colour[lab]].copy()
    part = L.tile_part[lab]
    g = np.zeros(lab.shape, bool)
    g[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    g[1:, :] |= lab[1:, :] != lab[:-1, :]
    img[g] *= 0.78
    if numbered:
        b = np.zeros(lab.shape, bool)
        d = part[:, 1:] != part[:, :-1]
        b[:, 1:] |= d
        b[:, :-1] |= d
        d = part[1:, :] != part[:-1, :]
        b[1:, :] |= d
        b[:-1, :] |= d
        img[b] = lattice
    return np.clip(img, 0, 255).astype(np.uint8), part


def _text(d, xy, s, size, fill=INK, anchor="la", stroke=0, stroke_fill=(255, 255, 255)):
    d.text(xy, s, font=ui_font(size), fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=stroke_fill)


def _fit(im: Image.Image, w, h, resample=Image.NEAREST):
    k = min(w / im.width, h / im.height)
    return im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), resample), k


def _lines(d, x, y, lines, size=21, gap=32, width=1120):
    font = ui_font(size)
    for ln in lines:
        words, cur = ln.split(" "), ""
        for w in words:
            t = (cur + " " + w).strip()
            if d.textlength(t, font=font) > width and cur:
                d.text((x, y), cur, font=font, fill=INK)
                y += gap
                cur = w
            else:
                cur = t
        d.text((x, y), cur, font=font, fill=INK)
        y += gap
    return y


def write_guide(L: Layout, out_dir: str, mode: str, backing_ci: int, plate_of_part: dict,
                plates_info: list) -> list[str]:
    m = L.mosaic
    s = m.settings
    mm = s.mm_per_px
    H, W = L.label.shape
    pal = s.palette
    pieces = mode == Mode.PIECES
    rgb, part_r = _colour_raster(L, numbered=pieces)
    pages = []
    w_mm, h_mm = m.size_mm
    nx, ny = L.grid

    # ---------------- page 1: overview
    pg = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    d = ImageDraw.Draw(pg)
    _text(d, (60, 50), "Tessera — Montaj kılavuzu", 46)
    mode_tr = ("Tek renk tablalar · AMS gerekmez (her parça tek renk)" if pieces
               else "Çok renkli paneller · AMS gerekir (AMS = Bambu'nun çok renkli besleyicisi; her panel tek seferde basılır)")
    n_things = len(L.parts) if pieces else len(L.panels)
    _text(d, (60, 110), f"{mode_tr}", 22)
    _text(d, (60, 142), f"{w_mm / 10:.0f} × {h_mm / 10:.0f} cm · {nx} × {ny} = {len(L.panels)} panel · "
                        + (f"{n_things} parça" if pieces else f"{n_things} panel yerleştirilecek"), 24)
    art = Image.fromarray(rgb)
    pr = L.tile_panel[L.label]
    edge = np.zeros(pr.shape, bool)
    edge[:, 1:] |= pr[:, 1:] != pr[:, :-1]
    edge[1:, :] |= pr[1:, :] != pr[:-1, :]
    a = np.asarray(art).copy()
    a[ndi.binary_dilation(edge, iterations=2)] = (0, 120, 255)
    fitted, k = _fit(Image.fromarray(a), PAGE_W - 120, PAGE_H - 560, Image.LANCZOS)
    ox, oy = (PAGE_W - fitted.width) // 2, 190
    pg.paste(fitted, (ox, oy))
    for pn in L.panels:
        try:
            big = max(getattr(pn.region, "geoms", [pn.region]), key=lambda q: q.area)
            c = shapely.ops.polylabel(big, 1.0)
        except Exception:
            c = pn.region.representative_point()
        px, py = ox + c.x / mm * k, oy + (H - c.y / mm) * k
        d.ellipse((px - 38, py - 38, px + 38, py + 38), fill=(255, 255, 255), outline=(0, 120, 255), width=5)
        _text(d, (px, py), pn.name, 30, anchor="mm")
    y = oy + fitted.height + 24
    _text(d, (60, y), "Paneller duvarda bu düzende yan yana gelir (önden görünüş). Mavi çizgiler panel sınırları.", 21)
    y += 40
    for i, f in enumerate(pal):
        d.rectangle((60, y, 100, y + 30), fill=tuple(f.rgb), outline=INK)
        n = sum(1 for p in L.parts if p.colour == i)
        fin = FINISH_TR.get(f.finish, f.finish)
        extra = "  ← kafes (arka ızgara) rengi" if i == backing_ci else ""
        what = f"{n} parça" if pieces else f"{n} renk bölgesi"
        _text(d, (115, y + 2), f"{L.letters[i]} = {f.name} ({fin})  ·  {what}{extra}", 22)
        y += 40
    y += 6
    if pieces:
        ex = L.parts[0].pid if L.parts else "Kı2-031"
        lines = [f"Parça kodu: renk kodu + panel + sıra no. Örnek: {ex}. Kod parçanın altında aynalı yazılıdır; "
                 "aynı numara kafesteki cebin içinde de yazar.",
                 "Hassas parça (haritada pembe daire): numarası parçaya sığmaz; tabla görselinde ve kafes cebinde bulun."]
    else:
        lines = ["Çok renkli panellerde parça numarası yoktur: her panel renkleri yerinde, tek nesne olarak basılır."]
    _lines(d, 60, y, lines)
    pages.append(pg)

    # ---------------- page 2: board, joining, hanging
    pg = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    d = ImageDraw.Draw(pg)
    _text(d, (60, 50), "Levha, birleştirme ve duvara asma", 40)
    board_w, board_h = w_mm / 10, h_mm / 10
    y = _lines(d, 60, 130, [
        f"Gerekenler: {board_w:.0f} × {board_h:.0f} cm, 4–6 mm MDF ya da kontrplak levha "
        + ("(ŞART: kafes açık ızgaradır, levhasız taşınmaz)" if pieces else "(paneller birbirine levha ile bağlanır)")
        + "; ahşap/PLA yapıştırıcı ya da çift taraflı köpük bant; 6 vida (4 × 30 mm).",
        "1. Levhanın önüne kurşun kalemle bir dik açı çiz (sol alt köşe). Panelleri genel haritadaki düzende, "
        "P1 sol üstten başlayarak yerleştir.",
        "2. Panel kenarları taş sınırlarını izler, bu yüzden zikzaklıdır: komşu paneller yapboz gibi yalnız "
        "tek bir konumda birbirine geçer. Önce kuru dene, sonra her paneli arkasından yapıştır.",
        "3. Her kafesin ÜST (ön) yüzüne oyulmuş 'P1 ÜST ↑' yazısı panel adını ve yukarı yönü gösterir; "
        "kafesin arka yüzü düzdür.",
        ("4. Parçaları kafeslere yerleştir (panel sayfalarına bak). Parçalar ızgara çubuklarına oturur; "
         "her parçanın altına birkaç nokta yapıştırıcı sür.") if pieces else
        "4. AMS panelleri dolu tabanlıdır; arkalarına bolca yapıştırıcı sürüp levhaya bastır.",
        "5. Duvara asma (fransız çıta): " + ("kafes dosyasının son tablasındaki" if pieces else
                                             "0_Aski_citalari.3mf dosyasındaki") + " 2 askı çıtasını bas. "
        "Birini levhanın arkasına, üst kenardan 5 cm aşağıya, eğimli yüzü aşağı ve duvara bakacak şekilde "
        "3 vidayla tak. Diğerini duvara terazili olarak, eğimli yüzü yukarı ve duvardan dışarı bakacak "
        "şekilde dübel ve vidayla tak. Levhayı üstten çıtaya oturt: iki eğim birbirine kilitlenir.",
        "Hafif eserlerde (2 kg altı) çıta yerine levhanın arkasına iki testere dişli resim askısı da yeter.",
    ])
    # simple drawing of the cleat principle
    y += 30
    x0 = 200
    d.rectangle((x0, y, x0 + 40, y + 300), fill=(200, 190, 170), outline=INK)            # wall
    _text(d, (x0 - 10, y + 310), "duvar", 20)
    d.polygon([(x0 + 40, y + 60), (x0 + 70, y + 60), (x0 + 70, y + 90), (x0 + 40, y + 120)],
              fill=(120, 120, 120), outline=INK)                                       # wall cleat
    d.rectangle((x0 + 100, y + 20, x0 + 115, y + 300), fill=(170, 140, 100), outline=INK)  # board
    _text(d, (x0 + 125, y + 260), "MDF levha + kafes + parçalar", 20)
    d.polygon([(x0 + 100, y + 40), (x0 + 72, y + 40), (x0 + 72, y + 56), (x0 + 100, y + 84)],
              fill=(80, 80, 80), outline=INK)                                          # board cleat
    _text(d, (x0 + 125, y + 50), "← levhadaki çıta (eğim aşağı)", 20)
    _text(d, (x0 + 125, y + 100), "← duvardaki çıta (eğim yukarı)", 20)
    pages.append(pg)

    # ---------------- one page per panel
    plates_by_panel: dict[str, list] = {}
    for p in plates_info:
        for pn_name in p.get("panel", []):
            plates_by_panel.setdefault(pn_name, []).append(p)
    for pn in L.panels:
        pg = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        d = ImageDraw.Draw(pg)
        x0, y0, x1, y1 = pn.bounds
        c0, c1 = int(max(0, x0 / mm - 2)), int(min(W, x1 / mm + 2))
        r0, r1 = int(max(0, H - y1 / mm - 2)), int(min(H, H - y0 / mm + 2))
        crop = rgb[r0:r1, c0:c1].copy()
        mask = (L.tile_panel[L.label[r0:r1, c0:c1]] != pn.index)
        crop[mask] = (crop[mask] * 0.25 + 255 * 0.75).astype(np.uint8)
        fitted, k = _fit(Image.fromarray(crop), PAGE_W - 100, 1150)
        ox, oy = (PAGE_W - fitted.width) // 2, 120
        _text(d, (50, 40), f"Panel {pn.name}", 44)
        what = f"{len(pn.parts)} parça" if pieces else "tek nesne (çok renkli)"
        _text(d, (330, 52), f"satır {pn.row + 1}, sütun {pn.col + 1} · {pn.size[0]:.0f}×{pn.size[1]:.0f} mm · {what}", 24)
        pg.paste(fitted, (ox, oy))
        hassas = []
        if pieces:
            fs = int(np.clip(k * 2.6, 11, 22))
            for i in pn.parts:
                pt = L.parts[i]
                g = pt.outline if pt.outline is not None else pt.footprint
                try:
                    big = max(getattr(g, "geoms", [g]), key=lambda q: q.area)
                    c = shapely.ops.polylabel(big, 0.3)
                except Exception:
                    c = g.representative_point()
                px, py = ox + (c.x / mm - c0) * k, oy + (H - c.y / mm - r0) * k
                if pt.kind == "hassas":
                    hassas.append(pt.number)
                    r = max(6, fs * 0.7)
                    d.ellipse((px - r, py - r, px + r, py + r), outline=(255, 0, 160), width=2)
                    _text(d, (px + r + 2, py), str(pt.number), max(10, fs - 2), anchor="lm", stroke=2)
                else:
                    _text(d, (px, py), str(pt.number), fs, anchor="mm", stroke=2)
        y = oy + fitted.height + 20
        _text(d, (50, y), "Yerleştirme sırası", 28)
        y += 40
        if pieces:
            steps = [f"1. Kafes {pn.name} ({pal[backing_ci].name}) ızgarasını levhadaki yerine yapıştır; ön yüzdeki 'ÜST' oku yukarı.",
                     "2. Büyük parçalardan başla: her parça yalnızca kendi cebine oturur (çıkıntılı çerçeve).",
                     "3. Numaralar haritada, kafesin cebinde ve parçanın altında aynıdır.",
                     f"4. Koyu ince çizgiler ({pal[backing_ci].name}) kafesle birlikte basıldı; ayrı parça değildir."]
            if hassas:
                steps.append(f"5. Hassas parçalar (pembe daire): {', '.join(str(n) for n in sorted(hassas))}. Cımbızla, en son yerleştir.")
        else:
            steps = [f"1. Panel {pn.name} AMS ile tek seferde, renkleri yerinde basılır.",
                     "2. Paneli levhadaki yerine yapıştır; panel adının yanındaki 'ÜST' oku yukarı bakmalı."]
        y = _lines(d, 50, y, steps, 20, 30)
        y += 6
        _text(d, (50, y), "Bu panelin tablaları:", 24)
        y += 34
        lst = plates_by_panel.get(pn.name, [])
        by_file: dict[str, list] = {}
        for p in lst:
            label = "Kafes" if p.get("tur") == "kafes" else p["renk"]
            by_file.setdefault((label, p["dosya"]), []).append(p["tabla"])
        for (renk, dosya), nums in by_file.items():
            _text(d, (70, y), f"{renk}: {dosya} → tabla {', '.join(str(n) for n in sorted(set(nums)))}", 20)
            y += 28
        pages.append(pg)

    out = []
    names = ["01_montaj_00_genel.png", "01_montaj_00_levha_ve_aski.png"] + \
            [f"01_montaj_{pn.name}.png" for pn in L.panels]
    for name, pg in zip(names, pages):
        p = os.path.join(out_dir, name)
        pg.save(p, optimize=True)
        out.append(p)
    pdf = os.path.join(out_dir, "01_montaj_kilavuzu.pdf")
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=150.0)
    out.insert(0, pdf)
    return out


def write_guide_stones(L, out_dir: str, mode: str, plates_info: list) -> list[str]:
    """Round 6 guide: overview with the template page grid + how to assemble from the 1:1 template."""
    m = L.mosaic
    s = m.settings
    pal = s.palette
    pieces = mode == Mode.PIECES
    w_mm, h_mm = m.size_mm
    pages = []
    # overview: stones drawn in their colours, template pages as a grid
    pg = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    d = ImageDraw.Draw(pg)
    _text(d, (60, 50), "Tessera — Montaj kılavuzu", 46)
    mode_tr = ("Tek renk tablalar · AMS gerekmez (her taş ayrı basılır)" if pieces
               else "Çok renkli paneller · AMS gerekir (taşlar ince bir tabanın üstünde basılır)")
    _text(d, (60, 110), mode_tr, 22)
    _text(d, (60, 142), f"{w_mm / 10:.0f} × {h_mm / 10:.0f} cm · {len(L.stones)} taş · "
                        f"1:1 şablon {L.pages[0]} × {L.pages[1]} A4 sayfa", 24)
    k = min((PAGE_W - 120) / w_mm, (PAGE_H - 560) / h_mm)
    ox, oy = (PAGE_W - w_mm * k) / 2, 190
    if pieces and m.plate is not None:
        d.polygon([(ox + x * k, oy + y * k) for x, y in m.plate.exterior.coords], fill=tuple(pal[m.plate_colour].rgb))
    for st in L.stones:
        d.polygon([(ox + x * k, oy + (h_mm - y) * k) for x, y in st.poly.exterior.coords], fill=tuple(pal[st.colour].rgb))
    from .template import page_grid
    cols, rows, cw, ch = page_grid(w_mm, h_mm, "A4")
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for r in range(rows):
        for c in range(cols):
            x0 = c * (cw - 10)
            yt = r * (ch - 10)
            d.rectangle((ox + x0 * k, oy + yt * k, ox + min(w_mm, x0 + cw) * k, oy + min(h_mm, yt + ch) * k),
                        outline=(0, 110, 230), width=2)
            _text(d, (ox + (x0 + 8) * k, oy + (yt + 8) * k), f"{letters[r % 26]}{c + 1}", 22, fill=(0, 80, 200),
                  stroke=3)
    y = oy + h_mm * k + 24
    _text(d, (60, y), "Mavi çerçeveler 1:1 şablonun A4 sayfalarıdır (A3 dosyası da var).", 21)
    y += 40
    for i, f in enumerate(pal):
        d.rectangle((60, y, 100, y + 30), fill=tuple(f.rgb), outline=INK)
        nn = sum(1 for st in L.stones if st.colour == i)
        what = f"tek parça taban ({len(L.panels)} tabla)" if pieces and m.plate is not None and i == m.plate_colour and not nn else f"{nn} taş"
        _text(d, (115, y + 2), f"{L.letters[i]} = {f.name} ({FINISH_TR.get(f.finish, f.finish)})  ·  {what}", 22)
        y += 40
    pages.append(pg)
    # steps
    pg = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    d = ImageDraw.Draw(pg)
    _text(d, (60, 50), "Nasıl dizilir?", 40)
    base = pieces and m.plate is not None
    steps = [
        f"1. Tabanı bas: 02_tablalar/0_Taban dosyası. {len(L.panels)} parça, rengi {pal[m.plate_colour].name}.",
        "2. Taban birden çok parçaysa düz bir levhanın üstünde yan yana diz ve yapıştır (adları kılavuzdaki sıradadır)."
        if len(L.panels) > 1 else "2. Tabanı masaya koy; istersen önce duvara ya da levhaya yapıştır.",
    ] if base else [
        "1. Şablonu yazdır: 00_sablon_1-1_A4.pdf (ya da A3) dosyasını %100 / 'gerçek boyut' ile yazdır. "
        "Her sayfadaki 100 mm cetveli ölç.",
        "2. Sayfaları birleştir: kırmızı şeritler komşu sayfayla üst üste gelir; kırmızı artı işaretlerini "
        "çakıştırıp bantla. Sayfa adları (A1, A2, B1 …) genel haritadaki mavi çerçevelerdir.",
        "3. Zemin: şablonu levhanın (MDF/kontrplak) ya da duvarın üstüne bantla. İstersen şablonun altına "
        "karbon kâğıdı koyup çizgileri zemine geçir.",
    ]
    if base:
        steps += [
            "3. Taşları renk renk bas. Her taş tabanda kendi yuvasına oturur; yuvanın dibinde taşın numarası yazar.",
            "4. 8 mm'den büyük taşların kodu altında aynalı yazılıdır; kodun son sayısı yuvadaki numaradır. "
            "Küçük taşları tabla haritasından (02_tablalar/onizleme) ya da 1:1 şablondan bul.",
            "5. Yuvaya bir damla yapıştırıcı koy, taşı bastır. Kâğıt şablona gerek yok; istersen yine yazdırabilirsin.",
        ]
    elif pieces:
        steps += [
            "4. Tablalar şablon sayfalarına göre sıralıdır: önce A1 sayfasının taşlarını içeren tablaları bas; "
            "o sayfayı dizerken sıradaki tablalar basılır.",
            "5. Her taşın kodu şablonda yazar. 8 mm'den büyük taşların kodu altında da aynalı yazılıdır. "
            "Küçük taşları tabla haritasındaki (02_tablalar/onizleme) yerlerinden tanırsın.",
            "6. Yapıştır: taşın altına bir damla jel süper yapıştırıcı ya da şeffaf silikon; şablondaki yerine bastır. "
            f"Taşlar arasında {str(round(s.gap_mm, 1)).replace('.', ',')} mm boşluk kalır; derz gerekmez.",
            "7. Yan basılan Freud taşları (kavisli yüz) tabladan çıkınca yatırılır; kavisli yüz dışa bakar.",
        ]
        if getattr(s, "backing", "yok") == "kafes":
            steps.append("8. İsteğe bağlı kafes: taşlar kafesin ceplerine oturur; kafesi önce levhaya yapıştır.")
    else:
        steps += ["4. Her panel tek seferde basılır: taşlar ince bir tabanın üstündedir. Panelleri şablondaki "
                  "yerlerine yan yana yapıştır."]
    steps += ["Asma: levhanın arkasına fransız çıta (45°) ya da iki testere dişli askı; doğrudan duvara "
              "yapıştırıyorsan askı gerekmez."]
    _lines(d, 60, 130, steps)
    pages.append(pg)
    out = []
    for name, pg in zip(["01_montaj_00_genel.png", "01_montaj_00_dizme.png"], pages):
        p = os.path.join(out_dir, name)
        pg.save(p, optimize=True)
        out.append(p)
    pdf = os.path.join(out_dir, "01_montaj_kilavuzu.pdf")
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=150.0)
    out.insert(0, pdf)
    return out
