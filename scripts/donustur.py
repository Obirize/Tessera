"""Herhangi bir görseli mozaiğe çevirir ve önizlemeleri kaydeder.

Kullanım:
    .venv\\Scripts\\python scripts\\donustur.py <görsel> [genişlik_cm] [renk_sayısı]

Çıktı: cikti/<görsel_adı>/  (once_sonra.png, mozaik.png, acili.png, sayilar.json)
Renkler görselden otomatik seçilir (ileride: senin filament listenden).
Bölgeler: varsa algoritma ekibinin otomatik önerisi, yoksa her yer "taşlar".
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy.cluster.vq import kmeans2  # noqa: E402

from tessera.model import Filament, Settings  # noqa: E402
from tessera.pipeline import build_mosaic, load_image  # noqa: E402
from tessera.render import render_cut, render_oblique  # noqa: E402


def auto_palette(img, k):
    try:
        from tessera.pipeline.palette import auto_palette as ap  # anti-alias/tram-aware
        return ap(img, k)
    except ImportError:
        px = img.reshape(-1, 3)[:: max(1, img.shape[0] * img.shape[1] // 60000)].astype(float)
        cent, lab = kmeans2(px, k, seed=1, minit="++")
        order = np.argsort(-np.bincount(lab, minlength=k))
        return [Filament(f"Renk {i + 1}", tuple(int(v) for v in cent[j])) for i, j in enumerate(order)]


def main():
    src = sys.argv[1]
    width_cm = float(sys.argv[2]) if len(sys.argv) > 2 else 40
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    name = os.path.splitext(os.path.basename(src))[0]
    if name.lower() in ("kaynak", "source", "image", "gorsel"):
        name = os.path.basename(os.path.dirname(os.path.abspath(src))) or name
    out = os.path.join("cikti", name)
    os.makedirs(out, exist_ok=True)

    s = Settings(width_mm=width_cm * 10)
    img = load_image(src, s)
    try:  # same as the app: real Bambu filaments from the library
        from tessera.pipeline.palette import load_filaments, suggest_palette
        s.palette = [p["filament"] for p in suggest_palette(img, load_filaments(), k, owned_only=False)]
    except Exception:
        s.palette = auto_palette(img, k)
    zone = None
    try:
        from tessera.pipeline import suggest_zones  # algoritma ekibi ekleyince devreye girer
        zone, _ = suggest_zones(img, s)
    except Exception:
        pass
    m = build_mosaic(img, s, zone, source=src)
    top = render_cut(m, 1600)
    Image.fromarray(top).save(os.path.join(out, "mozaik.png"))
    Image.fromarray(render_oblique(top, m)).save(os.path.join(out, "acili.png"))

    a = Image.open(src).convert("RGB")
    th = 900
    a = a.resize((int(a.width * th / a.height), th), Image.LANCZOS)
    b = Image.fromarray(top).resize(a.size, Image.LANCZOS)
    sbs = Image.new("RGB", (a.width * 2 + 60, th + 70), (245, 243, 238))
    sbs.paste(a, (20, 50))
    sbs.paste(b, (a.width + 40, 50))
    try:
        f = ImageFont.truetype("arialbd.ttf", 26)
    except OSError:
        f = ImageFont.load_default()
    d = ImageDraw.Draw(sbs)
    d.text((20, 12), "Görsel", fill=(30, 30, 30), font=f)
    w_mm, h_mm = m.size_mm
    used = [f.name for i, f in enumerate(s.palette) if (m.colour == i).any()]
    stats = dict(m.stats)
    try:  # same numbers as the app (slicer-calibrated)
        from tessera.export.api import estimate
        stats["tahmin"] = estimate(m)
    except Exception:
        pass
    est = stats.get("tahmin", {})
    head = f"Tessera: {len(used)} renk, {w_mm / 10:.0f} x {h_mm / 10:.0f} cm"
    if est:
        head += f" · {est.get('parca')} parça · ~{est.get('saat')} saat · {est.get('gram', 0) / 1000:.2f} kg"
    d.text((a.width + 40, 12), head, fill=(30, 30, 30), font=f)
    d.text((20, th + 58 - 16), "Renkler: " + ", ".join(used), fill=(60, 60, 60), font=ImageFont.load_default())
    sbs.save(os.path.join(out, "once_sonra.png"))
    with open(os.path.join(out, "sayilar.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2, default=str)
    print("Hazır:", os.path.abspath(out))


if __name__ == "__main__":
    main()
