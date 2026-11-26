"""The program's own cat as a template picture: flat, only main colours, square to the edge.

usage: sablon_kedi.py <kedi_3.json>      writes "ornekler/sablonlar/Tessera kedisi.png"
"""
import json
import os
import sys

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tessera.pipeline.palette import hex_rgb  # noqa: E402
from tessera.pipeline.renk import MAIN  # noqa: E402

PAINT = {n: hex_rgb(h) for n, h in MAIN}
OF = {"bg": "Kırmızı", "ear_in": "Pembe", "nose": "Pembe", "pupil": "Siyah"}       # everything else is the white fur
SIZE, SS, HALF = 1600, 3, 0.97                                                     # px, supersampling, half width shown

d = json.load(open(sys.argv[1], encoding="utf-8"))
n = SIZE * SS
xs = [p[0] for s in d["stones"] for p in s["pts"]]
ys = [p[1] for s in d["stones"] for p in s["pts"]]
cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 / HALF
im = Image.new("RGB", (n, n), PAINT["Siyah"])
dr = ImageDraw.Draw(im)
for s in sorted(d["stones"], key=lambda s: -s["area"]):
    if s["region"] == "eye":                      # one blue, one amber, like the icon
        col = PAINT["Mavi" if s["cx"] < cx else "Altın"]
    else:
        col = PAINT[OF.get(s["region"], "Beyaz")]
    dr.polygon([((x - cx) / half * n / 2 + n / 2, n / 2 - (y - cy) / half * n / 2) for x, y in s["pts"]], fill=col)
out = os.path.join(ROOT, "ornekler", "sablonlar")
os.makedirs(out, exist_ok=True)
p = os.path.join(out, "Tessera kedisi.png")
im.resize((SIZE, SIZE), Image.LANCZOS).save(p, optimize=True)
print(p, os.path.getsize(p) // 1024, "KB", sorted({s["region"] for s in d["stones"]}))
