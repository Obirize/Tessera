"""Blender layers (acilis_blender.py, mode "katman") -> the small files the opening card needs.

usage: acilis_hazirla.py <katman_dir> <kedi_n.json>      writes tessera/web/acilis/
"""
import json
import os
import sys

from PIL import Image, ImageDraw

src, cat = sys.argv[1], json.load(open(sys.argv[2], encoding="utf-8"))
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tessera", "web", "acilis")
os.makedirs(out, exist_ok=True)
d = json.load(open(os.path.join(src, "katman.json")))
im = {n: Image.open(os.path.join(src, n + ".png")).convert("RGB") for n in ("tam", "levha")}
R = im["tam"].width


def px(x, y):
    return ((x - d["cx"]) / d["scale"] + 0.5) * R, (0.5 - (y - d["cy"]) / d["scale"]) * R


# cut the plate out of the paper ground: the card paints its own paper and shadow
m = Image.new("L", (R * 4, R * 4), 0)
ImageDraw.Draw(m).polygon([tuple(4 * v for v in px(*p)) for p in cat["plate"]], fill=255)
m = m.resize((R, R), Image.LANCZOS)
for n, i in im.items():
    i.putalpha(m)
    i.save(os.path.join(out, n + ".webp"), quality=94, method=6, exact=False)
stones = [{"region": s["region"], "pts": [[round(v / R, 5) for v in px(*p)] for p in s["pts"]]} for s in d["stones"]]
json.dump({"plate": [[round(v / R, 5) for v in px(*p)] for p in cat["plate"]], "stones": stones},
          open(os.path.join(out, "taslar.json"), "w"))
print({f: os.path.getsize(os.path.join(out, f)) // 1024 for f in os.listdir(out)}, "KB")
