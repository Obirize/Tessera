"""rapor.json + rapor.txt (Turkish summary)."""
from __future__ import annotations

import json
import os
from collections import defaultdict

from ..geometry.layout import Layout
from ..model import Mode


def write_report(L: Layout, out_dir: str, res: dict) -> list[str]:
    s = L.mosaic.settings
    mode = res["mod"]
    w, h = L.mosaic.size_mm
    lines = []
    lines.append("TESSERA ÜRETİM RAPORU")
    lines.append("=" * 40)
    lines.append(f"Boyut: {w / 10:.1f} × {h / 10:.1f} cm   Mod: "
                 + ("Tek renk tablalar · AMS gerekmez (her tabla tek renk, renk değişimi yok)" if mode == Mode.PIECES
                    else "Çok renkli paneller · AMS gerekir (AMS = Bambu'nun çok renkli besleyicisi; panel tek seferde basılır)"))
    lines.append(f"Paneller: {res['panels']} adet (en fazla {s.panel_max_mm:.0f} mm), boyutlar: "
                 + ", ".join(f"{a}×{b}" for a, b in res["panel_boyutlari_mm"]) + " mm")
    if mode == Mode.PIECES:
        lines.append(f"Yerleştirilecek parça: {res['parts']} adet ({res.get('hassas_parca', 0)} hassas)"
                     + ("" if s.merge_same_colour else " (aynı renk birleştirme KAPALI)"))
    else:
        lines.append(f"Yerleştirilecek parça: {res['parts']} panel (her panel tek nesne; "
                     f"{res.get('renk_bolgesi', 0)} renk bölgesi içinde)")
    lines.append(f"Kafes (arka ızgara) rengi: {res['kafes_renk']}. Kafes açık ızgaradır: MDF/kontrplak levha gerekir.")
    lines.append(f"Ağırlık: ~{res.get('kg_m2', 0)} kg/m² (parçalar ~{res.get('parca_gram', 0)} g, "
                 f"kafes ~{res.get('kafes_gram', 0)} g)")
    lines.append("")
    lines.append("Renk başına:")
    by_col = defaultdict(lambda: [0, 0.0, 0.0])
    for p in res["plates"]:
        p = dict(p, renk=(f"Kafes ({p['renk']})" if p.get("tur") == "kafes" else
                          "Askı çıtası" if p.get("tur") == "aski" else p["renk"]))
        by_col[p["renk"]][0] += 1
        by_col[p["renk"]][1] += p["gram"]
        by_col[p["renk"]][2] += p["saat"]
    for f in s.palette:
        n = res["renk_parca"].get(f.name, 0)
        what = "parça" if mode == Mode.PIECES else "bölge"
        lines.append(f"  {f.name:<14} {n:>5} {what}   ~{res['renk_gram'].get(f.name, 0):>5} g")
    lines.append("")
    lines.append("Tablalar:")
    if mode == Mode.PIECES:
        for name, (n, g, hh) in by_col.items():
            lines.append(f"  {name:<16} {n:>3} tabla   ~{g:>5.0f} g   ~{hh:>5.1f} saat")
    else:
        for p in res["plates"]:
            if p.get("tur") == "aski":
                lines.append(f"  Askı çıtaları: {p['dosya']}, ~{p['gram']} g, ~{p['saat']:.1f} saat")
                continue
            lines.append(f"  Panel {p['panel'][0]:<4} tabla {p['tabla']:>2}: ~{p['gram']:>4} g, ~{p['saat']:.1f} saat, "
                         f"~{p['degisim']} renk değişimi, ~{p['purge_gram']} g değişim atığı")
    lines.append(f"  TOPLAM: {len(res['plates'])} tabla, ~{res['grams']} g filament, ~{res['hours']} saat baskı")
    lines.append("")
    if mode == Mode.AMS and res.get("purge"):
        pg = res["purge"]
        lines.append(f"Renk değişimi (TAHMİN): ~{pg['degisim']} değişim, ~{pg['purge_gram']} g değişim atığı, "
                     f"~{pg['saat']} saat değişim süresi.")
        lines.append("  Değişim atığı = renk değiştirirken yazıcının attığı filament. Her katmanda bulunan renk "
                     "sayısı - 1 değişim, değişim başına ~0,5 g. Kesin değer Bambu Studio'nun dilimleme sonucundadır.")
        lines.append("")
    else:
        lines.append("Renk değişimi: 0 (her tabla tek renk). Değişim atığı: 0 g.")
        lines.append("")
    if res.get("numarasiz_parca"):
        lines.append(f"Not: {res['numarasiz_parca']} parçanın altına kod sığmadı; kodları tabla görselinde "
                     "(02_tablalar/onizleme) ve kafesin cebinde yazılı.")
    if res["warnings"]:
        lines.append("Uyarılar:")
        for wv in res["warnings"]:
            lines.append(f"  - {wv}")
    lines.append("")
    lines.append("Dosyalar: 01_montaj_kilavuzu.pdf (kılavuz), 02_tablalar/*.3mf (Bambu Studio'da aç), "
                 "02_tablalar/onizleme/*.png (tabla haritaları), 03_stl/ (yedek).")
    lines.append("Tabla sırası: " + ("panel sırasıyla (P1 basılırken P1 dizilebilir)" if res.get("tabla_sirasi") == "panel"
                                     else "renk başına en sıkı dizilim"))
    lines.append("Gram ve saat değerleri tahmindir (Bambu Studio 0.20 mm Standart, %15 dolgu ile kalibre).")
    txt = os.path.join(out_dir, "rapor.txt")
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    js = os.path.join(out_dir, "rapor.json")
    data = {k: v for k, v in res.items() if k != "files"}
    data["dosyalar"] = [os.path.relpath(f, out_dir) for f in res["files"]]
    data["boru_hatti_istatistik"] = L.mosaic.stats
    data["yerlesim_istatistik"] = L.stats
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return [txt, js]


def write_report_stones(L, out_dir: str, res: dict) -> list[str]:
    s = L.mosaic.settings
    mode = res["mod"]
    w, h = L.mosaic.size_mm
    pieces = mode == Mode.PIECES
    lines = ["TESSERA ÜRETİM RAPORU", "=" * 40,
             f"Boyut: {w / 10:.1f} × {h / 10:.1f} cm   Taş stili: {getattr(s, 'style', 'klasik')}   Mod: "
             + ("Tek renk tablalar · AMS gerekmez" if pieces else "Çok renkli paneller · AMS gerekir"),
             f"Taş: {res['tas']} ayrı taş (aralarında {s.gap_mm:.1f} mm boşluk)"
             + ("" if pieces else f", {res['panels']} çok renkli panel"),
             f"Ağırlık: ~{res['kg_m2']} kg/m², toplam ~{res['grams']} g, ~{res['hours']} saat baskı",
             f"Şablon: 1:1, {res['sablon_sayfa'][0]} × {res['sablon_sayfa'][1]} A4 sayfa (A3 dosyası da var)", ""]
    lines.append("Renk başına:")
    for f in s.palette:
        lines.append(f"  {f.name:<14} {res['renk_parca'].get(f.name, 0):>5} taş   ~{res['renk_gram'].get(f.name, 0):>5} g")
    lines.append("")
    lines.append(f"Tablalar: {len(res['plates'])}")
    by = defaultdict(lambda: [0, 0, 0.0])
    for p in res["plates"]:
        key = p["renk"] if p.get("tur") == "parca" else ("Kafes" if p.get("tur") == "kafes" else p["renk"])
        by[key][0] += 1
        by[key][1] += p["gram"]
        by[key][2] += p["saat"]
    for k_, (n, g, hh) in by.items():
        lines.append(f"  {k_:<16} {n:>3} tabla  ~{g:>5} g  ~{hh:>5.1f} saat")
    if not pieces and res.get("purge"):
        lines.append(f"Renk değişimi (TAHMİN): ~{res['purge']['degisim']} değişim, ~{res['purge']['purge_gram']} g "
                     "değişim atığı.")
        lines.append(f"Taban rengi (boşluklarda görünür): {res.get('taban_renk')}")
    if pieces:
        lines.append(f"Altında kodu olan taş: {res['kodlu_tas']} (8 mm ve üstü); diğerleri şablon ve tabla haritasından.")
        if res.get("yan_basilan"):
            lines.append(f"Yan basılan Freud taşı: {res['yan_basilan']} (kavisli yüz pürüzsüz duvar olarak basılır)")
    if res["warnings"]:
        lines.append("Uyarılar:")
        lines.extend(f"  - {wv}" for wv in res["warnings"])
    lines += ["", "Dosyalar: 00_sablon_1-1_A4.pdf / _A3.pdf (1:1 şablon), 01_montaj_kilavuzu.pdf, "
              "02_tablalar/*.3mf (Bambu Studio), 02_tablalar/onizleme/*.png (tabla haritaları), 03_stl/ (yedek).",
              "Gram ve saat değerleri tahmindir (Bambu Studio CLI ile kalibre)."]
    txt = os.path.join(out_dir, "rapor.txt")
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    js = os.path.join(out_dir, "rapor.json")
    data = {k: v for k, v in res.items() if k != "files"}
    data["dosyalar"] = [os.path.relpath(f, out_dir) for f in res["files"]]
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, default=str)
    return [txt, js]
