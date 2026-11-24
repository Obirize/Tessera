# Test varyantları (sanat ekibi)

İki kalite ölçütünün (poster + mürekkep/tram çizgi roman) zorlaştırılmış halleri.
Hepsi `scripts/sanat_murekkep.py` ile yeniden üretilir; `scripts/sanat_varyant_test.py` hepsini
hızlı yoldan (otomatik palet, `donustur.py` mantığı) geçirip `docs/img/sanat_var_*.png` kaydeder.

| Dosya | Ne test eder |
|---|---|
| `poster_yuzler.png` | Posterin yüzlere kırpılmış hali (848 px). Aynı fiziksel boyutta yüzler ~1.9× büyür: taş boyu ile yüz ölçeği ilişkisi, göz/kaş/dudak okunurluğu. |
| `poster_2renk.png` | Siyah + krem eşikleme (kırmızı siyaha düşer). Renk kaybında figür/zemin ayrımının yalnız kabartma ve dil ile taşınıp taşınamadığı. |
| `poster_5renk.png` | k-means ile 5 renk: 3 gerçek renk + kenar yumuşatmasından gelen 2 ara ton (pembe, kahve). Otomatik paletin kenar halelerine renk harcayıp harcamadığı, ince hale parçaları. |
| `poster_600px.png` | Düşük çözünürlüklü kaynak (600 px, 42 cm için 0.7 mm/px). Büyütmede yumuşak kenarlar, kaybolan ince çizgiler, tramsız görselde yanlış tram algılama. |
| `poster_3000px.png` | Bikübik büyütülmüş yüksek çözünürlük (3000 px). Büyük dosyada hız, otomatik paletin kararlılığı. |
| `murekkep_tramsiz.png` | Mürekkep görselinin düz gri tonlu hali (nokta yok). Tram çözücüsü kapalıyken taban çizgi: griler doğru 3. renge gidiyor mu. |
| `murekkep_kaba_tram.png` | Aynı görsel, iri nokta tramı (18 px aralık ≈ 6 mm). Tram çözücünün iri noktalarda çalışıp çalışmadığı; yoksa binlerce nokta parçası çıkar. |
| `murekkep_90derece.png` | İnce tramlı görselin 90° döndürülmüşü. Tram açısı/yön bağımlılığı, yatay format, ışın merkezinin yer değiştirmesi. |

Asıl kaynaklar: `ornekler/poster/kaynak.webp`, `ornekler/murekkep_tram/kaynak.png` (özgün; başlıklı gezgin,
sivri oval yüz plakası, kapalı hilal gözler, perçinli alın bandı; yeşil yok, dikdörtgen plaka yok, ağız ızgarası yok).
