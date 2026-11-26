"""Opening a picture. Only numpy and Pillow: the window shows pictures before the pipeline is loaded."""
from __future__ import annotations

import numpy as np
from PIL import Image


def open_rgb(path_or_img) -> Image.Image:
    """Open any supported picture as RGB. Transparent areas are laid on white (a transparent
    logo must not become a black mosaic). Raises ValueError with a Turkish message for
    missing, empty, corrupt or unsupported files."""
    if isinstance(path_or_img, np.ndarray):
        a = path_or_img
        if a.ndim == 2:
            a = np.stack([a] * 3, -1)
        if a.ndim != 3 or a.shape[2] not in (3, 4) or min(a.shape[:2]) < 2:
            raise ValueError("Görsel okunamadı: beklenen renkli bir resim (yükseklik × genişlik × 3).")
        im = Image.fromarray(a.astype(np.uint8))
    elif isinstance(path_or_img, Image.Image):
        im = path_or_img
    else:
        try:
            im = Image.open(path_or_img)
            im.load()
        except FileNotFoundError:
            raise
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as e:
            name = str(path_or_img).replace("\\", "/").split("/")[-1]
            raise ValueError(f"Görsel açılamadı: '{name}' bozuk ya da desteklenmeyen bir biçimde. "
                             f"PNG, JPG veya WEBP olarak yeniden kaydedip dene.") from e
    try:
        has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
        if has_alpha:
            rgba = im.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, rgba)
        im = im.convert("RGB")
    except (OSError, ValueError) as e:
        raise ValueError("Görsel açılamadı: renk biçimi desteklenmiyor. PNG ya da JPG olarak kaydedip dene.") from e
    if min(im.size) < 2:
        raise ValueError("Görsel çok küçük (en az 2 × 2 piksel olmalı).")
    return im
