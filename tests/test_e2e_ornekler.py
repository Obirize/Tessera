"""Critic/QA team: the end-to-end matrix (test_e2e.py) for the examples other than the poster and the
ink print. A file of its own so that the two slow halves run side by side; the checks are the same."""
from __future__ import annotations

import numpy as np
import pytest

from test_e2e import HERE, ROOT, built, build_mosaic, check_estimates, check_example, check_min_feature  # noqa: F401

from qa_ortak import EXAMPLES  # noqa: E402  (test_e2e puts scripts/ on the path)


@pytest.mark.parametrize("name", [n for n in EXAMPLES if n not in HERE])
def test_example_invariants(built, name):  # noqa: F811
    check_example(built, name)


@pytest.mark.parametrize("name", ["portre", "demir_maske"])
def test_min_feature_width(built, name):  # noqa: F811
    check_min_feature(built, name)


@pytest.mark.parametrize("name", ["orman_masali"])
def test_estimate_both_modes_fast_and_sane(built, name):  # noqa: F811
    check_estimates(built, name)


def test_no_halo_specks_on_new_image():
    """ornekler/portre loaded the way the UI loads a new image: library palette + 'Genel' preset."""
    from scipy import ndimage as ndi
    from qa_ortak import apply_preset, load_presets
    from tessera.model import Settings
    from tessera.pipeline import load_filaments, load_image, to_filament
    lib = {f.name: f for f in (to_filament(e) for e in load_filaments())}
    s = Settings(palette=[lib[n] for n in ("Siyah", "Kırmızı", "Altın", "Fildişi Beyazı")], width_mm=424)
    pr = load_presets()["Genel"]
    apply_preset(s, pr)
    s.default_vocab = int(pr.get("default_vocab", 0))
    m = build_mosaic(load_image(str(ROOT / "ornekler" / "portre" / "kaynak.png"), s), s, None)
    lab, n = ndi.label(m.colour == 1)
    area = np.bincount(lab.ravel())[1:] * s.mm_per_px ** 2
    assert int((area < 4).sum()) <= 10, f"{int((area < 4).sum())} kırmızı kırıntı (<4 mm²)"
