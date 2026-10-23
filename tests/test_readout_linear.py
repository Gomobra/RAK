"""Properti matematis readout linear.

Yang diuji adalah klaim yang bisa dibuktikan, bukan asumsi:

1. Standardizer menghasilkan mean 0 dan std 1, memakai statistik data
   fit saja.
2. Ridge memisahkan data yang memang terpisah secara linear.
3. Alpha yang dipakai selalu berasal dari grid yang diberikan.
4. Save/load tidak mengubah prediksi.
5. Fit tidak memutasi input.
6. Bias tidak diregularisasi: dengan fitur nol, prediksi = kelas
   mayoritas, bahkan pada alpha sangat besar.
"""

from __future__ import annotations

import torch

from rak.readout import RidgeReadout, Standardizer


def _blobs(
    n_per: int = 60, n_classes: int = 3, d: int = 6, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    centers = torch.zeros(n_classes, d)
    for c in range(n_classes):
        centers[c, c] = 3.0
    xs, ys = [], []
    for c in range(n_classes):
        xs.append(centers[c] + torch.randn(n_per, d, generator=g))
        ys.append(torch.full((n_per,), c, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


def test_standardizer_zero_mean_unit_std() -> None:
    x = torch.randn(200, 5, generator=torch.Generator().manual_seed(1)) * 4 + 7
    s = Standardizer.fit(x)
    z = s.transform(x)
    assert z.mean(dim=0).abs().max() < 1e-9
    assert (z.std(dim=0) - 1.0).abs().max() < 1e-6


def test_standardizer_uses_fit_statistics_only() -> None:
    x = torch.randn(50, 3, generator=torch.Generator().manual_seed(2))
    y = torch.randn(10, 3, generator=torch.Generator().manual_seed(3)) + 100
    s = Standardizer.fit(x)
    expected = ((y - x.mean(dim=0)) / x.std(dim=0).clamp_min(1e-9)).double()
    assert torch.allclose(s.transform(y), expected)


def test_ridge_separates_linear_data() -> None:
    x, y = _blobs()
    model = RidgeReadout().fit(x, y)
    assert float((model.predict(x) == y).double().mean()) > 0.95


def test_alpha_comes_from_grid() -> None:
    x, y = _blobs()
    grid = (0.5, 5.0, 50.0)
    model = RidgeReadout(alphas=grid).fit(x, y)
    assert model.alpha_ in grid


def test_save_load_roundtrip(tmp_path) -> None:
    x, y = _blobs()
    model = RidgeReadout().fit(x, y)
    path = tmp_path / "readout.pt"
    model.save(path)
    again = RidgeReadout.load(path)
    assert torch.equal(model.predict(x), again.predict(x))
    assert again.alpha_ == model.alpha_
    assert again.transform == model.transform


def test_fit_does_not_mutate_inputs() -> None:
    x, y = _blobs()
    x0, y0 = x.clone(), y.clone()
    RidgeReadout().fit(x, y)
    assert torch.equal(x, x0) and torch.equal(y, y0)


def test_bias_is_not_regularized() -> None:
    """Fitur nol: satu-satunya informasi adalah prior kelas."""
    x = torch.zeros(40, 4)
    y = torch.cat([torch.zeros(30, dtype=torch.long), torch.ones(10, dtype=torch.long)])
    model = RidgeReadout(alphas=(1e6,)).fit(x, y, val_fraction=0.5, seed=0)
    # Bias harus tetap belajar prior mayoritas walau alpha raksasa.
    assert int(model.predict(torch.zeros(1, 4))[0]) == 0


def test_explicit_validation_split_is_used() -> None:
    x, y = _blobs()
    # Data _blobs tersusun per kelas; acak dulu supaya validasi
    # memuat semua kelas.
    perm = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(4))
    x, y = x[perm], y[perm]
    model = RidgeReadout().fit(
        x[:120], y[:120], x_val=x[120:], y_val=y[120:]
    )
    assert 0.0 <= model.val_accuracy_ <= 1.0
    assert model.val_accuracy_ > 0.9


def test_log1p_transform_runs() -> None:
    x, y = _blobs()
    x = x.abs()
    model = RidgeReadout(transform="log1p").fit(x, y)
    assert model.alpha_ in model.alphas
    assert model.predict(x).shape == y.shape
