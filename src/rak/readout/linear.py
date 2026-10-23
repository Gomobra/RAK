"""Readout linear untuk spike count.

Jaringan spike belajar tanpa label. Modul ini hanya memetakan vektor
spike count menjadi kelas. Tidak ada backprop ke jaringan; bobot SNN
tidak disentuh sama sekali.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

__all__ = ["RidgeReadout", "Standardizer"]


@dataclass
class Standardizer:
    """Z-score per fitur, statistik dari data fit saja."""

    mean: Tensor
    std: Tensor

    @classmethod
    def fit(cls, x: Tensor) -> "Standardizer":
        x = x.to(torch.float64)
        return cls(
            mean=x.mean(dim=0),
            std=x.std(dim=0).clamp_min(1e-9),
        )

    def transform(self, x: Tensor) -> Tensor:
        return (x.to(torch.float64) - self.mean) / self.std


@dataclass
class RidgeReadout:
    """Ridge regression tertutup, dengan bias yang tidak diregularisasi.

    Fitur distandarisasi memakai statistik data fit. Alpha dipilih dari
    `alphas` berdasarkan akurasi pada data validasi yang diberikan
    pemanggil (bukan test). Setelah alpha terpilih, model di-fit ulang
    pada SELURUH data fit.
    """

    transform: str = "raw"
    alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    alpha_: float = 1.0
    val_accuracy_: float = 0.0
    scaler_: Standardizer | None = field(default=None, repr=False)
    weights_: Tensor | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    def _features(self, x: Tensor) -> Tensor:
        x = x.to(torch.float64)
        if self.transform == "raw":
            return x
        if self.transform == "log1p":
            return torch.log1p(x.clamp_min(0.0))
        raise ValueError(f"transform tidak dikenal: {self.transform}")

    @staticmethod
    def _solve(x: Tensor, y: Tensor, alpha: float) -> Tensor:
        """(X'X + alpha I)^-1 X'Y dengan kolom terakhir = bias."""
        d = x.shape[1]
        reg = alpha * torch.eye(d, dtype=x.dtype)
        reg[-1, -1] = 0.0  # bias tidak diregularisasi
        return torch.linalg.solve(x.T @ x + reg, x.T @ y)

    @staticmethod
    def _one_hot(y: Tensor, n_classes: int) -> Tensor:
        out = torch.zeros(y.shape[0], n_classes, dtype=torch.float64)
        out[torch.arange(y.shape[0]), y.to(torch.long)] = 1.0
        return out

    def _with_bias(self, x: Tensor) -> Tensor:
        return torch.cat(
            [x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=1
        )

    # ------------------------------------------------------------------
    def fit(
        self,
        x: Tensor,
        y: Tensor,
        *,
        x_val: Tensor | None = None,
        y_val: Tensor | None = None,
        val_fraction: float = 0.2,
        seed: int = 0,
    ) -> "RidgeReadout":
        """Pilih alpha di validasi, lalu fit ulang di seluruh data.

        Kalau `x_val` tidak diberikan, data dipecah internal memakai
        `val_fraction`. Test tidak boleh dipakai di sini.
        """
        x = x.to(torch.float64)
        y = y.to(torch.long)
        if x.shape[0] != y.shape[0]:
            raise ValueError("jumlah baris x dan y tidak sama")

        if x_val is None:
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(x.shape[0], generator=g)
            n_fit = max(1, int((1.0 - val_fraction) * x.shape[0]))
            fit_idx, val_idx = perm[:n_fit], perm[n_fit:]
            x_fit, y_fit = x[fit_idx], y[fit_idx]
            x_val, y_val = x[val_idx], y[val_idx]
        else:
            x_fit, y_fit = x, y
            assert y_val is not None
            x_val = x_val.to(torch.float64)
            y_val = y_val.to(torch.long)

        f_fit = self._features(x_fit)
        scaler = Standardizer.fit(f_fit)
        x_fit_z = self._with_bias(scaler.transform(f_fit))
        x_val_z = self._with_bias(scaler.transform(self._features(x_val)))
        y_fit_hot = self._one_hot(y_fit, int(y.max().item()) + 1)

        best_alpha, best_acc = float(self.alphas[0]), -1.0
        for alpha in self.alphas:
            w = self._solve(x_fit_z, y_fit_hot, alpha)
            acc = float(
                (x_val_z @ w).argmax(dim=1).eq(y_val).double().mean()
            )
            if acc > best_acc:
                best_alpha, best_acc = float(alpha), acc

        # Refit pada seluruh data dengan alpha terpilih.
        f_all = self._features(x)
        scaler = Standardizer.fit(f_all)
        x_all_z = self._with_bias(scaler.transform(f_all))
        self.weights_ = self._solve(
            x_all_z, self._one_hot(y, int(y.max().item()) + 1), best_alpha
        )
        self.scaler_ = scaler
        self.alpha_ = best_alpha
        self.val_accuracy_ = best_acc
        return self

    def decision_function(self, x: Tensor) -> Tensor:
        if self.scaler_ is None or self.weights_ is None:
            raise RuntimeError("RidgeReadout belum di-fit")
        z = self._with_bias(self.scaler_.transform(self._features(x)))
        return z @ self.weights_

    def predict(self, x: Tensor) -> Tensor:
        return self.decision_function(x).argmax(dim=1)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        if self.scaler_ is None or self.weights_ is None:
            raise RuntimeError("RidgeReadout belum di-fit")
        torch.save(
            {
                "transform": self.transform,
                "alpha": self.alpha_,
                "val_accuracy": self.val_accuracy_,
                "mean": self.scaler_.mean,
                "std": self.scaler_.std,
                "weights": self.weights_,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RidgeReadout":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        out = cls(
            transform=blob["transform"],
            alpha_=float(blob["alpha"]),
            val_accuracy_=float(blob.get("val_accuracy", 0.0)),
        )
        out.scaler_ = Standardizer(mean=blob["mean"], std=blob["std"])
        out.weights_ = blob["weights"]
        return out
