"""Ridge readout (Phase 3)."""

from __future__ import annotations

import torch
from torch import Tensor


class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: Tensor):
        x = x.to(torch.float64)
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0).clamp_min(1e-9)
        return self

    def transform(self, x: Tensor) -> Tensor:
        return (x.to(torch.float64) - self.mean) / self.std


class RidgeReadout:
    def __init__(self, alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)):
        self.alphas = alphas
        self.alpha_ = alphas[0]
        self.scaler_ = None
        self.weights_ = None
        self.val_accuracy_ = 0.0

    def _features(self, x: Tensor) -> Tensor:
        return x.to(torch.float64)

    def fit(self, x: Tensor, y: Tensor, x_val=None, y_val=None, val_fraction=0.2, seed=0):
        x = x.to(torch.float64)
        y = y.to(torch.long)
        if x_val is None:
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(x.shape[0], generator=g)
            n_fit = max(1, int((1 - val_fraction) * x.shape[0]))
            fit_idx, val_idx = perm[:n_fit], perm[n_fit:]
            x_fit, y_fit = x[fit_idx], y[fit_idx]
            x_val, y_val = x[val_idx], y[val_idx]
        else:
            x_fit, y_fit = x, y
            x_val = x_val.to(torch.float64)
            y_val = y_val.to(torch.long)

        scaler = Standardizer().fit(x_fit)
        x_fit_z = torch.cat([scaler.transform(x_fit), torch.ones(x_fit.shape[0], 1)], dim=1)
        x_val_z = torch.cat([scaler.transform(x_val), torch.ones(x_val.shape[0], 1)], dim=1)
        n_classes = int(y.max().item()) + 1
        y_hot = torch.zeros(y_fit.shape[0], n_classes)
        y_hot[torch.arange(y_fit.shape[0]), y_fit] = 1.0

        best_alpha, best_acc = float(self.alphas[0]), -1.0
        for alpha in self.alphas:
            reg = alpha * torch.eye(x_fit_z.shape[1])
            reg[-1, -1] = 0.0
            w = torch.linalg.solve(x_fit_z.T @ x_fit_z + reg, x_fit_z.T @ y_hot)
            acc = float((x_val_z @ w).argmax(1).eq(y_val).float().mean())
            if acc > best_acc:
                best_alpha, best_acc = float(alpha), acc

        # Refit on all
        scaler = Standardizer().fit(x)
        x_all_z = torch.cat([scaler.transform(x), torch.ones(x.shape[0], 1)], dim=1)
        y_all_hot = torch.zeros(y.shape[0], n_classes)
        y_all_hot[torch.arange(y.shape[0]), y] = 1.0
        reg = best_alpha * torch.eye(x_all_z.shape[1])
        reg[-1, -1] = 0.0
        self.weights_ = torch.linalg.solve(x_all_z.T @ x_all_z + reg, x_all_z.T @ y_all_hot)
        self.scaler_ = scaler
        self.alpha_ = best_alpha
        self.val_accuracy_ = best_acc
        return self

    def predict(self, x: Tensor) -> Tensor:
        x = x.to(torch.float64)
        z = torch.cat([self.scaler_.transform(x), torch.ones(x.shape[0], 1)], dim=1)
        return (z @ self.weights_).argmax(dim=1)

    def save(self, path):
        torch.save({"alpha": self.alpha_, "mean": self.scaler_.mean,
                    "std": self.scaler_.std, "weights": self.weights_}, path)

    @classmethod
    def load(cls, path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        r = cls()
        r.alpha_ = float(blob["alpha"])
        r.scaler_ = Standardizer()
        r.scaler_.mean = blob["mean"]
        r.scaler_.std = blob["std"]
        r.weights_ = blob["weights"]
        return r