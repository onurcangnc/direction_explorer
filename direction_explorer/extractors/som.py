"""Piras et al. 2026 SOM-based multi-direction refusal extraction.

Pure PyTorch SOM (no MiniSom dependency), deterministic given seed. The
math here is preserved byte-for-byte from the pre-refactor monolith — do
not change PCA init, neighborhood schedule, or update rule unless you
explicitly mean to invalidate cached directions on disk.
"""

from __future__ import annotations

import torch

from direction_explorer.core.activations import collect_layer_activations
from direction_explorer.extractors.base import (
    ExtractedDirection,
    ExtractionResult,
    ExtractorBase,
)


class SimpleSOM:
    """Self-Organizing Map — Kohonen (2013) algorithm, rectangular grid,
    Gaussian neighborhood, time-decaying learning rate. fp32 on CPU."""

    def __init__(
        self,
        grid_size=(4, 4),
        input_dim: int = 3072,
        learning_rate: float = 0.01,
        sigma: float = 0.3,
        n_iterations: int = 10000,
        topology: str = "rectangular",
        seed: int = 42,
    ):
        if topology != "rectangular":
            raise ValueError(f"Only 'rectangular' topology is supported (got {topology}).")
        self.grid_rows, self.grid_cols = int(grid_size[0]), int(grid_size[1])
        self.n_neurons = self.grid_rows * self.grid_cols
        self.input_dim = int(input_dim)
        self.learning_rate = float(learning_rate)
        self.sigma = float(sigma)
        self.n_iterations = int(n_iterations)
        self.seed = int(seed)
        self._gen = torch.Generator().manual_seed(self.seed)
        self._weights = torch.zeros(self.n_neurons, self.input_dim, dtype=torch.float32)
        self._positions = [(i // self.grid_cols, i % self.grid_cols)
                           for i in range(self.n_neurons)]
        self._fitted = False

    @property
    def neurons(self) -> torch.Tensor:
        return self._weights

    @property
    def neuron_lattice_positions(self) -> list:
        return list(self._positions)

    def _pca_init(self, X: torch.Tensor) -> None:
        Xc = X - X.mean(dim=0, keepdim=True)
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        pc1 = Vh[0]
        pc2 = Vh[1] if Vh.shape[0] > 1 else torch.zeros_like(pc1)
        proj = Xc @ torch.stack([pc1, pc2], dim=1)
        pc1_min, pc1_max = float(proj[:, 0].min()), float(proj[:, 0].max())
        pc2_min, pc2_max = float(proj[:, 1].min()), float(proj[:, 1].max())
        if abs(pc1_max - pc1_min) < 1e-9:
            pc1_max = pc1_min + 1.0
        if abs(pc2_max - pc2_min) < 1e-9:
            pc2_max = pc2_min + 1.0

        mean_X = X.mean(dim=0)
        for i in range(self.n_neurons):
            r, c = self._positions[i]
            tr = r / max(self.grid_rows - 1, 1)
            tc = c / max(self.grid_cols - 1, 1)
            v1 = pc1_min + tr * (pc1_max - pc1_min)
            v2 = pc2_min + tc * (pc2_max - pc2_min)
            self._weights[i] = mean_X + v1 * pc1 + v2 * pc2

    def fit(self, X: torch.Tensor) -> None:
        if X.dim() != 2:
            raise ValueError(f"X must be 2D, got shape {tuple(X.shape)}")
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"X dim mismatch: expected {self.input_dim}, got {X.shape[1]}"
            )
        X = X.detach().to("cpu", dtype=torch.float32)
        n_samples = X.shape[0]
        if n_samples == 0:
            self._fitted = True
            return

        self._pca_init(X)

        T = float(self.n_iterations)
        sigma = self.sigma
        # Sigma is in lattice units; for a small grid we keep it small.
        sigma_abs = max(sigma * max(self.grid_rows, self.grid_cols), 0.5)
        two_sig_sq = 2.0 * (sigma_abs ** 2)

        positions = torch.tensor(self._positions, dtype=torch.float32)
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        lattice_dist_sq = (diff[..., 0] ** 2 + diff[..., 1] ** 2)

        for t in range(self.n_iterations):
            idx = int(torch.randint(0, n_samples, (1,), generator=self._gen).item())
            x = X[idx]
            d2 = ((self._weights - x) ** 2).sum(dim=1)
            bmu = int(torch.argmin(d2).item())
            alpha_t = self.learning_rate / (1.0 + 2.0 * t / T)
            h = torch.exp(-lattice_dist_sq[bmu] / two_sig_sq)
            self._weights += (alpha_t * h).unsqueeze(1) * (x.unsqueeze(0) - self._weights)

        self._fitted = True

    def assign_bmus(self, X: torch.Tensor) -> torch.Tensor:
        X = X.detach().to("cpu", dtype=torch.float32)
        chunk = 512
        out = torch.zeros(X.shape[0], dtype=torch.long)
        for s in range(0, X.shape[0], chunk):
            e = min(s + chunk, X.shape[0])
            sub = X[s:e].unsqueeze(1) - self._weights.unsqueeze(0)
            d2 = (sub ** 2).sum(dim=2)
            out[s:e] = d2.argmin(dim=1)
        return out


def compute_som_directions(
    harmful_activations: torch.Tensor,
    harmless_activations: torch.Tensor,
    grid_size=(4, 4),
    n_iterations: int = 10000,
    learning_rate: float = 0.01,
    sigma: float = 0.3,
    seed: int = 42,
) -> list[dict]:
    """Train SOM on harmful activations, then for each neuron i:
       direction_i = unit(w_i − μ_harmless). Returns a list of dicts sorted
       by cluster_size descending."""
    H = harmful_activations.detach().to("cpu", dtype=torch.float32)
    N = harmless_activations.detach().to("cpu", dtype=torch.float32)
    if H.dim() != 2 or N.dim() != 2:
        raise ValueError("activations must be 2D")
    n_h, d_in = H.shape
    if N.shape[1] != d_in:
        raise ValueError(f"d_in mismatch: harmful={d_in}, harmless={N.shape[1]}")

    som = SimpleSOM(
        grid_size=grid_size,
        input_dim=d_in,
        learning_rate=learning_rate,
        sigma=sigma,
        n_iterations=n_iterations,
        topology="rectangular",
        seed=seed,
    )
    som.fit(H)

    mu_harmless = N.mean(dim=0)
    bmus = som.assign_bmus(H)
    neurons = som.neurons
    positions = som.neuron_lattice_positions
    n_neurons = neurons.shape[0]

    cluster_sizes = torch.zeros(n_neurons, dtype=torch.long)
    for i in range(n_neurons):
        cluster_sizes[i] = int((bmus == i).sum().item())
    total_h = max(int(n_h), 1)

    results = []
    for i in range(n_neurons):
        diff = neurons[i] - mu_harmless
        raw_norm = float(diff.norm().item())
        if raw_norm < 1e-9:
            direction = torch.zeros_like(diff)
        else:
            direction = diff / raw_norm
        cluster_member_mask = (bmus == i)
        cluster_size = int(cluster_member_mask.sum().item())
        if cluster_size > 0:
            members = H[cluster_member_mask]
            tightness = float(((members - neurons[i]) ** 2).sum(dim=1).sqrt().mean().item())
        else:
            tightness = float("nan")
        results.append({
            "lattice_position": tuple(positions[i]),
            "neuron_index": i,
            "direction": direction,
            "raw_norm": raw_norm,
            "cluster_size": cluster_size,
            "cluster_share": cluster_size / total_h,
            "cluster_tightness": tightness,
        })

    results.sort(key=lambda r: -r["cluster_size"])
    return results


class SOMExtractor(ExtractorBase):
    method_name = "som_md"

    def compute(
        self,
        harmful_prompts: list[str],
        harmless_prompts: list[str],
        layer: int,
        som_grid_rows: int = 4,
        som_grid_cols: int = 4,
        som_iterations: int = 10000,
        som_learning_rate: float = 0.01,
        som_sigma: float = 0.3,
        som_seed: int = 42,
        **kwargs,
    ) -> ExtractionResult:
        ctx = self.ctx
        rows = max(2, min(int(som_grid_rows), 8))
        cols = max(2, min(int(som_grid_cols), 8))
        n_iter = max(1000, min(int(som_iterations), 50000))

        H_acts = collect_layer_activations(ctx, harmful_prompts, layer)
        N_acts = collect_layer_activations(ctx, harmless_prompts, layer)

        som_results = compute_som_directions(
            H_acts, N_acts,
            grid_size=(rows, cols),
            n_iterations=n_iter,
            learning_rate=float(som_learning_rate),
            sigma=float(som_sigma),
            seed=int(som_seed),
        )

        mu_harmless_norm = float(N_acts.mean(dim=0).norm().item())
        mu_harmful_norm = float(H_acts.mean(dim=0).norm().item())

        directions: list[ExtractedDirection] = []
        for rank, r in enumerate(som_results):
            i = int(r["neuron_index"])
            layer_key = f"{layer}_som_n{i}"
            d_vec = r["direction"]
            raw_norm_i = float(r["raw_norm"])
            cluster_share = float(r["cluster_share"])
            normalized_score_i = cluster_share  # use cluster_share as the score
            tightness_val = r["cluster_tightness"]
            tightness_clean = (
                float(tightness_val) if tightness_val == tightness_val else None
            )
            directions.append(ExtractedDirection(
                layer_key=layer_key,
                direction=d_vec.detach().to("cpu"),
                raw_norm=raw_norm_i,
                normalized_score=normalized_score_i,
                metadata={
                    "lattice_position": list(r["lattice_position"]),
                    "neuron_index": i,
                    "cluster_size": int(r["cluster_size"]),
                    "cluster_share": cluster_share,
                    "cluster_tightness": tightness_clean,
                    "som_grid_rows": rows,
                    "som_grid_cols": cols,
                    "rank_by_cluster_size": rank,
                    "display_label": f"L{layer} (SOM n[{i // cols},{i % cols}])",
                },
            ))

        return ExtractionResult(
            method=self.method_name,
            layer=layer,
            directions=directions,
            summary={
                "som_grid_rows": rows,
                "som_grid_cols": cols,
                "n_neurons": rows * cols,
                "harmful_centroid_norm": mu_harmful_norm,
                "harmless_centroid_norm": mu_harmless_norm,
            },
        )
