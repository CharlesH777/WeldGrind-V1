"""Learnable surface set selection and conditional rectified-flow DiT.

Coordinates passed to the model use the same ROI-centered normalization for
cloud, seeds and ordered trajectory. Only the I/O layer converts millimetres.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .geometry import GeometryConfig, GeometryEncoder


@dataclass
class ModelConfig:
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    seeds: int = 64
    trajectory_points: int = 128
    heads: int = 8
    selector_layers: int = 2
    dit_layers: int = 6
    mlp_ratio: int = 4
    selection_temperature: float = 0.5
    noise_std: float = 0.5

    def __post_init__(self):
        if min(self.seeds, self.trajectory_points, self.selector_layers, self.dit_layers, self.heads, self.mlp_ratio) < 1:
            raise ValueError("model sizes must be positive")
        if self.geometry.dim % self.heads or self.geometry.dim < 8:
            raise ValueError("geometry.dim must be >=8 and divisible by heads")
        if self.selection_temperature <= 0 or self.noise_std <= 0:
            raise ValueError("selection_temperature and noise_std must be positive")
        if self.trajectory_points < 2:
            raise ValueError("ordered trajectory requires >=2 points")

    @classmethod
    def from_dict(cls, value: dict) -> "ModelConfig":
        value = dict(value)
        value["geometry"] = GeometryConfig(**value.get("geometry", {}))
        return cls(**value)

    def to_dict(self) -> dict:
        return asdict(self)


class SetAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ratio: int):
        super().__init__()
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.self_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(4)])
        self.ff = nn.Sequential(nn.Linear(dim, dim * ratio), nn.GELU(), nn.Linear(dim * ratio, dim))

    def forward(self, queries: Tensor, memory: Tensor, valid: Tensor) -> Tensor:
        q = self.norms[0](queries)
        m = self.norms[1](memory)
        queries = queries + self.cross(q, m, m, key_padding_mask=~valid, need_weights=False)[0]
        q = self.norms[2](queries)
        queries = queries + self.self_attention(q, q, q, need_weights=False)[0]
        return queries + self.ff(self.norms[3](queries))


class LearnableSetSelector(nn.Module):
    """Ordered queries select an unordered set of *observed* surface points.

    Forward selection is deterministic and without replacement while enough
    distinct valid candidates remain. Training uses a straight-through softmax
    estimator; a convex soft coordinate alone would drift off curved surfaces.
    If ROI has fewer points than queries, repetition is explicit in seed_unique.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.geometry.dim
        self.count = config.seeds
        self.temperature = config.selection_temperature
        self.queries = nn.Parameter(torch.randn(config.seeds, dim) * 0.02)
        self.position = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([
            SetAttentionBlock(dim, config.heads, config.mlp_ratio) for _ in range(config.selector_layers)
        ])
        self.query_projection = nn.Linear(dim, dim)
        self.key_projection = nn.Linear(dim, dim)
        self.feature_norm = nn.LayerNorm(dim)

    def forward(self, xyz: Tensor, features: Tensor, valid: Tensor) -> dict[str, Tensor]:
        if not valid.any(dim=1).all():
            raise ValueError("each sample requires at least one valid surface candidate")
        memory = features + self.position(xyz)
        queries = self.queries[None].expand(xyz.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, memory, valid)
        logits = self.query_projection(queries) @ self.key_projection(memory).transpose(-1, -2)
        logits = logits / math.sqrt(features.shape[-1])
        logits = logits.masked_fill(~valid[:, None], float("-inf"))
        all_soft = (logits / self.temperature).softmax(-1)
        used = torch.zeros_like(valid)
        selected, assignments, unique_flags = [], [], []
        for index in range(self.count):
            available = valid & ~used
            has_unused = available.any(dim=-1)
            available = torch.where(has_unused[:, None], available, valid)
            # This mask must have its own storage; later mask updates must not
            # invalidate tensors saved by softmax/masked_fill for backward.
            scores = (logits[:, index] / self.temperature).masked_fill(~available, float("-inf"))
            soft = scores.softmax(-1)
            hard_index = scores.argmax(-1)
            hard = F.one_hot(hard_index, features.shape[1]).to(features.dtype)
            weights = hard + (soft - soft.detach()) if self.training else hard
            selected.append(hard_index)
            assignments.append(weights)
            unique_flags.append(has_unused)
            used = used | hard.bool()
        weights = torch.stack(assignments, dim=1)
        return {
            "seed_xyz": weights @ xyz,
            "seed_features": self.feature_norm(weights @ features + queries),
            "selected_indices": torch.stack(selected, dim=1),
            "seed_unique": torch.stack(unique_flags, dim=1),
            "soft_assignments": all_soft,
        }


def sinusoidal_embedding(values: Tensor, dim: int) -> Tensor:
    half = dim // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=values.device, dtype=values.dtype) / max(half - 1, 1))
    angles = values[..., None] * frequencies
    embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
    return F.pad(embedding, (0, dim - embedding.shape[-1]))


class ConditionalDiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ratio: int):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(dim, elementwise_affine=False) for _ in range(3)])
        self.ff = nn.Sequential(nn.Linear(dim, dim * ratio), nn.GELU(), nn.Linear(dim * ratio, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 9))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        # Nonzero residual gates permit all branches to receive gradients on
        # the first iteration, including the learnable surface selector.
        with torch.no_grad():
            for offset in (2, 5, 8):
                self.modulation[-1].bias[offset * dim:(offset + 1) * dim].fill_(0.1)

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale[:, None]) + shift[:, None]

    def forward(self, x: Tensor, seeds: Tensor, condition: Tensor) -> Tensor:
        s1, a1, g1, s2, a2, g2, s3, a3, g3 = self.modulation(condition).chunk(9, -1)
        q = self.modulate(self.norms[0](x), s1, a1)
        x = x + g1[:, None] * self.self_attention(q, q, q, need_weights=False)[0]
        q = self.modulate(self.norms[1](x), s2, a2)
        x = x + g2[:, None] * self.cross_attention(q, seeds, seeds, need_weights=False)[0]
        return x + g3[:, None] * self.ff(self.modulate(self.norms[2](x), s3, a3))


class ConditionalTrajectoryDiT(nn.Module):
    """Predict dx/dt for a complete ordered trajectory, conditioned on a set."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.geometry.dim
        self.dim = dim
        self.length = config.trajectory_points
        self.input = nn.Linear(3, dim)
        self.seed_position = nn.Sequential(nn.Linear(3, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
        self.seed_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([
            ConditionalDiTBlock(dim, config.heads, config.mlp_ratio) for _ in range(config.dit_layers)
        ])
        self.final_norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, 3)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)

    def forward(self, trajectory: Tensor, time: Tensor, seed_xyz: Tensor, seed_features: Tensor) -> Tensor:
        if trajectory.shape[1:] != (self.length, 3):
            raise ValueError(f"trajectory must have shape [B,{self.length},3]")
        positions = torch.arange(self.length, dtype=trajectory.dtype, device=trajectory.device)
        x = self.input(trajectory) + sinusoidal_embedding(positions, self.dim)[None]
        seeds = self.seed_norm(seed_features + self.seed_position(seed_xyz))
        condition = self.time_mlp(sinusoidal_embedding(time.reshape(-1) * 1000, self.dim)) + seeds.mean(1)
        for block in self.blocks:
            x = block(x, seeds, condition)
        return self.output(self.final_norm(x))


class WeldGrindModel(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.geometry = GeometryEncoder(self.config.geometry)
        self.selector = LearnableSetSelector(self.config)
        self.dit = ConditionalTrajectoryDiT(self.config)

    def encode(self, points: Tensor, roi_mask: Tensor, point_valid: Tensor | None = None) -> dict[str, Tensor]:
        geometry = self.geometry(points, roi_mask, point_valid)
        seeds = self.selector(geometry["candidate_xyz"], geometry["candidate_features"], geometry["candidate_valid"])
        return {**geometry, **seeds}

    def forward(self, points: Tensor, roi_mask: Tensor, noisy_trajectory: Tensor, time: Tensor,
                point_valid: Tensor | None = None) -> dict[str, Tensor]:
        condition = self.encode(points, roi_mask, point_valid)
        return {**condition, "velocity": self.dit(noisy_trajectory, time, condition["seed_xyz"], condition["seed_features"])}

    def flow_loss(self, points: Tensor, roi_mask: Tensor, trajectory: Tensor,
                  point_valid: Tensor | None = None, weights: dict | None = None,
                  generator: torch.Generator | None = None) -> dict[str, Tensor]:
        settings = {"flow": 1.0, "endpoint": 0.1, "coverage": 0.05, "diversity": 0.01, "smoothness": 0.001}
        if weights:
            unknown = set(weights) - set(settings)
            if unknown:
                raise ValueError(f"unknown loss weights: {sorted(unknown)}")
            settings.update(weights)
        noise = torch.randn(trajectory.shape, device=trajectory.device, dtype=trajectory.dtype, generator=generator) * self.config.noise_std
        time = torch.rand(trajectory.shape[0], device=trajectory.device, dtype=trajectory.dtype, generator=generator)
        noisy = noise.lerp(trajectory, time[:, None, None])
        output = self(points, roi_mask, noisy, time, point_valid)
        velocity = output["velocity"]
        estimate = noisy + (1 - time[:, None, None]) * velocity
        flow = F.mse_loss(velocity, trajectory - noise)
        endpoint = F.mse_loss(estimate, trajectory)
        distances = torch.cdist(output["candidate_xyz"], output["seed_xyz"]).square().amin(-1)
        valid = output["candidate_valid"]
        coverage = (distances * valid).sum() / valid.sum().clamp_min(1)
        assignments = output["soft_assignments"]
        pair_overlap = assignments @ assignments.transpose(1, 2)
        off_diagonal = ~torch.eye(self.config.seeds, dtype=torch.bool, device=points.device)
        diversity = pair_overlap[:, off_diagonal].mean() if self.config.seeds > 1 else pair_overlap.sum() * 0
        smoothness = (estimate[:, 2:] - 2 * estimate[:, 1:-1] + estimate[:, :-2]).square().mean() if self.config.trajectory_points > 2 else estimate.sum() * 0
        losses = {"flow": flow, "endpoint": endpoint, "coverage": coverage, "diversity": diversity, "smoothness": smoothness}
        losses["loss"] = sum(settings[name] * value for name, value in losses.items())
        return losses

    @torch.no_grad()
    def sample(self, points: Tensor, roi_mask: Tensor, point_valid: Tensor | None = None,
               steps: int = 32, method: str = "heun", generator: torch.Generator | None = None) -> dict[str, Tensor]:
        if steps < 1 or method not in ("euler", "heun"):
            raise ValueError("steps must be positive and method euler or heun")
        if self.training:
            raise RuntimeError("call model.eval() before sampling")
        condition = self.encode(points, roi_mask, point_valid)
        x = torch.randn((points.shape[0], self.config.trajectory_points, 3), device=points.device, dtype=points.dtype, generator=generator) * self.config.noise_std
        dt = 1.0 / steps
        for index in range(steps):
            time = x.new_full((x.shape[0],), index * dt)
            v = self.dit(x, time, condition["seed_xyz"], condition["seed_features"])
            predicted = x + dt * v
            if method == "heun":
                v_next = self.dit(predicted, time + dt, condition["seed_xyz"], condition["seed_features"])
                predicted = x + 0.5 * dt * (v + v_next)
            x = predicted
        return {**condition, "trajectory": x}
