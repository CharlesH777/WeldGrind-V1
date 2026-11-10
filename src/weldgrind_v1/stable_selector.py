"""Bounded, balanced surface-set selection without target coordinates.

Parameter names intentionally match LearnableSetSelector for audited warm
loading. The numerical behavior changes: per-token normalization, bounded
cosine scores, explicit rectangular Sinkhorn balance, and observed-geometry
repulsion prevent unbounded shared scores from choosing a single ROI corner.
No ground-truth point, trajectory, sample identifier or memory bank is used.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F

from .model import LearnableSetSelector, ModelConfig


def balanced_log_assignments(logits: Tensor, valid: Tensor, iterations: int = 12) -> Tensor:
    """Rectangular transport: query rows sum to 1, valid columns to Q/Nvalid.

    Invalid columns are exactly zero after exponentiation. Invalid all-minus-
    infinity column reductions are avoided, preventing NaN padding gradients.
    Computation is FP32, including inside an outer mixed-precision context.
    """
    if logits.ndim != 3 or valid.shape != (logits.shape[0], logits.shape[2]) or valid.dtype != torch.bool:
        raise ValueError("Expected logits [B,Q,N] and boolean valid [B,N]")
    if iterations < 1 or not valid.any(-1).all():
        raise ValueError("Positive Sinkhorn iterations and at least one valid point are required")
    log_mass = logits.float().masked_fill(~valid[:, None], -torch.inf)
    target_log_column = (logits.shape[1] / valid.sum(-1).float()).log()[:, None]
    for _ in range(iterations):
        log_mass = log_mass - torch.logsumexp(log_mass, dim=-1, keepdim=True)
        # Padded columns have no mass. Reduce finite dummies for these columns
        # instead of differentiating logsumexp([-inf,...,-inf]).
        column_sum = torch.logsumexp(log_mass.masked_fill(~valid[:, None], 0.), dim=1)
        correction = (target_log_column - column_sum).masked_fill(~valid, 0.)
        log_mass = log_mass + correction[:, None]
    return log_mass - torch.logsumexp(log_mass, dim=-1, keepdim=True)


class StableLearnableSetSelector(LearnableSetSelector):
    """Drop-in ROI-only 64-query selector with a bounded soft-gradient path.

    All historical trainable keys remain unchanged; ``load_state_dict(...,
    strict=True)`` can load the old selector. This is an initialization transfer,
    not an exact-behavior continuation. Keep geometry/selector parameters FP32;
    an external autocast context is disabled for this module's calculations.

    The hard step greedily balances learned transport scores and a bounded
    distance-to-already-selected penalty. Distances use observed XYZ only.
    ``soft_assignments`` is the balanced transport used for diagnostics/losses;
    ``sampling_soft_assignments`` is the without-replacement ST distribution.
    """

    def __init__(self, config: ModelConfig, *, temperature_floor: float = .2,
                 temperature_ceiling: float = 2., sinkhorn_iterations: int = 12,
                 spatial_weight: float = 2.):
        super().__init__(config)
        if (not math.isfinite(temperature_floor) or not math.isfinite(temperature_ceiling)
                or not 0 < temperature_floor <= temperature_ceiling):
            raise ValueError("Temperature bounds must be finite and positive")
        if not isinstance(sinkhorn_iterations, int) or sinkhorn_iterations < 1:
            raise ValueError("Sinkhorn iterations must be a positive integer")
        if not math.isfinite(spatial_weight) or spatial_weight < 0:
            raise ValueError("Spatial repulsion weight must be finite and nonnegative")
        self.temperature_floor = float(temperature_floor)
        self.temperature_ceiling = float(temperature_ceiling)
        self.sinkhorn_iterations = sinkhorn_iterations
        self.spatial_weight = float(spatial_weight)
        self.temperature = min(max(float(config.selection_temperature), temperature_floor), temperature_ceiling)

    def forward(self, xyz: Tensor, features: Tensor, valid: Tensor) -> dict:
        if (xyz.ndim != 3 or xyz.shape[-1] != 3 or features.ndim != 3
                or xyz.shape[:2] != features.shape[:2] or features.shape[-1] != self.queries.shape[-1]
                or valid.shape != xyz.shape[:2] or valid.dtype != torch.bool):
            raise ValueError("Expected aligned XYZ [B,N,3], features [B,N,D], boolean valid [B,N]")
        if not xyz.is_floating_point() or not features.is_floating_point():
            raise ValueError("XYZ and features must be floating point")
        if not valid.any(-1).all():
            raise ValueError("Each sample needs a valid observed ROI candidate")
        if not torch.isfinite(xyz[valid]).all() or not torch.isfinite(features[valid]).all():
            raise ValueError("Valid ROI XYZ/features must be finite")
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self._forward_fp32(xyz.float(), features.float(), valid)

    def _forward_fp32(self, xyz: Tensor, features: Tensor, valid: Tensor) -> dict:
        dim = features.shape[-1]
        safe_xyz = xyz.masked_fill(~valid[..., None], 0.)
        safe_features = features.masked_fill(~valid[..., None], 0.)
        # Bound observation/query contribution independently; old queries can
        # have ~4000 projected norm and must not drown out real scene features.
        memory = F.layer_norm(safe_features, (dim,)) + F.layer_norm(self.position(safe_xyz), (dim,))
        memory = memory.masked_fill(~valid[..., None], 0.)
        queries = self.queries[None].expand(xyz.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, memory, valid)
        query = F.normalize(self.query_projection(queries), dim=-1, eps=1e-6)
        key = F.normalize(self.key_projection(memory), dim=-1, eps=1e-6)
        # Explicit clamp only protects floating-point roundoff around cosine's
        # mathematical bound. No learned logit scale can diverge.
        cosine = (query @ key.transpose(-1, -2)).clamp(-1., 1.)
        temperature = min(max(self.temperature, self.temperature_floor), self.temperature_ceiling)
        logits = cosine / temperature
        log_transport = balanced_log_assignments(logits, valid, self.sinkhorn_iterations)
        transport = log_transport.exp()
        used = torch.zeros_like(valid)
        with torch.no_grad():
            high = safe_xyz.masked_fill(~valid[..., None], -torch.inf).amax(1)
            low = safe_xyz.masked_fill(~valid[..., None], torch.inf).amin(1)
            diameter_squared = (high - low).square().sum(-1).clamp_min(1e-6)
            nearest_selected_distance = torch.zeros_like(valid, dtype=torch.float32)
        selected, weights, sampling_soft, unique = [], [], [], []
        for index in range(self.count):
            available = valid & ~used
            has_unused = available.any(-1)
            available = torch.where(has_unused[:, None], available, valid)
            scores = log_transport[:, index] + self.spatial_weight * nearest_selected_distance
            scores = scores.masked_fill(~available, -torch.inf)
            soft = scores.softmax(-1)
            chosen = scores.argmax(-1)
            hard = F.one_hot(chosen, xyz.shape[1]).to(torch.float32)
            weights.append(hard + (soft - soft.detach()) if self.training else hard)
            selected.append(chosen)
            sampling_soft.append(soft)
            unique.append(has_unused)
            used = used | hard.bool()
            with torch.no_grad():
                picked_xyz = safe_xyz.gather(1, chosen[:, None, None].expand(-1, 1, 3))
                distance = (safe_xyz - picked_xyz).square().sum(-1) / diameter_squared[:, None]
                nearest_selected_distance = distance if index == 0 else torch.minimum(nearest_selected_distance, distance)
        assignments = torch.stack(weights, 1)
        observed_features = assignments @ safe_features
        seed_features = self.feature_norm(F.layer_norm(observed_features, (dim,)) + F.layer_norm(queries, (dim,)))
        with torch.no_grad():
            safe_log = log_transport.masked_fill(~valid[:, None], 0.)
            entropy = -(transport * safe_log).sum(-1)
            target_columns = (self.count / valid.sum(-1).float())[:, None]
            diagnostics = {
                "temperature": logits.new_tensor(temperature),
                "logits_min": logits.masked_select(valid[:, None].expand_as(logits)).amin(),
                "logits_max": logits.masked_select(valid[:, None].expand_as(logits)).amax(),
                "mean_entropy": entropy.mean(), "mean_max_probability": transport.amax(-1).mean(),
                "row_sum_max_error": (transport.sum(-1) - 1).abs().amax(),
                "column_sum_max_error": ((transport.sum(1) - target_columns).masked_fill(~valid, 0.)).abs().amax(),
                "query_projected_norm_mean": self.query_projection(queries).norm(dim=-1).mean(),
            }
        return {"seed_xyz": assignments @ safe_xyz, "seed_features": seed_features,
                "selected_indices": torch.stack(selected, 1), "seed_unique": torch.stack(unique, 1),
                "soft_assignments": transport, "sampling_soft_assignments": torch.stack(sampling_soft, 1),
                "diagnostics": diagnostics}
