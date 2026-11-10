"""Observed-scale conditioning and explicit pure-noise program supervision.

This is a separate reviewed training strategy. Existing geometry, selector and
DiT parameter keys stay unchanged for a warm start. Scale is the measured ROI
normalization scale, never a target-program statistic. No planner, phase/pass
template or GT trajectory is an inference input.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .program_model import APPROACH, CONTACT, ProgramConfig, ProgramFlowModel, segment_smoothness


DEFAULT_METRIC_LOSS_WEIGHTS = {
    "flow": 1.0, "endpoint": .1, "phase": .1,
    "coverage": .05, "diversity": .01, "smoothness": .001,
    "conditional_t0": 1.0, "conditional_phase": .2,
    "physical_flow": .5, "physical_t0": 1.0,
}


class MetricProgramFlowModel(ProgramFlowModel):
    """Complete-program flow with an observed physical-scale condition.

    Every public encoding/generation call requires scale_mm: a finite positive
    scalar or [B] ROI scale from observation preprocessing. The added encoder's
    last layer is zero initialized, so loading an old ProgramFlowModel state
    changes only the four missing scale_encoder parameter keys and initially
    preserves the old vector field.
    """

    def __init__(self, config: ProgramConfig | None = None, *, physical_reference_mm: float = 20.,
                 low_time_fraction: float = .5, low_time_max: float = .25):
        super().__init__(config)
        if not math.isfinite(physical_reference_mm) or physical_reference_mm <= 0:
            raise ValueError("physical_reference_mm must be finite and positive")
        if not 0 <= low_time_fraction <= 1 or not 0 < low_time_max <= 1:
            raise ValueError("Invalid early-flow-time sampling mixture")
        self.physical_reference_mm = float(physical_reference_mm)
        self.low_time_fraction = float(low_time_fraction)
        self.low_time_max = float(low_time_max)
        dim = self.config.geometry.dim
        self.scale_encoder = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.scale_encoder[-1].weight)
        nn.init.zeros_(self.scale_encoder[-1].bias)

    @staticmethod
    def _observed_scale(scale_mm, points: Tensor) -> Tensor:
        scale = torch.as_tensor(scale_mm, device=points.device, dtype=points.dtype)
        if scale.ndim == 0 or scale.shape == (1,):
            scale = scale.reshape(1).expand(points.shape[0])
        elif scale.shape != (points.shape[0],):
            raise ValueError("scale_mm must be a scalar or [B] observed ROI scale")
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("scale_mm must be finite and positive")
        return scale

    def encode(self, points: Tensor, roi_mask: Tensor, point_valid: Tensor | None = None, *, scale_mm) -> dict[str, Tensor]:
        scale = self._observed_scale(scale_mm, points)
        condition = super().encode(points, roi_mask, point_valid)
        embedding = self.scale_encoder(torch.log(scale / self.physical_reference_mm)[:, None])
        return {**condition, "seed_features": condition["seed_features"] + embedding[:, None],
                "observed_scale_mm": scale}

    def forward(self, points: Tensor, roi_mask: Tensor, noisy_trajectory: Tensor, time: Tensor,
                point_valid: Tensor | None = None, *, scale_mm) -> dict[str, Tensor]:
        condition = self.encode(points, roi_mask, point_valid, scale_mm=scale_mm)
        return {**condition, **self.dit(noisy_trajectory, time, condition["seed_xyz"], condition["seed_features"])}

    def flow_loss(self, points: Tensor, roi_mask: Tensor, trajectory: Tensor,
                  point_valid: Tensor | None = None, *, scale_mm, phase_labels: Tensor | None = None,
                  segment_ids: Tensor | None = None, weights: dict | None = None,
                  generator: torch.Generator | None = None) -> dict[str, Tensor]:
        if trajectory.ndim != 3 or trajectory.shape[1:] != (self.config.trajectory_points, 3):
            raise ValueError(f"trajectory must be [B,{self.config.trajectory_points},3]")
        if phase_labels is not None:
            if phase_labels.shape != trajectory.shape[:2] or phase_labels.dtype != torch.long:
                raise ValueError("phase_labels must be int64 [B,T]")
            if ((phase_labels < CONTACT) | (phase_labels > APPROACH)).any():
                raise ValueError("phase_labels must use CONTACT0/RETRACT1/TRANSFER2/APPROACH3")
        settings = dict(DEFAULT_METRIC_LOSS_WEIGHTS)
        if weights:
            unknown = set(weights) - set(settings)
            if unknown:
                raise ValueError(f"Unknown metric-program loss weights: {sorted(unknown)}")
            settings.update(weights)
        if any(not math.isfinite(float(value)) or value < 0 for value in settings.values()):
            raise ValueError("Metric-program loss weights must be finite and nonnegative")
        if settings["conditional_t0"] < 1 or settings["conditional_phase"] < .2:
            raise ValueError("Explicit t0 supervision requires conditional_t0 >= 1 and conditional_phase >= .2")

        # Encode observations exactly once. Both DiT branches share this graph.
        condition = self.encode(points, roi_mask, point_valid, scale_mm=scale_mm)
        noise = torch.randn(trajectory.shape, device=trajectory.device, dtype=trajectory.dtype,
                            generator=generator) * self.config.noise_std
        time = torch.rand(trajectory.shape[0], device=trajectory.device, dtype=trajectory.dtype, generator=generator)
        if self.low_time_fraction > 0:
            early = torch.rand(time.shape, device=time.device, dtype=time.dtype, generator=generator) < self.low_time_fraction
            time = torch.where(early, time * self.low_time_max, time)
        noisy = noise.lerp(trajectory, time[:, None, None])
        teacher = self.dit(noisy, time, condition["seed_xyz"], condition["seed_features"])
        # Crucially, this branch sees the original pure noise, never noisy x_t,
        # target labels, segment/pass metadata or any teacher prediction.
        pure = self.dit(noise, torch.zeros_like(time), condition["seed_xyz"], condition["seed_features"])
        target_velocity = trajectory - noise
        estimate = noisy + (1 - time[:, None, None]) * teacher["velocity"]
        estimate_t0 = noise + pure["velocity"]
        flow = F.mse_loss(teacher["velocity"], target_velocity)
        endpoint = F.mse_loss(estimate, trajectory)
        conditional_t0 = F.mse_loss(estimate_t0, trajectory)
        if phase_labels is None:
            phase = teacher["velocity"].sum() * 0
            conditional_phase = pure["velocity"].sum() * 0
        else:
            labels = phase_labels.reshape(-1)
            phase = F.cross_entropy(teacher["phase_logits"].reshape(-1, self.config.phase_classes), labels)
            conditional_phase = F.cross_entropy(pure["phase_logits"].reshape(-1, self.config.phase_classes), labels)
        factor = condition["observed_scale_mm"].to(trajectory.dtype)[:, None, None] / self.physical_reference_mm
        physical_flow = ((teacher["velocity"] - target_velocity) * factor).square().mean()
        physical_t0 = ((estimate_t0 - trajectory) * factor).square().mean()
        distances = torch.cdist(condition["candidate_xyz"], condition["seed_xyz"]).square().amin(-1)
        valid = condition["candidate_valid"]
        coverage = (distances * valid).sum() / valid.sum().clamp_min(1)
        assignments = condition["soft_assignments"]
        overlap = assignments @ assignments.transpose(1, 2)
        off_diagonal = ~torch.eye(self.config.seeds, dtype=torch.bool, device=points.device)
        diversity = overlap[:, off_diagonal].mean() if self.config.seeds > 1 else overlap.sum() * 0
        smoothness = segment_smoothness(estimate, segment_ids)
        losses = {"flow": flow, "endpoint": endpoint, "phase": phase, "coverage": coverage,
                  "diversity": diversity, "smoothness": smoothness, "conditional_t0": conditional_t0,
                  "conditional_phase": conditional_phase, "physical_flow": physical_flow, "physical_t0": physical_t0}
        losses["loss"] = sum(settings[name] * value for name, value in losses.items())
        return losses

    @torch.no_grad()
    def sample(self, points: Tensor, roi_mask: Tensor, point_valid: Tensor | None = None,
               steps: int = 32, method: str = "heun", generator: torch.Generator | None = None,
               initial_noise: Tensor | None = None, *, scale_mm) -> dict[str, Tensor]:
        if not isinstance(steps, int) or steps < 1 or method not in ("euler", "heun"):
            raise ValueError("steps must be a positive integer and method euler or heun")
        if self.training:
            raise RuntimeError("Call model.eval() before metric-program sampling")
        condition = self.encode(points, roi_mask, point_valid, scale_mm=scale_mm)
        shape = (points.shape[0], self.config.trajectory_points, 3)
        if initial_noise is None:
            x = torch.randn(shape, device=points.device, dtype=points.dtype, generator=generator) * self.config.noise_std
        else:
            if not isinstance(initial_noise, Tensor) or initial_noise.shape != shape or not initial_noise.is_floating_point():
                raise ValueError(f"initial_noise must be floating point with shape {shape}")
            if not torch.isfinite(initial_noise).all():
                raise ValueError("initial_noise must be finite")
            x = initial_noise.to(device=points.device, dtype=points.dtype).clone()
        dt = 1.0 / steps
        for index in range(steps):
            time = x.new_full((x.shape[0],), index * dt)
            velocity = self.dit(x, time, condition["seed_xyz"], condition["seed_features"])["velocity"]
            predicted = x + dt * velocity
            if method == "heun":
                next_velocity = self.dit(predicted, time + dt, condition["seed_xyz"], condition["seed_features"])["velocity"]
                predicted = x + .5 * dt * (velocity + next_velocity)
            x = predicted
        final = self.dit(x, x.new_ones(x.shape[0]), condition["seed_xyz"], condition["seed_features"])
        return {**condition, "trajectory": x, "phase_logits": final["phase_logits"],
                "phases": final["phase_logits"].argmax(-1)}
