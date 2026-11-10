"""Same-capacity selector repair and differentiable autonomous program supervision.

Historical models and loss implementations stay unchanged. Old metric-v3 keys
load strictly, but replacing the selector intentionally changes its algorithm.
GT positions/phases/segments enter losses only, never autonomous generation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .metric_program_model import DEFAULT_METRIC_LOSS_WEIGHTS, MetricProgramFlowModel
from .program_model import ProgramConfig, segment_smoothness
from .stable_selector import StableLearnableSetSelector


DEFAULT_ROLLOUT_LOSS_WEIGHTS = {
    "rollout_position": 1., "rollout_phase": .5, "rollout_smoothness": .001,
}


@dataclass(frozen=True)
class RolloutRepairConfig:
    rollout_steps: int = 4
    rollout_method: str = "euler"
    class_balanced_rollout_phase: bool = True
    class_weight_cap: float = 4.
    # Historical repaired checkpoints omitted these fields. Defaults preserve
    # their 20 mm scaling and eager activation behavior on reload.
    rollout_reference_mm: float = 20.
    rollout_activation_checkpoint: bool = False

    def __post_init__(self):
        if isinstance(self.rollout_steps, bool) or not isinstance(self.rollout_steps, int) or self.rollout_steps < 1:
            raise ValueError("rollout_steps must be a positive integer")
        if self.rollout_method not in ("euler", "heun"):
            raise ValueError("rollout_method must be euler or heun")
        if not isinstance(self.class_balanced_rollout_phase, bool):
            raise ValueError("class_balanced_rollout_phase must be boolean")
        if (isinstance(self.class_weight_cap, bool) or not isinstance(self.class_weight_cap, (int, float))
                or not math.isfinite(self.class_weight_cap) or self.class_weight_cap < 1.):
            raise ValueError("class_weight_cap must be finite and >= 1")
        if (isinstance(self.rollout_reference_mm, bool) or not isinstance(self.rollout_reference_mm, (int, float))
                or not math.isfinite(self.rollout_reference_mm) or self.rollout_reference_mm <= 0):
            raise ValueError("rollout_reference_mm must be finite and positive")
        if not isinstance(self.rollout_activation_checkpoint, bool):
            raise ValueError("rollout_activation_checkpoint must be boolean")


class RepairedMetricProgramFlowModel(MetricProgramFlowModel):
    """Bounded selector + actual generated-endpoint losses, with no extra weights.

    ``rollout_enabled`` is an explicit per-update scheduling choice owned by the
    finite-budget trainer. False preserves all base losses and avoids the extra
    rollout; its scalar marker is never included in the weighted objective.
    Training uses a differentiable observation-only solver, not ``sample()``,
    whose inherited inference API correctly remains eval-only and no-grad.
    """

    def __init__(self, config: ProgramConfig | None = None, *, repair_options=None,
                 selector_options: dict | None = None, physical_reference_mm: float = 20.,
                 low_time_fraction: float = .5, low_time_max: float = .25):
        super().__init__(config, physical_reference_mm=physical_reference_mm,
                         low_time_fraction=low_time_fraction, low_time_max=low_time_max)
        self.repair_options = (repair_options if isinstance(repair_options, RolloutRepairConfig)
                               else RolloutRepairConfig(**(repair_options or {})))
        self.selector = StableLearnableSetSelector(self.config, **(selector_options or {}))

    def scale_reference_metadata(self) -> dict[str, float]:
        """Expose both independent references for checkpoint/experiment receipts."""
        return {"base_physical_reference_mm": self.physical_reference_mm,
                "rollout_reference_mm": self.repair_options.rollout_reference_mm,
                "rollout_quadratic_scale_relative_to_base":
                    (self.physical_reference_mm / self.repair_options.rollout_reference_mm) ** 2}

    def differentiable_rollout(self, points: Tensor, roi_mask: Tensor, point_valid: Tensor | None = None,
                               *, scale_mm, steps: int | None = None, method: str | None = None,
                               initial_noise: Tensor | None = None,
                               generator: torch.Generator | None = None) -> dict[str, Tensor]:
        """Observation-only Euler/Heun integration, including t=1 phase decoding.

        There is no teacher forcing, stop-gradient, target prefix, phase plan,
        segment boundary, GT pass index, or target surface in this interface.
        The same condition is encoded once and reused at every solver step.
        """
        steps = self.repair_options.rollout_steps if steps is None else steps
        method = self.repair_options.rollout_method if method is None else method
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1 or method not in ("euler", "heun"):
            raise ValueError("steps must be positive integer and method euler or heun")
        condition = self.encode(points, roi_mask, point_valid, scale_mm=scale_mm)
        def predict(state, time):
            if self.repair_options.rollout_activation_checkpoint and torch.is_grad_enabled():
                # Non-reentrant checkpointing supports dict outputs, preserves
                # parameter/condition gradients and the DiT's AMP context.
                return checkpoint(self.dit, state, time, condition["seed_xyz"],
                                  condition["seed_features"], use_reentrant=False)
            return self.dit(state, time, condition["seed_xyz"], condition["seed_features"])
        shape = (points.shape[0], self.config.trajectory_points, 3)
        if initial_noise is None:
            x = torch.randn(shape, device=points.device, dtype=points.dtype,
                            generator=generator) * self.config.noise_std
        else:
            if (not isinstance(initial_noise, Tensor) or initial_noise.shape != shape
                    or not initial_noise.is_floating_point() or not torch.isfinite(initial_noise).all()):
                raise ValueError(f"initial_noise must be finite floating point with shape {shape}")
            x = initial_noise.to(device=points.device, dtype=points.dtype).clone()
        dt = 1. / steps
        for index in range(steps):
            time = x.new_full((x.shape[0],), index * dt)
            velocity = predict(x, time)["velocity"]
            predicted = x + dt * velocity
            if method == "heun":
                next_velocity = predict(predicted, time + dt)["velocity"]
                predicted = x + .5 * dt * (velocity + next_velocity)
            x = predicted
        final = predict(x, x.new_ones(x.shape[0]))
        return {**condition, "trajectory": x, "phase_logits": final["phase_logits"],
                "phases": final["phase_logits"].argmax(-1)}

    def _rollout_phase_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        weight = None
        if self.repair_options.class_balanced_rollout_phase:
            # Current TRAIN batch labels only. No validation class frequencies,
            # checkpoint selection or hidden class-template input is involved.
            counts = torch.bincount(labels.reshape(-1), minlength=self.config.phase_classes).to(logits.dtype)
            weight = (counts.sum() / (self.config.phase_classes * counts.clamp_min(1.))).sqrt()
            weight = weight.clamp(max=self.repair_options.class_weight_cap)
            weight = torch.where(counts > 0, weight, torch.zeros_like(weight))
        return F.cross_entropy(logits.reshape(-1, self.config.phase_classes), labels.reshape(-1), weight=weight)

    def flow_loss(self, points: Tensor, roi_mask: Tensor, trajectory: Tensor,
                  point_valid: Tensor | None = None, *, scale_mm, phase_labels: Tensor | None = None,
                  segment_ids: Tensor | None = None, weights: dict | None = None,
                  generator: torch.Generator | None = None, rollout_enabled: bool = True) -> dict[str, Tensor]:
        if not isinstance(rollout_enabled, bool):
            raise ValueError("rollout_enabled must be an explicit boolean")
        settings = {**DEFAULT_METRIC_LOSS_WEIGHTS, **DEFAULT_ROLLOUT_LOSS_WEIGHTS}
        if weights:
            unknown = set(weights) - set(settings)
            if unknown:
                raise ValueError(f"Unknown repaired-program loss weights: {sorted(unknown)}")
            settings.update(weights)
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in settings.values()):
            raise ValueError("Loss weights must be finite nonnegative numbers")
        # Keep all old loss terms and their original definitions. The base
        # encoder pass is shared by its teacher/t0 branches as before.
        losses = super().flow_loss(points, roi_mask, trajectory, point_valid, scale_mm=scale_mm,
            phase_labels=phase_labels, segment_ids=segment_ids,
            weights={key: settings[key] for key in DEFAULT_METRIC_LOSS_WEIGHTS}, generator=generator)
        zero = losses["loss"] * 0.
        additional = {key: zero for key in DEFAULT_ROLLOUT_LOSS_WEIGHTS}
        if rollout_enabled:
            # This branch makes a new independent Gaussian draw. GT trajectory,
            # phase and segments are not passed to the autonomous integrator.
            generated = self.differentiable_rollout(points, roi_mask, point_valid,
                                                   scale_mm=scale_mm, generator=generator)
            scale = generated["observed_scale_mm"].to(trajectory.dtype)[:, None, None]
            # This reference belongs only to autonomous endpoint supervision;
            # changing it does not change the historical teacher/t0 objective.
            residual_physical = (generated["trajectory"] - trajectory) * (scale / self.repair_options.rollout_reference_mm)
            additional["rollout_position"] = residual_physical.square().mean()
            additional["rollout_phase"] = (self._rollout_phase_loss(generated["phase_logits"], phase_labels)
                                            if phase_labels is not None else zero)
            # Penalize high-frequency *error* inside each GT segment; preserve
            # actual curved-path second differences rather than flattening them.
            additional["rollout_smoothness"] = segment_smoothness(residual_physical, segment_ids)
        losses["loss"] = losses["loss"] + sum(settings[key] * value for key, value in additional.items())
        losses.update(additional)
        losses["rollout_executed"] = torch.tensor(rollout_enabled, device=points.device, dtype=torch.bool)
        return losses
