"""Conditional flow over a complete multi-pass XYZ grinding program.

The geometry condition remains 1024 candidates / 64 selected seeds by default.
The output is a separate 1024-token ordered program, including all contact and
air moves supplied by the dataset. No pass plan, phase template, segment IDs,
toolpath anchors, or ground-truth actions are inference conditions. Four phase
classes are auxiliary supervised outputs, not inputs or a hand-written planner.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .geometry import GeometryEncoder
from .model import ConditionalDiTBlock, LearnableSetSelector, ModelConfig, sinusoidal_embedding


CONTACT, RETRACT, TRANSFER, APPROACH = 0, 1, 2, 3
PHASE_NAMES = {CONTACT: "contact", RETRACT: "retract", TRANSFER: "transfer", APPROACH: "approach"}


@dataclass
class ProgramConfig(ModelConfig):
    """Flat ModelConfig-compatible schema with a full-program output length.

    ``geometry.candidates`` is the geometry input memory count and remains
    independent of ``trajectory_points``. Supported output lengths are any
    integer >=2, including 512/1024/2048. Existing 128-point model checkpoints
    are a different architecture and must not be silently loaded as programs.
    """

    trajectory_points: int = 1024
    phase_classes: int = 4

    def __post_init__(self):
        super().__post_init__()
        if self.phase_classes != len(PHASE_NAMES):
            raise ValueError("Full-program phase_classes must be four: CONTACT/RETRACT/TRANSFER/APPROACH")


class ProgramTrajectoryDiT(nn.Module):
    """Shared ordered DiT tokens predict velocity and auxiliary action phase."""

    def __init__(self, config: ProgramConfig):
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
        self.phase_head = nn.Linear(dim, config.phase_classes)
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)
        nn.init.normal_(self.phase_head.weight, std=0.02)
        nn.init.zeros_(self.phase_head.bias)

    def forward_tokens(self, trajectory: Tensor, time: Tensor, seed_xyz: Tensor, seed_features: Tensor) -> Tensor:
        if trajectory.ndim != 3 or trajectory.shape[1:] != (self.length, 3):
            raise ValueError(f"Full program must have shape [B,{self.length},3]")
        positions = torch.arange(self.length, dtype=trajectory.dtype, device=trajectory.device)
        x = self.input(trajectory) + sinusoidal_embedding(positions, self.dim)[None]
        seeds = self.seed_norm(seed_features + self.seed_position(seed_xyz))
        condition = self.time_mlp(sinusoidal_embedding(time.reshape(-1) * 1000, self.dim)) + seeds.mean(1)
        for block in self.blocks:
            x = block(x, seeds, condition)
        return self.final_norm(x)

    def forward(self, trajectory: Tensor, time: Tensor, seed_xyz: Tensor, seed_features: Tensor) -> dict[str, Tensor]:
        tokens = self.forward_tokens(trajectory, time, seed_xyz, seed_features)
        return {"velocity": self.output(tokens), "phase_logits": self.phase_head(tokens)}


def segment_smoothness(trajectory: Tensor, segment_ids: Tensor | None = None) -> Tensor:
    """Penalize second differences only inside one annotated source segment.

    Segment IDs are used solely by this training loss. Missing IDs disable the
    regularizer rather than smoothing across unknown contact/air corners.
    Negative IDs denote unlabelled points and contribute no smoothness loss.
    """
    if segment_ids is None:
        return trajectory.sum() * 0
    if segment_ids.shape != trajectory.shape[:2] or segment_ids.dtype != torch.long:
        raise ValueError("segment_ids must be int64 [B,T]")
    if trajectory.shape[1] < 3:
        return trajectory.sum() * 0
    same_segment = ((segment_ids[:, :-2] == segment_ids[:, 1:-1])
                    & (segment_ids[:, 1:-1] == segment_ids[:, 2:])
                    & (segment_ids[:, :-2] >= 0))
    second_difference = trajectory[:, 2:] - 2 * trajectory[:, 1:-1] + trajectory[:, :-2]
    values = second_difference.square().mean(-1)
    return (values * same_segment).sum() / same_segment.sum().clamp_min(1)


class ProgramFlowModel(nn.Module):
    """Learn complete contact/air XYZ and descriptive phases from geometry.

    The model does not synthesize unlabelled initial approach/final retract
    motions. Its supervision must contain the full intended program; evaluation
    must state the source program's actual start/end scope. Raw sampled air
    points are deliberately not projected back onto the observed ROI.
    """

    def __init__(self, config: ProgramConfig | None = None):
        super().__init__()
        self.config = config or ProgramConfig()
        if not isinstance(self.config, ProgramConfig):
            raise TypeError("ProgramFlowModel requires ProgramConfig, not the single-stroke ModelConfig")
        self.geometry = GeometryEncoder(self.config.geometry)
        self.selector = LearnableSetSelector(self.config)
        self.dit = ProgramTrajectoryDiT(self.config)

    def encode(self, points: Tensor, roi_mask: Tensor, point_valid: Tensor | None = None) -> dict[str, Tensor]:
        geometry = self.geometry(points, roi_mask, point_valid)
        seeds = self.selector(geometry["candidate_xyz"], geometry["candidate_features"], geometry["candidate_valid"])
        return {**geometry, **seeds}

    def forward(self, points: Tensor, roi_mask: Tensor, noisy_trajectory: Tensor, time: Tensor,
                point_valid: Tensor | None = None) -> dict[str, Tensor]:
        condition = self.encode(points, roi_mask, point_valid)
        prediction = self.dit(noisy_trajectory, time, condition["seed_xyz"], condition["seed_features"])
        return {**condition, **prediction}

    def flow_loss(self, points: Tensor, roi_mask: Tensor, trajectory: Tensor,
                  point_valid: Tensor | None = None, *, phase_labels: Tensor | None = None,
                  segment_ids: Tensor | None = None, weights: dict | None = None,
                  generator: torch.Generator | None = None) -> dict[str, Tensor]:
        if trajectory.ndim != 3 or trajectory.shape[1:] != (self.config.trajectory_points, 3):
            raise ValueError(f"trajectory must be [B,{self.config.trajectory_points},3]")
        if phase_labels is not None:
            if phase_labels.shape != trajectory.shape[:2] or phase_labels.dtype != torch.long:
                raise ValueError("phase_labels must be int64 [B,T]")
            if ((phase_labels < CONTACT) | (phase_labels > APPROACH)).any():
                raise ValueError("phase_labels must use CONTACT0/RETRACT1/TRANSFER2/APPROACH3")
        settings = {"flow": 1.0, "endpoint": 0.1, "phase": 0.1,
                    "coverage": 0.05, "diversity": 0.01, "smoothness": 0.001}
        if weights:
            unknown = set(weights) - set(settings)
            if unknown:
                raise ValueError(f"Unknown program loss weights: {sorted(unknown)}")
            settings.update(weights)
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in settings.values()):
            raise ValueError("Program loss weights must be finite and nonnegative")
        noise = torch.randn(trajectory.shape, device=trajectory.device, dtype=trajectory.dtype,
                            generator=generator) * self.config.noise_std
        time = torch.rand(trajectory.shape[0], device=trajectory.device, dtype=trajectory.dtype, generator=generator)
        noisy = noise.lerp(trajectory, time[:, None, None])
        output = self(points, roi_mask, noisy, time, point_valid)
        velocity = output["velocity"]
        estimate = noisy + (1 - time[:, None, None]) * velocity
        # Every ordered XYZ token is supervised, including all air moves.
        flow = F.mse_loss(velocity, trajectory - noise)
        endpoint = F.mse_loss(estimate, trajectory)
        phase = (F.cross_entropy(output["phase_logits"].reshape(-1, self.config.phase_classes), phase_labels.reshape(-1))
                 if phase_labels is not None else velocity.sum() * 0)
        distances = torch.cdist(output["candidate_xyz"], output["seed_xyz"]).square().amin(-1)
        valid = output["candidate_valid"]
        coverage = (distances * valid).sum() / valid.sum().clamp_min(1)
        assignments = output["soft_assignments"]
        pair_overlap = assignments @ assignments.transpose(1, 2)
        off_diagonal = ~torch.eye(self.config.seeds, dtype=torch.bool, device=points.device)
        diversity = pair_overlap[:, off_diagonal].mean() if self.config.seeds > 1 else pair_overlap.sum() * 0
        smoothness = segment_smoothness(estimate, segment_ids)
        losses = {"flow": flow, "endpoint": endpoint, "phase": phase, "coverage": coverage,
                  "diversity": diversity, "smoothness": smoothness}
        losses["loss"] = sum(settings[name] * value for name, value in losses.items())
        return losses

    @torch.no_grad()
    def sample(self, points: Tensor, roi_mask: Tensor, point_valid: Tensor | None = None,
               steps: int = 32, method: str = "heun", generator: torch.Generator | None = None,
               initial_noise: Tensor | None = None) -> dict[str, Tensor]:
        """Generate from random x0; explicit initial_noise is already std-scaled.

        Passing independently seeded x0 per image makes initial noise invariant
        to evaluation batch packing. This is a noise input, not a GT trajectory
        or action template. Generator is used only when initial_noise is absent.
        """
        if not isinstance(steps, int) or steps < 1 or method not in ("euler", "heun"):
            raise ValueError("steps must be a positive integer and method euler or heun")
        if self.training:
            raise RuntimeError("Call model.eval() before full-program sampling")
        condition = self.encode(points, roi_mask, point_valid)
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
                predicted = x + 0.5 * dt * (velocity + next_velocity)
            x = predicted
        final = self.dit(x, x.new_ones(x.shape[0]), condition["seed_xyz"], condition["seed_features"])
        return {**condition, "trajectory": x, "phase_logits": final["phase_logits"],
                "phases": final["phase_logits"].argmax(-1)}
