"""Observable process conditioning and distribution-consistent conditional FM.

Historical bounded models are unchanged. Only a zero-initialized process encoder
adds parameters. No chart direction, phase, pass, segment, or GT trajectory is an
inference condition. Teacher endpoints and terminal phases are training targets;
independently sampled programs are never regressed against an unrelated GT mode.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .bounded_scratch_program_model import ARMS, BoundedScratchProgramModel
from .program_model import ProgramConfig


FORMAT = "weldgrind-process-conditioned-cfm-v1"
STRATEGY = "observable_process_standard_cfm_terminal_teacher_phase_v1"
PROCESS_PARAMETER_NAMES = (
    "tool_effective_radius_mm", "layer_depth_mm", "clearance_mm", "overlap",
)
PROCESS_PARAMETER_UNITS = ("mm", "mm", "mm", "fraction")
PROCESS_CONDITION_SCHEMA = "weldgrind-known-process-conditions-v1"
PROCESS_NORMALIZATION = {
    "positive_parameter_reference_mm": (1., 1., 1.),
    "positive_parameter_transform": "log(value / 1 mm)",
    "overlap_transform": "logit(clamp(overlap, 1e-4, 1-1e-4))",
    "overlap_logit_epsilon": 1e-4,
    "statistics_source": "fixed physical units; no learned or GT-derived statistics",
}
NEW_PROCESS_PARAMETER_NAMES = (
    "process_encoder.0.weight", "process_encoder.0.bias",
    "process_encoder.2.weight", "process_encoder.2.bias",
)
DEFAULT_PROCESS_FLOW_LOSS_WEIGHTS = {
    "flow": 1., "endpoint": .1, "physical_flow": .5,
    "phase": .1, "terminal_phase": .5, "coverage": .05, "diversity": .01,
}


def normalize_process_params(process_params: Tensor, *, batch_size: int, device=None) -> Tensor:
    """Validate explicit [B,4] observations, then express them in fixed units.

    The first three positive quantities are in mm. Overlap is a dimensionless
    fraction in [0,1). Its transform is bounded near the valid interval endpoints
    only to avoid a logarithmic infinity, without guessing missing process data.
    """
    if not isinstance(process_params, Tensor) or not process_params.is_floating_point():
        raise ValueError("process_params must be an explicit floating Tensor [B,4]")
    if process_params.shape != (batch_size, len(PROCESS_PARAMETER_NAMES)):
        raise ValueError("process_params must have shape [B,4] in the documented order")
    values = process_params.to(device=device, dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError("process_params must be finite and representable in FP32")
    if (values[:, :3] <= 0).any():
        raise ValueError("Tool radius, layer depth and clearance must be positive mm")
    overlap = values[:, 3:]
    if ((overlap < 0) | (overlap >= 1)).any():
        raise ValueError("Process overlap must satisfy 0 <= overlap < 1")
    epsilon = PROCESS_NORMALIZATION["overlap_logit_epsilon"]
    # The unit references are each exactly 1 mm, not sample/target statistics.
    return torch.cat((values[:, :3].log(), torch.logit(overlap.clamp(epsilon, 1-epsilon))), dim=-1)


def _weights(overrides=None):
    result = dict(DEFAULT_PROCESS_FLOW_LOSS_WEIGHTS)
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ValueError("Loss weights must be a mapping")
        unknown = set(overrides)-set(result)
        if unknown:
            raise ValueError(f"Unknown process-CFM loss weights: {sorted(unknown)}")
        result.update(overrides)
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError("Loss weights must be finite nonnegative numbers")
    if result["flow"] <= 0:
        raise ValueError("The standard conditional flow loss must retain a positive weight")
    return result


class ProcessConditionedFlowModel(BoundedScratchProgramModel):
    """The bounded architecture plus four explicitly observed process values."""

    def __init__(self, config=None, *, bounds=None, activation_checkpoint=True,
                 selector_expanded=False, objective_migration=None):
        super().__init__(config, bounds=bounds, activation_checkpoint=activation_checkpoint,
                         selector_expanded=selector_expanded)
        if objective_migration is not None and not isinstance(objective_migration, dict):
            raise ValueError("objective_migration must be a dictionary or None")
        self.objective_migration = deepcopy(objective_migration or {})
        self.loss_weights = _weights(self.objective_migration.get("loss_weights"))
        self.class_balanced_phase = self.objective_migration.get("class_balanced_phase", True)
        self.class_weight_cap = self.objective_migration.get("class_weight_cap", 4.)
        if not isinstance(self.class_balanced_phase, bool):
            raise ValueError("class_balanced_phase must be boolean")
        if (isinstance(self.class_weight_cap, bool) or not isinstance(self.class_weight_cap, (int, float))
                or not math.isfinite(self.class_weight_cap) or self.class_weight_cap < 1):
            raise ValueError("class_weight_cap must be finite and >= 1")
        dim = self.config.geometry.dim
        self.process_encoder = nn.Sequential(nn.Linear(4, dim), nn.SiLU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.process_encoder[-1].weight)
        nn.init.zeros_(self.process_encoder[-1].bias)

    def process_conditioning_metadata(self):
        return {"parameter_names": list(PROCESS_PARAMETER_NAMES),
                "parameter_units": list(PROCESS_PARAMETER_UNITS),
                "normalization": deepcopy(PROCESS_NORMALIZATION),
                "new_parameter_names": list(NEW_PROCESS_PARAMETER_NAMES),
                "new_parameter_count": sum(p.numel() for p in self.process_encoder.parameters()),
                "strategy": STRATEGY, "loss_weights": dict(self.loss_weights),
                "class_balanced_phase": self.class_balanced_phase,
                "class_weight_cap": self.class_weight_cap}

    def load_bounded_state_dict(self, state_dict):
        """Explicit audited migration of all old keys; only four new keys remain."""
        current = self.state_dict()
        new = set(NEW_PROCESS_PARAMETER_NAMES)
        expected = set(current)-new
        if set(state_dict) != expected:
            raise ValueError("Parent state must contain exactly the original bounded-model keys")
        for name, value in state_dict.items():
            if not isinstance(value, Tensor) or value.shape != current[name].shape or value.dtype != current[name].dtype:
                raise ValueError(f"Parent tensor shape/dtype mismatch: {name}")
        if any(torch.count_nonzero(value).item() for value in (
                self.process_encoder[-1].weight, self.process_encoder[-1].bias)):
            raise ValueError("Use a fresh zero-initialized process encoder for parent migration")
        current.update(state_dict)
        self.load_state_dict(current, strict=True)
        names = [name for name, _ in self.named_parameters() if name not in new]
        return {"copied_parameter_names": names, "new_parameter_names": sorted(new),
                "new_parameter_count": sum(p.numel() for p in self.process_encoder.parameters()),
                "parent_keys_preserved": True, "new_process_last_layer_zero": True}

    def encode(self, points, roi_mask, point_valid=None, *, scale_mm, process_params):
        normalized = normalize_process_params(process_params, batch_size=points.shape[0], device=points.device)
        condition = super().encode(points, roi_mask, point_valid, scale_mm=scale_mm)
        embedding = self.process_encoder(normalized).to(condition["seed_features"].dtype)
        return {**condition, "seed_features": condition["seed_features"]+embedding[:, None]}

    def forward(self, points, roi_mask, noisy_trajectory, time, point_valid=None, *, scale_mm, process_params):
        condition = self.encode(points, roi_mask, point_valid, scale_mm=scale_mm, process_params=process_params)
        return {**condition, **self.dit(noisy_trajectory, time, condition["seed_xyz"], condition["seed_features"])}

    def differentiable_rollout(self, points, roi_mask, point_valid=None, *, scale_mm, process_params,
                               steps=None, method=None, initial_noise=None, generator=None):
        condition = self.encode(points, roi_mask, point_valid, scale_mm=scale_mm, process_params=process_params)
        shape = (points.shape[0], self.config.trajectory_points, 3)
        if initial_noise is None:
            noise = torch.randn(shape, device=points.device, dtype=points.dtype,
                                generator=generator)*self.config.noise_std
        else:
            if (not isinstance(initial_noise, Tensor) or not initial_noise.is_floating_point()
                    or initial_noise.shape != shape or not torch.isfinite(initial_noise).all()):
                raise ValueError(f"initial_noise must be finite floating point {shape}")
            noise = initial_noise.to(points).clone()
        return self._integrate(condition, noise, 8 if steps is None else steps,
                               'euler' if method is None else method)

    @torch.no_grad()
    def sample(self, points, roi_mask, point_valid=None, steps=8, method='euler',
               generator=None, initial_noise=None, *, scale_mm, process_params):
        if self.training:
            raise RuntimeError("Call eval before process-conditioned sampling")
        return self.differentiable_rollout(points, roi_mask, point_valid, scale_mm=scale_mm,
            process_params=process_params, steps=steps, method=method,
            initial_noise=initial_noise, generator=generator)

    def _phase_loss(self, logits, labels):
        # CE and class counts stay FP32 even though the DiT may run in BF16.
        weight = None
        if self.class_balanced_phase:
            counts = torch.bincount(labels.reshape(-1), minlength=self.config.phase_classes).float()
            weight = (counts.sum()/(self.config.phase_classes*counts.clamp_min(1))).sqrt()
            weight = weight.clamp(max=self.class_weight_cap)
            weight = torch.where(counts > 0, weight, torch.zeros_like(weight))
        return F.cross_entropy(logits.float().reshape(-1, self.config.phase_classes),
                               labels.reshape(-1), weight=weight)

    def flow_loss(self, points, roi_mask, trajectory, point_valid=None, *, scale_mm,
                  process_params, phase_labels, segment_ids=None, generator=None, weights=None):
        """Supervise the coupled teacher vector field, never unrelated rollout modes.

        Segment IDs are accepted for the existing batch interface but do not
        enter the objective. The endpoint term equals (1-t)^2 times velocity
        MSE. Its weight and physical scale depend only on time/observations, so
        they retain the standard conditional-FM optimum. Terminal phase targets
        are decoded on the clean GT state *only in training*.
        """
        if weights is not None and not isinstance(weights, Mapping):
            raise ValueError("Loss weights must be a mapping")
        settings = _weights({**self.loss_weights, **({} if weights is None else weights)})
        if trajectory.shape != (points.shape[0], self.config.trajectory_points, 3):
            raise ValueError("Invalid full-program target shape")
        if not trajectory.is_floating_point() or not torch.isfinite(trajectory).all():
            raise ValueError("Trajectory supervision must be finite floating point")
        if (phase_labels.shape != trajectory.shape[:2] or phase_labels.dtype != torch.long
                or ((phase_labels < 0) | (phase_labels >= self.config.phase_classes)).any()):
            raise ValueError("Phase supervision must be int64 [B,T] using classes 0..3")
        condition = self.encode(points, roi_mask, point_valid, scale_mm=scale_mm, process_params=process_params)
        noise = torch.randn(trajectory.shape, device=points.device, dtype=points.dtype,
                            generator=generator)*self.config.noise_std
        t = self.uniform_times(points.shape[0], points.device, points.dtype, generator)
        noisy = noise.lerp(trajectory, t[:, None, None])
        teacher = self.dit(noisy, t, condition['seed_xyz'], condition['seed_features'])
        terminal = self.dit(trajectory, torch.ones_like(t), condition['seed_xyz'], condition['seed_features'])
        target_velocity = trajectory-noise
        estimate = noisy+(1-t[:, None, None])*teacher['velocity']
        factor = condition['observed_scale_mm'][:, None, None]/self.physical_reference_mm
        valid = condition['candidate_valid']
        distances = torch.cdist(condition['candidate_xyz'], condition['seed_xyz']).square().amin(-1)
        soft = condition['soft_assignments']
        overlap = soft@soft.transpose(1, 2)
        off = ~torch.eye(self.config.seeds, device=points.device, dtype=torch.bool)
        losses = {
            'flow': F.mse_loss(teacher['velocity'], target_velocity),
            'endpoint': F.mse_loss(estimate, trajectory),
            'physical_flow': ((teacher['velocity']-target_velocity)*factor).square().mean(),
            'phase': self._phase_loss(teacher['phase_logits'], phase_labels),
            'terminal_phase': self._phase_loss(terminal['phase_logits'], phase_labels),
            'coverage': (distances*valid).sum()/valid.sum().clamp_min(1),
            'diversity': overlap[:, off].mean() if self.config.seeds > 1 else overlap.sum()*0.,
        }
        losses['loss'] = sum(settings[name]*value for name, value in losses.items())
        return losses


def make_process_conditioned_model(model_config: dict, arm: str, objective_migration: dict | None = None) -> nn.Module:
    """Construct an explicit arm for audited state/optimizer migration by its runner."""
    if arm not in ARMS:
        raise ValueError("Unknown process-conditioned arm")
    config = ProgramConfig.from_dict(model_config)
    if (config.selector_layers, config.dit_layers) != ARMS[arm]:
        raise ValueError("Model configuration does not match the requested arm")
    return ProcessConditionedFlowModel(config, selector_expanded=arm == 'selector6',
                                       objective_migration=objective_migration)


def _checked_process_binding(migration):
    """Check the saved manifest identity; Dataset checks raw sources on loading."""
    if not isinstance(migration, dict) or migration.get('strategy') != STRATEGY:
        raise ValueError("An explicit process objective migration is required")
    path = migration.get("process_manifest")
    expected_hash = migration.get("process_manifest_sha256")
    if not isinstance(path, str) or not path:
        raise ValueError("Checkpoint has no explicit process manifest")
    if (not isinstance(expected_hash, str) or len(expected_hash) != 64
            or any(c not in '0123456789abcdef' for c in expected_hash)):
        raise ValueError("Checkpoint has no valid process manifest SHA256")
    path = Path(path).resolve(strict=True)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError("Process manifest SHA256 changed since training")
    manifest = json.loads(payload)
    if (manifest.get('schema') != PROCESS_CONDITION_SCHEMA
            or manifest.get('parameter_names') != list(PROCESS_PARAMETER_NAMES)
            or manifest.get('parameter_units') != list(PROCESS_PARAMETER_UNITS)):
        raise ValueError("Process manifest schema, parameter order or units differ")
    rows = manifest.get('samples')
    if not isinstance(rows, list) or not rows or manifest.get('count') != len(rows):
        raise ValueError("Process manifest count/samples disagree")
    ids = [row.get('sample_id') for row in rows]
    if any(not isinstance(sid, str) or not sid for sid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Process manifest sample IDs are missing or duplicated")
    try:
        values = torch.tensor([row['process_params'] for row in rows], dtype=torch.float32)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Process manifest has malformed explicit parameters") from error
    normalize_process_params(values, batch_size=len(rows), device='cpu')
    for row in rows:
        sources = row.get('sources', {})
        for name in ('asset_metadata', 'trajectory_meta'):
            source = sources.get(name, {})
            fingerprint = source.get('sha256')
            if (not isinstance(source.get('path'), str) or not source['path']
                    or not isinstance(source.get('bytes'), int) or source['bytes'] <= 0
                    or not isinstance(fingerprint, str) or len(fingerprint) != 64
                    or any(c not in '0123456789abcdef' for c in fingerprint)):
                raise ValueError("Process manifest lacks raw metadata provenance")
    return {'process_manifest': str(path), 'process_manifest_sha256': expected_hash,
            'schema': PROCESS_CONDITION_SCHEMA, 'count': len(rows),
            'parameter_names': list(PROCESS_PARAMETER_NAMES)}


def load_process_conditioned_model(checkpoint, *, device='cpu'):
    """Load only an explicit new-objective checkpoint with bound process schema.

    This never treats an old bounded checkpoint as a conditioned model or guesses
    process settings. All new encoder keys must exist. The returned model still
    requires callers to provide process_params for each observation.
    """
    from .geometry_acceleration import accelerate_geometry
    from .program_training_v3 import enable_dit_bf16
    state = torch.load(Path(checkpoint), map_location='cpu', weights_only=True, mmap=True)
    metadata = state.get('metadata', {})
    objective = metadata.get('current_objective', {})
    factory_id = f'{make_process_conditioned_model.__module__}:{make_process_conditioned_model.__qualname__}'
    if (state.get('format') != FORMAT or not isinstance(objective, dict)
            or objective.get('strategy') != STRATEGY or objective.get('model_factory') != factory_id
            or metadata.get('objective_strategy') != STRATEGY):
        raise ValueError("An explicit process-conditioned CFM checkpoint/objective is required")
    migration = objective.get('migration')
    binding = _checked_process_binding(migration)
    model = make_process_conditioned_model(state['model_config'], metadata.get('arm'), migration)
    expected_state = model.state_dict()
    saved_state = state.get('model', {})
    if set(saved_state) != set(expected_state):
        raise ValueError("Checkpoint must contain every process-conditioned model key")
    for name, value in saved_state.items():
        if (not isinstance(value, Tensor) or value.shape != expected_state[name].shape
                or value.dtype != expected_state[name].dtype):
            raise ValueError(f"Conditioned checkpoint tensor shape/dtype mismatch: {name}")
    model.load_state_dict(saved_state, strict=True)
    model.to(device)
    accelerate_geometry(model)
    if metadata.get('dit_bf16'):
        enable_dit_bf16(model, device)
    model.loaded_process_binding = binding
    model.eval()
    return model, state
