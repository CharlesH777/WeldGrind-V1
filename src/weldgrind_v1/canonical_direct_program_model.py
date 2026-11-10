"""Deterministic one-forward neural decoding of a canonical ordered program.

This reuses the process-conditioned bounded architecture/state, not the CFM or
energy objective. Every query is zero and every decoder time is zero. The output
head directly represents the normalized trajectory; phases come from that same
forward. Canonical GT paths, segments and phase labels are training targets only.
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

from .bounded_scratch_program_model import ARMS
from .program_model import ProgramConfig
from .process_conditioned_flow_model import (
    ProcessConditionedFlowModel, FORMAT, STRATEGY as PROCESS_CFM_STRATEGY, _checked_process_binding,
)


STRATEGY = 'canonical_process_deterministic_t0_program_v3'
CANONICAL_SCHEMA = 'weldgrind-world-canonical-program-targets-v1'
CANONICAL_DEFINITION_NAME = 'world_axis_contact_preserving_canonical_v1'
POSITION_REFERENCE_MM = 2.
DISPLACEMENT_HORIZONS = (1, 4, 16, 64)
D2_POSITION_REFERENCE_MM = .25
DEFAULT_DIRECT_LOSS_WEIGHTS = {
    'position': 1., 'displacement': .5, 'd2_error': .1, 'segment_endpoint': .5,
    'phase': 1., 'coverage': .05, 'diversity': .01,
}


def _fixed_weights(overrides=None):
    result = dict(DEFAULT_DIRECT_LOSS_WEIGHTS)
    if overrides is not None:
        if not isinstance(overrides, Mapping) or set(overrides)-set(result):
            raise ValueError('Unknown canonical-direct loss weights')
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value != result[name] for name, value in overrides.items()):
            raise ValueError('Keep the explicit fixed canonical-direct supervision weights')
        result.update(overrides)
    return result


def segment_endpoint_mask(segment_ids: Tensor, *, shape) -> Tensor:
    if (not isinstance(segment_ids, Tensor) or segment_ids.shape != shape
            or segment_ids.dtype != torch.long or (segment_ids < 0).any()):
        raise ValueError('Direct supervision needs fully labelled int64 segment_ids [B,T]')
    starts = torch.ones_like(segment_ids, dtype=torch.bool)
    ends = torch.ones_like(starts)
    starts[:, 1:] = segment_ids[:, 1:] != segment_ids[:, :-1]
    ends[:, :-1] = segment_ids[:, :-1] != segment_ids[:, 1:]
    return starts | ends


def direct_program_supervision(prediction: Tensor, target: Tensor, *, scale_mm: Tensor,
                               segment_ids: Tensor) -> dict[str, Tensor]:
    """Physical residual losses; genuine GT curvature incurs exactly zero loss.

    A difference of two independent endpoint errors has variance factor2. A
    second difference with coefficients (1,-2,1) has factor6. These fixed unit
    references do not normalize by target variance, length or phase frequency.
    """
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError('Direct trajectories must be matching [B,T,3]')
    batch, length, _ = prediction.shape
    if length <= max(DISPLACEMENT_HORIZONS):
        raise ValueError('Canonical direct supervision needs T>64 for its fixed horizons')
    endpoint = segment_endpoint_mask(segment_ids, shape=prediction.shape[:2])
    scale = torch.as_tensor(scale_mm, dtype=torch.float32, device=prediction.device)
    if scale.ndim == 0 or scale.shape == (1,):
        scale = scale.reshape(1).expand(batch)
    if scale.shape != (batch,) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Direct scale must be finite positive observed mm [B]')
    error_mm = (prediction.float()-target.float())*scale[:, None, None]
    residual = error_mm/POSITION_REFERENCE_MM
    displacement = torch.stack([((error_mm[:, h:]-error_mm[:, :-h])
        /(math.sqrt(2)*POSITION_REFERENCE_MM)).square().mean() for h in DISPLACEMENT_HORIZONS]).mean()
    d2 = error_mm[:, 2:]-2*error_mm[:, 1:-1]+error_mm[:, :-2]
    return {'position': residual.square().mean(), 'displacement': displacement,
            'd2_error': (d2/(math.sqrt(6)*D2_POSITION_REFERENCE_MM)).square().mean(),
            'segment_endpoint': (residual.square().mean(-1)*endpoint).sum()/endpoint.sum().clamp_min(1)}


class CanonicalDirectProgramModel(ProcessConditionedFlowModel):
    """Same parameter/state keys, explicit fixed-zero/t0 direct program semantics."""

    def __init__(self, config=None, *, bounds=None, activation_checkpoint=True,
                 selector_expanded=False, objective_migration=None):
        if objective_migration is not None and not isinstance(objective_migration, dict):
            raise ValueError('objective_migration must be a dictionary or None')
        migration = deepcopy(objective_migration or {})
        direct_weights = _fixed_weights(migration.get('loss_weights'))
        base_migration = deepcopy(migration)
        # The architecture parent validates CFM keys; direct weights belong only
        # to this new objective and must never be fed to its CFM validator.
        base_migration.pop('loss_weights', None)
        super().__init__(config, bounds=bounds, activation_checkpoint=activation_checkpoint,
                         selector_expanded=selector_expanded, objective_migration=base_migration)
        if self.config.trajectory_points <= max(DISPLACEMENT_HORIZONS):
            raise ValueError('Canonical direct supervision requires all fixed horizons, T>64')
        self.objective_migration = migration
        self.direct_loss_weights = direct_weights
        self.loss_weights = dict(direct_weights)

    def direct_supervision_metadata(self):
        return {'strategy': STRATEGY, 'objective_kind': 'canonical direct neural regression, not conditional FM',
                'query': 'fixed all-zero [B,T,3]', 'decoder_time': 0., 'decoder_forwards': 1,
                'trajectory': 'same t0 output head; x0=0 so trajectory=legacy velocity head',
                'phase_decoder_time': 0., 'generation_method': 'direct_t0', 'generation_steps': 1,
                'stochastic_latent': False, 'nonzero_initial_noise_policy': 'explicit rejection',
                'position_reference_mm': POSITION_REFERENCE_MM,
                'displacement_horizons': list(DISPLACEMENT_HORIZONS),
                'displacement_reference_mm': math.sqrt(2)*POSITION_REFERENCE_MM,
                'd2_position_reference_mm': D2_POSITION_REFERENCE_MM,
                'd2_error_reference_mm': math.sqrt(6)*D2_POSITION_REFERENCE_MM,
                'd2_scope': 'second difference of physical prediction-minus-GT error, not absolute curvature',
                'segment_endpoint_scope': 'first/last tokens in each annotated GT segment run; supervision only',
                'loss_weights': dict(self.direct_loss_weights)}

    def process_conditioning_metadata(self):
        metadata = super().process_conditioning_metadata()
        metadata.update(strategy=STRATEGY, loss_weights=dict(self.direct_loss_weights),
                        direct_supervision=self.direct_supervision_metadata())
        return metadata

    def forward(self, points, roi_mask, point_valid=None, *, scale_mm, process_params):
        condition = self.encode(points, roi_mask, point_valid, scale_mm=scale_mm, process_params=process_params)
        queries = points.new_zeros((points.shape[0], self.config.trajectory_points, 3))
        time = points.new_zeros(points.shape[0])
        output = self.dit(queries, time, condition['seed_xyz'], condition['seed_features'])
        return {**condition, 'trajectory': output['velocity'], 'phase_logits': output['phase_logits'],
                'phases': output['phase_logits'].argmax(-1)}

    def _validate_direct_query(self, points, steps, method, initial_noise):
        if not isinstance(steps, int) or isinstance(steps, bool) or steps != 1 or method != 'direct_t0':
            raise ValueError('Canonical direct decoding requires steps=1 and method=direct_t0')
        if initial_noise is not None:
            if (not isinstance(initial_noise, Tensor) or not initial_noise.is_floating_point()
                    or initial_noise.shape != (points.shape[0], self.config.trajectory_points, 3)
                    or not torch.isfinite(initial_noise).all() or torch.count_nonzero(initial_noise).item()):
                raise ValueError('Direct initial_noise, if supplied, must be finite correctly-shaped all-zero queries')

    def differentiable_rollout(self, points, roi_mask, point_valid=None, *, scale_mm, process_params,
                               steps=1, method='direct_t0', initial_noise=None, generator=None):
        """Compatibility name for the same deterministic one-forward decoder.

        generator is accepted for callers sharing an interface, but no random
        operation consumes it. An explicit nonzero latent is rejected.
        """
        self._validate_direct_query(points, steps, method, initial_noise)
        return self(points, roi_mask, point_valid, scale_mm=scale_mm, process_params=process_params)

    @torch.no_grad()
    def sample(self, points, roi_mask, point_valid=None, steps=1, method='direct_t0',
               generator=None, initial_noise=None, *, scale_mm, process_params):
        if self.training:
            raise RuntimeError('Call eval before canonical direct decoding')
        return self.differentiable_rollout(points, roi_mask, point_valid, scale_mm=scale_mm,
            process_params=process_params, steps=steps, method=method, initial_noise=initial_noise, generator=generator)

    def flow_loss(self, points, roi_mask, trajectory, point_valid=None, *, scale_mm,
                  process_params, phase_labels, segment_ids=None, generator=None, weights=None):
        """Runner-compatible name for direct supervision, with no flow/noise loss."""
        settings = _fixed_weights(weights)
        if (trajectory.shape != (points.shape[0], self.config.trajectory_points, 3)
                or not trajectory.is_floating_point() or not torch.isfinite(trajectory).all()):
            raise ValueError('Canonical target must be finite floating point [B,T,3]')
        if (phase_labels.shape != trajectory.shape[:2] or phase_labels.dtype != torch.long
                or ((phase_labels < 0) | (phase_labels >= self.config.phase_classes)).any()):
            raise ValueError('Canonical phases must be int64[B,T] using classes0..3')
        segment_endpoint_mask(segment_ids, shape=trajectory.shape[:2])
        output = self(points, roi_mask, point_valid, scale_mm=scale_mm, process_params=process_params)
        losses = direct_program_supervision(output['trajectory'], trajectory,
            scale_mm=output['observed_scale_mm'], segment_ids=segment_ids)
        losses['phase'] = self._phase_loss(output['phase_logits'], phase_labels)
        valid = output['candidate_valid']
        distances = torch.cdist(output['candidate_xyz'], output['seed_xyz']).square().amin(-1)
        soft = output['soft_assignments']
        overlap = soft@soft.transpose(1, 2)
        off = ~torch.eye(self.config.seeds, device=points.device, dtype=torch.bool)
        losses['coverage'] = (distances*valid).sum()/valid.sum().clamp_min(1)
        losses['diversity'] = overlap[:, off].mean() if self.config.seeds > 1 else overlap.sum()*0.
        losses['loss'] = sum(settings[name]*value for name, value in losses.items())
        return losses


def make_canonical_direct_program_model(model_config: dict, arm: str,
                                         objective_migration: dict | None = None) -> nn.Module:
    if arm not in ARMS:
        raise ValueError('Unknown canonical direct arm')
    config = ProgramConfig.from_dict(model_config)
    if (config.selector_layers, config.dit_layers) != ARMS[arm]:
        raise ValueError('Model configuration does not match canonical direct arm')
    return CanonicalDirectProgramModel(config, selector_expanded=arm == 'selector6',
                                       objective_migration=objective_migration)


def _checked_canonical_binding(migration, process_binding):
    path = migration.get('canonical_manifest')
    expected = migration.get('canonical_manifest_sha256')
    definition_expected = migration.get('canonical_definition_sha256')
    for value in (expected, definition_expected):
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Canonical manifest and definition SHA256 must be explicit')
    if not isinstance(path, str) or not path:
        raise ValueError('An explicit canonical target manifest is required')
    path = Path(path).resolve(strict=True)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError('Canonical target manifest changed since training')
    manifest = json.loads(payload)
    definition = manifest.get('definition')
    if (manifest.get('schema') != CANONICAL_SCHEMA or manifest.get('target_definition') != CANONICAL_DEFINITION_NAME
            or not isinstance(definition, dict) or definition.get('name') != CANONICAL_DEFINITION_NAME):
        raise ValueError('Unsupported canonical target schema or definition')
    definition_hash = hashlib.sha256(json.dumps(definition, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()
    if definition_hash != definition_expected or manifest.get('definition_sha256') != definition_expected:
        raise ValueError('Canonical definition identity differs from the checkpoint')
    rows = manifest.get('samples')
    if not isinstance(rows, list) or not rows or manifest.get('count') != len(rows):
        raise ValueError('Canonical manifest count/samples disagree')
    verification = manifest.get('verification', {})
    if verification.get('passed') is not True or verification.get('checked_samples') != len(rows):
        raise ValueError('Canonical targets lack complete successful verification')
    ids = [row.get('sample_id') for row in rows]
    if len(set(ids)) != len(ids) or any(not isinstance(sid, str) or not sid for sid in ids):
        raise ValueError('Canonical target IDs are missing or duplicated')
    process = json.loads(Path(process_binding['process_manifest']).read_text())
    if set(ids) != {row['sample_id'] for row in process['samples']}:
        raise ValueError('Canonical targets and process conditions do not bind the same sample set')
    for row in rows:
        if not isinstance(row.get('target_npz'), str) or not row['target_npz']:
            raise ValueError('Canonical target path is missing')
        for key in ('target_sha256', 'source_npz_sha256'):
            value = row.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Canonical target/source provenance is missing')
    return {'canonical_manifest': str(path), 'canonical_manifest_sha256': expected,
            'canonical_definition_sha256': definition_hash, 'schema': CANONICAL_SCHEMA,
            'target_definition': CANONICAL_DEFINITION_NAME, 'count': len(rows)}


def load_canonical_direct_program_model(checkpoint, *, device='cpu'):
    from .geometry_acceleration import accelerate_geometry
    from .program_training_v3 import enable_dit_bf16
    state = torch.load(Path(checkpoint), map_location='cpu', weights_only=True, mmap=True)
    metadata = state.get('metadata', {})
    objective = metadata.get('current_objective', {})
    factory_id = f'{make_canonical_direct_program_model.__module__}:{make_canonical_direct_program_model.__qualname__}'
    if (state.get('format') != FORMAT or not isinstance(objective, dict)
            or objective.get('strategy') != STRATEGY or metadata.get('objective_strategy') != STRATEGY
            or objective.get('model_factory') != factory_id):
        raise ValueError('An explicit canonical-direct checkpoint is required')
    migration = objective.get('migration')
    if not isinstance(migration, dict) or migration.get('strategy') != STRATEGY:
        raise ValueError('Canonical-direct migration identity is required')
    process_binding = _checked_process_binding({**migration, 'strategy': PROCESS_CFM_STRATEGY})
    canonical_binding = _checked_canonical_binding(migration, process_binding)
    model = make_canonical_direct_program_model(state['model_config'], metadata.get('arm'), migration)
    expected, saved = model.state_dict(), state.get('model', {})
    if set(saved) != set(expected):
        raise ValueError('All canonical-direct state keys must be explicit')
    for name, value in saved.items():
        if not isinstance(value, Tensor) or value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f'Canonical-direct tensor shape/dtype mismatch: {name}')
    model.load_state_dict(saved, strict=True)
    model.to(device)
    accelerate_geometry(model)
    if metadata.get('dit_bf16'):
        enable_dit_bf16(model, device)
    model.loaded_process_binding = process_binding
    model.loaded_canonical_binding = canonical_binding
    model.eval()
    return model, state
