"""Canonical direct decoding with an explicitly different ROI voxel frame.

All parameters, known controls, fixed zero/t0 queries, label definitions and
direct loss weights are inherited unchanged. Reusing an old state is exact;
the geometry forward function deliberately changes and is not equivalent to
the bounding-box-origin parent. Loading therefore requires a distinct strategy,
factory identity and a fixed geometry-frame declaration.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import Tensor, nn

from .bounded_scratch_program_model import ARMS
from .canonical_direct_program_model import (
    CanonicalDirectProgramModel, FORMAT, _checked_canonical_binding,
)
from .process_conditioned_flow_model import (
    STRATEGY as PROCESS_CFM_STRATEGY, _checked_process_binding,
)
from .process_augmented_canonical_data import (
    PROCESS_AUGMENTATION_SCHEMA, PROCESS_AUGMENTATION_SEED, VARIANT_DEPTH_MM,
)
from .process_conditioned_data import PROCESS_PARAMETER_NAMES, _positive
from .program_model import ProgramConfig
from .roi_lattice_geometry import (
    GEOMETRY_FRAME_POLICY, RoiLatticeSparseVoxelBackbone, install_roi_lattice_geometry,
)


STRATEGY = 'canonical_process_direct_t0_roi_lattice_v4'


def _migration(value):
    if value is None:
        return {'strategy': STRATEGY, 'geometry_frame': deepcopy(GEOMETRY_FRAME_POLICY)}
    if not isinstance(value, dict):
        raise ValueError('ROI-lattice objective_migration must be a dictionary or None')
    if value.get('strategy') != STRATEGY:
        raise ValueError('ROI-lattice migration must explicitly declare its new strategy')
    if value.get('geometry_frame') != GEOMETRY_FRAME_POLICY:
        raise ValueError('ROI-lattice migration must declare the exact fixed geometry_frame policy')
    return deepcopy(value)


class RoiLatticeDirectProgramModel(CanonicalDirectProgramModel):
    """Same learned state and direct objective; explicit ROI-lattice geometry."""

    def __init__(self, config=None, *, bounds=None, activation_checkpoint=True,
                 selector_expanded=False, objective_migration=None):
        migration = _migration(objective_migration)
        super().__init__(config, bounds=bounds, activation_checkpoint=activation_checkpoint,
                         selector_expanded=selector_expanded, objective_migration=migration)
        self.geometry_installation_audit = install_roi_lattice_geometry(self)

    def direct_supervision_metadata(self):
        metadata = super().direct_supervision_metadata()
        metadata.update(strategy=STRATEGY, geometry_frame=deepcopy(GEOMETRY_FRAME_POLICY),
                        parent_state_compatible=True, parent_forward_equivalent=False)
        return metadata

    def process_conditioning_metadata(self):
        metadata = super().process_conditioning_metadata()
        metadata.update(strategy=STRATEGY, geometry_frame=deepcopy(GEOMETRY_FRAME_POLICY),
                        direct_supervision=self.direct_supervision_metadata())
        return metadata


def make_roi_lattice_direct_program_model(model_config: dict, arm: str,
                                           objective_migration: dict | None = None) -> nn.Module:
    if arm not in ARMS:
        raise ValueError('Unknown ROI-lattice direct arm')
    config = ProgramConfig.from_dict(model_config)
    if (config.selector_layers, config.dit_layers) != ARMS[arm]:
        raise ValueError('Model configuration does not match ROI-lattice direct arm')
    return RoiLatticeDirectProgramModel(config, selector_expanded=arm == 'selector6',
                                       objective_migration=objective_migration)


def _digest(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError(f'Invalid ROI-lattice {name} SHA256 binding')
    return value


def _checked_augmentation_binding(migration, process_binding, canonical_binding):
    """Check optional train-label provenance, without reading any target into generation."""
    path, expected = migration.get('process_augmentation_manifest'), migration.get('process_augmentation_manifest_sha256')
    if path is None and expected is None:
        return None
    if not isinstance(path, str) or not path:
        raise ValueError('Process augmentation manifest path and SHA256 must be paired')
    _digest(expected, 'process augmentation manifest')
    path = Path(path).resolve(strict=True)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError('Process augmentation manifest SHA256 changed since training')
    frozen = json.loads(payload)
    if frozen.get('schema') != PROCESS_AUGMENTATION_SCHEMA:
        raise ValueError('Unsupported process augmentation schema')
    for key, required in (('process_manifest_sha256', process_binding['process_manifest_sha256']),
                          ('canonical_manifest_sha256', canonical_binding['canonical_manifest_sha256']),
                          ('canonical_definition_sha256', canonical_binding['canonical_definition_sha256'])):
        if frozen.get(key) != required:
            raise ValueError(f'Process augmentation {key} differs from the checkpoint')
    if isinstance(frozen.get('augmentation_seed'), bool) or frozen.get('augmentation_seed') != PROCESS_AUGMENTATION_SEED:
        raise ValueError('Process augmentation seed changed')
    eligible, variants = frozen.get('eligible_sample_ids'), frozen.get('variants')
    if (not isinstance(eligible, list) or any(not isinstance(s, str) or not s for s in eligible)
            or len(set(eligible)) != len(eligible) or not isinstance(variants, list)
            or len(variants) != len(eligible) * 2 or frozen.get('variant_count', len(variants)) != len(variants)):
        raise ValueError('Process augmentation sample/variant count is incomplete')
    process = json.loads(Path(process_binding['process_manifest']).read_text())
    canonical = json.loads(Path(canonical_binding['canonical_manifest']).read_text())
    pindex = {row['sample_id']: row for row in process['samples']}
    cindex = {row['sample_id']: row for row in canonical['samples']}
    seen = set()
    for sid in eligible:
        if (sid not in pindex or sid not in cindex
                or any(row.get('optimizer_split') != 'train' or row.get('cohort') != 'new'
                       for row in (pindex[sid], cindex[sid]))):
            raise ValueError('Process augmentation must contain only original new training samples')
    for row in variants:
        if not isinstance(row, dict):
            raise ValueError('Malformed process augmentation variant record')
        sid, variant = row.get('sample_id'), row.get('variant_id')
        key = (sid, variant)
        if sid not in eligible or variant not in VARIANT_DEPTH_MM or key in seen:
            raise ValueError('Duplicate or ineligible process augmentation variant')
        seen.add(key)
        if (row.get('original_optimizer_split') != 'train'
                or row.get('source_npz_sha256') != cindex[sid]['source_npz_sha256']):
            raise ValueError('Process augmentation source/split binding changed')
        values = row.get('process_params')
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError('Process augmentation requires four known control values')
        values = [_positive(v, name) for name, v in zip(PROCESS_PARAMETER_NAMES, values)]
        if (values[1] != VARIANT_DEPTH_MM[variant]
                or any(not math.isclose(values[i], pindex[sid]['process_params'][i], rel_tol=1e-12, abs_tol=1e-12)
                       for i in (0, 2, 3))):
            raise ValueError('Process augmentation changed an unrelated process control')
        proof = row.get('verification', {})
        if proof.get('passed') is not True or proof.get('material_replay_passed') is not True:
            raise ValueError('Process augmentation variant lacks successful replay verification')
        relative = row.get('target_npz')
        if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
                or '..' in Path(relative).parts):
            raise ValueError('Process augmentation target path must be portable')
        _digest(row.get('target_sha256'), 'augmentation target')
    if seen != {(sid, variant) for sid in eligible for variant in VARIANT_DEPTH_MM}:
        raise ValueError('Each eligible sample needs both frozen process variants')
    if 'verification' in frozen:
        proof = frozen['verification']
        if proof.get('passed') is not True or proof.get('checked_variants') != len(variants):
            raise ValueError('Process augmentation export verification is incomplete')
    return {'process_augmentation_manifest': str(path), 'process_augmentation_manifest_sha256': expected,
            'schema': PROCESS_AUGMENTATION_SCHEMA, 'eligible_sample_count': len(eligible), 'variant_count': len(variants)}


def load_roi_lattice_direct_program_model(checkpoint, *, device='cpu'):
    """Strict new-geometry loader; old direct loaders reject this strategy identity."""
    from .geometry_acceleration import accelerate_geometry
    from .program_training_v3 import enable_dit_bf16
    state = torch.load(Path(checkpoint), map_location='cpu', weights_only=True, mmap=True)
    metadata = state.get('metadata', {})
    objective = metadata.get('current_objective', {})
    factory_id = f'{make_roi_lattice_direct_program_model.__module__}:{make_roi_lattice_direct_program_model.__qualname__}'
    if (state.get('format') != FORMAT or not isinstance(objective, dict)
            or objective.get('strategy') != STRATEGY or metadata.get('objective_strategy') != STRATEGY
            or objective.get('model_factory') != factory_id):
        raise ValueError('An explicit ROI-lattice direct checkpoint/strategy/factory is required')
    migration = _migration(objective.get('migration'))
    if not isinstance(objective.get('migration'), dict):
        raise ValueError('ROI-lattice checkpoint needs an explicit migration identity')
    if 'geometry_frame' in metadata and metadata['geometry_frame'] != GEOMETRY_FRAME_POLICY:
        raise ValueError('Checkpoint geometry_frame differs from the fixed ROI-lattice policy')
    process_binding = _checked_process_binding({**migration, 'strategy': PROCESS_CFM_STRATEGY})
    canonical_binding = _checked_canonical_binding(migration, process_binding)
    augmentation_binding = _checked_augmentation_binding(migration, process_binding, canonical_binding)
    model = make_roi_lattice_direct_program_model(state['model_config'], metadata.get('arm'), migration)
    expected, saved = model.state_dict(), state.get('model', {})
    if set(saved) != set(expected):
        raise ValueError('All ROI-lattice direct state keys must be explicit')
    for name, value in saved.items():
        if not isinstance(value, Tensor) or value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f'ROI-lattice direct tensor shape/dtype mismatch: {name}')
    model.load_state_dict(saved, strict=True)
    model.to(device)
    accelerate_geometry(model)
    if metadata.get('dit_bf16'):
        enable_dit_bf16(model, device)
    model.loaded_process_binding = process_binding
    model.loaded_canonical_binding = canonical_binding
    model.loaded_augmentation_binding = augmentation_binding
    model.loaded_geometry_frame = deepcopy(GEOMETRY_FRAME_POLICY)
    model.eval()
    return model, state


def evaluate_roi_lattice_subset(model, data, indices, output, *, seed=20261109,
                                batch_size=8, check=None):
    """One unchanged strict evaluation, with explicit new input-frame provenance."""
    from .canonical_direct_evaluation import evaluate_canonical_direct_subset
    from .runtime import write_json
    if (not isinstance(model, RoiLatticeDirectProgramModel)
            or not isinstance(model.geometry.backbone, RoiLatticeSparseVoxelBackbone)
            or model.process_conditioning_metadata().get('geometry_frame') != GEOMETRY_FRAME_POLICY):
        raise ValueError('ROI-lattice evaluation requires the explicit new model and geometry policy')
    result = evaluate_canonical_direct_subset(model, data, indices, output,
        seed=seed, batch_size=batch_size, check=check)
    result.update(model_strategy=STRATEGY, geometry_frame=deepcopy(GEOMETRY_FRAME_POLICY),
                  input_encoding_changed=True, targets_and_thresholds_unchanged=True)
    write_json(Path(output) / 'summary.json', result)
    return result
