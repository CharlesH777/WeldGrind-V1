"""Validation and loading of cached canonical supervision."""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .process_conditioned_data import ProcessConditionedProgramDataset, _read_source_conditions, _identity_sha
from .program_data import _segment_modes, resample_complete_program
from .primitive_program_contract import validate_primitive_program

CANONICAL_SCHEMA = 'weldgrind-world-canonical-program-targets-v1'
TARGET_DEFINITION = 'world_axis_contact_preserving_canonical_v1'
CANONICAL_DEFINITION = {
    'name': TARGET_DEFINITION, 'version': 1,
    'contact_points': 'each source raw stroke exactly retained forward or reversed; no splitting merging or snapping',
    'pass_order': 'preserve contiguous source instance pass snapshot and cut-depth groups',
    'stroke_axis': 'PCA of contact endpoint displacement outer products projected into physical nominal tangent plane',
    'lane_order': 'world lane centroid then world tangent centroid within each source pass',
    'world_sign': 'largest absolute component positive; world XYZ priority within 1e-8 tie',
    'serpentine': 'first contact curve in each pass follows positive canonical tangent then alternates',
    'physical_normal': 'source nominal geometry normal used offline only; source chart U and V signs do not determine ordering',
    'air': 'retract transfer approach on original planar safe-height or original analytic offset shell using source planner helpers',
    'source_scope': 'no initial approach or final retract added',
    'trajectory_points': 1024, 'maximum_representation_error_mm': .02,
    'cross_chart_float_tolerance_mm': 1e-4,
}


def definition_sha256(definition=CANONICAL_DEFINITION):
    return hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()


def _sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): value.update(block)
    return value.hexdigest()














def _target_path(root, relative):
    path=Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Canonical target paths must be relative to their immutable manifest')
    resolved=(root/path).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError('Missing or escaping canonical target file')
    return resolved


def validate_canonical_manifest(path, *, process_manifest_sha256=None):
    """Validate the frozen target definition and complete per-sample certificates."""
    path=Path(path).resolve()
    data=json.loads(path.read_text())
    if data.get('schema')!=CANONICAL_SCHEMA or data.get('target_definition')!=TARGET_DEFINITION:
        raise ValueError('Unsupported canonical target manifest schema/definition')
    if data.get('definition')!=CANONICAL_DEFINITION or data.get('definition_sha256')!=definition_sha256(data.get('definition')):
        raise ValueError('Canonical target definition changed')
    if process_manifest_sha256 is not None and data.get('process_manifest_sha256')!=process_manifest_sha256:
        raise ValueError('Canonical targets bind a different process manifest')
    rows=data.get('samples')
    if not isinstance(rows,list) or not rows or data.get('count')!=len(rows):
        raise ValueError('Canonical target sample count is incomplete')
    seen=set()
    for row in rows:
        sid=row.get('sample_id')
        if not isinstance(sid,str) or not sid or sid in seen:
            raise ValueError('Canonical sample IDs must be unique')
        seen.add(sid)
        for field in ('target_sha256','source_npz_sha256'):
            digest=row.get(field)
            if not isinstance(digest,str) or len(digest)!=64 or any(x not in '0123456789abcdef' for x in digest):
                raise ValueError(f'Invalid canonical {field}')
        relative=Path(row.get('target_npz',''))
        if not row.get('target_npz') or relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Canonical target path must be portable')
        proof=row.get('certificate',{})
        for field in ('all_contact_world_point_multisets_exact','all_pass_snapshot_depth_annotations_exact',
                      'pass_causal_order_preserved','no_initial_approach_or_final_retract_added','air_direction_geometry_verified'):
            if proof.get(field) is not True:
                raise ValueError('Incomplete canonical contact/pass/air certificate')
    verification=data.get('verification',{})
    if verification.get('passed') is not True or verification.get('checked_samples')!=len(rows):
        raise ValueError('Canonical targets have not passed complete export verification')
    return data




class CanonicalProcessProgramDataset(ProcessConditionedProgramDataset):
    """Cached canonical labels plus the original fixed-length labels for dual scoring."""
    def __init__(self,manifest,max_points=2048,trajectory_points=1024,seed=20261109,max_error_mm=.02,
                 *,process_manifest,canonical_manifest,source_root_overrides=None):
        if trajectory_points!=1024 or max_error_mm!=.02:
            raise ValueError('Canonical target definition requires 1024 points and the original .02mm error budget')
        super().__init__(manifest,max_points=max_points,trajectory_points=trajectory_points,seed=seed,
                         max_error_mm=max_error_mm,process_manifest=process_manifest,
                         source_root_overrides=source_root_overrides)
        self.canonical_manifest=Path(canonical_manifest).resolve()
        self.canonical_manifest_sha256=_sha256(self.canonical_manifest)
        frozen=validate_canonical_manifest(self.canonical_manifest,process_manifest_sha256=self.process_manifest_sha256)
        self.canonical_definition_sha256=frozen['definition_sha256']
        self.target_definition=TARGET_DEFINITION
        self._canonical_records={row['sample_id']:row for row in frozen['samples']}
        splits={row.get('optimizer_split') for row in self.rows}
        if len(splits)!=1:raise ValueError('One canonical data manifest must have one optimizer split')
        binding=frozen['manifests'].get(next(iter(splits)),{})
        if binding.get('count')!=len(self.rows) or binding.get('ordered_row_identity_sha256')!=_identity_sha(self.rows):
            raise ValueError('Canonical target sample identities/order do not match the data manifest')
        for row in self.rows:
            record=self._canonical_records.get(row['sample_id'])
            if record is None or any(record.get(key)!=row.get(key) for key in ('sample_id','cohort','optimizer_split')):
                raise ValueError('Canonical target identity or optimizer split changed')
            if record['sources']!=self._process_records[row['sample_id']]['sources']:
                raise ValueError('Canonical target metadata fingerprints do not match the process manifest')

    def __getitem__(self,index):
        row=self.rows[index];record=self._canonical_records[row['sample_id']]
        source=Path(row['npz'])
        if not source.is_absolute():source=self.manifest.parent/source
        if _sha256(source)!=record['source_npz_sha256']:
            raise ValueError('Original source NPZ SHA256 differs from the canonical target binding')
        target_file=_target_path(self.canonical_manifest.parent,record['target_npz'])
        if _sha256(target_file)!=record['target_sha256']:
            raise ValueError('Canonical target NPZ SHA256 mismatch')
        sample=super().__getitem__(index)
        sample['source_trajectory']=sample['trajectory']
        sample['source_phase_labels']=sample['phase_labels']
        sample['source_segment_ids']=sample['segment_ids']
        with np.load(target_file,allow_pickle=False) as arrays:
            if (str(arrays['target_definition'].item())!=TARGET_DEFINITION
                    or str(arrays['canonical_definition_sha256'].item())!=self.canonical_definition_sha256
                    or str(arrays['source_npz_sha256'].item())!=record['source_npz_sha256']
                    or str(arrays['units'].item())!='mm'):
                raise ValueError('Canonical target NPZ definition/source binding changed')
            trajectory=arrays['trajectory'].copy()
            if trajectory.shape!=(1024,3) or not np.isfinite(trajectory).all():
                raise ValueError('Canonical target must be a finite full 1024-point world-mm program')
            sample['trajectory']=(torch.from_numpy(trajectory)-sample['center'])/sample['scale']
            for name in arrays.files:
                if name.startswith('trajectory_'):
                    value=arrays[name].copy()
                    if value.shape!=(1024,):raise ValueError('Canonical label shape must be fixed 1024')
                    sample[name]=torch.from_numpy(value)
            sample['segment_count']=torch.tensor(len(arrays['resampled_segment_ranges']),dtype=torch.long)
            sample['source_point_count']=torch.tensor(int(arrays['source_point_count']),dtype=torch.long)
        labels=sample['trajectory_mode']
        if (labels.dtype!=torch.long or ((labels<0)|(labels>3)).any()
                or not torch.equal(sample['trajectory_contact'],labels==0)
                or not sample['trajectory_valid'].all() or sample['trajectory_stop'].sum()!=1
                or not sample['trajectory_stop'][-1]):
            raise ValueError('Canonical phase/contact/valid/stop labels are malformed')
        sample['phase_labels']=labels;sample['segment_ids']=sample['trajectory_segment']
        sample['target_definition']=TARGET_DEFINITION
        sample['canonical_target_sha']=record['target_sha256']
        sample['canonical_definition_sha256']=self.canonical_definition_sha256
        return sample




