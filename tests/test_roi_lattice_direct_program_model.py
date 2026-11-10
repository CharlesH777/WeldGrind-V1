"""A changed observation frame requires exact state migration and explicit identity."""
from copy import deepcopy
import hashlib
import inspect
import json

import pytest
import torch

from weldgrind_v1.bounded_scratch_program_model import initialized_arm
from weldgrind_v1.canonical_direct_program_model import (
    CANONICAL_SCHEMA, CANONICAL_DEFINITION_NAME, DEFAULT_DIRECT_LOSS_WEIGHTS,
    make_canonical_direct_program_model, load_canonical_direct_program_model,
)
from weldgrind_v1.geometry import GeometryConfig
from weldgrind_v1.process_conditioned_flow_model import (
    PROCESS_CONDITION_SCHEMA, PROCESS_PARAMETER_NAMES, PROCESS_PARAMETER_UNITS,
)
from weldgrind_v1.process_augmented_canonical_data import PROCESS_AUGMENTATION_SCHEMA
from weldgrind_v1.program_model import ProgramConfig
from weldgrind_v1.roi_lattice_direct_program_model import (
    FORMAT, STRATEGY, RoiLatticeDirectProgramModel, make_roi_lattice_direct_program_model,
    load_roi_lattice_direct_program_model, evaluate_roi_lattice_subset,
)
from weldgrind_v1.roi_lattice_geometry import GEOMETRY_FRAME_POLICY, RoiLatticeSparseVoxelBackbone


def cfg():
    return ProgramConfig(geometry=GeometryConfig(dim=24, candidates=12, channels=(4, 6, 8, 10),
        neighbors=4, branch_dim=6, voxel_size=.13), seeds=4, trajectory_points=72,
        heads=4, selector_layers=2, dit_layers=6, mlp_ratio=2)


def migration(**extra):
    return {'strategy': STRATEGY, 'geometry_frame': deepcopy(GEOMETRY_FRAME_POLICY), **extra}


def models(arm='base'):
    original = initialized_arm(arm, config=cfg(), seed=53, activation_checkpoint=True)
    parent = make_canonical_direct_program_model(original.config.to_dict(), arm)
    # Migration includes the four trained process parameters, not just bounded state.
    parent.load_bounded_state_dict(original.state_dict())
    with torch.no_grad():
        parent.process_encoder[-1].weight.copy_(torch.randn(parent.process_encoder[-1].weight.shape,
            generator=torch.Generator().manual_seed(173)) * .01)
    new = make_roi_lattice_direct_program_model(parent.config.to_dict(), arm, migration())
    new.load_state_dict(parent.state_dict(), strict=True)
    return parent, new


def inputs():
    gen = torch.Generator().manual_seed(41)
    points = torch.rand(2, 18, 6, generator=gen)
    points[:, :, :3] -= .45
    roi = torch.zeros(2, 18, dtype=torch.bool); roi[:, :12] = True
    valid = torch.ones_like(roi)
    target = torch.randn(2, 72, 3, generator=gen) * .3
    phase = (torch.arange(72) // 18).remainder(4)[None].repeat(2, 1)
    process = torch.tensor([[3., .25, 1.5, .2], [5., .5, 2., .4]])
    return points, roi, valid, target, phase, process


@pytest.mark.parametrize('arm', ['base', 'dit12', 'selector6'])
def test_every_state_key_value_shape_dtype_and_parameter_count_survives_migration(arm):
    old, new = models(arm)
    assert set(old.state_dict()) == set(new.state_dict())
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())
    assert set(dict(old.named_parameters())) == set(dict(new.named_parameters()))
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key])
        assert value.shape == new.state_dict()[key].shape and value.dtype == new.state_dict()[key].dtype
    assert isinstance(new.geometry.backbone, RoiLatticeSparseVoxelBackbone)
    assert new.direct_loss_weights == old.direct_loss_weights == DEFAULT_DIRECT_LOSS_WEIGHTS
    assert new.geometry_installation_audit['parameter_objects_preserved']


def test_forward_is_an_explicit_semantic_change_despite_exact_state():
    old, new = models(); old.eval(); new.eval()
    points, roi, valid, _, _, process = inputs()
    kwargs = dict(scale_mm=torch.tensor([4., 9.]), process_params=process)
    with torch.no_grad():
        before, after = old.sample(points, roi, valid, **kwargs), new.sample(points, roi, valid, **kwargs)
    assert torch.equal(before['candidate_xyz'], after['candidate_xyz'])
    assert not torch.equal(before['candidate_features'], after['candidate_features'])
    assert not torch.equal(before['trajectory'], after['trajectory'])
    meta = new.direct_supervision_metadata()
    assert meta['parent_state_compatible'] and not meta['parent_forward_equivalent']


def test_zero_t0_single_forward_and_phase_head_consistency_without_gt_or_rng():
    _, model = models(); model.eval()
    points, roi, valid, _, _, process = inputs()
    calls = []
    original = model.dit.forward
    def observed(x, t, *a, **kw):
        calls.append((x.detach().clone(), t.detach().clone()))
        return original(x, t, *a, **kw)
    model.dit.forward = observed
    generator = torch.Generator().manual_seed(71); rng = generator.get_state().clone()
    kwargs = dict(scale_mm=torch.tensor([4., 9.]), process_params=process)
    a = model.sample(points, roi, valid, generator=generator, **kwargs)
    b = model.sample(points, roi, valid, initial_noise=torch.zeros(2, 72, 3), **kwargs)
    assert len(calls) == 2 and all(torch.count_nonzero(x) == torch.count_nonzero(t) == 0 for x, t in calls)
    assert torch.equal(a['trajectory'], b['trajectory']) and torch.equal(a['phase_logits'], b['phase_logits'])
    assert torch.equal(a['phases'], a['phase_logits'].argmax(-1))
    assert torch.equal(rng, generator.get_state())
    for method in (RoiLatticeDirectProgramModel.forward, RoiLatticeDirectProgramModel.sample,
                   RoiLatticeDirectProgramModel.differentiable_rollout):
        params = inspect.signature(method).parameters
        assert params['process_params'].default is inspect.Parameter.empty
        assert not set(params) & {'trajectory', 'target', 'phase_labels', 'segment_ids', 'chart', 'pass_index'}
    with pytest.raises(TypeError, match='process_params'):
        model.sample(points, roi, valid, scale_mm=torch.tensor([4., 9.]))
    with pytest.raises(ValueError, match='all-zero'):
        model.sample(points, roi, valid, initial_noise=torch.ones(2, 72, 3), **kwargs)


@pytest.mark.parametrize('arm', ['base', 'dit12', 'selector6'])
def test_direct_losses_same_t0_and_finite_gradients_reach_all_parts(arm):
    _, model = models(arm)
    points, roi, valid, target, phase, process = inputs()
    calls = []
    original = model.dit.forward
    def observed(x, t, *a, **kw):
        calls.append((x.detach().clone(), t.detach().clone()))
        return original(x, t, *a, **kw)
    model.dit.forward = observed
    loss = model.flow_loss(points, roi, target, valid, scale_mm=torch.tensor([4., 9.]),
        process_params=process, phase_labels=phase, segment_ids=phase)
    assert len(calls) == 1 and torch.count_nonzero(calls[0][0]) == torch.count_nonzero(calls[0][1]) == 0
    assert set(loss) == set(DEFAULT_DIRECT_LOSS_WEIGHTS) | {'loss'}
    assert all(torch.isfinite(v) for v in loss.values())
    loss['loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for parameter in (model.geometry.backbone.convolutions[0].weight, model.geometry.fusion[0].weight,
                      model.selector.queries, model.dit.output.weight, model.dit.phase_head.weight,
                      *model.process_encoder.parameters()):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_metadata_declares_frame_strategy_and_unchanged_direct_protocol():
    _, model = models()
    metadata = model.process_conditioning_metadata()
    direct = metadata['direct_supervision']
    assert metadata['strategy'] == direct['strategy'] == STRATEGY
    assert metadata['geometry_frame'] == direct['geometry_frame'] == GEOMETRY_FRAME_POLICY
    assert direct['decoder_forwards'] == 1 and direct['decoder_time'] == direct['phase_decoder_time'] == 0
    assert direct['query'] == 'fixed all-zero [B,T,3]'
    assert metadata['loss_weights'] == direct['loss_weights'] == DEFAULT_DIRECT_LOSS_WEIGHTS
    metadata['geometry_frame']['name'] = 'mutation'
    assert model.process_conditioning_metadata()['geometry_frame'] == GEOMETRY_FRAME_POLICY


@pytest.mark.parametrize('bad', [{'strategy': 'old'}, {'strategy': STRATEGY},
    {'strategy': STRATEGY, 'geometry_frame': {'name': 'other'}}, False])
def test_factory_refuses_ambiguous_or_old_frame_identity(bad):
    with pytest.raises(ValueError):
        make_roi_lattice_direct_program_model(cfg().to_dict(), 'base', bad)


def write(path, value):
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, augmented=True):
    _, model = models()
    source = tmp_path / 'source.json'; source.write_text('{}')
    fingerprint = {'path': str(source), 'bytes': 2, 'sha256': sha(source)}
    process = {'schema': PROCESS_CONDITION_SCHEMA, 'parameter_names': list(PROCESS_PARAMETER_NAMES),
        'parameter_units': list(PROCESS_PARAMETER_UNITS), 'count': 1,
        'samples': [{'sample_id': 'test', 'cohort': 'new', 'optimizer_split': 'train',
                     'process_params': [.7, .5, 4., .65],
                     'sources': {'asset_metadata': fingerprint, 'trajectory_meta': fingerprint}}]}
    process_path = tmp_path / 'process.json'; write(process_path, process)
    definition = {'name': CANONICAL_DEFINITION_NAME, 'version': 1}
    definition_sha = hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    canonical = {'schema': CANONICAL_SCHEMA, 'target_definition': CANONICAL_DEFINITION_NAME,
        'definition': definition, 'definition_sha256': definition_sha, 'count': 1,
        'verification': {'passed': True, 'checked_samples': 1},
        'samples': [{'sample_id': 'test', 'cohort': 'new', 'optimizer_split': 'train',
                     'target_npz': 'test.npz', 'target_sha256': 'a' * 64, 'source_npz_sha256': 'b' * 64}]}
    canonical_path = tmp_path / 'canonical.json'; write(canonical_path, canonical)
    migrate = migration(process_manifest=str(process_path), process_manifest_sha256=sha(process_path),
        canonical_manifest=str(canonical_path), canonical_manifest_sha256=sha(canonical_path),
        canonical_definition_sha256=definition_sha)
    augmentation_path = tmp_path / 'augmentation.json'
    if augmented:
        variants = [{'sample_id': 'test', 'original_optimizer_split': 'train', 'variant_id': name,
            'source_npz_sha256': 'b' * 64, 'process_params': [.7, depth, 4., .65],
            'target_npz': name + '.npz', 'target_sha256': 'c' * 64,
            'verification': {'passed': True, 'material_replay_passed': True}}
            for name, depth in (('depth_0p375', .375), ('depth_0p25', .25))]
        augmentation = {'schema': PROCESS_AUGMENTATION_SCHEMA,
            'process_manifest_sha256': sha(process_path), 'canonical_manifest_sha256': sha(canonical_path),
            'canonical_definition_sha256': definition_sha, 'augmentation_seed': 20261110,
            'eligible_sample_ids': ['test'], 'variants': variants, 'variant_count': 2,
            'verification': {'passed': True, 'checked_variants': 2}}
        write(augmentation_path, augmentation)
        migrate.update(process_augmentation_manifest=str(augmentation_path),
                       process_augmentation_manifest_sha256=sha(augmentation_path))
    objective = {'strategy': STRATEGY,
        'model_factory': 'weldgrind_v1.roi_lattice_direct_program_model:make_roi_lattice_direct_program_model',
        'migration': migrate}
    state = {'format': FORMAT, 'model_config': model.config.to_dict(), 'model': model.state_dict(),
        'metadata': {'arm': 'base', 'objective_strategy': STRATEGY, 'current_objective': objective,
                     'geometry_frame': deepcopy(GEOMETRY_FRAME_POLICY)}}
    return model, state, process_path, canonical_path, augmentation_path


@pytest.mark.parametrize('augmented', [False, True])
def test_loader_requires_new_identity_and_preserves_exact_full_state(tmp_path, augmented):
    model, state, _, _, _ = fixture(tmp_path, augmented=augmented)
    path = tmp_path / 'model.pt'; torch.save(state, path)
    loaded, _ = load_roi_lattice_direct_program_model(path)
    assert loaded.loaded_geometry_frame == GEOMETRY_FRAME_POLICY
    assert (loaded.loaded_augmentation_binding is not None) == augmented
    assert loaded.loaded_canonical_binding['target_definition'] == CANONICAL_DEFINITION_NAME
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)
    with pytest.raises(ValueError):
        load_canonical_direct_program_model(path)


@pytest.mark.parametrize('bad', ['factory', 'strategy', 'migration_strategy', 'missing_frame', 'frame', 'metadata_frame',
    'process_sha', 'canonical_sha', 'definition', 'aug_pair', 'aug_sha', 'aug_schema', 'aug_binding', 'aug_split',
    'aug_source', 'aug_control', 'aug_duplicate', 'aug_proof', 'aug_incomplete', 'state_missing', 'state_dtype', 'state_shape'])
def test_loader_rejects_wrong_strategy_frame_provenance_and_state(tmp_path, bad):
    _, state, process_path, canonical_path, aug_path = fixture(tmp_path)
    metadata = state['metadata']; objective = metadata['current_objective']; migrate = objective['migration']
    if bad == 'factory': objective['model_factory'] = 'old:factory'
    elif bad == 'strategy': objective['strategy'] = 'old'
    elif bad == 'migration_strategy': migrate['strategy'] = 'old'
    elif bad == 'missing_frame': migrate.pop('geometry_frame')
    elif bad == 'frame': migrate['geometry_frame']['name'] = 'other'
    elif bad == 'metadata_frame': metadata['geometry_frame']['name'] = 'other'
    elif bad == 'process_sha': process_path.write_text(process_path.read_text() + '\n')
    elif bad == 'canonical_sha': canonical_path.write_text(canonical_path.read_text() + '\n')
    elif bad == 'definition': migrate['canonical_definition_sha256'] = 'd' * 64
    elif bad == 'aug_pair': migrate.pop('process_augmentation_manifest_sha256')
    elif bad == 'aug_sha': aug_path.write_text(aug_path.read_text() + '\n')
    elif bad.startswith('aug_'):
        aug = json.loads(aug_path.read_text())
        if bad == 'aug_schema': aug['schema'] = 'other'
        elif bad == 'aug_binding': aug['canonical_manifest_sha256'] = 'd' * 64
        elif bad == 'aug_split': aug['variants'][0]['original_optimizer_split'] = 'val'
        elif bad == 'aug_source': aug['variants'][0]['source_npz_sha256'] = 'd' * 64
        elif bad == 'aug_control': aug['variants'][0]['process_params'][2] += 1
        elif bad == 'aug_duplicate': aug['variants'][1] = deepcopy(aug['variants'][0])
        elif bad == 'aug_proof': aug['variants'][0]['verification']['material_replay_passed'] = False
        elif bad == 'aug_incomplete': aug['verification']['checked_variants'] = 1
        write(aug_path, aug); migrate['process_augmentation_manifest_sha256'] = sha(aug_path)
    else:
        key = next(iter(state['model']))
        if bad == 'state_missing': state['model'].pop(key)
        elif bad == 'state_dtype': state['model'][key] = state['model'][key].double()
        elif bad == 'state_shape': state['model'][key] = state['model'][key].reshape(-1)[:1]
    path = tmp_path / 'invalid.pt'; torch.save(state, path)
    with pytest.raises(ValueError):
        load_roi_lattice_direct_program_model(path)


def test_evaluator_calls_old_strict_evaluation_once_and_adds_explicit_provenance(tmp_path, monkeypatch):
    import weldgrind_v1.canonical_direct_evaluation as original
    _, model = models()
    calls = []
    def strict(*a, **kw):
        calls.append((a, kw)); tmp_path.joinpath('evaluation').mkdir()
        return {'count': 1, 'mean': {'ordered_rmse_mm': 2.}, 'thresholds': {'untouched': 1},
                'source_thresholds': {'untouched': 1}, 'source_gt_evaluated_separately': True}
    monkeypatch.setattr(original, 'evaluate_canonical_direct_subset', strict)
    result = evaluate_roi_lattice_subset(model, object(), [0], tmp_path / 'evaluation', batch_size=1)
    assert len(calls) == 1
    assert result['model_strategy'] == STRATEGY and result['geometry_frame'] == GEOMETRY_FRAME_POLICY
    assert result['input_encoding_changed'] and result['targets_and_thresholds_unchanged']
    assert result['thresholds'] == result['source_thresholds'] == {'untouched': 1}
    assert json.loads((tmp_path / 'evaluation/summary.json').read_text()) == result
    old, _ = models()
    with pytest.raises(ValueError, match='explicit new model'):
        evaluate_roi_lattice_subset(old, object(), [0], tmp_path / 'wrong')
    assert len(calls) == 1
