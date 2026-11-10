import inspect

import pytest
import torch

from weldgrind_v1.heightmap_models import (HeightMapFieldUNet, HeatMapTrajectoryDiffusion,
                                          axial_tangent, field_loss)


def inputs(batch=2, size=16):
    return torch.randn(batch, 15, size, size), torch.tensor([[.2, .1, 1., .5]]).repeat(batch, 1)


def tiny_diffusion():
    return HeatMapTrajectoryDiffusion(dict(image_width=8, sequence_width=8, condition_dimensions=16, timesteps=20))


def test_pixel_heads_are_finite_physical_and_all_parameters_receive_gradients():
    model = HeightMapFieldUNet(dict(widths=[8, 16], condition_dimensions=16))
    features, process = inputs()
    result = model(features, process)
    assert result['cut_depth_normalized'].shape == (2, 1, 16, 16)
    assert (result['cut_depth_normalized'] >= 0).all()
    torch.testing.assert_close(result['tangent_axis'].norm(dim=1), torch.ones(2, 16, 16))
    mask = torch.ones(2, 16, 16, dtype=torch.bool)
    target_axis = torch.zeros(2, 2, 16, 16); target_axis[:, 0] = 1.
    losses = field_loss(result, cut_depth_normalized=torch.full((2, 16, 16), .1), cut_valid=mask,
                        tangent_axis=target_axis, axis_valid=mask, contact_target=mask.float())
    losses['loss'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_opposite_strokes_have_the_same_nonzero_axis_target():
    velocity = torch.tensor([[1., 2.], [-1., -2.]])
    encoded = axial_tangent(velocity)
    torch.testing.assert_close(encoded[0], encoded[1])
    torch.testing.assert_close(encoded.norm(dim=-1), torch.ones(2))
    assert torch.equal(velocity.mean(0), torch.zeros(2))


def test_invalid_field_labels_cannot_poison_masked_objective():
    model = HeightMapFieldUNet(dict(widths=[8, 16], condition_dimensions=16))
    features, process = inputs(batch=1)
    output = model(features, process)
    valid = torch.zeros(1, 16, 16, dtype=torch.bool); valid[:, :8] = True
    target = torch.full((1, 16, 16), float('nan')); target[valid] = .1
    direction = torch.full((1, 2, 16, 16), float('nan')); direction[:, 0, :8] = 1.; direction[:, 1, :8] = 0.
    loss = field_loss(output, cut_depth_normalized=target, cut_valid=valid,
                      tangent_axis=direction, axis_valid=valid, contact_target=target)
    assert torch.isfinite(loss['loss'])
    loss['loss'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_diffusion_broadcast_and_oracle_noise_recover_the_same_clean_teacher():
    model = tiny_diffusion()
    features, process = inputs()
    clean = torch.randn(2, 17, 3)
    noise = torch.randn_like(clean)
    time = torch.tensor([0, 19])
    a = model.alpha_bar[time, None, None]
    torch.testing.assert_close(model.q_sample(clean, time, noise), a.sqrt() * clean + (1 - a).sqrt() * noise)
    model.denoise = lambda noisy, timestep, condition, tokens: noise
    result = model.loss(features, process, clean, torch.zeros(2, 17, dtype=torch.long), timestep=time, noise=noise)
    assert result['noise'] == 0
    torch.testing.assert_close(result['predicted_x0'], clean, atol=1e-6, rtol=1e-6)


def test_diffusion_train_phase_uses_detached_self_estimated_clean_state():
    model = tiny_diffusion()
    features, process = inputs()
    seen = []
    original = model.phase_logits
    def capture(clean, condition):
        seen.append(clean)
        return original(clean, condition)
    model.phase_logits = capture
    target = torch.randn(2, 17, 3)
    result = model.loss(features, process, target, torch.arange(17)[None].repeat(2, 1) % 4)
    torch.testing.assert_close(seen[0], result['predicted_x0'].detach())
    assert not seen[0].requires_grad
    assert not torch.equal(seen[0], target)
    result['loss'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_sampling_is_seeded_without_global_rng_or_gt_or_train_mode_changes():
    model = tiny_diffusion()
    features, process = inputs(batch=1)
    state = torch.get_rng_state().clone()
    one = model.sample(features, process, points=19, steps=4, seed=5)
    two = model.sample(features, process, points=19, steps=4, seed=5)
    three = model.sample(features, process, points=19, steps=4, seed=6)
    torch.testing.assert_close(one['trajectory_normalized'], two['trajectory_normalized'], atol=0, rtol=0)
    assert not torch.equal(one['trajectory_normalized'], three['trajectory_normalized'])
    assert torch.equal(state, torch.get_rng_state())
    assert model.training
    assert one['end_token_index'] == 18 and one['ground_truth_input'] is False
    assert not {'target_xyz', 'target_phases', 'pass_ids', 'chart'} & set(inspect.signature(model.sample).parameters)


@pytest.mark.parametrize('kind', ['field', 'diffusion'])
def test_nonfinite_observation_is_rejected(kind):
    features, process = inputs(batch=1); features[0, 0, 0, 0] = float('nan')
    model = HeightMapFieldUNet(dict(widths=[8, 16], condition_dimensions=16)) if kind == 'field' else tiny_diffusion()
    with pytest.raises(ValueError):
        model(features, process) if kind == 'field' else model.sample(features, process, points=8, steps=2)


@pytest.mark.parametrize('points,steps', [(3, 4), (16, 0), (16, 21), (True, 2)])
def test_invalid_sampling_budget_is_rejected(points, steps):
    model = tiny_diffusion(); features, process = inputs(batch=1)
    with pytest.raises(ValueError):
        model.sample(features, process, points=points, steps=steps)
