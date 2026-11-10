"""Observation-frame invariance and strict legacy migration, without training."""

import copy
import types

import pytest
import torch
from torch import nn

from weldgrind_v1.geometry import GeometryConfig, GeometryEncoder, SparseVoxelBackbone
from weldgrind_v1.geometry_acceleration import accelerate_geometry, vectorized_sparse_forward
from weldgrind_v1.roi_lattice_geometry import (
    GEOMETRY_FRAME_POLICY, RoiLatticeSparseVoxelBackbone, install_roi_lattice_geometry,
)


def config(voxel_size=.125):
    return GeometryConfig(dim=12, candidates=8, channels=(4, 5, 6, 7),
                          branch_dim=4, neighbors=8, voxel_size=voxel_size)


def near_observations(dtype=torch.float64, voxel_size=.125):
    # Mixed negative/positive cells, with multiple continuous points per cell.
    cells = torch.tensor([[-6, -4, -2], [-5, -3, -1], [-4, -2, 0],
                          [-2, -1, 1], [-1, 0, 2], [0, 1, 3],
                          [1, 2, 4], [2, 3, 5], [3, -2, -4]], dtype=dtype)
    xyz = torch.cat(((cells + .25) * voxel_size, (cells + .6) * voxel_size))
    rgb = torch.linspace(.1, .9, len(xyz) * 3, dtype=dtype).reshape(-1, 3)
    roi = torch.arange(len(xyz)).remainder(3) != 0
    return xyz, rgb, roi


def model_fixture(dtype=torch.float64):
    model = nn.Module()
    model.geometry = GeometryEncoder(config()).to(dtype=dtype)
    model.other_layer = nn.Linear(3, 2).to(dtype=dtype)
    return model


@pytest.mark.parametrize("accelerated", [False, True])
@pytest.mark.parametrize("far_cell", [-256, 256])
def test_disconnected_far_observation_cannot_change_near_multistage_centers_or_features(accelerated, far_cell):
    torch.manual_seed(71)
    backbone = RoiLatticeSparseVoxelBackbone(config()).double().eval()
    if accelerated:
        for layer in (*backbone.convolutions, *backbone.refinements):
            layer.forward = types.MethodType(vectorized_sparse_forward, layer)
    xyz, rgb, roi = near_observations()
    far_xyz = torch.tensor([[far_cell + .375, far_cell + .375, far_cell + .375]], dtype=xyz.dtype) * .125
    with torch.no_grad():
        near = backbone(xyz, rgb, roi)
        full = backbone(torch.cat((xyz, far_xyz)), torch.cat((rgb, torch.ones(1, 3, dtype=rgb.dtype))),
                        torch.cat((roi, torch.tensor([False]))))
    for (near_centers, near_features), (full_centers, full_features) in zip(near, full):
        is_near = full_centers.abs().amax(-1) < 2
        assert torch.equal(full_centers[is_near], near_centers)
        torch.testing.assert_close(full_features[is_near], near_features, atol=2e-12, rtol=2e-12)


def test_integer_center_restoration_avoids_float_cancellation_from_large_storage_shift():
    backbone = RoiLatticeSparseVoxelBackbone(config(.03)).eval()
    xyz, rgb, roi = near_observations(torch.float32, .03)
    far = torch.tensor([[-100000., .001, .001]])
    with torch.no_grad():
        near = backbone(xyz, rgb, roi)
        full = backbone(torch.cat((xyz, far)), torch.cat((rgb, torch.zeros(1, 3))),
                        torch.cat((roi, torch.tensor([False]))))
    for (centers, features), (full_centers, full_features) in zip(near, full):
        mask = full_centers.abs().amax(-1) < 2
        assert torch.equal(full_centers[mask], centers)
        torch.testing.assert_close(full_features[mask], features, atol=2e-6, rtol=2e-6)


def test_signed_negative_cells_keep_their_global_coarse_parent_and_nonnegative_storage():
    backbone = RoiLatticeSparseVoxelBackbone(config()).double()
    cells = torch.tensor([[-9, -1, 7], [-8, -8, -8], [-1, -1, -1],
                          [0, 0, 0], [1, 2, 3], [7, 8, 9]])
    xyz = (cells.double() + .25) * .125
    observed_coordinates = []
    handles = [layer.register_forward_pre_hook(lambda _module, args: observed_coordinates.append(args[0].clone()))
               for layer in backbone.convolutions]
    stages = backbone(xyz, torch.ones_like(xyz) * .3, torch.ones(len(xyz), dtype=torch.bool))
    for handle in handles:
        handle.remove()
    assert all((coordinates >= 0).all() for coordinates in observed_coordinates)
    assert observed_coordinates[0].amin(0).tolist() == [7, 0, 0]
    for stage, (centers, _) in enumerate(stages):
        parents = torch.unique(torch.div(cells, 2 ** stage, rounding_mode="floor"), dim=0, sorted=True)
        expected = (parents.double() + .5) * (.125 * 2 ** stage)
        assert torch.equal(centers, expected)


def test_translation_by_stride_aligned_offset_preserves_occupancy_with_translated_centers():
    backbone = RoiLatticeSparseVoxelBackbone(config()).double()
    xyz, rgb, roi = near_observations()
    delta = torch.tensor([8., -16., 24.], dtype=xyz.dtype) * .125
    original = backbone(xyz, rgb, roi)
    moved = backbone(xyz + delta, rgb, roi)
    for (centers, _), (moved_centers, moved_features) in zip(original, moved):
        assert torch.equal(moved_centers, centers + delta)
        assert torch.isfinite(moved_features).all()
    # Continuous VFE coordinates intentionally express the fixed ROI frame;
    # this test does not assert invariance to changing that physical frame.


def test_reference_and_accelerated_hierarchy_match_outputs_and_finite_gradients():
    torch.manual_seed(43)
    reference = RoiLatticeSparseVoxelBackbone(config()).double()
    fast = copy.deepcopy(reference)
    for layer in (*fast.convolutions, *fast.refinements):
        layer.forward = types.MethodType(vectorized_sparse_forward, layer)
    xyz, rgb, roi = near_observations()
    xyz = xyz.requires_grad_()
    rgb = rgb.requires_grad_()
    fast_xyz = xyz.detach().clone().requires_grad_()
    fast_rgb = rgb.detach().clone().requires_grad_()
    reference_stages = reference(xyz, rgb, roi)
    fast_stages = fast(fast_xyz, fast_rgb, roi)
    for (centers, features), (fast_centers, fast_features) in zip(reference_stages, fast_stages):
        assert torch.equal(centers, fast_centers)
        torch.testing.assert_close(fast_features, features, atol=2e-12, rtol=2e-12)
    sum(features.square().mean() for _, features in reference_stages).backward()
    sum(features.square().mean() for _, features in fast_stages).backward()
    for left, right in ((xyz, fast_xyz), (rgb, fast_rgb)):
        assert torch.isfinite(left.grad).all() and left.grad.abs().sum() > 0
        torch.testing.assert_close(right.grad, left.grad, atol=2e-11, rtol=2e-11)
    for (name, parameter), (fast_name, fast_parameter) in zip(reference.named_parameters(), fast.named_parameters()):
        assert fast_name == name
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(fast_parameter.grad, parameter.grad, atol=2e-11, rtol=2e-11)


def test_legacy_state_copies_strictly_and_installer_preserves_parameters_rng_modes_and_optimizer_references():
    torch.manual_seed(79)
    model = model_fixture()
    accelerate_geometry(model)
    model.eval()
    model.geometry.backbone.convolutions[0].train()
    next(model.parameters()).requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters())
    original = model.geometry.backbone
    before_parameters = tuple(model.named_parameters())
    before_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    before_modes = {name: module.training for name, module in model.named_modules()}
    before_requires_grad = {name: parameter.requires_grad for name, parameter in before_parameters}
    before_rng = torch.get_rng_state().clone()
    before_convolution_forward = original.convolutions[0].forward
    before_sampler = model.geometry.point_sampler
    audit = install_roi_lattice_geometry(model)
    assert type(model.geometry.backbone) is RoiLatticeSparseVoxelBackbone
    assert audit["installed"] and audit["rng_consumed"] is False
    assert audit["geometry_frame_policy"] == GEOMETRY_FRAME_POLICY
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert {name: module.training for name, module in model.named_modules()} == before_modes
    assert {name: parameter.requires_grad for name, parameter in model.named_parameters()} == before_requires_grad
    assert model.geometry.backbone.convolutions[0].forward == before_convolution_forward
    assert model.geometry.point_sampler is before_sampler
    after_parameters = tuple(model.named_parameters())
    assert [name for name, _ in after_parameters] == [name for name, _ in before_parameters]
    assert all(old is new for (_, old), (_, new) in zip(before_parameters, after_parameters))
    assert all(old is new for old, (_, new) in zip(optimizer.param_groups[0]["params"], after_parameters))
    after_state = model.state_dict()
    assert after_state.keys() == before_state.keys()
    for name, value in before_state.items():
        assert after_state[name].dtype == value.dtype and after_state[name].shape == value.shape
        assert torch.equal(after_state[name], value), name
    standalone = RoiLatticeSparseVoxelBackbone(original.config).double()
    standalone.load_state_dict(original.state_dict(), strict=True)
    for name, value in standalone.state_dict().items():
        assert torch.equal(value, original.state_dict()[name])
    before_repeat = torch.get_rng_state().clone()
    assert install_roi_lattice_geometry(model)["installed"] is False
    assert torch.equal(torch.get_rng_state(), before_repeat)


def test_roi_candidates_and_fps_are_unchanged_by_installation_or_disconnected_context():
    torch.manual_seed(37)
    model = model_fixture()
    xyz, rgb, roi = near_observations()
    points = torch.cat((xyz, rgb), -1)[None]
    legacy = model.geometry(points, roi[None])
    install_roi_lattice_geometry(model)
    updated = model.geometry(points, roi[None])
    far = points.new_tensor([[[-50., -50., -50., .2, .3, .4]]])
    with_far = model.geometry(torch.cat((points, far), 1), torch.cat((roi[None], torch.tensor([[False]])), 1))
    for name in ("candidate_xyz", "candidate_valid", "candidate_roi"):
        assert torch.equal(updated[name], legacy[name])
        assert torch.equal(updated[name], with_far[name])
    torch.testing.assert_close(updated["candidate_features"], with_far["candidate_features"], atol=2e-12, rtol=2e-12)


def test_installer_rejects_an_unrelated_backbone():
    model = nn.Module()
    model.geometry = nn.Module()
    model.geometry.backbone = nn.Identity()
    with pytest.raises(TypeError, match="SparseVoxelBackbone"):
        install_roi_lattice_geometry(model)


@pytest.mark.parametrize("xyz", [torch.empty(0, 3), torch.zeros(2, 4), torch.tensor([[float('nan'), 0., 0.]])])
def test_invalid_observations_fail_before_sparse_indexing(xyz):
    backbone = RoiLatticeSparseVoxelBackbone(config())
    with pytest.raises(ValueError, match="ROI lattice"):
        backbone(xyz, torch.zeros_like(xyz), torch.ones(len(xyz), dtype=torch.bool))
