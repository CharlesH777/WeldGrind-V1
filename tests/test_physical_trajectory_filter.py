"""Geometric properties of the CPU filter; no trained-model/robot claims.

Fixtures specify observable noisy clouds and model-predicted phases. Ground
truth curves appear only in assertions measuring the effect of filtering.
"""

import inspect
import json

import numpy as np
import pytest

from weldgrind_v1.physical_trajectory_filter import (
    PhysicalFilterConfig,
    continuity_metrics,
    filter_trajectory,
)


def plane_cloud():
    rng = np.random.default_rng(117)
    x, y = np.meshgrid(np.linspace(-3, 23, 53), np.linspace(-5, 5, 25))
    return np.column_stack((x.ravel(), y.ravel(), rng.normal(0, .015, x.size)))


def phase_logits(phases):
    logits = np.full((len(phases), 4), -8., dtype=np.float64)
    logits[np.arange(len(phases)), phases] = 8.
    return logits


def noisy_contact():
    rng = np.random.default_rng(818)
    truth = np.column_stack((np.linspace(0, 12, 121), np.zeros((121, 2))))
    noisy = truth + rng.normal(0, .15, truth.shape)
    noisy[[27, 63, 94]] += [.2, .4, .8]
    noisy[[0, -1]] = truth[[0, -1]]
    return truth, noisy


def sample_rmse(prediction, target):
    return np.sqrt(np.mean(np.sum((prediction - target) ** 2, axis=1)))


def test_noisy_contact_accuracy_and_spatial_continuity_improve_without_mutating_input():
    truth, noisy = noisy_contact()
    cloud = plane_cloud()
    before, cloud_before = noisy.copy(), cloud.copy()
    phases = np.zeros(len(noisy), dtype=np.int64)
    result = filter_trajectory(noisy, cloud, phase_logits=phase_logits(phases),
                               camera_position_mm=np.array([0., 0., 30.]))
    filtered = result['trajectory_filtered_mm']
    assert filtered.shape == noisy.shape
    assert np.isfinite(filtered).all()
    assert sample_rmse(filtered, truth) < .85 * sample_rmse(noisy, truth)
    assert np.mean(np.diff(filtered, n=3, axis=0) ** 2) < .65 * np.mean(np.diff(noisy, n=3, axis=0) ** 2)
    np.testing.assert_array_equal(noisy, before)
    np.testing.assert_array_equal(cloud, cloud_before)
    np.testing.assert_array_equal(filtered[[0, -1]], noisy[[0, -1]])
    np.testing.assert_array_equal(result['predicted_phases'], phases)
    probabilities = result['phase_probabilities']
    assert probabilities.shape == (len(noisy), 4)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all() and (probabilities <= 1).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1., atol=1e-6)
    json.dumps(result['report'], allow_nan=False)
    json.dumps(continuity_metrics(filtered, phases), allow_nan=False)


def test_noise_free_right_angle_is_preserved_within_one_predicted_phase():
    incoming = np.column_stack((np.linspace(0, 10, 51), np.zeros((51, 2))))
    outgoing = np.column_stack((np.full(50, 10.), np.linspace(.2, 10, 50), np.zeros(50)))
    trajectory = np.concatenate((incoming, outgoing))
    result = filter_trajectory(trajectory, plane_cloud(),
                               predicted_phases=np.zeros(len(trajectory), dtype=np.int64))
    # A real geometric turn can occur without a discrete motion-phase change.
    # It must not be rounded away merely to improve a difference statistic.
    deviations = np.linalg.norm(result['trajectory_filtered_mm'] - trajectory, axis=1)
    assert deviations.max() < .02
    assert result['report']['protected_geometric_corners'] >= 1


def test_isolated_spike_on_straight_path_is_removed_instead_of_protected_as_corner():
    truth = np.column_stack((np.linspace(0, 10, 101), np.zeros((101, 2))))
    noisy = truth.copy()
    noisy[50, 1] = 1.5
    result = filter_trajectory(noisy, plane_cloud(),
                               predicted_phases=np.zeros(len(noisy), dtype=np.int64))
    filtered = result['trajectory_filtered_mm']
    assert np.linalg.norm(filtered[50] - truth[50]) < .6
    assert sample_rmse(filtered, truth) < .6 * sample_rmse(noisy, truth)
    assert result['report']['spike_candidates'] >= 1


def test_air_motion_keeps_retract_transfer_and_approach_instead_of_snapping_to_cloud():
    contact = np.column_stack((np.linspace(0, 12, 40), np.zeros((40, 2))))
    retract = np.column_stack((np.full(25, 12.), np.zeros(25), np.linspace(0, 4, 25)))
    transfer = np.column_stack((np.linspace(12, 18, 40), np.zeros(40), np.full(40, 4.)))
    approach = np.column_stack((np.full(25, 18.), np.zeros(25), np.linspace(4, 0, 25)))
    trajectory = np.concatenate((contact, retract, transfer, approach))
    phases = np.repeat(np.arange(4), [40, 25, 40, 25])
    filtered = filter_trajectory(trajectory, plane_cloud(), phase_logits=phase_logits(phases),
                                 camera_position_mm=np.array([0., 0., 30.]))['trajectory_filtered_mm']
    # The physically necessary 4 mm air clearance must survive filtering.
    assert filtered[70:100, 2].min() > 3.8
    assert filtered[60, 2] > 3.1
    assert filtered[110, 2] > 2.7
    assert filtered[40:65, 2].max() > 3.8
    assert filtered[105:, 2].max() > 3.8
    np.testing.assert_array_equal(filtered[[0, -1]], trajectory[[0, -1]])


def test_later_pass_can_cut_below_initial_observed_surface():
    first = np.column_stack((np.linspace(0, 12, 40), np.zeros((40, 2))))
    retract = np.column_stack((np.full(25, 12.), np.zeros(25), np.linspace(0, 4, 25)))
    transfer = np.column_stack((np.linspace(12, 0, 40), np.zeros(40), np.full(40, 4.)))
    approach = np.column_stack((np.zeros((25, 2)), np.linspace(4, -1.5, 25)))
    second = np.column_stack((np.linspace(0, 12, 40), np.zeros(40), np.full(40, -1.5)))
    trajectory = np.concatenate((first, retract, transfer, approach, second))
    phases = np.repeat([0, 1, 2, 3, 0], [40, 25, 40, 25, 40])
    result = filter_trajectory(trajectory, plane_cloud(), phase_logits=phase_logits(phases),
                               camera_position_mm=np.array([0., 0., 30.]))
    # A first-frame cloud does not describe material after earlier passes.
    np.testing.assert_allclose(result['trajectory_filtered_mm'][138:162, 2], -1.5, atol=.1)
    np.testing.assert_array_equal(result['predicted_phases'], phases)


def test_observed_plane_corrects_only_outside_contact_with_camera_oriented_normal():
    trajectory = np.column_stack((np.linspace(0, 12, 61), np.zeros(61), np.ones(61)))
    config = PhysicalFilterConfig(smoothness_weight=0., jerk_weight=0.)
    camera = np.array([0., 0., 30.])
    contact = filter_trajectory(trajectory, plane_cloud(),
                                predicted_phases=np.zeros(61, dtype=np.int64),
                                camera_position_mm=camera, config=config)
    air = filter_trajectory(trajectory, plane_cloud(),
                            predicted_phases=np.full(61, 2, dtype=np.int64),
                            camera_position_mm=camera, config=config)
    assert contact['trajectory_filtered_mm'][10:-10, 2].mean() < .85
    np.testing.assert_allclose(air['trajectory_filtered_mm'][10:-10], trajectory[10:-10], atol=.02)
    # Reversing camera side reverses the single-sided physical correction.
    below = trajectory.copy()
    below[:, 2] = -1.
    reverse = filter_trajectory(below, plane_cloud(),
                                predicted_phases=np.zeros(61, dtype=np.int64),
                                camera_position_mm=-camera, config=config)
    assert reverse['trajectory_filtered_mm'][10:-10, 2].mean() > -.85


def test_filter_is_equivariant_to_rigid_frame_change_with_camera_transformed():
    _, trajectory = noisy_contact()
    cloud = plane_cloud()
    camera = np.array([0., 0., 30.])
    axis = np.array([1., 2., 3.]) / np.sqrt(14.)
    cross = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    angle = .73
    rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
    translation = np.array([17., -9., 4.])
    logits = phase_logits(np.zeros(len(trajectory), dtype=np.int64))
    base = filter_trajectory(trajectory, cloud, phase_logits=logits, camera_position_mm=camera)
    rotated = filter_trajectory(trajectory @ rotation.T + translation,
                                cloud @ rotation.T + translation,
                                phase_logits=logits,
                                camera_position_mm=camera @ rotation.T + translation)
    expected = base['trajectory_filtered_mm'] @ rotation.T + translation
    np.testing.assert_allclose(rotated['trajectory_filtered_mm'], expected, atol=3e-4, rtol=0)
    np.testing.assert_array_equal(rotated['predicted_phases'], base['predicted_phases'])


def test_correction_trust_region_bounds_changes_and_preserves_endpoints():
    _, noisy = noisy_contact()
    config = PhysicalFilterConfig(max_correction_mm=.08)
    result = filter_trajectory(noisy, plane_cloud(),
                               predicted_phases=np.zeros(len(noisy), dtype=np.int64), config=config)
    changes = np.linalg.norm(result['trajectory_filtered_mm'] - noisy, axis=1)
    assert changes.max() <= .08 + 1e-6
    assert changes.max() > .001
    np.testing.assert_array_equal(result['trajectory_filtered_mm'][[0, -1]], noisy[[0, -1]])


def test_rgb_columns_do_not_change_geometric_constraint():
    _, trajectory = noisy_contact()
    cloud = plane_cloud()
    rgb_cloud = np.column_stack((cloud, np.full_like(cloud, .5)))
    phases = np.zeros(len(trajectory), dtype=np.int64)
    first = filter_trajectory(trajectory, cloud, predicted_phases=phases)
    second = filter_trajectory(trajectory, rgb_cloud, predicted_phases=phases)
    np.testing.assert_allclose(first['trajectory_filtered_mm'], second['trajectory_filtered_mm'], atol=1e-7)


def test_weak_isolated_phase_flicker_is_removed_without_imposing_a_stage_template():
    truth, _ = noisy_contact()
    expected = np.repeat([2, 0, 2, 1], [30, 30, 30, 31])
    logits = phase_logits(expected)
    # A weak one-token alternate hypothesis should not split the contact run.
    logits[45] = [1., -4., 1.4, -4.]
    assert logits[45].argmax() == 2
    result = filter_trajectory(truth, plane_cloud(), phase_logits=logits)
    np.testing.assert_array_equal(result['predicted_phases'], expected)


def test_missing_phase_keeps_unknown_status_and_disables_contact_surface_correction():
    trajectory = np.column_stack((np.linspace(0, 12, 61), np.zeros(61), np.ones(61)))
    result = filter_trajectory(trajectory, plane_cloud(), camera_position_mm=np.array([0., 0., 30.]),
                               config=PhysicalFilterConfig(smoothness_weight=0., jerk_weight=0.))
    np.testing.assert_array_equal(result['predicted_phases'], np.full(61, -1, dtype=np.int64))
    np.testing.assert_allclose(result['trajectory_filtered_mm'], trajectory, atol=.02)


@pytest.mark.parametrize('kwargs', [
    {'max_correction_mm': -1.},
    {'max_correction_mm': float('nan')},
    {'surface_tolerance_mm': -1.},
    {'surface_weight': -1.},
    {'smoothness_weight': -1.},
    {'jerk_weight': float('inf')},
    {'phase_switch_cost': -1.},
    {'max_removal_depth_mm': -1.},
])
def test_invalid_physical_configuration_is_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        config = PhysicalFilterConfig(**kwargs)
        _, noisy = noisy_contact()
        filter_trajectory(noisy, plane_cloud(), predicted_phases=np.zeros(len(noisy), dtype=int), config=config)


@pytest.mark.parametrize('kind', ['nonfinite_trajectory', 'bad_trajectory_shape', 'bad_cloud_shape',
                                  'empty_roi', 'bad_roi_shape', 'bad_logits', 'bad_phases'])
def test_invalid_observations_and_predictions_fail_explicitly(kind):
    _, trajectory = noisy_contact()
    cloud = plane_cloud()
    kwargs = {'predicted_phases': np.zeros(len(trajectory), dtype=np.int64)}
    if kind == 'nonfinite_trajectory':
        trajectory[30, 0] = np.nan
    elif kind == 'bad_trajectory_shape':
        trajectory = trajectory[:, :2]
    elif kind == 'bad_cloud_shape':
        cloud = cloud[:, :2]
    elif kind == 'empty_roi':
        kwargs['roi_mask'] = np.zeros(len(cloud), dtype=bool)
    elif kind == 'bad_roi_shape':
        kwargs['roi_mask'] = np.ones(len(cloud) - 1, dtype=bool)
    elif kind == 'bad_logits':
        kwargs = {'phase_logits': np.zeros((len(trajectory), 3))}
    elif kind == 'bad_phases':
        kwargs['predicted_phases'][40] = 4
    with pytest.raises((ValueError, TypeError)):
        filter_trajectory(trajectory, cloud, **kwargs)


def test_filter_api_has_no_ground_truth_conditions():
    parameters = inspect.signature(filter_trajectory).parameters
    forbidden = {'target_mm', 'target_phases', 'segment_ids', 'pass_index', 'surface_snapshot', 'ground_truth'}
    assert forbidden.isdisjoint(parameters)
    _, trajectory = noisy_contact()
    with pytest.raises(TypeError):
        filter_trajectory(trajectory, plane_cloud(), target_mm=trajectory)
