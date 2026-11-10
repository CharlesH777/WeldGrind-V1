"""Physical support tests for observation-only surrounding-stock fitting."""
import dataclasses
import json

import numpy as np
import pytest

from weldgrind_v1.background_surface import BackgroundSurfaceConfig, ObservedBackgroundSurface


def cloud(kind='plane', *, noise=.01, seed=31, half_size=4., count=71):
    rng = np.random.default_rng(seed)
    axis = np.linspace(-half_size, half_size, count)
    x, y = np.meshgrid(axis, axis)
    if kind == 'plane':
        z = .1 * x - .04 * y
    elif kind == 'quadratic':
        z = .08 * x * x + .04 * y * y + .03 * x * y
    elif kind == 'cylinder':
        z = np.sqrt(12. ** 2 - x * x) - 12.
    elif kind == 'sphere':
        z = np.sqrt(12. ** 2 - x * x - y * y) - 12.
    elif kind == 'fold':
        z = .8 * np.abs(x)
    else:
        raise ValueError(kind)
    roi = (x * x + y * y) <= 1.
    bump = .8 * np.maximum(1. - x * x - y * y, 0.) ** 2
    xyz = np.column_stack((x.ravel(), y.ravel(), (z + bump + rng.normal(0., noise, z.shape)).ravel()))
    return xyz, roi.ravel()


@pytest.mark.parametrize('kind', ['plane', 'quadratic', 'cylinder', 'sphere'])
def test_recovers_surrounding_surface_without_fitting_the_roi_bump(kind):
    xyz, roi = cloud(kind)
    original, original_roi = xyz.copy(), roi.copy()
    fit = ObservedBackgroundSurface(xyz, roi, camera_position_mm=[0., 0., 30.])
    q = np.array([[0., 0., .7], [.4, -.2, .8], [-.3, .3, .6]])
    result = fit.query(q)
    assert fit.report['available'], fit.report
    assert fit.report['model'] == ('plane' if kind == 'plane' else 'quadratic')
    assert np.all(result['confidence'] > .7)
    x, y = q[:, 0], q[:, 1]
    if kind == 'plane':
        expected = .1 * x - .04 * y
    elif kind == 'quadratic':
        expected = .08 * x * x + .04 * y * y + .03 * x * y
    elif kind == 'cylinder':
        expected = np.sqrt(12. ** 2 - x * x) - 12.
    else:
        expected = np.sqrt(12. ** 2 - x * x - y * y) - 12.
    # Anchors lie at chart coordinates rather than the world's vertical ray;
    # verify their fitted normal coordinate against the analytic surface.
    anchor = result['anchors_mm']
    if kind == 'plane':
        residual = anchor[:, 2] - (.1 * anchor[:, 0] - .04 * anchor[:, 1])
    elif kind == 'quadratic':
        residual = anchor[:, 2] - (.08 * anchor[:, 0] ** 2 + .04 * anchor[:, 1] ** 2 + .03 * anchor[:, 0] * anchor[:, 1])
    elif kind == 'cylinder':
        residual = anchor[:, 2] - (np.sqrt(12. ** 2 - anchor[:, 0] ** 2) - 12.)
    else:
        residual = anchor[:, 2] - (np.sqrt(12. ** 2 - anchor[:, 0] ** 2 - anchor[:, 1] ** 2) - 12.)
    assert np.max(np.abs(residual)) < .025
    assert np.all(result['signed_distance_mm'] > .5)
    assert np.allclose(np.linalg.norm(result['normals'], axis=1), 1.)
    assert np.array_equal(xyz, original) and np.array_equal(roi, original_roi)
    assert fit.report['source'] == 'observed_nonroi_point_cloud_only'
    json.dumps(fit.report, allow_nan=False)


def test_holes_inside_sam_hull_and_sparse_ring_outliers_do_not_become_the_base():
    xyz, roi = cloud('plane', noise=.008)
    rng = np.random.default_rng(8)
    # False negative SAM pixels inside its outer boundary must not leak the
    # original bump into the surrounding-stock fit.
    holes = roi & (np.linalg.norm(xyz[:, :2], axis=1) < .6)
    roi[holes] = False
    candidates = np.flatnonzero(~roi & (np.linalg.norm(xyz[:, :2], axis=1) > 1.3))
    contaminated = rng.choice(candidates, 140, replace=False)
    xyz[contaminated, 2] += 1.2
    fit = ObservedBackgroundSurface(xyz, roi, camera_position_mm=[0., 0., 30.])
    result = fit.query(np.array([[0., 0., .5], [.3, -.2, .5]]))
    assert np.all(result['confidence'] > .5), fit.report
    base = result['anchors_mm']
    assert np.max(np.abs(base[:, 2] - (.1 * base[:, 0] - .04 * base[:, 1]))) < .03


def test_normal_air_height_is_supported_but_tangential_extrapolation_is_not():
    xyz, roi = cloud()
    fit = ObservedBackgroundSurface(xyz, roi, camera_position_mm=[0., 0., 30.])
    q = np.array([[0., 0., 0.], [0., 0., 6.], [4., 0., 0.], [0., 0., 25.]])
    result = fit.query(q)
    assert np.all(result['confidence'][:2] > .5)
    assert np.all(result['confidence'][2:] == 0.)
    assert np.allclose(result['normals'][0], result['normals'][1], atol=.002)


def test_observation_and_camera_rigid_transform_gives_same_surface():
    xyz, roi = cloud('quadratic', count=61)
    camera = np.array([0., 0., 30.])
    q = np.array([[0., 0., .7], [.4, -.2, .8], [-.3, .3, 4.]])
    rotation, _ = np.linalg.qr(np.random.default_rng(43).normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    translation = np.array([18., -11., 4.])
    first = ObservedBackgroundSurface(xyz, roi, camera_position_mm=camera).query(q)
    second = ObservedBackgroundSurface(xyz @ rotation.T + translation, roi,
        camera_position_mm=camera @ rotation.T + translation).query(q @ rotation.T + translation)
    assert np.all(first['confidence'] > .5)
    assert np.allclose(second['anchors_mm'], first['anchors_mm'] @ rotation.T + translation, atol=1e-7)
    assert np.allclose(second['normals'], first['normals'] @ rotation.T, atol=1e-7)
    assert np.allclose(first['signed_distance_mm'], second['signed_distance_mm'], atol=1e-7)
    assert np.allclose(first['confidence'], second['confidence'], atol=1e-7)


def test_missing_camera_disables_oriented_evidence_without_inventing_a_sign():
    xyz, roi = cloud()
    fit = ObservedBackgroundSurface(xyz, roi)
    result = fit.query(np.array([[0., 0., .7]]))
    assert fit.report['available']
    assert result['oriented'] is False
    assert result['confidence'][0] == 0.


@pytest.mark.parametrize('case', ['empty', 'all_roi', 'no_roi', 'line', 'one_sided', 'fold', 'nonfinite'])
def test_missing_or_unsafe_background_support_is_a_finite_zero_confidence_fallback(case):
    xyz, roi = cloud('fold' if case == 'fold' else 'plane')
    if case == 'empty':
        xyz, roi = np.empty((0, 3)), np.empty(0, dtype=bool)
    elif case == 'all_roi':
        roi[:] = True
    elif case == 'no_roi':
        roi[:] = False
    elif case == 'line':
        xyz = np.column_stack((np.linspace(-3., 3., 80), np.zeros((80, 2))))
        roi = np.arange(80) % 4 == 0
    elif case == 'one_sided':
        keep = roi | (xyz[:, 0] > 1.3)
        xyz, roi = xyz[keep], roi[keep]
    elif case == 'nonfinite':
        xyz[~roi] = np.nan
    fit = ObservedBackgroundSurface(xyz, roi, camera_position_mm=[0., 0., 30.])
    result = fit.query(np.array([[0., 0., .7]]))
    assert result['confidence'][0] == 0., fit.report
    for key in ('anchors_mm', 'normals', 'signed_distance_mm', 'confidence', 'sigma_mm'):
        assert np.isfinite(result[key]).all()
    json.dumps(fit.report, allow_nan=False)


def test_nonfinite_rows_are_removed_and_rgb_is_not_used():
    xyz, roi = cloud()
    rgb = np.full_like(xyz, np.nan)
    with_bad = np.vstack((np.column_stack((xyz, rgb)), np.full((2, 6), np.inf)))
    mask = np.r_[roi, False, True]
    fit = ObservedBackgroundSurface(with_bad, mask, camera_position_mm=[0., 0., 30.])
    result = fit.query(np.array([[0., 0., .7]]))
    assert result['confidence'][0] > .5
    assert fit.report['nonfinite_points_removed'] == 2


def test_query_empty_and_invalid_shapes():
    xyz, roi = cloud()
    fit = ObservedBackgroundSurface(xyz, roi, camera_position_mm=[0., 0., 30.])
    assert fit.query(np.empty((0, 3)))['normals'].shape == (0, 3)
    for query in ([0., 0., 1.], [[np.nan, 0., 1.]], np.empty((2, 4))):
        with pytest.raises(ValueError):
            fit.query(query)
    with pytest.raises(ValueError):
        ObservedBackgroundSurface(xyz, roi.astype(int))
    with pytest.raises(ValueError):
        ObservedBackgroundSurface(xyz, roi, camera_position_mm=[0., np.nan, 3.])


@pytest.mark.parametrize('name,value', [('annulus_gap_mm', -1.), ('annulus_width_mm', 0.),
    ('min_points', 5), ('robust_iterations', True), ('max_slope', np.inf),
    ('min_inlier_fraction', 1.), ('min_view_cosine', 0.), ('max_points', 20),
    ('max_annulus_width_mm', 1.), ('quadratic_min_relative_improvement', np.nan)])
def test_configuration_rejects_invalid_values(name, value):
    with pytest.raises(ValueError):
        dataclasses.replace(BackgroundSurfaceConfig(), **{name: value})
