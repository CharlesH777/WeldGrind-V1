"""Observation-only nominal surface interpolation across a SAM defect region.

The support is a ring of observed, non-ROI points outside the ROI's projected
convex hull. The interior is a hole to interpolate, not data to fit. No GT
surface, path, phase, pass or cutting height is accepted by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree


@dataclass(frozen=True)
class BackgroundSurfaceConfig:
    annulus_gap_mm: float = .25
    annulus_width_mm: float = 2.
    annulus_width_scale: float = .5
    max_annulus_width_mm: float = 4.
    min_points: int = 32
    min_quadratic_points: int = 60
    max_points: int = 4096
    robust_iterations: int = 6
    noise_floor_mm: float = .01
    max_residual_sigma_mm: float = .15
    model_residual_floor_mm: float = .04
    max_model_residual_noise_ratio: float = 4.
    min_inlier_fraction: float = .7
    min_tangent_ratio: float = .025
    quadratic_min_relative_improvement: float = .25
    quadratic_min_absolute_improvement_mm: float = .015
    max_design_condition: float = 1e4
    max_slope: float = 1.5
    min_view_cosine: float = .1
    max_query_normal_distance_mm: float = 20.

    def __post_init__(self):
        nonnegative = {'annulus_gap_mm', 'annulus_width_scale',
                       'quadratic_min_absolute_improvement_mm'}
        positive = {'annulus_width_mm', 'max_annulus_width_mm', 'noise_floor_mm',
                    'max_residual_sigma_mm', 'model_residual_floor_mm',
                    'max_model_residual_noise_ratio', 'max_design_condition', 'max_slope',
                    'max_query_normal_distance_mm'}
        fractions = {'min_inlier_fraction', 'min_tangent_ratio',
                     'quadratic_min_relative_improvement', 'min_view_cosine'}
        for name in nonnegative | positive | fractions:
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.number))
                    or not math.isfinite(float(value))):
                raise ValueError(f'{name} must be a finite number')
            if ((name in nonnegative and value < 0) or (name in positive and value <= 0)
                    or (name in fractions and not 0 < value < 1)):
                raise ValueError(f'{name} is outside its valid range')
        for name in ('min_points', 'min_quadratic_points', 'max_points', 'robust_iterations'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.min_points < 6 or self.min_quadratic_points < 12:
            raise ValueError('min_points >= 6 and min_quadratic_points >= 12 are required')
        if self.max_points < max(self.min_points, self.min_quadratic_points):
            raise ValueError('max_points must accommodate both support minima')
        if self.max_annulus_width_mm < self.annulus_width_mm:
            raise ValueError('max_annulus_width_mm must be >= annulus_width_mm')


def _stable_unique(points):
    # Preserve input order. Axis-aligned voxels or lexicographic subsampling
    # would introduce a new rotation-dependent geometry approximation.
    _, index = np.unique(points, axis=0, return_index=True)
    return points[np.sort(index)]


def _subsample(points, count):
    return points if len(points) <= count else points[np.linspace(0, len(points) - 1, count, dtype=int)]


def _design(uv, quadratic):
    u, v = uv.T
    if quadratic:
        return np.column_stack((np.ones(len(uv)), u, v, u * u, u * v, v * v))
    return np.column_stack((np.ones(len(uv)), u, v))


def _fit(uv, height, spatial, quadratic, config):
    matrix = _design(uv, quadratic)
    weights = spatial.copy()
    coefficients = np.zeros(matrix.shape[1])
    for _ in range(config.robust_iterations):
        weighted = matrix * np.sqrt(weights[:, None])
        singular = np.linalg.svd(weighted, compute_uv=False)
        if (len(singular) != matrix.shape[1] or singular[-1] <= 1e-12
                or singular[0] / singular[-1] > config.max_design_condition):
            return None
        coefficients = np.linalg.lstsq(weighted, height * np.sqrt(weights), rcond=None)[0]
        residual = height - matrix @ coefficients
        median = float(np.median(residual))
        sigma = 1.4826 * float(np.median(np.abs(residual - median)))
        huber_delta = 2.5 * max(sigma, config.noise_floor_mm)
        weights = spatial * np.minimum(1., huber_delta / np.maximum(np.abs(residual - median), 1e-12))
    residual = height - matrix @ coefficients
    median = float(np.median(residual))
    sigma = 1.4826 * float(np.median(np.abs(residual - median)))
    inliers = np.abs(residual - median) <= max(3. * sigma, 3. * config.noise_floor_mm)
    if not inliers.any():
        return None
    rms = float(np.sqrt(np.average(residual[inliers] ** 2, weights=spatial[inliers])))
    return {'coefficients': coefficients, 'sigma': max(sigma, rms), 'rms': rms,
            'inliers': inliers, 'inlier_fraction': float(inliers.mean()),
            'condition': float(singular[0] / singular[-1])}


def _local_noise_sigma(uv, height, config):
    """Estimate measurement noise with leave-one-out small local planes.

    A global quadratic can approximate a crease while leaving spatially
    coherent error. Its residual must be compatible with the observed local
    noise, not merely smaller than a permissive absolute noise ceiling.
    """
    count = min(17, len(uv))
    distance, index = cKDTree(uv).query(uv, k=count)
    delta = uv[index[:, 1:]] - uv[:, None]
    bandwidth = np.maximum(np.median(distance[:, 1:], axis=1), 1e-9)
    design = np.concatenate((np.ones((*delta.shape[:2], 1)), delta / bandwidth[:, None, None]), axis=2)
    coefficients = np.einsum('mij,mj->mi', np.linalg.pinv(design, rcond=1e-8), height[index[:, 1:]])
    residual = height - coefficients[:, 0]
    sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    return max(sigma, config.noise_floor_mm)


class ObservedBackgroundSurface:
    """Robust plane/quadratic interpolation of the observed surrounding stock.

    A positive signed distance faces the calibrated camera. Unsupported queries
    remain finite with zero confidence. Query support is tangential: ordinary
    retract heights are supported; only offsets over 20 mm are rejected by
    default. ``anchors_mm`` and ``normals`` define a tangent plane at each
    query's chart coordinates, not an orthogonal closest-point projection.
    """

    def __init__(self, points_mm, roi_mask, *, camera_position_mm=None, config=None):
        self.config = config if isinstance(config, BackgroundSurfaceConfig) else BackgroundSurfaceConfig(**(config or {}))
        cfg = self.config
        points = np.asarray(points_mm, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] not in (3, 6):
            raise ValueError('points_mm must be [N,3] or [N,6]')
        if roi_mask is None:
            mask = np.zeros(len(points), dtype=bool)
        else:
            mask = np.asarray(roi_mask)
            if mask.shape != (len(points),) or mask.dtype != np.bool_:
                raise ValueError('roi_mask must be a boolean [N] array')
        self.camera_position_mm = None
        if camera_position_mm is not None:
            camera = np.asarray(camera_position_mm, dtype=np.float64)
            if camera.shape != (3,) or not np.isfinite(camera).all():
                raise ValueError('camera_position_mm must be finite [3]')
            self.camera_position_mm = camera.copy()
        finite = np.isfinite(points[:, :3]).all(axis=1)
        roi = _stable_unique(points[finite & mask, :3])
        background = _stable_unique(points[finite & ~mask, :3])
        self.support_points_mm = np.empty((0, 3))
        self._available = False
        self.report = {
            'algorithm': 'nonroi_annulus_robust_monge_plane_or_quadratic_v1',
            'units': 'mm', 'source': 'observed_nonroi_point_cloud_only',
            'uses_ground_truth': False, 'input_points': int(len(points)),
            'roi_points': int(len(roi)), 'nonroi_points': int(len(background)),
            'nonfinite_points_removed': int((~finite).sum()),
            'oriented': self.camera_position_mm is not None,
            'model': 'unavailable', 'available': False, 'support_points': 0,
            'query_support': 'inlier_ring_convex_hull_in_tangent_chart',
            'normal_distance_limit_mm': float(cfg.max_query_normal_distance_mm),
        }
        if len(roi) < 6 or len(background) < cfg.min_points:
            self._disable('insufficient_roi_or_nonroi_points')
            return
        # PCA is used only for an observation-derived coordinate chart. The
        # resulting height model is fitted exclusively to surrounding points.
        roi_sample = _subsample(roi, cfg.max_points)
        origin = roi_sample.mean(axis=0)
        centered = roi_sample - origin
        covariance = centered.T @ centered / len(centered)
        eigenvalues, vectors = np.linalg.eigh(covariance)
        if eigenvalues[1] <= 1e-12 or eigenvalues[1] / max(eigenvalues[2], 1e-12) < cfg.min_tangent_ratio:
            self._disable('degenerate_roi_footprint')
            return
        normal = vectors[:, 0]
        if self.camera_position_mm is not None:
            if np.dot(normal, self.camera_position_mm - origin) < 0:
                normal = -normal
        else:
            if normal[np.argmax(np.abs(normal))] < 0:
                normal = -normal
        u_axis = vectors[:, 2]
        v_axis = np.cross(normal, u_axis)
        basis = np.column_stack((u_axis, v_axis, normal))
        roi_local = (roi_sample - origin) @ basis
        try:
            roi_hull = ConvexHull(roi_local[:, :2])
        except QhullError:
            self._disable('degenerate_roi_hull')
            return
        radius = float(np.quantile(np.linalg.norm(roi_local[:, :2], axis=1), .9))
        width = min(cfg.max_annulus_width_mm, max(cfg.annulus_width_mm, cfg.annulus_width_scale * radius))
        local = (background - origin) @ basis
        nearest, _ = cKDTree(roi_local[:, :2]).query(local[:, :2])
        outside_roi_hull = np.zeros(len(local), dtype=bool)
        for equation in roi_hull.equations:
            outside_roi_hull |= local[:, :2] @ equation[:2] + equation[2] > 1e-10
        # A chart-overlapping remote/back face is not surrounding stock. This
        # broad geometric gate is independent of subsequent query heights.
        normal_extent = float(np.ptp(roi_local[:, 2]))
        nearby_height = np.abs(local[:, 2]) <= max(1.5 * width, normal_extent + 1.)
        selected = (outside_roi_hull & (nearest >= cfg.annulus_gap_mm)
                    & (nearest <= cfg.annulus_gap_mm + width) & nearby_height)
        ring = _subsample(background[selected], cfg.max_points)
        self.report.update(annulus_width_mm=float(width), annulus_gap_mm=float(cfg.annulus_gap_mm),
                           annulus_candidate_points=int(selected.sum()), roi_tangent_radius_mm=radius)
        if len(ring) < cfg.min_points:
            self._disable('insufficient_annulus_support')
            return
        ring_local = (ring - origin) @ basis
        tangent_covariance = np.cov(ring_local[:, :2].T, bias=True)
        tangent_eigenvalues = np.linalg.eigvalsh(tangent_covariance)
        tangent_ratio = float(tangent_eigenvalues[0] / max(tangent_eigenvalues[1], 1e-12))
        if tangent_eigenvalues[0] <= 1e-12 or tangent_ratio < cfg.min_tangent_ratio:
            self._disable('degenerate_annulus_footprint')
            return
        scale = max(float(np.sqrt(np.mean(np.sum(ring_local[:, :2] ** 2, axis=1)))), .01)
        uv = ring_local[:, :2] / scale
        spatial = np.exp(-.5 * np.sum(uv ** 2, axis=1))
        plane = _fit(uv, ring_local[:, 2], spatial, False, cfg)
        quadratic = _fit(uv, ring_local[:, 2], spatial, True, cfg) if len(ring) >= cfg.min_quadratic_points else None
        if plane is None:
            self._disable('rank_deficient_plane_design')
            return
        chosen, model = plane, 'plane'
        if quadratic is not None:
            improvement = plane['rms'] - quadratic['rms']
            if (improvement >= cfg.quadratic_min_absolute_improvement_mm
                    and improvement >= cfg.quadratic_min_relative_improvement * max(plane['rms'], cfg.noise_floor_mm)
                    and quadratic['inlier_fraction'] >= cfg.min_inlier_fraction):
                chosen, model = quadratic, 'quadratic'
        self.report.update(plane_sigma_mm=float(plane['sigma']), plane_rms_mm=float(plane['rms']),
                           quadratic_sigma_mm=None if quadratic is None else float(quadratic['sigma']),
                           quadratic_rms_mm=None if quadratic is None else float(quadratic['rms']),
                           fit_sigma_mm=float(chosen['sigma']), inlier_fraction=float(chosen['inlier_fraction']),
                           design_condition=float(chosen['condition']), tangent_ratio=tangent_ratio)
        if chosen['sigma'] > cfg.max_residual_sigma_mm or chosen['inlier_fraction'] < cfg.min_inlier_fraction:
            self._disable('no_reliable_smooth_surface_fit')
            return
        local_noise = _local_noise_sigma(uv, ring_local[:, 2], cfg)
        model_limit = max(cfg.model_residual_floor_mm, cfg.max_model_residual_noise_ratio * local_noise)
        self.report.update(local_measurement_noise_sigma_mm=float(local_noise),
                           model_residual_limit_mm=float(model_limit))
        if chosen['sigma'] > model_limit:
            self._disable('surface_model_misfit_exceeds_observed_local_noise')
            return
        inlier_local = ring_local[chosen['inliers']]
        if len(inlier_local) < cfg.min_points:
            self._disable('insufficient_fit_inliers')
            return
        try:
            support_hull = ConvexHull(inlier_local[:, :2])
        except QhullError:
            self._disable('degenerate_inlier_hull')
            return
        # A single-sided sliver is not evidence for a surface across the hole.
        # Query-specific convex-hull checks below remain stricter than this
        # coarse ring-coverage criterion.
        angles = np.arctan2(inlier_local[:, 1], inlier_local[:, 0])
        sectors = np.floor((angles + np.pi) * (8. / (2. * np.pi))).astype(int) % 8
        occupied = len(np.unique(sectors))
        center_inside = bool(np.all(support_hull.equations[:, 2] <= 1e-9))
        self.report.update(occupied_ring_sectors=int(occupied), roi_center_inside_support=center_inside)
        if occupied < 6 or not center_inside:
            self._disable('annulus_does_not_surround_roi')
            return
        coefficients = np.zeros(6)
        coefficients[:len(chosen['coefficients'])] = chosen['coefficients']
        self._origin, self._basis, self._scale = origin.copy(), basis.copy(), scale
        self._coefficients, self._hull_equations = coefficients, support_hull.equations.copy()
        self._sigma = float(chosen['sigma'])
        self._fit_confidence = (math.exp(-(self._sigma / cfg.max_residual_sigma_mm) ** 2)
                                * chosen['inlier_fraction']
                                * min(1., max(0., (tangent_ratio - cfg.min_tangent_ratio) / .15)))
        self.support_points_mm = ring[chosen['inliers']].copy()
        self._tree = cKDTree(self.support_points_mm)
        self._available = True
        self.report.update(available=True, model=model, reason='supported_observed_annulus',
                           support_points=int(len(self.support_points_mm)),
                           fit_confidence=float(self._fit_confidence))

    def _disable(self, reason):
        self.report.update(available=False, model='unavailable', reason=reason)

    def query(self, query_mm):
        query = np.asarray(query_mm, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != 3 or not np.isfinite(query).all():
            raise ValueError('query_mm must be finite [N,3]')
        count = len(query)
        result = {'anchors_mm': np.zeros((count, 3)), 'normals': np.zeros((count, 3)),
                  'signed_distance_mm': np.zeros(count), 'nearest_distance_mm': np.zeros(count),
                  'confidence': np.zeros(count), 'sigma_mm': np.zeros(count),
                  'oriented': self.camera_position_mm is not None, 'supported': np.zeros(count, dtype=bool)}
        if not count or not self._available:
            return result
        local = (query - self._origin) @ self._basis
        uv = local[:, :2] / self._scale
        u, v = uv.T
        c = self._coefficients
        height = _design(uv, True) @ c
        du = (c[1] + 2. * c[3] * u + c[4] * v) / self._scale
        dv = (c[2] + c[4] * u + 2. * c[5] * v) / self._scale
        anchors = np.column_stack((local[:, :2], height)) @ self._basis.T + self._origin
        normals = np.column_stack((-du, -dv, np.ones(count))) @ self._basis.T
        normals /= np.maximum(np.linalg.norm(normals, axis=1), 1e-12)[:, None]
        signed = np.einsum('ij,ij->i', query - anchors, normals)
        # Real support includes the annulus hole but excludes every point
        # outside the inlier ring's convex hull. No data-dependent GT margin.
        inside = np.all(local[:, :2] @ self._hull_equations[:, :2].T
                        + self._hull_equations[:, 2] <= 1e-9, axis=1)
        supported = (inside & (np.hypot(du, dv) <= self.config.max_slope)
                     & (np.abs(signed) <= self.config.max_query_normal_distance_mm))
        confidence = self._fit_confidence * supported.astype(float)
        if self.camera_position_mm is None:
            confidence.fill(0.)
        else:
            view = self.camera_position_mm - anchors
            cosine = np.einsum('ij,ij->i', view, normals) / np.maximum(np.linalg.norm(view, axis=1), 1e-12)
            confidence *= np.clip((cosine - self.config.min_view_cosine) / .2, 0., 1.)
        nearest, _ = self._tree.query(query)
        result.update(anchors_mm=anchors, normals=normals, signed_distance_mm=signed,
                      confidence=confidence, sigma_mm=np.full(count, self._sigma),
                      nearest_distance_mm=nearest, supported=supported)
        return result
