"""Robust local planes from observed SAM-selected point clouds, in millimetres.

No target trajectory, machining chart or target surface enters this module.
Planes describe the *observed* surface, not the material surface after a pass.
Normals have an outward interpretation only when an observation camera is given.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SurfaceConfig:
    voxel_size_mm: float = .25
    neighbors: int = 24
    radius_mm: float = 2.5
    outlier_mad_scale: float = 4.
    min_neighbors: int = 8
    max_normal_sigma_mm: float = .35
    max_planarity_ratio: float = .25
    min_tangent_ratio: float = .02
    robust_iterations: int = 3

    def __post_init__(self):
        for name in ("voxel_size_mm", "radius_mm", "outlier_mad_scale",
                     "max_normal_sigma_mm", "max_planarity_ratio", "min_tangent_ratio"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
                raise ValueError(f"{name} must be finite and positive")
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("neighbors", "min_neighbors", "robust_iterations"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_neighbors < 3 or self.neighbors < self.min_neighbors:
            raise ValueError("neighbors >= min_neighbors >= 3 is required")
        if self.max_planarity_ratio >= 1 or self.min_tangent_ratio >= 1:
            raise ValueError("planarity and tangent ratios must be less than one")


def _mad(values):
    median = np.median(values)
    return median, 1.4826 * np.median(np.abs(values - median))


def _voxel_representatives(xyz, size):
    """Select actual observed rows nearest each voxel centroid; never synthesize XYZ."""
    coordinates = np.floor((xyz - xyz.min(axis=0)) / size).astype(np.int64)
    _, inverse, counts = np.unique(coordinates, axis=0, return_inverse=True, return_counts=True)
    means = np.column_stack([np.bincount(inverse, weights=xyz[:, axis]) / counts for axis in range(3)])
    distances = np.sum((xyz - means[inverse]) ** 2, axis=1)
    order = np.lexsort((distances, inverse))
    starts = np.r_[0, np.flatnonzero(np.diff(inverse[order])) + 1]
    return xyz[order[starts]]


class ObservedSurface:
    """Denoised observations and a confidence-gated robust local plane query.

    ``anchors_mm`` are fitted plane centres. ``signed_distance_mm`` equals
    dot(query - anchor, normal). With a camera, positive distance faces the
    camera; without a camera its sign has no physical inside/outside meaning.
    Unsupported queries remain finite but have zero confidence.
    """

    def __init__(self, points_mm, roi_mask=None, *, camera_position_mm=None, config=None):
        self.config = config if isinstance(config, SurfaceConfig) else SurfaceConfig(**(config or {}))
        points = np.asarray(points_mm, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] not in (3, 6) or not len(points):
            raise ValueError("points_mm must be nonempty [N,3] or [N,6]")
        if roi_mask is None:
            mask = np.ones(len(points), dtype=bool)
        else:
            mask = np.asarray(roi_mask)
            if mask.shape != (len(points),) or mask.dtype != np.bool_:
                raise ValueError("roi_mask must be a boolean [N] array")
        selected = points[mask, :3]
        finite = np.isfinite(selected).all(axis=1)
        xyz = selected[finite]
        if not len(xyz):
            raise ValueError("SAM ROI contains no finite observed XYZ points")
        self.camera_position_mm = None
        if camera_position_mm is not None:
            camera = np.asarray(camera_position_mm, dtype=np.float64)
            if camera.shape != (3,) or not np.isfinite(camera).all():
                raise ValueError("camera_position_mm must be finite [3]")
            self.camera_position_mm = camera.copy()
        unique = np.unique(xyz, axis=0)
        voxels = _voxel_representatives(unique, self.config.voxel_size_mm)
        retained = np.ones(len(voxels), dtype=bool)
        # A density ratio, in addition to a global robust distance test, keeps
        # valid but sparsely sampled patches. A lone distant point has dense
        # neighbours with much smaller neighbour distances and is rejected.
        if len(voxels) > self.config.min_neighbors:
            tree = cKDTree(voxels)
            distances, indices = tree.query(voxels, k=min(self.config.neighbors + 1, len(voxels)))
            local_distance = np.median(distances[:, 1:], axis=1)
            neighbor_distance = np.median(local_distance[indices[:, 1:]], axis=1)
            ratio = local_distance / np.maximum(neighbor_distance, 1e-12)
            distance_median, distance_mad = _mad(local_distance)
            ratio_median, ratio_mad = _mad(ratio)
            distance_limit = max(distance_median + self.config.outlier_mad_scale * distance_mad,
                                 2.5 * distance_median)
            ratio_limit = max(2.5, ratio_median + self.config.outlier_mad_scale * ratio_mad)
            retained = ~((local_distance > distance_limit) & (ratio > ratio_limit))
        self.points_mm = np.ascontiguousarray(voxels[retained])
        if not len(self.points_mm):
            raise ValueError("No observed surface remains after outlier filtering")
        self._tree = cKDTree(self.points_mm)
        self.report = {
            "input_points": int(len(points)), "roi_points": int(mask.sum()),
            "nonfinite_roi_points_removed": int((~finite).sum()),
            "duplicate_points_removed": int(len(xyz) - len(unique)),
            "voxel_points": int(len(voxels)),
            "statistical_outliers_removed": int((~retained).sum()),
            "retained_points": int(len(self.points_mm)),
            "normal_orientation": "toward_observation_camera" if self.camera_position_mm is not None else "unoriented",
            "source": "observed_point_cloud_only", "units": "mm",
        }

    def query(self, query_mm):
        query = np.asarray(query_mm, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != 3 or not np.isfinite(query).all():
            raise ValueError("query_mm must be finite [M,3]")
        count = len(query)
        result = {
            "anchors_mm": np.zeros((count, 3)), "normals": np.zeros((count, 3)),
            "signed_distance_mm": np.zeros(count), "nearest_distance_mm": np.zeros(count),
            "confidence": np.zeros(count), "sigma_mm": np.zeros(count),
            "oriented": self.camera_position_mm is not None,
        }
        if not count:
            return result
        nearest_distances, nearest = self._tree.query(query)
        result["nearest_distance_mm"] = nearest_distances
        result["anchors_mm"] = self.points_mm[nearest].copy()
        if len(self.points_mm) < self.config.min_neighbors:
            return result
        k = min(self.config.neighbors, len(self.points_mm))
        # Query neighbourhoods around the nearest observation, so the normal
        # estimate does not become a function of the trajectory's normal offset.
        distances, indices = self._tree.query(self.points_mm[nearest], k=k)
        neighborhoods = self.points_mm[indices]
        support = distances <= self.config.radius_mm
        bandwidth = np.maximum(np.minimum(distances[:, -1], self.config.radius_mm), self.config.voxel_size_mm)
        spatial_weights = np.exp(-.5 * (distances / bandwidth[:, None]) ** 2) * support
        weights = spatial_weights.copy()
        sigma = np.zeros(count)
        for iteration in range(self.config.robust_iterations):
            normalizer = np.maximum(weights.sum(axis=1), 1e-12)
            anchors = np.einsum("mk,mkj->mj", weights, neighborhoods) / normalizer[:, None]
            centered = neighborhoods - anchors[:, None]
            covariance = np.einsum("mk,mki,mkj->mij", weights, centered, centered) / normalizer[:, None, None]
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 0)
            normals = eigenvectors[:, :, 0]
            residual = np.einsum("mkj,mj->mk", centered, normals)
            # Unsupported neighbours carry zero spatial weight. NaN-median
            # ignores those rows; each neighbourhood includes its central point.
            supported_residual = np.where(support, residual, np.nan)
            residual_median = np.nanmedian(supported_residual, axis=1)
            robust_sigma = 1.4826 * np.nanmedian(np.abs(supported_residual - residual_median[:, None]), axis=1)
            sigma = np.sqrt(np.sum(weights * residual ** 2, axis=1) / normalizer)
            if iteration + 1 < self.config.robust_iterations:
                huber = np.minimum(1., 2.5 * np.maximum(robust_sigma[:, None], .005) / np.maximum(np.abs(residual), 1e-12))
                weights = spatial_weights * huber
        if self.camera_position_mm is not None:
            view = self.camera_position_mm - anchors
            dot = np.einsum("ij,ij->i", normals, view)
            normals *= np.where(dot < 0, -1., 1.)[:, None]
            view_cosine = np.abs(dot) / np.maximum(np.linalg.norm(view, axis=1), 1e-12)
        else:
            # A deterministic sign is useful numerically but is never called
            # outward. It is not a substitute for camera calibration.
            major = np.argmax(np.abs(normals), axis=1)
            normals *= np.where(normals[np.arange(count), major] < 0, -1., 1.)[:, None]
            view_cosine = np.ones(count)
        plane_ratio = eigenvalues[:, 0] / np.maximum(eigenvalues[:, 1], 1e-12)
        tangent_ratio = eigenvalues[:, 1] / np.maximum(eigenvalues[:, 2], 1e-12)
        confidence = np.clip(1. - plane_ratio / self.config.max_planarity_ratio, 0., 1.)
        confidence *= np.clip(1. - sigma / self.config.max_normal_sigma_mm, 0., 1.)
        confidence *= np.clip((view_cosine - .05) / .15, 0., 1.)
        valid = ((support.sum(axis=1) >= self.config.min_neighbors)
                 & (nearest_distances <= self.config.radius_mm)
                 & (tangent_ratio >= self.config.min_tangent_ratio)
                 & (eigenvalues[:, 1] > 1e-12))
        # Reject tangential extrapolation beyond the neighbourhood footprint.
        # Eight support directions preserve corners better than a spherical
        # distance gate alone. Half a voxel permits ordinary sampling gaps.
        tangent_basis = eigenvectors[:, :, 1:]
        tangent_neighbors = np.einsum("mkj,mjl->mkl", centered, tangent_basis)
        tangent_query = np.einsum("mj,mjl->ml", query - anchors, tangent_basis)
        for angle in np.arange(8) * (np.pi / 4):
            direction = np.array([np.cos(angle), np.sin(angle)])
            footprint = tangent_neighbors @ direction
            upper = np.max(np.where(support, footprint, -np.inf), axis=1)
            query_extent = tangent_query @ direction
            valid &= query_extent <= upper + .5 * self.config.voxel_size_mm
        confidence *= valid
        result.update(anchors_mm=anchors, normals=normals,
                      signed_distance_mm=np.einsum("ij,ij->i", query - anchors, normals),
                      confidence=confidence, sigma_mm=sigma)
        return result
