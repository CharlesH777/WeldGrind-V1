"""Observation-only prototype decoder for the pixel cutting-depth/axis field.

The field predicts total removal, not a per-pass cutting command. A declared
layer depth splits that field into passes. Contact references and downward
cutting commands are saved separately; no GT phase/path/nominal mesh is read.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class FieldPlannerConfig:
    contact_probability: float = .5
    minimum_cut_depth_mm: float = .02
    sampling_radius_fraction: float = .4
    minimum_sample_spacing_mm: float = .1
    max_passes: int = 32
    max_contact_points: int = 32768

    def __post_init__(self):
        if (not 0 <= self.contact_probability <= 1 or self.minimum_cut_depth_mm < 0
                or self.sampling_radius_fraction <= 0 or self.minimum_sample_spacing_mm <= 0
                or self.max_passes < 1 or self.max_contact_points < 2
                or not all(math.isfinite(float(v)) for v in
                           (self.contact_probability, self.minimum_cut_depth_mm,
                            self.sampling_radius_fraction, self.minimum_sample_spacing_mm))):
            raise ValueError('Invalid field planner configuration')


def _unit(value, fallback=None):
    value = np.asarray(value, dtype=float)
    length = np.linalg.norm(value)
    if length < 1e-10:
        if fallback is None:
            raise ValueError('Degenerate observation geometry')
        return _unit(fallback)
    return value / length


def plan_field_program(cut_depth_mm, tangent_axis, contact_probability, *,
                       points_world_mm, base_anchors_world_mm, base_normals_world,
                       observation_valid, fit_supported, roi_mask, process_params,
                       config=None):
    """Decode a dominant-axis raster with honest, variable program length.

    Invalid/concave pixels never request cutting. Gaps become air moves rather
    than cutting bridges. This initial decoder uses one dominant predicted
    axis for a patch; it does not claim arbitrary streamline planning.
    """
    cfg = config if isinstance(config, FieldPlannerConfig) else FieldPlannerConfig(**(config or {}))
    depth = np.asarray(cut_depth_mm, dtype=float)
    if depth.ndim != 2:
        raise ValueError('cut_depth_mm must be [H,W]')
    shape = depth.shape
    axis = np.asarray(tangent_axis, dtype=float)
    if axis.shape == (2,) + shape:
        axis = axis.transpose(1, 2, 0)
    probability = np.asarray(contact_probability, dtype=float)
    points = np.asarray(points_world_mm, dtype=float)
    anchors = np.asarray(base_anchors_world_mm, dtype=float)
    normals = np.asarray(base_normals_world, dtype=float)
    if (axis.shape != shape + (2,) or probability.shape != shape or points.shape != shape + (3,)
            or anchors.shape != points.shape or normals.shape != points.shape):
        raise ValueError('Field and observed geometry grids must align')
    valid = np.asarray(observation_valid, bool) & np.asarray(fit_supported, bool)
    roi = np.asarray(roi_mask, bool)
    if valid.shape != shape or roi.shape != shape:
        raise ValueError('Observation/fit/ROI masks must align')
    if (not np.isfinite(depth).all() or not np.isfinite(axis).all() or not np.isfinite(probability).all()
            or (depth < 0).any() or ((probability < 0) | (probability > 1)).any()):
        raise ValueError('Predictions must be finite, nonnegative depth and probabilities in [0,1]')
    valid &= np.isfinite(points).all(-1) & np.isfinite(anchors).all(-1) & np.isfinite(normals).all(-1)
    valid &= np.linalg.norm(normals, axis=-1) > .9
    parameters = np.asarray(process_params, float)
    if (parameters.shape != (4,) or not np.isfinite(parameters).all()
            or (parameters[:3] <= 0).any() or not 0 <= parameters[3] < 1):
        raise ValueError('process_params=[radius,layer_depth,clearance,overlap] in mm')
    radius, layer, clearance, overlap = parameters
    normal_grid = normals / np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
    observed_height = np.einsum('...i,...i->...', points - anchors, normal_grid)
    clipped_depth = np.minimum(depth, np.maximum(observed_height, 0.))
    eligible = valid & roi & (probability >= cfg.contact_probability) & (clipped_depth >= cfg.minimum_cut_depth_mm)
    empty = {'trajectory_reference_mm': np.empty((0, 3)), 'trajectory_command_mm': np.empty((0, 3)),
             'predicted_phases': np.empty(0, np.int64), 'cut_depth_mm': np.empty(0),
             'outward_normals_world': np.empty((0, 3)), 'velocity_direction_world': np.empty((0, 3)),
             'end_token_index': None,
             'report': {'status': 'no_supported_cut', 'uses_ground_truth': False, 'speed_magnitude_supervised': False}}
    if not eligible.any():
        return empty
    mean_normal = _unit(normal_grid[eligible].mean(0))
    # Image-axis derivatives come from calibrated observed base positions.
    horizontal = valid[:, 1:] & valid[:, :-1]
    vertical = valid[1:] & valid[:-1]
    du = (anchors[:, 1:] - anchors[:, :-1])[horizontal]
    dv = (anchors[1:] - anchors[:-1])[vertical]
    # A crop can be enlarged with nearest-neighbour calibrated geometry.
    # Repeated source pixels carry no spatial derivative information.
    du = du[np.linalg.norm(du, axis=1) > 1e-9]
    dv = dv[np.linalg.norm(dv, axis=1) > 1e-9]
    if not len(du) or not len(dv):
        raise ValueError('Insufficient adjacent observation support for pixel/world axes')
    mean_du, mean_dv = np.median(du, axis=0), np.median(dv, axis=0)
    weighted_axis = (axis[eligible] * probability[eligible, None]).sum(0)
    if np.linalg.norm(weighted_axis) < 1e-6:
        empty['report']['status'] = 'unresolved_predicted_axis'
        return empty
    angle = .5 * math.atan2(weighted_axis[1], weighted_axis[0])
    u = _unit(mean_du * math.cos(angle) + mean_dv * math.sin(angle))
    u = _unit(u - (u @ mean_normal) * mean_normal)
    v = _unit(np.cross(mean_normal, u))
    origin = anchors[eligible].mean(0)
    projected = np.stack(((anchors - origin) @ u, (anchors - origin) @ v), -1)
    supported_indices = np.flatnonzero(valid.ravel())
    tree = cKDTree(projected.reshape(-1, 2)[supported_indices])
    pixel_spacing = float(np.median(np.r_[np.linalg.norm(du, axis=1), np.linalg.norm(dv, axis=1)]))
    spacing = max(cfg.minimum_sample_spacing_mm, radius * cfg.sampling_radius_fraction)
    lane_spacing = 2 * radius * (1 - overlap)
    passes = int(np.ceil(clipped_depth[eligible].max() / layer))
    if passes > cfg.max_passes:
        raise ValueError('Predicted field exceeds declared max_passes; no silent truncation')
    strokes = []
    for pass_index in range(passes):
        active = eligible & (clipped_depth > pass_index * layer + 1e-10)
        if not active.any():
            continue
        uv = projected[active]
        lane_values = np.arange(uv[:, 1].min(), uv[:, 1].max() + .5 * lane_spacing, lane_spacing)
        if not len(lane_values):
            lane_values = np.array([uv[:, 1].mean()])
        for lane_index, lane_value in enumerate(lane_values):
            travel = np.arange(uv[:, 0].min(), uv[:, 0].max() + .5 * spacing, spacing)
            queries = np.column_stack((travel, np.full(len(travel), lane_value)))
            distance, nearest = tree.query(queries)
            flat_ids = supported_indices[nearest]
            selected = (distance <= max(1.25 * pixel_spacing, .55 * spacing)) & active.ravel()[flat_ids]
            indices = flat_ids if lane_index % 2 == 0 else flat_ids[::-1]
            selected = selected if lane_index % 2 == 0 else selected[::-1]
            boundaries = np.r_[0, np.flatnonzero(np.diff(selected.astype(int))) + 1, len(selected)]
            for start, stop in zip(boundaries[:-1], boundaries[1:]):
                if not selected[start]:
                    continue
                chosen = indices[start:stop]
                chosen = chosen[np.r_[True, chosen[1:] != chosen[:-1]]]
                if len(chosen) < 2:
                    continue
                h = observed_height.ravel()[chosen]
                n = normal_grid.reshape(-1, 3)[chosen]
                base = anchors.reshape(-1, 3)[chosen]
                remaining = clipped_depth.ravel()[chosen] - pass_index * layer
                cut = np.minimum(layer, remaining)
                reference = base + np.maximum(h - pass_index * layer, 0.)[:, None] * n
                command = reference - cut[:, None] * n
                strokes.append((reference, command, n, cut, pass_index))
    if not strokes:
        empty['report']['status'] = 'no_supported_contiguous_stroke'
        return empty
    if sum(len(row[0]) for row in strokes) > cfg.max_contact_points:
        raise ValueError('Predicted field exceeds declared point budget; no silent truncation')
    highest = float(np.max(points[valid] @ mean_normal)) + clearance
    references, commands, outward, phases, cuts, pass_ids = [], [], [], [], [], []
    for stroke_index, (reference, command, normal, cut, pass_index) in enumerate(strokes):
        if stroke_index:
            previous = references[-1][-1]
            previous_command = commands[-1][-1]
            previous_normal = outward[-1][-1]
            up1 = previous + (highest - previous @ mean_normal) * mean_normal
            up2 = reference[0] + (highest - reference[0] @ mean_normal) * mean_normal
            for phase, segment, command_segment, segment_normals in (
                    (1, np.stack((previous, up1)), np.stack((previous_command, up1)), np.stack((previous_normal, previous_normal))),
                    (2, np.stack((up1, up2)), np.stack((up1, up2)), np.stack((previous_normal, normal[0]))),
                    (3, np.stack((up2, reference[0])), np.stack((up2, command[0])), np.stack((normal[0], normal[0])))):
                references.append(segment); commands.append(command_segment); outward.append(segment_normals)
                phases.append(np.full(2, phase, np.int64)); cuts.append(np.zeros(2)); pass_ids.append(np.full(2, pass_index))
        references.append(reference); commands.append(command); outward.append(normal)
        phases.append(np.zeros(len(reference), np.int64)); cuts.append(cut); pass_ids.append(np.full(len(reference), pass_index))
    references, commands = np.concatenate(references), np.concatenate(commands)
    labels = np.concatenate(phases)
    directions = np.zeros_like(references)
    for index in range(len(references) - 1):
        if labels[index] == labels[index + 1]:
            delta = references[index + 1] - references[index]
            if np.linalg.norm(delta) > 1e-10:
                directions[index] = delta / np.linalg.norm(delta)
    if len(directions) > 1:
        directions[-1] = directions[-2]
    return {'trajectory_reference_mm': references, 'trajectory_command_mm': commands,
            'predicted_phases': labels, 'cut_depth_mm': np.concatenate(cuts),
            'outward_normals_world': np.concatenate(outward), 'velocity_direction_world': directions,
            'pass_indices': np.concatenate(pass_ids), 'end_token_index': len(references) - 1,
            'report': {'status': 'planned', 'uses_ground_truth': False, 'units': 'mm',
                       'field_representation': 'total_removal_depth_and_unoriented_pixel_tangent_axis',
                       'speed_magnitude_supervised': False, 'passes': passes, 'contact_strokes': len(strokes),
                       'point_count': len(references), 'fixed_1024_claimed': False,
                       'initial_approach_or_final_retract_added': False,
                       'depth_clipped_to_observed_positive_height_pixels': int(np.count_nonzero(depth > np.maximum(observed_height, 0.))),
                       'clearance_plane_height_mm': highest, 'tool_radius_mm': radius,
                       'layer_depth_mm': layer, 'overlap': overlap,
                       'limitations': ['dominant-axis local raster, not arbitrary learned streamline topology',
                                       'spatial samples have no calibrated feed-rate timestamps',
                                       'no full tool/cell collision or force validation']}}
