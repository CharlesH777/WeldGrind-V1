"""Observation-only, bounded spatial filtering of complete grinding programs.

The point index is NOT execution time. Second/third differences regularize
spatial jitter; they are not a claim of robot acceleration/jerk feasibility.
No GT path, segment, pass, chart normal, or target surface is accepted here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

import numpy as np
from scipy import sparse
from scipy.ndimage import median_filter
from scipy.sparse.linalg import splu

from .pointcloud_surface import ObservedSurface, SurfaceConfig
from .background_surface import BackgroundSurfaceConfig, ObservedBackgroundSurface


@dataclass(frozen=True)
class PhysicalFilterConfig:
    smoothness_weight: float = 4.
    jerk_weight: float = .5
    surface_weight: float = 2.
    surface_tolerance_mm: float = .3
    max_removal_depth_mm: float | None = None
    max_correction_mm: float = 2.
    huber_delta_mm: float = .75
    outlier_mad_scale: float = 4.
    phase_switch_cost: float = 1.5
    contact_probability_threshold: float = .65
    corner_angle_degrees: float = 65.
    corner_window: int = 3
    boundary_join_distance_mm: float = .5
    boundary_join_weight: float = 4.
    admm_rho: float = 4.
    max_iterations: int = 600
    tolerance_mm: float = 1e-5
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    background: BackgroundSurfaceConfig | None = None
    background_weight: float = 8.
    background_tolerance_mm: float = .15
    preserve_air_normal_height: bool = True
    air_normal_tolerance_mm: float = .15
    global_continuity_guard: bool = False

    def __post_init__(self):
        nonnegative = {"smoothness_weight", "jerk_weight", "surface_weight", "surface_tolerance_mm",
                       "phase_switch_cost", "boundary_join_distance_mm", "boundary_join_weight",
                       "background_weight", "background_tolerance_mm", "air_normal_tolerance_mm"}
        positive = {"max_correction_mm", "huber_delta_mm", "outlier_mad_scale", "admm_rho", "tolerance_mm"}
        for name in nonnegative | positive | {"contact_probability_threshold", "corner_angle_degrees"}:
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
                    or not math.isfinite(float(value))):
                raise ValueError(f"{name} must be a finite number")
            if name in nonnegative and value < 0 or name in positive and value <= 0:
                raise ValueError(f"{name} is outside its valid range")
        if not 0 <= self.contact_probability_threshold <= 1 or not 0 < self.corner_angle_degrees < 180:
            raise ValueError("contact probability must be in [0,1] and corner angle in (0,180)")
        for name in ("corner_window", "max_iterations"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.corner_window < 3:
            raise ValueError("corner_window must be at least three for independent side fits")
        if self.max_removal_depth_mm is not None:
            value = self.max_removal_depth_mm
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
                    or not math.isfinite(float(value)) or value < 0):
                raise ValueError("max_removal_depth_mm must be None or finite and nonnegative")
        if isinstance(self.surface, dict):
            object.__setattr__(self, "surface", SurfaceConfig(**self.surface))
        elif not isinstance(self.surface, SurfaceConfig):
            raise ValueError("surface must be SurfaceConfig or its configuration dict")
        if isinstance(self.background, dict):
            object.__setattr__(self, "background", BackgroundSurfaceConfig(**self.background))
        elif self.background is not None and not isinstance(self.background, BackgroundSurfaceConfig):
            raise ValueError("background must be None, BackgroundSurfaceConfig or its configuration dict")
        if not isinstance(self.preserve_air_normal_height, bool):
            raise ValueError("preserve_air_normal_height must be a boolean")
        if not isinstance(self.global_continuity_guard, bool):
            raise ValueError("global_continuity_guard must be a boolean")


def _xyz(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) < 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite [N,3] with N >= 2")
    return result.copy()


def _phases(length, phase_logits, predicted_phases, switch_cost):
    if phase_logits is not None:
        logits = np.asarray(phase_logits, dtype=np.float64)
        if logits.shape != (length, 4) or not np.isfinite(logits).all():
            raise ValueError("phase_logits must be finite [N,4]")
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
    else:
        probabilities = np.full((length, 4), .25)
    if predicted_phases is not None:
        labels = np.asarray(predicted_phases)
        if (labels.shape != (length,) or not np.issubdtype(labels.dtype, np.integer)
                or (labels < 0).any() or (labels > 3).any()):
            raise ValueError("predicted_phases must be integer [N] in [0,3]")
        if phase_logits is None:
            # Explicit but finite confidence; a label does not imply certainty.
            probabilities.fill(.05)
            probabilities[np.arange(length), labels] = .85
    if phase_logits is None and predicted_phases is None:
        return np.full(length, -1, dtype=np.int64), probabilities
    costs = -np.log(np.maximum(probabilities, np.finfo(float).tiny))
    accumulated = costs[0].copy()
    back = np.zeros((length, 4), dtype=np.int64)
    transitions = switch_cost * (1 - np.eye(4))
    # Symmetric Potts penalty: no required CONTACT/RETRACT/etc. template.
    for index in range(1, length):
        scores = accumulated[:, None] + transitions
        back[index] = scores.argmin(axis=0)
        accumulated = costs[index] + scores.min(axis=0)
    labels = np.empty(length, dtype=np.int64)
    labels[-1] = accumulated.argmin()
    for index in range(length - 2, -1, -1):
        labels[index] = back[index + 1, labels[index + 1]]
    return labels, probabilities


def _regularizer(path, phases, config):
    length = len(path)
    differences = np.diff(path, axis=0)
    step = np.linalg.norm(differences, axis=1)
    deviation = np.zeros(length)
    deviation[1:-1] = np.linalg.norm(path[1:-1] - .5 * (path[:-2] + path[2:]), axis=1)
    med = median_filter(deviation, size=9, mode="nearest")
    mad = 1.4826 * median_filter(np.abs(deviation - med), size=9, mode="nearest")
    reversal = np.zeros(length, dtype=bool)
    cosine = np.einsum("ij,ij->i", differences[:-1], differences[1:]) / np.maximum(step[:-1] * step[1:], 1e-12)
    reversal[1:-1] = cosine < -.5
    spike = reversal & (deviation > np.maximum(config.huber_delta_mm, med + config.outlier_mad_scale * mad))
    fidelity = np.ones(length)
    fidelity[spike] = np.maximum(.05, np.minimum(1., config.huber_delta_mm / deviation[spike]))

    corners = np.zeros(length, dtype=bool)
    width = config.corner_window
    # Fit each side separately, excluding the centre token. Averaging across
    # a junction can erase a real right angle; raw chords misidentify isolated
    # spikes as corners. Reject directions whose signal is dominated by noise.
    fit_index = np.arange(width, dtype=float)
    fit_index -= fit_index.mean()
    fit_denominator = float(fit_index @ fit_index)
    for index in range(width, length - width):
        if (width < 3 or spike[index]
                or not np.all(phases[index - width:index + width + 1] == phases[index])):
            continue
        side_vectors = []
        for side in (path[index - width:index], path[index + 1:index + width + 1]):
            center = side.mean(axis=0)
            slope = (fit_index[:, None] * (side - center)).sum(axis=0) / fit_denominator
            noise_rms = float(np.sqrt(np.mean(np.sum((side - center - fit_index[:, None] * slope) ** 2, axis=1))))
            chord = slope * (width - 1)
            if np.linalg.norm(chord) <= max(1e-6, 3 * noise_rms):
                break
            side_vectors.append(chord)
        if len(side_vectors) != 2:
            continue
        before, after = side_vectors
        denom = np.linalg.norm(before) * np.linalg.norm(after)
        if denom > 1e-12 and np.dot(before, after) / denom < math.cos(math.radians(config.corner_angle_degrees)):
            corners[index] = True
    # Never smooth across a decoded phase transition or a geometrical corner.
    # Preserve fidelity at corners rather than fitting a straight shortcut.
    fidelity[corners] *= 8.
    hessian = sparse.diags(fidelity, format="csc")
    for order, coefficients, weight in ((2, [1., -2., 1.], config.smoothness_weight),
                                         (3, [-1., 3., -3., 1.], config.jerk_weight)):
        starts = [i for i in range(length - order)
                  if np.all(phases[i:i + order + 1] == phases[i]) and not corners[i + 1:i + order].any()]
        if starts and weight:
            rows = np.repeat(np.arange(len(starts)), order + 1)
            columns = np.asarray(starts)[:, None] + np.arange(order + 1)
            matrix = sparse.coo_matrix((np.tile(coefficients, len(starts)), (rows, columns.ravel())), shape=(len(starts), length)).tocsc()
            hessian = hessian + weight * (matrix.T @ matrix)
    joins = np.flatnonzero((phases[:-1] != phases[1:]) & (step <= config.boundary_join_distance_mm))
    if len(joins) and config.boundary_join_weight:
        rows = np.repeat(np.arange(len(joins)), 2)
        matrix = sparse.coo_matrix((np.tile([-1., 1.], len(joins)), (rows, np.column_stack((joins, joins + 1)).ravel())), shape=(len(joins), length)).tocsc()
        hessian = hessian + config.boundary_join_weight * (matrix.T @ matrix)
    return hessian, fidelity, spike, corners, joins


def continuity_metrics(trajectory_mm, phases=None):
    """Spatial diagnostics only, in mm per index difference, never per second."""
    path = _xyz(trajectory_mm, "trajectory_mm")
    labels = None if phases is None else np.asarray(phases)
    if labels is not None and labels.shape != (len(path),):
        raise ValueError("phases must align with trajectory_mm")
    result = {"domain": "spatial_sequence_not_execution_time", "points": len(path)}
    for order, name in ((1, "step"), (2, "second_difference"), (3, "third_difference")):
        values = np.linalg.norm(np.diff(path, n=order, axis=0), axis=1)
        result[name + "_rms_mm"] = float(np.sqrt(np.mean(values ** 2))) if len(values) else 0.
        result[name + "_max_mm"] = float(values.max()) if len(values) else 0.
        result[name + "_p95_mm"] = float(np.percentile(values, 95)) if len(values) else 0.
        if labels is not None:
            same = np.array([np.all(labels[i:i + order + 1] == labels[i]) for i in range(len(values))])
            result[name + "_within_phase_rms_mm"] = float(np.sqrt(np.mean(values[same] ** 2))) if same.any() else 0.
    return result


def _project_correction(offset, radius, normals, air_mask, air_tolerance):
    """Exact Euclidean projection onto a ball intersected with an air slab.

    Both convex sets are centred at the raw point. Sequential clipping is
    feasible but is not their intersection's proximal operator for ADMM.
    """
    length = np.linalg.norm(offset, axis=1)
    projected = offset * np.minimum(1., radius / np.maximum(length, 1e-12))[:, None]
    if normals is not None and air_mask.any():
        projected_normal = np.einsum("ij,ij->i", projected, normals)
        active = air_mask & (np.abs(projected_normal) > air_tolerance)
        if active.any():
            original_normal = np.einsum("ij,ij->i", offset, normals)
            clipped_normal = np.clip(original_normal, -air_tolerance, air_tolerance)
            tangent = offset - original_normal[:, None] * normals
            tangent_radius = np.sqrt(np.maximum(radius ** 2 - clipped_normal ** 2, 0.))
            tangent_scale = np.minimum(1., tangent_radius / np.maximum(np.linalg.norm(tangent, axis=1), 1e-12))
            projected[active] = (tangent * tangent_scale[:, None] + clipped_normal[:, None] * normals)[active]
    return projected


def _guard_continuity(raw, candidate):
    """Choose a GT-free convex blend that decreases both global D2 and D3.

    Each squared difference norm is a convex quadratic in the blend fraction.
    Staying before both one-dimensional minimizers prevents phase-boundary
    artifacts from increasing either norm. Convex blending preserves the
    correction balls, air slabs, and original endpoints.
    """
    difference = candidate - raw
    fraction = 1.
    for order in (2, 3):
        original_difference = np.diff(raw, n=order, axis=0)
        update_difference = np.diff(difference, n=order, axis=0)
        squared_update = float(np.sum(update_difference ** 2))
        if squared_update > 1e-24:
            slope_half = float(np.sum(original_difference * update_difference))
            fraction = min(fraction, max(0., -slope_half / squared_update))
    return raw + fraction * difference, float(fraction)


def filter_trajectory(trajectory_mm, points_mm, *, roi_mask=None, phase_logits=None,
                      predicted_phases=None, camera_position_mm=None, config=None):
    """Filter an ordered predicted program using only observations and predictions.

    The default surface term corrects high-confidence contact points on the
    camera-facing side of a fitted surface, outside a tolerance. It does NOT
    pull later-pass points out of already removed material. An inward depth
    limit requires an explicitly supplied material-removal allowance.

    Optional non-ROI background fitting adds a nominal-base lower boundary,
    retaining the original observed defect surface as an upper boundary.
    Later contact passes may occupy different heights inside this corridor.
    Air points with reliable background support preserve their original local
    normal height within a noise tolerance while their tangential coordinates
    may be smoothed. Zero air tolerance preserves that height exactly.

    With no camera or no predicted phase, the oriented surface terms are off.
    Endpoints and a per-point correction ball are hard constraints. The result
    contains the feasible ADMM ball variable even if convergence is incomplete.
    """
    cfg = config if isinstance(config, PhysicalFilterConfig) else PhysicalFilterConfig(**(config or {}))
    raw = _xyz(trajectory_mm, "trajectory_mm")
    phases, probabilities = _phases(len(raw), phase_logits, predicted_phases, cfg.phase_switch_cost)
    surface = ObservedSurface(points_mm, roi_mask, camera_position_mm=camera_position_mm, config=cfg.surface)
    planes = surface.query(raw)
    confidence = np.asarray(planes["confidence"])
    eligible = ((phases == 0) & (probabilities[:, 0] >= cfg.contact_probability_threshold)
                & (confidence > 0) & bool(planes["oriented"]))
    strengths = cfg.surface_weight * confidence * probabilities[:, 0] * eligible
    background_surface = None
    background_planes = None
    background_strengths = np.zeros(len(raw))
    background_eligible = np.zeros(len(raw), dtype=bool)
    air_height_preserved = np.zeros(len(raw), dtype=bool)
    inconsistent_corridor = np.zeros(len(raw), dtype=bool)
    if cfg.background is not None:
        background_surface = ObservedBackgroundSurface(
            points_mm, roi_mask, camera_position_mm=camera_position_mm, config=cfg.background)
        background_planes = background_surface.query(raw)
        background_confidence = np.asarray(background_planes["confidence"])
        supported = (background_confidence > 0) & bool(background_planes["oriented"])
        base_normals = np.asarray(background_planes["normals"])
        base_anchors = np.asarray(background_planes["anchors_mm"])
        alignment = np.einsum("ij,ij->i", base_normals, planes["normals"])
        # Intersect each upper tangent plane along the nominal-base normal.
        # Contradictory observed evidence must not become competing penalties.
        upper_height = np.einsum("ij,ij->i", planes["anchors_mm"] - base_anchors,
                                 planes["normals"]) / np.maximum(alignment, 1e-12)
        inconsistent_corridor = supported & (confidence > 0) & (
            (alignment < .5) | (upper_height < -cfg.background_tolerance_mm - cfg.surface_tolerance_mm))
        supported &= ~inconsistent_corridor
        strengths[inconsistent_corridor] = 0.
        eligible[inconsistent_corridor] = False
        background_eligible = ((phases == 0) &
                               (probabilities[:, 0] >= cfg.contact_probability_threshold) & supported)
        background_strengths = (cfg.background_weight * background_confidence *
                                probabilities[:, 0] * background_eligible)
        air_height_preserved = supported & np.isin(phases, [1, 2, 3]) & cfg.preserve_air_normal_height
    use_background_prox = bool(np.any(background_strengths))
    hessian, fidelity, spikes, corners, joins = _regularizer(raw, phases, cfg)
    rho = cfg.admm_rho
    split_count = 3 if use_background_prox else 2
    solve = splu((hessian + split_count * rho * sparse.eye(len(raw), format="csc")).tocsc()).solve
    rhs_prior = fidelity[:, None] * raw
    normals = planes["normals"]
    anchors = planes["anchors_mm"]
    ball = raw.copy()
    plane = raw.copy()
    dual_ball = np.zeros_like(raw)
    dual_plane = np.zeros_like(raw)
    base_plane = raw.copy()
    dual_base = np.zeros_like(raw)
    residual = dual_residual = 0.
    converged = False
    for iteration in range(cfg.max_iterations):
        split_sum = ball - dual_ball + plane - dual_plane
        if use_background_prox:
            split_sum = split_sum + base_plane - dual_base
        state = solve(rhs_prior + rho * split_sum)
        previous_ball, previous_plane = ball.copy(), plane.copy()
        previous_base = base_plane.copy()
        offset = state + dual_ball - raw
        ball = raw + _project_correction(offset, cfg.max_correction_mm,
                                         base_normals if background_planes is not None else None,
                                         air_height_preserved, cfg.air_normal_tolerance_mm)
        ball[[0, -1]] = raw[[0, -1]]
        proposal = state + dual_plane
        signed = np.einsum("ij,ij->i", proposal - anchors, normals)
        violation = np.maximum(signed - cfg.surface_tolerance_mm, 0.)
        if cfg.max_removal_depth_mm is not None:
            violation += np.minimum(signed + cfg.max_removal_depth_mm, 0.)
        plane = proposal - ((strengths / (rho + strengths)) * violation)[:, None] * normals
        if use_background_prox:
            base_proposal = state + dual_base
            base_signed = np.einsum("ij,ij->i", base_proposal - base_anchors, base_normals)
            base_violation = np.minimum(base_signed + cfg.background_tolerance_mm, 0.)
            base_plane = base_proposal - ((background_strengths / (rho + background_strengths)) *
                                          base_violation)[:, None] * base_normals
            dual_base += state - base_plane
        dual_ball += state - ball
        dual_plane += state - plane
        residual = float(max(np.linalg.norm(state - ball, axis=1).max(), np.linalg.norm(state - plane, axis=1).max()))
        dual_residual = float(rho * max(np.linalg.norm(ball - previous_ball, axis=1).max(), np.linalg.norm(plane - previous_plane, axis=1).max()))
        if use_background_prox:
            residual = max(residual, float(np.linalg.norm(state - base_plane, axis=1).max()))
            dual_residual = max(dual_residual, float(rho * np.linalg.norm(base_plane - previous_base, axis=1).max()))
        if max(residual, dual_residual) <= cfg.tolerance_mm:
            converged = True
            break
    candidate_continuity = continuity_metrics(ball, phases)
    continuity_blend = 1.
    if cfg.global_continuity_guard:
        ball, continuity_blend = _guard_continuity(raw, ball)
    correction = np.linalg.norm(ball - raw, axis=1)
    final_signed = np.einsum("ij,ij->i", ball - anchors, normals)
    raw_violation = np.maximum(planes["signed_distance_mm"] - cfg.surface_tolerance_mm, 0.)
    final_violation = np.maximum(final_signed - cfg.surface_tolerance_mm, 0.)
    if cfg.max_removal_depth_mm is not None:
        raw_violation += np.maximum(-cfg.max_removal_depth_mm - planes["signed_distance_mm"], 0.)
        final_violation += np.maximum(-cfg.max_removal_depth_mm - final_signed, 0.)
    report = {
        "algorithm": ("background_surface_corridor_phase_aware_bounded_spatial_ADMM_v2"
                      if cfg.background is not None else "robust_observed_planes_phase_aware_bounded_spatial_ADMM_v1"),
        "units": "mm", "configuration": asdict(cfg), "points": len(raw),
        "uses_ground_truth": False, "execution_time_available": False,
        "robot_dynamics_validated": False,
        "surface_mode": ("nominal_background_lower_and_observed_defect_upper_contact_corridor"
                         if cfg.background is not None else "camera_facing_contact_envelope"),
        "inward_depth_limit_enabled": cfg.max_removal_depth_mm is not None,
        "surface_term_enabled": bool(np.any(strengths)), "surface_eligible_points": int(eligible.sum()),
        "surface_cloud": surface.report, "spike_candidates": int(spikes.sum()),
        "protected_geometric_corners": int(corners.sum()), "short_phase_boundary_joins": int(len(joins)),
        "phase_changes": int(np.count_nonzero(phases != np.argmax(probabilities, axis=1))) if (phases >= 0).any() else 0,
        "correction_mean_mm": float(correction.mean()), "correction_max_mm": float(correction.max()),
        "endpoints_preserved": bool(np.array_equal(ball[[0, -1]], raw[[0, -1]])),
        "surface_exterior_violation_raw_mean_mm": float(raw_violation[eligible].mean()) if eligible.any() else 0.,
        "surface_exterior_violation_filtered_mean_mm": float(final_violation[eligible].mean()) if eligible.any() else 0.,
        "iterations": iteration + 1, "converged": converged,
        "primal_residual_mm": residual, "dual_residual_mm": dual_residual,
        "raw_continuity": continuity_metrics(raw, phases), "filtered_continuity": continuity_metrics(ball, phases),
        "global_continuity_guard": {
            "enabled": cfg.global_continuity_guard, "blend_fraction": continuity_blend,
            "uses_ground_truth": False, "orders": [2, 3],
            "candidate_continuity": candidate_continuity,
            "solver_residuals_describe": "pre-guard ADMM candidate",
            "physical_penalties": "soft; continuity blending can reduce the correction",
        },
        "limitations": ["sequence index is not time; no velocity/acceleration/jerk limits are certified",
                        "observed surface is pre-machining, not a later-pass target surface",
                        "soft local surface evidence is not a collision or tool/TCP feasibility certificate"],
    }
    result = {"trajectory_filtered_mm": ball, "predicted_phases": phases,
              "phase_probabilities": probabilities, "report": report}
    if background_planes is not None:
        base_raw_signed = np.einsum("ij,ij->i", raw - base_anchors, base_normals)
        base_final_signed = np.einsum("ij,ij->i", ball - base_anchors, base_normals)
        raw_lower = np.maximum(-cfg.background_tolerance_mm - base_raw_signed, 0.)
        final_lower = np.maximum(-cfg.background_tolerance_mm - base_final_signed, 0.)
        # Re-query the final curve for diagnostics; the optimization uses fixed
        # local tangent planes so that its objective stays convex.
        final_base = background_surface.query(ball)
        checked = background_eligible & (np.asarray(final_base["confidence"]) > 0)
        final_curve_lower = np.maximum(-cfg.background_tolerance_mm -
                                       np.asarray(final_base["signed_distance_mm"]), 0.)
        air_checked = air_height_preserved & (np.asarray(final_base["confidence"]) > 0)
        air_curve_change = np.abs(np.asarray(final_base["signed_distance_mm"]) -
                                  np.asarray(background_planes["signed_distance_mm"]))
        final_observed = surface.query(ball)
        upper_checked = eligible & (np.asarray(final_observed["confidence"]) > 0)
        final_observed_violation = np.maximum(np.asarray(final_observed["signed_distance_mm"]) -
                                              cfg.surface_tolerance_mm, 0.)
        report.update({
            "background_cloud": background_surface.report,
            "background_eligible_points": int(background_eligible.sum()),
            "inconsistent_corridor_points": int(inconsistent_corridor.sum()),
            "air_normal_height_preserved_points": int(air_height_preserved.sum()),
            "air_normal_height_tolerance_mm": cfg.air_normal_tolerance_mm,
            "air_normal_height_max_change_mm": float(np.max(np.abs(
                base_final_signed[air_height_preserved] - base_raw_signed[air_height_preserved])))
                if air_height_preserved.any() else 0.,
            "air_final_curve_height_checked_points": int(air_checked.sum()),
            "air_final_curve_height_max_change_mm": float(air_curve_change[air_checked].max())
                if air_checked.any() else None,
            "background_lower_violation_raw_mean_mm": float(raw_lower[background_eligible].mean())
                if background_eligible.any() else 0.,
            "background_lower_violation_filtered_mean_mm": float(final_lower[background_eligible].mean())
                if background_eligible.any() else 0.,
            "background_final_curve_checked_points": int(checked.sum()),
            "background_final_curve_lower_violation_mean_mm": float(final_curve_lower[checked].mean())
                if checked.any() else 0.,
            "surface_final_query_checked_points": int(upper_checked.sum()),
            "surface_final_query_exterior_violation_mean_mm": float(final_observed_violation[upper_checked].mean())
                if upper_checked.any() else None,
            "contact_surface_constraints": "soft; original endpoints and bounded correction remain hard",
        })
        result.update(background_anchors_mm=base_anchors, background_normals=base_normals,
                      background_confidence=np.asarray(background_planes["confidence"]),
                      background_contact_eligible=background_eligible,
                      background_air_height_preserved=air_height_preserved)
    return result
