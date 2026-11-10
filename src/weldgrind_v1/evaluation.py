"""Ordered, raw-trajectory evaluation for the fixed small-set overfit stage.

This evaluates one fixed first-pass contact stroke per image. It does not
evaluate a complete grinding program, a held-out generalization set, removal
quality, robot reachability, or snapped/projected trajectories. All physical
errors are in millimetres. No GPU work or checkpoint loading occurs on import.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

from .data import WeldDataset, prepare_point_cloud
from .runtime import write_json


@dataclass(frozen=True)
class OverfitThresholds:
    """Provisional memorization gate; these are not machining tolerances.

    Every sample must meet every limit using the raw model output. A jump is
    a predicted segment longer than max(jump_absolute_mm,
    jump_gt_multiplier * longest ground-truth segment).
    """

    ordered_rmse_mm: float = 1.0
    pointwise_max_mm: float = 3.0
    surface_mean_mm: float = 1.0
    surface_max_mm: float = 3.0
    jump_absolute_mm: float = 3.0
    jump_gt_multiplier: float = 3.0

    def __post_init__(self):
        if any(not np.isfinite(value) or value <= 0 for value in asdict(self).values()):
            raise ValueError("All overfit thresholds must be finite and positive")


def _xyz(value, name: str, minimum: int = 1) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or len(value) < minimum or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite [N,3] with N >= {minimum}")
    return value


def restore_millimetres(normalized_xyz, center_mm, scale_mm) -> np.ndarray:
    """Apply the exact ROI frame used for both point cloud and target path."""
    normalized_xyz = _xyz(normalized_xyz, "normalized_xyz")
    if isinstance(center_mm, torch.Tensor):
        center_mm = center_mm.detach().cpu().numpy()
    if isinstance(scale_mm, torch.Tensor):
        scale_mm = scale_mm.detach().cpu().numpy()
    center = np.asarray(center_mm, dtype=np.float64)
    scale = np.asarray(scale_mm, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all() or scale.ndim != 0 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Normalization requires finite center [3] and positive scalar scale")
    return normalized_xyz * float(scale) + center


def _nearest_distance(query: np.ndarray, reference: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    """Exact Euclidean nearest distances, bounding temporary array memory."""
    distances = np.full(len(query), np.inf, dtype=np.float64)
    for begin in range(0, len(query), 256):
        chunk = query[begin:begin + 256]
        best = np.full(len(chunk), np.inf, dtype=np.float64)
        for start in range(0, len(reference), chunk_size):
            squared = ((chunk[:, None] - reference[None, start:start + chunk_size]) ** 2).sum(-1)
            best = np.minimum(best, squared.min(-1))
        distances[begin:begin + len(chunk)] = np.sqrt(best)
    return distances


def trajectory_metrics(prediction_mm, target_mm, surface_xyz_mm, *, jump_threshold_mm: float = 3.0) -> dict:
    """Compare ordered, equally sampled paths, without reversal or alignment.

    ordered_rmse_mm = sqrt(mean_i ||prediction_i - target_i||_2^2).
    chamfer_mm is the mean of the two directed, *unsquared* nearest-point
    distances. It ignores order and is therefore only an auxiliary metric.
    Surface distances refer to observed point samples, not signed penetration.
    """
    prediction = _xyz(prediction_mm, "prediction_mm", minimum=2)
    target = _xyz(target_mm, "target_mm", minimum=2)
    surface = _xyz(surface_xyz_mm, "surface_xyz_mm")
    if prediction.shape != target.shape:
        raise ValueError("Ordered prediction and target must have identical [T,3] shapes")
    if not np.isfinite(jump_threshold_mm) or jump_threshold_mm <= 0:
        raise ValueError("jump_threshold_mm must be finite and positive")
    error = np.linalg.norm(prediction - target, axis=-1)
    surface_distance = _nearest_distance(prediction, surface)
    forward = _nearest_distance(prediction, target)
    backward = _nearest_distance(target, prediction)
    steps = np.linalg.norm(np.diff(prediction, axis=0), axis=-1)
    gt_steps = np.linalg.norm(np.diff(target, axis=0), axis=-1)
    return {
        "ordered_rmse_mm": float(np.sqrt(np.mean(error ** 2))),
        "coordinate_rmse_mm": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "pointwise_mean_mm": float(error.mean()), "pointwise_max_mm": float(error.max()),
        "pointwise_p95_mm": float(np.percentile(error, 95)),
        "start_error_mm": float(error[0]), "end_error_mm": float(error[-1]),
        "chamfer_mm": float((forward.mean() + backward.mean()) * 0.5),
        "prediction_to_target_mean_mm": float(forward.mean()),
        "target_to_prediction_mean_mm": float(backward.mean()),
        "surface_mean_mm": float(surface_distance.mean()), "surface_max_mm": float(surface_distance.max()),
        "surface_p95_mm": float(np.percentile(surface_distance, 95)),
        "step_mean_mm": float(steps.mean()), "step_max_mm": float(steps.max()),
        "target_step_max_mm": float(gt_steps.max()),
        "step_jump_threshold_mm": float(jump_threshold_mm),
        "step_jump_count": int((steps > jump_threshold_mm).sum()),
        "prediction_arc_length_mm": float(steps.sum()), "target_arc_length_mm": float(gt_steps.sum()),
        "points": len(prediction),
    }


def assess_overfit(metrics: dict, thresholds: OverfitThresholds | dict | None = None) -> dict:
    """Return every ordered/raw condition, keeping the failed limits visible."""
    limits = thresholds if isinstance(thresholds, OverfitThresholds) else OverfitThresholds(**(thresholds or {}))
    max_step = max(limits.jump_absolute_mm, limits.jump_gt_multiplier * metrics["target_step_max_mm"])
    checks = {
        "ordered_rmse": metrics["ordered_rmse_mm"] <= limits.ordered_rmse_mm,
        "pointwise_max": metrics["pointwise_max_mm"] <= limits.pointwise_max_mm,
        "raw_surface_mean": metrics["surface_mean_mm"] <= limits.surface_mean_mm,
        "raw_surface_max": metrics["surface_max_mm"] <= limits.surface_max_mm,
        "raw_step_max": metrics["step_max_mm"] <= max_step,
    }
    return {"passed": bool(all(checks.values())), "checks": {key: bool(value) for key, value in checks.items()},
            "allowed_step_max_mm": float(max_step), "thresholds": asdict(limits)}


def fixed_sample_seed(sample_id: str, base_seed: int = 20261108) -> int:
    """Noise and cloud-subset seed stable for an image across manifest ordering."""
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a nonempty string")
    digest = hashlib.sha256(f"{int(base_seed)}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 31 - 1)


def _roi_source(arrays, row) -> str:
    source = row.get("roi_source")
    if "roi_source" in arrays:
        stored = str(arrays["roi_source"].item())
        if source is not None and source != stored:
            raise ValueError("NPZ and manifest disagree about ROI source")
        source = stored
    return str(source) if source is not None else "unspecified"


@torch.no_grad()
def evaluate_manifest(model, manifest: str | Path, output_dir: str | Path, *,
                      max_points: int = 4096, flow_steps: int = 32, seed: int = 20261108,
                      thresholds: OverfitThresholds | dict | None = None,
                      require_predicted_roi: bool = True, roi_gate_report: dict | str | Path | None = None,
                      method: str = "heun") -> dict:
    """Evaluate fixed first-pass single-stroke targets and retain raw artifacts.

    For the actual pipeline use ROI source ``sam3_prediction`` in each NPZ or
    manifest row, and supply the SAM stage's saved report with top-level
    ``passed: true``. Sources from ground-truth annotations are rejected by
    default. Without a verified segmentation gate this still permits diagnostic
    trajectory evaluation, but overall_gate_passed is always false.

    Each sample receives a stable noise seed derived from its ID. Normalization
    uses its complete *predicted* ROI, independent of the target trajectory.
    No snapping, GT ROI replacement, reversal, or rigid alignment is performed.
    """
    if not isinstance(flow_steps, int) or flow_steps < 1:
        raise ValueError("flow_steps must be a positive integer")
    if method not in ("euler", "heun"):
        raise ValueError("method must be euler or heun")
    limits = thresholds if isinstance(thresholds, OverfitThresholds) else OverfitThresholds(**(thresholds or {}))
    if isinstance(roi_gate_report, (str, Path)):
        gate_path = str(Path(roi_gate_report).resolve())
        segmentation_report = json.loads(Path(roi_gate_report).read_text())
    else:
        gate_path = None
        segmentation_report = roi_gate_report or {}
    if not isinstance(segmentation_report, dict):
        raise ValueError("roi_gate_report must be a JSON mapping")
    segmentation_gate_passed = segmentation_report.get("passed") is True
    dataset = WeldDataset(manifest, max_points=max_points,
                          trajectory_points=model.config.trajectory_points, seed=seed)
    # Do not overwrite a completed evaluation: old reports and per-image paths
    # are review evidence, especially while comparing overfit checkpoints.
    output = Path(output_dir).expanduser().resolve()
    prediction_dir = output / "predictions"
    if ((output / "summary.json").exists() or (output / "details.jsonl").exists()
            or (prediction_dir.is_dir() and any(prediction_dir.iterdir()))):
        raise ValueError("Evaluation output already contains results; choose a new directory")
    prediction_dir.mkdir(parents=True, exist_ok=True)
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    was_training = model.training
    model.eval()
    details, known_ids = [], set()
    try:
        for index, row in enumerate(dataset.rows):
            path = Path(row["npz"])
            if not path.is_absolute():
                path = dataset.manifest.parent / path
            path = path.resolve()
            sample = dataset[index]  # Enforce finite mm units and single-contact-stroke contract.
            sample_id = sample["sample_id"]
            sample_seed = fixed_sample_seed(sample_id, seed)
            if sample_id in known_ids:
                raise ValueError("Evaluation sample_id values must be unique")
            known_ids.add(sample_id)
            with np.load(path, allow_pickle=False) as arrays:
                points, roi = arrays["points"].copy(), arrays["roi_mask"].astype(bool)
                source = _roi_source(arrays, row)
                target_scope = (str(arrays["target_scope"].item()) if "target_scope" in arrays
                                else row.get("target_scope", "unspecified_single_contact_stroke"))
            if require_predicted_roi and source != "sam3_prediction":
                raise ValueError(f"Evaluation requires actual SAM 3 predicted ROI, got {source!r} for {sample_id}")
            # Cloud sampling also uses the image's fixed seed, not manifest index.
            prepared = prepare_point_cloud(points, roi, max_points=max_points, seed=sample_seed)
            generator = torch.Generator(device=device).manual_seed(sample_seed)
            prediction = model.sample(prepared["points"][None].to(device), prepared["roi_mask"][None].to(device),
                                      prepared["point_valid"][None].to(device), steps=flow_steps, method=method, generator=generator)
            normalized_prediction = prediction["trajectory"][0].detach().cpu()
            raw_mm = restore_millimetres(normalized_prediction, prepared["center"], prepared["scale"])
            target_mm = restore_millimetres(sample["trajectory"], sample["center"], sample["scale"])
            surface = np.asarray(points[roi, :3], dtype=np.float64)
            gt_max_step = float(np.linalg.norm(np.diff(target_mm, axis=0), axis=-1).max())
            jump_limit = max(limits.jump_absolute_mm, limits.jump_gt_multiplier * gt_max_step)
            metrics = trajectory_metrics(raw_mm, target_mm, surface, jump_threshold_mm=jump_limit)
            gate = assess_overfit(metrics, limits)
            slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)[:80].strip(".") or "sample"
            digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8]
            artifact = prediction_dir / f"{index:04d}_{slug}_{digest}.npz"
            errors = np.linalg.norm(raw_mm - target_mm, axis=-1)
            np.savez_compressed(
                artifact, trajectory_raw_normalized=normalized_prediction.numpy(), trajectory_raw_mm=raw_mm,
                target_normalized=sample["trajectory"].numpy(), target_mm=target_mm,
                observed_roi_xyz_mm=surface, pointwise_error_mm=errors,
                raw_surface_distance_mm=_nearest_distance(raw_mm, surface),
                step_lengths_mm=np.linalg.norm(np.diff(raw_mm, axis=0), axis=-1),
                normalization_center_mm=prepared["center"].numpy(), normalization_scale_mm=prepared["scale"].numpy(),
                noise_seed=np.array(sample_seed), sample_id=np.array(sample_id), units=np.array("mm"),
                roi_source=np.array(source), target_scope=np.array(target_scope), snap_to_surface=np.array(False),
            )
            detail = {"sample_id": sample_id, "input": str(path), "prediction": str(artifact),
                      "noise_seed": sample_seed, "roi_source": source, "roi_points": int(roi.sum()),
                      "target_scope": target_scope, "metrics": metrics, "overfit_gate": gate}
            details.append(detail)
            with (output / "details.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(detail, ensure_ascii=False, allow_nan=False) + "\n")
    finally:
        model.train(was_training)
    metric_names = list(details[0]["metrics"])
    means = {key: float(np.mean([item["metrics"][key] for item in details])) for key in metric_names}
    maxima = {key: float(max(item["metrics"][key] for item in details)) for key in metric_names}
    trajectory_gate_passed = all(item["overfit_gate"]["passed"] for item in details)
    summary = {
        "stage": "fixed_small_set_raw_trajectory_overfit", "samples": len(details),
        "target_definition": "one fixed first-pass single contact stroke per image; not a complete grinding program",
        "prediction_definition": "raw ordered surface-contact XYZ; no snapping, reversal, alignment, or GT ROI substitution",
        "metric_definition": {"ordered_rmse_mm": "sqrt(mean_i ||prediction_i-target_i||_2^2)",
                              "chamfer_mm": "half the sum of directed mean unsquared nearest-point distances; auxiliary only",
                              "surface": "nearest observed ROI point distance, not signed penetration or removal quality"},
        "trajectory_gate_passed": bool(trajectory_gate_passed),
        "segmentation_gate_passed": bool(segmentation_gate_passed),
        "overall_gate_passed": bool(trajectory_gate_passed and segmentation_gate_passed and require_predicted_roi),
        "passed": bool(trajectory_gate_passed and segmentation_gate_passed and require_predicted_roi),
        "all_samples_use_verified_predicted_roi_source": bool(require_predicted_roi),
        "roi_gate_report": gate_path, "segmentation_gate_report": segmentation_report,
        "thresholds": asdict(limits), "mean_per_sample_metrics": means, "worst_per_sample_metrics": maxima,
        "base_seed": int(seed), "noise_seed_policy": "SHA256(base_seed:sample_id) first 8 bytes modulo 2^31-1",
        "flow_steps": flow_steps, "flow_method": method, "max_points": max_points, "manifest": str(dataset.manifest),
        "details": str(output / "details.jsonl"), "predictions": str(prediction_dir),
        "generalization_claim": False, "machining_quality_claim": False,
    }
    write_json(output / "summary.json", summary)
    return summary
