"""Batched noise-to-program evaluation of all 300 memorization images.

Only raw generated XYZ and phase predictions count. Ground-truth phase, pass
and segment annotations are evaluation labels; none are passed to sample().
Air moves correctly leave the surface. Later contact passes refer to changed
material, so this evaluator has no full-program surface-attraction gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from .evaluation import fixed_sample_seed, restore_millimetres, _nearest_distance
from .program_data import PROGRAM_SCHEMA, ProgramDataset
from .program_model import CONTACT, PHASE_NAMES
from .runtime import write_json


COLLECTION_COUNT = 300


@dataclass(frozen=True)
class ProgramOverfitThresholds:
    """Explicit per-image memorization limits, not machining tolerances."""

    ordered_rmse_mm: float = 2.0
    pointwise_p95_mm: float = 3.0
    pointwise_max_mm: float = 6.0
    phase_accuracy: float = .95
    nonempty_phase_recall: float = .90
    duplicate_boundary_gap_max_mm: float = 1.0

    def __post_init__(self):
        if any(not math.isfinite(float(value)) or value <= 0 for value in asdict(self).values()):
            raise ValueError("Program thresholds must be finite and positive")
        if self.phase_accuracy > 1 or self.nonempty_phase_recall > 1:
            raise ValueError("Phase accuracy and recall thresholds cannot exceed one")


def _numpy(value, dtype=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _runs(phases: np.ndarray) -> list[dict]:
    starts = np.r_[0, np.flatnonzero(np.diff(phases) != 0) + 1]
    ends = np.r_[starts[1:], len(phases)]
    return [{"phase": int(phases[start]), "name": PHASE_NAMES[int(phases[start])],
             "start": int(start), "end": int(end), "point_count": int(end - start)}
            for start, end in zip(starts, ends)]


def _position_group(errors: np.ndarray, selected: np.ndarray) -> dict:
    values = errors[selected]
    if not len(values):
        return {"points": 0, "ordered_rmse_mm": None, "pointwise_mean_mm": None,
                "pointwise_p95_mm": None, "pointwise_max_mm": None}
    return {"points": len(values), "ordered_rmse_mm": float(np.sqrt(np.mean(values ** 2))),
            "pointwise_mean_mm": float(values.mean()), "pointwise_p95_mm": float(np.percentile(values, 95)),
            "pointwise_max_mm": float(values.max())}


def program_metrics(prediction_mm, target_mm, predicted_phases, target_phases, segment_ids,
                    *, duplicate_tolerance_mm: float = 1e-4) -> tuple[dict, dict[str, np.ndarray]]:
    """Ordered world-mm errors, phase confusion and source-segment continuity.

    Segment joins are inspected at GT boundary indices without using those
    indices during inference. A duplicated GT join is a source endpoint shared
    by the next segment; its generated gap is a continuity error. Flow t is a
    generation variable, not execution time, and step lengths are not speeds.
    """
    prediction, target = _numpy(prediction_mm, np.float64), _numpy(target_mm, np.float64)
    if (prediction.ndim != 2 or prediction.shape[1] != 3 or len(prediction) < 2
            or prediction.shape != target.shape or not np.isfinite(prediction).all()
            or not np.isfinite(target).all()):
        raise ValueError("Program prediction and target must be equally shaped finite [T,3]")
    predicted_phases, target_phases, segments = map(_numpy, (predicted_phases, target_phases, segment_ids))
    length = len(prediction)
    for name, value in (("predicted_phases", predicted_phases), ("target_phases", target_phases)):
        if value.shape != (length,) or not np.issubdtype(value.dtype, np.integer) or ((value < 0) | (value > 3)).any():
            raise ValueError(f"{name} must be integer [T] with four phase classes")
    if (segments.shape != (length,) or not np.issubdtype(segments.dtype, np.integer) or (segments < 0).any()):
        raise ValueError("segment_ids must be nonnegative integer [T]")
    errors = np.linalg.norm(prediction - target, axis=-1)
    confusion = np.zeros((4, 4), dtype=np.int64)
    np.add.at(confusion, (target_phases, predicted_phases), 1)
    recalls = {PHASE_NAMES[phase]: (float(confusion[phase, phase] / confusion[phase].sum())
                                   if confusion[phase].sum() else None) for phase in range(4)}
    boundaries = np.flatnonzero(segments[1:] != segments[:-1]) + 1
    starts, ends = np.r_[0, boundaries], np.r_[boundaries, length]
    if len(np.unique(segments[starts])) != len(starts):
        raise ValueError("Each source segment ID must occupy one contiguous sequence range")
    pred_gaps = np.linalg.norm(prediction[boundaries] - prediction[boundaries - 1], axis=-1)
    target_gaps = np.linalg.norm(target[boundaries] - target[boundaries - 1], axis=-1)
    duplicate = target_gaps <= duplicate_tolerance_mm
    endpoint_indices = np.column_stack((starts, ends - 1))
    endpoint_errors = errors[endpoint_indices]
    steps = np.linalg.norm(np.diff(prediction, axis=0), axis=-1)
    target_steps = np.linalg.norm(np.diff(target, axis=0), axis=-1)
    groups = {PHASE_NAMES[phase]: _position_group(errors, target_phases == phase) for phase in range(4)}
    groups["air"] = _position_group(errors, target_phases != CONTACT)
    metrics = {
        "ordered_rmse_mm": float(np.sqrt(np.mean(errors ** 2))),
        "pointwise_mean_mm": float(errors.mean()), "pointwise_p95_mm": float(np.percentile(errors, 95)),
        "pointwise_max_mm": float(errors.max()), "start_error_mm": float(errors[0]), "end_error_mm": float(errors[-1]),
        "phase_accuracy": float(np.mean(predicted_phases == target_phases)),
        "phase_confusion": confusion.tolist(), "phase_recall": recalls,
        "minimum_nonempty_phase_recall": min(value for value in recalls.values() if value is not None),
        "groups": groups, "segments": len(starts), "segment_boundaries": len(boundaries),
        "segment_endpoint_mean_mm": float(endpoint_errors.mean()), "segment_endpoint_max_mm": float(endpoint_errors.max()),
        "boundary_prediction_gap_mean_mm": float(pred_gaps.mean()) if len(pred_gaps) else 0.,
        "boundary_prediction_gap_max_mm": float(pred_gaps.max()) if len(pred_gaps) else 0.,
        "boundary_target_gap_max_mm": float(target_gaps.max()) if len(target_gaps) else 0.,
        "duplicate_boundary_count": int(duplicate.sum()),
        "duplicate_boundary_gap_max_mm": float(pred_gaps[duplicate].max()) if duplicate.any() else 0.,
        "step_mean_mm": float(steps.mean()), "step_max_mm": float(steps.max()),
        "target_step_max_mm": float(target_steps.max()),
        "prediction_arc_length_mm": float(steps.sum()), "target_arc_length_mm": float(target_steps.sum()),
        "predicted_phase_runs": _runs(predicted_phases), "target_phase_runs": _runs(target_phases),
        "predicted_phase_run_count": len(_runs(predicted_phases)), "target_phase_run_count": len(_runs(target_phases)),
        "terminal_predicted_phase": int(predicted_phases[-1]), "terminal_target_phase": int(target_phases[-1]),
        "stop_definition": "fixed_sequence_end_only_no_learned_stop_head", "fixed_end_token_index": length - 1,
    }
    arrays = {"pointwise_error_mm": errors, "phase_confusion": confusion,
              "segment_start_indices": starts, "segment_end_indices_inclusive": ends - 1,
              "segment_endpoint_errors_mm": endpoint_errors, "boundary_indices": boundaries,
              "boundary_prediction_gap_mm": pred_gaps, "boundary_target_gap_mm": target_gaps,
              "boundary_is_duplicated_gt_endpoint": duplicate, "step_lengths_mm": steps}
    return metrics, arrays


def assess_program_overfit(metrics: dict, thresholds: ProgramOverfitThresholds | dict | None = None) -> dict:
    limits = thresholds if isinstance(thresholds, ProgramOverfitThresholds) else ProgramOverfitThresholds(**(thresholds or {}))
    checks = {"full_ordered_rmse": metrics["ordered_rmse_mm"] <= limits.ordered_rmse_mm,
              "pointwise_p95": metrics["pointwise_p95_mm"] <= limits.pointwise_p95_mm,
              "pointwise_max": metrics["pointwise_max_mm"] <= limits.pointwise_max_mm,
              "phase_accuracy": metrics["phase_accuracy"] >= limits.phase_accuracy,
              "all_nonempty_phase_recall": metrics["minimum_nonempty_phase_recall"] >= limits.nonempty_phase_recall,
              "duplicated_segment_join_gap": metrics["duplicate_boundary_gap_max_mm"] <= limits.duplicate_boundary_gap_max_mm}
    return {"passed": bool(all(checks.values())), "checks": {key: bool(value) for key, value in checks.items()}}


def _load_segmentation_gate(value, sample_ids: list[str]) -> tuple[dict, bool]:
    report = json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value
    if not isinstance(report, dict):
        raise ValueError("roi_gate_report must be a mapping or JSON path")
    records = report.get("samples", [])
    ids = [row.get("sample_id") for row in records] if isinstance(records, list) else []
    passed = (report.get("passed") is True and report.get("count") == COLLECTION_COUNT
              and report.get("inference_kind") == "fresh_rgb_sam3_processor"
              and report.get("gt_mask_used_as_inference_input") is not True
              and len(ids) == COLLECTION_COUNT and len(set(ids)) == COLLECTION_COUNT
              and set(sample_ids).issubset(ids))
    return report, bool(passed)


def _first_snapshot_diagnostic(sample: dict, prediction_mm: np.ndarray, target_phases: np.ndarray) -> dict:
    if "trajectory_snapshot_index" not in sample:
        return {"available": False, "reason": "missing_snapshot_annotations", "used_in_gate": False}
    selected = (target_phases == CONTACT) & (_numpy(sample["trajectory_snapshot_index"]) == 0)
    if "trajectory_pass_index" in sample:
        selected &= _numpy(sample["trajectory_pass_index"]) == 0
    valid_roi = _numpy(sample["roi_mask"], bool) & _numpy(sample["point_valid"], bool)
    if not selected.any() or not valid_roi.any():
        return {"available": False, "reason": "no_initial_snapshot_contact_or_visible_roi", "used_in_gate": False}
    xyz = restore_millimetres(_numpy(sample["points"])[valid_roi, :3], sample["center"], sample["scale"])
    distances = _nearest_distance(prediction_mm[selected], xyz)
    return {"available": True, "points": int(selected.sum()), "mean_mm": float(distances.mean()),
            "max_mm": float(distances.max()), "reference": "sampled_observed_initial_sam_roi",
            "scope": "GT contact tokens in pass0/snapshot0 only; later material surfaces differ", "used_in_gate": False}


@torch.no_grad()
def evaluate_program_manifest(model, manifest, output_dir, *, max_points=2048, batch_size=8,
                              flow_steps=16, method="euler", seed=20261108,
                              roi_gate_report, prepared_samples=None,
                              thresholds: ProgramOverfitThresholds | dict | None = None) -> dict:
    """Run true batched noise-to-program sampling and retain every raw image.

    A cached CPU ProgramDataset sample list can bypass repeated source RDP and
    input sampling. Cache sample IDs must match the immutable manifest order.
    CPU noise is generated separately from stable image seeds, then packed into
    the GPU inference batch. Initial noise is invariant to batch size; different
    GPU batch kernels may still introduce ordinary floating-point differences.
    """
    if any(not isinstance(value, int) or value < 1 for value in (max_points, batch_size, flow_steps)):
        raise ValueError("max_points, batch_size and flow_steps must be positive integers")
    if method not in ("euler", "heun"):
        raise ValueError("method must be euler or heun")
    limits = thresholds if isinstance(thresholds, ProgramOverfitThresholds) else ProgramOverfitThresholds(**(thresholds or {}))
    dataset = ProgramDataset(manifest, max_points=max_points, trajectory_points=model.config.trajectory_points, seed=seed)
    ids = [row["sample_id"] for row in dataset.rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Program evaluation requires unique nonempty sample IDs")
    segmentation_report, segmentation_passed = _load_segmentation_gate(roi_gate_report, ids)
    if prepared_samples is not None and len(prepared_samples) != len(dataset):
        raise ValueError("prepared_samples must match the full manifest count")
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
    records, all_confusion, actual_batches = [], np.zeros((4, 4), dtype=np.int64), []
    try:
        for begin in range(0, len(dataset), batch_size):
            indices = list(range(begin, min(begin + batch_size, len(dataset))))
            samples = [prepared_samples[index] if prepared_samples is not None else dataset[index] for index in indices]
            for index, sample in zip(indices, samples):
                if sample.get("sample_id") != ids[index]:
                    raise ValueError("Cached sample ID/order disagrees with manifest")
                if any(isinstance(value, torch.Tensor) and value.device.type != "cpu" for value in sample.values()):
                    raise ValueError("prepared_samples must remain CPU tensors")
                if sample["points"].shape != (max_points, 6):
                    raise ValueError("Cached point count must match max_points")
                path = Path(dataset.rows[index]["npz"])
                if not path.is_absolute():
                    path = dataset.manifest.parent / path
                with np.load(path, allow_pickle=False) as arrays:
                    if (str(arrays["roi_source"].item()) != "sam3_prediction"
                            or str(arrays["program_schema"].item()) != PROGRAM_SCHEMA
                            or str(arrays["units"].item()) != "mm"):
                        raise ValueError("Full-program inference requires actual SAM-predicted ROI and mm program schema")
            inference = default_collate([{key: sample[key] for key in ("points", "roi_mask", "point_valid")} for sample in samples])
            noises = torch.stack([torch.randn((model.config.trajectory_points, 3),
                       generator=torch.Generator().manual_seed(fixed_sample_seed(sample["sample_id"], seed))) * model.config.noise_std
                       for sample in samples])
            # No trajectory, phase, segment, pass, stop or anchor labels enter sample().
            result = model.sample(inference["points"].to(device), inference["roi_mask"].to(device),
                                  inference["point_valid"].to(device), steps=flow_steps, method=method,
                                  initial_noise=noises.to(device=device, dtype=inference["points"].dtype))
            generated = _numpy(result["trajectory"], np.float32)
            logits = _numpy(result["phase_logits"], np.float32)
            expected_shape = (len(samples), model.config.trajectory_points, 3)
            if generated.shape != expected_shape or logits.shape != (*expected_shape[:2], 4) or not np.isfinite(logits).all():
                raise ValueError("Sampler must return full raw [B,T,3] and finite [B,T,4] phase logits")
            actual_batches.append(len(samples))
            for offset, (index, sample) in enumerate(zip(indices, samples)):
                sample_id = sample["sample_id"]
                raw_mm = restore_millimetres(generated[offset], sample["center"], sample["scale"])
                target_mm = restore_millimetres(sample["trajectory"], sample["center"], sample["scale"])
                target_phases = _numpy(sample["phase_labels"], np.int64)
                segment_ids = _numpy(sample["segment_ids"], np.int64)
                predicted_phases = logits[offset].argmax(-1)
                metrics, arrays = program_metrics(raw_mm, target_mm, predicted_phases, target_phases, segment_ids)
                gate = assess_program_overfit(metrics, limits)
                all_confusion += arrays["phase_confusion"]
                diagnostic = _first_snapshot_diagnostic(sample, raw_mm, target_phases)
                slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)[:80].strip(".") or "sample"
                suffix = hashlib.sha256(sample_id.encode()).hexdigest()[:8]
                artifact = prediction_dir / f"{index:04d}_{slug}_{suffix}.npz"
                fixed_end = np.zeros(model.config.trajectory_points, dtype=bool)
                fixed_end[-1] = True
                metadata_arrays = {key: _numpy(sample[key]) for key in
                                   ("trajectory_pass_index", "trajectory_snapshot_index", "trajectory_boundary", "trajectory_stop")
                                   if key in sample}
                np.savez_compressed(artifact, trajectory_raw_normalized=generated[offset], trajectory_raw_mm=raw_mm,
                    phase_logits=logits[offset], predicted_phases=predicted_phases, target_normalized=_numpy(sample["trajectory"]),
                    target_mm=target_mm, target_phases=target_phases, segment_ids=segment_ids,
                    initial_noise_normalized=noises[offset].numpy(),
                    normalization_center_mm=_numpy(sample["center"]), normalization_scale_mm=_numpy(sample["scale"]),
                    noise_seed=np.array(fixed_sample_seed(sample_id, seed)), fixed_length_end_mask=fixed_end,
                    sample_id=np.array(sample_id), roi_source=np.array("sam3_prediction"), units=np.array("mm"),
                    snap_to_surface=np.array(False), **arrays, **metadata_arrays)
                detail = {"sample_id": sample_id, "source_split": sample.get("source_split", "unspecified"),
                          "prediction": str(artifact), "noise_seed": fixed_sample_seed(sample_id, seed),
                          "metrics": metrics, "gate": gate, "first_snapshot_roi_diagnostic": diagnostic}
                records.append(detail)
                with (output / "details.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(detail, ensure_ascii=False, allow_nan=False) + "\n")
    finally:
        model.train(was_training)
    scalar_keys = ("ordered_rmse_mm", "pointwise_mean_mm", "pointwise_p95_mm", "pointwise_max_mm", "start_error_mm",
                   "end_error_mm", "phase_accuracy", "minimum_nonempty_phase_recall", "segment_endpoint_max_mm",
                   "duplicate_boundary_gap_max_mm", "step_max_mm")
    means = {key: float(np.mean([row["metrics"][key] for row in records])) for key in scalar_keys}
    worst = {key: float((min if key in ("phase_accuracy", "minimum_nonempty_phase_recall") else max)(
             row["metrics"][key] for row in records)) for key in scalar_keys}
    groups = {}
    for name in ("contact", "air", "retract", "transfer", "approach"):
        values = [row["metrics"]["groups"][name] for row in records if row["metrics"]["groups"][name]["points"]]
        groups[name] = {"images": len(values), "points": sum(row["points"] for row in values),
                        "mean_ordered_rmse_mm": float(np.mean([row["ordered_rmse_mm"] for row in values])) if values else None,
                        "worst_ordered_rmse_mm": max((row["ordered_rmse_mm"] for row in values), default=None),
                        "worst_pointwise_max_mm": max((row["pointwise_max_mm"] for row in values), default=None)}
    raw_passed = all(row["gate"]["passed"] for row in records)
    count_passed = len(records) == COLLECTION_COUNT and len(set(ids)) == COLLECTION_COUNT
    passed = bool(raw_passed and count_passed and segmentation_passed)
    summary = {
        "stage": "all300_full_program_noise_to_data_overfit", "passed": passed, "overall_gate_passed": passed,
        "trajectory_gate_passed": bool(raw_passed), "collection_count_passed": bool(count_passed),
        "segmentation_gate_passed": segmentation_passed, "count": len(records), "required_count": COLLECTION_COUNT,
        "passed_images": sum(row["gate"]["passed"] for row in records),
        "failed_sample_ids": [row["sample_id"] for row in records if not row["gate"]["passed"]],
        "mean": means, "worst": worst, "groups": groups, "phase_confusion": all_confusion.tolist(),
        "pooled_phase_accuracy": float(np.trace(all_confusion) / all_confusion.sum()),
        "thresholds": asdict(limits), "batch_size": batch_size, "actual_batch_sizes": actual_batches,
        "flow_steps": flow_steps, "flow_method": method, "base_seed": int(seed), "device": str(device),
        "noise_seed_policy": "independent CPU SHA256(base_seed:sample_id) seeds; initial x0 saved for every image",
        "inference_kind": "batched_raw_noise_to_program_ode_not_teacher_denoising",
        "prediction_definition": "ordered raw XYZ including contact and all air moves; no snapping or GT action conditions",
        "generation_time_definition": "flow t in [0,1] is a generation coordinate, not physical execution time",
        "stop_definition": "fixed_sequence_end_only_no_learned_stop_head",
        "program_scope": "all labelled passes/air segments; source has no initial approach or final retract",
        "surface_gate": False, "first_snapshot_roi_distance_used_in_gate": False,
        "segmentation_report": segmentation_report, "manifest": str(dataset.manifest),
        "manifest_sha256": hashlib.sha256(dataset.manifest.read_bytes()).hexdigest(),
        "cached_prepared_samples": prepared_samples is not None, "generalization_claim": False, "machining_quality_claim": False,
        "details": str(output / "details.jsonl"), "predictions": str(prediction_dir),
        "per_image": [{"sample_id": row["sample_id"], "prediction": row["prediction"], "gate": row["gate"],
                       "metrics": {key: row["metrics"][key] for key in scalar_keys}} for row in records],
    }
    write_json(output / "summary.json", summary)
    return summary
