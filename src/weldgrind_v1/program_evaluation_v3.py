"""V3 full-program evaluation with explicit observed physical-scale conditions.

Metrics, gates, fixed per-image noise and raw artifacts deliberately match the
original evaluator. The only additional inference condition is sample['scale']
from the observed SAM ROI. Ground-truth paths, phases, passes and target scales
remain supervision/evaluation labels, never inputs to model.sample().
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from .evaluation import fixed_sample_seed, restore_millimetres
from .program_data import PROGRAM_SCHEMA, ProgramDataset
from .program_evaluation import (
    COLLECTION_COUNT, ProgramOverfitThresholds, _numpy, _load_segmentation_gate,
    _first_snapshot_diagnostic, program_metrics, assess_program_overfit,
)
from .runtime import write_json


@torch.no_grad()
def evaluate_metric_program_manifest(model, manifest, output_dir, *, max_points=2048, batch_size=8,
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
            for sample in samples:
                observed_scale = sample.get("scale")
                if (not isinstance(observed_scale, torch.Tensor) or observed_scale.ndim != 0
                        or not torch.isfinite(observed_scale) or observed_scale <= 0):
                    raise ValueError("Observed ROI scale must be a finite positive CPU scalar tensor")
            inference = default_collate([{key: sample[key] for key in ("points", "roi_mask", "point_valid", "scale")} for sample in samples])
            noises = torch.stack([torch.randn((model.config.trajectory_points, 3),
                       generator=torch.Generator().manual_seed(fixed_sample_seed(sample["sample_id"], seed))) * model.config.noise_std
                       for sample in samples])
            # No trajectory, phase, segment, pass, stop or anchor labels enter sample().
            result = model.sample(inference["points"].to(device), inference["roi_mask"].to(device),
                                  inference["point_valid"].to(device),
                                  scale_mm=inference["scale"].to(device=device, dtype=inference["points"].dtype).reshape(-1),
                                  steps=flow_steps, method=method,
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
                    observed_scale_condition_mm=_numpy(sample["scale"]),
                    noise_seed=np.array(fixed_sample_seed(sample_id, seed)), fixed_length_end_mask=fixed_end,
                    sample_id=np.array(sample_id), roi_source=np.array("sam3_prediction"), units=np.array("mm"),
                    snap_to_surface=np.array(False), **arrays, **metadata_arrays)
                detail = {"sample_id": sample_id, "source_split": sample.get("source_split", "unspecified"),
                          "prediction": str(artifact), "noise_seed": fixed_sample_seed(sample_id, seed),
                          "observed_scale_condition_mm": float(sample["scale"]),
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
        "stage": "all300_full_program_metric_conditioned_noise_to_data_overfit_v3",
        "evaluation_format": "weldgrind-metric-program-evaluation-v3",
        "observable_scale_condition": True,
        "scale_condition_source": "ProgramDataset sample['scale']; observed SAM-predicted ROI normalization scale, never target extent",
        "passed": passed, "overall_gate_passed": passed,
        "trajectory_gate_passed": bool(raw_passed), "collection_count_passed": bool(count_passed),
        "segmentation_gate_passed": segmentation_passed, "count": len(records), "required_count": COLLECTION_COUNT,
        "passed_images": sum(row["gate"]["passed"] for row in records),
        "failed_sample_ids": [row["sample_id"] for row in records if not row["gate"]["passed"]],
        "mean": means, "worst": worst, "groups": groups, "phase_confusion": all_confusion.tolist(),
        "pooled_phase_accuracy": float(np.trace(all_confusion) / all_confusion.sum()),
        "thresholds": asdict(limits), "batch_size": batch_size, "actual_batch_sizes": actual_batches,
        "flow_steps": flow_steps, "flow_method": method, "base_seed": int(seed), "device": str(device),
        "noise_seed_policy": "independent CPU SHA256(base_seed:sample_id) seeds; initial x0 saved for every image",
        "inference_kind": "batched_observed_scale_conditioned_raw_noise_to_program_ode_not_teacher_denoising",
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
