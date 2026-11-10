"""Deterministic t0 generation scored against canonical and untouched source GT."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from .evaluation import restore_millimetres
from .process_conditioned_data import PROCESS_PARAMETER_NAMES, PROCESS_PARAMETER_UNITS
from .process_conditioned_evaluation import _isolated_evaluation
from .program_evaluation import ProgramOverfitThresholds, assess_program_overfit, program_metrics
from .runtime import write_json


CANONICAL_SCHEMA = "weldgrind-world-canonical-program-targets-v1"
CANONICAL_DEFINITION = "world_axis_contact_preserving_canonical_v1"
CANONICAL_STRATEGY = "canonical_process_deterministic_t0_program_v3"
INITIAL_STATE_PROTOCOL = "explicit_all_zero_direct_t0"


def validate_canonical_manifest(path, expected_sha256, expected_definition_sha256):
    """Bind a deliberate target-definition change without changing source inputs."""
    for name, value in (("manifest SHA256", expected_sha256), ("definition SHA256", expected_definition_sha256)):
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Canonical {name} must be explicit")
    path = Path(path).resolve(strict=True)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Canonical manifest SHA256 changed")
    value = json.loads(payload)
    definition = value.get("definition")
    if (value.get("schema") != CANONICAL_SCHEMA or not isinstance(definition, dict)
            or definition.get("name") != CANONICAL_DEFINITION):
        raise ValueError("Unsupported canonical target schema or definition")
    actual_definition = hashlib.sha256(json.dumps(definition, sort_keys=True, separators=(",", ":"),
                                                 ensure_ascii=False).encode("utf-8")).hexdigest()
    if actual_definition != expected_definition_sha256 or value.get("definition_sha256") != actual_definition:
        raise ValueError("Canonical definition SHA256 changed")
    count, records = value.get("count"), value.get("samples")
    verification = value.get("verification", {})
    if (isinstance(count, bool) or not isinstance(count, int) or count < 1
            or not isinstance(records, list) or len(records) != count
            or verification.get("passed") is not True or verification.get("checked_samples") != count):
        raise ValueError("Canonical targets require completed full-sample verification")
    ids = [row.get("sample_id") for row in records]
    if any(not isinstance(sid, str) or not sid for sid in ids) or len(set(ids)) != count:
        raise ValueError("Canonical target sample identities are invalid or duplicated")
    return {"schema": CANONICAL_SCHEMA, "name": CANONICAL_DEFINITION,
            "canonical_manifest": str(path), "canonical_manifest_sha256": expected_sha256,
            "canonical_definition_sha256": expected_definition_sha256, "count": count,
            "source_targets_retained": True, "changed_target_is_not_same_gt_improvement": True}


def _dataset_binding(data):
    required = ("canonical_manifest", "canonical_manifest_sha256", "canonical_definition_sha256")
    if any(getattr(data, name, None) is None for name in required):
        raise ValueError("Canonical evaluation requires a bound canonical manifest and definition")
    return validate_canonical_manifest(*(getattr(data, name) for name in required))


def _validate_sample(sample, length, binding):
    if sample.get("target_definition") != binding["name"]:
        raise ValueError("Sample target definition differs from the frozen canonical definition")
    if sample.get("canonical_definition_sha256") != binding["canonical_definition_sha256"]:
        raise ValueError("Sample canonical definition SHA256 changed")
    target_sha = sample.get("canonical_target_sha")
    if not isinstance(target_sha, str) or len(target_sha) != 64 or any(c not in "0123456789abcdef" for c in target_sha):
        raise ValueError("Sample must preserve its canonical target SHA256")
    for prefix in ("", "source_"):
        trajectory = sample[prefix + "trajectory"]
        if (not isinstance(trajectory, torch.Tensor) or trajectory.shape != (length, 3)
                or not torch.is_floating_point(trajectory) or not torch.isfinite(trajectory).all()):
            raise ValueError("Both canonical and source GT must be finite complete programs")
        for field in ("phase_labels", "segment_ids"):
            tensor = sample[prefix + field]
            if not isinstance(tensor, torch.Tensor) or tensor.shape != (length,) or tensor.dtype != torch.long:
                raise ValueError("Both target variants must retain fixed phase and segment labels")
        phases = sample[prefix + "phase_labels"]
        if ((phases < 0) | (phases >= 4)).any():
            raise ValueError("Phase GT must use the original four classes")


def _mean(records, key):
    return {name: float(np.mean([record[key][name] for record in records]))
            for name, value in records[0][key].items() if isinstance(value, (int, float))}


def evaluate_canonical_direct_subset(model, data, indices, output, *, seed=20261109,
                                       batch_size=8, check=None):
    """Call sample once with actual zero queries, then score both GT definitions.

    Only pointcloud/ROI/valid/observed scale and known process values enter sample.
    There is no random-noise generation, hidden GT conditioning, phase template,
    second t1 decoder, trajectory filtering, assignment relaxation or gate change.
    """
    indices = list(indices)
    if (not indices or len(set(indices)) != len(indices)
            or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(data.rows) for i in indices)):
        raise ValueError("Evaluation requires nonempty unique immutable dataset indices")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if check is not None and not callable(check):
        raise ValueError("check must be callable")
    binding = _dataset_binding(data)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    anchor = next(model.parameters(), None)
    device = anchor.device if anchor is not None else torch.device("cpu")
    length = model.config.trajectory_points
    records, actual_batches = [], []
    confusion, source_confusion = np.zeros((4, 4), dtype=np.int64), np.zeros((4, 4), dtype=np.int64)
    with _isolated_evaluation(model, device):
        for begin in range(0, len(indices), batch_size):
            if check is not None:
                check()
            chosen = indices[begin:begin + batch_size]
            samples = [data[index] for index in chosen]
            for sample in samples:
                _validate_sample(sample, length, binding)
            observation = default_collate([{key: sample[key] for key in
                ("points", "roi_mask", "point_valid", "scale", "process_params")} for sample in samples])
            process = observation["process_params"]
            if (process.shape != (len(samples), 4) or not torch.is_floating_point(process)
                    or not torch.isfinite(process).all() or (process[:, :3] <= 0).any()
                    or ((process[:, 3] < 0) | (process[:, 3] >= 1)).any()):
                raise ValueError("Evaluation requires four valid known process parameters")
            initial_state = torch.zeros((len(samples), length, 3), dtype=observation["points"].dtype, device=device)
            actual_initial_state = initial_state.detach().float().cpu().clone()
            generated = model.sample(observation["points"].to(device), observation["roi_mask"].to(device),
                observation["point_valid"].to(device), scale_mm=observation["scale"].to(device),
                process_params=process.to(device), initial_noise=initial_state, steps=1, method="direct_t0")
            shape = (len(samples), length, 3)
            if (generated["trajectory"].shape != shape or generated["phase_logits"].shape != (*shape[:2], 4)
                    or not torch.isfinite(generated["trajectory"]).all()
                    or not torch.isfinite(generated["phase_logits"]).all()):
                raise ValueError("Direct t0 sampler must return finite complete XYZ and phase logits")
            if "phases" in generated and not torch.equal(generated["phases"], generated["phase_logits"].argmax(-1)):
                raise ValueError("Direct t0 phases must come from the returned same-forward logits")
            actual_batches.append(len(samples))
            for offset, (sample, index) in enumerate(zip(samples, chosen)):
                normalized = generated["trajectory"][offset].float().cpu()
                logits = generated["phase_logits"][offset].float().cpu()
                phases = logits.argmax(-1).numpy()
                prediction = restore_millimetres(normalized, sample["center"], sample["scale"])
                target = restore_millimetres(sample["trajectory"], sample["center"], sample["scale"])
                source_target = restore_millimetres(sample["source_trajectory"], sample["center"], sample["scale"])
                metrics, arrays = program_metrics(prediction, target, phases, sample["phase_labels"], sample["segment_ids"])
                source_metrics, source_arrays = program_metrics(prediction, source_target, phases,
                    sample["source_phase_labels"], sample["source_segment_ids"])
                confusion += arrays["phase_confusion"]
                source_confusion += source_arrays["phase_confusion"]
                artifact = output / f"{begin + offset:04d}.npz"
                np.savez_compressed(artifact, sample_id=np.array(sample["sample_id"]), prediction_mm=prediction,
                    prediction_normalized=normalized.numpy(), phase_logits=logits.numpy(), predicted_phases=phases,
                    target_mm=target, canonical_target_mm=target, target_normalized=sample["trajectory"].numpy(),
                    target_phases=sample["phase_labels"].numpy(), segment_ids=sample["segment_ids"].numpy(),
                    source_target_mm=source_target, source_target_normalized=sample["source_trajectory"].numpy(),
                    source_target_phases=sample["source_phase_labels"].numpy(),
                    source_segment_ids=sample["source_segment_ids"].numpy(),
                    initial_noise_normalized=actual_initial_state[offset].numpy(),
                    initial_state_protocol=np.array(INITIAL_STATE_PROTOCOL), generation_method=np.array("direct_t0"),
                    generation_steps=np.array(1), phase_decode_time=np.array(0.),
                    process_params=process[offset].numpy(), process_parameter_names=np.asarray(PROCESS_PARAMETER_NAMES),
                    process_parameter_units=np.asarray(PROCESS_PARAMETER_UNITS),
                    target_definition=np.array(sample["target_definition"]),
                    canonical_target_sha=np.array(sample["canonical_target_sha"]),
                    canonical_definition_sha256=np.array(binding["canonical_definition_sha256"]),
                    canonical_manifest_sha256=np.array(binding["canonical_manifest_sha256"]),
                    normalization_center_mm=sample["center"].numpy(), normalization_scale_mm=sample["scale"].numpy(),
                    units=np.array("mm"), **arrays, **{f"source_{key}": value for key, value in source_arrays.items()})
                records.append({"sample_id": sample["sample_id"], "manifest_index": index,
                    "cohort": data.rows[index].get("cohort"), "shape": data.rows[index].get("shape"),
                    "prediction": str(artifact.resolve()), "process_params": process[offset].tolist(),
                    "target_definition": sample["target_definition"], "canonical_target_sha": sample["canonical_target_sha"],
                    "metrics": metrics, "gate": assess_program_overfit(metrics),
                    "source_metrics": source_metrics, "source_gate": assess_program_overfit(source_metrics)})
    recalls = np.diag(confusion) / np.maximum(confusion.sum(1), 1)
    source_recalls = np.diag(source_confusion) / np.maximum(source_confusion.sum(1), 1)
    result = {"count": len(records), "mean": _mean(records, "metrics"),
        "phase_confusion": confusion.tolist(), "phase_recall": recalls.tolist(),
        "balanced_phase_accuracy": float(recalls.mean()),
        "passed_images": sum(record["gate"]["passed"] for record in records),
        "gates": [{"sample_id": row["sample_id"], **row["gate"]} for row in records],
        "source_mean": _mean(records, "source_metrics"), "source_phase_confusion": source_confusion.tolist(),
        "source_phase_recall": source_recalls.tolist(), "source_balanced_phase_accuracy": float(source_recalls.mean()),
        "source_passed_images": sum(record["source_gate"]["passed"] for record in records),
        "source_gates": [{"sample_id": row["sample_id"], **row["source_gate"]} for row in records],
        "method": "direct_t0", "flow_steps": 1, "phase_decode_time": 0., "seed": seed,
        "initial_state_protocol": INITIAL_STATE_PROTOCOL, "random_initial_noise_generated": False,
        "actual_batch_sizes": actual_batches, "gt_as_inference_input": False,
        "known_process_parameters_as_inference_input": True, "target_definition": binding,
        "process_manifest_sha256": getattr(data, "process_manifest_sha256", None),
        "thresholds": asdict(ProgramOverfitThresholds()), "source_thresholds": asdict(ProgramOverfitThresholds()),
        "changed_target_is_not_same_gt_improvement": True, "source_gt_evaluated_separately": True,
        "selection_policy": "explicit immutable indices; no target-error ranking",
        "is_full_validation_gate": False, "per_image": records}
    write_json(output / "summary.json", result)
    return result
