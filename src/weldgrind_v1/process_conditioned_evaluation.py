"""Strict observation-and-known-process evaluation with the original GT gates."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import default_collate

from .evaluation import fixed_sample_seed, restore_millimetres
from .process_conditioned_data import PROCESS_PARAMETER_NAMES, PROCESS_PARAMETER_UNITS
from .program_evaluation import ProgramOverfitThresholds, assess_program_overfit, program_metrics
from .runtime import write_json


@contextmanager
def _isolated_evaluation(model, device):
    training = model.training
    state = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    numpy_state, python_state = np.random.get_state(), random.getstate()
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        torch.set_rng_state(state)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        model.train(training)


def evaluate_process_conditioned_subset(model, data, indices, output, *, seed=20261109,
                                         batch_size=8, check=None):
    """Match the repair evaluator interface; only known conditions enter sample.

    Trajectories, phases, segments and chart frames are never sampler arguments.
    Every generated program is scored against the original ordered GT using
    exactly the original ProgramOverfitThresholds, with no assignment relaxation.
    """
    indices = list(indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Evaluation needs nonempty unique indices")
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= len(data.rows) for i in indices):
        raise ValueError("Evaluation indices must be valid integers")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Evaluation batch size must be a positive integer")
    if check is not None and not callable(check):
        raise ValueError("Evaluation stop check must be callable")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    anchor = next(model.parameters(), None)
    device = anchor.device if anchor is not None else torch.device("cpu")
    records, actual_batches = [], []
    confusion = np.zeros((4, 4), dtype=np.int64)
    with _isolated_evaluation(model, device):
        for begin in range(0, len(indices), batch_size):
            if check is not None:
                check()
            chosen = indices[begin:begin + batch_size]
            samples = [data[index] for index in chosen]
            observation = default_collate([{key: sample[key] for key in
                ("points", "roi_mask", "point_valid", "scale", "process_params")} for sample in samples])
            process = observation["process_params"]
            if (process.shape != (len(samples), 4) or not torch.is_floating_point(process)
                    or not torch.isfinite(process).all() or (process[:, :3] <= 0).any()
                    or ((process[:, 3] < 0) | (process[:, 3] >= 1)).any()):
                raise ValueError("Evaluation requires four valid known process parameters per sample")
            noise = torch.stack([torch.randn((model.config.trajectory_points, 3),
                generator=torch.Generator().manual_seed(fixed_sample_seed(sample["sample_id"], seed)))
                * model.config.noise_std for sample in samples])
            generated = model.sample(observation["points"].to(device), observation["roi_mask"].to(device),
                observation["point_valid"].to(device), scale_mm=observation["scale"].to(device),
                process_params=process.to(device), initial_noise=noise.to(device), steps=8, method="euler")
            shape = (len(samples), model.config.trajectory_points, 3)
            if (generated["trajectory"].shape != shape or generated["phase_logits"].shape != (*shape[:2], 4)
                    or not torch.isfinite(generated["trajectory"]).all()
                    or not torch.isfinite(generated["phase_logits"]).all()):
                raise ValueError("Sampler must return finite complete programs and four-class phase logits")
            actual_batches.append(len(samples))
            for offset, (sample, index) in enumerate(zip(samples, chosen)):
                normalized = generated["trajectory"][offset].float().cpu()
                logits = generated["phase_logits"][offset].float().cpu()
                predicted = restore_millimetres(normalized, sample["center"], sample["scale"])
                target = restore_millimetres(sample["trajectory"], sample["center"], sample["scale"])
                phase = logits.argmax(-1).numpy()
                metrics, arrays = program_metrics(predicted, target, phase, sample["phase_labels"], sample["segment_ids"])
                confusion += arrays["phase_confusion"]
                selector = {}
                if "soft_assignments" in generated:
                    soft = generated["soft_assignments"][offset].float().cpu()
                    selector = {"entropy_mean": float(-(soft * soft.clamp_min(1e-30).log()).sum(-1).mean()),
                        "max_probability_mean": float(soft.amax(-1).mean()),
                        "soft_top1_unique": int(soft.argmax(-1).unique().numel())}
                if "selected_indices" in generated:
                    selector["hard_unique"] = int(generated["selected_indices"][offset].unique().numel())
                sid = sample["sample_id"]
                artifact = output / f"{begin + offset:04d}.npz"
                np.savez_compressed(artifact, sample_id=np.array(sid), prediction_mm=predicted,
                    prediction_normalized=normalized.numpy(), phase_logits=logits.numpy(), target_mm=target,
                    target_normalized=sample["trajectory"].numpy(), predicted_phases=phase,
                    target_phases=sample["phase_labels"].numpy(), segment_ids=sample["segment_ids"].numpy(),
                    initial_noise_normalized=noise[offset].numpy(),
                    noise_seed=np.array(fixed_sample_seed(sid, seed)), process_params=process[offset].numpy(),
                    process_parameter_names=np.asarray(PROCESS_PARAMETER_NAMES),
                    process_parameter_units=np.asarray(PROCESS_PARAMETER_UNITS),
                    normalization_center_mm=sample["center"].numpy(),
                    normalization_scale_mm=sample["scale"].numpy(), units=np.array("mm"), **arrays)
                records.append({"sample_id": sid, "manifest_index": index, "cohort": data.rows[index].get("cohort"),
                    "shape": data.rows[index].get("shape"), "prediction": str(artifact.resolve()),
                    "process_params": process[offset].tolist(), "noise_seed": fixed_sample_seed(sid, seed),
                    "metrics": metrics, "gate": assess_program_overfit(metrics), "selector": selector})
    mean = {key: float(np.mean([record["metrics"][key] for record in records])) for key in records[0]["metrics"]
            if isinstance(records[0]["metrics"][key], (int, float))}
    recalls = np.diag(confusion) / np.maximum(confusion.sum(1), 1)
    result = {"count": len(records), "mean": mean, "phase_confusion": confusion.tolist(),
        "phase_recall": recalls.tolist(), "balanced_phase_accuracy": float(recalls.mean()),
        "passed_images": sum(record["gate"]["passed"] for record in records),
        "selection_policy": "explicit immutable dataset indices; no GT error ranking",
        "method": "euler", "flow_steps": 8, "seed": seed, "actual_batch_sizes": actual_batches,
        "gt_as_inference_input": False, "known_process_parameters_as_inference_input": True,
        "process_parameter_names": list(PROCESS_PARAMETER_NAMES), "process_parameter_units": list(PROCESS_PARAMETER_UNITS),
        "process_manifest_sha256": getattr(data, "process_manifest_sha256", None),
        "thresholds": asdict(ProgramOverfitThresholds()), "is_full_validation_gate": False, "per_image": records}
    write_json(output / "summary.json", result)
    return result
