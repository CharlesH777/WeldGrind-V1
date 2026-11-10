"""Finite-budget training and millimetre-space inference helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
import yaml

from .data import WeldDataset, prepare_point_cloud
from .model import ModelConfig, WeldGrindModel


def read_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict) or "model" not in value:
        raise ValueError("configuration must contain a model mapping")
    ModelConfig.from_dict(value["model"])
    return value


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_checkpoint(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def batch_indices(step: int, batch_size: int, count: int, seed: int) -> list[int]:
    """Shuffle by epoch and absolute sample position, reproducible after resume."""
    if min(batch_size, count) < 1 or step < 0:
        raise ValueError("invalid batch parameters")
    permutations = {}
    result = []
    for position in range(step * batch_size, (step + 1) * batch_size):
        epoch, offset = divmod(position, count)
        if epoch not in permutations:
            permutations[epoch] = torch.randperm(count, generator=torch.Generator().manual_seed(seed + epoch))
        result.append(int(permutations[epoch][offset]))
    return result


def dataset_signature(dataset: WeldDataset) -> str:
    """Bind relative manifests to actual files without reading all dataset bytes.

    This is a resolved-path/size/mtime identity guard, not a content hash of
    every NPZ. Deliberately forged metadata or in-place concurrent writes are
    outside the immutable-input contract; actual datasets must be versioned.
    """
    records = [{"manifest": str(dataset.manifest)}]
    for row in dataset.rows:
        path = Path(row["npz"])
        if not path.is_absolute():
            path = dataset.manifest.parent / path
        path = path.resolve()
        stat = path.stat()
        records.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def train(config_path: str | Path, manifest: str | Path, output: str | Path,
          device: str = "cpu", steps: int | None = None, resume: str | Path | None = None,
          step_callback=None) -> dict:
    config = read_config(config_path)
    model_config = ModelConfig.from_dict(config["model"])
    settings = config.get("training", {})
    seed = int(settings.get("seed", 20261108))
    batch_size = int(settings.get("batch_size", 2))
    save_every = int(settings.get("save_every", 100))
    total_steps = int(steps if steps is not None else settings.get("steps", 10000))
    if min(batch_size, save_every, total_steps) < 1:
        raise ValueError("batch_size, save_every and steps must be positive")
    output = Path(output).resolve()
    # A run directory owns its stop request, state and checkpoints. Accidentally
    # reusing it would erase the evidence needed to resume/review a prior run.
    if (output / "latest.json").exists() and resume is None:
        raise ValueError("output already contains a run; use a new directory or --resume")
    manifest = Path(manifest).resolve()
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    torch.manual_seed(seed)
    model = WeldGrindModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings.get("lr", 0.0003)),
                                  weight_decay=float(settings.get("weight_decay", 0.01)))
    dataset = WeldDataset(manifest, max_points=int(config.get("data", {}).get("max_points", 4096)),
                          trajectory_points=model_config.trajectory_points, seed=seed)
    if len(dataset) < 1:
        raise ValueError("manifest contains no samples")
    data_identity = dataset_signature(dataset)
    completed = 0
    metrics = {}
    if resume is not None:
        state = torch.load(resume, map_location=device, weights_only=True)
        if steps is None:
            total_steps = int(state["target_steps"])
        if state["model_config"] != model_config.to_dict() or state["manifest_sha256"] != manifest_hash:
            raise ValueError("resume requires identical model configuration and manifest")
        if state["dataset_signature"] != data_identity:
            raise ValueError("resume requires identical resolved dataset files (path/size/mtime)")
        if state["training_settings"] != settings or state["data_settings"] != config.get("data", {}):
            raise ValueError("resume requires identical training and data settings")
        if total_steps > state["target_steps"]:
            raise ValueError("resume cannot automatically extend the original absolute step budget")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        completed = int(state["step"])
        metrics = state.get("metrics", {})
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.startswith("cuda") and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng"]])
    if completed > total_steps:
        raise ValueError("checkpoint step is beyond the requested absolute budget")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    start = time.monotonic()
    status = "running"

    def save(status_name):
        checkpoint = output / "checkpoints" / f"step_{completed:08d}.pt"
        save_checkpoint(checkpoint, {
            "format": "weldgrind-v1-0.1", "model_config": model_config.to_dict(),
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": completed,
            "target_steps": total_steps, "manifest_sha256": manifest_hash,
            "dataset_signature": data_identity,
            "training_settings": settings, "data_settings": config.get("data", {}),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
            "metrics": metrics,
        })
        write_json(output / "latest.json", {
            "checkpoint": str(checkpoint), "step": completed, "target_steps": total_steps,
            "status": status_name, "local_save_complete": True,
        })

    model.train()
    try:
        while completed < total_steps:
            if (output / "REQUEST_STOP").exists():
                status = "stopped"
                break
            indices = batch_indices(completed, batch_size, len(dataset), seed)
            batch = default_collate([dataset[index] for index in indices])
            batch = {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            losses = model.flow_loss(batch["points"], batch["roi_mask"], batch["trajectory"],
                                     batch["point_valid"], settings.get("loss_weights"))
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("nonfinite loss; training stopped without retry")
            losses["loss"].backward()
            finite_gradients = [torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None]
            if finite_gradients and not torch.stack(finite_gradients).all():
                raise FloatingPointError("nonfinite gradient; training stopped without retry")
            optimizer.step()
            completed += 1
            metrics = {name: float(value.detach()) for name, value in losses.items()}
            progress = {
                "step": completed, "target_steps": total_steps, "status": "running", "metrics": metrics,
                "elapsed_seconds": time.monotonic() - start, "sample_ids": batch.get("sample_id", []),
            }
            write_json(output / "progress.json", progress)
            print(json.dumps(progress, ensure_ascii=False), flush=True)
            if step_callback is not None and step_callback(model, completed):
                status = "gate_passed"
                break
            if completed % save_every == 0 and completed < total_steps:
                save("running")
    except Exception as error:
        write_json(output / "progress.json", {
            "step": completed, "target_steps": total_steps, "status": "failed", "error": str(error),
            "elapsed_seconds": time.monotonic() - start, "automatic_retry": False,
        })
        raise
    if status == "running":
        status = "completed"
    save(status)
    result = {"status": status, "step": completed, "target_steps": total_steps, "metrics": metrics,
              "elapsed_seconds": time.monotonic() - start, "output": str(output)}
    write_json(output / "progress.json", result)
    return result


def load_model(checkpoint: str | Path, device: str = "cpu") -> tuple[WeldGrindModel, dict]:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if state.get("format") != "weldgrind-v1-0.1":
        raise ValueError("unsupported checkpoint format")
    model = WeldGrindModel(ModelConfig.from_dict(state["model_config"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


@torch.no_grad()
def predict(model: WeldGrindModel, points: np.ndarray, roi_mask: np.ndarray,
            max_points: int = 4096, steps: int = 32, seed: int = 0,
            snap_to_surface: bool = False) -> dict[str, np.ndarray]:
    prepared = prepare_point_cloud(points, roi_mask, max_points=max_points, seed=seed)
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    output = model.sample(prepared["points"][None].to(device), prepared["roi_mask"][None].to(device),
                          prepared["point_valid"][None].to(device), steps=steps, generator=generator)
    normalized_path = output["trajectory"][0].cpu()
    center, scale = prepared["center"], prepared["scale"]
    raw_path = normalized_path * scale + center
    # Report deviation against the complete observed ROI even without optional
    # postprojection. Chunking bounds memory for large registered input clouds.
    observed = torch.as_tensor(np.asarray(points)[np.asarray(roi_mask).astype(bool), :3], dtype=raw_path.dtype)
    nearest_distance = torch.full((len(raw_path),), float("inf"), dtype=raw_path.dtype)
    projected = torch.zeros_like(raw_path)
    for chunk in observed.split(8192):
        distances, nearest = torch.cdist(raw_path[None], chunk[None], compute_mode="donot_use_mm_for_euclid_dist")[0].min(-1)
        closer = distances < nearest_distance
        projected[closer] = chunk[nearest[closer]]
        nearest_distance = torch.minimum(nearest_distance, distances)
    path = projected if snap_to_surface else raw_path
    return {
        "trajectory_mm": path.numpy(), "trajectory_raw_mm": raw_path.numpy(),
        "trajectory_projected_mm": projected.numpy(), "raw_surface_distance_mm": nearest_distance.numpy(),
        "step_lengths_mm": torch.linalg.vector_norm(path.diff(dim=0), dim=-1).numpy(),
        "seeds_mm": (output["seed_xyz"][0].cpu() * scale + center).numpy(),
        "seed_unique": output["seed_unique"][0].cpu().numpy(),
        "candidate_xyz_mm": (output["candidate_xyz"][0].cpu() * scale + center).numpy(),
        "candidate_valid": output["candidate_valid"][0].cpu().numpy(),
        "normalization_center_mm": center.numpy(), "normalization_scale_mm": scale.numpy(),
    }
