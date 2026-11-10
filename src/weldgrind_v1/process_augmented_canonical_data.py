"""Explicit, training-only depth variants over immutable canonical observations.

Evaluation always reads the original program.  The runner supplies an original
cached sample and an absolute optimizer step to ``training_item``; there is no
mutable epoch, RNG, or augmentation hidden in ``__getitem__``.  Variant targets
and their uncanonicalized source labels are offline, fingerprint-bound labels,
never generation conditions.  Only the declared layer depth changes as input.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from .canonical_program_data import (
    CanonicalProcessProgramDataset, TARGET_DEFINITION, _sha256, _target_path,
)
from .process_conditioned_data import PROCESS_PARAMETER_NAMES, _positive


PROCESS_AUGMENTATION_SCHEMA = "weldgrind-process-depth-augmentation-v1"
PROCESS_AUGMENTATION_SEED = 20261110
SUPERVISION_VARIANTS = ("original", "depth_0p375", "depth_0p25")
VARIANT_DEPTH_MM = {"depth_0p375": .375, "depth_0p25": .25}

_BOOL_LABELS = ("trajectory_contact", "trajectory_cut_level_valid", "trajectory_boundary",
                "trajectory_valid", "trajectory_stop")
_INT_LABELS = ("trajectory_kind", "trajectory_mode", "trajectory_segment",
               "trajectory_pass_index", "trajectory_snapshot_index")
_LABELS = _BOOL_LABELS + _INT_LABELS + ("trajectory_cut_level_mm",)


def _digest(value, field):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"Invalid augmentation {field}")
    return value


def _process_values(value):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("Augmentation process_params must contain four values")
    return [_positive(v, name) for name, v in zip(PROCESS_PARAMETER_NAMES, value)]


def _validate_manifest(path, dataset):
    frozen = json.loads(path.read_text())
    if frozen.get("schema") != PROCESS_AUGMENTATION_SCHEMA:
        raise ValueError("Unsupported process augmentation manifest schema")
    for key, expected in (
        ("process_manifest_sha256", dataset.process_manifest_sha256),
        ("canonical_manifest_sha256", dataset.canonical_manifest_sha256),
        ("canonical_definition_sha256", dataset.canonical_definition_sha256),
    ):
        if _digest(frozen.get(key), key) != expected:
            raise ValueError(f"Augmentation {key} binding changed")
    seed = frozen.get("augmentation_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != PROCESS_AUGMENTATION_SEED:
        raise ValueError("Augmentation seed must be the frozen 20261110 seed")
    eligible = frozen.get("eligible_sample_ids")
    if (not isinstance(eligible, list) or any(not isinstance(s, str) or not s for s in eligible)
            or len(set(eligible)) != len(eligible)):
        raise ValueError("Augmentation eligible sample IDs must be unique")
    variants = frozen.get("variants")
    if not isinstance(variants, list) or len(variants) != 2 * len(eligible):
        raise ValueError("Each eligible sample requires exactly two depth variants")
    if "variant_count" in frozen and frozen["variant_count"] != len(variants):
        raise ValueError("Augmentation variant count changed")
    indexed = {sid: {} for sid in eligible}
    for sid in eligible:
        canonical = dataset._canonical_records.get(sid)
        process = dataset._process_records.get(sid)
        if (canonical is None or process is None or canonical.get("optimizer_split") != "train"
                or process.get("optimizer_split") != "train"
                or canonical.get("cohort") != "new" or process.get("cohort") != "new"):
            raise ValueError("Only original new training samples may be augmentation eligible")
    paths = set()
    for row in variants:
        if not isinstance(row, dict):
            raise ValueError("Augmentation variant records must be mappings")
        sid, variant = row.get("sample_id"), row.get("variant_id")
        if sid not in indexed or variant not in VARIANT_DEPTH_MM or variant in indexed[sid]:
            raise ValueError("Duplicate, ineligible or unknown augmentation variant")
        canonical, process = dataset._canonical_records[sid], dataset._process_records[sid]
        if (row.get("original_optimizer_split") != "train"
                or ("cohort" in row and row["cohort"] != canonical["cohort"])):
            raise ValueError("Augmentation cannot contain validation/test or changed cohort records")
        if _digest(row.get("source_npz_sha256"), "source_npz_sha256") != canonical["source_npz_sha256"]:
            raise ValueError("Augmentation original source NPZ binding changed")
        values = _process_values(row.get("process_params"))
        original = process["process_params"]
        if (values[1] != VARIANT_DEPTH_MM[variant]
                or any(not math.isclose(values[i], original[i], rel_tol=1e-12, abs_tol=1e-12)
                       for i in (0, 2, 3))):
            raise ValueError("Only the declared layer_depth_mm may change for a depth variant")
        proof = row.get("verification", {})
        if proof.get("passed") is not True or proof.get("material_replay_passed") is not True:
            raise ValueError("Augmentation requires passed physical and material replay verification")
        _digest(row.get("target_sha256"), "target_sha256")
        if not isinstance(row.get("target_npz"), str) or not row["target_npz"]:
            raise ValueError("Augmentation target paths must be nonempty relative strings")
        target = _target_path(path.parent, row.get("target_npz", ""))
        if target in paths:
            raise ValueError("Each augmentation variant requires its own target file")
        paths.add(target)
        indexed[sid][variant] = row
    if any(set(rows) != set(VARIANT_DEPTH_MM) for rows in indexed.values()):
        raise ValueError("Each eligible sample requires both frozen depth variants")
    return frozen, indexed


def _array(arrays, name, shape, dtype):
    if name not in arrays:
        raise ValueError(f"Missing augmentation target array {name}")
    value = arrays[name].copy()
    if value.shape != shape or value.dtype != dtype:
        raise ValueError(f"Invalid augmentation {name} shape/dtype")
    if not np.isfinite(value).all():
        raise ValueError(f"Non-finite augmentation {name}")
    return value


def _load_target(root, row, definition_sha):
    target_file = _target_path(root, row["target_npz"])
    if _sha256(target_file) != row["target_sha256"]:
        raise ValueError("Augmentation target NPZ SHA256 mismatch")
    with np.load(target_file, allow_pickle=False) as arrays:
        for key, expected in (("target_definition", TARGET_DEFINITION),
                              ("canonical_definition_sha256", definition_sha),
                              ("source_npz_sha256", row["source_npz_sha256"]), ("units", "mm")):
            if key not in arrays or arrays[key].shape != () or str(arrays[key].item()) != expected:
                raise ValueError(f"Augmentation NPZ {key} binding changed")
        process = _array(arrays, "process_params", (4,), np.dtype("float32"))
        if not np.array_equal(process, np.asarray(row["process_params"], dtype=np.float32)):
            raise ValueError("Augmentation NPZ process_params differ from the manifest")
        result = {"trajectory": _array(arrays, "trajectory", (1024, 3), np.dtype("float32"))}
        for key in _BOOL_LABELS:
            result[key] = _array(arrays, key, (1024,), np.dtype("bool"))
        for key in _INT_LABELS:
            result[key] = _array(arrays, key, (1024,), np.dtype("int64"))
        result["trajectory_cut_level_mm"] = _array(arrays, "trajectory_cut_level_mm", (1024,), np.dtype("float32"))
        extra = {key for key in arrays.files if key.startswith("trajectory_")} - set(_LABELS)
        if extra:
            raise ValueError(f"Unknown augmentation trajectory labels: {sorted(extra)}")
        for key in ("source_trajectory_mm", "source_phase_labels", "source_segment_ids"):
            xyz = key == "source_trajectory_mm"
            result[key] = _array(arrays, key, (1024, 3) if xyz else (1024,),
                                 np.dtype("float32" if xyz else "int64"))
        if "resampled_segment_ranges" not in arrays:
            raise ValueError("Missing augmentation resampled segment ranges")
        ranges = arrays["resampled_segment_ranges"].copy()
        if (ranges.dtype != np.int64 or ranges.ndim != 2 or ranges.shape[1] != 2 or len(ranges) == 0
                or ranges[0, 0] != 0 or ranges[-1, 1] != 1024 or (ranges[:, 1] <= ranges[:, 0]).any()
                or not np.array_equal(ranges[:-1, 1], ranges[1:, 0])):
            raise ValueError("Malformed augmentation resampled segment ranges")
        if "source_point_count" not in arrays or arrays["source_point_count"].shape != ():
            raise ValueError("Missing augmentation source_point_count")
        source_count = arrays["source_point_count"].item()
        if isinstance(source_count, bool) or not isinstance(source_count, int) or source_count < 2:
            raise ValueError("Invalid augmentation source_point_count")
        if "segment_reconstruction_max_error_mm" not in arrays:
            raise ValueError("Missing augmentation representation error certificate")
        errors = arrays["segment_reconstruction_max_error_mm"]
        if (errors.shape != (len(ranges),) or not np.isfinite(errors).all()
                or (errors < 0).any() or (errors > .02 + 1e-8).any()):
            raise ValueError("Augmentation exceeds the canonical representation error budget")
    phase, contact = result["trajectory_mode"], result["trajectory_contact"]
    if ((phase < 0).any() or (phase > 3).any() or phase[0] != 0 or phase[-1] != 0
            or not np.array_equal(contact, phase == 0)
            or not np.array_equal(result["trajectory_cut_level_valid"], contact)
            or not result["trajectory_valid"].all() or result["trajectory_stop"].sum() != 1
            or not result["trajectory_stop"][-1]):
        raise ValueError("Malformed augmentation phase/contact/valid/stop labels")
    for key in ("trajectory_pass_index", "trajectory_snapshot_index"):
        if (result[key][contact] < 0).any() or not (result[key][~contact] == -1).all():
            raise ValueError("Augmentation contact pass/snapshot labels are inconsistent")
    if (not np.array_equal(result["trajectory_kind"], (~contact).astype(np.int64))):
        raise ValueError("Augmentation trajectory kind disagrees with contact labels")
    boundary = np.zeros(1024, dtype=bool)
    for segment, (start, end) in enumerate(ranges):
        boundary[[start, end - 1]] = True
        if not (result["trajectory_segment"][start:end] == segment).all() or len(set(phase[start:end])) != 1:
            raise ValueError("Augmentation segment/phase labels disagree with their ranges")
    if not np.array_equal(boundary, result["trajectory_boundary"]):
        raise ValueError("Augmentation segment boundary labels disagree with their ranges")
    source_phase, source_ids = result["source_phase_labels"], result["source_segment_ids"]
    if ((source_phase < 0).any() or (source_phase > 3).any() or source_phase[0] != 0 or source_phase[-1] != 0
            or source_ids[0] != 0 or (np.diff(source_ids) < 0).any() or (np.diff(source_ids) > 1).any()):
        raise ValueError("Malformed augmentation uncanonicalized source phase/segment labels")
    if any(len(set(source_phase[source_ids == segment])) != 1 for segment in np.unique(source_ids)):
        raise ValueError("Augmentation uncanonicalized source phase/segment labels disagree")
    result["segment_count"] = len(ranges)
    result["source_point_count"] = source_count
    result["process_params"] = process
    return result


class ProcessAugmentedCanonicalProgramDataset(CanonicalProcessProgramDataset):
    """Original evaluation labels, with deterministic explicit train-only variants."""
    def __init__(self, manifest, max_points=2048, trajectory_points=1024, seed=20261109,
                 max_error_mm=.02, *, process_manifest, canonical_manifest,
                 process_augmentation_manifest=None, source_root_overrides=None):
        super().__init__(manifest, max_points=max_points, trajectory_points=trajectory_points,
                         seed=seed, max_error_mm=max_error_mm, process_manifest=process_manifest,
                         canonical_manifest=canonical_manifest, source_root_overrides=source_root_overrides)
        self.process_augmentation_manifest = (Path(process_augmentation_manifest).resolve()
                                              if process_augmentation_manifest is not None else None)
        self.process_augmentation_manifest_sha256 = (_sha256(self.process_augmentation_manifest)
                                                     if self.process_augmentation_manifest else None)
        self._augmentation_records = {}
        self.augmentation_seed = PROCESS_AUGMENTATION_SEED
        if self.process_augmentation_manifest:
            _, self._augmentation_records = _validate_manifest(self.process_augmentation_manifest, self)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["supervision_variant"] = "original"
        sample["source_sample_id"] = self.rows[index]["sample_id"]
        return sample

    def training_item(self, index, *, absolute_step, base_sample):
        """Return a shallow copy; never mutate a tensor held by the runner's LRU."""
        row = self.rows[index]
        if row.get("optimizer_split") != "train":
            raise ValueError("training_item is forbidden for validation/test samples")
        if isinstance(absolute_step, bool) or not isinstance(absolute_step, Integral) or absolute_step < 0:
            raise ValueError("Augmentation requires an explicit nonnegative absolute optimizer step")
        sid = row["sample_id"]
        if (not isinstance(base_sample, dict) or base_sample.get("sample_id") != sid
                or base_sample.get("source_sample_id") != sid
                or base_sample.get("supervision_variant") != "original"
                or base_sample.get("canonical_target_sha") != self._canonical_records[sid]["target_sha256"]
                or base_sample.get("canonical_definition_sha256") != self.canonical_definition_sha256):
            raise ValueError("training_item requires the matching unaugmented canonical base sample")
        process = base_sample.get("process_params")
        expected = torch.tensor(self._process_records[sid]["process_params"], dtype=torch.float32)
        center, scale = base_sample.get("center"), base_sample.get("scale")
        if (not isinstance(process, torch.Tensor) or process.dtype != torch.float32 or process.shape != (4,)
                or process.device.type != "cpu" or not torch.equal(process, expected)
                or not isinstance(center, torch.Tensor) or center.shape != (3,) or not center.is_floating_point()
                or not torch.isfinite(center).all() or not isinstance(scale, torch.Tensor)
                or scale.numel() != 1 or not scale.is_floating_point() or not torch.isfinite(scale).all() or scale.item() <= 0):
            raise ValueError("Invalid original base process parameters or world-mm normalization")
        sample = dict(base_sample)
        variants = self._augmentation_records.get(sid)
        if variants is None:
            return sample
        payload = f"{self.augmentation_seed}:{sid}:{int(absolute_step)}".encode("utf-8")
        slot = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 3
        variant = SUPERVISION_VARIANTS[slot]
        if variant == "original":
            return sample
        record = variants[variant]
        target = _load_target(self.process_augmentation_manifest.parent, record, self.canonical_definition_sha256)
        sample["trajectory"] = (torch.from_numpy(target["trajectory"]) - center) / scale
        for key in _LABELS:
            sample[key] = torch.from_numpy(target[key])
        sample["phase_labels"] = sample["trajectory_mode"]
        sample["segment_ids"] = sample["trajectory_segment"]
        sample["source_trajectory"] = (torch.from_numpy(target["source_trajectory_mm"]) - center) / scale
        sample["source_phase_labels"] = torch.from_numpy(target["source_phase_labels"])
        sample["source_segment_ids"] = torch.from_numpy(target["source_segment_ids"])
        sample["segment_count"] = torch.tensor(target["segment_count"], dtype=torch.long)
        sample["source_point_count"] = torch.tensor(target["source_point_count"], dtype=torch.long)
        sample["process_params"] = torch.from_numpy(target["process_params"])
        sample["canonical_target_sha"] = record["target_sha256"]
        sample["supervision_variant"] = variant
        return sample
