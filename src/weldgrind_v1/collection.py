"""Frozen train-only overfit subsets and actual-SAM-prediction ROI export.

The Isaac collection remains read-only. Target XYZ is one deterministic longest
stroke from pass zero / snapshot zero, never an air move or a complete repair.
GT segmentation files are never read to construct the model's conditioning ROI.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


REQUIRED_PATHS = ("rgb", "camera", "camera_pointcloud", "trajectory_world", "trajectory_meta")
PREDICTION_SOURCE = "sam3_prediction"
TARGET_SCOPE = "longest_single_contact_stroke_in_first_pass_not_complete_material_removal"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Dataset path escapes root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _index(dataset_root: str | Path) -> tuple[Path, list[dict], str]:
    root = Path(dataset_root).expanduser().resolve()
    bundle = _read_json(root / "training_bundle_manifest.json")
    validation = _read_json(root / "training_bundle_validation.json")
    if bundle.get("status") != "complete" or validation.get("passed") is not True:
        raise ValueError("Collection training bundle must be complete and validated before subset selection/export")
    index_path = _inside(root, bundle.get("index", "training_index.jsonl"))
    index_hash = sha256_file(index_path)
    if bundle.get("index_sha256") != index_hash:
        raise ValueError("training index fingerprint does not match completed bundle")
    rows = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    if len(rows) != bundle.get("sample_count") or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Collection index count/sample identifiers are inconsistent")
    return root, rows, index_hash


def _stroke(root: Path, row: dict) -> tuple[np.ndarray, dict]:
    meta = _read_json(_inside(root, row["paths"]["trajectory_meta"]))
    with _inside(root, row["paths"]["trajectory_world"]).open("rb") as handle:
        trajectory = np.load(handle, allow_pickle=False)
    if (trajectory.ndim != 2 or trajectory.shape[1] != 3 or len(trajectory) < 2
            or not np.isfinite(trajectory).all()):
        raise ValueError("Collection trajectory must be finite [M,3]")
    definition = str(meta.get("contact_reference_definition", "")).lower()
    if (meta.get("status") != "validated" or meta.get("units") != "mm"
            or meta.get("coordinate_frame") != "world"
            or meta.get("representation") != "ordered_xyz_surface_contact_reference"
            or not ("before each pass" in definition or "pre-pass" in definition)
            or meta.get("segment_ranges_convention") != "half-open [start, end)"
            or meta.get("point_count") != len(trajectory)):
        raise ValueError("Trajectory lacks validated world-mm pre-pass contact-reference semantics")
    candidates = []
    previous_end = 0
    for index, segment in enumerate(meta.get("segment_ranges", [])):
        start, end = segment.get("start"), segment.get("end")
        if (not isinstance(start, int) or not isinstance(end, int) or start < previous_end
                or not 0 <= start < end <= len(trajectory)):
            raise ValueError("Invalid/overlapping half-open trajectory segments")
        previous_end = end
        if (segment.get("contact") is True and segment.get("kind") == "grind_stroke"
                and segment.get("pass_index") == 0 and segment.get("surface_snapshot_index") == 0
                and segment.get("instance_id") == row.get("defect_instance_id")):
            curve = trajectory[start:end]
            length = float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())
            if len(curve) >= 2 and length > 0:
                candidates.append((length, -index, index, segment, curve))
    if not candidates:
        raise ValueError(f"No unambiguous nonzero first-pass/snapshot-zero contact stroke: {row['sample_id']}")
    length, _, index, segment, curve = max(candidates, key=lambda item: (item[0], item[1]))
    return np.asarray(curve, dtype=np.float32), {
        "scope": TARGET_SCOPE, "source_segment_index": index,
        "source_segment": segment, "arc_length_mm": length, "source_point_count": len(curve),
        "selection_rule": "maximum_world_arc_length; ties_choose_lowest_segment_index",
        "contact_reference_definition": meta["contact_reference_definition"],
    }


def _freeze_row(root: Path, row: dict) -> dict:
    if row.get("split") != "train" or row.get("grindable") is not True:
        raise ValueError("Overfit subset may contain only grindable train-source samples")
    if row.get("units") != "mm" or row.get("coordinate_frame") != "world":
        raise ValueError("Collection source must use world coordinates in mm")
    missing = set(REQUIRED_PATHS) - set(row.get("paths", {}))
    if missing:
        raise ValueError(f"Collection sample missing source paths: {sorted(missing)}")
    frozen = dict(row)
    fingerprints = {}
    keys = REQUIRED_PATHS + (("asset_metadata",) if "asset_metadata" in row["paths"] else ())
    for key in keys:
        path = _inside(root, row["paths"][key])
        fingerprints[key] = {"path": row["paths"][key], "sha256": sha256_file(path), "bytes": path.stat().st_size}
    frozen["source_fingerprints"] = fingerprints
    _, frozen["trajectory_selection"] = _stroke(root, row)
    return frozen


def select_overfit_subset(dataset_root: str | Path, count: int = 6) -> list[dict]:
    """Deterministically cover shape/code pairs using train-source RGB images.

    Selection is prospective and never consults SAM predictions. Use
    ``freeze_subset`` to persist before training; export never reselects samples.
    """
    if not isinstance(count, int) or count < 1:
        raise ValueError("subset count must be a positive integer")
    root, rows, _ = _index(dataset_root)
    candidates = sorted((row for row in rows if row.get("split") == "train" and row.get("grindable") is True),
                        key=lambda row: row["sample_id"])
    if len(candidates) < count:
        raise ValueError("Not enough grindable train-source samples for the requested fixed subset")
    selected, shapes, codes, pairs = [], set(), set(), set()
    while len(selected) < count:
        def score(row):
            shape, code = row.get("shape"), row.get("defect_code")
            return (int(shape not in shapes) + int(code not in codes), int((shape, code) not in pairs))
        # Stable max preserves lexicographically first sample on equal scores.
        chosen = max(candidates, key=score)
        candidates.remove(chosen)
        selected.append(chosen)
        shapes.add(chosen.get("shape"))
        codes.add(chosen.get("defect_code"))
        pairs.add((chosen.get("shape"), chosen.get("defect_code")))
    return [_freeze_row(root, row) for row in selected]


def _selection_digest(selected: list[dict]) -> str:
    serialized = json.dumps(selected, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _load_fixed_subset(root: Path, rows: list[dict], index_hash: str, subset_file: Path) -> tuple[dict, list[dict]]:
    frozen = _read_json(subset_file)
    if (frozen.get("schema") != "weldgrind-fixed-overfit-subset-v1"
            or Path(frozen.get("dataset_root", "")).resolve() != root
            or frozen.get("dataset_index_sha256") != index_hash):
        raise ValueError("Fixed subset belongs to a different dataset/index; automatic reselection is forbidden")
    selected = frozen.get("selected")
    if (not isinstance(selected, list) or not selected or frozen.get("count") != len(selected)
            or frozen.get("selection_sha256") != _selection_digest(selected)):
        raise ValueError("Fixed subset has invalid count or selection fingerprint")
    indexed = {row["sample_id"]: row for row in rows}
    if len({row["sample_id"] for row in selected}) != len(selected):
        raise ValueError("Fixed subset cannot contain duplicate samples")
    for row in selected:
        original = indexed.get(row["sample_id"])
        if original is None or _freeze_row(root, original) != row:
            raise ValueError(f"Frozen source/trajectory selection changed: {row['sample_id']}; do not replace samples")
    return frozen, selected


def freeze_subset(dataset_root: str | Path, subset_file: str | Path, count: int = 6) -> Path:
    """Persist a fixed list exclusively; an existing valid list is never replaced."""
    root, rows, index_hash = _index(dataset_root)
    path = Path(subset_file).expanduser().resolve()
    if path.is_relative_to(root):
        raise ValueError("Fixed subset must be written outside the read-only source collection")
    if path.exists():
        frozen, _ = _load_fixed_subset(root, rows, index_hash, path)
        if frozen["count"] != count:
            raise ValueError("Existing subset count is frozen; use an explicit new experiment instead of replacing it")
        return path
    selected = select_overfit_subset(root, count)
    value = {
        "schema": "weldgrind-fixed-overfit-subset-v1", "dataset_root": str(root),
        "dataset_index_sha256": index_hash, "count": count, "selection_sha256": _selection_digest(selected),
        "purpose": "train_source_small_set_overfit_only", "selected": selected,
        "selection_rule": "greedy_shape_and_defect_code_coverage_then_pair_coverage; sample_id_lexical_tiebreak",
        "target_scope": TARGET_SCOPE, "automatic_sample_replacement": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return path


def _scalar(arrays, key):
    if key not in arrays or arrays[key].ndim != 0:
        raise ValueError(f"SAM prediction requires scalar provenance field {key}")
    return arrays[key].item()


def _load_prediction(root: Path, row: dict, predictions: Path) -> tuple[np.ndarray, dict]:
    path = (predictions / f"{row['sample_id']}.npz").resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Actual SAM 3 prediction is missing; GT mask cannot substitute: {path}")
    if path.is_relative_to(root):
        raise ValueError("SAM predictions must be separate inference artifacts outside the source dataset")
    # Catch accidental links to any source annotation or label file.
    for key in ("labels_2d", "defect_mask_png", "grinding_mask_png", "coco_annotations"):
        if key in row["paths"] and path == (root / row["paths"][key]).resolve():
            raise ValueError("GT segmentation cannot be used as sam_prediction")
    with np.load(path, allow_pickle=False) as arrays:
        if _scalar(arrays, "source") != PREDICTION_SOURCE or _scalar(arrays, "inference_executed") is not True:
            raise ValueError("Mask must be an executed SAM 3 prediction, never a GT mask")
        if _scalar(arrays, "sample_id") != row["sample_id"]:
            raise ValueError("SAM prediction sample_id does not match fixed sample")
        rgb_hash = _scalar(arrays, "rgb_sha256")
        if rgb_hash != row["source_fingerprints"]["rgb"]["sha256"]:
            raise ValueError("SAM prediction does not match frozen source RGB fingerprint")
        checkpoint_hash = _scalar(arrays, "checkpoint_sha256")
        if (not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64
                or any(character not in "0123456789abcdef" for character in checkpoint_hash)):
            raise ValueError("SAM prediction requires the actual checkpoint SHA256")
        mask = arrays["mask"] if "mask" in arrays else None
        expected_shape = tuple(reversed(row["image_size_wh"]))
        if mask is None or mask.dtype != np.bool_ or mask.shape != expected_shape:
            raise ValueError(f"SAM prediction mask must be bool [H,W] at source RGB resolution {expected_shape}")
        mask = mask.copy()
    return mask, {
        "path": str(path), "sha256": sha256_file(path), "source": PREDICTION_SOURCE,
        "inference_executed": True, "checkpoint_sha256": checkpoint_hash, "rgb_sha256": rgb_hash,
    }


def _camera_cloud(root: Path, row: dict, mask: np.ndarray) -> tuple[dict, np.ndarray]:
    camera = _read_json(_inside(root, row["paths"]["camera"]))
    if (camera.get("capture_status") != "accepted" or camera.get("world_units") != "mm"
            or camera.get("image_size_wh") != row["image_size_wh"]):
        raise ValueError("Camera capture must be accepted with matching resolution and world-mm coordinates")
    with Image.open(_inside(root, row["paths"]["rgb"])) as image:
        if list(image.size) != row["image_size_wh"]:
            raise ValueError("Source RGB dimensions do not match calibration/index")
    with np.load(_inside(root, row["paths"]["camera_pointcloud"]), allow_pickle=False) as arrays:
        # Explicit allow-list: no instance_ids/grinding_mask/semantic GT is read.
        xyz = arrays["xyz_world_mm"].astype(np.float32)
        camera_xyz = arrays["xyz_camera_cv_mm"].astype(np.float32)
        rgb = arrays["rgb"]
        uv = arrays["pixel_uv"]
    n = len(xyz)
    if (xyz.shape != (n, 3) or camera_xyz.shape != (n, 3) or rgb.shape != (n, 3)
            or uv.shape != (n, 2) or rgb.dtype != np.uint8 or not np.issubdtype(uv.dtype, np.integer)
            or not np.isfinite(xyz).all() or not np.isfinite(camera_xyz).all()):
        raise ValueError("Malformed visible RGB-D point cloud; expected finite world/camera XYZ, uint8 RGB and integer pixel_uv")
    width, height = row["image_size_wh"]
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height) & (camera_xyz[:, 2] > 0)
    if not valid.all():
        raise ValueError("Visible camera point cloud contains invalid pixel_uv or nonpositive optical depth")
    # pixel_uv is an exact source pixel index. Isaac uses centres at u+.5,v+.5;
    # rounding a reprojected centre would shift half the points to neighbours.
    roi = mask[uv[:, 1], uv[:, 0]]
    if not roi.any():
        raise ValueError(f"Actual SAM prediction has no visible cloud ROI: {row['sample_id']}")
    transform = np.asarray(camera["world_from_camera_cv_mm"], dtype=np.float64)
    intrinsics = np.asarray(camera["intrinsics_K"], dtype=np.float64)
    if (transform.shape != (4, 4) or intrinsics.shape != (3, 3)
            or not np.isfinite(transform).all() or not np.isfinite(intrinsics).all()):
        raise ValueError("Invalid camera calibration matrices")
    exported = {
        "points": np.column_stack([xyz, rgb.astype(np.float32) / 255]).astype(np.float32),
        "roi_mask": roi, "mask": mask, "point_uv": uv.astype(np.int32),
        "intrinsics": intrinsics, "camera_from_world": np.linalg.inv(transform),
        "pixel_center_offset": np.array(float(camera.get("pixel_center_offset", 0.5))),
        "units": np.array("mm"), "coordinate_frame": np.array("world"),
        "roi_source": np.array(PREDICTION_SOURCE),
    }
    return exported, roi


def export_collection(dataset_root: str | Path, subset_file: str | Path,
                      predictions_dir: str | Path, output_dir: str | Path) -> Path:
    """Export a frozen set only after actual SAM predictions exist for every RGB.

    Prediction NPZ files must carry mask, source='sam3_prediction', sample_id,
    inference_executed=True, rgb_sha256 and checkpoint_sha256. Source annotations
    are never a fallback. All inputs are preflighted before output creation.
    """
    root, indexed, index_hash = _index(dataset_root)
    subset_path = Path(subset_file).expanduser().resolve()
    frozen, selected = _load_fixed_subset(root, indexed, index_hash, subset_path)
    predictions = Path(predictions_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.is_relative_to(root):
        raise ValueError("Export output must be outside the read-only source collection")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing collection export: {output}")
    prepared = []
    checkpoint_hashes = set()
    for row in selected:
        mask, prediction = _load_prediction(root, row, predictions)
        arrays, roi = _camera_cloud(root, row, mask)
        curve, selection = _stroke(root, row)
        if selection != row["trajectory_selection"]:
            raise ValueError("Frozen trajectory selection changed; automatic replacement is forbidden")
        segment = {"start": 0, "end": len(curve), "contact": True, "kind": "grind_stroke",
                   "pass_index": 0, "surface_snapshot_index": 0}
        arrays.update(trajectory=curve, contact=np.ones(len(curve), dtype=bool),
                      segment_ranges=np.array(json.dumps([segment])),
                      trajectory_scope=np.array(TARGET_SCOPE))
        metadata = {
            "sample_id": row["sample_id"], "split": "train", "source_split": "train",
            "shape": row["shape"], "defect_code": row["defect_code"],
            "units": "mm", "coordinate_frame": "world", "roi_source": PREDICTION_SOURCE,
            "roi_points": int(roi.sum()), "point_count": len(roi),
            "source_fingerprints": row["source_fingerprints"], "prediction": prediction,
            "trajectory_selection": selection, "scope": "small_fixed_train_subset_overfit_only",
            "complete_material_removal_claim": False,
        }
        checkpoint_hashes.add(prediction["checkpoint_sha256"])
        prepared.append((row["sample_id"], arrays, metadata))
    if len(checkpoint_hashes) != 1:
        raise ValueError("All fixed-subset masks must come from the same actual SAM checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for sample_id, arrays, metadata in prepared:
        filename = f"{sample_id}.npz"
        np.savez_compressed(output / filename, **arrays)
        (output / f"{sample_id}.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        records.append({"npz": filename, "sample_id": sample_id, "split": "train", "units": "mm",
                        "source_split": "train", "roi_source": PREDICTION_SOURCE,
                        "trajectory_scope": TARGET_SCOPE})
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
    (output / "export_receipt.json").write_text(json.dumps({
        "schema": "weldgrind-collection-export-v1", "dataset_root": str(root),
        "dataset_index_sha256": index_hash, "fixed_subset_file": str(subset_path),
        "fixed_subset_sha256": sha256_file(subset_path), "selection_sha256": frozen["selection_sha256"],
        "sample_ids": [row["sample_id"] for row in selected], "count": len(selected),
        "sam_checkpoint_sha256": next(iter(checkpoint_hashes)), "gt_mask_used_as_input": False,
        "roi_source": PREDICTION_SOURCE, "target_scope": TARGET_SCOPE,
        "complete_material_removal_claim": False, "manifest_sha256": sha256_file(manifest),
    }, indent=2, ensure_ascii=False) + "\n")
    return manifest
