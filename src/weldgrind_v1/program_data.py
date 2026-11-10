"""Complete ordered grinding programs, including every air move and pass.

This is a supervision/representation adapter, not a hand-written action planner.
The full XYZ sequence can be modelled directly with conditional Flow Matching.
Contact, segment, pass and cutting-plane annotations remain separate labels.
The source collection is read-only and inference ROI must come from SAM 3.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .collection import _index, _load_prediction, sha256_file
from .data import prepare_point_cloud
from .predicted_rgbd_cloud import POINTCLOUD_SOURCE, load_predicted_rgbd_cloud, mask_postprocess


PROGRAM_SCHEMA = "weldgrind-complete-program-v1"
SOURCE_KINDS = {"grind_stroke": 0, "air_move": 1}
ACTION_MODES = {"contact": 0, "retract": 1, "transfer": 2, "approach": 3}


def _segment_modes(segments, trajectory=None, normal_world=None):
    """Derive labels from existing ordered air triplets, never generate motion.

    The source only calls these air_move. Its generator emits three explicit
    segments per transition. The collection loader additionally verifies their
    chart-normal geometry before assigning these descriptive auxiliary labels.
    """
    result = np.zeros(len(segments), dtype=np.int64)
    index = 0
    while index < len(segments):
        if segments[index]["contact"]:
            index += 1
            continue
        end = index
        while end < len(segments) and not segments[end]["contact"]:
            end += 1
        if end - index != 3 or index == 0 or end == len(segments):
            raise ValueError("Air transitions must retain the source's three segments between contact strokes")
        result[index:end] = [1, 2, 3]
        if normal_world is not None:
            normal = np.asarray(normal_world, dtype=np.float64)
            normal = normal / np.linalg.norm(normal)
            for offset in range(3):
                segment = segments[index + offset]
                curve = np.asarray(trajectory[segment["start"]:segment["end"]], dtype=np.float64)
                displacement = curve - curve[0]
                height = displacement @ normal
                lateral = displacement - height[:, None] * normal
                if offset == 0 and not (height[-1] > 0 and np.linalg.norm(lateral, axis=-1).max() <= 1e-4):
                    raise ValueError("Source retract segment is not an upward chart-normal move")
                if offset == 1 and np.abs(height).max() > 1e-4:
                    raise ValueError("Source transfer segment does not retain chart clearance height")
                if offset == 2 and not (height[-1] < 0 and np.linalg.norm(lateral, axis=-1).max() <= 1e-4):
                    raise ValueError("Source approach segment is not a downward chart-normal move")
        index = end
    return result


def _curve(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Each source segment must be finite [N,3], N>=2")
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=-1) > 1e-10]
    unique = points[keep]
    if len(unique) < 2:
        raise ValueError("Each source segment requires nonzero arc length")
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(unique, axis=0), axis=-1))]
    return unique, arc


def _interpolate(points, arc, positions):
    return np.column_stack([np.interp(positions, arc, points[:, axis]) for axis in range(3)])


def _anchors(points, arc, tolerance):
    """RDP-like anchors using actual *arc-parameter* interpolation error.

    This preserves segment-internal height corners, not just overall endpoints.
    It does not identify or dictate retract/transfer/approach action templates.
    """
    selected = {0, len(points) - 1}
    pending = [(0, len(points) - 1)]
    while pending:
        first, last = pending.pop()
        if last <= first + 1:
            continue
        ratio = ((arc[first + 1:last] - arc[first]) / (arc[last] - arc[first]))[:, None]
        estimate = points[first] + ratio * (points[last] - points[first])
        error = np.linalg.norm(points[first + 1:last] - estimate, axis=-1)
        relative = int(error.argmax())
        if error[relative] > tolerance:
            index = first + 1 + relative
            selected.add(index)
            pending.extend([(first, index), (index, last)])
    return np.array(sorted(selected), dtype=np.int64)


def _integer_allocation(weights, count):
    if count == 0:
        return np.zeros(len(weights), dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    quotas = weights / weights.sum() * count
    result = np.floor(quotas).astype(np.int64)
    left = count - int(result.sum())
    order = np.argsort(-(quotas - result), kind="stable")
    result[order[:left]] += 1
    return result


def _validate_segments(trajectory, segments):
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3 or not np.isfinite(trajectory).all():
        raise ValueError("Full program must be finite [N,3]")
    if not isinstance(segments, list) or not segments:
        raise ValueError("Full program must contain explicit ordered segment metadata")
    cursor = 0
    curves = []
    for segment in segments:
        begin, end = segment.get("start"), segment.get("end")
        if (not isinstance(begin, int) or not isinstance(end, int) or begin != cursor
                or not begin + 2 <= end <= len(trajectory)):
            raise ValueError("Half-open segments must exactly cover the ordered raw trajectory without gaps/overlap")
        kind = segment.get("kind")
        if kind not in SOURCE_KINDS or segment.get("contact") is not (kind == "grind_stroke"):
            raise ValueError("Source kind/contact labels must explicitly agree")
        if kind == "grind_stroke":
            if (not isinstance(segment.get("pass_index"), int) or segment["pass_index"] < 0
                    or not isinstance(segment.get("surface_snapshot_index"), int)
                    or segment["surface_snapshot_index"] < 0
                    or not np.isfinite(segment.get("cut_level_chart_h_mm", np.nan))):
                raise ValueError("Contact segments require pass, snapshot and independent cutting-plane annotations")
        curve, arc = _curve(trajectory[begin:end])
        if cursor and not np.allclose(trajectory[cursor - 1], trajectory[begin], rtol=0, atol=1e-4):
            raise ValueError("Adjacent source segments are disconnected; do not invent connecting motion")
        curves.append((curve, arc))
        cursor = end
    if cursor != len(trajectory):
        raise ValueError("All original trajectory points must be covered by source segments")
    return trajectory, curves


def resample_complete_program(trajectory, segments: list[dict], count: int = 1024,
                              max_error_mm: float = 0.02) -> dict[str, np.ndarray]:
    """Represent the entire program at fixed length without removing air moves.

    Every segment keeps both endpoints and error-adaptive internal anchors.
    Remaining tokens are distributed over anchor intervals by arc length.
    With internal anchor tolerance max_error_mm/2, insertion of true source-curve
    samples retains an arc-parameter reconstruction bound <=max_error_mm.
    Insufficient capacity fails explicitly instead of dropping or merging stages.
    """
    if not isinstance(count, int) or count < 2:
        raise ValueError("Program sequence length must be an integer >=2")
    if not np.isfinite(max_error_mm) or max_error_mm <= 0:
        raise ValueError("max_error_mm must be finite and positive")
    raw, curves = _validate_segments(trajectory, segments)
    segment_modes = _segment_modes(segments)
    anchors = [_anchors(curve, arc, max_error_mm * 0.5) for curve, arc in curves]
    required = sum(len(indices) for indices in anchors)
    if count < required:
        raise ValueError(f"Sequence capacity {count} cannot preserve this complete program: requires {required} anchors; increase length")
    intervals = [arc[indices[1:]] - arc[indices[:-1]] for (_, arc), indices in zip(curves, anchors)]
    additional = _integer_allocation(np.concatenate(intervals), count - required)
    xyz, contacts, kinds, modes, ids, passes, snapshots, cut_levels, cut_valid, boundaries = [], [], [], [], [], [], [], [], [], []
    ranges, errors = [], []
    interval_cursor, token_cursor = 0, 0
    for index, ((curve, arc), indices, segment) in enumerate(zip(curves, anchors, segments)):
        positions = [arc[indices[0]]]
        for first, last in zip(indices[:-1], indices[1:]):
            extras = int(additional[interval_cursor])
            positions.extend(np.linspace(arc[first], arc[last], extras + 2)[1:].tolist())
            interval_cursor += 1
        positions = np.asarray(positions)
        sampled = _interpolate(curve, arc, positions)
        # Independent round-trip check in source arc coordinates, before FP32 I/O.
        reconstructed = _interpolate(sampled, positions, arc)
        maximum = float(np.linalg.norm(reconstructed - curve, axis=-1).max())
        if maximum > max_error_mm + 1e-8:
            raise RuntimeError("Program resampling exceeded its representation error budget")
        errors.append(maximum)
        n = len(sampled)
        contact = segment["contact"]
        xyz.append(sampled)
        contacts.extend([contact] * n)
        kinds.extend([SOURCE_KINDS[segment["kind"]]] * n)
        modes.extend([segment_modes[index]] * n)
        ids.extend([index] * n)
        passes.extend([segment.get("pass_index", -1)] * n)
        snapshots.extend([segment.get("surface_snapshot_index", -1)] * n)
        cut_levels.extend([segment.get("cut_level_chart_h_mm", 0.0)] * n)
        cut_valid.extend([contact] * n)
        marked = np.zeros(n, dtype=bool)
        marked[[0, -1]] = True
        boundaries.extend(marked.tolist())
        ranges.append([token_cursor, token_cursor + n])
        token_cursor += n
    stop = np.zeros(count, dtype=bool)
    stop[-1] = True
    return {
        "trajectory": np.concatenate(xyz).astype(np.float32),
        "trajectory_contact": np.asarray(contacts, dtype=bool),
        "trajectory_kind": np.asarray(kinds, dtype=np.int64),
        "trajectory_mode": np.asarray(modes, dtype=np.int64),
        "trajectory_segment": np.asarray(ids, dtype=np.int64),
        "trajectory_pass_index": np.asarray(passes, dtype=np.int64),
        "trajectory_snapshot_index": np.asarray(snapshots, dtype=np.int64),
        "trajectory_cut_level_mm": np.asarray(cut_levels, dtype=np.float32),
        "trajectory_cut_level_valid": np.asarray(cut_valid, dtype=bool),
        "trajectory_boundary": np.asarray(boundaries, dtype=bool),
        "trajectory_valid": np.ones(count, dtype=bool), "trajectory_stop": stop,
        "resampled_segment_ranges": np.asarray(ranges, dtype=np.int64),
        "segment_reconstruction_max_error_mm": np.asarray(errors, dtype=np.float64),
        "minimum_anchor_count": np.array(required, dtype=np.int64),
        "source_point_count": np.array(len(raw), dtype=np.int64),
    }


def load_collection_program(dataset_root, row: dict, *, trajectory_points=1024,
                            segment_points=128, max_error_mm=0.02) -> dict:
    """Read every raw point and segment; include per-segment 128-point labels."""
    root = Path(dataset_root).resolve()
    trajectory_path = (root / row["paths"]["trajectory_world"]).resolve()
    meta_path = (root / row["paths"]["trajectory_meta"]).resolve()
    for path in [trajectory_path, meta_path]:
        if not path.is_relative_to(root):
            raise ValueError("Program source path escapes collection")
    metadata = json.loads(meta_path.read_text())
    if (metadata.get("status") != "validated" or metadata.get("units") != "mm"
            or metadata.get("coordinate_frame") != "world"
            or metadata.get("segment_ranges_convention") != "half-open [start, end)"
            or metadata.get("representation") != "ordered_xyz_surface_contact_reference"):
        raise ValueError("Full program requires validated world-mm contact-reference metadata")
    raw = np.load(trajectory_path, allow_pickle=False)
    if metadata.get("point_count") != len(raw):
        raise ValueError("Program metadata point count mismatch")
    segments = metadata["segment_ranges"]
    raw, curves = _validate_segments(raw, segments)
    # These are known machining-chart calibration metadata, not predicted path
    # labels. They validate descriptive air-mode annotations only.
    if "asset_metadata" in row["paths"]:
        asset_path = (root / row["paths"]["asset_metadata"]).resolve()
        if not asset_path.is_relative_to(root):
            raise ValueError("Asset metadata path escapes collection")
        scene = json.loads(asset_path.read_text())["scene"]
        if scene.get("chart", {}).get("kind") in ("planar_box_top", "cylindrical_axial_side", "spherical_front_patch"):
            from .primitive_program_contract import validate_primitive_program
            normal_file = metadata.get("normal_file")
            if not isinstance(normal_file, str):
                raise ValueError("Primitive trajectory requires a normal_file")
            normal_path = (meta_path.parent / normal_file).resolve()
            if not normal_path.is_relative_to(root):
                raise ValueError("Primitive normal path escapes source collection")
            validate_primitive_program(scene, metadata, raw, normal_path=normal_path)
        else:
            rotation = np.asarray(scene["world_from_local"]["rotation"], dtype=np.float64)
            basis = np.asarray(scene["chart"]["basis_columns"], dtype=np.float64)
            _segment_modes(segments, raw, rotation @ basis[:, 2])
    if not isinstance(segment_points, int) or segment_points < 2:
        raise ValueError("segment_points must be an integer >=2")
    fixed_segments = []
    for curve, arc in curves:
        sampled = _interpolate(curve, arc, np.linspace(0, arc[-1], segment_points))
        fixed_segments.append(sampled)
    full = resample_complete_program(raw, segments, trajectory_points, max_error_mm)
    return {**full, "raw_trajectory_mm": raw.astype(np.float32),
            "segment_trajectories_mm": np.stack(fixed_segments).astype(np.float32),
            "segment_ranges": np.array(json.dumps(segments, ensure_ascii=False)),
            "program_metadata": np.array(json.dumps(metadata, ensure_ascii=False)),
            "program_schema": np.array(PROGRAM_SCHEMA), "units": np.array("mm"),
            "coordinate_frame": np.array("world"), "source_split": np.array(row["split"]),
            "source_segment_count": np.array(len(segments), dtype=np.int64)}


def _source_signature(root, row):
    keys = ["rgb", "depth", "camera", "trajectory_world", "trajectory_meta"]
    if "asset_metadata" in row["paths"]:
        keys.append("asset_metadata")
    result = {}
    for key in keys:
        if key not in row.get("paths", {}):
            raise ValueError(f"Program source requires {key}; legacy GT-sampled camera clouds cannot substitute")
        path = (root / row["paths"][key]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Missing or external program source file")
        result[key] = {"path": row["paths"][key], "sha256": sha256_file(path), "bytes": path.stat().st_size}
    return result


def freeze_program_selection(dataset_root, selection_file) -> Path:
    """Freeze all 300 source images for explicitly authorized memorization.

    Original train/val/test labels remain source_split records; this experiment
    intentionally trains on all of them and does not claim held-out evaluation.
    """
    root, rows, index_hash = _index(dataset_root)
    path = Path(selection_file).resolve()
    if path.is_relative_to(root):
        raise ValueError("Selection must be outside the read-only source collection")
    selected = []
    for row in rows:
        if row.get("units") != "mm" or row.get("coordinate_frame") != "world" or row.get("grindable") is not True:
            raise ValueError("Every selected program must be grindable world-mm supervision")
        selected.append({**row, "source_fingerprints": _source_signature(root, row)})
    selection_hash = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
    value = {"schema": "weldgrind-full-collection-overfit-selection-v1", "dataset_root": str(root),
             "dataset_index_sha256": index_hash, "count": len(selected), "selected": selected,
             "selection_sha256": selection_hash, "include_all_source_splits": True,
             "purpose": "all_images_memorization_not_heldout_evaluation", "generalization_claim": False}
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("Frozen program selection changed; automatic replacement is forbidden")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def export_program_collection(dataset_root, selection_file, predictions_dir, output_dir,
                               *, trajectory_points=1024, segment_points=128,
                               max_error_mm=0.02, roi_max_points=16384,
                               context_max_points=8192, cloud_seed=20261108) -> Path:
    """Export complete programs conditioned on actual SAM-predicted ROI only."""
    root, rows, index_hash = _index(dataset_root)
    selection = json.loads(Path(selection_file).read_text())
    if (Path(selection.get("dataset_root", "")).resolve() != root
            or selection.get("dataset_index_sha256") != index_hash):
        raise ValueError("Program selection must belong to the same completed collection")
    selected = selection.get("selected", [])
    indexed = {row["sample_id"]: row for row in rows}
    if (not selected or len(selected) != len(rows)
            or {row["sample_id"] for row in selected} != set(indexed)):
        raise ValueError("This authorized overfit export must include every source image exactly once")
    output = Path(output_dir).resolve()
    if output.is_relative_to(root) or output.exists():
        raise ValueError("Use a new output directory outside the read-only source collection")
    # Validate actual prediction presence and all fingerprints before writing.
    prepared, checkpoint_hashes = [], set()
    for frozen in selected:
        row = indexed[frozen["sample_id"]]
        signatures = _source_signature(root, row)
        if any(frozen.get("source_fingerprints", {}).get(key) != value for key, value in signatures.items()):
            raise ValueError("Program source changed after selection was frozen")
        raw_mask, prediction = _load_prediction(root, frozen, Path(predictions_dir).resolve())
        mask, mask_metadata = mask_postprocess(raw_mask, kernel_size=3,
            opening_iterations=0, dilation_iterations=1)
        if mask_metadata["processed_roi_empty"]:
            raise ValueError(f"SAM downstream ROI postprocess is empty for {row['sample_id']}: "
                f"raw_pixels={mask_metadata['raw_pixel_count']}, "
                f"processed_pixels={mask_metadata['processed_pixel_count']}; "
                "no GT fallback or manufactured ROI is permitted")
        arrays, cloud_metadata = load_predicted_rgbd_cloud(root, row, mask,
            roi_max_points=roi_max_points, context_max_points=context_max_points, seed=cloud_seed)
        arrays.update(raw_sam_mask=raw_mask.copy(),
            mask_postprocess_metadata=np.array(json.dumps(mask_metadata, sort_keys=True)),
            raw_sam_mask_pixel_count=np.array(mask_metadata["raw_pixel_count"], dtype=np.int64),
            processed_roi_mask_pixel_count=np.array(mask_metadata["processed_pixel_count"], dtype=np.int64))
        cloud_metadata["mask_postprocess"] = mask_metadata
        target = load_collection_program(root, row, trajectory_points=trajectory_points,
                                         segment_points=segment_points, max_error_mm=max_error_mm)
        arrays.update(target)
        checkpoint_hashes.add(prediction["checkpoint_sha256"])
        prepared.append((row, arrays, prediction, cloud_metadata, signatures))
    if len(checkpoint_hashes) != 1:
        raise ValueError("All full-program ROI predictions must use the same SAM checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    manifest_rows = []
    cloud_statistics = []
    for row, arrays, prediction, cloud_metadata, signatures in prepared:
        sample_id = row["sample_id"]
        npz_path = output / f"{sample_id}.npz"
        np.savez_compressed(npz_path, **arrays)
        entry = {"npz": npz_path.name, "sample_id": sample_id, "split": "overfit_all",
                 "source_split": row["split"], "units": "mm", "roi_source": "sam3_prediction",
                 "program_schema": PROGRAM_SCHEMA, "pointcloud_source": POINTCLOUD_SOURCE,
                 "target_scope": "complete_ordered_program_including_all_air_and_contact_segments"}
        manifest_rows.append(entry)
        cloud_statistics.append({"sample_id": sample_id, **cloud_metadata})
        (output / f"{sample_id}.json").write_text(json.dumps({**entry,
            "roi_points": cloud_metadata["sampled_predicted_roi_point_count"],
            "pointcloud": cloud_metadata,
            "prediction": prediction, "source_fingerprints": signatures,
            "program_points": int(arrays["source_point_count"]), "segments": int(arrays["source_segment_count"]),
            "max_representation_error_mm": float(arrays["segment_reconstruction_max_error_mm"].max())}, indent=2) + "\n")
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in manifest_rows))
    (output / "export_receipt.json").write_text(json.dumps({
        "schema": PROGRAM_SCHEMA, "dataset_root": str(root), "dataset_index_sha256": index_hash,
        "selection_sha256": sha256_file(selection_file), "count": len(selected),
        "sam_checkpoint_sha256": next(iter(checkpoint_hashes)), "roi_source": "sam3_prediction",
        "trajectory_points": trajectory_points, "segment_points": segment_points,
        "max_representation_error_mm": max_error_mm, "all_source_splits_used_for_overfit": True,
        "gt_mask_used_as_input": False, "generalization_claim": False,
        "source_program_has_initial_approach": False, "source_program_has_final_retract": False,
        "pointcloud_source": POINTCLOUD_SOURCE, "source_camera_pointcloud_read": False,
        "gt_defect_or_instance_sampling_used": False, "roi_max_points": roi_max_points,
        "context_max_points": context_max_points, "pointcloud_sampling_seed": cloud_seed,
        "source_depth_is_object_foreground_cropped": True,
        "sam_raw_prediction_and_iou_evaluation_unchanged": True,
        "roi_mask_postprocess": {"kernel_shape": "square", "kernel_size": 3,
            "opening_iterations": 0, "dilation_iterations": 1,
            "outside_image_value": False, "empty_result_policy": "explicit_failure_no_fallback"},
        "pointcloud_statistics_per_sample": cloud_statistics,
    }, indent=2) + "\n")
    return manifest


class ProgramDataset(Dataset):
    """Full-program data contract for direct ordered XYZ Flow Matching.

    All trajectory_* annotations are supervision, not inference-time inputs.
    No source pass/mode template or GT surface snapshot is fed to the encoder.
    """
    def __init__(self, manifest, max_points=2048, trajectory_points=1024, seed=20261108,
                 max_error_mm=0.02):
        self.manifest = Path(manifest).resolve()
        self.rows = [json.loads(line) for line in self.manifest.read_text().splitlines() if line.strip()]
        if not self.rows:
            raise ValueError("Program manifest must not be empty")
        self.max_points, self.trajectory_points = max_points, trajectory_points
        self.seed, self.max_error_mm = seed, max_error_mm

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row["npz"])
        if not path.is_absolute():
            path = self.manifest.parent / path
        with np.load(path, allow_pickle=False) as arrays:
            if (str(arrays["program_schema"].item()) != PROGRAM_SCHEMA
                    or str(arrays["units"].item()) != "mm"
                    or str(arrays["roi_source"].item()) != "sam3_prediction"
                    or "pointcloud_source" not in arrays
                    or str(arrays["pointcloud_source"].item()) != POINTCLOUD_SOURCE):
                raise ValueError("ProgramDataset requires full-program world-mm targets, actual SAM ROI and RGB-D backprojection without GT defect sampling")
            prepared = prepare_point_cloud(arrays["points"], arrays["roi_mask"], self.max_points, self.seed + index)
            segments = json.loads(str(arrays["segment_ranges"].item()))
            target = resample_complete_program(arrays["raw_trajectory_mm"], segments, self.trajectory_points, self.max_error_mm)
        prepared["trajectory"] = (torch.from_numpy(target["trajectory"]) - prepared["center"]) / prepared["scale"]
        for name, value in target.items():
            if name.startswith("trajectory_"):
                prepared[name] = torch.from_numpy(value)
        prepared["phase_labels"] = prepared["trajectory_mode"]
        prepared["segment_ids"] = prepared["trajectory_segment"]
        prepared["sample_id"] = row["sample_id"]
        prepared["source_split"] = row.get("source_split", "unspecified")
        prepared["segment_count"] = torch.tensor(len(segments), dtype=torch.long)
        prepared["source_point_count"] = torch.tensor(int(target["source_point_count"]), dtype=torch.long)
        return prepared
