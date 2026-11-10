"""SAM-conditioned point clouds reconstructed from noisy RGB-D pixels.

The supplied collection camera_pointcloud is intentionally never opened here:
its defect-preserving sampling uses ground-truth defect/instance information.
This adapter uses only RGB, noisy optical-Z depth, calibration and the actual
SAM-predicted binary mask. Source depth already excludes non-object pixels;
that renderer preprocessing is recorded rather than called raw sensor depth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


POINTCLOUD_SOURCE = "sam_prediction_and_noisy_rgbd_backprojection_no_gt_defect_sampling"
DEPTH_PREPROCESSING = "source_depth_already_object_foreground_cropped_by_renderer"


def _binary_square_dilation(mask, kernel_size):
    radius = kernel_size // 2
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask)
    for y in range(kernel_size):
        for x in range(kernel_size):
            shifted = padded[y:y+height, x:x+width]
            result |= shifted
    return result


def mask_postprocess(raw, kernel_size=3, opening_iterations=0,
                     dilation_iterations=1) -> tuple[np.ndarray, dict]:
    """Downstream ROI dilation only; never opens/erodes or alters SAM's raw mask.

    The user's final instruction explicitly disables opening. A nonzero opening
    argument is rejected rather than silently removing a small predicted region.
    Dilation uses a square kernel and constant False outside the image. Empty
    source masks remain empty and are explicitly marked for the export caller.
    """
    raw = np.asarray(raw)
    if raw.ndim != 2 or not raw.size or raw.dtype != np.bool_:
        raise ValueError("Raw SAM mask must be nonempty bool [H,W]")
    if (not isinstance(kernel_size, int) or isinstance(kernel_size, bool)
            or kernel_size < 1 or kernel_size % 2 != 1):
        raise ValueError("kernel_size must be a positive odd integer")
    for name, value in [("opening_iterations", opening_iterations), ("dilation_iterations", dilation_iterations)]:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if opening_iterations != 0:
        raise ValueError("Opening is disabled: use dilation only, without erosion or denoising")
    processed = raw.copy()
    for _ in range(dilation_iterations):
        processed = _binary_square_dilation(processed, kernel_size)
    metadata = {
        "schema": "sam-predicted-downstream-roi-postprocess-v1",
        "operations": ["binary_dilation"],
        "kernel_shape": "square", "kernel_size": kernel_size,
        "opening_iterations": opening_iterations, "dilation_iterations": dilation_iterations,
        "outside_image_value": False, "raw_mask_preserved": True,
        "raw_pixel_count": int(raw.sum()),
        "processed_pixel_count": int(processed.sum()),
        "opening_applied": False, "erosion_applied": False,
        "removed_raw_pixel_count": int((raw & ~processed).sum()),
        "processed_roi_empty": bool(not processed.any()),
        "scope": "downstream pointcloud ROI only; SAM training/raw-prediction/IoU evaluation unchanged",
    }
    return processed, metadata


def _limit(value, name, allow_zero=False):
    if not isinstance(value, int) or isinstance(value, bool) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be an integer {'>=0' if allow_zero else '>0'}")


def fixed_cloud_seed(sample_id: str, seed: int = 20261108) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()[:8], "big") % (2**63 - 1)


def _sample_pixels(candidates, limit, rng):
    if len(candidates) <= limit:
        return candidates.copy()
    # Uniform without replacement, then canonical row-major order. No repeated
    # coordinates are manufactured to fill a fixed quota for a small ROI.
    return np.sort(rng.choice(candidates, size=limit, replace=False))


def backproject_predicted_rgbd(rgb, noisy_depth_mm, camera: dict, predicted_mask,
                               *, roi_max_points=16384, context_max_points=8192,
                               seed=20261108) -> tuple[dict[str, np.ndarray], dict]:
    """Reconstruct unique visible XYZRGB pixels, in world millimetres.

    Pixel indices are integers (u,v); rays use (u+pixel_center_offset,
    v+pixel_center_offset), preserving this collection's explicit .5 convention.
    Context is uniformly sampled outside the *entire* predicted mask, so it
    cannot add hidden GT-selected defect points or exceed the ROI point cap.
    """
    _limit(roi_max_points, "roi_max_points")
    _limit(context_max_points, "context_max_points", allow_zero=True)
    rgb = np.asarray(rgb)
    depth = np.asarray(noisy_depth_mm, dtype=np.float64)
    mask = np.asarray(predicted_mask)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("RGB must be uint8 [H,W,3]")
    if depth.shape != rgb.shape[:2] or mask.shape != depth.shape or mask.dtype != np.bool_:
        raise ValueError("noisy_depth_mm and bool SAM mask must match RGB [H,W]")
    if (camera.get("capture_status") != "accepted" or camera.get("world_units") != "mm"
            or camera.get("image_size_wh") != [rgb.shape[1], rgb.shape[0]]):
        raise ValueError("Camera must identify an accepted matching-resolution world-mm capture")
    definition = str(camera.get("depth_definition", ""))
    if definition != "distance_to_image_plane, along optical +Z":
        raise ValueError("Depth must explicitly be optical-Z distance_to_image_plane in millimetres")
    offset = camera.get("pixel_center_offset")
    if not isinstance(offset, (int, float)) or not np.isfinite(offset) or not 0 <= offset < 1:
        raise ValueError("Calibration must explicitly supply pixel_center_offset in [0,1)")
    intrinsics = np.asarray(camera.get("intrinsics_K"), dtype=np.float64)
    transform = np.asarray(camera.get("world_from_camera_cv_mm"), dtype=np.float64)
    if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
            or intrinsics[0,0] <= 0 or intrinsics[1,1] <= 0
            or not np.allclose(intrinsics[2], [0,0,1]) or abs(np.linalg.det(intrinsics)) < 1e-12):
        raise ValueError("intrinsics_K must be a nonsingular finite optical-camera calibration")
    if (transform.shape != (4,4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0,0,0,1])):
        raise ValueError("world_from_camera_cv_mm must be finite homogeneous [4,4]")
    rotation = transform[:3,:3]
    if (not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
        raise ValueError("Camera extrinsic rotation must be rigid and right-handed")
    valid = np.isfinite(depth) & (depth > 0)
    roi_candidates = np.flatnonzero(valid & mask)
    context_candidates = np.flatnonzero(valid & ~mask)
    if not len(roi_candidates):
        raise ValueError("Actual SAM prediction contains no finite positive-depth ROI pixel")
    roi_seed, context_seed = np.random.SeedSequence(seed).spawn(2)
    roi_indices = _sample_pixels(roi_candidates, roi_max_points, np.random.default_rng(roi_seed))
    context_indices = _sample_pixels(context_candidates, context_max_points, np.random.default_rng(context_seed))
    flat_indices = np.concatenate([roi_indices, context_indices])
    v, u = np.divmod(flat_indices, depth.shape[1])
    pixel_uv = np.column_stack([u,v]).astype(np.int32)
    homogeneous_pixels = np.column_stack([u+offset, v+offset, np.ones(len(u))])
    rays = homogeneous_pixels @ np.linalg.inv(intrinsics).T
    camera_xyz = rays * (depth[v,u] / rays[:,2])[:,None]
    world_xyz = camera_xyz @ rotation.T + transform[:3,3]
    if not np.isfinite(world_xyz).all() or np.any(np.abs(world_xyz) > np.finfo(np.float32).max):
        raise ValueError("Backprojected world coordinates are not finite float32-representable XYZ")
    roi = np.r_[np.ones(len(roi_indices),dtype=bool), np.zeros(len(context_indices),dtype=bool)]
    points = np.column_stack([world_xyz, rgb[v,u].astype(np.float32)/255]).astype(np.float32)
    metadata = {
        "pointcloud_source": POINTCLOUD_SOURCE, "roi_source": "sam3_prediction",
        "units": "mm", "coordinate_frame": "world", "depth_field": "noisy_depth_mm",
        "source_depth_preprocessing": DEPTH_PREPROCESSING,
        "source_depth_is_object_foreground_cropped": True,
        "source_camera_pointcloud_read": False, "gt_defect_sampling_used": False,
        "gt_defect_or_instance_labels_read": False,
        "valid_depth_rule": "finite(noisy_depth_mm) and noisy_depth_mm>0; persisted valid masks not read",
        "roi_sampling": "all valid SAM ROI pixels up to cap; uniform without replacement when capped",
        "context_sampling": "uniform valid depth pixels outside complete SAM prediction; no ROI pixel duplication",
        "sampling_seed": int(seed), "roi_max_points": roi_max_points, "context_max_points": context_max_points,
        "valid_depth_pixel_count": int(valid.sum()), "predicted_mask_pixel_count": int(mask.sum()),
        "predicted_mask_invalid_depth_pixel_count": int((mask & ~valid).sum()),
        "valid_predicted_roi_pixel_count": len(roi_candidates),
        "sampled_predicted_roi_point_count": len(roi_indices),
        "context_pool_pixel_count": len(context_candidates),
        "sampled_context_point_count": len(context_indices), "point_count": len(points),
        "unique_pixel_count": len(np.unique(flat_indices)), "pixel_center_offset": float(offset),
    }
    arrays = {
        "points": points, "roi_mask": roi, "mask": mask,
        "point_uv": pixel_uv, "xyz_camera_cv_mm": camera_xyz.astype(np.float32),
        "intrinsics": intrinsics, "camera_from_world": np.linalg.inv(transform),
        "pixel_center_offset": np.array(float(offset)), "units": np.array("mm"),
        "coordinate_frame": np.array("world"), "roi_source": np.array("sam3_prediction"),
        "pointcloud_source": np.array(POINTCLOUD_SOURCE),
        "source_depth_preprocessing": np.array(DEPTH_PREPROCESSING),
        "source_depth_is_object_foreground_cropped": np.array(True),
        "valid_predicted_roi_pixel_count": np.array(len(roi_candidates),dtype=np.int64),
        "sampled_predicted_roi_point_count": np.array(len(roi_indices),dtype=np.int64),
        "sampled_context_point_count": np.array(len(context_indices),dtype=np.int64),
        "predicted_mask_invalid_depth_pixel_count": np.array(int((mask & ~valid).sum()),dtype=np.int64),
        "pointcloud_metadata": np.array(json.dumps(metadata, sort_keys=True)),
    }
    return arrays, metadata


def load_predicted_rgbd_cloud(dataset_root, row: dict, predicted_mask, *,
                              roi_max_points=16384, context_max_points=8192,
                              seed=20261108) -> tuple[dict[str, np.ndarray], dict]:
    """Read only RGB/noisy depth/calibration files listed in the source index."""
    root = Path(dataset_root).resolve()
    paths = {}
    for key in ["rgb", "depth", "camera"]:
        if key not in row.get("paths", {}):
            raise ValueError(f"Predicted RGB-D reconstruction requires source path {key}; no legacy cloud fallback")
        path = (root / row["paths"][key]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or external RGB-D source path: {key}")
        paths[key] = path
    with Image.open(paths["rgb"]) as image:
        rgb = np.asarray(image.convert("RGB"))
    with np.load(paths["depth"], allow_pickle=False) as source:
        if "noisy_depth_mm" not in source.files:
            raise ValueError("Source depth NPZ lacks noisy_depth_mm; clean depth/legacy cloud cannot substitute")
        noisy_depth = source["noisy_depth_mm"]
    camera = json.loads(paths["camera"].read_text())
    if camera.get("image_size_wh") != row.get("image_size_wh"):
        raise ValueError("Indexed image size and RGB-D calibration disagree")
    return backproject_predicted_rgbd(rgb, noisy_depth, camera, predicted_mask,
        roi_max_points=roi_max_points, context_max_points=context_max_points,
        seed=fixed_cloud_seed(row["sample_id"],seed))
