"""Explicit millimetre data contracts and tiny RGB-D fixtures for code checks.

No weld training data is supplied or imported. A manifest is JSONL, each row
containing ``npz`` and optionally ``sample_id``. Each NPZ contains ``points``
[N,6] (XYZ mm, RGB 0..1), ``roi_mask`` [N] and one continuous surface-contact
``trajectory`` [M,3] in the same frame/unit. Split discontinuous strokes before
export: air moves and multiple contact strokes are rejected when annotated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _point_arrays(points, roi_mask):
    points = np.asarray(points, dtype=np.float32)
    roi = np.asarray(roi_mask)
    if points.ndim != 2 or points.shape[1] != 6 or not len(points):
        raise ValueError("points must be nonempty [N,6]: XYZ mm followed by RGB [0,1]")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite XYZRGB values")
    if np.any(points[:, 3:] < 0) or np.any(points[:, 3:] > 1):
        raise ValueError("RGB values must be in [0,1]")
    if roi.shape != (len(points),) or not np.isin(roi, [False, True]).all():
        raise ValueError("roi_mask must be binary [N]")
    roi = roi.astype(bool)
    if not roi.any():
        raise ValueError("ROI is empty: no surface seeds or trajectory can be inferred")
    return points, roi


def prepare_point_cloud(points, roi_mask, max_points: int = 4096, seed: int = 0) -> dict[str, torch.Tensor]:
    """Normalize in the full ROI frame, then sample and pad cloud/context.

    At least min(64, number of ROI points, max_points) unique ROI points survive
    sampling. Normalization is computed before sampling and shared by the whole
    cloud and trajectory. ``center`` is ROI bounding-box centre and ``scale`` is
    its maximum radius, clamped to 1 mm. Padding is explicitly invalid.
    """
    if not isinstance(max_points, int) or max_points < 1:
        raise ValueError("max_points must be a positive integer")
    points, roi = _point_arrays(points, roi_mask)
    roi_xyz = points[roi, :3]
    center = (roi_xyz.min(axis=0) + roi_xyz.max(axis=0)) * 0.5
    scale = np.float32(max(float(np.linalg.norm(roi_xyz - center, axis=1).max()), 1.0))
    count = min(len(points), max_points)
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        roi_indices = np.flatnonzero(roi)
        roi_budget = min(len(roi_indices), max_points, max(min(64, max_points), max_points // 2))
        selected_roi = rng.choice(roi_indices, roi_budget, replace=False)
        remaining = np.setdiff1d(np.arange(len(points)), selected_roi, assume_unique=False)
        selected_other = rng.choice(remaining, max_points - roi_budget, replace=False)
        selected = np.concatenate([selected_roi, selected_other])
        rng.shuffle(selected)
    else:
        selected = np.arange(len(points))
    sampled = points[selected].copy()
    sampled[:, :3] = (sampled[:, :3] - center) / scale
    padded = np.zeros((max_points, 6), dtype=np.float32)
    padded[:count] = sampled
    padded_roi, valid = np.zeros(max_points, dtype=bool), np.zeros(max_points, dtype=bool)
    padded_roi[:count], valid[:count] = roi[selected], True
    return {
        "points": torch.from_numpy(padded), "roi_mask": torch.from_numpy(padded_roi),
        "point_valid": torch.from_numpy(valid), "center": torch.from_numpy(center.copy()),
        "scale": torch.tensor(scale, dtype=torch.float32),
    }


def resample_trajectory(points, count: int = 128) -> np.ndarray:
    """Resample one continuous ordered stroke by arc length, retaining endpoints."""
    curve = np.asarray(points, dtype=np.float64)
    if curve.ndim != 2 or curve.shape[1] != 3 or len(curve) < 2 or not np.isfinite(curve).all():
        raise ValueError("trajectory must be finite [M,3] with M >= 2")
    if not isinstance(count, int) or count < 2:
        raise ValueError("trajectory point count must be an integer >= 2")
    keep = np.r_[True, np.linalg.norm(np.diff(curve, axis=0), axis=1) > 1e-10]
    curve = curve[keep]
    if len(curve) < 2:
        raise ValueError("trajectory must have nonzero total arc length")
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    sample_arc = np.linspace(0, arc[-1], count)
    return np.column_stack([np.interp(sample_arc, arc, curve[:, axis]) for axis in range(3)]).astype(np.float32)


def _single_stroke_contract(arrays, row: dict, length: int):
    units = str(arrays["units"].item()) if "units" in arrays else row.get("units", "mm")
    if units != "mm":
        raise ValueError("NPZ contract requires XYZ and trajectory in mm; convert units explicitly")
    if "contact" in arrays:
        contact = np.asarray(arrays["contact"])
        if contact.shape != (length,) or not np.all(contact == 1):
            raise ValueError("Air moves/mixed contact labels require explicit stroke segmentation before training")
    segments = row.get("segment_ranges")
    if "segment_ranges" in arrays:
        segments = json.loads(str(arrays["segment_ranges"].item()))
    if segments is not None:
        if (len(segments) != 1 or not segments[0].get("contact", False)
                or segments[0].get("start") != 0 or segments[0].get("end") != length):
            raise ValueError("Each sample must contain exactly one full continuous contact stroke; split segments first")


class WeldDataset(Dataset):
    """Generic supervised contract, ready for future real RGB-D/trajectory data."""

    def __init__(self, manifest: str | Path, max_points: int = 4096, trajectory_points: int = 128, seed: int = 0):
        self.manifest = Path(manifest).expanduser().resolve()
        if not isinstance(max_points, int) or max_points < 1:
            raise ValueError("max_points must be a positive integer")
        if not isinstance(trajectory_points, int) or trajectory_points < 2:
            raise ValueError("trajectory_points must be an integer >= 2")
        self.max_points, self.trajectory_points, self.seed = max_points, trajectory_points, seed
        self.rows = []
        for line_number, line in enumerate(self.manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("npz"), str):
                raise ValueError(f"Manifest line {line_number} must contain a string npz path")
            self.rows.append(row)
        if not self.rows:
            raise ValueError("Manifest must contain at least one sample")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        path = Path(row["npz"])
        if not path.is_absolute():
            path = self.manifest.parent / path
        with np.load(path, allow_pickle=False) as arrays:
            required = {"points", "roi_mask", "trajectory"}
            if not required.issubset(arrays.files):
                raise ValueError(f"Missing required NPZ keys in {path}: {sorted(required - set(arrays.files))}")
            curve = arrays["trajectory"]
            _single_stroke_contract(arrays, row, len(curve))
            batch = prepare_point_cloud(arrays["points"], arrays["roi_mask"], self.max_points, self.seed + index)
            curve = resample_trajectory(curve, self.trajectory_points)
        batch["trajectory"] = (torch.from_numpy(curve) - batch["center"]) / batch["scale"]
        batch["sample_id"] = row.get("sample_id", path.stem)
        return batch


