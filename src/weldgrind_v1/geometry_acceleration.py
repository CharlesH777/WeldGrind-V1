"""Equivalent geometry kernels for the fixed, non-augmented overfit experiment.

Vectorization removes 27 host-synchronizing nonzero/gather/GEMM loops per
convolution. FPS caches *indices only*, never learned features or target paths.
Checkpoint parameter names and the original 27-offset convolution stay intact.
"""

from __future__ import annotations

import hashlib
import types

import torch

from .geometry import OccupiedSparseConv3d, farthest_point_indices


def vectorized_sparse_forward(self, coordinates, features):
    if coordinates.ndim != 2 or coordinates.shape[-1] != 3 or len(coordinates) != len(features) or not len(coordinates):
        raise ValueError("Sparse convolution requires nonempty aligned [V,3] coordinates and features")
    output_coords = coordinates if self.stride == 1 else torch.unique(
        torch.div(coordinates, 2, rounding_mode="floor"), dim=0, sorted=True)
    shape = coordinates.amax(0) + 1
    keys = (coordinates[:, 0] * shape[1] + coordinates[:, 1]) * shape[2] + coordinates[:, 2]
    keys, order = keys.sort()
    query = output_coords[:, None, :] * self.stride + self.offsets[None]
    valid = ((query >= 0) & (query < shape)).all(-1)
    query_keys = (query[..., 0] * shape[1] + query[..., 1]) * shape[2] + query[..., 2]
    lookup = torch.searchsorted(keys, query_keys).clamp(max=len(keys) - 1)
    valid = valid & (keys[lookup] == query_keys)
    gathered = features[order[lookup]] * valid[..., None].to(features.dtype)
    # Same sum of 27 offset-specific learned matrix products as the reference.
    # Keeping geometry FP32 avoids unsafe global bf16 index_add accumulation.
    result = gathered.reshape(len(output_coords), -1) @ self.weight.reshape(-1, self.weight.shape[-1])
    return output_coords, result + self.bias


class CachedInputFPS:
    """Cache deterministic FPS on the observed XYZ, with exact input signatures.

    The small D2H signature transfer replaces thousands of tiny CUDA launches.
    Coordinates are detached for sampling exactly as in the reference FPS.
    A changed cloud/ROI/order produces a different signature. No labels enter.
    """

    def __init__(self, maximum_entries=512):
        self.maximum_entries = maximum_entries
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def __call__(self, xyz, count):
        cpu = xyz.detach().cpu().contiguous()
        signature = (hashlib.sha256(cpu.numpy().tobytes()).hexdigest(), tuple(cpu.shape), str(cpu.dtype),
                     count, str(xyz.device))
        if signature in self.cache:
            self.hits += 1
            return self.cache[signature]
        self.misses += 1
        indices = farthest_point_indices(cpu, count).to(xyz.device)
        if len(self.cache) >= self.maximum_entries:
            self.cache.pop(next(iter(self.cache)))
        self.cache[signature] = indices
        return indices


def accelerate_geometry(model):
    """Install before training; do not change active model code during a run."""
    changed = 0
    for layer in model.geometry.modules():
        if isinstance(layer, OccupiedSparseConv3d):
            layer.forward = types.MethodType(vectorized_sparse_forward, layer)
            changed += 1
    sampler = CachedInputFPS()
    model.geometry.point_sampler = sampler
    return {"sparse_layers_vectorized": changed, "fps": sampler,
            "feature_cache": False, "target_inputs": False, "geometry_dtype": "float32"}
