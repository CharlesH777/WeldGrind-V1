"""PV-Ground-inspired sparse geometry encoder, with a portable Torch backend.

This is a new model, not a checkpoint-compatible copy of PV-Ground.  Its
occupied-coordinate convolutions have a separate learned matrix for each of
the 27 spatial offsets.  No dense volume or CUDA extension is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch
from torch import Tensor, nn


@dataclass
class GeometryConfig:
    dim: int = 256
    candidates: int = 1024
    voxel_size: float = 0.03
    channels: tuple[int, ...] = (64, 64, 128, 128)
    neighbors: int = 16
    branch_dim: int = 48
    radius_multiplier: float = 2.5
    query_chunk_size: int = 256
    backend: str = "torch"

    def __post_init__(self):
        self.channels = tuple(self.channels)
        if self.backend != "torch":
            raise ValueError("Only the explicit 'torch' sparse backend is implemented")
        if len(self.channels) != 4 or min(self.channels) < 1:
            raise ValueError("channels must contain four positive stage widths")
        if min(self.dim, self.candidates, self.neighbors, self.branch_dim, self.query_chunk_size) < 1:
            raise ValueError("Geometry dimensions, counts and chunk size must be positive")
        if self.voxel_size <= 0 or self.radius_multiplier <= 0:
            raise ValueError("Voxel size and radius multiplier must be positive")


@torch.no_grad()
def farthest_point_indices(xyz: Tensor, count: int) -> Tensor:
    """Deterministic FPS with unique row indices, even for duplicate points.

    FPS selects input rows, so it neither moves points nor invents seed
    coordinates.  Empty input returns empty indices; count must be positive.
    """
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError("xyz must have shape [N,3]")
    if count < 1:
        raise ValueError("count must be positive")
    n = len(xyz)
    count = min(count, n)
    chosen = torch.empty(count, dtype=torch.long, device=xyz.device)
    if not n:
        return chosen
    distances = xyz.new_full((n,), float("inf"))
    used = torch.zeros(n, dtype=torch.bool, device=xyz.device)
    current = ((xyz - xyz.mean(0)) ** 2).sum(-1).argmax()
    for i in range(count):
        chosen[i] = current
        used[current] = True
        distances = torch.minimum(distances, ((xyz - xyz[current]) ** 2).sum(-1))
        distances.masked_fill_(used, -1)
        current = distances.argmax()
    return chosen


def _mean_voxel_features(coordinates: Tensor, features: Tensor) -> tuple[Tensor, Tensor]:
    coordinates, inverse, counts = torch.unique(
        coordinates, dim=0, sorted=True, return_inverse=True, return_counts=True,
    )
    sums = features.new_zeros((len(coordinates), features.shape[-1]))
    sums.index_add_(0, inverse, features)
    return coordinates, sums / counts.to(features.dtype).unsqueeze(-1)


class OccupiedSparseConv3d(nn.Module):
    """3x3x3 convolution over occupied coordinates, with stride one or two.

    At stride one this is a submanifold convolution.  At stride two the output
    coordinate set consists of occupied parent cells floor(input/2).  This
    deliberate restriction differs from spconv's activation expansion, while
    retaining learned, offset-specific spatial convolution and receptive field.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("Sparse convolution supports stride one or two")
        self.stride = stride
        self.weight = nn.Parameter(torch.empty(27, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("offsets", torch.tensor(list(product((-1, 0, 1), repeat=3))), persistent=False)
        nn.init.normal_(self.weight, std=(27 * in_channels) ** -0.5)

    def forward(self, coordinates: Tensor, features: Tensor) -> tuple[Tensor, Tensor]:
        if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
            raise ValueError("Sparse coordinates must be [V,3]")
        if len(coordinates) != len(features) or not len(coordinates):
            raise ValueError("Sparse convolution needs nonempty, aligned coordinates/features")
        output_coords = coordinates if self.stride == 1 else torch.unique(
            torch.div(coordinates, 2, rounding_mode="floor"), dim=0, sorted=True,
        )
        # Coordinates are nonnegative. Mixed-radix keys are collision-free
        # inside bounds; the explicit bounds mask also rejects negative queries.
        shape = coordinates.amax(0) + 1
        keys = (coordinates[:, 0] * shape[1] + coordinates[:, 1]) * shape[2] + coordinates[:, 2]
        keys, order = keys.sort()
        output = features.new_zeros((len(output_coords), self.weight.shape[-1]))
        for kernel_index, offset in enumerate(self.offsets):
            query = output_coords * self.stride + offset
            inside = ((query >= 0) & (query < shape)).all(-1)
            query_keys = (query[:, 0] * shape[1] + query[:, 1]) * shape[2] + query[:, 2]
            lookup = torch.searchsorted(keys, query_keys).clamp(max=len(keys) - 1)
            matches = inside & (keys[lookup] == query_keys)
            rows = matches.nonzero(as_tuple=True)[0]
            if len(rows):
                contribution = features[order[lookup[rows]]] @ self.weight[kernel_index]
                output.index_add_(0, rows, contribution)
        return output_coords, output + self.bias


class SparseVoxelBackbone(nn.Module):
    """MeanVFE followed by four sparse stages at strides 1, 2, 4, and 8."""

    def __init__(self, config: GeometryConfig):
        super().__init__()
        self.config = config
        widths = (7, *config.channels)
        self.convolutions = nn.ModuleList([
            OccupiedSparseConv3d(widths[i], widths[i + 1], stride=1 if i == 0 else 2)
            for i in range(4)
        ])
        self.refinements = nn.ModuleList([
            OccupiedSparseConv3d(width, width) for width in config.channels
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for width in config.channels])
        self.refine_norms = nn.ModuleList([nn.LayerNorm(width) for width in config.channels])
        self.activation = nn.GELU()

    def forward(self, xyz: Tensor, rgb: Tensor, roi: Tensor) -> list[tuple[Tensor, Tensor]]:
        # Per-cloud origin makes feature geometry invariant to translations.
        origin = xyz.amin(0)
        local_xyz = xyz - origin
        coordinates = torch.floor(local_xyz / self.config.voxel_size + 1e-6).long()
        values = torch.cat((local_xyz, rgb, roi.to(xyz.dtype).unsqueeze(-1)), -1)
        coordinates, features = _mean_voxel_features(coordinates, values)
        stages = []
        for stage, (conv, refine, norm, refine_norm) in enumerate(zip(
            self.convolutions, self.refinements, self.norms, self.refine_norms,
        )):
            coordinates, features = conv(coordinates, features)
            features = self.activation(norm(features))
            _, residual = refine(coordinates, features)
            features = self.activation(refine_norm(residual) + features)
            centers = (coordinates.to(xyz.dtype) + 0.5) * (self.config.voxel_size * 2 ** stage) + origin
            stages.append((centers, features))
        return stages


class PointVoxelAggregation(nn.Module):
    """Radius-limited kNN grouping, relative-position MLP, and masked max pool."""

    def __init__(self, source_dim: int, config: GeometryConfig):
        super().__init__()
        self.config = config
        self.mlp = nn.Sequential(
            nn.Linear(source_dim + 3, config.branch_dim), nn.GELU(),
            nn.Linear(config.branch_dim, config.branch_dim), nn.GELU(),
        )

    def forward(self, query: Tensor, xyz: Tensor, features: Tensor, radius: float) -> Tensor:
        groups = []
        for chunk in query.split(self.config.query_chunk_size):
            distance = torch.cdist(chunk.unsqueeze(0), xyz.unsqueeze(0)).squeeze(0)
            distance, neighbors = distance.topk(min(self.config.neighbors, len(xyz)), largest=False)
            valid = distance <= radius
            offsets = (xyz[neighbors] - chunk[:, None, :]) / radius
            grouped = self.mlp(torch.cat((offsets, features[neighbors]), -1))
            grouped = grouped.masked_fill(~valid[..., None], -torch.inf)
            pooled = grouped.amax(1)
            pooled = torch.where(valid.any(1, keepdim=True), pooled, torch.zeros_like(pooled))
            groups.append(pooled)
        return torch.cat(groups)


class GeometryEncoder(nn.Module):
    """Encode full-cloud context and return padded surface-only ROI candidates.

    Args to forward:
        points: [B,N,6], normalized XYZ followed by RGB in [0,1].
        roi_mask: [B,N] boolean; identifies grindable observed surface points.
        point_valid: optional [B,N] boolean mask for variable-size input padding.

    Returns candidate_xyz [B,K,3], candidate_features [B,K,D], and boolean
    candidate_valid/candidate_roi [B,K]. Only true-valid entries may be selected.
    A sample with no valid ROI fails explicitly rather than generating a path.
    """

    def __init__(self, config: GeometryConfig | None = None):
        super().__init__()
        self.config = config or GeometryConfig()
        self.backbone = SparseVoxelBackbone(self.config)
        self.raw_aggregation = PointVoxelAggregation(4, self.config)
        self.voxel_aggregations = nn.ModuleList([
            PointVoxelAggregation(width, self.config) for width in self.config.channels
        ])
        self.fusion = nn.Sequential(
            nn.Linear(self.config.branch_dim * 5, self.config.dim),
            nn.LayerNorm(self.config.dim), nn.GELU(),
        )

    def forward(self, points: Tensor, roi_mask: Tensor, point_valid: Tensor | None = None) -> dict[str, Tensor]:
        if points.ndim != 3 or points.shape[-1] != 6 or points.shape[0] == 0:
            raise ValueError("points must be nonempty [B,N,6]")
        if not points.is_floating_point():
            raise ValueError("points must be floating point")
        if roi_mask.shape != points.shape[:2] or roi_mask.dtype != torch.bool:
            raise ValueError("roi_mask must be boolean [B,N]")
        if point_valid is None:
            point_valid = torch.ones_like(roi_mask)
        if point_valid.shape != roi_mask.shape or point_valid.dtype != torch.bool:
            raise ValueError("point_valid must be boolean [B,N]")
        if not torch.isfinite(points[point_valid]).all():
            raise ValueError("Valid points must be finite")
        valid_rgb = points[..., 3:][point_valid]
        if ((valid_rgb < 0) | (valid_rgb > 1)).any():
            raise ValueError("RGB must be normalized to [0,1]")
        all_xyz, all_features, all_valid = [], [], []
        for sample, roi, valid in zip(points, roi_mask, point_valid):
            sample, roi = sample[valid], roi[valid]
            if not len(sample) or not roi.any():
                raise ValueError("Each sample requires at least one valid ROI surface point")
            xyz, rgb = sample[:, :3], sample[:, 3:]
            roi_xyz = xyz[roi]
            sampler = getattr(self, "point_sampler", farthest_point_indices)
            selected = sampler(roi_xyz, self.config.candidates)
            candidate_xyz = roi_xyz[selected]
            raw_features = torch.cat((rgb, roi.to(rgb.dtype).unsqueeze(-1)), -1)
            radius = self.config.radius_multiplier * self.config.voxel_size
            fused = [self.raw_aggregation(candidate_xyz, xyz, raw_features, radius)]
            for stage, ((voxel_xyz, voxel_features), aggregation) in enumerate(zip(
                self.backbone(xyz, rgb, roi), self.voxel_aggregations,
            )):
                fused.append(aggregation(candidate_xyz, voxel_xyz, voxel_features, radius * 2 ** stage))
            features = self.fusion(torch.cat(fused, -1))
            padding = self.config.candidates - len(candidate_xyz)
            all_xyz.append(torch.cat((candidate_xyz, candidate_xyz.new_zeros((padding, 3)))))
            all_features.append(torch.cat((features, features.new_zeros((padding, self.config.dim)))))
            all_valid.append(torch.arange(self.config.candidates, device=points.device) < len(candidate_xyz))
        candidate_valid = torch.stack(all_valid)
        return {
            "candidate_xyz": torch.stack(all_xyz),
            "candidate_features": torch.stack(all_features),
            "candidate_valid": candidate_valid,
            "candidate_roi": candidate_valid.clone(),
        }
