"""A fixed voxel lattice in the existing observed-ROI normalized frame.

The full cloud's minimum is only an integer storage shift, aligned to the
complete stride-8 hierarchy. It never changes voxel membership, continuous
position features, or the physical centres of cells near the ROI. This module
uses observations only; it does not estimate a target or a nominal surface.
"""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor, nn

from .geometry import SparseVoxelBackbone, _mean_voxel_features


GEOMETRY_FRAME_POLICY = {
    "name": "roi_normalized_signed_voxel_lattice_v1",
    "coordinate_frame": "existing observed SAM ROI bbox-centred, max-radius-normalized XYZ",
    "lattice_origin_normalized": [0.0, 0.0, 0.0],
    "signed_cell": "floor(xyz / voxel_size + 1e-6)",
    "storage_shift": "floor_divide(signed_cell.amin(0), 8) * 8",
    "hierarchy_alignment": 8,
    "continuous_vfe_position": "original ROI-normalized xyz; no full-cloud-min subtraction",
    "stage_center": "(coordinates + floor_divide(storage_shift, 2**stage) + 0.5) * voxel_size * 2**stage; integer addition before float conversion",
    "candidate_sampling": "unchanged actual observed ROI points",
    "target_geometry_used": False,
}


class RoiLatticeSparseVoxelBackbone(SparseVoxelBackbone):
    """The original learned sparse hierarchy with an ROI-fixed coordinate frame.

    Inputs and outputs, submodules, state keys and parameter shapes are exactly
    those of ``SparseVoxelBackbone``. Input ``xyz`` is already normalized by
    ``prepare_point_cloud``; callers must keep that observed-ROI normalization.
    """

    def forward(self, xyz: Tensor, rgb: Tensor, roi: Tensor) -> list[tuple[Tensor, Tensor]]:
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
            raise ValueError("ROI lattice requires nonempty [N,3] observed normalized XYZ")
        if not xyz.is_floating_point() or not torch.isfinite(xyz).all():
            raise ValueError("ROI lattice XYZ must be finite floating point")
        if rgb.shape != xyz.shape or roi.shape != (len(xyz),):
            raise ValueError("ROI lattice requires aligned XYZ/RGB/ROI rows")

        signed = torch.floor(xyz / self.config.voxel_size + 1e-6).long()
        aligned_min = torch.div(signed.amin(0), 8, rounding_mode="floor") * 8
        coordinates = signed - aligned_min
        values = torch.cat((xyz, rgb, roi.to(xyz.dtype).unsqueeze(-1)), -1)
        coordinates, features = _mean_voxel_features(coordinates, values)
        stages = []
        for stage, (conv, refine, norm, refine_norm) in enumerate(zip(
            self.convolutions, self.refinements, self.norms, self.refine_norms,
        )):
            coordinates, features = conv(coordinates, features)
            features = self.activation(norm(features))
            _, residual = refine(coordinates, features)
            features = self.activation(refine_norm(residual) + features)
            # Restore signed cell indices while still int64. Converting a
            # large nonnegative storage index before cancellation would lose
            # subvoxel precision near the ROI when a far point shifts storage.
            signed_coarse = coordinates + torch.div(aligned_min, 2 ** stage, rounding_mode="floor")
            centers = (signed_coarse.to(xyz.dtype) + 0.5) * (self.config.voxel_size * 2 ** stage)
            stages.append((centers, features))
        return stages


def install_roi_lattice_geometry(model) -> dict:
    """Replace the backbone container, preserving every existing Parameter.

    Child layers are reused and their state is loaded strictly, so their values,
    parameter order/identity, gradients, optimizer references, dtype/device,
    train/eval modes and optional accelerated forwards survive. Construction
    performs no random initialization and consumes no RNG. Install when creating
    a new explicit model/factory, never by editing a running training process.
    """
    geometry = getattr(model, "geometry", None)
    original = getattr(geometry, "backbone", None)
    if type(original) not in (SparseVoxelBackbone, RoiLatticeSparseVoxelBackbone):
        raise TypeError("Expected model.geometry.backbone to be the original or ROI lattice SparseVoxelBackbone")
    if (len(original.convolutions) != 4
            or [layer.stride for layer in original.convolutions] != [1, 2, 2, 2]):
        raise ValueError("ROI lattice alignment requires the original stride-1/2/4/8 hierarchy")

    before = tuple(model.named_parameters())
    installed = type(original) is not RoiLatticeSparseVoxelBackbone
    if installed:
        # Calling the normal constructor would initialize new random weights,
        # replace Parameter objects and invalidate an existing optimizer. Only
        # the parameter-free module container needs replacement here.
        replacement = RoiLatticeSparseVoxelBackbone.__new__(RoiLatticeSparseVoxelBackbone)
        nn.Module.__init__(replacement)
        replacement.config = original.config
        for name, child in original.named_children():
            replacement.add_module(name, child)
        replacement.training = original.training
        replacement.load_state_dict(original.state_dict(), strict=True)
        geometry.backbone = replacement

    after = tuple(model.named_parameters())
    if len(before) != len(after) or any(a != b or p is not q for (a, p), (b, q) in zip(before, after)):
        raise RuntimeError("ROI lattice installation changed parameter identity or order")
    return {
        "installed": installed,
        "backbone_class": f"{RoiLatticeSparseVoxelBackbone.__module__}:{RoiLatticeSparseVoxelBackbone.__name__}",
        "geometry_frame_policy": deepcopy(GEOMETRY_FRAME_POLICY),
        "parameter_names": [name for name, _ in after],
        "parameter_count": sum(parameter.numel() for _, parameter in after),
        "parameter_objects_preserved": True,
        "state_keys_shapes_dtypes_values_preserved": True,
        "rng_consumed": False,
    }
