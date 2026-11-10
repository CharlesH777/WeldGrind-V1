"""SAM 3 masks and calibrated projection onto an XYZRGB point cloud.

Projection uses optical camera coordinates (X right, Y down, Z forward).
XYZ, the optional depth image and ``occlusion_tolerance`` share the same unit.
Pixel coordinates are (u, v), rounded to the nearest pixel centre.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def project_mask_to_points(
    mask: np.ndarray,
    xyz: np.ndarray,
    intrinsics: np.ndarray | None = None,
    camera_from_world: np.ndarray | None = None,
    point_uv: np.ndarray | None = None,
    *,
    depth_image: np.ndarray | None = None,
    occlusion_tolerance: float = 1.0,
    min_depth: float = 1e-6,
) -> np.ndarray:
    """Return visible points covered by ``mask`` without guessing calibration.

    ``xyz`` is in the world frame when ``camera_from_world`` is supplied;
    otherwise it must already be in the optical camera frame. Explicit
    ``point_uv`` can replace intrinsics for a cloud with known RGB registration.
    A nearest-depth z buffer rejects occluded cloud points; an optional measured
    registered depth image additionally checks absolute depth agreement.
    """
    mask = np.asarray(mask)
    xyz = np.asarray(xyz, dtype=np.float64)
    if mask.ndim != 2 or not mask.size:
        raise ValueError("mask must have nonempty shape [H,W]")
    if not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError("mask must contain finite binary values")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N,3]")
    if not np.isfinite(occlusion_tolerance) or occlusion_tolerance < 0:
        raise ValueError("occlusion_tolerance must be finite and nonnegative")
    if not np.isfinite(min_depth) or min_depth <= 0:
        raise ValueError("min_depth must be finite and positive")
    camera_xyz = xyz
    if camera_from_world is not None:
        transform = np.asarray(camera_from_world, dtype=np.float64)
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1])):
            raise ValueError("camera_from_world must be a finite homogeneous [4,4] matrix")
        rotation = transform[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
            raise ValueError("camera_from_world rotation must be a proper rigid rotation")
        camera_xyz = xyz @ rotation.T + transform[:3, 3]
    z = camera_xyz[:, 2]
    valid = np.isfinite(camera_xyz).all(axis=1) & (z > min_depth)
    if point_uv is not None:
        uv = np.asarray(point_uv, dtype=np.float64)
        if uv.shape != (len(xyz), 2):
            raise ValueError("point_uv must have shape [N,2] in (u,v) order")
    else:
        if intrinsics is None:
            raise ValueError("Provide calibrated intrinsics or explicit registered point_uv")
        k = np.asarray(intrinsics, dtype=np.float64)
        if (k.shape != (3, 3) or not np.isfinite(k).all()
                or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1])):
            raise ValueError("intrinsics must be a finite optical-camera [3,3] matrix")
        projected = camera_xyz @ k.T
        uv = projected[:, :2] / np.where(valid, z, 1)[:, None]
    height, width = mask.shape
    valid &= np.isfinite(uv).all(axis=1)
    # Bound floating pixels before integer conversion, including extreme but
    # finite coordinates, to avoid integer overflow on malformed/outside points.
    valid &= (uv[:, 0] >= -0.5) & (uv[:, 0] < width - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] < height - 0.5)
    # Invalid floats never reach integer casts or array indexing.
    pixel = np.floor(np.where(valid[:, None], uv, 0) + 0.5).astype(np.int64)
    valid &= (pixel[:, 0] >= 0) & (pixel[:, 0] < width) & (pixel[:, 1] >= 0) & (pixel[:, 1] < height)
    selected = np.zeros(len(xyz), dtype=bool)
    ids = np.flatnonzero(valid)
    if not len(ids):
        return selected
    flat_pixel = pixel[ids, 1] * width + pixel[ids, 0]
    z_buffer = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(z_buffer, flat_pixel, z[ids])
    visible = z[ids] <= z_buffer[flat_pixel] + occlusion_tolerance
    if depth_image is not None:
        depth = np.asarray(depth_image, dtype=np.float64)
        if depth.shape != mask.shape:
            raise ValueError("depth_image must have the same [H,W] shape as mask")
        measured = depth[pixel[ids, 1], pixel[ids, 0]]
        visible &= np.isfinite(measured) & (measured > min_depth) & (np.abs(measured - z[ids]) <= occlusion_tolerance)
    selected[ids] = visible & mask.astype(bool)[pixel[ids, 1], pixel[ids, 0]]
    return selected


class Sam3Segmenter:
    """Lazy local-checkpoint SAM 3 adapter; all retained instances are unioned.

    ``predict`` accepts a PIL image, image path or uint8 RGB HWC array. No model
    is downloaded and no substitute segmentation is generated on import errors.
    Inspect ``last_metadata`` for instance count and the union policy.
    """

    def __init__(self, checkpoint: str, device: str = "cpu", confidence: float = 0.5, *, bpe_path: str | None = None):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Local SAM 3 checkpoint does not exist: {self.checkpoint}")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must lie in [0,1]")
        self.device, self.confidence, self.bpe_path = device, confidence, bpe_path
        self._processor = None
        self.last_metadata: dict = {}

    def _load(self):
        if self._processor is None:
            try:
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
            except ImportError as exc:
                raise ImportError("Install the local 01_projects/sam3 package and its dependencies for SAM 3 inference") from exc
            model = build_sam3_image_model(
                checkpoint_path=str(self.checkpoint), load_from_HF=False,
                device=self.device, bpe_path=self.bpe_path, eval_mode=True,
            )
            self._processor = Sam3Processor(model, device=self.device, confidence_threshold=self.confidence)
        return self._processor

    def predict(self, image, text: str = "weld seam") -> np.ndarray:
        from PIL import Image
        if not isinstance(text, str) or not text.strip():
            raise ValueError("SAM 3 text prompt must be nonempty")
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                image = opened.convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
                raise ValueError("image array must be uint8 RGB [H,W,3]")
            image = Image.fromarray(array, mode="RGB")
        processor = self._load()
        state = processor.set_image(image)
        result = processor.set_text_prompt(prompt=text, state=state)
        masks = result["masks"]
        if hasattr(masks, "detach"):
            masks = masks.detach().cpu().numpy()
        masks = np.asarray(masks, dtype=bool)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim != 3 or masks.shape[1:] != (image.height, image.width):
            raise ValueError("Unexpected SAM 3 output shape; expected [instances,1,H,W] or [instances,H,W]")
        self.last_metadata = {
            "backend": "sam3", "checkpoint": str(self.checkpoint), "prompt": text,
            "confidence": self.confidence, "instance_policy": "union_all_retained_instances",
            "instance_count": len(masks), "camera_frame": "optical_x_right_y_down_z_forward",
        }
        return np.any(masks, axis=0)
