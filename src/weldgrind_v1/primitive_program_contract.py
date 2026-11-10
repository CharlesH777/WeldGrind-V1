"""Strict analytic checks for box/cylinder/sphere contact-reference programs.

The primitive chart stores true curved geometry. Its basis columns describe a
centre tangent frame and must not be used as an affine chart on curved parts.
"""
from pathlib import Path

import numpy as np


KINDS = {"box": "planar_box_top", "cylinder": "cylindrical_axial_side", "sphere": "spherical_front_patch"}


def analytic_coordinates(scene, world_xyz):
    xyz = np.asarray(world_xyz, dtype=np.float64)
    pose = scene["world_from_local"]
    rotation = np.asarray(pose["rotation"], dtype=np.float64)
    translation = np.asarray(pose["translation_mm"], dtype=np.float64)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all()
            or rotation.shape != (3, 3) or translation.shape != (3,)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-7)):
        raise ValueError("Primitive XYZ and source SE3 must be finite world-mm geometry")
    shape = scene["shape"]
    if KINDS.get(shape) != scene.get("chart", {}).get("kind"):
        raise ValueError("Primitive shape/chart kind mismatch")
    local = (xyz - translation) @ rotation
    if shape == "box":
        uvh = local
        normals = np.tile([0., 0., 1.], (len(local), 1))
    else:
        radius = float(scene["dimensions_mm"]["radius"])
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Curved chart radius must be positive")
        x, y, z = local.T
        if shape == "cylinder":
            theta = np.arctan2(y, z)
            uvh = np.column_stack([x, radius * theta, np.hypot(y, z) - radius])
            normals = np.column_stack([np.zeros(len(local)), np.sin(theta), np.cos(theta)])
        else:
            u = np.arctan2(x, z)
            v = np.arctan2(y, np.hypot(x, z))
            uvh = np.column_stack([radius * u, radius * v, np.linalg.norm(local, axis=1) - radius])
            normals = np.column_stack([np.cos(v) * np.sin(u), np.sin(v), np.cos(v) * np.cos(u)])
    return uvh, normals @ rotation.T


def validate_primitive_program(scene, metadata, trajectory, *, normal_path=None, normals_world=None,
                               tolerance_mm=.01):
    """Check explicit air modes against actual offset-surface geometry.

    The 0.01mm tolerance covers saved float32 chord and transform error; it is
    much smaller than the source contact spacing and clearance. No path changes.
    """
    if not 0 < tolerance_mm <= .02:
        raise ValueError("Primitive contract tolerance must remain <=0.02mm")
    xyz = np.asarray(trajectory, dtype=np.float64)
    uvh, analytical_normals = analytic_coordinates(scene, xyz)
    if normals_world is None:
        if not normal_path or not Path(normal_path).is_file():
            raise ValueError("Primitive programs require their saved per-point normal file")
        normals_world = np.load(normal_path, allow_pickle=False)
    normals = np.asarray(normals_world, dtype=np.float64)
    if (normals.shape != xyz.shape or not np.isfinite(normals).all()
            or np.max(np.abs(np.linalg.norm(normals, axis=1) - 1), initial=0) > 1e-4
            or np.max(np.linalg.norm(normals - analytical_normals, axis=1), initial=0) > .001):
        raise ValueError("Saved per-point outward normals disagree with primitive geometry")
    segments = metadata["segment_ranges"]
    modes = []
    expected = "contact"
    for segment in segments:
        mode = segment.get("motion_mode")
        if mode not in ("contact", "retract", "transfer", "approach"):
            raise ValueError("Primitive segment requires explicit motion_mode")
        if bool(segment["contact"]) != (mode == "contact"):
            raise ValueError("Primitive contact/motion_mode disagree")
        if mode != expected:
            raise ValueError("Primitive transition must retain contact/retract/transfer/approach order")
        expected = {"contact": "retract", "retract": "transfer", "transfer": "approach", "approach": "contact"}[mode]
        modes.append(mode)
    if not modes or modes[0] != "contact" or modes[-1] != "contact":
        raise ValueError("Primitive source must start/end with actual contact strokes")
    for i, segment in enumerate(segments):
        curve = uvh[segment["start"]:segment["end"]]
        mode = modes[i]
        if len(curve) < 2:
            raise ValueError("Primitive segments require nonzero source curves")
        if mode == "contact":
            if np.min(curve[:, 2]) < float(segment["cut_level_chart_h_mm"]) - tolerance_mm:
                raise ValueError("Contact reference penetrates its own scheduled cut level")
            continue
        clearance = float(segment.get("clearance_chart_h_mm", np.nan))
        if not np.isfinite(clearance):
            raise ValueError("Primitive air motion requires explicit offset-shell clearance")
        if mode in ("retract", "approach"):
            if np.max(np.linalg.norm(curve[:, :2] - curve[0, :2], axis=1)) > tolerance_mm:
                raise ValueError("Primitive retract/approach must follow its own radial normal")
            delta = np.diff(curve[:, 2])
            if mode == "retract" and (curve[-1, 2] <= curve[0, 2] or np.min(delta) < -tolerance_mm):
                raise ValueError("Primitive retract must move outward")
            if mode == "approach" and (curve[-1, 2] >= curve[0, 2] or np.max(delta) > tolerance_mm):
                raise ValueError("Primitive approach must move inward")
            end = -1 if mode == "retract" else 0
            if abs(curve[end, 2] - clearance) > tolerance_mm:
                raise ValueError("Primitive lift endpoint does not reach declared offset shell")
        elif np.max(np.abs(curve[:, 2] - clearance)) > tolerance_mm:
            raise ValueError("Primitive transfer must retain actual curved offset-shell clearance")
        triplet_start = i - (0 if mode == "retract" else 1 if mode == "transfer" else 2)
        if any(abs(float(s.get("clearance_chart_h_mm", np.nan)) - clearance) > 1e-7
               for s in segments[triplet_start:triplet_start + 3]):
            raise ValueError("Air triplet clearance annotations disagree")
    return {"passed": True, "chart_kind": scene["chart"]["kind"], "point_count": len(xyz),
            "normal_source": metadata.get("normal_file"), "tolerance_mm": tolerance_mm,
            "normals_max_error": float(np.max(np.linalg.norm(normals - analytical_normals, axis=1), initial=0)),
            "geometry_changed": False}
