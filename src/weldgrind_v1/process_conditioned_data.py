"""Known machining conditions, separately bound to immutable source metadata.

The four scalars describe the tool and requested process. They are read only
from tool configuration metadata, never inferred from a trajectory or phase.
The original NPZ observations, normalization and supervision stay unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from .program_data import ProgramDataset


PROCESS_PARAMETER_NAMES = ("tool_effective_radius_mm", "layer_depth_mm", "clearance_mm", "overlap")
PROCESS_PARAMETER_UNITS = ("mm", "mm", "mm", "fraction")
PROCESS_CONDITION_SCHEMA = "weldgrind-known-process-conditions-v1"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Process parameter {field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (result < 0 if field == "overlap" else result <= 0):
        raise ValueError(f"Process parameter {field} must be finite and positive")
    if field == "overlap" and result >= 1:
        raise ValueError("Process overlap must lie in [0, 1)")
    return result


def _parameters(asset, trajectory_meta):
    """Read explicit process declarations; no access to program labels is needed."""
    if not isinstance(asset, dict) or not isinstance(trajectory_meta, dict):
        raise ValueError("Process metadata must be JSON mappings")
    scene = asset.get("scene", {})
    configured = asset.get("config", {}).get("tool", {})
    declared = trajectory_meta.get("tool", {})
    if not isinstance(scene, dict) or not isinstance(configured, dict) or not isinstance(declared, dict):
        raise ValueError("Process sources must contain tool configuration mappings")
    radius = declared.get("radius_mm", scene.get("tool_effective_radius_mm"))
    values = [radius, declared.get("maximum_layer_depth_mm", configured.get("maximum_layer_depth_mm")),
              declared.get("clearance_mm", configured.get("clearance_mm")),
              declared.get("lane_overlap", configured.get("lane_overlap"))]
    result = [_positive(value, name) for name, value in zip(PROCESS_PARAMETER_NAMES, values)]
    explicit_fields = {
        "tool_effective_radius_mm": "trajectory_meta.tool.radius_mm" if "radius_mm" in declared else "asset_metadata.scene.tool_effective_radius_mm",
        "layer_depth_mm": "trajectory_meta.tool.maximum_layer_depth_mm" if "maximum_layer_depth_mm" in declared else "asset_metadata.config.tool.maximum_layer_depth_mm",
        "clearance_mm": "trajectory_meta.tool.clearance_mm" if "clearance_mm" in declared else "asset_metadata.config.tool.clearance_mm",
        "overlap": "trajectory_meta.tool.lane_overlap" if "lane_overlap" in declared else "asset_metadata.config.tool.lane_overlap",
    }
    comparisons = [(scene.get("tool_effective_radius_mm"), result[0], "scene tool radius"),
                   (declared.get("effective_contact_width_mm"), result[0] * 2, "declared tool width")]
    for key, value in zip(("maximum_layer_depth_mm", "clearance_mm", "lane_overlap"), result[1:]):
        if key in configured:
            comparisons.append((configured[key], value, f"configured {key}"))
    for value, expected, name in comparisons:
        if value is not None and not math.isclose(_positive(value, "overlap" if "overlap" in name else name), expected,
                                                  rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"Process source declarations disagree: {name}")
    return result, explicit_fields


def _source_root(row, manifest, overrides):
    cohort = row.get("cohort")
    value = overrides.get(cohort, row.get("source_root"))
    if not isinstance(value, (str, Path)):
        raise ValueError("Every sample needs an explicit source_root or source-root override")
    path = Path(value)
    if not path.is_absolute():
        # Portable dataset bundles place program_inputs/ and raw/ beside each other.
        base = manifest.parent.parent if manifest.parent.name in ("program_inputs", "runtime_inputs") else manifest.parent
        path = base / path
    return path.resolve()


def _read_source_conditions(row, manifest, *, source_root_overrides=None, expected_sources=None):
    overrides = source_root_overrides or {}
    npz = Path(row["npz"])
    if not npz.is_absolute():
        npz = manifest.parent / npz
    sidecar = json.loads(npz.with_suffix(".json").read_text())
    if sidecar.get("sample_id") != row["sample_id"]:
        raise ValueError("Process sidecar sample ID differs from the data manifest")
    fingerprints = sidecar.get("source_fingerprints", {})
    root = _source_root(row, manifest, overrides)
    documents, sources = {}, {}
    for kind in ("asset_metadata", "trajectory_meta"):
        record = fingerprints.get(kind)
        if not isinstance(record, dict) or set(("path", "sha256", "bytes")) - set(record):
            raise ValueError(f"Missing fingerprinted {kind} process source")
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Process {kind} path must be relative to the source collection")
        source = (root / relative).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError(f"Missing or escaping process source: {kind}")
        canonical = {"path": relative.as_posix(), "sha256": record["sha256"], "bytes": record["bytes"]}
        if expected_sources is not None and canonical != expected_sources.get(kind):
            raise ValueError(f"Process {kind} fingerprint differs from the frozen process manifest")
        payload = source.read_bytes()
        if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"Process {kind} source SHA256/size mismatch")
        documents[kind] = json.loads(payload)
        sources[kind] = canonical
    parameters, fields = _parameters(documents["asset_metadata"], documents["trajectory_meta"])
    return parameters, fields, sources


def _row_identity(row):
    return {key: row.get(key) for key in ("sample_id", "cohort", "optimizer_split")}


def _identity_sha(rows):
    payload = json.dumps([_row_identity(row) for row in rows], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def export_process_manifest(manifests, output, *, source_root_overrides=None):
    """Freeze declared conditions and their source fingerprints without touching NPZs.

    ``manifests`` maps split names to JSONL files. Paths are provenance only;
    loading binds row identities/order and source SHA, so data roots may move.
    The output is created exclusively and cannot silently overwrite a snapshot.
    """
    records, bindings, seen = [], {}, set()
    for split, path in manifests.items():
        manifest = Path(path).resolve()
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError("Cannot freeze an empty process split")
        bindings[split] = {"original_manifest": str(manifest), "original_manifest_sha256": _sha256(manifest),
                           "count": len(rows), "ordered_row_identity_sha256": _identity_sha(rows)}
        for row in rows:
            sid = row["sample_id"]
            if sid in seen or row.get("optimizer_split") != split:
                raise ValueError("Process sample IDs must be unique and optimizer splits must agree")
            seen.add(sid)
            values, fields, sources = _read_source_conditions(row, manifest, source_root_overrides=source_root_overrides)
            records.append({**_row_identity(row), "process_params": values, "declared_fields": fields,
                            "sources": sources, "original_source_root": str(_source_root(row, manifest, source_root_overrides or {}))})
    result = {"schema": PROCESS_CONDITION_SCHEMA, "parameter_names": list(PROCESS_PARAMETER_NAMES),
              "parameter_units": list(PROCESS_PARAMETER_UNITS), "count": len(records),
              "manifests": bindings, "samples": records,
              "source_policy": "explicit tool configuration only; no values inferred from trajectories, phases, passes or chart axes",
              "trajectory_or_npz_reexported": False}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


class ProcessConditionedProgramDataset(ProgramDataset):
    """Original sample tensors plus four known, fingerprint-verified conditions."""
    def __init__(self, manifest, max_points=2048, trajectory_points=1024, seed=20261109,
                 max_error_mm=.02, *, process_manifest=None, source_root_overrides=None):
        super().__init__(manifest, max_points=max_points, trajectory_points=trajectory_points,
                         seed=seed, max_error_mm=max_error_mm)
        self.source_root_overrides = source_root_overrides or {}
        self.process_manifest = Path(process_manifest).resolve() if process_manifest is not None else None
        self.process_manifest_sha256 = _sha256(self.process_manifest) if self.process_manifest else None
        self._process_records = None
        if self.process_manifest:
            frozen = json.loads(self.process_manifest.read_text())
            if (frozen.get("schema") != PROCESS_CONDITION_SCHEMA
                    or frozen.get("parameter_names") != list(PROCESS_PARAMETER_NAMES)
                    or frozen.get("parameter_units") != list(PROCESS_PARAMETER_UNITS)):
                raise ValueError("Unsupported process manifest schema or parameter order")
            records = frozen.get("samples")
            if not isinstance(records, list) or frozen.get("count") != len(records):
                raise ValueError("Invalid process manifest sample count")
            indexed = {}
            for record in records:
                sid = record.get("sample_id")
                if not isinstance(sid, str) or sid in indexed:
                    raise ValueError("Duplicate or invalid process manifest sample ID")
                values = record.get("process_params")
                if not isinstance(values, list) or len(values) != 4:
                    raise ValueError("Frozen process parameters must contain exactly four values")
                [_positive(value, name) for name, value in zip(PROCESS_PARAMETER_NAMES, values)]
                indexed[sid] = record
            split_names = {row.get("optimizer_split") for row in self.rows}
            if len(split_names) != 1:
                raise ValueError("One data manifest must contain a single optimizer split")
            split = next(iter(split_names))
            binding = frozen.get("manifests", {}).get(split, {})
            if (binding.get("count") != len(self.rows)
                    or binding.get("ordered_row_identity_sha256") != _identity_sha(self.rows)):
                raise ValueError("Dataset identities/order do not match the frozen process manifest")
            for row in self.rows:
                record = indexed.get(row["sample_id"])
                if record is None or _row_identity(record) != _row_identity(row):
                    raise ValueError("Dataset row identity differs from its process record")
            self._process_records = indexed

    def __getitem__(self, index):
        row = self.rows[index]
        record = self._process_records[row["sample_id"]] if self._process_records is not None else None
        values, _, _ = _read_source_conditions(row, self.manifest,
            source_root_overrides=self.source_root_overrides,
            expected_sources=record["sources"] if record is not None else None)
        if record is not None and any(not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                                      for a, b in zip(values, record["process_params"])):
            raise ValueError("Frozen process parameters disagree with their source metadata")
        sample = super().__getitem__(index)
        sample["process_params"] = torch.tensor(values, dtype=torch.float32)
        return sample


def main():
    parser = argparse.ArgumentParser(description="Freeze known tool/process parameters; no training or GPU execution")
    parser.add_argument("--plan", type=Path, required=True, help="Plan JSON with a manifests split mapping")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    result = export_process_manifest(plan["manifests"], args.output)
    print(json.dumps({"count": result["count"], "output": str(args.output.resolve()),
                      "sha256": _sha256(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
