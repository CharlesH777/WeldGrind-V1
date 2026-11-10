"""Function-preserving DiT depth expansion for an auditable capacity arm.

Every repaired parent block is copied to an even index; a trainable block with
zero residual gates follows it. The copied function is unchanged before an
optimizer update. Geometry, selector, observation scale and output heads are
transferred by exact name. Added blocks never receive fabricated Adam history.
"""
from __future__ import annotations

from copy import deepcopy
import re
from collections.abc import Mapping

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .model import sinusoidal_embedding
from .program_model import ProgramConfig, ProgramTrajectoryDiT
from .repaired_metric_program_model import RepairedMetricProgramFlowModel


PARENT_FORMAT = "weldgrind-complete-program-repaired-metric-v1"
PARENT_STRATEGY = "bounded_selector_autonomous_rollout_v1"
EXPANDED_FORMAT = "weldgrind-complete-program-expanded-metric-v1"
EXPANDED_STRATEGY = "expanded_identity_interleaved_dit_v1"
_BLOCK_KEY = re.compile(r"^dit\.blocks\.([0-9]+)\.(.+)$")


class ExpandedProgramTrajectoryDiT(ProgramTrajectoryDiT):
    """Identical DiT math with optional per-block activation recomputation.

    Recompute both teacher/t0 calls and autonomous calls, rather than only the
    outer rollout. No wrapper modules are introduced, preserving parameter keys.
    """

    def __init__(self, config, *, dit_block_checkpoint=True):
        if not isinstance(dit_block_checkpoint, bool):
            raise ValueError("dit_block_checkpoint must be boolean")
        super().__init__(config)
        self.dit_block_checkpoint = dit_block_checkpoint

    def forward_tokens(self, trajectory: Tensor, time: Tensor, seed_xyz: Tensor,
                       seed_features: Tensor) -> Tensor:
        if trajectory.ndim != 3 or trajectory.shape[1:] != (self.length, 3):
            raise ValueError(f"Full program must have shape [B,{self.length},3]")
        positions = torch.arange(self.length, dtype=trajectory.dtype, device=trajectory.device)
        x = self.input(trajectory) + sinusoidal_embedding(positions, self.dim)[None]
        seeds = self.seed_norm(seed_features + self.seed_position(seed_xyz))
        condition = self.time_mlp(sinusoidal_embedding(time.reshape(-1) * 1000, self.dim)) + seeds.mean(1)
        for block in self.blocks:
            if self.dit_block_checkpoint and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, seeds, condition, use_reentrant=False)
            else:
                x = block(x, seeds, condition)
        return self.final_norm(x)


class ExpandedMetricProgramFlowModel(RepairedMetricProgramFlowModel):
    """An exactly doubled-depth repaired model, with no change in width/inputs."""

    def __init__(self, config=None, *, source_dit_layers=None, dit_block_checkpoint=True,
                 **repair_kwargs):
        config = config or ProgramConfig(dit_layers=12)
        if not isinstance(config, ProgramConfig):
            raise TypeError("Expanded model requires ProgramConfig")
        if source_dit_layers is None:
            if config.dit_layers % 2:
                raise ValueError("Expanded DiT depth must be even")
            source_dit_layers = config.dit_layers // 2
        if (isinstance(source_dit_layers, bool) or not isinstance(source_dit_layers, int)
                or source_dit_layers < 1 or config.dit_layers != 2 * source_dit_layers):
            raise ValueError("Expanded depth must equal twice the positive source depth")
        super().__init__(config, **repair_kwargs)
        self.source_dit_layers = source_dit_layers
        self.dit_block_checkpoint = dit_block_checkpoint
        self.dit = ExpandedProgramTrajectoryDiT(config, dit_block_checkpoint=dit_block_checkpoint)
        # Identity residual gates, not zero attention/FF weights: gates receive
        # gradients on update one; the newly opened branches can learn next.
        with torch.no_grad():
            for block in self.dit.blocks[1::2]:
                linear = block.modulation[-1]
                for gate in (2, 5, 8):
                    rows = slice(gate * self.dit.dim, (gate + 1) * self.dit.dim)
                    linear.weight[rows].zero_()
                    linear.bias[rows].zero_()


def expanded_program_config(source):
    """Make a new config; no caller-owned configuration is mutated."""
    config = source if isinstance(source, ProgramConfig) else ProgramConfig.from_dict(source)
    values = deepcopy(config.to_dict())
    values["dit_layers"] *= 2
    return ProgramConfig.from_dict(values)


def _model_options(metadata):
    for name in ("metric_model_options", "repair_options"):
        if not isinstance(metadata.get(name), dict):
            raise ValueError(f"Checkpoint must explicitly record {name}")
    selectors = deepcopy(metadata.get("selector_options", {}))
    if not isinstance(selectors, dict):
        raise ValueError("selector_options must be a dict")
    class_name = selectors.pop("class", "StableLearnableSetSelector")
    if class_name != "StableLearnableSetSelector":
        raise ValueError("Expansion requires the repaired bounded selector")
    return {**deepcopy(metadata["metric_model_options"]),
            "repair_options": deepcopy(metadata["repair_options"]),
            "selector_options": selectors}


def expand_from_repaired_state(parent, *, device="cpu", initialization_seed=20261110,
                               dit_block_checkpoint=True):
    """Warm-expand a complete repaired state; return the model and exact audit.

    ``source_parameter_names`` records the source canonical order required to
    interpret historical unnamed Adam param_groups. ``parameter_name_map`` is
    the only supported name migration; new blocks belong to an empty-state
    third optimizer group owned by the runner. No optimizer is created here.
    """
    if not isinstance(parent, Mapping):
        raise ValueError("parent must be an already loaded complete checkpoint mapping")
    metadata = parent.get("metadata", {})
    if (parent.get("format") != PARENT_FORMAT
            or metadata.get("model_strategy") != PARENT_STRATEGY):
        raise ValueError("Warm expansion accepts only an explicitly tagged repaired parent")
    if (isinstance(initialization_seed, bool) or not isinstance(initialization_seed, int)
            or not 0 <= initialization_seed < 2 ** 63):
        raise ValueError("initialization_seed must be a nonnegative int below 2**63")
    if not isinstance(dit_block_checkpoint, bool):
        raise ValueError("dit_block_checkpoint must be boolean")
    source_config = ProgramConfig.from_dict(parent["model_config"])
    target_config = expanded_program_config(source_config)
    options = _model_options(metadata)
    # CPU initialization is reproducible, without changing CPU or CUDA training
    # RNG. manual_seed() is deliberately avoided: it also seeds CUDA generators.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(initialization_seed)
        with torch.device("meta"):
            template = RepairedMetricProgramFlowModel(source_config, **options)
        model = ExpandedMetricProgramFlowModel(target_config,
            source_dit_layers=source_config.dit_layers,
            dit_block_checkpoint=dit_block_checkpoint, **options)
    source_state = parent.get("model", {})
    template_state = template.state_dict()
    if set(source_state) != set(template_state):
        raise ValueError("Parent model keys do not exactly match its repaired source architecture")
    source_parameter_names = [name for name, _ in template.named_parameters()]
    state_name_map = {}
    expanded_state = model.state_dict()
    for name, value in source_state.items():
        if (not isinstance(value, Tensor) or value.shape != template_state[name].shape
                or value.dtype != template_state[name].dtype):
            raise ValueError(f"Parent tensor shape/dtype changed: {name}")
        match = _BLOCK_KEY.fullmatch(name)
        destination = (f"dit.blocks.{2 * int(match.group(1))}.{match.group(2)}" if match else name)
        if destination not in expanded_state or expanded_state[destination].shape != value.shape:
            raise ValueError(f"Unmappable parent tensor: {name}")
        expanded_state[destination] = value
        state_name_map[name] = destination
    if len(set(state_name_map.values())) != len(state_name_map):
        raise ValueError("Parent state mapping must be one-to-one")
    model.load_state_dict(expanded_state, strict=True)
    parameter_name_map = {name: state_name_map[name] for name in source_parameter_names}
    copied = set(parameter_name_map.values())
    new_parameter_names = [name for name, _ in model.named_parameters() if name not in copied]
    if any(not _BLOCK_KEY.fullmatch(name) or int(_BLOCK_KEY.fullmatch(name).group(1)) % 2 != 1
           for name in new_parameter_names):
        raise ValueError("Only newly inserted odd DiT blocks may add parameters")
    source_count = sum(parameter.numel() for parameter in template.parameters())
    target_count = sum(parameter.numel() for parameter in model.parameters())
    audit = {"source_dit_layers": source_config.dit_layers, "target_dit_layers": target_config.dit_layers,
        "dit_block_checkpoint": dit_block_checkpoint, "initialization_seed": initialization_seed,
        "source_config": source_config.to_dict(), "target_config": target_config.to_dict(),
        "source_parameter_names": source_parameter_names, "parameter_name_map": parameter_name_map,
        "state_name_map": state_name_map, "new_parameter_names": new_parameter_names,
        "source_parameters": source_count, "target_parameters": target_count,
        "added_parameters": target_count - source_count, "capacity_ratio": target_count / source_count,
        "source_state_tensors_transferred": len(state_name_map),
        "all_source_state_transferred": True, "initial_function_preserving": True,
        "identity_initialization": "new odd blocks have exactly zero residual-gate weights/biases",
        "optimizer_migration": "preserve original two groups by explicit name map; new odd-block parameters have no moments/step0",
        "new_checkpoint_format": EXPANDED_FORMAT, "new_model_strategy": EXPANDED_STRATEGY}
    return model.to(device), audit


def load_expanded_model_from_state(state, *, device="cpu"):
    """Strict expanded-format model factory; no legacy class interpretation."""
    metadata = state.get("metadata", {})
    expansion = metadata.get("expansion", {})
    if (state.get("format") != EXPANDED_FORMAT
            or metadata.get("model_strategy") != EXPANDED_STRATEGY):
        raise ValueError("Expected explicitly tagged expanded metric-program state")
    config = ProgramConfig.from_dict(state["model_config"])
    for name in ("source_dit_layers", "target_dit_layers", "dit_block_checkpoint"):
        if name not in expansion:
            raise ValueError(f"Expanded metadata must record {name}")
    if config.dit_layers != expansion["target_dit_layers"]:
        raise ValueError("Expanded metadata and model_config depths disagree")
    if ("target_config" in expansion
            and ProgramConfig.from_dict(expansion["target_config"]).to_dict() != config.to_dict()):
        raise ValueError("Recorded expanded target config disagrees with checkpoint")
    if ("source_config" in expansion
            and expanded_program_config(expansion["source_config"]).to_dict() != config.to_dict()):
        raise ValueError("Expanded configuration must change only the source DiT depth")
    model = ExpandedMetricProgramFlowModel(config,
        source_dit_layers=expansion["source_dit_layers"],
        dit_block_checkpoint=expansion["dit_block_checkpoint"], **_model_options(metadata))
    model.load_state_dict(state["model"], strict=True)
    return model.to(device)
