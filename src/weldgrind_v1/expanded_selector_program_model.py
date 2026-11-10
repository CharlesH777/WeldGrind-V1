"""Increase only point-cloud selector capacity with an identity warm start.

Each original SetAttentionBlock is followed by two new zero-gated residual
blocks. The DiT, geometry encoder, feature width, seeds and program length stay
unchanged. Distinct metadata prevents interpreting this arm as DiT expansion.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import re

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .expanded_metric_program_model import ExpandedProgramTrajectoryDiT
from .model import SetAttentionBlock
from .program_model import ProgramConfig
from .repaired_metric_program_model import RepairedMetricProgramFlowModel
from .stable_selector import StableLearnableSetSelector


PARENT_FORMAT = "weldgrind-complete-program-repaired-metric-v1"
PARENT_STRATEGY = "bounded_selector_autonomous_rollout_v1"
SELECTOR_EXPANDED_FORMAT = "weldgrind-complete-program-selector-expanded-metric-v1"
SELECTOR_EXPANDED_STRATEGY = "selector_identity_interleaved_depth_v1"
_BLOCK_KEY = re.compile(r"^selector\.blocks\.([0-9]+)\.(.+)$")


class IdentityGatedSetAttentionBlock(SetAttentionBlock):
    """Trainable cross/self/FF branches, initially an exact identity function."""

    def __init__(self, dim, heads, ratio, *, activation_checkpoint=True):
        if not isinstance(activation_checkpoint, bool):
            raise ValueError("activation_checkpoint must be boolean")
        super().__init__(dim, heads, ratio)
        self.residual_gates = nn.Parameter(torch.zeros(3))
        self.activation_checkpoint = activation_checkpoint

    def _forward(self, queries, memory, valid):
        q = self.norms[0](queries)
        m = self.norms[1](memory)
        queries = queries + self.residual_gates[0] * self.cross(
            q, m, m, key_padding_mask=~valid, need_weights=False)[0]
        q = self.norms[2](queries)
        queries = queries + self.residual_gates[1] * self.self_attention(q, q, q, need_weights=False)[0]
        return queries + self.residual_gates[2] * self.ff(self.norms[3](queries))

    def forward(self, queries, memory, valid):
        # StableLearnableSetSelector already disables autocast here. Preserve
        # this FP32 computation through non-reentrant activation recomputation.
        if self.activation_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward, queries, memory, valid, use_reentrant=False)
        return self._forward(queries, memory, valid)


class SelectorExpandedStableSelector(StableLearnableSetSelector):
    """The same bounded/Sinkhorn selector, with three times as many blocks."""

    def __init__(self, config, *, source_selector_layers=None,
                 selector_block_checkpoint=True, **selector_options):
        if source_selector_layers is None:
            if config.selector_layers % 3:
                raise ValueError("Expanded selector depth must be divisible by three")
            source_selector_layers = config.selector_layers // 3
        if (isinstance(source_selector_layers, bool) or not isinstance(source_selector_layers, int)
                or source_selector_layers < 1 or config.selector_layers != 3 * source_selector_layers):
            raise ValueError("Selector depth must equal three times its positive source depth")
        if not isinstance(selector_block_checkpoint, bool):
            raise ValueError("selector_block_checkpoint must be boolean")
        super().__init__(config, **selector_options)
        self.source_selector_layers = source_selector_layers
        self.selector_block_checkpoint = selector_block_checkpoint
        self.blocks = nn.ModuleList([
            SetAttentionBlock(config.geometry.dim, config.heads, config.mlp_ratio)
            if index % 3 == 0 else IdentityGatedSetAttentionBlock(
                config.geometry.dim, config.heads, config.mlp_ratio,
                activation_checkpoint=selector_block_checkpoint)
            for index in range(config.selector_layers)])


class SelectorExpandedMetricProgramFlowModel(RepairedMetricProgramFlowModel):
    """Selector-only expansion; the DiT has exactly the supplied source depth."""

    def __init__(self, config=None, *, source_selector_layers=None,
                 selector_block_checkpoint=True, dit_block_checkpoint=True,
                 selector_options=None, **repair_kwargs):
        config = config or ProgramConfig(selector_layers=6)
        if not isinstance(config, ProgramConfig):
            raise TypeError("Selector-expanded model requires ProgramConfig")
        if not isinstance(dit_block_checkpoint, bool):
            raise ValueError("dit_block_checkpoint must be boolean")
        super().__init__(config, selector_options=selector_options, **repair_kwargs)
        self.selector = SelectorExpandedStableSelector(config,
            source_selector_layers=source_selector_layers,
            selector_block_checkpoint=selector_block_checkpoint, **(selector_options or {}))
        self.source_selector_layers = self.selector.source_selector_layers
        self.selector_block_checkpoint = selector_block_checkpoint
        self.dit_block_checkpoint = dit_block_checkpoint
        # This subclass provides activation recomputation only. It neither
        # changes DiT depth nor adds any parameter or identity DiT layer.
        self.dit = ExpandedProgramTrajectoryDiT(config, dit_block_checkpoint=dit_block_checkpoint)


def selector_expanded_program_config(source):
    config = source if isinstance(source, ProgramConfig) else ProgramConfig.from_dict(source)
    values = deepcopy(config.to_dict())
    values["selector_layers"] *= 3
    return ProgramConfig.from_dict(values)


def _model_options(metadata, *, expanded=False):
    for key in ("metric_model_options", "repair_options"):
        if not isinstance(metadata.get(key), dict):
            raise ValueError(f"Checkpoint must explicitly record {key}")
    options = deepcopy(metadata.get("selector_options", {}))
    if not isinstance(options, dict):
        raise ValueError("selector_options must be a dict")
    selector_class = options.pop("class", "StableLearnableSetSelector")
    accepted = {"StableLearnableSetSelector"}
    if expanded:
        accepted.add("SelectorExpandedStableSelector")
    if selector_class not in accepted:
        raise ValueError("Expected the bounded repaired selector's numerical settings")
    return {**deepcopy(metadata["metric_model_options"]),
            "repair_options": deepcopy(metadata["repair_options"]), "selector_options": options}


def expand_selector_from_repaired_state(parent, *, device="cpu", initialization_seed=20261110,
                                        selector_block_checkpoint=True, dit_block_checkpoint=True):
    """Copy a repaired checkpoint, expanding selector blocks i -> 3*i only.

    The audit's canonical source order and explicit name map are required for
    migration of the historical unnamed two Adam groups. The runner owns the
    empty-state third Adam group for all newly inserted selector parameters.
    """
    if not isinstance(parent, Mapping):
        raise ValueError("parent must be a complete loaded repaired checkpoint mapping")
    metadata = parent.get("metadata", {})
    if (parent.get("format") != PARENT_FORMAT
            or metadata.get("model_strategy") != PARENT_STRATEGY):
        raise ValueError("Selector expansion accepts only a tagged repaired source checkpoint")
    if (isinstance(initialization_seed, bool) or not isinstance(initialization_seed, int)
            or not 0 <= initialization_seed < 2 ** 63):
        raise ValueError("initialization_seed must be a nonnegative integer below 2**63")
    for flag in (selector_block_checkpoint, dit_block_checkpoint):
        if not isinstance(flag, bool):
            raise ValueError("Activation checkpoint flags must be boolean")
    source_config = ProgramConfig.from_dict(parent["model_config"])
    target_config = selector_expanded_program_config(source_config)
    options = _model_options(metadata)
    # Seed only the CPU default generator, and restore its state after setup;
    # do not alter the continuing arm's CPU/CUDA training noise sequence.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(initialization_seed)
        with torch.device("meta"):
            template = RepairedMetricProgramFlowModel(source_config, **options)
        model = SelectorExpandedMetricProgramFlowModel(target_config,
            source_selector_layers=source_config.selector_layers,
            selector_block_checkpoint=selector_block_checkpoint,
            dit_block_checkpoint=dit_block_checkpoint, **options)
    source_state = parent.get("model", {})
    reference = template.state_dict()
    if set(source_state) != set(reference):
        raise ValueError("Parent keys must exactly match the repaired source architecture")
    target_state = model.state_dict()
    state_name_map = {}
    for name, value in source_state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != reference[name].shape
                or value.dtype != reference[name].dtype):
            raise ValueError(f"Parent tensor shape or dtype changed: {name}")
        match = _BLOCK_KEY.fullmatch(name)
        destination = f"selector.blocks.{3 * int(match.group(1))}.{match.group(2)}" if match else name
        if destination not in target_state or target_state[destination].shape != value.shape:
            raise ValueError(f"Cannot migrate source state tensor {name}")
        target_state[destination] = value
        state_name_map[name] = destination
    if len(set(state_name_map.values())) != len(state_name_map):
        raise ValueError("State migration must be one-to-one")
    model.load_state_dict(target_state, strict=True)
    source_names = [name for name, _ in template.named_parameters()]
    parameter_name_map = {name: state_name_map[name] for name in source_names}
    copied = set(parameter_name_map.values())
    new_names = [name for name, _ in model.named_parameters() if name not in copied]
    if any(not _BLOCK_KEY.fullmatch(name) or int(_BLOCK_KEY.fullmatch(name).group(1)) % 3 == 0
           for name in new_names):
        raise ValueError("Added capacity must belong only to inserted selector blocks")
    source_count = sum(parameter.numel() for parameter in template.parameters())
    target_count = sum(parameter.numel() for parameter in model.parameters())
    audit = {"source_selector_layers": source_config.selector_layers,
        "target_selector_layers": target_config.selector_layers,
        "selector_block_checkpoint": selector_block_checkpoint, "dit_block_checkpoint": dit_block_checkpoint,
        "dit_layers": source_config.dit_layers, "initialization_seed": initialization_seed,
        "source_config": source_config.to_dict(), "target_config": target_config.to_dict(),
        "source_parameter_names": source_names, "parameter_name_map": parameter_name_map,
        "state_name_map": state_name_map, "new_parameter_names": new_names,
        "source_parameters": source_count, "target_parameters": target_count,
        "source_selector_parameters": sum(parameter.numel() for parameter in template.selector.parameters()),
        "target_selector_parameters": sum(parameter.numel() for parameter in model.selector.parameters()),
        "added_parameters": target_count - source_count, "capacity_ratio": target_count / source_count,
        "all_source_state_transferred": True, "source_state_tensors_transferred": len(state_name_map),
        "only_selector_capacity_increased": True, "geometry_and_dit_unchanged": True,
        "initial_function_preserving": True,
        "identity_initialization": "each old selector block is followed by two new blocks with three zero scalar residual gates",
        "optimizer_migration": "old two Adam groups migrate by name; new selector blocks use an empty-state third group",
        "new_checkpoint_format": SELECTOR_EXPANDED_FORMAT,
        "new_model_strategy": SELECTOR_EXPANDED_STRATEGY}
    return model.to(device), audit


def load_selector_expanded_model_from_state(state, *, device="cpu"):
    """Strict selector-expanded factory; do not read this arm as larger DiT."""
    metadata = state.get("metadata", {})
    expansion = metadata.get("selector_expansion", {})
    if (state.get("format") != SELECTOR_EXPANDED_FORMAT
            or metadata.get("model_strategy") != SELECTOR_EXPANDED_STRATEGY):
        raise ValueError("Expected explicitly tagged selector-expanded metric-program state")
    config = ProgramConfig.from_dict(state["model_config"])
    for key in ("source_selector_layers", "target_selector_layers", "dit_layers",
                "selector_block_checkpoint", "dit_block_checkpoint"):
        if key not in expansion:
            raise ValueError(f"Selector expansion metadata must record {key}")
    if config.selector_layers != expansion["target_selector_layers"] or config.dit_layers != expansion["dit_layers"]:
        raise ValueError("Checkpoint selector/DiT depths disagree with recorded selector-only expansion")
    if ("target_config" in expansion
            and ProgramConfig.from_dict(expansion["target_config"]).to_dict() != config.to_dict()):
        raise ValueError("Target config audit disagrees with the selector-expanded checkpoint")
    if ("source_config" in expansion
            and selector_expanded_program_config(expansion["source_config"]).to_dict() != config.to_dict()):
        raise ValueError("Only selector depth may change from the source architecture")
    model = SelectorExpandedMetricProgramFlowModel(config,
        source_selector_layers=expansion["source_selector_layers"],
        selector_block_checkpoint=expansion["selector_block_checkpoint"],
        dit_block_checkpoint=expansion["dit_block_checkpoint"], **_model_options(metadata, expanded=True))
    model.load_state_dict(state["model"], strict=True)
    return model.to(device)
