"""Fresh, bounded time-conditioned program models; historical runs stay intact.

All arms share an untrained common initialization. No historical learned weight,
optimizer moment, target template or target-derived inference condition is used.
"""
from dataclasses import dataclass, asdict
from copy import deepcopy
import math
import re

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .model import ConditionalDiTBlock, sinusoidal_embedding
from .program_model import ProgramConfig, ProgramTrajectoryDiT, segment_smoothness
from .repaired_metric_program_model import RepairedMetricProgramFlowModel, RolloutRepairConfig
from .metric_program_model import DEFAULT_METRIC_LOSS_WEIGHTS
from .expanded_selector_program_model import SelectorExpandedStableSelector

FORMAT = "weldgrind-bounded-scratch-program-v1"
STRATEGY = "bounded_time_qknorm_uniform_flow_shared_integrator_v1"
ARMS = {"base": (2, 6), "dit12": (2, 12), "selector6": (6, 6)}


@dataclass(frozen=True)
class Bounds:
    attention_scale: float = 8.0
    shift: float = 2.0
    scale_delta: float = .9
    gate: float = 1.0
    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("All numerical bounds must be finite and positive")
        if self.scale_delta >= 1:
            raise ValueError("Modulation must retain a positive scale")


class NormalizedAttention(nn.MultiheadAttention):
    """Headwise Q/K normalization bounds scores independently of learned norms."""
    def __init__(self, *args, logit_scale=8., **kw):
        super().__init__(*args, **kw)
        self.logit_scale = float(logit_scale)
        if not self.batch_first or not self._qkv_same_embed_dim or self.dropout:
            raise ValueError("Program attention requires batch-first equal widths and no dropout")

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if need_weights or attn_mask is not None or is_causal:
            raise ValueError("Only unmasked program attention with optional padding is supported")
        b, n, dim = query.shape
        head = dim // self.num_heads
        weights = self.in_proj_weight.chunk(3)
        biases = self.in_proj_bias.chunk(3) if self.in_proj_bias is not None else [None]*3
        q, k, v = [F.linear(x, w, a).reshape(b, -1, self.num_heads, head).transpose(1, 2)
                   for x, w, a in zip((query, key, value), weights, biases)]
        # FP32 normalization retains small-gradient information under BF16 AMP.
        q = F.normalize(q.float(), dim=-1, eps=1e-6).to(v.dtype)
        k = F.normalize(k.float(), dim=-1, eps=1e-6).to(v.dtype)
        mask = None
        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool or key_padding_mask.shape != (b, key.shape[1]):
                raise ValueError("Padding mask must be boolean [B,Nkey]")
            if key_padding_mask.all(-1).any():
                raise ValueError("At least one key must be valid")
            mask = ~key_padding_mask[:, None, None, :]
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                               dropout_p=0., scale=self.logit_scale)
        return self.out_proj(output.transpose(1, 2).reshape(b, n, dim)), None


def normalize_attention_modules(module, scale):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            replacement = NormalizedAttention(child.embed_dim, child.num_heads,
                batch_first=True, dropout=0., bias=child.in_proj_bias is not None,
                logit_scale=scale)
            replacement.load_state_dict(child.state_dict(), strict=True)
            setattr(module, name, replacement)
        else:
            normalize_attention_modules(child, scale)


class BoundedDiTBlock(ConditionalDiTBlock):
    def __init__(self, dim, heads, ratio, bounds):
        super().__init__(dim, heads, ratio)
        self.bounds = bounds
        normalize_attention_modules(self, bounds.attention_scale)

    def modulation_parts(self, condition):
        raw = self.modulation(condition).chunk(9, -1)
        limits = (self.bounds.shift, self.bounds.scale_delta, self.bounds.gate)*3
        return tuple(limit * torch.tanh(x/limit) for x, limit in zip(raw, limits))

    def forward(self, x, seeds, condition):
        s1, a1, g1, s2, a2, g2, s3, a3, g3 = self.modulation_parts(condition)
        q = self.modulate(self.norms[0](x), s1, a1)
        x = x + g1[:, None] * self.self_attention(q, q, q, need_weights=False)[0]
        q = self.modulate(self.norms[1](x), s2, a2)
        x = x + g2[:, None] * self.cross_attention(q, seeds, seeds, need_weights=False)[0]
        return x + g3[:, None] * self.ff(self.modulate(self.norms[2](x), s3, a3))


class BoundedProgramDiT(ProgramTrajectoryDiT):
    def __init__(self, cfg, bounds, activation_checkpoint=True):
        super().__init__(cfg)
        self.blocks = nn.ModuleList([BoundedDiTBlock(self.dim, cfg.heads, cfg.mlp_ratio, bounds)
                                     for _ in range(cfg.dit_layers)])
        self.condition_norm = nn.LayerNorm(self.dim, elementwise_affine=False)
        self.activation_checkpoint = activation_checkpoint

    def forward_tokens(self, trajectory, time, seed_xyz, seed_features):
        if trajectory.shape[1:] != (self.length, 3):
            raise ValueError("Wrong complete program shape")
        positions = torch.arange(self.length, device=trajectory.device, dtype=torch.float32)
        x = self.input(trajectory) + sinusoidal_embedding(positions, self.dim)[None]
        seeds = self.seed_norm(seed_features + self.seed_position(seed_xyz))
        condition = self.condition_norm(
            self.time_mlp(sinusoidal_embedding(time.reshape(-1).float()*1000, self.dim)) + seeds.mean(1))
        for block in self.blocks:
            if self.activation_checkpoint and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, seeds, condition, use_reentrant=False)
            else:
                x = block(x, seeds, condition)
        return self.final_norm(x)


class BoundedScratchProgramModel(RepairedMetricProgramFlowModel):
    def __init__(self, config=None, *, bounds=None, activation_checkpoint=True,
                 selector_expanded=False):
        config = config or ProgramConfig()
        self.bounds = bounds if isinstance(bounds, Bounds) else Bounds(**(bounds or {}))
        super().__init__(config, repair_options=RolloutRepairConfig(
            rollout_steps=8, rollout_reference_mm=2., rollout_activation_checkpoint=False))
        if selector_expanded:
            if config.selector_layers != 6:
                raise ValueError("Expanded selector requires six blocks")
            self.selector = SelectorExpandedStableSelector(config, source_selector_layers=2,
                selector_block_checkpoint=activation_checkpoint)
        normalize_attention_modules(self.selector, self.bounds.attention_scale)
        self.dit = BoundedProgramDiT(config, self.bounds, activation_checkpoint)
        self.activation_checkpoint = activation_checkpoint
        self.selector_expanded = selector_expanded

    @staticmethod
    def uniform_times(batch, device, dtype, generator=None):
        # Each minibatch covers the entire flow interval, with no t0-biased mixture.
        t = (torch.arange(batch, device=device, dtype=dtype) +
             torch.rand(batch, device=device, dtype=dtype, generator=generator))/batch
        return t[torch.randperm(batch, device=device, generator=generator)]

    def _integrate(self, condition, initial_noise, steps=8, method="euler"):
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1 or method not in ("euler", "heun"):
            raise ValueError("Invalid finite integration")
        x = initial_noise.clone()
        for index in range(steps):
            t = x.new_full((x.shape[0],), index/steps)
            v = self.dit(x, t, condition['seed_xyz'], condition['seed_features'])['velocity']
            proposed = x + v/steps
            if method == 'heun':
                v2 = self.dit(proposed, t+1/steps, condition['seed_xyz'], condition['seed_features'])['velocity']
                proposed = x + (v+v2)/(2*steps)
            x = proposed
        final = self.dit(x, x.new_ones(x.shape[0]), condition['seed_xyz'], condition['seed_features'])
        return {**condition, 'trajectory':x, 'phase_logits':final['phase_logits'],
                'phases':final['phase_logits'].argmax(-1)}

    def differentiable_rollout(self, points, roi_mask, point_valid=None, *, scale_mm,
                               steps=None, method=None, initial_noise=None, generator=None):
        condition = self.encode(points, roi_mask, point_valid, scale_mm=scale_mm)
        shape = (points.shape[0], self.config.trajectory_points, 3)
        noise = (torch.randn(shape, device=points.device, dtype=points.dtype, generator=generator)*self.config.noise_std
                 if initial_noise is None else initial_noise.to(points).clone())
        if noise.shape != shape or not torch.isfinite(noise).all():
            raise ValueError("Invalid initial noise")
        return self._integrate(condition, noise, 8 if steps is None else steps,
                               'euler' if method is None else method)

    @torch.no_grad()
    def sample(self, points, roi_mask, point_valid=None, steps=8, method='euler',
               generator=None, initial_noise=None, *, scale_mm):
        if self.training:
            raise RuntimeError("Call eval before sampling")
        return self.differentiable_rollout(points, roi_mask, point_valid, scale_mm=scale_mm,
            steps=steps, method=method, initial_noise=initial_noise, generator=generator)

    def flow_loss(self, points, roi_mask, trajectory, point_valid=None, *, scale_mm,
                  phase_labels, segment_ids=None, generator=None, weights=None):
        settings = {**DEFAULT_METRIC_LOSS_WEIGHTS, 'rollout_position':1.,
                    'rollout_phase':.5, 'rollout_smoothness':.001}
        if weights:
            if set(weights)-set(settings): raise ValueError("Unknown loss weights")
            settings.update(weights)
        if trajectory.shape != (points.shape[0], self.config.trajectory_points, 3):
            raise ValueError("Invalid target shape")
        if phase_labels.shape != trajectory.shape[:2] or phase_labels.dtype != torch.long:
            raise ValueError("Phase labels must be int64 [B,T]")
        condition = self.encode(points, roi_mask, point_valid, scale_mm=scale_mm)
        noise = torch.randn(trajectory.shape, device=points.device, dtype=points.dtype,
                            generator=generator)*self.config.noise_std
        t = self.uniform_times(points.shape[0], points.device, points.dtype, generator)
        noisy = noise.lerp(trajectory, t[:,None,None])
        teacher = self.dit(noisy, t, condition['seed_xyz'], condition['seed_features'])
        pure = self.dit(noise, torch.zeros_like(t), condition['seed_xyz'], condition['seed_features'])
        v = trajectory-noise
        estimate = noisy+(1-t[:,None,None])*teacher['velocity']
        direct = noise+pure['velocity']
        factor = condition['observed_scale_mm'][:,None,None]/20.
        labels = phase_labels.reshape(-1)
        ce = lambda logits: F.cross_entropy(logits.reshape(-1,4),labels)
        valid = condition['candidate_valid']
        distances = torch.cdist(condition['candidate_xyz'],condition['seed_xyz']).square().amin(-1)
        soft = condition['soft_assignments']
        overlap = soft@soft.transpose(1,2)
        off = ~torch.eye(self.config.seeds,device=points.device,dtype=torch.bool)
        rollout_noise = torch.randn(trajectory.shape, device=points.device, dtype=points.dtype,
                                    generator=generator)*self.config.noise_std
        generated = self._integrate(condition, rollout_noise)
        residual = (generated['trajectory']-trajectory)*condition['observed_scale_mm'][:,None,None]/2.
        losses = {'flow':F.mse_loss(teacher['velocity'],v), 'endpoint':F.mse_loss(estimate,trajectory),
            'phase':ce(teacher['phase_logits']), 'conditional_t0':F.mse_loss(direct,trajectory),
            'conditional_phase':ce(pure['phase_logits']),
            'physical_flow':((teacher['velocity']-v)*factor).square().mean(),
            'physical_t0':((direct-trajectory)*factor).square().mean(),
            'coverage':(distances*valid).sum()/valid.sum().clamp_min(1),
            'diversity':overlap[:,off].mean() if self.config.seeds>1 else overlap.sum()*0,
            'smoothness':segment_smoothness(estimate,segment_ids),
            'rollout_position':residual.square().mean(),
            'rollout_phase':self._rollout_phase_loss(generated['phase_logits'],phase_labels),
            'rollout_smoothness':segment_smoothness(residual,segment_ids)}
        losses['loss'] = sum(settings[k]*v for k,v in losses.items())
        return losses


def initialized_arm(arm, *, seed=20261110, config=None, activation_checkpoint=True):
    """Random base initialization followed by identity depth insertion, never training."""
    if arm not in ARMS: raise ValueError("Unknown arm")
    base_cfg = config or ProgramConfig()
    if (base_cfg.selector_layers,base_cfg.dit_layers)!=(2,6):
        raise ValueError("Common initialization requires selector2/DiT6")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        base = BoundedScratchProgramModel(base_cfg, activation_checkpoint=activation_checkpoint)
        if arm=='base': return base
        cfg_dict=deepcopy(base_cfg.to_dict())
        cfg_dict['selector_layers'],cfg_dict['dit_layers']=ARMS[arm]
        torch.manual_seed(seed+1)
        model=BoundedScratchProgramModel(ProgramConfig.from_dict(cfg_dict),
            activation_checkpoint=activation_checkpoint,selector_expanded=arm=='selector6')
        target=model.state_dict()
        for name,value in base.state_dict().items():
            match=re.match(r'^(dit|selector)\.blocks\.(\d+)\.(.+)$',name)
            mapped=name
            if match:
                prefix,index,suffix=match.groups()
                factor=2 if arm=='dit12' and prefix=='dit' else 3 if arm=='selector6' and prefix=='selector' else 1
                mapped=f'{prefix}.blocks.{int(index)*factor}.{suffix}'
            target[mapped]=value.clone()
        model.load_state_dict(target,strict=True)
        if arm=='dit12':
            with torch.no_grad():
                for block in model.dit.blocks[1::2]:
                    for gate in (2,5,8):
                        rows=slice(gate*model.dit.dim,(gate+1)*model.dit.dim)
                        block.modulation[-1].weight[rows].zero_()
                        block.modulation[-1].bias[rows].zero_()
        return model
