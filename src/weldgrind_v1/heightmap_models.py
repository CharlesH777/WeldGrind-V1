"""Independent height-map field and conditional trajectory-diffusion models.

The image U-Net follows the local B2 velocity-field family's residual/FiLM
topology. Its conditions, physical units, targets and output heads are new.
No language model, old checkpoint, GT phase or GT geometry is an input.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels):
    return math.gcd(8, channels)


def _finite(value, name):
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} must be finite')


def axial_tangent(vector, *, dim=-1, eps=1e-8):
    """Encode an unoriented 2-D tangent as cos(2 theta), sin(2 theta).

    Opposite cutting strokes at the same pixel have exactly the same target,
    rather than averaging their velocities to zero. This is an axis, not m/s.
    """
    if vector.shape[dim] != 2:
        raise ValueError('2-D tangent required')
    unit = F.normalize(vector, dim=dim, eps=eps)
    u, v = unit.unbind(dim)
    return torch.stack((u.square() - v.square(), 2 * u * v), dim=dim)


class _ImageBlock(nn.Module):
    def __init__(self, incoming, outgoing, condition_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(incoming), incoming)
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(outgoing), outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1)
        self.film = nn.Linear(condition_dim, 2 * outgoing)
        self.skip = nn.Conv2d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()

    def forward(self, image, condition):
        hidden = self.conv1(F.silu(self.norm1(image)))
        scale, shift = self.film(F.silu(condition)).chunk(2, -1)
        hidden = self.norm2(hidden) * (1 + .9 * scale.tanh()[:, :, None, None])
        hidden = hidden + 2 * shift.tanh()[:, :, None, None]
        return self.skip(image) + self.conv2(F.silu(hidden))


@dataclass(frozen=True)
class HeightMapFieldConfig:
    input_channels: int = 15
    process_dimensions: int = 4
    widths: tuple[int, ...] = (16, 32, 64, 96)
    condition_dimensions: int = 128

    def __post_init__(self):
        if isinstance(self.widths, list):
            object.__setattr__(self, 'widths', tuple(self.widths))
        if (self.input_channels < 1 or self.process_dimensions != 4 or len(self.widths) < 2
                or any(not isinstance(x, int) or x < 8 for x in self.widths)
                or self.condition_dimensions < 8):
            raise ValueError('Invalid heightmap field model configuration')


class HeightMapFieldUNet(nn.Module):
    """Single-pass pixel depth/axial-direction field with explicit process ratios."""
    def __init__(self, config=None):
        super().__init__()
        self.config = config if isinstance(config, HeightMapFieldConfig) else HeightMapFieldConfig(**(config or {}))
        cfg = self.config
        self.process = nn.Sequential(nn.Linear(4, cfg.condition_dimensions), nn.SiLU(),
                                     nn.Linear(cfg.condition_dimensions, cfg.condition_dimensions))
        self.stem = nn.Conv2d(cfg.input_channels, cfg.widths[0], 3, padding=1)
        self.down = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for index, width in enumerate(cfg.widths):
            if index:
                self.down.append(nn.Conv2d(cfg.widths[index - 1], width, 3, stride=2, padding=1))
            self.blocks.append(_ImageBlock(width, width, cfg.condition_dimensions))
        self.up = nn.ModuleList([_ImageBlock(cfg.widths[i] + cfg.widths[i - 1], cfg.widths[i - 1], cfg.condition_dimensions)
                                 for i in range(len(cfg.widths) - 1, 0, -1)])
        self.head = nn.Conv2d(cfg.widths[0], 4, 1)
        with torch.no_grad():
            self.head.bias[0] = -3.

    def forward(self, features, process_conditions):
        if features.ndim != 4 or features.shape[1] != self.config.input_channels or min(features.shape[-2:]) < 8:
            raise ValueError('features must be [B,C,H,W], spatial size >=8')
        if process_conditions.shape != (features.shape[0], 4):
            raise ValueError('process_conditions must be [B,4] metric ratios')
        _finite(features, 'features'); _finite(process_conditions, 'process_conditions')
        condition = self.process(process_conditions)
        hidden = self.stem(features)
        skips = []
        for index, block in enumerate(self.blocks):
            if index:
                hidden = self.down[index - 1](hidden)
            hidden = block(hidden, condition)
            skips.append(hidden)
        for block, skip in zip(self.up, reversed(skips[:-1])):
            hidden = F.interpolate(hidden, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            hidden = block(torch.cat((hidden, skip), 1), condition)
        raw = self.head(hidden)
        return {'cut_depth_normalized': F.softplus(raw[:, :1]),
                'tangent_axis': F.normalize(raw[:, 1:3], dim=1, eps=1e-6),
                'contact_logits': raw[:, 3:4], 'raw_field': raw}


def _image_target(value, prediction, channels=1):
    if value.ndim == 3 and channels == 1:
        value = value[:, None]
    if value.shape != prediction.shape:
        raise ValueError('Field target and prediction dimensions disagree')
    return value


def _masked_mean(value, mask):
    mask = mask.bool()
    selected = value[mask]
    if not selected.numel():
        return value.sum() * 0.
    _finite(selected, 'supervised values')
    return selected.mean()


def field_loss(output, *, cut_depth_normalized, cut_valid, tangent_axis,
               axis_valid, contact_target, axis_coherence=None):
    """GT enters only this training objective; invalid labels may be NaN."""
    depth = output['cut_depth_normalized']
    target = _image_target(cut_depth_normalized, depth)
    mask = _image_target(cut_valid, depth).bool()
    if (target[mask] < 0).any():
        raise ValueError('Cut depth is nonnegative; concavity must not request removal')
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    depth_loss = _masked_mean(F.smooth_l1_loss(depth, safe_target, reduction='none', beta=.05), mask)
    axis = output['tangent_axis']
    if tangent_axis.shape != axis.shape:
        raise ValueError('tangent_axis must be [B,2,H,W]')
    axis_mask = _image_target(axis_valid, depth).bool()
    safe_axis = torch.where(axis_mask.expand_as(axis), tangent_axis, torch.zeros_like(axis))
    if axis_mask.any():
        _finite(safe_axis[axis_mask.expand_as(axis)], 'axis labels')
    cosine = (axis * safe_axis).sum(1, keepdim=True).clamp(-1., 1.)
    axis_error = 1. - cosine
    if axis_coherence is not None:
        coherence = _image_target(axis_coherence, depth)
        axis_error = axis_error * torch.where(axis_mask, coherence, torch.zeros_like(coherence)).clamp(0., 1.)
    axis_loss = _masked_mean(axis_error, axis_mask)
    contact = _image_target(contact_target, depth).to(depth.dtype)
    safe_contact = torch.where(mask, contact, torch.zeros_like(contact))
    contact_loss = _masked_mean(F.binary_cross_entropy_with_logits(output['contact_logits'], safe_contact, reduction='none'), mask)
    total = depth_loss + .25 * axis_loss + .1 * contact_loss
    return {'loss': total, 'depth': depth_loss, 'axis': axis_loss, 'contact': contact_loss}


def _time_embedding(timestep, dimension):
    half = dimension // 2
    frequency = torch.exp(-math.log(10000.) * torch.arange(half, device=timestep.device) / max(half - 1, 1))
    angle = timestep.float()[:, None] * frequency[None]
    embedding = torch.cat((angle.sin(), angle.cos()), -1)
    return F.pad(embedding, (0, dimension - embedding.shape[-1]))


class _SequenceBlock(nn.Module):
    def __init__(self, incoming, outgoing, condition_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(incoming), incoming)
        self.conv1 = nn.Conv1d(incoming, outgoing, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(outgoing), outgoing)
        self.conv2 = nn.Conv1d(outgoing, outgoing, 3, padding=1)
        self.film = nn.Linear(condition_dim, 2 * outgoing)
        self.skip = nn.Conv1d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()

    def forward(self, sequence, condition):
        hidden = self.conv1(F.silu(self.norm1(sequence)))
        scale, shift = self.film(F.silu(condition)).chunk(2, -1)
        hidden = self.norm2(hidden) * (1 + .9 * scale.tanh()[:, :, None]) + 2 * shift.tanh()[:, :, None]
        return self.skip(sequence) + self.conv2(F.silu(hidden))


@dataclass(frozen=True)
class HeightMapDiffusionConfig:
    input_channels: int = 15
    image_width: int = 16
    sequence_width: int = 32
    condition_dimensions: int = 128
    timesteps: int = 1000
    beta_start: float = .0001
    beta_end: float = .02
    phase_weight: float = .1

    def __post_init__(self):
        if (self.input_channels < 1 or min(self.image_width, self.sequence_width, self.condition_dimensions) < 8
                or self.timesteps < 2 or not 0 < self.beta_start < self.beta_end < 1
                or not math.isfinite(self.phase_weight) or self.phase_weight < 0):
            raise ValueError('Invalid trajectory diffusion configuration')


class HeatMapTrajectoryDiffusion(nn.Module):
    """Image-conditioned 1-D U-Net DDPM, with self-predicted-clean phase state."""
    def __init__(self, config=None):
        super().__init__()
        self.config = config if isinstance(config, HeightMapDiffusionConfig) else HeightMapDiffusionConfig(**(config or {}))
        cfg = self.config
        widths = (cfg.image_width, cfg.image_width * 2, cfg.image_width * 4)
        self.process = nn.Sequential(nn.Linear(4, cfg.condition_dimensions), nn.SiLU(),
                                     nn.Linear(cfg.condition_dimensions, cfg.condition_dimensions))
        self.image_stem = nn.Conv2d(cfg.input_channels, widths[0], 3, padding=1)
        self.image_blocks = nn.ModuleList([_ImageBlock(w, w, cfg.condition_dimensions) for w in widths])
        self.image_down = nn.ModuleList([nn.Conv2d(widths[i], widths[i + 1], 3, stride=2, padding=1) for i in range(2)])
        self.image_projection = nn.Linear(widths[-1], cfg.condition_dimensions)
        self.time_projection = nn.Sequential(nn.Linear(cfg.condition_dimensions, cfg.condition_dimensions), nn.SiLU(),
                                             nn.Linear(cfg.condition_dimensions, cfg.condition_dimensions))
        width = cfg.sequence_width
        self.sequence_stem = nn.Conv1d(3, width, 3, padding=1)
        self.seq0 = _SequenceBlock(width, width, cfg.condition_dimensions)
        self.down1 = nn.Conv1d(width, width * 2, 3, stride=2, padding=1)
        self.seq1 = _SequenceBlock(width * 2, width * 2, cfg.condition_dimensions)
        self.down2 = nn.Conv1d(width * 2, width * 4, 3, stride=2, padding=1)
        self.seq2 = _SequenceBlock(width * 4, width * 4, cfg.condition_dimensions)
        self.query = nn.Linear(width * 4, cfg.condition_dimensions)
        self.key = nn.Linear(cfg.condition_dimensions, cfg.condition_dimensions)
        self.value = nn.Linear(cfg.condition_dimensions, cfg.condition_dimensions)
        self.attention_output = nn.Linear(cfg.condition_dimensions, width * 4)
        self.up1 = _SequenceBlock(width * 6, width * 2, cfg.condition_dimensions)
        self.up0 = _SequenceBlock(width * 3, width, cfg.condition_dimensions)
        self.noise_head = nn.Conv1d(width, 3, 1)
        self.phase_stem = nn.Conv1d(3, width, 3, padding=1)
        self.phase_block = _SequenceBlock(width, width, cfg.condition_dimensions)
        self.phase_head = nn.Conv1d(width, 4, 1)
        beta = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps, dtype=torch.float64)
        self.register_buffer('alpha_bar', torch.cumprod(1. - beta, dim=0).float())

    def encode(self, features, process_conditions):
        if features.ndim != 4 or features.shape[1] != self.config.input_channels or min(features.shape[-2:]) < 8:
            raise ValueError('features must be [B,C,H,W] with spatial size >=8')
        if process_conditions.shape != (len(features), 4):
            raise ValueError('process_conditions must be [B,4]')
        _finite(features, 'features'); _finite(process_conditions, 'process_conditions')
        process = self.process(process_conditions)
        hidden = self.image_stem(features)
        for index, block in enumerate(self.image_blocks):
            if index:
                hidden = self.image_down[index - 1](hidden)
            hidden = block(hidden, process)
        tokens = F.adaptive_avg_pool2d(hidden, (4, 4)).flatten(2).transpose(1, 2)
        tokens = self.image_projection(tokens)
        return process + tokens.mean(1), tokens

    @staticmethod
    def _extract(values, timestep, reference):
        return values[timestep].reshape((len(reference),) + (1,) * (reference.ndim - 1))

    def q_sample(self, clean, timestep, noise):
        if clean.ndim != 3 or clean.shape[-1] != 3 or clean.shape != noise.shape:
            raise ValueError('clean and noise must have matching [B,L,3] shape')
        if timestep.shape != (len(clean),) or timestep.dtype != torch.long or (timestep < 0).any() or (timestep >= self.config.timesteps).any():
            raise ValueError('timestep must be long[B] inside diffusion schedule')
        a = self._extract(self.alpha_bar, timestep, clean)
        return a.sqrt() * clean + (1 - a).sqrt() * noise

    def denoise(self, noisy, timestep, condition, tokens):
        if noisy.ndim != 3 or noisy.shape[-1] != 3 or noisy.shape[1] < 4:
            raise ValueError('noisy trajectory must be [B,L,3], L>=4')
        _finite(noisy, 'noisy trajectory')
        embedding = condition + self.time_projection(_time_embedding(timestep / self.config.timesteps, self.config.condition_dimensions))
        h0 = self.seq0(self.sequence_stem(noisy.transpose(1, 2)), embedding)
        h1 = self.seq1(self.down1(h0), embedding)
        h2 = self.seq2(self.down2(h1), embedding)
        query = F.normalize(self.query(h2.transpose(1, 2)), dim=-1, eps=1e-6)
        key = F.normalize(self.key(tokens), dim=-1, eps=1e-6)
        attention = (8. * query @ key.transpose(1, 2)).softmax(-1)
        h2 = h2 + self.attention_output(attention @ self.value(tokens)).transpose(1, 2)
        h = self.up1(torch.cat((F.interpolate(h2, size=h1.shape[-1], mode='linear', align_corners=False), h1), 1), embedding)
        h = self.up0(torch.cat((F.interpolate(h, size=h0.shape[-1], mode='linear', align_corners=False), h0), 1), embedding)
        return self.noise_head(h).transpose(1, 2)

    def phase_logits(self, clean_estimate, condition):
        hidden = self.phase_block(self.phase_stem(clean_estimate.transpose(1, 2)), condition)
        return self.phase_head(hidden).transpose(1, 2)

    def loss(self, features, process_conditions, target_xyz, target_phases, valid=None, *, generator=None,
             timestep=None, noise=None):
        if target_xyz.ndim != 3 or target_xyz.shape[-1] != 3 or target_phases.shape != target_xyz.shape[:2]:
            raise ValueError('Training labels require XYZ[B,L,3] and phases[B,L]')
        _finite(target_xyz, 'target trajectory')
        if target_phases.dtype != torch.long or (target_phases < 0).any() or (target_phases > 3).any():
            raise ValueError('target_phases must be long[B,L] in [0,3]')
        mask = torch.ones_like(target_phases, dtype=torch.bool) if valid is None else valid.bool()
        if mask.shape != target_phases.shape:
            raise ValueError('valid must align with trajectory tokens')
        if timestep is None:
            timestep = torch.randint(self.config.timesteps, (len(features),), device=features.device, generator=generator)
        if noise is None:
            noise = torch.randn(target_xyz.shape, device=target_xyz.device, dtype=target_xyz.dtype, generator=generator)
        noisy = self.q_sample(target_xyz, timestep, noise)
        condition, tokens = self.encode(features, process_conditions)
        predicted_noise = self.denoise(noisy, timestep, condition, tokens)
        noise_loss = _masked_mean((predicted_noise - noise).square(), mask[:, :, None].expand_as(noise))
        a = self._extract(self.alpha_bar, timestep, noisy)
        clean_estimate = (noisy - (1 - a).sqrt() * predicted_noise) / a.sqrt()
        # Same state protocol during train and decode: phase sees the model's
        # own clean estimate, never the teacher trajectory or teacher phase.
        logits = self.phase_logits(clean_estimate.detach(), condition)
        phase_error = F.cross_entropy(logits.transpose(1, 2), target_phases, reduction='none')
        phase_loss = _masked_mean(phase_error * a[:, 0, 0][:, None], mask)
        total = noise_loss + self.config.phase_weight * phase_loss
        return {'loss': total, 'noise': noise_loss, 'phase': phase_loss,
                'predicted_x0': clean_estimate, 'phase_logits': logits, 'timestep': timestep}

    @torch.no_grad()
    def sample(self, features, process_conditions, *, points=1024, steps=32, seed=0):
        if isinstance(points, bool) or not isinstance(points, int) or points < 4:
            raise ValueError('points must be an integer >=4')
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= self.config.timesteps:
            raise ValueError('steps must be inside diffusion schedule')
        mode = self.training
        self.eval()
        try:
            generator = torch.Generator(device=features.device).manual_seed(int(seed))
            value = torch.randn((len(features), points, 3), device=features.device, dtype=features.dtype, generator=generator)
            condition, tokens = self.encode(features, process_conditions)
            schedule = torch.linspace(self.config.timesteps - 1, 0, steps, device=features.device).round().long()
            for index, step in enumerate(schedule):
                timestep = step.expand(len(features))
                predicted_noise = self.denoise(value, timestep, condition, tokens)
                a = self._extract(self.alpha_bar, timestep, value)
                clean = (value - (1 - a).sqrt() * predicted_noise) / a.sqrt()
                previous = self.alpha_bar[schedule[index + 1]] if index + 1 < len(schedule) else value.new_tensor(1.)
                value = previous.sqrt() * clean + (1 - previous).sqrt() * predicted_noise
                _finite(value, 'generated trajectory')
            logits = self.phase_logits(value, condition)
            return {'trajectory_normalized': value, 'phase_logits': logits,
                    'predicted_phases': logits.argmax(-1), 'end_token_index': points - 1,
                    'method': 'conditional_DDIM_eta0', 'seed': int(seed), 'steps': steps,
                    'phase_state': 'self_predicted_x0', 'ground_truth_input': False}
        finally:
            self.train(mode)


def model_from_config(variant, config):
    if variant == 'field':
        return HeightMapFieldUNet(config)
    if variant == 'diffusion':
        return HeatMapTrajectoryDiffusion(config)
    raise ValueError('variant must be field or diffusion')


def model_config(model):
    return asdict(model.config)
