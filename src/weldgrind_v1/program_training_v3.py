"""Reviewed metric/conditioning correction within the parent's absolute budget.

This is an explicit strategy revision after saved diagnostics, not a failed-job
retry. It preserves the parent weights/Adam/RNG, adds observed scale, and spends
only the optimizer updates remaining before the original absolute final step.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from .geometry_acceleration import accelerate_geometry
from .program_data import ProgramDataset
from .program_model import ProgramConfig
from .metric_program_model import MetricProgramFlowModel
from .runtime import save_checkpoint, write_json


class PreparedProgramSamples(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        return self.samples[index]


class FiniteEpochBatches:
    """Deterministic full-coverage epochs, including the final smaller batch."""
    def __init__(self, count, batch_size, steps, seed=20261108, start_step=0):
        if min(count, batch_size, steps) < 1 or not 0 <= start_step <= steps:
            raise ValueError('Invalid finite batch schedule')
        self.count, self.batch_size, self.steps = count, batch_size, steps
        self.seed, self.start_step = seed, start_step
        self.batches_per_epoch = math.ceil(count / batch_size)
    def __len__(self):
        return self.steps - self.start_step
    def __iter__(self):
        epoch_cached, order = -1, None
        for step in range(self.start_step, self.steps):
            epoch, offset = divmod(step, self.batches_per_epoch)
            if epoch != epoch_cached:
                order = torch.randperm(self.count, generator=torch.Generator().manual_seed(self.seed + epoch)).tolist()
                epoch_cached = epoch
            yield order[offset*self.batch_size:(offset+1)*self.batch_size]


def enable_dit_bf16(model, device):
    """AMP only the transformer. Geometry/coordinates/voxel accumulation stay FP32."""
    if not str(device).startswith('cuda'):
        return False
    original = model.dit.forward
    def forward(*args, **kwargs):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = original(*args, **kwargs)
        return {key: value.float() for key, value in result.items()}
    model.dit.forward = forward
    return True


def train_metric_program(config_path, manifest, output_dir, *, sam_gate, parent_checkpoint,
                         stop_file=None, device='cuda'):
    config = yaml.safe_load(Path(config_path).read_text())
    settings = config['training']
    model_config = ProgramConfig.from_dict(config['model'])
    parent_path=Path(parent_checkpoint).resolve()
    parent=torch.load(parent_path,map_location='cpu',weights_only=True,mmap=True)
    if parent.get('format')!='weldgrind-complete-program-v1':
        raise ValueError('This reviewed correction warm-starts the saved original full-program checkpoint only')
    batch_size, workers = int(settings['batch_size']), int(settings.get('num_workers', 4))
    steps, eval_every = int(settings['steps']), int(settings.get('eval_every', 500))
    save_every, seed = int(settings.get('save_every', 500)), int(settings.get('seed', 20261108))
    if min(batch_size, steps, eval_every, save_every) < 1 or workers < 0:
        raise ValueError('Finite budget and batches must be positive')
    if steps>parent['target_steps']:
        raise ValueError('Correction cannot increase the original absolute optimizer budget')
    if parent['model_config']!=model_config.to_dict():
        raise ValueError('Correction preserves the original full-program architecture and lengths')
    output = Path(output_dir).resolve()
    if (output/'latest.json').exists():
        raise ValueError('This program run already has a checkpoint; no automatic restart')
    gate_path = Path(sam_gate).resolve()
    gate = json.loads(gate_path.read_text())
    if gate.get('passed') is not True:
        raise ValueError('Actual SAM inference must pass before trajectory fitting')
    manifest = Path(manifest).resolve()
    data = ProgramDataset(manifest, max_points=int(config['data']['max_points']),
                          trajectory_points=model_config.trajectory_points, seed=seed)
    expected = int(settings.get('expected_images', 300))
    if len(data) != expected:
        raise ValueError(f'Every image must be used: expected {expected}, got {len(data)}')
    output.mkdir(parents=True, exist_ok=True)
    stop = Path(stop_file) if stop_file else output/'REQUEST_STOP'
    start = time.monotonic()
    def progress(status, **extra):
        record = {'status': status, 'stage':'metric_conditioned_full_program_remaining_budget', 'step': completed,
                  'target_steps': steps, 'batch_size': batch_size, 'num_workers': workers,
                  'images': len(data), 'seen_unique_images': len(seen), 'image_exposures': exposures,
                  'elapsed_seconds': time.monotonic()-start, **extra}
        write_json(output/'progress.json', record)
        print(json.dumps(record,ensure_ascii=False),flush=True)
        return record
    completed=int(parent['step'])
    resumed_from=completed
    exposures=int(parent['image_exposures'])
    seen=set(parent['seen_sample_ids'])
    if completed>=steps:
        raise ValueError('No remaining optimizer budget; automatic extension is forbidden')
    progress('preparing_cpu_program_cache')
    with ThreadPoolExecutor(max_workers=max(1,workers),thread_name_prefix='program-prepare') as executor:
        samples = list(executor.map(data.__getitem__, range(len(data))))
    if len({sample['sample_id'] for sample in samples}) != len(samples):
        raise ValueError('Program sample identifiers must be unique')
    torch.manual_seed(seed)
    model_options=config.get('metric_model',{})
    model = MetricProgramFlowModel(model_config,**model_options).to(device)
    incompatible=model.load_state_dict(parent['model'],strict=False)
    if incompatible.unexpected_keys or any(not name.startswith('scale_encoder.') for name in incompatible.missing_keys):
        raise ValueError('Warm start may miss only the newly observed-scale encoder')
    acceleration = accelerate_geometry(model)
    amp = enable_dit_bf16(model,device) if config.get('performance',{}).get('dit_bf16',True) else False
    old_parameters=[p for name,p in model.named_parameters() if not name.startswith('scale_encoder.')]
    scale_parameters=[p for name,p in model.named_parameters() if name.startswith('scale_encoder.')]
    optimizer = torch.optim.AdamW(old_parameters,lr=float(settings.get('lr',.0003)),
                                  weight_decay=float(settings.get('weight_decay',.01)))
    optimizer.load_state_dict(parent['optimizer'])
    optimizer.add_param_group({'params':scale_parameters,'lr':float(settings.get('lr',.0003))})
    torch.set_rng_state(parent['torch_rng'].cpu())
    if str(device).startswith('cuda') and parent.get('cuda_rng') is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in parent['cuda_rng']])
    schedule = FiniteEpochBatches(len(samples),batch_size,steps,seed,start_step=completed)
    options = {'batch_sampler': schedule, 'num_workers': workers,
               'pin_memory': str(device).startswith('cuda')}
    if workers:
        options.update(persistent_workers=True,prefetch_factor=2)
    loader = DataLoader(PreparedProgramSamples(samples),**options)
    metadata = {'config':config,'manifest':str(manifest),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
                'sam_gate':str(gate_path),'sample_ids':[s['sample_id'] for s in samples],
                'source_split_counts':{k:sum(s['source_split']==k for s in samples) for k in ['train','val','test']},
                'geometry_runtime':'equivalent vectorized27-offset sparse convolution; cached input FPS indices only',
                'geometry_fp32':True,'dit_bf16':amp,'full_program_tokens':model_config.trajectory_points,
                'no_gt_phase_or_action_template_as_input':True,'no_gt_roi_substitution':True,
                'sam_gate_user_threshold':'all300 mean IoU > 0.90; no per-image minimum hard gate',
                'early_loss_threshold':float(settings.get('loss_overfit_threshold',.08)),
                'early_loss_window_steps':2*schedule.batches_per_epoch,
                'automatic_retry':False,'automatic_budget_extension':False}
    if metadata['manifest_sha256']!=parent['metadata']['manifest_sha256']:
        raise ValueError('Correction must use the same immutable all300 program input manifest')
    metadata.update(parent_checkpoint=str(parent_path),resumed_from_step=resumed_from,
                    original_absolute_budget=int(parent['target_steps']),remaining_updates=steps-completed,
                    metric_model_options=model_options,old_adam_restored=True,new_scale_encoder_group=True,
                    correction='observed scale + physical error + explicit pure-noise conditional learning')
    del parent
    write_json(output/'config.json',metadata)
    metrics, last_evaluation = {}, None
    history = deque(maxlen=2*schedule.batches_per_epoch)
    durations = deque(maxlen=40)
    result_status = 'running'
    def save(status):
        path = output/'checkpoints'/f'step_{completed:08d}.pt'
        save_checkpoint(path,{'format':'weldgrind-complete-program-metric-v3','model_config':model_config.to_dict(),
            'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':completed,'target_steps':steps,
            'metadata':metadata,'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else None,
            'seen_sample_ids':sorted(seen),'image_exposures':exposures,'loss_history':list(history),'metrics':metrics})
        write_json(output/'latest.json',{'checkpoint':str(path),'step':completed,'target_steps':steps,
                  'status':status,'local_save_complete':True})
        return path
    def evaluate(step,method='euler',ode_steps=8):
        from .program_evaluation_v3 import evaluate_metric_program_manifest
        report = evaluate_metric_program_manifest(model,manifest,output/'evaluations'/f'step_{step:08d}_{method}{ode_steps}',
                 max_points=int(config['data']['max_points']),batch_size=batch_size,flow_steps=ode_steps,
                 method=method,seed=seed,roi_gate_report=gate_path,prepared_samples=samples)
        write_json(output/'gate_latest.json',report)
        from .program_visualization import render_program_previews
        render_program_previews(report,manifest,output/'evaluations'/f'step_{step:08d}_{method}{ode_steps}'/'pictures',step=step)
        return report
    model.train()
    try:
        for batch in loader:
            if stop.exists():
                result_status='stopped'
                break
            tick=time.monotonic()
            ids=batch['sample_id']
            batch={key:value.to(device,non_blocking=True) if isinstance(value,torch.Tensor) else value for key,value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            losses=model.flow_loss(batch['points'],batch['roi_mask'],batch['trajectory'],batch['point_valid'],
                    scale_mm=batch['scale'],phase_labels=batch['phase_labels'],segment_ids=batch['segment_ids'],weights=settings.get('loss_weights'))
            if not bool(torch.isfinite(losses['loss'])):
                raise FloatingPointError('Nonfinite program loss; no automatic retry')
            losses['loss'].backward()
            finite=[torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
            if finite and not bool(torch.stack(finite).all()):
                raise FloatingPointError('Nonfinite program gradient; no automatic retry')
            optimizer.step()
            completed+=1
            exposures+=len(ids)
            seen.update(ids)
            metrics={name:float(value.detach()) for name,value in losses.items()}
            history.append(metrics['loss'])
            duration=time.monotonic()-tick
            durations.append(duration)
            extra={'actual_micro_batch':len(ids),'epoch':completed/schedule.batches_per_epoch,'metrics':metrics,
                   'resumed_from_step':resumed_from,'updates_in_corrected_stage':completed-resumed_from,
                   'step_seconds':duration,'samples_per_second':len(ids)/max(duration,1e-9),
                   'rolling_mean_loss':sum(history)/len(history),
                   'estimated_remaining_update_seconds':(steps-completed)*sum(durations)/len(durations),
                   'fps_cache_hits':acceleration['fps'].hits,'fps_cache_misses':acceleration['fps'].misses}
            if str(device).startswith('cuda'):
                extra.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                             cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            progress('running',**extra)
            loss_fitted=(len(history)==history.maxlen and len(seen)==len(samples)
                         and sum(history)/len(history)<=metadata['early_loss_threshold'])
            if completed%save_every==0:
                save('running')
            if loss_fitted:
                result_status='loss_overfit'
                break
            if completed%eval_every==0:
                last_evaluation=evaluate(completed)
                if last_evaluation.get('passed'):
                    last_evaluation=evaluate(completed,method='heun',ode_steps=32)
                    if last_evaluation.get('passed'):
                        result_status='gate_passed'
                        break
            if completed>=steps:
                break
        if result_status=='running':
            result_status='budget_exhausted'
        checkpoint=save(result_status)
        if result_status!='stopped' and result_status!='gate_passed':
            last_evaluation=evaluate(completed,method='heun',ode_steps=32)
        passed=bool(last_evaluation and last_evaluation.get('passed'))
        report=progress(result_status,passed=passed,checkpoint=str(checkpoint),metrics=metrics,
                        evaluation=last_evaluation,loss_early_stopped=result_status=='loss_overfit')
        write_json(output/'result.json',report)
        return report
    except Exception as error:
        checkpoint=save('failed')
        report=progress('failed',passed=False,error=str(error),checkpoint=str(checkpoint),automatic_retry=False)
        write_json(output/'failure.json',report)
        raise
    finally:
        del loader
