"""Bounded audited real-text FP32 CPU/CUDA smoke and instrumentation runner."""
from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.data import load_audited_bundle
from src.evaluate import evaluate
from src.model import MoETransformer
from src.train import TrainingSession, _atomic_json, _json_hash, _sha256_file, write_metrics_csv


def run_real_preflight(config_path, bundle_path, output_dir, *, device='cpu', resume=False):
    selected = torch.device(device)
    if selected.type not in {'cpu','cuda'}:
        raise ValueError('supported devices are cpu and cuda')
    if selected.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; GPU execution has not been verified')
    config=json.loads(Path(config_path).read_text())
    data=load_audited_bundle(bundle_path)
    out=Path(output_dir)
    if not resume and (out/'checkpoints/latest.json').exists():
        raise ValueError('output already has a checkpoint; use --resume or a fresh output')
    mc=config['model']; tc=config['training']
    random.seed(mc['seed']);np.random.seed(mc['seed']);torch.manual_seed(mc['seed'])
    if selected.type=='cuda': torch.cuda.manual_seed_all(mc['seed'])
    model=MoETransformer(mc)
    source_files={p.name:_sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    identity={'protocol_id':config['protocol_id'],'source_revision':_json_hash(source_files),
              'data_manifest_hash':data['bundle_hash'],
              'tokenizer_revision':data['spec']['manifest']['tokenizer']['revision']['value'],
              'eval_revision':_json_hash(data['batches']['dev']),'source_kind':'measured'}
    session=TrainingSession(model,mc,tc,data['batches']['train'],identity,out,device=device,audit_bundle=bundle_path)
    if resume: session.restore_latest()
    warmup_steps=config.get('warmup_steps',1)
    measurement_steps=config.get('measurement_steps',3)
    if type(warmup_steps) is not int or warmup_steps < 0 or type(measurement_steps) is not int or measurement_steps <= 0:
        raise ValueError('warmup and measurement steps must be bounded integers')
    def sync():
        if selected.type=='cuda': torch.cuda.synchronize(selected)
    if warmup_steps:
        warmup=session.fit(max_optimizer_steps=warmup_steps)
        if warmup['status']=='failed':
            _atomic_json(out/'real_preflight.json',{'status':'failed','phase':'warmup',**warmup})
            return {'status':'failed',**warmup}
    sync()
    if selected.type=='cuda': torch.cuda.reset_peak_memory_stats(selected)
    before=session.trained_target_tokens
    save_before=session.checkpoint_seconds
    started=time.perf_counter()
    result=session.fit(max_optimizer_steps=measurement_steps)
    sync();elapsed=time.perf_counter()-started
    checkpoint_seconds=session.checkpoint_seconds-save_before
    measured_tokens=result['trained_target_tokens']-before
    gpu_allocated=torch.cuda.max_memory_allocated(selected) if selected.type=='cuda' else None
    gpu_reserved=torch.cuda.max_memory_reserved(selected) if selected.type=='cuda' else None
    evaluation=None;eval_seconds=None
    if result['status']!='failed':
        sync();started=time.perf_counter()
        evaluation=evaluate(model,data['batches']['dev'],device=device)
        sync();eval_seconds=time.perf_counter()-started
        metric={'run_id':result['run_id'],'protocol_id':config['protocol_id'],
                'config_id':mc['config_id'],'family':mc['family'],'cell':mc['cell'],'seed':mc['seed'],
                'split':'dev','checkpoint_id':result['checkpoint_id'],
                'trained_target_tokens':result['trained_target_tokens'],
                'eval_target_tokens':evaluation['eval_target_tokens'],'loss_nats':evaluation['loss_nats'],
                'status':result['status'],'source_kind':'measured'}
        metrics_path=out/'metrics.csv'
        # No-op resumes do not add a duplicate evaluation primary key.
        import csv
        existing=list(csv.DictReader(metrics_path.open())) if metrics_path.exists() else []
        if not any(r['run_id']==metric['run_id'] and r['checkpoint_id']==metric['checkpoint_id'] and r['split']=='dev' for r in existing):
            write_metrics_csv(metrics_path,metric)
    report={**result,'source_kind':'measured','purpose':'pipeline_smoke_only','device':str(selected),
            'precision':'float32','data_manifest_hash':data['bundle_hash'],'eval_revision':identity['eval_revision'],
            'source_file_hashes':source_files,'hardware':torch.cuda.get_device_name(selected) if selected.type=='cuda' else platform.processor(),
            'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'python_version':platform.python_version(),
            'warmup_steps_requested':warmup_steps,'measurement_steps_requested':measurement_steps,
            'measured_target_tokens':measured_tokens,'measurement_seconds_including_checkpoint':elapsed,
            'checkpoint_seconds_during_measurement':checkpoint_seconds,
            'tokens_per_second_including_checkpoint':measured_tokens/elapsed if measured_tokens else None,
            'tokens_per_second_excluding_checkpoint':measured_tokens/(elapsed-checkpoint_seconds) if measured_tokens and elapsed>checkpoint_seconds else None,
            'timing_scope':'instrumented FP32 trainer; host routing diagnostics and finite checks retained',
            'evaluation_seconds':eval_seconds,'eval_target_tokens':evaluation['eval_target_tokens'] if evaluation else None,
            'loss_nats':evaluation['loss_nats'] if evaluation else None,
            'gpu_memory_bytes':gpu_allocated,'gpu_reserved_bytes':gpu_reserved,
            'gpu_measurement_status':'measured_training_window' if selected.type=='cuda' else 'not_measured_cpu_only'}
    _atomic_json(out/'real_preflight.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--bundle',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--device',default='cpu')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    report=run_real_preflight(args.config,args.bundle,args.output,device=args.device,resume=args.resume)
    print(json.dumps(report,indent=2))
    return 1 if report['status']=='failed' else 0


if __name__=='__main__':
    raise SystemExit(main())
