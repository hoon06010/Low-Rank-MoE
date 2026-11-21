"""Real FineWeb-Edu CPU setup verification; outputs are not hypothesis results."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import torch
from src.real_preflight import run_real_preflight
from src.train import _atomic_json, _sha256_file

ROOT=Path(__file__).resolve().parents[1]

def equal(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray): return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def checkpoint(directory):
    p=json.loads((directory/'checkpoints/latest.json').read_text())
    path=directory/'checkpoints'/p['filename']
    assert _sha256_file(path)==p['sha256']
    return torch.load(path,map_location='cpu',weights_only=False)

def main():
    torch.set_num_threads(2)
    cfg=ROOT/'configs/real_cpu_smoke.json'
    bundle=ROOT/'data/prepared/fineweb_10k/bundle.json'
    full_dir=ROOT/'results/preflight/real_cpu_continuous_v2'
    resumed_dir=ROOT/'results/preflight/real_cpu_resumed_v2'
    full=run_real_preflight(cfg,bundle,full_dir)
    print('continuous',full['status'],full['trained_target_tokens'],flush=True)
    half=json.loads(cfg.read_text());half.update(warmup_steps=0,measurement_steps=1)
    half_path=ROOT/'reports/m9/interrupt_config.json';_atomic_json(half_path,half)
    partial=run_real_preflight(half_path,bundle,resumed_dir)
    print('interrupted',partial['status'],partial['trained_target_tokens'],flush=True)
    assert partial['status']=='incomplete' and partial['trained_target_tokens']==512
    resumed=run_real_preflight(cfg,bundle,resumed_dir,resume=True)
    print('resumed',resumed['status'],resumed['trained_target_tokens'],flush=True)
    a,b=checkpoint(full_dir),checkpoint(resumed_dir)
    fields=['model_state_dict','optimizer_state_dict','scheduler_state_dict','rng_state','data_order','cursor','epoch','trained_target_tokens','optimizer_step']
    checks={field:equal(a[field],b[field]) for field in fields}
    checks['dev_loss_exact_equal']=full['loss_nats']==resumed['loss_nats']
    checks['both_completed']=full['status']==resumed['status']=='completed'
    checks['expected_training_targets']=full['trained_target_tokens']==resumed['trained_target_tokens']==2048
    report={'purpose':'pipeline_smoke_only','source_kind':'measured','device':'cpu','torch_threads':torch.get_num_threads(),
            'checks':checks,'all_passed':all(checks.values()),'trained_targets_per_run':2048,
            'eval_targets_per_run':full['eval_target_tokens'],'data_manifest_hash':full['data_manifest_hash'],
            'continuous_run_id':full['run_id'],'resumed_run_id':resumed['run_id'],
            'gpu_status':'not_run_no_cuda','config_sha256':_sha256_file(cfg),
            'note':'Tiny randomly initialized model; no research contrast or performance claim.'}
    _atomic_json(ROOT/'reports/m9/real_cpu_verification.json',report)
    print(json.dumps(report,indent=2),flush=True)
    assert report['all_passed'],report

if __name__=='__main__':main()
