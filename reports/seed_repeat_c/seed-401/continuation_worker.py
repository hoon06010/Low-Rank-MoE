"""Private CUDA step1→step2 worker; invoked only by a reserved parent runner.

The local envelope is an operational capability, not a cryptographic sandbox.
Standalone CLI work without its living parent reservation is rejected.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import drive_runner as runner

def validate_reservation(path,branch,*,now=None):
    now=time.time() if now is None else now
    path=Path(path); branch=Path(branch)
    if path.is_symlink() or not path.is_file(): raise ValueError('invalid parent reservation')
    record=json.loads(path.read_text())
    if (not isinstance(record.get('id'),str) or len(record['id'])!=32
            or any(c not in '0123456789abcdef' for c in record['id'])
            or record.get('status')!='reserved' or record.get('id')!=os.environ.get('SEED401_RESERVED_INVOCATION')
            or record.get('parent_pid')!=os.getppid() or record.get('expires_at',0)<=now
            or record.get('spec',{}).get('seed')!=401 or record.get('spec',{}).get('phase')!='recovery'
            or not record.get('run_id')):
        raise ValueError('invalid/expired parent reservation')
    base=(runner.HERE/'operational-control').resolve()
    if (not path.resolve().is_relative_to(base) or not branch.resolve().is_relative_to(path.parent.resolve())
            or branch.is_symlink() or not branch.is_dir()):
        raise ValueError('branch must belong to reserved recovery temporary directory')
    expected=runner.baseline.build_spec(401,record['spec']['cell'],'recovery')
    if record['spec']!=expected: raise ValueError('reserved frozen spec mismatch')
    return record

def advance(path,branch):
    record=validate_reservation(path,branch); spec=record['spec']; gate=record['gate']
    runner.validate_gate(gate,spec)
    runner.hardware_gate(gate)
    import torch
    session=runner.make_training_session(spec,branch)
    session.restore_latest()
    pointer,checkpoint=runner._checkpoint(branch)
    captured=session._checkpoint_payload(pointer['checkpoint_id'])
    expected=dict(spec,run_id=record['run_id'])
    restored=runner.comparator.compare_payloads(runner.comparator.verified_load(checkpoint,pointer['sha256']),captured,expected)
    if session.optimizer_step!=1 or session.cursor!=8 or session.trained_target_tokens!=4096:
        raise ValueError('continuation requires the same verified step1 checkpoint')
    result=session.fit(max_optimizer_steps=1)
    if result['status']=='failed' or session.optimizer_step!=2 or session.cursor!=16 or session.trained_target_tokens!=8192:
        raise ValueError('continuation did not reach exact step2 counters')
    pointer,checkpoint=runner._checkpoint(branch)
    rows=[json.loads(line) for line in (Path(branch)/'run_manifest.jsonl').read_text().splitlines()]
    if len(rows)!=1: raise ValueError('continuation single manifest required')
    rows[0].update(config_file_sha256=spec['config_file_sha256'],execution_source_revision=spec['source_revision'],
                   experiment_id=spec['experiment_id'],train_order_policy=spec['train_order_policy'])
    (Path(branch)/'run_manifest.jsonl').write_text(json.dumps(rows[0],sort_keys=True)+'\n')
    summary={'status':'continuation-branch-verified','run_id':session.run_id,'step1_restore':restored,
             'optimizer_step':2,'cursor':16,'trained_target_tokens':8192,'checkpoint':pointer,
             'checkpoint_sha256':runner.digest(checkpoint),'execution_device':'cuda','process_id':os.getpid()}
    runner._atomic_json(Path(branch)/'continuation-branch.json',summary)
    del session,captured
    torch.cuda.empty_cache()
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reservation',type=Path,required=True); parser.add_argument('--branch',type=Path,required=True)
    args=parser.parse_args(); advance(args.reservation,args.branch)

if __name__=='__main__': main()
