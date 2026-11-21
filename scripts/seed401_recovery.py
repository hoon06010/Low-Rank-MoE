"""Additive, pinned AdamW recovery; never edits the frozen training payload."""
import csv
import json
import math
from pathlib import Path
import shutil
import signal
import time
import traceback
import uuid
from scripts import seed401_local as local

PIN = {'snapshot_sha256':'7c8e95ac4e38ebe95b65a770f504f68f318eb2e7cb80ee47450eed6da0a559e8',
       'run_id':'c024d26e-d6c7-4fd5-b86e-67f4564fdc40', 'optimizer_step':128,
       'trained_target_tokens':524288,
       'checkpoint_sha256':'039f14163e1feebfc264bb151c6e2805499a6372192b2f8c9968b5912af9edf1',
       'main_reserved_seconds':12720}
FAILURES = {
 'startup-failure-20261007-2023': (1791372149,1791374100),
 'startup-failure-20261007-2116': (1791375300,1791375600),
}


def validate_ledger(ledger):
    local.remaining(ledger,'main'); local.remaining(ledger,'pilot')
    seen=set()
    for row in ledger['sessions']:
        if row['id'] in seen: raise ValueError('duplicate ledger session')
        seen.add(row['id'])
        if row['id'] in FAILURES:
            start,end=FAILURES[row['id']]
            if (row.get('status')!='reviewed_failure' or row.get('original_status')!='failed'
                or row.get('phase')!='main' or row.get('reserved_seconds')!=5400
                or row.get('allocated_at')!=start or row.get('ended_at_upper_bound')!=end
                or row.get('runtime')!='unknown-'+row['id'] or row.get('training_started') is not False
                or row.get('review')!='User requested new-notebook workaround after both startup failures; preserve full reservation.'):
                raise ValueError('reviewed failure identity/reservation mismatch')
        elif row.get('status')!='closed':
            raise ValueError('unresolved or unapproved session')
        if not isinstance(row.get('reserved_seconds'),(int,float)) or not math.isfinite(row['reserved_seconds']) or row['reserved_seconds']<=0:
            raise ValueError('invalid reservation')
    if not set(FAILURES).issubset(seen): raise ValueError('missing historical failure charges')


def verify_run_identity(out,payload):
    out=Path(out)
    rows=[json.loads(x) for x in (out/'run_manifest.jsonl').read_text().splitlines() if x.strip()]
    if len(rows)!=1 or rows[0].get('run_id')!=payload['run_id']:
        raise ValueError('checkpoint/manifest run identity mismatch')
    records=[(rows[0],'checkpoint_step')]
    if (out/'mini_report.json').exists(): records.append((json.loads((out/'mini_report.json').read_text()),'optimizer_step'))
    for record,key in records:
        if record.get('run_id')!=payload['run_id'] or record.get('status')=='failed':
            raise ValueError('report run identity mismatch or terminal failure')
        step=record.get(key)
        if type(step) is not int or not 0<=step<=payload['optimizer_step'] or record.get('trained_target_tokens')!=step*4096:
            raise ValueError('report/manifest counters mismatch')
        if step==payload['optimizer_step'] and record.get('checkpoint_id')!=payload['checkpoint_id']:
            raise ValueError('report/manifest checkpoint mismatch')
    if payload['trained_target_tokens']!=payload['optimizer_step']*4096 or payload.get('cursor',payload['optimizer_step']*8)!=payload['optimizer_step']*8 or payload.get('epoch',0)!=0:
        raise ValueError('checkpoint counters mismatch')


def restore_recovery(source,destination,pin=None):
    """Pin must come from a separately saved receipt, never from the source itself."""
    import torch
    source,destination=Path(source).resolve(),Path(destination).resolve()
    pin=dict(PIN if pin is None else pin)
    if local.sha(source/'snapshot.json')!=pin['snapshot_sha256']: raise ValueError('snapshot pin mismatch')
    if destination.exists() and any(destination.iterdir()): raise ValueError('restore requires empty destination')
    spec=json.loads((source/'snapshot.json').read_text())
    if spec.get('schema')!='seed401-local-v1' or 'ledger.json' not in spec['files']: raise ValueError('invalid snapshot')
    total=0
    for name,rec in spec['files'].items():
        p=source/name
        if not p.resolve().is_relative_to(source) or p.is_symlink() or not p.is_file() or p.stat().st_size!=rec['bytes'] or local.sha(p)!=rec['sha256']:
            raise ValueError('snapshot checksum mismatch')
        total+=rec['bytes']
    parent=destination.parent
    while not parent.exists(): parent=parent.parent
    if shutil.disk_usage(parent).free < total+3*1024**3: raise ValueError('insufficient copy disk space')
    ledger=json.loads((source/'ledger.json').read_text()); validate_ledger(ledger)
    if sum(r['reserved_seconds'] for r in ledger['sessions'] if r['phase']=='main')!=pin['main_reserved_seconds']:
        raise ValueError('main reservation pin mismatch')
    out=source/'main/LR-HL'; pointer,cp=local.pointer(out)
    if pointer['sha256']!=pin['checkpoint_sha256']: raise ValueError('checkpoint pin mismatch')
    payload=torch.load(cp,map_location='cpu',weights_only=False)
    for key in ('run_id','optimizer_step','trained_target_tokens'):
        if payload[key]!=pin[key]: raise ValueError('HL identity pin mismatch')
    if pointer['checkpoint_id']!=payload['checkpoint_id']: raise ValueError('checkpoint pointer mismatch')
    verify_run_identity(out,payload)
    for relative,expected in pin.get('runs',{}).items():
        target=source/relative
        if not target.resolve().is_relative_to(source): raise ValueError('unsafe run pin path')
        ptr,checkpoint=local.pointer(target)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if any(saved[k]!=expected[k] for k in ('run_id','optimizer_step','trained_target_tokens','checkpoint_id')) or ptr['sha256']!=expected['checkpoint_sha256']:
            raise ValueError('run pin mismatch')
        verify_run_identity(target,saved)
    destination.mkdir(parents=True,exist_ok=True)
    for name in [*spec['files'],'snapshot.json']:
        target=destination/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source/name,target)
    return pin


def missing_evaluation_steps(step, existing):
    expected=list(range(256,step+1,256))
    if len(existing)!=len(set(existing)) or any(x not in expected for x in existing): raise ValueError('invalid evaluation counters')
    missing=[s for s in expected if s not in existing]
    if any(s!=step for s in missing): raise ValueError('missing historical evaluation; cannot reconstruct from current checkpoint')
    return missing


def repair_boundary_evaluation(out,config_path,bundle_path,device='cuda'):
    """Restore exact frozen identity and fill only the current checkpoint evaluation."""
    import torch
    from scripts import run_colab_mini as mini
    from scripts.seed_repeat_policy import source_hashes
    from src.train import TrainingSession,_json_hash
    from src.model import MoETransformer
    out=Path(out); config=mini._config(Path(config_path)); mini._validate_grid(Path(config_path),config)
    audited=mini.load_audited_bundle(bundle_path)
    pointer,cp=local.pointer(out); payload=torch.load(cp,map_location='cpu',weights_only=False)
    verify_run_identity(out,payload)
    identity={'protocol_id':config['protocol_id'],'source_revision':_json_hash(source_hashes()),
      'data_manifest_hash':audited['bundle_hash'],'tokenizer_revision':audited['spec']['manifest']['tokenizer']['revision']['value'],
      'eval_revision':_json_hash(audited['batches']['dev']),'source_kind':'measured'}
    session=TrainingSession(MoETransformer(config['model']),config['model'],config['training'],audited['batches']['train'],identity,out,device=device,audit_bundle=bundle_path)
    session.restore_latest()
    rows=list(csv.DictReader((out/'metrics.csv').open())) if (out/'metrics.csv').exists() else []
    for row in rows:
        if row['run_id']!=session.run_id or row['seed']!='401' or row['config_id']!=config['model']['config_id'] or row['split']!='dev' or not math.isfinite(float(row['loss_nats'])):
            raise ValueError('evaluation identity mismatch')
    for row in rows:
        if int(row['trained_target_tokens'])==payload['trained_target_tokens'] and row['checkpoint_id']!=payload['checkpoint_id']:
            raise ValueError('current evaluation checkpoint mismatch')
    steps=[int(r['trained_target_tokens'])//4096 for r in rows]
    if any(int(r['trained_target_tokens'])%4096 for r in rows): raise ValueError('evaluation token mismatch')
    missing=missing_evaluation_steps(session.optimizer_step,steps)
    complete=session.optimizer_step==2048
    result=session.result('completed' if complete else 'incomplete')
    evaluation=None
    if missing:
        evaluation=mini._eval_metric(session,audited['batches']['dev'],config,result,out,device)
    if session.optimizer_step and session.optimizer_step%256==0:
        batch=audited['batches']['dev'][0]
        with torch.no_grad():
            _,_,diagnostic=session.model(torch.as_tensor(batch['input_ids'],dtype=torch.long,device=device),torch.as_tensor(batch['attention_mask'],dtype=torch.bool,device=device))
        mini.append_routing_diagnostic(out/'routing_diagnostics.jsonl',result,diagnostic)
    if complete:
        rows=list(csv.DictReader((out/'metrics.csv').open()))
        report_path=out/'mini_report.json'; report=json.loads(report_path.read_text()) if report_path.exists() else {}
        last=rows[-1]
        if last['checkpoint_id']!=result['checkpoint_id']: raise ValueError('final evaluation checkpoint mismatch')
        report.update(source_kind='measured',purpose='mini_exploration_only',precision='float32',pilot=False,device=str(device),parameter_count=sum(p.numel() for p in session.model.parameters()),actual_tokens_total=session.trained_target_tokens,optimizer_steps_this_invocation=0,actual_tokens_this_invocation=0)
        report.update(result,model=config['model'],training=session.training_config,source_file_hashes=source_hashes(),data_manifest_hash=identity['data_manifest_hash'],eval_revision=identity['eval_revision'],loss_nats=float(last['loss_nats']),eval_target_tokens=int(last['eval_target_tokens']),recovered_evaluation_only=True)
        local.verify_final(out,report,config['model']['cell'])
        local.write(report_path,report); session._write_run_manifest('completed','final evaluation recovered without training')
    return result


def session_budget(allocated_at,stop_at,now,quota_hours):
    if any(not math.isfinite(v) for v in (allocated_at,stop_at,now,quota_hours)) or allocated_at>now or quota_hours<=0:
        raise ValueError('invalid time/quota')
    seconds=min(5400,stop_at-allocated_at)
    if seconds<=now-allocated_at+local.SAVE_RESERVE+300 or quota_hours*3600<seconds:
        raise ValueError('insufficient session time including save reserve')
    return seconds


def run_recovery(quota_hours,allocated_at,stop_at,state_dir=local.STATE):
    """One operator-supervised runtime; always save Output before GPU shutdown."""
    import fcntl
    import gc
    import torch
    from scripts.run_colab_mini import run_colab_mini
    state_dir=Path(state_dir)
    seconds=session_budget(allocated_at,stop_at,time.time(),quota_hours)
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0)!='Tesla T4': raise ValueError('Tesla T4 required')
    torch.cuda.set_device(0); local.validate_payload()
    with (state_dir/'operator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=json.loads((state_dir/'ledger.json').read_text()); validate_ledger(ledger)
        reports={c:json.loads((state_dir/'pilot'/f'LR-{c}'/'mini_report.json').read_text()) for c in local.ORDER}
        local.pilot_estimate(reports)
        runtime=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        row=reserve_recovery(ledger,seconds,runtime)
        row.update(allocated_at=allocated_at,stop_at=stop_at,quota_hours_observed=quota_hours)
        local.write(state_dir/'ledger.json',ledger)
        deadline=allocated_at+seconds-local.SAVE_RESERVE
        def timeout(*_): raise TimeoutError('save deadline reached')
        previous=signal.signal(signal.SIGALRM,timeout); signal.setitimer(signal.ITIMER_REAL,max(1,deadline-time.time()))
        try:
            def boundary(session,stage):
                if shutil.disk_usage(state_dir).free<3*1024**3 or sum(p.stat().st_size for p in state_dir.rglob('*') if p.is_file())>=30*1024**3: raise ValueError('disk budget exceeded')
                if session.epoch!=0 or session.cursor!=session.optimizer_step*8 or session.trained_target_tokens!=session.optimizer_step*4096: raise ValueError('training counters mismatch')
                if stage in ('initial','after_chunk','after_report'): local.retain_two(session.output_dir)
                return time.time()<deadline-300
            for cell in local.ORDER:
                out=state_dir/'main'/f'LR-{cell}'; config=local.ROOT/f'configs/seed_repeat_c/seed-401/LR-{cell}.json'; bundle=local.ROOT/'data/prepared/colab_free_mini_v1/bundle.json'
                resume=(out/'checkpoints/latest.json').exists()
                if resume and time.time()<deadline-300:
                    restored=repair_boundary_evaluation(out,config,bundle)
                    if restored['status']=='completed': continue
                while time.time()<deadline-300:
                    resume=(out/'checkpoints/latest.json').exists()
                    row['cell']=cell; local.write(state_dir/'ledger.json',ledger)
                    report=run_colab_mini(config,bundle,out,device='cuda',resume=resume,max_steps=2048,max_seconds=deadline-time.time()-180,boundary_hook=boundary)
                    if report['status']=='failed': raise RuntimeError('training failed; preserve original state')
                    if report['status']=='completed': local.verify_final(out,report,cell); break
                    gc.collect(); torch.cuda.empty_cache()
                if time.time()>=deadline-300: break
            row.update(status='closed',elapsed_allocation_seconds=time.time()-allocated_at)
        except BaseException:
            row['status']='failed'; (state_dir/f'failure-{row["id"]}.log').write_text(traceback.format_exc()); raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous)
            local.write(state_dir/'ledger.json',ledger)
            digest=local.seal(state_dir)
            print('SNAPSHOT:',digest,'Save Output, verify Successful, then stop GPU.',flush=True)
    return ledger


def make_pin(state_dir):
    """Call after final console closure and seal; retain this receipt outside Output."""
    import torch
    state_dir=Path(state_dir)
    ledger=json.loads((state_dir/'ledger.json').read_text()); validate_ledger(ledger)
    spec=json.loads((state_dir/'snapshot.json').read_text())
    for name,rec in spec['files'].items():
        path=state_dir/name
        if not path.resolve().is_relative_to(state_dir.resolve()) or path.is_symlink() or path.stat().st_size!=rec['bytes'] or local.sha(path)!=rec['sha256']:
            raise ValueError('snapshot must be sealed after console closes')
    runs={}
    for out in sorted((state_dir/'main').glob('LR-*')):
        ptr,cp=local.pointer(out); saved=torch.load(cp,map_location='cpu',weights_only=False)
        verify_run_identity(out,saved)
        runs[str(out.relative_to(state_dir))]={**{k:saved[k] for k in ('run_id','optimizer_step','trained_target_tokens','checkpoint_id')},'checkpoint_sha256':ptr['sha256']}
    hl=runs['main/LR-HL']
    return {**hl,'snapshot_sha256':local.sha(state_dir/'snapshot.json'),
            'main_reserved_seconds':sum(r['reserved_seconds'] for r in ledger['sessions'] if r['phase']=='main'),'runs':runs}


def reserve_recovery(ledger,seconds,runtime):
    """New explicit authorization lifts total user caps; historical limits stay intact."""
    validate_ledger(ledger)
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0<seconds<=5400:
        raise ValueError('invalid session reservation')
    if any(r['runtime']==runtime for r in ledger['sessions']): raise ValueError('fresh runtime required')
    ledger['continuation_authorization']={'reference':'User: 전부 진행해. 시간 한도는 상관없어.',
      'total_user_time_cap_seconds':None,'session_limit_seconds':5400,'save_reserve_seconds':1020,
      'cost_limit_krw':0,'historical_reservations_preserved':True}
    row={'id':uuid.uuid4().hex,'runtime':runtime,'phase':'main','reserved_seconds':seconds,
         'status':'running','started_at':time.time(),'authorization_reference':ledger['continuation_authorization']['reference']}
    ledger['sessions'].append(row)
    return row
