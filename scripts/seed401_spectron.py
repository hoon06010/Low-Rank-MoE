"""Separate fixed-recipe seed401 Spectron runner. Never modifies frozen AdamW code.
CUDA pilots and fresh-runtime optimizer restoration are required before main.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,platform,random,signal,time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.data import load_audited_bundle
from src.model import MoETransformer
from src.train import TrainingSession,_atomic_json,_json_hash,_sha256_file,_target_count
from scripts.run_colab_mini import _eval_metric,append_routing_diagnostic
from scripts.seed401_local import pointer,retain_two,seal,remaining,verify_final
EXPERIMENT='seed401-spectron-fixed-recipe-v1'
PURPOSE='mini_exploration_only'
GLOBAL_BATCH=4096
ROW_TARGETS=512
CHUNK_STEPS=32
EVAL_INTERVAL=1048576
ORDER=('HS','LL')
GPU_DEVICE='cuda:0'

def runtime_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()

def load_config(path):
    c=json.loads(Path(path).read_text())
    cell=c.get('model',{}).get('cell')
    if cell not in ORDER:raise ValueError('only Spectron HS/LL allowed')
    expected=json.loads((ROOT/f'configs/seed_repeat_c/seed-401/LR-{cell}.json').read_text())
    expected['experiment_id']=EXPERIMENT
    expected['training'].update(recipe_id=EXPERIMENT,optimizer='spectron',spectron_momentum=.95,spectron_ns_steps=5,spectron_power_steps=1,spectron_weight_decay=0.)
    if c!=expected:raise ValueError('fixed Spectron config mismatch')
    return c

def source_identity():
    names=[*sorted((ROOT/'src').glob('*.py')),Path(__file__),ROOT/'scripts/run_colab_mini.py',ROOT/'scripts/seed401_local.py']
    return {str(p.relative_to(ROOT)):_sha256_file(p) for p in names}

def state_hash(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            t=x.detach().cpu().contiguous();h.update(str((str(t.dtype),list(t.shape))).encode());h.update(t.numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=lambda k:(type(k).__name__,str(k))):h.update(repr(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(str(type(x)).encode())
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()

def validate_resources(r,now=None):
    try:
        values=[r[k] for k in ('allocated_at','stop_at','quota_hours','budget_seconds','save_reserve_seconds')]
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):raise ValueError('nonfinite resource input')
        allocated,stop,quota,budget,reserve=values; now=time.time() if now is None else now
        seconds=stop-allocated
        if not (0<=allocated<=now<stop-reserve and 1020<=reserve<seconds<=5400 and seconds<=budget and seconds<=quota*3600):raise ValueError('resource bounds invalid')
        return seconds
    except (KeyError,TypeError):raise ValueError('explicit resource inputs required') from None

def main_gate(state,adamw):
    state,adamw=Path(state),Path(adamw)
    estimate=0.;ids=set()
    for cell in ORDER:
        p=state/'preflight'/f'LR-{cell}'
        if not (p/'mini_report.json').exists() or not (p/'recovery_proof.json').exists():raise ValueError('actual CUDA pilot and recovery evidence required')
        r=json.loads((p/'mini_report.json').read_text()); proof=json.loads((p/'recovery_proof.json').read_text())
        ptr,_=pointer(p)
        if not (r.get('experiment_id')==EXPERIMENT and r.get('device','').startswith('cuda') and r.get('pilot_usable') is True and r.get('pilot_measurement_complete') is True and r.get('pilot_measured_steps')==100 and r.get('optimizer_step')==105 and r.get('trained_target_tokens')==430080 and r.get('status')!='failed' and r['model']['cell']==cell and r['model']['seed']==401 and r['training']['recipe_id']==EXPERIMENT and r['run_id'] not in ids):raise ValueError('CUDA pilot evidence invalid')
        if not (proof.get('experiment_id')==EXPERIMENT and proof.get('run_id')==r['run_id'] and proof.get('checkpoint_sha256')==ptr['sha256'] and proof.get('optimizer_state_hash')==r['optimizer_state_hash'] and proof.get('restored_runtime')!=r['runtime_id'] and proof.get('source_runtime')==r['runtime_id'] and proof.get('source_revision')==_json_hash(source_identity())):raise ValueError('fresh-runtime optimizer recovery evidence invalid')
        ids.add(r['run_id'])
        train,evaluation=r['elapsed_training_seconds_including_checkpoint'],r['evaluation_seconds_this_invocation']
        if not (math.isfinite(train) and train>0 and math.isfinite(evaluation) and evaluation>=0):raise ValueError('invalid pilot timing')
        estimate+=train*2048/100+evaluation*8
    adamw_gate(adamw)
    return estimate*1.3+3600

def adamw_gate(adamw):
    from scripts.analyze_seed_repeat_c import collect,ADAMW
    adamw=Path(adamw)
    for cell in ('HL','HS','LS','LL'):
        p=adamw/'main'/f'LR-{cell}'
        if not (p/'mini_report.json').exists():raise ValueError('all four AdamW results required first')
        r=json.loads((p/'mini_report.json').read_text())
        if r.get('status')!='completed' or r.get('training',{}).get('optimizer')!='adamw':raise ValueError('AdamW anchor incomplete')
        verify_final(p,r,cell)
        accepted,reason=collect({'experiment_id':ADAMW,'phase':'main','path':str(p)})
        if accepted is None or accepted['seed']!=401 or accepted['cell']!=cell:
            raise ValueError('AdamW anchor identity rejected: '+str(reason))

def run_spectron(config_path, bundle_path, output_dir, *, device="cuda", resume=False,
                   max_steps=32, max_seconds=None, pilot=False, boundary_hook=None, recovery_only=False):
    selected = torch.device(device)
    if selected.type not in {"cpu", "cuda"}:
        raise ValueError("supported devices are cpu and cuda")
    if selected.type == "cpu" and max_steps is None and not pilot:
        raise ValueError("CPU is reserved for bounded tests; specify --max-steps")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; no training was started")
    if max_steps is not None and (type(max_steps) is not int or max_steps <= 0):
        raise ValueError("max_steps must be a positive integer")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    if pilot and resume:
        raise ValueError("pilot requires a fresh run and cannot resume")

    config_path, bundle_path, out = Path(config_path).resolve(), Path(bundle_path).resolve(), Path(output_dir).resolve()
    config = load_config(config_path)
    audited = load_audited_bundle(bundle_path)
    if audited["spec"]["purpose"] != PURPOSE:
        raise ValueError(f"bundle purpose must be {PURPOSE}")
    if audited["spec"]["vocab_size"] != config["model"]["vocab_size"] or audited["spec"]["context_length"] != config["model"]["context_length"]:
        raise ValueError("model and mini bundle dimensions differ")
    batches = audited["batches"]["train"]
    counts = [_target_count(row) for row in batches]
    if not counts or any(count != ROW_TARGETS for count in counts):
        raise ValueError("every training row must contain exactly 512 valid targets")
    available = sum(counts)
    target = config["training"]["total_target_tokens"]
    if available < target:
        raise ValueError(f"mini bundle has {available} unique packed targets; budget requires {target}; cycling is forbidden")
    if target % GLOBAL_BATCH:
        raise ValueError("token budget is not aligned to the global target batch")

    if out.exists():
        if not resume and any(out.iterdir()):
            raise ValueError("output directory is not empty; use --resume or choose a fresh output")
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "run_manifest.jsonl"
    if resume:
        if not (out / "checkpoints/latest.json").is_file() or not manifest.is_file():
            raise ValueError("resume requires a checkpoint and run manifest")
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if not rows or any(row.get("status") == "failed" for row in rows):
            raise ValueError("failed run manifests cannot be resumed")
        if len({row.get("run_id") for row in rows}) != len(rows):
            raise ValueError("run manifest contains duplicate run IDs")
    elif (out / "checkpoints/latest.json").exists():
        raise ValueError("output already has a checkpoint; use --resume or a fresh output")

    source_hashes = source_identity()
    random.seed(config["model"]["seed"]); np.random.seed(config["model"]["seed"]); torch.manual_seed(config["model"]["seed"])
    if selected.type == "cuda":
        torch.cuda.manual_seed_all(config["model"]["seed"])
    identity = {
        "protocol_id": config["protocol_id"], "source_revision": _json_hash(source_hashes),
        "data_manifest_hash": audited["bundle_hash"],
        "tokenizer_revision": audited["spec"]["manifest"]["tokenizer"]["revision"]["value"],
        "eval_revision": _json_hash(audited["batches"]["dev"]), "source_kind": "measured",
    }
    session = TrainingSession(MoETransformer(config["model"]), config["model"], config["training"], batches,
                              identity, out, device=device, audit_bundle=bundle_path)
    if resume:
        session.restore_latest()
        check_resume_run(rows,session.run_id)
    elif boundary_hook is not None:
        # An overnight run needs a durable zero-step generation before training.
        session._save_checkpoint()
        session._write_run_manifest("incomplete", "initial checkpoint")
        if boundary_hook(session, "initial") is False:
            raise ValueError("overnight preflight rejected the initial checkpoint")

    # 역할: CUDA를 사용할 때 작업 완료를 기다려 실행 시간을 정확히 잴 수 있게 한다.
    def sync():
        if selected.type == "cuda":
            torch.cuda.synchronize(selected)

    if recovery_only:
        return {"optimizer_state_hash":state_hash(session.optimizer.state_dict()), "run_id":session.run_id, "optimizer_step":session.optimizer_step}

    start_step = session.optimizer_step
    invocation_checkpoint_seconds = session.checkpoint_seconds
    if selected.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected)
    elapsed_train = elapsed_eval = 0.0
    evaluation = None
    routing_diagnostic = None
    measured_tokens = 0
    warmup_tokens = 0
    warmup_elapsed = 0.0
    sync()
    invocation_started = time.perf_counter()
    if pilot:
        warm_start = session.optimizer_step
        warm_tokens_start = session.trained_target_tokens
        sync(); warm_started = time.perf_counter()
        warmup = session.fit(max_optimizer_steps=5)
        sync(); warmup_elapsed = time.perf_counter() - warm_started
        warmup_tokens = session.trained_target_tokens - warm_tokens_start
        if warmup["status"] == "failed":
            result = warmup
            _atomic_json(out / "mini_report.json", {**warmup, "purpose": PURPOSE, "phase": "pilot_warmup_failed"})
            return {**warmup, "purpose": PURPOSE}
        if session.optimizer_step - warm_start < 5:
            raise ValueError("token budget ended before pilot warmup completed")

    result = session.result('completed' if session.trained_target_tokens == target else 'incomplete')
    if not pilot and session.trained_target_tokens and session.trained_target_tokens % EVAL_INTERVAL == 0:
        if evaluation_due(out,session.trained_target_tokens,session.run_id,result['checkpoint_id']):
            sync(); eval_started=time.perf_counter()
            evaluation=_eval_metric(session,audited['batches']['dev'],config,result,out,device)
            sync(); elapsed_eval+=time.perf_counter()-eval_started
        else:
            with (out/'metrics.csv').open() as f:
                row=next(r for r in csv.DictReader(f) if r['run_id']==session.run_id and r['checkpoint_id']==result['checkpoint_id'])
            evaluation={'loss_nats':float(row['loss_nats']),'eval_target_tokens':int(row['eval_target_tokens'])}
        sync(); diagnostic_started=time.perf_counter()
        dev_batch=audited['batches']['dev'][0]
        with torch.no_grad():
            ids=torch.as_tensor(dev_batch['input_ids'],dtype=torch.long,device=selected)
            mask=torch.as_tensor(dev_batch['attention_mask'],dtype=torch.bool,device=selected)
            _,_,routing_diagnostic=session.model(ids,mask)
        append_routing_diagnostic(out/'routing_diagnostics.jsonl',result,routing_diagnostic)
        sync();elapsed_eval+=time.perf_counter()-diagnostic_started

    measured_start_step = session.optimizer_step
    invocation_checkpoint_seconds = session.checkpoint_seconds
    sync()
    cap = 100 if pilot else (max_steps if max_steps is not None else target // GLOBAL_BATCH)
    while session.trained_target_tokens < target and session.optimizer_step - measured_start_step < cap:
        if boundary_hook is not None and boundary_hook(session, "before_chunk") is False:
            break
        if max_seconds is not None and time.perf_counter() - invocation_started >= max_seconds:
            break
        previous_tokens = session.trained_target_tokens
        count = min(CHUNK_STEPS, cap - (session.optimizer_step - measured_start_step))
        if not pilot:
            until_eval = EVAL_INTERVAL - (previous_tokens % EVAL_INTERVAL)
            count = min(count, max(1, until_eval // GLOBAL_BATCH))
        sync()
        chunk_started = time.perf_counter()
        result = session.fit(max_optimizer_steps=count)
        sync(); elapsed_train += time.perf_counter() - chunk_started
        if result["status"] == "failed":
            break
        measured_tokens += session.trained_target_tokens - previous_tokens
        crossed = session.trained_target_tokens // EVAL_INTERVAL > previous_tokens // EVAL_INTERVAL
        final = result["status"] == "completed"
        if crossed or final or pilot and session.optimizer_step - measured_start_step >= 100:
            sync(); eval_started = time.perf_counter()
            evaluation = _eval_metric(session, audited["batches"]["dev"], config, result, out, device)
            dev_batch = audited["batches"]["dev"][0]
            with torch.no_grad():
                ids = torch.as_tensor(dev_batch["input_ids"], dtype=torch.long, device=selected)
                mask = torch.as_tensor(dev_batch["attention_mask"], dtype=torch.bool, device=selected)
                _, _, routing_diagnostic = session.model(ids, mask)
            append_routing_diagnostic(out / "routing_diagnostics.jsonl", result, routing_diagnostic)
            sync(); elapsed_eval += time.perf_counter() - eval_started
        if boundary_hook is not None and boundary_hook(session, "after_chunk") is False:
            break
        print(json.dumps({"status": result["status"], "optimizer_step": result["optimizer_step"],
                          "trained_target_tokens": result["trained_target_tokens"]}, separators=(",", ":")), flush=True)
        if result["status"] in {"completed", "failed"}:
            break

    sync()
    if "result" not in locals():
        result = session.result("incomplete")
    measured_elapsed = elapsed_train
    checkpoint_seconds = session.checkpoint_seconds - invocation_checkpoint_seconds
    allocated = torch.cuda.max_memory_allocated(selected) if selected.type == "cuda" else None
    reserved = torch.cuda.max_memory_reserved(selected) if selected.type == "cuda" else None
    report = {
        **result, "purpose": PURPOSE, "source_kind": "measured", "device": str(selected), "precision": "float32",
        "model": config["model"], "training": session.training_config,
        "last_step_diagnostics": session.last_step_diagnostics, "data_manifest_hash": audited["bundle_hash"],
        "source_file_hashes": source_hashes, "unique_train_target_tokens_available": available,
        "eval_revision": identity["eval_revision"], "actual_tokens_this_invocation": measured_tokens,
        "actual_tokens_total": session.trained_target_tokens, "optimizer_steps_this_invocation": session.optimizer_step - start_step,
        "elapsed_training_seconds_including_checkpoint": measured_elapsed,
        "checkpoint_seconds_this_invocation": checkpoint_seconds,
        "tokens_per_second_including_checkpoint": measured_tokens / measured_elapsed if measured_elapsed else None,
        "tokens_per_second_excluding_checkpoint": measured_tokens / (measured_elapsed - checkpoint_seconds) if measured_elapsed > checkpoint_seconds else None,
        "evaluation_seconds_this_invocation": elapsed_eval, "eval_target_tokens": evaluation["eval_target_tokens"] if evaluation else None,
        "loss_nats": evaluation["loss_nats"] if evaluation else None,
        "perplexity": math.exp(evaluation["loss_nats"]) if evaluation and evaluation["loss_nats"] < math.log(float.fromhex("0x1.fffffffffffffp+1023")) else None,
        "parameter_count": sum(parameter.numel() for parameter in session.model.parameters()),
        "routing_diagnostic_scope": "first_dev_batch_only" if evaluation else None,
        "routing_diagnostic": routing_diagnostic if evaluation else None,
        "gpu_memory_allocated_peak_bytes": allocated,
        "gpu_memory_reserved_peak_bytes": reserved, "hardware": torch.cuda.get_device_name(selected) if selected.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "python_version": platform.python_version(),
        "pilot": pilot, "pilot_warmup_steps": 5 if pilot else 0, "pilot_warmup_tokens": warmup_tokens,
        "pilot_warmup_seconds": warmup_elapsed if pilot else None,
        "pilot_measured_steps": session.optimizer_step - measured_start_step if pilot else None,
        "pilot_measurement_complete": (session.optimizer_step - measured_start_step >= 100) if pilot else None,
        "pilot_usable": bool(pilot and session.optimizer_step - measured_start_step >= 100 and result["status"] != "failed" and measured_elapsed > 0 and measured_tokens > 0),
        "pilot_elapsed_including_warmup_seconds": warmup_elapsed + elapsed_train if pilot else None,
        "analysis_boundary": "exploratory only; excluded from confirmatory analysis; single-seed results do not establish hypotheses or efficiency",
    }
    report.update(experiment_id=EXPERIMENT, phase="pilot" if pilot else "main", runtime_id=runtime_id(), optimizer_state_hash=state_hash(session.optimizer.state_dict()))
    _atomic_json(out / "mini_report.json", report)
    _atomic_json(out / "experiment_spec.json", {"experiment_id":EXPERIMENT,"recipe_id":EXPERIMENT,"phase":report["phase"],"seed":401,"cell":config["model"]["cell"],"config_hash":_json_hash(config),"source_revision":_json_hash(source_hashes)})
    if boundary_hook is not None:
        boundary_hook(session, "after_report")
    return report


def reserve_resources(state,r,phase,runtime,now=None):
    seconds=validate_resources(r,now)
    state=Path(state);state.mkdir(parents=True,exist_ok=True)
    p=state/'spectron_ledger.json'
    ledger=json.loads(p.read_text()) if p.exists() else {'experiment_id':EXPERIMENT,'authorization':'User: 전부 진행해. 시간 한도는 상관없어. No additional cost; retain historical charges.','sessions':[]}
    if ledger.get('experiment_id')!=EXPERIMENT:raise ValueError('ledger identity mismatch')
    if any(row['status']!='closed' for row in ledger['sessions']):raise ValueError('unresolved Spectron session; inspect, no automatic retry')
    if any(row['runtime']==runtime for row in ledger['sessions']):raise ValueError('save Output and use a fresh runtime')
    row={'runtime':runtime,'phase':phase,'reserved_seconds':seconds,'allocated_at':r['allocated_at'],'stop_at':r['stop_at'],'quota_hours_observed':r['quota_hours'],'status':'running'}
    ledger['sessions'].append(row);_atomic_json(p,ledger);return row


def execute(phase,state,bundle,resources,adamw):
    """One explicitly bounded CUDA session. Output save/stop is an operator action."""
    import fcntl
    import traceback
    state,adamw=Path(state),Path(adamw)
    if phase not in ('pilot','recovery','main'):raise ValueError('invalid phase')
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0)!='Tesla T4':raise ValueError('Tesla T4 CUDA required')
    validate_resources(resources)
    state.mkdir(parents=True,exist_ok=True)
    with (state/'operator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        row=reserve_resources(state,resources,phase,runtime_id())
        deadline=resources['stop_at']-resources['save_reserve_seconds']
        def timeout(*_):raise TimeoutError('training deadline; preserve and save Output')
        previous=signal.signal(signal.SIGALRM,timeout)
        signal.setitimer(signal.ITIMER_REAL,max(1,deadline-time.time()))
        try:
            adamw_gate(adamw)
            if phase=='main':
                estimate=main_gate(state,adamw)
                if estimate>resources['quota_hours']*3600:raise ValueError('Spectron estimate exceeds observed available quota')
            def boundary(session,stage):
                import shutil
                if shutil.disk_usage(state).free<3*1024**3:raise ValueError('less than 3 GiB free')
                if stage in ('initial','after_chunk','after_report'):retain_two(session.output_dir)
                if sum(p.stat().st_size for p in state.rglob('*') if p.is_file())>=30*1024**3:raise ValueError('30 GiB output limit reached')
                return time.time()<deadline-180 if stage=='before_chunk' else True
            for cell in ORDER:
                if time.time()>deadline-(1000 if phase=='pilot' else 300):break
                out=state/('preflight' if phase in ('pilot','recovery') else 'main')/f'LR-{cell}'
                config=ROOT/f'configs/seed401_spectron/LR-{cell}.json'
                if phase=='recovery':
                    r=json.loads((out/'mini_report.json').read_text())
                    if r.get('runtime_id')==runtime_id():raise ValueError('recovery requires a different runtime')
                    ptr,_=pointer(out)
                    proof=run_spectron(config,bundle,out,device=GPU_DEVICE,resume=True,recovery_only=True)
                    if proof['optimizer_state_hash']!=r.get('optimizer_state_hash') or proof['run_id']!=r['run_id'] or proof['optimizer_step']!=105:raise ValueError('optimizer recovery differs')
                    proof.update(experiment_id=EXPERIMENT,checkpoint_sha256=ptr['sha256'],source_runtime=r['runtime_id'],restored_runtime=runtime_id(),source_revision=_json_hash(source_identity()))
                    _atomic_json(out/'recovery_proof.json',proof)
                else:
                    prior=json.loads((out/'mini_report.json').read_text()) if (out/'mini_report.json').exists() else None
                    if prior and (phase=='pilot' or prior.get('status')=='completed'):
                        if phase=='pilot' and not prior.get('pilot_usable'):raise ValueError('failed pilot; no automatic retry')
                        pointer(out)
                        if phase=='main':verify_spectron_final(out,prior)
                        continue
                    report=run_spectron(config,bundle,out,device=GPU_DEVICE,resume=(out/'checkpoints/latest.json').exists(),pilot=phase=='pilot',max_steps=2048,max_seconds=deadline-time.time()-180,boundary_hook=boundary)
                    if report['status']=='failed' or phase=='pilot' and not report['pilot_usable']:raise ValueError('Spectron run failed or incomplete pilot')
                    if phase=='main' and report['status']=='completed':verify_spectron_final(out,report)
                    if phase=='main' and report['status']!='completed':break
            row['status']='closed'
        except BaseException:
            row['status']='failed';(state/'failure.log').write_text(traceback.format_exc());raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
            ledger=json.loads((state/'spectron_ledger.json').read_text());ledger['sessions'][-1].update(row)
            _atomic_json(state/'spectron_ledger.json',ledger)
            print('SPECTRON_SNAPSHOT',seal(state),flush=True)
            print('Quick Save + Save output; confirm Successful, then stop session. Retain full reservation.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('pilot','recovery','main'))
    p.add_argument('--state',required=True);p.add_argument('--bundle',required=True)
    p.add_argument('--resources',required=True);p.add_argument('--adamw',required=True)
    a=p.parse_args();execute(a.phase,a.state,a.bundle,json.loads(Path(a.resources).read_text()),a.adamw)



def restore_state(source,destination,expected_snapshot):
    import shutil
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if _sha256_file(source/'snapshot.json')!=expected_snapshot:raise ValueError('wrong snapshot identity')
    if destination.exists() and any(destination.iterdir()):raise ValueError('restore requires empty target')
    spec=json.loads((source/'snapshot.json').read_text())
    if spec.get('schema')!='seed401-local-v1' or 'spectron_ledger.json' not in spec.get('files',{}):raise ValueError('invalid snapshot')
    for name,r in spec['files'].items():
        p=source/name
        if not p.resolve().is_relative_to(source) or p.is_symlink() or p.stat().st_size!=r['bytes'] or _sha256_file(p)!=r['sha256']:raise ValueError('snapshot integrity mismatch')
    ledger=json.loads((source/'spectron_ledger.json').read_text())
    if ledger.get('experiment_id')!=EXPERIMENT or any(r['status']!='closed' for r in ledger['sessions']):raise ValueError('unresolved saved Spectron session')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(destination.parent).free<sum(r['bytes'] for r in spec['files'].values())+3*1024**3:raise ValueError('insufficient restore space')
    destination.mkdir(exist_ok=True)
    for name in [*spec['files'],'snapshot.json']:
        out=destination/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,out)

def verify_spectron_final(out,report):
    out=Path(out)
    if not (report.get('experiment_id')==EXPERIMENT and report.get('training',{}).get('recipe_id')==EXPERIMENT and report.get('status')=='completed' and report.get('optimizer_step')==2048 and report.get('trained_target_tokens')==8388608):raise ValueError('Spectron final identity or counters invalid')
    if not (out/'metrics.csv').is_file():raise ValueError('missing final metrics')
    with (out/'metrics.csv').open() as f:rows=list(csv.DictReader(f))
    if len(rows)!=8 or sorted(int(r['trained_target_tokens']) for r in rows)!=list(range(1048576,8388609,1048576)):raise ValueError('eight fixed dev evaluations required')
    if any(r['split']!='dev' or r['run_id']!=report['run_id'] or int(r['eval_target_tokens'])!=65536 or not math.isfinite(float(r['loss_nats'])) for r in rows):raise ValueError('invalid final evaluation')
    ptr,_=pointer(out)
    final=next(r for r in rows if int(r['trained_target_tokens'])==8388608)
    if final['checkpoint_id']!=ptr['checkpoint_id'] or report.get('checkpoint_id')!=ptr['checkpoint_id']:raise ValueError('final checkpoint differs')
    if float(final['loss_nats'])!=report.get('loss_nats'):raise ValueError('final CE differs')
    if any(int(r['seed'])!=401 or r['cell']!=report['model']['cell'] or r['config_id']!=report['model']['config_id'] for r in rows):raise ValueError('evaluation model identity differs')

def check_resume_run(rows,run_id):
    if not rows or any(r.get('run_id')!=run_id for r in rows):raise ValueError('checkpoint and manifest run identity mismatch')

def evaluation_due(out,tokens,run_id,checkpoint_id):
    if not tokens or tokens % EVAL_INTERVAL:return False
    p=Path(out)/'metrics.csv'
    if not p.exists():return True
    with p.open() as f:
        rows=list(csv.DictReader(f))
    return not any(r['run_id']==run_id and r['split']=='dev' and r['checkpoint_id']==checkpoint_id and int(r['trained_target_tokens'])==tokens for r in rows)

if __name__=='__main__':main()
