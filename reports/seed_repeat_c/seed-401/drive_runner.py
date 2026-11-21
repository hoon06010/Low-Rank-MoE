"""Explicit seed401 CUDA runner with direct Drive immutable control history.

Drive has no compare-and-swap lock: this runner REQUIRES one operator/account,
plus local flock and create/list duplicate exclusion. It cannot exclude a hostile
or concurrently starting remote process; no atomic remote-lock claim is made.
No automatic retries, stale-lock takeover, failure reset, or next-cell execution.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import uuid

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE)]
import drive_transport as transport
import checkpoint_retention as retention
import checkpoint_state_compare as comparator
from scripts import run_seed_repeat_c as baseline
from scripts import seed_repeat_policy as policy
from scripts.overnight_store import _atomic_json, _checkpoint, deadline, digest

CAP=30*1024**3
PREFIX='seed401-ledger-'

# This explicit review route is limited to the independently inspected failed HL.
REVIEW_FAILURE_ID='b91efcbd13d2429aabf3e27fe0344fff'
REVIEW_FAILED_RUN_ID='be7ee742-8a0b-4cde-9730-fb62473d5b05'
REVIEW_ORIGINAL_HEAD={'id':'REDACTED_PRIVATE_ID',
    'sha256':'871b0b0aa1bd7ffbc1d81995aba4ffde820abffc0f90065df8f3dd179db4a8e5'}
REVIEW_ORIGINAL_SEQ=35
REVIEW_ORIGINAL_ADAPTER='c411516e442143a2014a341c7a25539948c213541cc40ac0b346867f4549aa1f'

def prepare_failure_review(ledger,gate,spec,head,seq):
    """Copy-only explicit HUMAN review; never reset failed status or charges."""
    request=gate.get('reviewed_failure',{})
    if (request.get('approved') is not True or not isinstance(request.get('approval_reference'),str)
            or not request['approval_reference'].strip() or gate.get('operator_account')!='example-operator'
            or request.get('invocation_id')!=REVIEW_FAILURE_ID or request.get('failed_run_id')!=REVIEW_FAILED_RUN_ID):
        raise ValueError('exact new human failure review approval required')
    if (head!=REVIEW_ORIGINAL_HEAD or request.get('ledger_head')!=head or seq!=REVIEW_ORIGINAL_SEQ
            or request.get('ledger_seq',seq)!=seq):
        raise ValueError('review requires exact inspected ledger head and sequence')
    expected=baseline.build_spec(401,'HL','pilot')
    if spec!=expected or ledger.get('reviewed_failures') or REVIEW_FAILED_RUN_ID in ledger.get('archived_runs',{}):
        raise ValueError('review is limited to one fresh failed-HL pilot retry')
    unknown=[i for i in ledger['invocations'] if i.get('status')!='verified']
    if len(unknown)!=1 or unknown[0].get('id')!=REVIEW_FAILURE_ID or unknown[0].get('status')!='failed':
        raise ValueError('other or foreign failed/interrupted invocation prevents review')
    invocation=unknown[0]; failed=ledger['runs'].get(expected['output'],{})
    if (invocation.get('spec')!=expected or invocation.get('adapter_revision')!=REVIEW_ORIGINAL_ADAPTER
            or failed.get('spec')!=expected or failed.get('status')!='failed'
            or failed.get('invocation_id')!=REVIEW_FAILURE_ID or failed.get('run_id')!=REVIEW_FAILED_RUN_ID
            or failed.get('identity')!=dict(expected,run_id=REVIEW_FAILED_RUN_ID)
            or failed.get('adapter_revision')!=REVIEW_ORIGINAL_ADAPTER
            or failed.get('adapter_file_hashes')!=invocation.get('adapter_file_hashes')):
        raise ValueError('failed HL exact source/run/adapter identity mismatch')
    if any(row.get('status')=='failed' and key!=expected['output'] for key,row in ledger['runs'].items()):
        raise ValueError('foreign failed run prevents review')
    result=copy.deepcopy(ledger)
    result.setdefault('archived_runs',{})[REVIEW_FAILED_RUN_ID]=copy.deepcopy(failed)
    del result['runs'][expected['output']]
    result['reviewed_failures']={REVIEW_FAILURE_ID:{'kind':'human-reviewed-one-fresh-HL-retry',
        'request':copy.deepcopy(request),'original_head':copy.deepcopy(head),'original_seq':seq,
        'review_seq':seq+1,'original_invocation_sha256':policy.json_hash(invocation),
        'original_run_sha256':policy.json_hash(failed),'failed_run_id':REVIEW_FAILED_RUN_ID,
        'retry_output':expected['output'],'retry_invocation_id':None,'retry_status':'pending'}}
    return result

def validate_failed_history(ledger):
    """Exempt only the exact durably reviewed failure; all other unknowns block."""
    for invocation in ledger['invocations']:
        if invocation.get('status')=='verified': continue
        review=ledger.get('reviewed_failures',{}).get(invocation.get('id'),{})
        archived=ledger.get('archived_runs',{}).get(REVIEW_FAILED_RUN_ID,{})
        request=review.get('request',{})
        if (invocation.get('id')!=REVIEW_FAILURE_ID or invocation.get('status')!='failed'
                or review.get('kind')!='human-reviewed-one-fresh-HL-retry' or request.get('approved') is not True
                or request.get('invocation_id')!=REVIEW_FAILURE_ID or request.get('failed_run_id')!=REVIEW_FAILED_RUN_ID
                or request.get('ledger_head')!=REVIEW_ORIGINAL_HEAD or not request.get('approval_reference')
                or review.get('original_head')!=REVIEW_ORIGINAL_HEAD or review.get('original_seq')!=REVIEW_ORIGINAL_SEQ
                or review.get('review_seq')!=REVIEW_ORIGINAL_SEQ+1
                or review.get('original_invocation_sha256')!=policy.json_hash(invocation)
                or review.get('original_run_sha256')!=policy.json_hash(archived)
                or archived.get('status')!='failed' or archived.get('run_id')!=REVIEW_FAILED_RUN_ID
                or archived.get('spec')!=baseline.build_spec(401,'HL','pilot')):
            raise ValueError('failed/interrupted reservation requires operator review with exact durable reviewed archive')
        retry_id=review.get('retry_invocation_id')
        if retry_id:
            retry=[i for i in ledger['invocations'] if i.get('id')==retry_id]
            current=ledger['runs'].get(review.get('retry_output'),{})
            expected=baseline.build_spec(401,'HL','pilot')
            if (len(retry)!=1 or retry[0].get('status')!='verified' or review.get('retry_status')!='verified'
                    or retry[0].get('spec')!=expected or retry[0].get('reviewed_retry_of')!=REVIEW_FAILURE_ID
                    or not review.get('retry_run_id') or review['retry_run_id']==REVIEW_FAILED_RUN_ID
                    or current.get('status')!='verified' or current.get('spec')!=expected
                    or current.get('identity')!=dict(expected,run_id=review['retry_run_id'])):
                raise ValueError('reviewed retry failed/interrupted or ambiguous; new review required')

def validate_pending_retry(ledger,gate,spec,seq,*,mode,resume,run_id):
    review=ledger.get('reviewed_failures',{}).get(REVIEW_FAILURE_ID,{})
    if (not review or review.get('retry_invocation_id') or review.get('retry_status')!='pending'):
        raise ValueError('one reviewed retry is already consumed or absent')
    if (mode!='run' or resume or run_id is not None or spec!=baseline.build_spec(401,'HL','pilot')):
        raise ValueError('review permits only ONE fresh HL run, never failed-run resume')
    if seq!=review.get('review_seq'):
        raise ValueError('ambiguous pending review ledger sequence')
    if gate.get('reviewed_failure')!=review.get('request'):
        raise ValueError('fresh retry must bind exact human approval review')
    if spec['output'] in ledger['runs']:
        raise ValueError('pending retry canonical output is not free')

def preserve_failed_local_run(ledger,out,control_dir):
    """Move only exact owned failed artifacts; never delete/checkpoint-prune them."""
    failed=ledger['archived_runs'][REVIEW_FAILED_RUN_ID]
    archive=Path(control_dir)/'archived-failed-runs'/REVIEW_FAILED_RUN_ID
    def exact_manifest(path):
        if path.is_symlink() or any(p.is_symlink() for p in path.rglob('*')):
            raise ValueError('failed local artifact symlink rejected')
        rows=[json.loads(line) for line in (path/'run_manifest.jsonl').read_text().splitlines() if line.strip()]
        if len(rows)!=1 or rows[0].get('run_id')!=REVIEW_FAILED_RUN_ID:
            raise ValueError('local failed output identity is ambiguous')
    if Path(out).exists():
        exact_manifest(Path(out))
        if archive.exists(): raise ValueError('both canonical and archived failed outputs exist')
        archive.parent.mkdir(parents=True,exist_ok=True); Path(out).rename(archive)
    elif archive.exists(): exact_manifest(archive)
    # A fresh runtime may have no local failed artifacts; original remote archive
    # IDs and the full failed row remain preserved in the immutable ledger.
    registry=Path(control_dir)/'enrollments'
    enrollment=retention.registration_path(registry,failed['spec']['output'])
    if enrollment.exists():
        record=retention.load(enrollment)
        if record.get('seed')!=401 or record.get('output')!=failed['spec']['output'] or record.get('root')!=str(ROOT.resolve()):
            raise ValueError('failed enrollment identity mismatch')
        destination=registry/'archived'/REVIEW_FAILED_RUN_ID
        if destination.exists(): raise ValueError('ambiguous duplicate failed enrollment')
        destination.parent.mkdir(parents=True,exist_ok=True); enrollment.rename(destination)

def adapter_hashes():
    directory=Path(__file__).resolve().parent
    return {name:digest(directory/name) for name in ('drive_runner.py','continuation_worker.py',
        'drive_transport.py','checkpoint_retention.py','checkpoint_state_compare.py')}

def adapter_revision():
    return policy.json_hash(adapter_hashes())

class OperationalDrive(transport.DriveAPI):
    def account_id(self):
        value=self._json(self._request(transport.API+'/about?fields=user(permissionId)'))
        return value.get('user',{}).get('permissionId')
    def list_files(self,folder,prefix):
        transport.safe_id(folder)
        query=f"'{folder}' in parents and trashed = false"
        if prefix: query+=f" and name contains '{prefix}'"
        rows=[]; token=None
        while True:
            params={'q':query,'fields':'nextPageToken,files(id,name,size,md5Checksum,mimeType)','pageSize':1000}
            if token: params['pageToken']=token
            page=self._json(self._request(transport.API+'/files?'+urllib.parse.urlencode(params)))
            rows.extend(r for r in page.get('files',[]) if r['name'].startswith(prefix))
            token=page.get('nextPageToken')
            if not token: return rows
    def delete_owned(self,fid):
        with self._request(transport.API+'/files/'+transport.safe_id(fid),'DELETE'): pass

def empty_ledger():
    return {'sessions':{},'preflight_seconds':0,'main_seconds':0,'runs':{},'invocations':[]}

class Control:
    def __init__(self,backend,folder,operator,cache):
        self.backend=backend; self.folder=transport.safe_id(folder); self.operator=operator
        self.cache=Path(cache); self.cache.mkdir(parents=True,exist_ok=True)
        self.head=None; self.seq=0; self.lock_id=None; self.preserve_lock=False; self.verified_rows=[]; self.raw=None
    @contextlib.contextmanager
    def lock(self):
        # No lease expiry: ambiguity/crash requires human review of the lock/history.
        if self.backend.list_files(self.folder,'seed401-lock-'):
            raise ValueError('remote control lock exists; operator review required')
        with tempfile.TemporaryDirectory(dir=self.cache) as tmp:
            path=Path(tmp)/('seed401-lock-'+uuid.uuid4().hex+'.json')
            transport.write_json(path,{'operator':self.operator,'pid':os.getpid(),'created_at':time.time()})
            self.lock_id=transport.upload_verified(self.backend,self.folder,path,Path(tmp))
        rows=self.backend.list_files(self.folder,'seed401-lock-')
        if len(rows)!=1 or rows[0]['id']!=self.lock_id:
            # Own duplicate can be removed; foreign lock never touched.
            self.backend.delete_owned(self.lock_id); self.lock_id=None
            raise ValueError('duplicate remote control lock; operator review required')
        try:
            yield self
        finally:
            if self.lock_id and not self.preserve_lock:
                self.backend.delete_owned(self.lock_id); self.lock_id=None
    def check_lock(self):
        rows=self.backend.list_files(self.folder,'seed401-lock-')
        if len(rows)!=1 or rows[0]['id']!=self.lock_id:
            raise ValueError('exclusive control lock changed')
    def cache_history(self):
        if self.head:
            _atomic_json(self.cache/'verified-history.json',{'rows':self.verified_rows,'head':self.head,
                'raw':self.raw,'operator':self.operator})
    def load(self):
        rows=self.backend.list_files(self.folder,PREFIX)
        ordered=sorted(rows,key=lambda r:r['name'])
        slim=[{k:r.get(k) for k in ('id','name','md5Checksum')} for r in ordered]
        previous=None; result=empty_ledger(); start=0
        cached_path=self.cache/'verified-history.json'
        if cached_path.exists():
            cached=json.loads(cached_path.read_text()); raw=cached['raw']
            if (cached.get('operator')!=self.operator or hashlib.sha256(raw.encode()).hexdigest()!=cached['head']['sha256']
                    or slim[:len(cached['rows'])]!=cached['rows']):
                raise ValueError('verified remote control cache/history changed')
            start=len(cached['rows']); previous=cached['head']; result=json.loads(raw)['ledger']
            if not start or previous['id']!=cached['rows'][-1]['id']:
                raise ValueError('verified control cache head mismatch')
            self.raw=raw
        for number,row in enumerate(ordered[start:],start+1):
            if not row['name'].startswith(PREFIX+f'{number:08d}-'):
                raise ValueError('ambiguous or incomplete remote control history')
            with tempfile.TemporaryDirectory(dir=self.cache) as tmp:
                path=Path(tmp)/'ledger.json'
                value=transport.download_json(self.backend,row['id'],path)
                hashed=digest(path); raw=path.read_text()
            if (value.get('schema')!=1 or value.get('seq')!=number or value.get('previous')!=previous
                    or value.get('operator')!=self.operator):
                raise ValueError('remote ledger chain/operator mismatch')
            result=value['ledger']; previous={'id':row['id'],'sha256':hashed}; self.raw=raw
        self.head=previous; self.seq=len(ordered); self.verified_rows=slim
        self.cache_history()
        return copy.deepcopy(result)
    def commit(self,ledger):
        self.check_lock()
        # Re-list history before each commit, reject a changed/forked head.
        before=self.head; sequence=self.seq; self.load()
        if before!=self.head or sequence!=self.seq:
            raise ValueError('remote control history changed')
        with tempfile.TemporaryDirectory(dir=self.cache) as tmp:
            path=Path(tmp)/(PREFIX+f'{self.seq+1:08d}-'+uuid.uuid4().hex+'.json')
            transport.write_json(path,{'schema':1,'seq':self.seq+1,'previous':self.head,
                                     'operator':self.operator,'ledger':ledger})
            fid=transport.upload_verified(self.backend,self.folder,path,Path(tmp))
            hashed=digest(path); self.raw=path.read_text(); name=path.name
        self.head={'id':fid,'sha256':hashed}; self.seq+=1
        md5=hashlib.md5(self.raw.encode(),usedforsecurity=False).hexdigest()
        self.verified_rows.append({'id':fid,'name':name,'md5Checksum':md5})
        self.cache_history()
        self.check_lock()

def positive(value,name):
    if type(value) not in (float,int) or not math.isfinite(value) or value<=0:
        raise ValueError('invalid '+name)
    return value

def preflight_limit(gate):
    limit=gate.get('preflight_limit_seconds',6300)
    if type(limit) is not int or limit not in (6300,9900):
        raise ValueError('unsupported preflight time limit')
    if limit>6300 and (gate.get('preflight_time_extension_approved') is not True
            or not isinstance(gate.get('preflight_time_extension_approval_reference'),str)
            or not gate['preflight_time_extension_approval_reference'].strip()):
        raise ValueError('separate explicit preflight time extension approval required')
    return limit

def main_limit(gate):
    limit=gate.get('main_limit_seconds',28800)
    if type(limit) is not int or limit not in (28800,57600,86400):
        raise ValueError('unsupported main time limit')
    if limit>28800 and (gate.get('time_extension_approved') is not True
            or not isinstance(gate.get('time_extension_approval_reference'),str)
            or not gate['time_extension_approval_reference'].strip()):
        raise ValueError('separate explicit time extension approval required')
    return limit

def settle_successful_main(ledger,gate,spec,invocation_id,verified_commit,*,now=None):
    """Copy-only settlement after a durable verified MAIN checkpoint/ledger commit.

    The caller retains the full reserved ledger until this copy is independently
    committed. Unknown/failed writes never authorize further work or a refund.
    """
    now=time.time() if now is None else now
    if spec.get('seed')!=401 or spec.get('phase')!='main': raise ValueError('only main success can settle')
    if (not isinstance(verified_commit,dict) or not verified_commit.get('id')
            or not comparator.HEX.fullmatch(str(verified_commit.get('sha256','')))):
        raise ValueError('verified immutable ledger commit required')
    transport.safe_id(verified_commit['id'])
    matches=[i for i in ledger['invocations'] if i.get('id')==invocation_id]
    if len(matches)!=1 or matches[0].get('status')!='verified' or matches[0].get('spec')!=spec:
        raise ValueError('newly verified main invocation required')
    invocation=matches[0]
    if invocation.get('settlement') or invocation.get('session_id')!=gate['session_id']:
        raise ValueError('main invocation settlement identity already settled or mismatched')
    state=ledger['runs'].get(spec['output'],{}); evidence=state.get('last_verified_snapshot',{})
    identity=state.get('identity',{})
    if (state.get('status')!='verified' or state.get('spec')!=spec or not identity.get('run_id')
            or identity!=dict(spec,run_id=identity['run_id']) or evidence.get('identity')!=identity
            or evidence.get('status')!='remote-publish-verified' or evidence.get('generation_restore_verified') is not True
            or not evidence.get('checkpoint_id') or not comparator.HEX.fullmatch(str(evidence.get('checkpoint_sha256','')))):
        raise ValueError('verified exact checkpoint identity prerequisite missing')
    session=ledger['sessions'][gate['session_id']]
    floor=invocation.get('reservation_floor_before'); reserved=invocation.get('reserved_session_charge')
    if (session.get('stopped_at') is not None or session['started_at']!=gate['session_started_at']
            or session.get('last_phase')!='main' or reserved!=session['charged_seconds']
            or type(floor) not in (int,float) or not math.isfinite(floor) or floor<0):
        raise ValueError('main reservation accounting identity mismatch')
    elapsed=now-session['started_at']
    closing_reserve=positive(gate['reserve_seconds'],'closing reserve')
    charge=max(floor,elapsed+closing_reserve)
    if elapsed<0 or charge>5400: raise ValueError('session budget exhausted during settlement')
    refund=reserved-charge
    if refund>gate['invocation_seconds']+closing_reserve or ledger['main_seconds']-refund<0:
        raise ValueError('main settlement exceeds its own reservation')
    result=copy.deepcopy(ledger)
    result['sessions'][gate['session_id']]['charged_seconds']=charge
    result['main_seconds']-=refund
    if (result['preflight_seconds']>preflight_limit(gate) or result['main_seconds']>main_limit(gate)
            or result['preflight_seconds']+result['main_seconds']>gate.get('quota_seconds',float('inf'))):
        raise ValueError('resource budget exhausted during settlement')
    item=next(i for i in result['invocations'] if i['id']==invocation_id)
    item['settlement']={'status':'verified-success-only','verified_checkpoint_commit':verified_commit,
        'reserved_session_charge':reserved,'settled_session_charge':charge,'unused_seconds':max(0,refund),
        'observed_elapsed_seconds':elapsed,'closing_reserve_seconds':closing_reserve,
        'checkpoint_id':evidence['checkpoint_id'],'checkpoint_sha256':evidence['checkpoint_sha256']}
    return result

def validate_gate(gate,spec,*,now=None):
    now=time.time() if now is None else now
    main_limit(gate)
    preflight_limit(gate)
    if gate.get('approved') is not True or not gate.get('approval_reference') or spec['seed']!=401 or spec.get('phase') not in ('recovery','pilot','main'):
        raise ValueError('explicit seed401 approval required')
    if gate.get('provider')!='Kaggle' or gate.get('operator_account')!='example-operator' or gate.get('precision')!='float32':
        raise ValueError('approved Kaggle account/float32 resource scope mismatch')
    if not isinstance(gate.get('drive_account_id'),str) or not gate['drive_account_id']:
        raise ValueError('expected Drive account ID required')
    if gate.get('adapter_revision')!=adapter_revision():
        raise ValueError('operational adapter source revision mismatch')
    if gate.get('source_revision')!=spec['source_revision']:
        raise ValueError('frozen source identity mismatch')
    if gate.get('cost_limit')!=0 or gate.get('cost_per_hour')!=0:
        raise ValueError('only explicitly free resources authorized')
    if gate.get('single_operator_assertion') is not True:
        raise ValueError('single operator/account assertion required; Drive lock is not atomic')
    for key in ('operator_account','provider','device_name','torch_version','cuda_version','python_version','session_id'):
        if not isinstance(gate.get(key),str) or not gate[key].strip(): raise ValueError('missing '+key)
    for key in ('folder_id','control_folder_id'): transport.safe_id(gate.get(key))
    for key in ('session_started_at','invocation_seconds','reserve_seconds','quota_seconds','min_free_bytes','max_output_bytes'):
        positive(gate.get(key),key)
    if gate['max_output_bytes']>CAP: raise ValueError('output cap exceeds approved 30GiB')
    elapsed=now-gate['session_started_at']
    if elapsed<0 or elapsed+gate['invocation_seconds']+gate['reserve_seconds']>5400:
        raise ValueError('session budget exhausted')
    storage=gate.get('storage_evidence',{})
    if storage.get('status')!='fresh-cache-restore-verified' or storage.get('mature_transfer_verified') is not True:
        raise ValueError('fresh storage restore and mature transfer evidence required')
    if spec['phase'] in ('pilot','main'):
        validate_recovery(gate.get('recovery_evidence',{}),spec)
    if spec['phase']=='main': validate_pilots(gate.get('pilot_evidence',[]),spec)
    return gate

def validate_pilots(pilots,spec):
    if len(pilots)!=4 or {p.get('cell') for p in pilots}!=set(policy.CELLS):
        raise ValueError('four independent pilots required')
    if len({p.get('run_id') for p in pilots})!=4 or any(not p.get('run_id') for p in pilots):
        raise ValueError('four distinct pilot run identities required')
    for p in pilots:
        expected=baseline.build_spec(401,p['cell'],'pilot'); report=p.get('report',{})
        model=json.loads((ROOT/expected['config']).read_text())['model']
        if (p.get('seed')!=401 or p.get('source_revision')!=spec['source_revision']
                or p.get('data_manifest_hash')!=spec.get('data_manifest_hash')
                or p.get('output')!=expected['output'] or p.get('remote_verified') is not True
                or report.get('run_id')!=p['run_id'] or report.get('model',{}).get('cell')!=p['cell']
                or report.get('model')!=model
                or report.get('source_file_hashes')!=expected['source_file_hashes']
                or report.get('data_manifest_hash')!=expected['data_manifest_hash']
                or report.get('eval_revision')!=expected['eval_revision']
                or report.get('pilot_usable') is not True or report.get('pilot_measurement_complete') is not True
                or report.get('pilot_warmup_steps')!=5 or report.get('pilot_measured_steps')!=100
                or report.get('optimizer_step')!=105 or report.get('trained_target_tokens')!=430080
                or report.get('device')!='cuda' or report.get('precision')!='float32'
                or not math.isfinite(float(report.get('loss_nats',float('nan'))))):
            raise ValueError('pilot cell/source/data/eval identity or semantic evidence mismatch')

def validate_committed_preflight(ledger,gate,spec):
    if spec['phase'] not in ('pilot','main'): return
    recovery=gate['recovery_evidence']
    expected=baseline.build_spec(401,recovery['cell'],'recovery')
    row=ledger['runs'].get(expected['output'],{})
    validate_resume(row,expected,recovery['run_id'])
    history=row.get('runtime_restore_history',[])
    continuation=row.get('continuation',{})
    if (continuation.get('comparison')!=recovery.get('continuation_comparison')
            or continuation.get('left_sha256')!=recovery.get('continuation_left_sha256')
            or continuation.get('right_sha256')!=recovery.get('continuation_right_sha256')
            or len(continuation.get('proofs',[]))!=2):
        raise ValueError('continuation must match committed reserved branch proof')
    if recovery['step0_restore'] not in history or recovery['step1_restore'] not in history:
        raise ValueError('recovery comparator results must match committed runtime restore history')
    if spec['phase']!='main': return
    for pilot in gate['pilot_evidence']:
        expected=baseline.build_spec(401,pilot['cell'],'pilot')
        row=ledger['runs'].get(expected['output'],{})
        validate_resume(row,expected,pilot['run_id'])
        if (row.get('report')!=pilot['report'] or row.get('last_verified_snapshot',{}).get('identity')!=row['identity']
                or not row.get('generations') or any(g.get('generation_restore_verified') is not True for g in row['generations'])):
            raise ValueError('pilot must match committed verified control ledger/report')

def validate_recovery(record,spec):
    if record.get('cell') not in policy.CELLS or not record.get('run_id'):
        raise ValueError('CUDA recovery run identity required')
    expected=baseline.build_spec(401,record['cell'],'recovery')
    if record.get('output')!=expected['output'] or record.get('eval_revision')!=expected['eval_revision']:
        raise ValueError('CUDA recovery output/eval identity mismatch')
    for name,step,mode in (('step0_restore',0,'restored'),('step1_restore',1,'restored'),('continuation_comparison',2,'continuation')):
        result=record.get(name,{})
        excluded=comparator.UUID_FIELDS if mode=='continuation' else frozenset()
        fields=[field for field in comparator.FIELDS if field not in excluded]
        if (result.get('status')!='exact-state-verified' or result.get('mode')!=mode
                or result.get('optimizer_step')!=step or result.get('seed')!=401 or result.get('cell')!=record['cell']
                or result.get('checked_fields')!=fields or result.get('excluded_fields')!=sorted(excluded)):
            raise ValueError('CUDA recovery exact comparator field evidence required')
        if mode=='restored' and (result.get('run_id')!=record['run_id'] or result.get('execution_device')!='cuda'):
            raise ValueError('CUDA restored runtime identity mismatch')
    continuation=record['continuation_comparison']
    if any(continuation.get(k)!=record.get(k) for k in ('run_id','execution_device','source_revision','data_manifest_hash','eval_revision')):
        raise ValueError('CUDA continuation exact run/device/source identity required')
    if record['step1_restore'].get('fresh_runtime') is not True:
        raise ValueError('CUDA step1 fresh runtime restore required')
    for name in ('continuation_left_sha256','continuation_right_sha256'):
        if not comparator.HEX.fullmatch(str(record.get(name,''))):
            raise ValueError('CUDA continuation checkpoint hashes required')
    if (record.get('status')!='cuda-recovery-verified' or record.get('seed')!=401
            or record.get('source_revision')!=spec['source_revision']
            or record.get('data_manifest_hash')!=spec.get('data_manifest_hash')
            or record.get('fresh_runtime') is not True or not record.get('origin_runtime_id')
            or not record.get('restore_runtime_id') or record.get('origin_runtime_id')==record.get('restore_runtime_id')
            or record.get('step0_restore_exact') is not True or record.get('step1_restore_exact') is not True
            or record.get('step2_continuation_exact') is not True
            or record.get('execution_device')!='cuda' or record.get('precision')!='float32'
            or record.get('step1_tokens')!=4096 or record.get('step2_tokens')!=8192
            or record.get('step1_cursor')!=8 or record.get('step2_cursor')!=16):
        raise ValueError('CUDA recovery semantic evidence required')

def reserve(ledger,gate,phase,*,now=None):
    now=time.time() if now is None else now
    result=copy.deepcopy(ledger)
    # A runtime switch requires observed stop evidence; its unreserved idle tail
    # is charged to that session's last phase before any new GPU reservation.
    for previous_id,previous in result['sessions'].items():
        if previous_id==gate['session_id'] or previous.get('stopped_at') is not None: continue
        proof=gate.get('session_stop_evidence',{}).get(previous_id,{})
        stopped=proof.get('stopped_at')
        if (proof.get('started_at')!=previous['started_at'] or type(stopped) not in (int,float)
                or not math.isfinite(stopped) or stopped<previous['started_at'] or stopped>now):
            raise ValueError('previous session observed stop evidence required')
        elapsed_previous=stopped-previous['started_at']
        if elapsed_previous>5400: raise ValueError('previous session budget exceeded')
        tail=max(0,elapsed_previous-previous['charged_seconds'])
        previous_bucket='main_seconds' if previous['last_phase']=='main' else 'preflight_seconds'
        result[previous_bucket]+=tail
        previous.update(charged_seconds=max(previous['charged_seconds'],elapsed_previous),stopped_at=stopped)
    session=result['sessions'].get(gate['session_id'],{'started_at':gate['session_started_at'],'charged_seconds':0})
    if session['started_at']!=gate['session_started_at']: raise ValueError('session start identity changed')
    if session.get('stopped_at') is not None: raise ValueError('closed runtime session cannot be reopened')
    elapsed=now-session['started_at']
    charge=max(session['charged_seconds'],elapsed)+gate['invocation_seconds']+gate['reserve_seconds']
    if elapsed<0 or charge>5400: raise ValueError('session budget exhausted')
    idle_tail=max(0,elapsed-session['charged_seconds'])
    previous_bucket='main_seconds' if session.get('last_phase',phase)=='main' else 'preflight_seconds'
    result[previous_bucket]+=idle_tail
    delta=charge-session['charged_seconds']-idle_tail
    bucket='main_seconds' if phase=='main' else 'preflight_seconds'
    result[bucket]+=delta
    if result['preflight_seconds']>preflight_limit(gate) or result['main_seconds']>main_limit(gate) or result['preflight_seconds']+result['main_seconds']>gate.get('quota_seconds',float('inf')):
        raise ValueError('resource budget exhausted')
    result['sessions'][gate['session_id']]={'started_at':session['started_at'],'charged_seconds':charge,'last_phase':phase}
    return result

def validate_resume(row,spec,run_id):
    if row.get('status')!='verified': raise ValueError('resume requires verified stopped state')
    if row.get('spec')!=spec or row.get('identity')!=dict(spec,run_id=run_id):
        raise ValueError('resume exact run identity mismatch')

def verified_boundary(state,publish,step):
    try:
        evidence=publish(step)
    except BaseException:
        state['status']='failed'; raise
    state['last_verified_snapshot']=evidence
    return evidence

def cleanup_generations(rows,restore_verified,delete):
    if not restore_verified: return rows
    # Report refreshes archive the SAME checkpoint; keep two distinct states.
    # Prefer the newest complete report for each state, then its newest archive.
    selected={}; last_position={}; checkpoint_hashes={}
    for index,row in enumerate(rows):
        checkpoint_id=row.get('checkpoint_id'); checkpoint_sha=row.get('checkpoint_sha256')
        if not isinstance(checkpoint_id,str) or not checkpoint_id or not comparator.HEX.fullmatch(str(checkpoint_sha)):
            raise ValueError('invalid checkpoint identity/hash for retention')
        if checkpoint_id in checkpoint_hashes and checkpoint_hashes[checkpoint_id]!=checkpoint_sha:
            raise ValueError('same checkpoint identity has changed hash')
        checkpoint_hashes[checkpoint_id]=checkpoint_sha
        identity=(checkpoint_id,checkpoint_sha); last_position[identity]=index
        previous=selected.get(identity)
        if previous is None or row.get('report_complete') is True or previous.get('report_complete') is not True:
            selected[identity]=row
        for key in ('archive_id','receipt_id','pointer_id'): transport.safe_id(row[key])
    identities=sorted(selected,key=lambda key:last_position[key])[-2:]
    keep=[selected[key] for key in identities]
    keep_objects={id(row) for row in keep}
    protected_ids={row[key] for row in keep for key in ('archive_id','receipt_id','pointer_id')}
    # IDs supplied exclusively by this run's committed ledger, never name search.
    for row in rows:
        if id(row) in keep_objects: continue
        for key in ('archive_id','receipt_id','pointer_id'):
            if row[key] not in protected_ids: delete(row[key])
    return keep

def hardware_gate(gate):
    import torch
    if (not torch.cuda.is_available() or torch.cuda.get_device_name(0)!=gate['device_name']
            or str(torch.__version__)!=gate['torch_version'] or torch.version.cuda!=gate['cuda_version']
            or platform.python_version()!=gate['python_version']):
        raise ValueError('actual CUDA hardware/library gate mismatch')

def make_training_session(spec,out):
    """Instantiate the exact frozen audited trainer; caller restores immediately."""
    import torch
    from src.train import TrainingSession
    from src.model import MoETransformer
    from src.data import load_audited_bundle
    config=json.loads((ROOT/spec['config']).read_text()); audited=load_audited_bundle(ROOT/policy.BUNDLE)
    identity={'protocol_id':config['protocol_id'],'source_revision':spec['source_revision'],
              'data_manifest_hash':spec['data_manifest_hash'],'eval_revision':spec['eval_revision'],
              'tokenizer_revision':audited['spec']['manifest']['tokenizer']['revision']['value'],'source_kind':'measured'}
    return TrainingSession(MoETransformer(config['model']),config['model'],config['training'],
                            audited['batches']['train'],identity,out,device='cuda',audit_bundle=ROOT/policy.BUNDLE)

def restored_session(spec,out):
    """Capture actual restored runtime state BEFORE another optimizer step."""
    import torch
    session=make_training_session(spec,out)
    session.restore_latest()
    pointer,checkpoint=_checkpoint(out)
    payload=session._checkpoint_payload(pointer['checkpoint_id'])
    # Capture payload first, then load saved bytes. No RNG-consuming work precedes capture.
    captured=out/'restored-runtime-state.pt'; torch.save(payload,captured)
    expected=dict(spec,run_id=session.run_id)
    result=comparator.compare_payloads(comparator.verified_load(checkpoint,pointer['sha256']),payload,expected)
    result.update(run_id=session.run_id,execution_device='cuda',capture_sha256=digest(captured),
                  fresh_process_required=True)
    del session,payload
    torch.cuda.empty_cache()
    return result

def execute(spec,gate,*,mode='run',resume=False,run_id=None,max_steps=32,backend=None):
    if mode not in ('init','run','restore','continuation') or type(max_steps) is not int or not 1<=max_steps<=32:
        raise ValueError('mode/steps must be bounded to 1..32')
    if mode=='continuation' and spec['phase']!='recovery': raise ValueError('continuation restricted to recovery')
    if mode!='run' and resume or spec['phase']=='pilot' and (resume or mode!='run'):
        raise ValueError('invalid phase/mode/resume combination')
    # Rebuild owning baseline instead of trusting a hand-written spec.
    if spec!=baseline.build_spec(401,spec['cell'],spec['phase']): raise ValueError('baseline spec drift')
    validate_gate(gate,spec); hardware_gate(gate)
    backend=backend or OperationalDrive(); out=ROOT/spec['output']
    if backend.account_id()!=gate['drive_account_id']:
        raise ValueError('actual Drive operator account mismatch')
    control_dir=HERE/'operational-control'; control_dir.mkdir(exist_ok=True)
    state={'spec':spec,'status':'running','invocation_id':uuid.uuid4().hex,
           'origin_runtime_id':gate['session_id'],'origin_process_id':os.getpid(),
           'adapter_revision':gate['adapter_revision'],'adapter_file_hashes':adapter_hashes()}
    with (control_dir/'local.lock').open('a') as lock, tempfile.TemporaryDirectory(dir=control_dir) as temporary:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cache=Path(temporary); control=Control(backend,gate['control_folder_id'],gate['operator_account'],control_dir/'history-cache')
        with control.lock():
            ledger=control.load(); key=spec['output']
            pending_review=False
            if gate.get('reviewed_failure') and not ledger.get('reviewed_failures'):
                if mode!='run' or resume or run_id is not None:
                    raise ValueError('failure review allows only a fresh retry')
                reviewed=prepare_failure_review(ledger,gate,spec,control.head,control.seq)
                try: control.commit(reviewed)
                except BaseException:
                    control.preserve_lock=True
                    raise
                ledger=reviewed
            validate_failed_history(ledger)
            review=ledger.get('reviewed_failures',{}).get(REVIEW_FAILURE_ID)
            if review and review.get('retry_status')=='pending':
                validate_pending_retry(ledger,gate,spec,control.seq,mode=mode,resume=resume,run_id=run_id)
                preserve_failed_local_run(ledger,out,control_dir)
                pending_review=True
            if run_id in ledger.get('archived_runs',{}):
                raise ValueError('archived failed run can never resume')
            old=ledger['runs'].get(key)
            if resume or mode in ('restore','continuation'):
                if not old or not run_id: raise ValueError('resume/restore requires committed run ID')
                validate_resume(old,spec,run_id)
                if old.get('adapter_revision')!=gate['adapter_revision']:
                    raise ValueError('resume/restore operational adapter revision drift')
            elif old or out.exists(): raise ValueError('fresh run requires absent output and remote history')
            validate_committed_preflight(ledger,gate,spec)
            reservation_floor_before=ledger['sessions'].get(gate['session_id'],{}).get('charged_seconds',0)
            ledger=reserve(ledger,gate,spec['phase'])
            invocation={'id':state['invocation_id'],'spec':spec,'mode':mode,'status':'reserved',
                        'approval_reference':gate['approval_reference'],'session_id':gate['session_id'],
                        'adapter_revision':gate['adapter_revision'],'adapter_file_hashes':adapter_hashes(),
                        'reservation_floor_before':reservation_floor_before,
                        'reserved_session_charge':ledger['sessions'][gate['session_id']]['charged_seconds'],
                        'main_limit_seconds':main_limit(gate),
                        'preflight_limit_seconds':preflight_limit(gate),
                        'preflight_time_extension_approval_reference':gate.get('preflight_time_extension_approval_reference'),
                        'time_extension_approval_reference':gate.get('time_extension_approval_reference')}
            ledger['invocations'].append(invocation)
            if pending_review:
                review=ledger['reviewed_failures'][REVIEW_FAILURE_ID]
                review.update(retry_invocation_id=invocation['id'],retry_status='reserved')
                invocation['reviewed_retry_of']=REVIEW_FAILURE_ID
            # Immutable readback-verified reservation MUST precede CUDA model initialization.
            try: control.commit(ledger)
            except BaseException:
                control.preserve_lock=True
                raise
            enrollment=control_dir/'enrollments'
            if old:
                state['origin_runtime_id']=old['origin_runtime_id']
                state['origin_process_id']=old.get('origin_process_id')
                state['runtime_restore_history']=copy.deepcopy(old.get('runtime_restore_history',[]))
                state.update({k:old[k] for k in ('identity','run_id','optimizer_step','trained_target_tokens','last_verified_snapshot','continuation') if k in old})
            generations=copy.deepcopy(old.get('generations',[])) if old else []
            restore_verified=old.get('restore_verified',False) if old else False
            def storage_check(extra=0):
                used=sum(p.stat().st_size for p in (ROOT/'reports/seed_repeat_c').rglob('*') if p.is_file())
                quota=backend.quota()
                remote_used=int(quota.get('usage',-1)); remote_limit=int(quota.get('limit',-1))
                if remote_used<0 or remote_limit<0: raise ValueError('remote quota unavailable')
                if (used+extra>=gate['max_output_bytes'] or shutil.disk_usage(ROOT).free<gate['min_free_bytes']+extra
                        or remote_limit-remote_used<extra+gate['min_free_bytes']):
                    raise ValueError('storage budget exhausted')
                # Count measured blob/control/proof/orphan/package bytes in the private
                # operational folders, including immutable ledger snapshots and locks.
                owned_remote=0
                for folder_id in {gate['folder_id'],gate['control_folder_id']}:
                    for remote_file in backend.list_files(folder_id,''):
                        if remote_file.get('mimeType')=='application/vnd.google-apps.folder': continue
                        count=int(remote_file.get('size',-1))
                        if count<0: raise ValueError('remote output byte accounting unavailable')
                        owned_remote+=count
                if used+owned_remote+extra>=gate['max_output_bytes']: raise ValueError('combined output cap exhausted')
            def remember():
                ledger['runs'][key]={**state,'generations':generations,'restore_verified':restore_verified,
                                     'enrollment':retention.load(retention.registration_path(enrollment,key))}
                control.commit(ledger)
            def publish(step,*,report_complete=False):
                nonlocal generations, restore_verified
                pointer,checkpoint=_checkpoint(out)
                size=checkpoint.stat().st_size
                # Local checkpoint history plus original tar, staged archive, readback,
                # independently downloaded restore archive and extracted checkpoint overlap.
                selected=[p for p in out.rglob('*') if p.is_file() and (p.suffix!='.pt' or p==checkpoint)]
                archive_bound=sum(p.stat().st_size+1024 for p in selected)+10240
                storage_check(6*archive_bound+size+1024**2)
                rows=[json.loads(line) for line in (out/'run_manifest.jsonl').read_text().splitlines()]
                if len(rows)!=1: raise ValueError('single manifest required')
                row=rows[0]
                if run_id and row['run_id']!=run_id: raise ValueError('restored run ID changed')
                if row['run_id'] in ledger.get('archived_runs',{}):
                    raise ValueError('fresh retry reused archived failed run identity')
                row.update(experiment_id=spec['experiment_id'],train_order_policy=spec['train_order_policy'],
                           config_file_sha256=spec['config_file_sha256'],execution_source_revision=spec['source_revision'])
                from src.train import _atomic_bytes
                _atomic_bytes(out/'run_manifest.jsonl',lambda stream:stream.write((json.dumps(row,sort_keys=True)+'\n').encode()))
                state['identity']=dict(spec,run_id=row['run_id']); state['run_id']=row['run_id']
                with tempfile.TemporaryDirectory(dir=cache) as tmp:
                    folder=Path(tmp); evidence=transport.publish(backend,gate['folder_id'],out,state['identity'],folder/'publish',step=step,timeout_seconds=gate['reserve_seconds'])
                    receipt_path=folder/'receipt.json'
                    receipt=transport.download_json(backend,evidence['receipt_id'],receipt_path)
                    if digest(receipt_path)!=evidence['receipt_sha256']: raise ValueError('receipt readback hash mismatch')
                    # Prove each owned generation can be freshly extracted before any cleanup.
                    restored=transport.restore(backend,evidence['pointer_id'],folder/'verified-restore',state['identity'],folder/'restore-cache',timeout_seconds=gate['reserve_seconds'])
                    if restored['checkpoint_id']!=evidence['checkpoint_id'] or restored['checkpoint_sha256']!=evidence['checkpoint_sha256']:
                        raise ValueError('published generation restore mismatch')
                    storage_check()
                    evidence['generation_restore_verified']=True
                    evidence['report_complete']=report_complete
                    generations.append(evidence)
                    # Stopped callback boundary; retention helper rejects running states.
                    _atomic_json(out/'execution_state.json',dict(state,status='verified'))
                    retention.retain(key,receipt,evidence,apply=True,root=ROOT,registry=enrollment)
                # Fresh-cache extraction proves this generation; CUDA runtime restoration
                # remains a distinct gate and is never inferred from transport.restore.
                restore_verified=True
                state['last_verified_snapshot']=evidence
                remember()
                generations=cleanup_generations(generations,True,backend.delete_owned)
                return evidence
            def boundary(session,phase):
                baseline.validate_counters(session); storage_check()
                if phase=='before_chunk':
                    control.check_lock()
                    _,ckpt=_checkpoint(out); storage_check(10*ckpt.stat().st_size)
                    return mode!='init'
                if phase in ('initial','after_chunk','after_report'):
                    verified_boundary(state,lambda step:publish(step,report_complete=phase=='after_report'),session.optimizer_step)
                    state.update(status='verified',optimizer_step=session.optimizer_step,trained_target_tokens=session.trained_target_tokens)
                    if phase=='after_report': state['report_published_step']=session.optimizer_step
                    remember()
                return True
            try:
                with deadline(gate['invocation_seconds']):
                    storage_check()
                    if mode=='continuation':
                        if old.get('continuation'): raise ValueError('continuation proof already committed; reuse exact evidence')
                        if old.get('optimizer_step')!=1 or old.get('trained_target_tokens')!=4096:
                            raise ValueError('continuation requires committed verified step1')
                        evidence=old['last_verified_snapshot']; storage_check(12*evidence['archive_bytes']+1024**2)
                        expires=time.time()+gate['invocation_seconds']
                        envelope=cache/'reserved-continuation.json'
                        transport.write_json(envelope,{'status':'reserved','id':state['invocation_id'],
                            'parent_pid':os.getpid(),'expires_at':expires,'spec':spec,'gate':gate,'run_id':run_id})
                        envelope.chmod(0o600)
                        branches=[]; proofs=[]
                        for index in (1,2):
                            branch=cache/('branch-'+str(index))
                            transport.restore(backend,evidence['pointer_id'],branch,old['identity'],cache/('branch-download-'+str(index)),timeout_seconds=gate['reserve_seconds'])
                            control.check_lock()
                            environment=dict(os.environ,SEED401_RESERVED_INVOCATION=state['invocation_id'])
                            subprocess.run([sys.executable,str(HERE/'continuation_worker.py'),'--reservation',str(envelope),'--branch',str(branch)],
                                env=environment,check=True,timeout=max(0.1,expires-time.time()),capture_output=True)
                            summary=json.loads((branch/'continuation-branch.json').read_text())
                            if summary.get('run_id')!=run_id or summary.get('optimizer_step')!=2:
                                raise ValueError('continuation child identity/counters mismatch')
                            pointer,checkpoint=_checkpoint(branch); branches.append((checkpoint,pointer['sha256']))
                            proof=transport.publish(backend,gate['folder_id'],branch,old['identity'],cache/('branch-publish-'+str(index)),step=2,timeout_seconds=gate['reserve_seconds'])
                            transport.restore(backend,proof['pointer_id'],cache/('branch-verify-'+str(index)),old['identity'],cache/('branch-proof-download-'+str(index)),timeout_seconds=gate['reserve_seconds'])
                            proof['generation_restore_verified']=True; proofs.append(proof)
                            storage_check()
                        comparison=comparator.compare_files(branches[0][0],branches[0][1],branches[1][0],branches[1][1],old['identity'],mode='continuation')
                        comparison.update(run_id=run_id,execution_device='cuda',source_revision=spec['source_revision'],
                            data_manifest_hash=spec['data_manifest_hash'],eval_revision=spec['eval_revision'])
                        state.update(status='verified',continuation={'comparison':comparison,
                            'left_sha256':branches[0][1],'right_sha256':branches[1][1],'proofs':proofs,
                            'origin_pointer_id':evidence['pointer_id'],'run_id':run_id,'session_id':gate['session_id']})
                    elif mode=='restore':
                        if out.exists(): raise ValueError('fresh restore requires absent output')
                        identity=old['identity']; evidence=old['last_verified_snapshot']
                        storage_check(4*evidence['archive_bytes']+1024**2)
                        with tempfile.TemporaryDirectory(dir=cache) as tmp:
                            transport.restore(backend,evidence['pointer_id'],out,identity,Path(tmp)/'restore',timeout_seconds=gate['reserve_seconds'])
                        ep=retention.registration_path(enrollment,key); ep.parent.mkdir(parents=True,exist_ok=True)
                        if ep.exists() and retention.load(ep)!=old['enrollment']: raise ValueError('enrollment mismatch')
                        if not ep.exists(): transport.write_json(ep,old['enrollment'])
                        state.update(identity=identity,run_id=run_id,optimizer_step=old.get('optimizer_step'),
                                     trained_target_tokens=old.get('trained_target_tokens'),last_verified_snapshot=evidence,
                                     runtime_restore=restored_session(spec,out),status='verified')
                        state['runtime_restore'].update(origin_runtime_id=state['origin_runtime_id'],
                            restore_runtime_id=gate['session_id'],restore_process_id=os.getpid(),
                            fresh_runtime=state['origin_runtime_id']!=gate['session_id'])
                        state.setdefault('runtime_restore_history',[]).append(state['runtime_restore'])
                        storage_check()
                        restore_verified=True
                    else:
                        if not resume: retention.register_new(key,root=ROOT,registry=enrollment)
                        else:
                            if not out.exists(): raise ValueError('resume output absent; restore first')
                            pointer,_=_checkpoint(out)
                            if pointer['checkpoint_id']!=old['last_verified_snapshot']['checkpoint_id'] or pointer['sha256']!=old['last_verified_snapshot']['checkpoint_sha256']:
                                raise ValueError('local checkpoint differs from committed remote history')
                        from scripts.run_colab_mini import run_colab_mini
                        report=run_colab_mini(ROOT/spec['config'],ROOT/policy.BUNDLE,out,device='cuda',resume=resume,
                                             max_steps=max_steps,pilot=spec['phase']=='pilot',boundary_hook=boundary)
                        if report['status']=='failed': raise ValueError('training failed; operator review required')
                        baseline.verify_final(out,report,spec)
                        if state.get('report_published_step')!=report['optimizer_step']:
                            verified_boundary(state,lambda step:publish(step,report_complete=True),report['optimizer_step'])
                        state.update(status='verified',report=report,optimizer_step=report['optimizer_step'],trained_target_tokens=report['trained_target_tokens'])
                    # Save owned history before removing old remote IDs; failed cleanup is conservative.
                    remember()
                    if restore_verified and len(generations)>2:
                        generations=cleanup_generations(generations,True,backend.delete_owned)
                        remember()
                    invocation['status']='verified'
                    if pending_review:
                        ledger['reviewed_failures'][REVIEW_FAILURE_ID].update(retry_status='verified',retry_run_id=state['run_id'])
                    control.commit(ledger)
                    _atomic_json(out/'execution_state.json',state)
                    if spec['phase']=='main':
                        # Checkpoint, stopped run, invocation and full reservation already
                        # committed/read back above. Keep the original ledger on any failure.
                        control.load()
                        settled=settle_successful_main(ledger,gate,spec,state['invocation_id'],control.head)
                        try:
                            control.commit(settled)
                        except BaseException:
                            control.preserve_lock=True
                            raise
                        ledger=settled
                    return state
            except BaseException as exc:
                state.update(status='failed',error_type=type(exc).__name__)
                invocation.update(status='failed',error_type=type(exc).__name__)
                if pending_review: ledger['reviewed_failures'][REVIEW_FAILURE_ID]['retry_status']='failed'
                ledger['runs'][key]={**state,'generations':generations,'restore_verified':restore_verified}
                # Reservation remains durable even if this failure publication also fails.
                try:
                    control.commit(ledger)
                    control.preserve_lock=False
                except BaseException:
                    control.preserve_lock=True
                if out.exists(): _atomic_json(out/'execution_state.json',state)
                raise

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('inspect','init','run','restore','continuation'))
    parser.add_argument('--cell',choices=policy.CELLS,required=True)
    parser.add_argument('--phase',choices=('recovery','pilot','main'),default='recovery')
    parser.add_argument('--gate',type=Path); parser.add_argument('--resume',action='store_true')
    parser.add_argument('--run-id'); parser.add_argument('--max-steps',type=int,default=32)
    args=parser.parse_args(argv); spec=baseline.build_spec(401,args.cell,args.phase)
    if args.mode=='inspect': print(json.dumps(spec,sort_keys=True,indent=2)); return
    if not args.gate: parser.error('--gate required for GPU modes')
    state=execute(spec,json.loads(args.gate.read_text()),mode=args.mode,resume=args.resume,run_id=args.run_id,max_steps=args.max_steps)
    print(json.dumps(state,sort_keys=True,indent=2))

if __name__=='__main__': main()
