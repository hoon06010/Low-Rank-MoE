"""Offline operational-gate tests: no credentials, network or GPU."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location('drive_runner', HERE/'drive_runner.py')
runner = importlib.util.module_from_spec(SPEC) if SPEC else None
if SPEC and Path(SPEC.origin).exists():
    SPEC.loader.exec_module(runner)

class GateTests(unittest.TestCase):
    def test_operational_runner_exists(self):
        self.assertTrue(callable(getattr(runner, 'execute', None)), 'direct Drive operational runner is missing')

    def test_charge_exhaustion_keeps_input_unchanged(self):
        ledger = {'sessions':{}, 'preflight_seconds':0, 'main_seconds':0, 'runs':{}, 'invocations':[]}
        gate={'session_id':'s','session_started_at':100, 'invocation_seconds':6000, 'reserve_seconds':400}
        with self.assertRaisesRegex(ValueError, 'budget|session'):
            runner.reserve(ledger, gate, 'recovery', now=100)
        self.assertEqual(ledger['sessions'], {})

    def test_phase_budgets_are_separate_and_idle_is_charged(self):
        ledger = {'sessions':{}, 'preflight_seconds':0, 'main_seconds':0, 'runs':{}, 'invocations':[]}
        gate={'session_id':'s','session_started_at':100, 'invocation_seconds':20, 'reserve_seconds':10}
        first=runner.reserve(ledger,gate,'recovery',now=110)
        second=runner.reserve(first,gate,'main',now=150)
        self.assertEqual(first['preflight_seconds'],40)
        self.assertEqual(second['main_seconds'],30)
        self.assertEqual(second['preflight_seconds'],50)

    def test_resume_requires_exact_run_identity(self):
        spec={'seed':401,'cell':'HL'}
        row={'spec':spec,'identity':dict(spec,run_id='same'),'status':'verified'}
        runner.validate_resume(row,spec,'same')
        with self.assertRaisesRegex(ValueError, 'identity'):
            runner.validate_resume(row,spec,'different')
        with self.assertRaisesRegex(ValueError, 'verified'):
            runner.validate_resume(dict(row,status='failed'),spec,'same')

    def test_main_requires_semantic_pilot_and_cuda_evidence(self):
        spec={'source_revision':'a'*64,'seed':401,'phase':'main'}
        gate={'approved':True,'approval_reference':'user','cost_limit':0,'cost_per_hour':0,
              'source_revision':'a'*64,'adapter_revision':runner.adapter_revision(),'single_operator_assertion':True,'operator_account':'example-operator',
              'folder_id':'folder','control_folder_id':'control','provider':'Kaggle','precision':'float32','drive_account_id':'drive-operator',
              'device_name':'T4','torch_version':'2','cuda_version':'12','python_version':'3',
              'session_id':'s','session_started_at':100,'invocation_seconds':20,'reserve_seconds':10,
              'quota_seconds':28800,'min_free_bytes':1024,'max_output_bytes':30*1024**3,
              'storage_evidence':{'status':'fresh-cache-restore-verified','mature_transfer_verified':True},
              'recovery_evidence':{},'pilot_evidence':[]}
        with self.assertRaisesRegex(ValueError, 'recovery'):
            runner.validate_gate(gate,spec,now=100)

class AddedGateTests(unittest.TestCase):
    def test_runtime_switch_requires_stop_proof_and_charges_idle_tail(self):
        first_gate={'session_id':'first','session_started_at':100,'invocation_seconds':20,'reserve_seconds':10}
        first=runner.reserve(runner.empty_ledger(),first_gate,'recovery',now=100)
        next_gate=dict(first_gate,session_id='next',session_started_at=3700)
        with self.assertRaisesRegex(ValueError,'stop evidence'):
            runner.reserve(first,next_gate,'recovery',now=3700)
        next_gate['session_stop_evidence']={'first':{'started_at':100,'stopped_at':3700}}
        charged=runner.reserve(first,next_gate,'recovery',now=3700)
        self.assertEqual(charged['preflight_seconds'],3630)
        self.assertEqual(charged['sessions']['first']['stopped_at'],3700)

    def test_repeated_calls_reserve_additional_time(self):
        ledger=runner.empty_ledger()
        gate={'session_id':'s','session_started_at':100,'invocation_seconds':20,'reserve_seconds':10}
        first=runner.reserve(ledger,gate,'recovery',now=100)
        second=runner.reserve(first,gate,'recovery',now=100)
        self.assertEqual(second['preflight_seconds'],60)

    def test_complete_semantic_recovery_and_pilots_match_committed_control(self):
        spec=runner.baseline.build_spec(401,'HL','main')
        recovery_spec=runner.baseline.build_spec(401,'HL','recovery')
        results=[]
        for step,mode in ((0,'restored'),(1,'restored'),(2,'continuation')):
            excluded=runner.comparator.UUID_FIELDS if mode=='continuation' else frozenset()
            results.append({'status':'exact-state-verified','mode':mode,'optimizer_step':step,
                'seed':401,'cell':'HL','checked_fields':[f for f in runner.comparator.FIELDS if f not in excluded],
                'excluded_fields':sorted(excluded),'run_id':'recovery-run','execution_device':'cuda',
                'fresh_runtime':step==1})
        recovery={'status':'cuda-recovery-verified','seed':401,'cell':'HL','run_id':'recovery-run',
                  'output':recovery_spec['output'],'eval_revision':spec['eval_revision'],
                  'source_revision':spec['source_revision'],'data_manifest_hash':spec['data_manifest_hash'],
                  'fresh_runtime':True,'origin_runtime_id':'origin','restore_runtime_id':'restored',
                  'step0_restore_exact':True,'step1_restore_exact':True,'step2_continuation_exact':True,
                  'execution_device':'cuda','precision':'float32','step1_tokens':4096,'step2_tokens':8192,
                  'step1_cursor':8,'step2_cursor':16,'step0_restore':results[0],'step1_restore':results[1],
                  'continuation_comparison':results[2],'continuation_left_sha256':'a'*64,'continuation_right_sha256':'b'*64}
        results[2].update(run_id='recovery-run',execution_device='cuda',source_revision=spec['source_revision'],data_manifest_hash=spec['data_manifest_hash'],eval_revision=spec['eval_revision'])
        runner.validate_recovery(recovery,spec)
        ledger=runner.empty_ledger()
        ledger['runs'][recovery_spec['output']]={'spec':recovery_spec,'identity':dict(recovery_spec,run_id='recovery-run'),
            'status':'verified','adapter_revision':'preserved-v2-recovery-evidence','runtime_restore_history':results[:2],
            'continuation':{'comparison':results[2],'left_sha256':'a'*64,'right_sha256':'b'*64,'proofs':[{},{}]}}
        pilots=[]
        for cell in runner.policy.CELLS:
            expected=runner.baseline.build_spec(401,cell,'pilot'); run_id='pilot-'+cell
            report={'run_id':run_id,'model':json.loads((runner.ROOT/expected['config']).read_text())['model'],
                    'source_file_hashes':expected['source_file_hashes'],'data_manifest_hash':expected['data_manifest_hash'],
                    'eval_revision':expected['eval_revision'],'pilot_usable':True,'pilot_measurement_complete':True,
                    'pilot_warmup_steps':5,'pilot_measured_steps':100,'optimizer_step':105,'trained_target_tokens':430080,
                    'device':'cuda','precision':'float32','loss_nats':1.0}
            pilots.append({'seed':401,'cell':cell,'run_id':run_id,'output':expected['output'],
                           'source_revision':expected['source_revision'],'data_manifest_hash':expected['data_manifest_hash'],
                           'remote_verified':True,'report':report})
            identity=dict(expected,run_id=run_id)
            ledger['runs'][expected['output']]={'status':'verified','adapter_revision':'preserved-v3-pilot-evidence','spec':expected,'identity':identity,'report':report,
                'last_verified_snapshot':{'identity':identity},'generations':[{'generation_restore_verified':True}]}
        runner.validate_pilots(pilots,spec)
        runner.validate_committed_preflight(ledger,{'recovery_evidence':recovery,'pilot_evidence':pilots},spec)
        ledger['runs'][pilots[0]['output']]['report']={}
        with self.assertRaisesRegex(ValueError,'committed'):
            runner.validate_committed_preflight(ledger,{'recovery_evidence':recovery,'pilot_evidence':pilots},spec)

    def test_main_rejects_one_pilot_relabelled_four_times(self):
        spec={'seed':401,'phase':'main','source_revision':'a'*64,'data_manifest_hash':'b'*64}
        report={'run_id':'same','pilot_usable':True,'pilot_measurement_complete':True,'pilot_warmup_steps':5,
                'pilot_measured_steps':100,'optimizer_step':105,'trained_target_tokens':430080,
                'device':'cuda','precision':'float32','loss_nats':1.0}
        rows=[{'seed':401,'cell':cell,'source_revision':'a'*64,'data_manifest_hash':'b'*64,
               'run_id':'same','remote_verified':True,'report':report} for cell in ('HL','HS','LS','LL')]
        with self.assertRaisesRegex(ValueError,'distinct|identity|cell'):
            runner.validate_pilots(rows,spec)

    def test_cuda_unavailable_never_creates_control_or_calls_backend(self):
        from types import SimpleNamespace
        backend=unittest.mock.Mock()
        spec={'seed':401,'cell':'HL','phase':'recovery'}
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
        with patch.object(runner.baseline,'build_spec',return_value=spec), patch.object(runner,'validate_gate'), patch.dict('sys.modules',{'torch':torch}):
            with self.assertRaisesRegex(ValueError,'CUDA'):
                runner.execute(spec,{},mode='init',backend=backend)
        self.assertEqual(backend.mock_calls,[])

class MainSettlementTests(unittest.TestCase):
    def fixture(self,*,phase='main',before=0,preflight=0,now=100):
        gate={'session_id':'s','session_started_at':100,'invocation_seconds':300,'reserve_seconds':90,'quota_seconds':108000}
        ledger=runner.empty_ledger(); ledger['preflight_seconds']=preflight
        if before: ledger['sessions']['s']={'started_at':100,'charged_seconds':before,'last_phase':'recovery'}
        reserved=runner.reserve(ledger,gate,phase,now=now)
        spec={'seed':401,'phase':phase,'output':'run'}; identity=dict(spec,run_id='same')
        state={'status':'verified','spec':spec,'identity':identity,'last_verified_snapshot':{'status':'remote-publish-verified',
            'identity':identity,'generation_restore_verified':True,'checkpoint_id':'cp','checkpoint_sha256':'a'*64}}
        reserved['runs']['run']=state
        reserved['invocations'].append({'id':'i','spec':spec,'session_id':'s','status':'verified',
            'reservation_floor_before':before,'reserved_session_charge':reserved['sessions']['s']['charged_seconds']})
        return reserved,gate,spec,{'id':'head','sha256':'b'*64}

    def test_verified_main_success_settles_to_elapsed_plus_one_closing_reserve(self):
        ledger,gate,spec,head=self.fixture()
        settled=runner.settle_successful_main(ledger,gate,spec,'i',head,now=120)
        self.assertEqual(settled['main_seconds'],110)
        self.assertEqual(settled['sessions']['s']['charged_seconds'],110)
        self.assertEqual(ledger['main_seconds'],390)
        second=runner.reserve(settled,gate,'main',now=120)
        second['invocations'].append({'id':'next','spec':spec,'session_id':'s','status':'verified',
            'reservation_floor_before':110,'reserved_session_charge':second['sessions']['s']['charged_seconds']})
        again=runner.settle_successful_main(second,gate,spec,'next',head,now=140)
        self.assertEqual(again['main_seconds'],130)

    def test_failed_or_ambiguous_main_never_refunds(self):
        ledger,gate,spec,head=self.fixture()
        for status in ('reserved','failed','settlement-pending'):
            ledger['invocations'][0]['status']=status
            with self.assertRaisesRegex(ValueError,'verified'):
                runner.settle_successful_main(ledger,gate,spec,'i',head,now=120)
            self.assertEqual(ledger['main_seconds'],390)
        ledger['invocations'][0]['status']='verified'
        with self.assertRaisesRegex(ValueError,'commit'):
            runner.settle_successful_main(ledger,gate,spec,'i',{},now=120)
        self.assertEqual(ledger['main_seconds'],390)

    def test_preflight_preserved_and_idle_charged_before_success_settlement(self):
        ledger,gate,spec,head=self.fixture(before=500,preflight=500,now=700)
        settled=runner.settle_successful_main(ledger,gate,spec,'i',head,now=750)
        self.assertEqual(settled['preflight_seconds'],600)
        self.assertEqual(settled['main_seconds'],140)
        next_gate=dict(gate,session_id='new',session_started_at=1000,
            session_stop_evidence={'s':{'started_at':100,'stopped_at':1000}})
        next_ledger=runner.reserve(settled,next_gate,'main',now=1000)
        self.assertEqual(next_ledger['preflight_seconds'],600)
        self.assertEqual(next_ledger['main_seconds'],690)

    def test_only_main_success_can_settle(self):
        ledger,gate,spec,head=self.fixture(phase='recovery')
        with self.assertRaisesRegex(ValueError,'main'):
            runner.settle_successful_main(ledger,gate,spec,'i',head,now=120)
        self.assertEqual(ledger['preflight_seconds'],390)

    def test_extended_main_limit_requires_separate_explicit_approval(self):
        self.assertEqual(runner.main_limit({}),28800)
        with self.assertRaisesRegex(ValueError,'extension'):
            runner.main_limit({'main_limit_seconds':57600})
        with self.assertRaisesRegex(ValueError,'extension'):
            runner.main_limit({'main_limit_seconds':57600,'time_extension_approved':True})
        self.assertEqual(runner.main_limit({'main_limit_seconds':57600,'time_extension_approved':True,
            'time_extension_approval_reference':'separate user approval'}),57600)

class PreflightExtensionTests(unittest.TestCase):
    def test_default_and_separate_preflight_approval_gate(self):
        self.assertEqual(runner.preflight_limit({}),6300)
        for gate in ({'preflight_limit_seconds':9900},
                     {'preflight_limit_seconds':9900,'preflight_time_extension_approved':True},
                     {'preflight_limit_seconds':9900,'time_extension_approved':True,'time_extension_approval_reference':'main approval'}):
            with self.assertRaisesRegex(ValueError,'preflight.*extension'):
                runner.preflight_limit(gate)
        gate={'preflight_limit_seconds':9900,'preflight_time_extension_approved':True,
              'preflight_time_extension_approval_reference':'separate user preflight approval'}
        self.assertEqual(runner.preflight_limit(gate),9900)
        self.assertEqual(runner.main_limit(gate),28800)
        with self.assertRaisesRegex(ValueError,'unsupported'):
            runner.preflight_limit(dict(gate,preflight_limit_seconds=10000))

    def test_extended_reservation_preserves_prior_failure_charge_and_cap(self):
        ledger=runner.empty_ledger(); ledger['preflight_seconds']=6200
        ledger['invocations']=[{'id':'prior-failure','status':'failed','reserved_seconds':400}]
        gate={'session_id':'new','session_started_at':100,'invocation_seconds':900,'reserve_seconds':60,'quota_seconds':108000}
        with self.assertRaisesRegex(ValueError,'budget'):
            runner.reserve(ledger,gate,'pilot',now=100)
        approved=dict(gate,preflight_limit_seconds=9900,preflight_time_extension_approved=True,
                      preflight_time_extension_approval_reference='separate user approval')
        charged=runner.reserve(ledger,approved,'pilot',now=100)
        self.assertEqual(charged['preflight_seconds'],7160)
        self.assertEqual(charged['invocations'],ledger['invocations'])
        self.assertEqual(charged['main_seconds'],0)
        self.assertEqual(ledger['preflight_seconds'],6200)
        too_used=runner.empty_ledger(); too_used['preflight_seconds']=9500
        with self.assertRaisesRegex(ValueError,'budget'):
            runner.reserve(too_used,approved,'pilot',now=100)
        self.assertEqual(too_used['preflight_seconds'],9500)

class FailureReviewTests(unittest.TestCase):
    def fixture(self):
        import copy
        path=HERE/'deployment/runtime-3-pilot-failure-ledger.json'
        record=json.loads(path.read_text()); ledger=copy.deepcopy(record['ledger'])
        spec=runner.baseline.build_spec(401,'HL','pilot')
        gate={'operator_account':'example-operator','reviewed_failure':{
            'invocation_id':'b91efcbd13d2429aabf3e27fe0344fff',
            'failed_run_id':'be7ee742-8a0b-4cde-9730-fb62473d5b05','ledger_head':record['head'],
            'approved':True,'approval_reference':'NEW human review approval'}}
        return ledger,gate,spec,record['head'],record['seq']

    def test_exact_human_review_preserves_failed_history_and_charge(self):
        ledger,gate,spec,head,seq=self.fixture(); old=ledger['runs'][spec['output']]
        reviewed=runner.prepare_failure_review(ledger,gate,spec,head,seq)
        self.assertNotIn(spec['output'],reviewed['runs'])
        self.assertEqual(reviewed['archived_runs'][gate['reviewed_failure']['failed_run_id']],old)
        self.assertEqual(reviewed['invocations'],ledger['invocations'])
        self.assertEqual(reviewed['preflight_seconds'],ledger['preflight_seconds'])
        self.assertEqual(reviewed['sessions'],ledger['sessions'])
        self.assertEqual(reviewed['archived_runs'][old['run_id']]['generations'],old['generations'])
        self.assertIn(spec['output'],ledger['runs'])
        runner.validate_failed_history(reviewed)
        runner.validate_pending_retry(reviewed,gate,spec,seq+1,mode='run',resume=False,run_id=None)

    def test_no_approval_tampered_head_and_other_failures_reject(self):
        import copy
        ledger,gate,spec,head,seq=self.fixture()
        for field,value in (('approved',False),('approval_reference',''),('invocation_id','foreign'),('failed_run_id','foreign')):
            bad=copy.deepcopy(gate); bad['reviewed_failure'][field]=value
            with self.assertRaises(ValueError): runner.prepare_failure_review(ledger,bad,spec,head,seq)
        with self.assertRaisesRegex(ValueError,'head|sequence'):
            runner.prepare_failure_review(ledger,gate,spec,dict(head,sha256='a'*64),seq)
        with self.assertRaisesRegex(ValueError,'head|sequence'):
            runner.prepare_failure_review(ledger,gate,spec,head,seq+1)
        foreign=copy.deepcopy(ledger); foreign['invocations'].append({'id':'other','status':'failed'})
        with self.assertRaisesRegex(ValueError,'other|foreign'):
            runner.prepare_failure_review(foreign,gate,spec,head,seq)
        unknown=copy.deepcopy(ledger); unknown['invocations'].append({'id':'crash','status':'reserved'})
        with self.assertRaisesRegex(ValueError,'other|foreign'):
            runner.prepare_failure_review(unknown,gate,spec,head,seq)

    def test_pending_review_cannot_resume_or_reenter_changed_head(self):
        ledger,gate,spec,head,seq=self.fixture()
        reviewed=runner.prepare_failure_review(ledger,gate,spec,head,seq)
        with self.assertRaisesRegex(ValueError,'fresh'):
            runner.validate_pending_retry(reviewed,gate,spec,seq+1,mode='run',resume=True,run_id=gate['reviewed_failure']['failed_run_id'])
        with self.assertRaisesRegex(ValueError,'fresh'):
            runner.validate_pending_retry(reviewed,gate,spec,seq+1,mode='restore',resume=False,run_id=gate['reviewed_failure']['failed_run_id'])
        with self.assertRaisesRegex(ValueError,'ambiguous|sequence'):
            runner.validate_pending_retry(reviewed,gate,spec,seq+2,mode='run',resume=False,run_id=None)
        record=reviewed['reviewed_failures'][gate['reviewed_failure']['invocation_id']]
        record['retry_invocation_id']='already-reserved'
        with self.assertRaisesRegex(ValueError,'consumed|retry'):
            runner.validate_pending_retry(reviewed,gate,spec,seq+1,mode='run',resume=False,run_id=None)

    def test_reviewed_failure_hash_tamper_and_foreign_failure_still_block(self):
        import copy
        ledger,gate,spec,head,seq=self.fixture()
        reviewed=runner.prepare_failure_review(ledger,gate,spec,head,seq)
        bad=copy.deepcopy(reviewed)
        bad['archived_runs'][gate['reviewed_failure']['failed_run_id']]['generations']=[]
        with self.assertRaisesRegex(ValueError,'review|archive'):
            runner.validate_failed_history(bad)
        bad=copy.deepcopy(reviewed); bad['invocations'].append({'id':'foreign','status':'failed'})
        with self.assertRaisesRegex(ValueError,'review|failed'):
            runner.validate_failed_history(bad)

    def test_local_failed_checkpoints_and_enrollment_are_preserved(self):
        ledger,gate,spec,head,seq=self.fixture()
        reviewed=runner.prepare_failure_review(ledger,gate,spec,head,seq)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); control=root/'control'; registry=control/'enrollments'
            runner.retention.register_new(spec['output'],root=root,registry=registry)
            out=root/spec['output']; (out/'checkpoints').mkdir(parents=True)
            payload=out/'checkpoints/failed-state.pt'; payload.write_bytes(b'failed checkpoint preserved')
            (out/'run_manifest.jsonl').write_text(json.dumps({'run_id':runner.REVIEW_FAILED_RUN_ID})+'\n')
            with patch.object(runner,'ROOT',root): runner.preserve_failed_local_run(reviewed,out,control)
            self.assertFalse(out.exists())
            archive=control/'archived-failed-runs'/runner.REVIEW_FAILED_RUN_ID
            self.assertEqual((archive/'checkpoints/failed-state.pt').read_bytes(),b'failed checkpoint preserved')
            self.assertTrue((registry/'archived'/runner.REVIEW_FAILED_RUN_ID).exists())
            self.assertFalse(runner.retention.registration_path(registry,spec['output']).exists())
            with patch.object(runner,'ROOT',root): runner.preserve_failed_local_run(reviewed,out,control)

    def test_86400_main_extension_is_separate_and_default_unchanged(self):
        self.assertEqual(runner.main_limit({}),28800)
        with self.assertRaisesRegex(ValueError,'extension'):
            runner.main_limit({'main_limit_seconds':86400})
        self.assertEqual(runner.main_limit({'main_limit_seconds':86400,'time_extension_approved':True,
            'time_extension_approval_reference':'NEW explicit24hour approval'}),86400)
        with self.assertRaisesRegex(ValueError,'unsupported'):
            runner.main_limit({'main_limit_seconds':90000,'time_extension_approved':True,
                'time_extension_approval_reference':'approval'})

class FakeDrive:
    def __init__(self): self.files={}; self.counter=0
    def upload(self,folder,name,path):
        self.counter+=1; fid=str(self.counter)
        self.files[fid]={'id':fid,'name':name,'folder':folder,'data':Path(path).read_bytes()}
        return {'id':fid}
    def metadata(self,fid):return {'id':fid,'size':str(len(self.files[fid]['data']))}
    def download(self,fid,path):Path(path).write_bytes(self.files[fid]['data'])
    def list_files(self,folder,prefix):
        return [{'id':k,'name':v['name'],'md5Checksum':__import__('hashlib').md5(v['data'],usedforsecurity=False).hexdigest(),'size':str(len(v['data']))} for k,v in self.files.items() if v['folder']==folder and v['name'].startswith(prefix)]
    def delete_owned(self,fid): del self.files[fid]
    def quota(self): return {'limit':str(10**12),'usage':'0'}
    def account_id(self): return 'drive-operator'

class ControlTests(unittest.TestCase):
    def test_immutable_ledger_chain_and_duplicate_lock_exclusion(self):
        drive=FakeDrive()
        with tempfile.TemporaryDirectory() as tmp:
            control=runner.Control(drive,'control','operator',Path(tmp))
            with control.lock():
                self.assertEqual(control.load()['invocations'],[])
                control.commit({'sessions':{},'preflight_seconds':0,'main_seconds':0,'runs':{},'invocations':[{'id':'i'}]})
                self.assertEqual(control.load()['invocations'],[{'id':'i'}])
                with self.assertRaisesRegex(ValueError,'lock'):
                    with runner.Control(drive,'control','operator',Path(tmp)).lock(): pass
            self.assertEqual(drive.list_files('control','seed401-lock-'),[])

    def test_persisted_verified_cache_avoids_history_download_and_detects_remote_mutation(self):
        drive=FakeDrive()
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp)
            control=runner.Control(drive,'control','operator',cache)
            with control.lock():
                control.load(); control.commit(runner.empty_ledger())
            with patch.object(drive,'download',wraps=drive.download) as downloaded:
                runner.Control(drive,'control','operator',cache).load()
                downloaded.assert_not_called()
            fid=drive.list_files('control',runner.PREFIX)[0]['id']
            drive.files[fid]['data']+=b' '
            with self.assertRaisesRegex(ValueError,'cache/history changed'):
                runner.Control(drive,'control','operator',cache).load()

    def test_publish_failure_prevents_next_chunk_and_records_failure(self):
        state={'status':'running'}
        seen=[]
        def publish(step): seen.append(step); raise RuntimeError('upload unavailable')
        with self.assertRaises(RuntimeError):
            runner.verified_boundary(state,publish,32)
        self.assertEqual(seen,[32])
        self.assertEqual(state['status'],'failed')
        self.assertNotIn('last_verified_snapshot',state)

    @staticmethod
    def generation(checkpoint,tag,*,complete=False,sha=None):
        return {'checkpoint_id':checkpoint,'checkpoint_sha256':sha or (checkpoint[-1]*64),
                'archive_id':tag+'a','receipt_id':tag+'r','pointer_id':tag+'p',
                'report_complete':complete,'report_metadata':tag}

    def test_cleanup_only_owned_old_generations_after_restore(self):
        rows=[self.generation('cp0','zero'),self.generation('cp1','one'),self.generation('cp2','two')]
        deleted=[]
        runner.cleanup_generations(rows,False,deleted.append)
        self.assertEqual(deleted,[])
        kept=runner.cleanup_generations(rows,True,deleted.append)
        self.assertEqual(deleted,['zeroa','zeror','zerop'])
        self.assertEqual(kept,rows[1:])

    def test_report_refresh_retains_prior_distinct_checkpoint(self):
        rows=[self.generation('cp0','zero',complete=True),self.generation('cp1','chunk'),self.generation('cp1','report',complete=True)]
        deleted=[]
        kept=runner.cleanup_generations(rows,True,deleted.append)
        self.assertEqual(kept,[rows[0],rows[2]])
        self.assertEqual(deleted,['chunka','chunkr','chunkp'])
        self.assertEqual(kept[-1]['report_metadata'],'report')

    def test_next_checkpoint_evicts_only_older_distinct_state_and_refresh_duplicate(self):
        rows=[self.generation('cp0','zero',complete=True),self.generation('cp1','one',complete=True),
              self.generation('cp2','chunk'),self.generation('cp2','report',complete=True)]
        deleted=[]
        kept=runner.cleanup_generations(rows,True,deleted.append)
        self.assertEqual(kept,[rows[1],rows[3]])
        self.assertEqual(deleted,['zeroa','zeror','zerop','chunka','chunkr','chunkp'])

    def test_complete_report_survives_later_non_report_duplicate(self):
        rows=[self.generation('cp0','zero',complete=True),self.generation('cp1','report',complete=True),
              self.generation('cp1','latechunk')]
        kept=runner.cleanup_generations(rows,True,lambda fid:None)
        self.assertEqual(kept,[rows[0],rows[1]])
        self.assertEqual(kept[-1]['report_metadata'],'report')

    def test_changed_sha_for_same_checkpoint_id_rejects_before_deletion(self):
        rows=[self.generation('cp1','one'),self.generation('cp1','changed',sha='a'*64)]
        deleted=[]
        with self.assertRaisesRegex(ValueError,'checkpoint.*hash'):
            runner.cleanup_generations(rows,True,deleted.append)
        self.assertEqual(deleted,[])


class OperationalIntegrationTests(unittest.TestCase):
    def fixture(self,root,phase='recovery'):
        import torch
        from types import SimpleNamespace
        output='reports/seed_repeat_c/seed-401/LR-HL' if phase=='main' else f'reports/seed_repeat_c/preflight/{phase}/seed-401/LR-HL'
        spec={'seed':401,'cell':'HL','phase':phase,'output':output,'config':'config.json',
              'experiment_id':runner.retention.EXPERIMENT,'train_order_policy':'fixed',
              'source_revision':'a'*64,'data_manifest_hash':'b'*64,'eval_revision':'c'*64,'config_file_sha256':'d'*64}
        gate={'session_id':'s','session_started_at':runner.time.time(),'invocation_seconds':60,
              'reserve_seconds':10,'quota_seconds':6300,'min_free_bytes':1024,'max_output_bytes':30*1024**3,
              'control_folder_id':'control','folder_id':'blobs','operator_account':'operator','approval_reference':'user','drive_account_id':'drive-operator','adapter_revision':runner.adapter_revision()}
        def trainer(config,bundle,out,**kw):
            out=Path(out); (out/'checkpoints').mkdir(parents=True)
            for step in range(kw['max_steps']+1):
                if step and not kw['boundary_hook'](session,'before_chunk'): break
                cid=str(step)*32; cp=out/'checkpoints'/('checkpoint-'+cid+'.pt')
                torch.save({'config':{'seed':401,'cell':'HL'},'run_id':'run-one',
                            'source_revision':'a'*64,'data_manifest_hash':'b'*64,'eval_revision':'c'*64},cp)
                pointer={'filename':cp.name,'checkpoint_id':cid,'sha256':runner.digest(cp)}
                (out/'checkpoints/latest.json').write_text(json.dumps(pointer))
                row={'run_id':'run-one','seed':401,'cell':'HL','checkpoint_id':cid,'checkpoint_step':step,
                     'data_manifest_hash':'b'*64,'eval_revision':'c'*64,'status':'incomplete','trained_target_tokens':step*4096}
                (out/'run_manifest.jsonl').write_text(json.dumps(row)+'\n')
                session=SimpleNamespace(optimizer_step=step,trained_target_tokens=step*4096,cursor=step*8,epoch=0)
                kw['boundary_hook'](session,'initial' if step==0 else 'after_chunk')
            return {'status':'incomplete','optimizer_step':step,'trained_target_tokens':step*4096,'run_id':'run-one'}
        return spec,gate,trainer

    def test_real_transport_retention_and_remote_control_at_mock_training_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); here=root/'reports/seed_repeat_c/seed-401'; here.mkdir(parents=True)
            spec,gate,trainer=self.fixture(root); drive=FakeDrive()
            with patch.object(runner,'ROOT',root),patch.object(runner,'HERE',here),patch.object(runner.baseline,'build_spec',return_value=spec),patch.object(runner,'validate_gate'),patch.object(runner,'hardware_gate'),patch('scripts.run_colab_mini.run_colab_mini',side_effect=trainer):
                state=runner.execute(spec,gate,backend=drive,max_steps=3)
            self.assertEqual(state['status'],'verified')
            self.assertEqual(state['run_id'],'run-one')
            self.assertEqual(len(list((root/spec['output']/'checkpoints').glob('*.pt'))),2)
            self.assertEqual(len(drive.list_files('blobs','commit-')),2)
            self.assertEqual(len(drive.list_files('blobs','generation-')),4)
            self.assertEqual(list((here/'operational-control').glob('tmp*')),[])
            self.assertEqual(drive.list_files('control','seed401-lock-'),[])
            ledger=runner.Control(drive,'control','operator',here).load()
            self.assertGreater(ledger['preflight_seconds'],0)
            self.assertEqual(ledger['invocations'][0]['status'],'verified')

    def reviewed_integration_fixture(self,root,drive):
        import copy
        spec,gate,trainer=self.fixture(root,phase='pilot')
        raw=json.loads((HERE/'deployment/runtime-3-pilot-failure-ledger.json').read_text())
        ledger=copy.deepcopy(raw['ledger']); old_key='reports/seed_repeat_c/preflight/pilot/seed-401/LR-HL'
        failed=ledger['runs'].pop(old_key); failed['spec']=spec; failed['identity']=dict(spec,run_id=runner.REVIEW_FAILED_RUN_ID)
        ledger['runs'][spec['output']]=failed
        next(i for i in ledger['invocations'] if i['id']==runner.REVIEW_FAILURE_ID)['spec']=spec
        gate.update(operator_account='example-operator',reviewed_failure={'invocation_id':runner.REVIEW_FAILURE_ID,
            'failed_run_id':runner.REVIEW_FAILED_RUN_ID,'approved':True,'approval_reference':'NEW human approval'})
        gate['session_stop_evidence']={key:{'started_at':session['started_at'],
            'stopped_at':session['started_at']+session['charged_seconds']} for key,session in ledger['sessions'].items() if session.get('stopped_at') is None}
        control=runner.Control(drive,'control','example-operator',root/'reports/seed_repeat_c/seed-401/operational-control/history-cache')
        with control.lock(): control.load(); control.commit(ledger)
        gate['reviewed_failure']['ledger_head']=dict(control.head)
        return spec,gate,trainer,ledger,dict(control.head),control.seq

    def test_failed_review_commit_blocks_training_and_preserves_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); here=root/'reports/seed_repeat_c/seed-401'; here.mkdir(parents=True); drive=FakeDrive()
            spec,gate,trainer,original,head,seq=self.reviewed_integration_fixture(root,drive)
            commit=runner.Control.commit
            def fail_review(control,ledger):
                if ledger.get('reviewed_failures'): raise OSError('review readback failed')
                return commit(control,ledger)
            with patch.object(runner,'ROOT',root),patch.object(runner,'HERE',here),patch.object(runner,'REVIEW_ORIGINAL_HEAD',head),patch.object(runner,'REVIEW_ORIGINAL_SEQ',seq),patch.object(runner.baseline,'build_spec',return_value=spec),patch.object(runner,'validate_gate'),patch.object(runner,'hardware_gate'),patch('scripts.run_colab_mini.run_colab_mini') as train,patch.object(runner.Control,'commit',new=fail_review):
                with self.assertRaisesRegex(OSError,'review readback failed'):
                    runner.execute(spec,gate,backend=drive,max_steps=1)
                train.assert_not_called()
            self.assertEqual(len(drive.list_files('control','seed401-lock-')),1)
            current=runner.Control(drive,'control','example-operator',here/'operational-control/history-cache').load()
            self.assertEqual(current,original)

    def test_retry_reservation_failure_after_review_holds_lock_before_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); here=root/'reports/seed_repeat_c/seed-401'; here.mkdir(parents=True); drive=FakeDrive()
            spec,gate,trainer,original,head,seq=self.reviewed_integration_fixture(root,drive)
            commit=runner.Control.commit
            def fail_reservation(control,ledger):
                if any(i.get('reviewed_retry_of') for i in ledger['invocations']):
                    raise OSError('retry reservation ambiguous')
                return commit(control,ledger)
            with patch.object(runner,'ROOT',root),patch.object(runner,'HERE',here),patch.object(runner,'REVIEW_ORIGINAL_HEAD',head),patch.object(runner,'REVIEW_ORIGINAL_SEQ',seq),patch.object(runner.baseline,'build_spec',return_value=spec),patch.object(runner,'validate_gate'),patch.object(runner,'hardware_gate'),patch.object(runner,'validate_committed_preflight'),patch('scripts.run_colab_mini.run_colab_mini') as train,patch.object(runner.Control,'commit',new=fail_reservation):
                with self.assertRaisesRegex(OSError,'retry reservation ambiguous'):
                    runner.execute(spec,gate,backend=drive,max_steps=1)
                train.assert_not_called()
            self.assertEqual(len(drive.list_files('control','seed401-lock-')),1)
            current=runner.Control(drive,'control','example-operator',here/'operational-control/history-cache').load()
            self.assertEqual(current['invocations'],original['invocations'])
            self.assertEqual(current['preflight_seconds'],original['preflight_seconds'])
            self.assertEqual(current['reviewed_failures'][runner.REVIEW_FAILURE_ID]['retry_status'],'pending')

    def test_review_allows_one_fresh_distinct_run_and_preserves_failed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); here=root/'reports/seed_repeat_c/seed-401'; here.mkdir(parents=True); drive=FakeDrive()
            spec,gate,trainer,original,head,seq=self.reviewed_integration_fixture(root,drive)
            with patch.object(runner,'ROOT',root),patch.object(runner,'HERE',here),patch.object(runner,'REVIEW_ORIGINAL_HEAD',head),patch.object(runner,'REVIEW_ORIGINAL_SEQ',seq),patch.object(runner.baseline,'build_spec',return_value=spec),patch.object(runner,'validate_gate'),patch.object(runner,'validate_committed_preflight'),patch.object(runner,'hardware_gate'),patch('scripts.run_colab_mini.run_colab_mini',side_effect=trainer):
                state=runner.execute(spec,gate,backend=drive,max_steps=1)
                current=runner.Control(drive,'control','example-operator',here/'operational-control/history-cache').load()
                runner.validate_failed_history(current)
                with self.assertRaisesRegex(ValueError,'fresh|absent'):
                    runner.execute(spec,gate,backend=drive,max_steps=1)
            self.assertNotEqual(state['run_id'],runner.REVIEW_FAILED_RUN_ID)
            self.assertEqual(current['archived_runs'][runner.REVIEW_FAILED_RUN_ID],original['runs'][spec['output']])
            old=next(i for i in original['invocations'] if i['id']==runner.REVIEW_FAILURE_ID)
            self.assertEqual(next(i for i in current['invocations'] if i['id']==runner.REVIEW_FAILURE_ID),old)
            self.assertGreaterEqual(current['preflight_seconds'],original['preflight_seconds'])
            self.assertEqual(current['reviewed_failures'][runner.REVIEW_FAILURE_ID]['retry_status'],'verified')

    def test_main_settlement_commits_actual_charge_after_verified_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); here=root/'reports/seed_repeat_c/seed-401'; here.mkdir(parents=True)
            spec,gate,trainer=self.fixture(root,phase='main'); drive=FakeDrive()
            with patch.object(runner,'ROOT',root),patch.object(runner,'HERE',here),patch.object(runner.baseline,'build_spec',return_value=spec),patch.object(runner,'validate_gate'),patch.object(runner,'validate_committed_preflight'),patch.object(runner,'hardware_gate'),patch('scripts.run_colab_mini.run_colab_mini',side_effect=trainer):
                runner.execute(spec,gate,backend=drive,max_steps=1)
            ledger=runner.Control(drive,'control','operator',here/'operational-control/history-cache').load()
            invocation=ledger['invocations'][-1]
            self.assertEqual(invocation['status'],'verified')
            self.assertEqual(invocation['settlement']['status'],'verified-success-only')
            self.assertLess(ledger['main_seconds'],invocation['reserved_session_charge'])
            self.assertEqual(ledger['preflight_seconds'],0)
            self.assertEqual(ledger['runs'][spec['output']]['last_verified_snapshot']['checkpoint_id'],invocation['settlement']['checkpoint_id'])

    def test_failed_settlement_keeps_full_main_reservation_and_failed_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); here=root/'reports/seed_repeat_c/seed-401'; here.mkdir(parents=True)
            spec,gate,trainer=self.fixture(root,phase='main'); drive=FakeDrive()
            original=runner.Control.commit
            def commit(control,ledger):
                if ledger['invocations'] and ledger['invocations'][-1].get('settlement'):
                    raise OSError('settlement readback failed')
                return original(control,ledger)
            with patch.object(runner,'ROOT',root),patch.object(runner,'HERE',here),patch.object(runner.baseline,'build_spec',return_value=spec),patch.object(runner,'validate_gate'),patch.object(runner,'validate_committed_preflight'),patch.object(runner,'hardware_gate'),patch('scripts.run_colab_mini.run_colab_mini',side_effect=trainer),patch.object(runner.Control,'commit',new=commit):
                with self.assertRaisesRegex(OSError,'settlement readback failed'):
                    runner.execute(spec,gate,backend=drive,max_steps=1)
            ledger=runner.Control(drive,'control','operator',here/'operational-control/history-cache').load()
            self.assertEqual(ledger['invocations'][-1]['status'],'failed')
            self.assertEqual(ledger['main_seconds'],ledger['invocations'][-1]['reserved_session_charge'])
            self.assertNotIn('settlement',ledger['invocations'][-1])

    def test_unknown_settlement_and_failure_marker_preserve_control_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); here=root/'reports/seed_repeat_c/seed-401'; here.mkdir(parents=True)
            spec,gate,trainer=self.fixture(root,phase='main'); drive=FakeDrive()
            original=runner.Control.commit
            def commit(control,ledger):
                invocation=ledger['invocations'][-1] if ledger['invocations'] else {}
                if invocation.get('settlement') or invocation.get('status')=='failed':
                    raise OSError('ambiguous remote write')
                return original(control,ledger)
            with patch.object(runner,'ROOT',root),patch.object(runner,'HERE',here),patch.object(runner.baseline,'build_spec',return_value=spec),patch.object(runner,'validate_gate'),patch.object(runner,'validate_committed_preflight'),patch.object(runner,'hardware_gate'),patch('scripts.run_colab_mini.run_colab_mini',side_effect=trainer),patch.object(runner.Control,'commit',new=commit):
                with self.assertRaisesRegex(OSError,'ambiguous remote write'):
                    runner.execute(spec,gate,backend=drive,max_steps=1)
            self.assertEqual(len(drive.list_files('control','seed401-lock-')),1)
            ledger=runner.Control(drive,'control','operator',here/'operational-control/history-cache').load()
            self.assertEqual(ledger['main_seconds'],ledger['invocations'][-1]['reserved_session_charge'])
            self.assertNotIn('settlement',ledger['invocations'][-1])

    def test_v3_run_resume_is_rejected_by_v4_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); here=root/'reports/seed_repeat_c/seed-401'; here.mkdir(parents=True)
            spec,gate,trainer=self.fixture(root); drive=FakeDrive()
            with patch.object(runner,'ROOT',root),patch.object(runner,'HERE',here),patch.object(runner.baseline,'build_spec',return_value=spec),patch.object(runner,'validate_gate'),patch.object(runner,'hardware_gate'),patch('scripts.run_colab_mini.run_colab_mini',side_effect=trainer):
                runner.execute(spec,gate,backend=drive,max_steps=1)
                control=runner.Control(drive,'control','operator',here/'operational-control/history-cache')
                with control.lock():
                    ledger=control.load()
                    ledger['runs'][spec['output']]['adapter_revision']='c411516e442143a2014a341c7a25539948c213541cc40ac0b346867f4549aa1f'
                    control.commit(ledger)
                with patch('scripts.run_colab_mini.run_colab_mini') as trained:
                    with self.assertRaisesRegex(ValueError,'adapter revision drift'):
                        runner.execute(spec,gate,resume=True,run_id='run-one',backend=drive)
                    trained.assert_not_called()

    def test_continuation_subprocesses_are_reserved_and_proofs_committed(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); here=root/'reports/seed_repeat_c/seed-401'; here.mkdir(parents=True)
            spec,gate,trainer=self.fixture(root); drive=FakeDrive(); calls=[]
            def child(command,**options):
                branch=Path(command[-1]); envelope=json.loads(Path(command[-3]).read_text())
                self.assertEqual(options['env']['SEED401_RESERVED_INVOCATION'],envelope['id'])
                current=runner.Control(drive,'control','operator',here/'operational-control/history-cache').load()
                self.assertEqual(current['invocations'][-1]['status'],'reserved')
                self.assertEqual(current['invocations'][-1]['mode'],'continuation')
                calls.append(command)
                cid=str(len(calls)+4)*32; cp=branch/'checkpoints'/('checkpoint-'+cid+'.pt')
                torch.save({'state':'same-step2'},cp)
                pointer={'filename':cp.name,'checkpoint_id':cid,'sha256':runner.digest(cp)}
                (branch/'checkpoints/latest.json').write_text(json.dumps(pointer))
                row=json.loads((branch/'run_manifest.jsonl').read_text())
                row.update(checkpoint_id=cid,checkpoint_step=2,trained_target_tokens=8192)
                (branch/'run_manifest.jsonl').write_text(json.dumps(row)+'\n')
                (branch/'continuation-branch.json').write_text(json.dumps({'run_id':'run-one','optimizer_step':2}))
            with patch.object(runner,'ROOT',root),patch.object(runner,'HERE',here),patch.object(runner.baseline,'build_spec',return_value=spec),patch.object(runner,'validate_gate'),patch.object(runner,'hardware_gate'),patch('scripts.run_colab_mini.run_colab_mini',side_effect=trainer):
                runner.execute(spec,gate,backend=drive,max_steps=1)
                with patch.object(runner.subprocess,'run',side_effect=child),patch.object(runner.comparator,'compare_files',return_value={'status':'exact-state-verified','mode':'continuation','optimizer_step':2}):
                    state=runner.execute(spec,gate,mode='continuation',run_id='run-one',backend=drive)
            self.assertEqual(len(calls),2)
            self.assertEqual(state['optimizer_step'],1)
            self.assertEqual(len(state['continuation']['proofs']),2)
            self.assertEqual(state['continuation']['comparison']['run_id'],'run-one')
            ledger=runner.Control(drive,'control','operator',here/'operational-control/history-cache').load()
            self.assertEqual(ledger['invocations'][-1]['status'],'verified')
            self.assertEqual(ledger['runs'][spec['output']]['continuation'],state['continuation'])

    def test_interrupted_reservation_blocks_all_later_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); here=root/'reports/seed_repeat_c/seed-401'; here.mkdir(parents=True)
            spec,gate,trainer=self.fixture(root); drive=FakeDrive()
            control=runner.Control(drive,'control','operator',here)
            with control.lock():
                ledger=control.load(); ledger['invocations'].append({'id':'crashed','status':'reserved'})
                control.commit(ledger)
            with patch.object(runner,'ROOT',root),patch.object(runner,'HERE',here),patch.object(runner.baseline,'build_spec',return_value=spec),patch.object(runner,'validate_gate'),patch.object(runner,'hardware_gate'),patch('scripts.run_colab_mini.run_colab_mini',side_effect=trainer) as trained:
                with self.assertRaisesRegex(ValueError,'operator review'):
                    runner.execute(spec,gate,backend=drive,max_steps=3)
                trained.assert_not_called()

    def test_initial_remote_publish_failure_stops_before_first_training_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); here=root/'reports/seed_repeat_c/seed-401'; here.mkdir(parents=True)
            spec,gate,trainer=self.fixture(root); drive=FakeDrive()
            with patch.object(runner,'ROOT',root),patch.object(runner,'HERE',here),patch.object(runner.baseline,'build_spec',return_value=spec),patch.object(runner,'validate_gate'),patch.object(runner,'hardware_gate'),patch('scripts.run_colab_mini.run_colab_mini',side_effect=trainer),patch.object(runner.transport,'publish',side_effect=RuntimeError('transfer failed')):
                with self.assertRaisesRegex(RuntimeError,'transfer failed'):
                    runner.execute(spec,gate,backend=drive,max_steps=3)
            ledger=runner.Control(drive,'control','operator',here).load()
            self.assertGreater(ledger['preflight_seconds'],0)
            self.assertEqual(ledger['invocations'][0]['status'],'failed')
            self.assertEqual(len(list((root/spec['output']/'checkpoints').glob('*.pt'))),1)
            with self.assertRaisesRegex(ValueError,'verified'):
                runner.validate_resume(ledger['runs'][spec['output']],spec,'run-one')

if __name__=='__main__': unittest.main()

