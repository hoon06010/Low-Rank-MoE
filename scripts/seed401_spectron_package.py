"""Build a separate, hash-verified portable Spectron payload; no GPU or network."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/seed401_spectron/package'
def build():
    OUT.mkdir(parents=True,exist_ok=True)
    original=json.loads((ROOT/'reports/seed_repeat_c/package/payload_manifest.json').read_text())
    paths=set(r['path'] for r in original['files'])
    paths.update(['scripts/analyze_seed_repeat_c.py','reports/seed_repeat_c/package/payload_manifest.json','scripts/seed401_spectron.py','scripts/seed401_spectron_package.py','scripts/seed401_local.py','scripts/prepare_seed401_local.py'])
    paths.update(f'configs/seed401_spectron/LR-{c}.json' for c in ('HS','LL'))
    rows={n:{'sha256':hashlib.sha256((ROOT/n).read_bytes()).hexdigest(),'bytes':(ROOT/n).stat().st_size} for n in sorted(paths)}
    manifest={'experiment_id':'seed401-spectron-fixed-recipe-v1','files':rows,'remote_downloads':original['remote_downloads'],'gpu_validation':'PENDING: actual finite pilots and different-runtime optimizer restore'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(OUT/'seed401-spectron-input.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in rows:z.write(ROOT/name,name)
        z.write(OUT/'manifest.json','spectron_manifest.json')
    h=hashlib.sha256((OUT/'seed401-spectron-input.zip').read_bytes()).hexdigest()
    (OUT/'sha256.json').write_text(json.dumps({'seed401-spectron-input.zip':h},indent=2)+'\n')
    return h
if __name__=='__main__':print(build())

def notebook():
    """Notebook uses exact manifest pin and explicit allocation/quota inputs."""
    manifest_hash=hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest()
    setup='''from pathlib import Path
import hashlib,json,sys,urllib.request,time,shutil
ROOT=Path('/kaggle/working/seed401_spectron_code')
ROOT.mkdir(parents=True,exist_ok=True)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
found=[p for p in Path('/kaggle/input').rglob('spectron_manifest.json') if sha(p)==EXPECTED_MANIFEST]
if len(found)!=1:raise ValueError('Attach exactly one private Spectron input matching this manifest')
source=found[0].parent;manifest=json.loads(found[0].read_text())
for name,r in manifest['files'].items():
    p=source/name
    if not p.resolve().is_relative_to(source.resolve()) or p.is_symlink() or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('payload mismatch: '+name)
    dest=ROOT/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
for r in manifest['remote_downloads']:
    target=ROOT/r['path'];target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and sha(target)==r['sha256']:continue
    if shutil.disk_usage(ROOT).free<10*1024**3:raise ValueError('10 GiB free required before raw download')
    started=time.monotonic();total=0
    with urllib.request.urlopen(r['url'],timeout=60) as stream,target.open('wb') as f:
        while block:=stream.read(8*1024**2):
            total+=len(block)
            if total>3*1024**3 or time.monotonic()-started>900:raise TimeoutError('bounded data download exceeded')
            f.write(block)
    if sha(target)!=r['sha256']:raise ValueError('raw data hash mismatch')
sys.path.insert(0,str(ROOT))
from scripts import seed401_spectron as spectron
from src.data import load_audited_bundle
BUNDLE=ROOT/'data/prepared/colab_free_mini_v1/bundle.json'
load_audited_bundle(BUNDLE)
STATE=Path('/kaggle/working/seed401_spectron_state')
if RESTORE_FROM:
    if not Path(RESTORE_FROM).resolve().is_relative_to(Path('/kaggle/input')):raise ValueError('Restore from Successful Output only')
    spectron.restore_state(RESTORE_FROM,STATE,RESTORE_SNAPSHOT)
print('LOCAL_PREPARATION_VERIFIED; actual CUDA gates are still required')
'''
    run='''if START_TRAINING:
    if ALLOCATED_AT is None or QUOTA_HOURS is None:raise ValueError('Enter actual allocation timestamp and observed remaining quota')
    resources={'allocated_at':ALLOCATED_AT,'stop_at':ALLOCATED_AT+5400,'quota_hours':QUOTA_HOURS,'budget_seconds':QUOTA_HOURS*3600,'save_reserve_seconds':1020}
    spectron.execute(PHASE,STATE,BUNDLE,resources,ADAMW_STATE)
else:
    print('Training OFF')
'''
    import textwrap
    journal_start="""from pathlib import Path
import json,time,traceback
STARTUP_JOURNAL=Path('/kaggle/working/seed401_spectron_startup_journal.json')
if STARTUP_JOURNAL.exists():startup_history=json.loads(STARTUP_JOURNAL.read_text())
elif STARTUP_HISTORY_FROM:
    prior=Path(STARTUP_HISTORY_FROM).resolve()
    if not prior.is_relative_to(Path('/kaggle/input')):raise ValueError('Startup history must come from saved Output')
    startup_history=json.loads(prior.read_text())
else:startup_history={'experiment_id':'seed401-spectron-fixed-recipe-v1','events':[]}
if startup_history.get('experiment_id')!='seed401-spectron-fixed-recipe-v1' or not isinstance(startup_history.get('events'),list):raise ValueError('wrong startup history')
startup_event={'runtime':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'phase':PHASE,'recorded_at':time.time(),'allocated_at':ALLOCATED_AT,'reserved_seconds':5400 if ALLOCATED_AT is not None else None,'status':'setup_started','accounting':'external startup reservation; merge by runtime with execution ledger, never double-charge'}
startup_history['events'].append(startup_event)
def save_startup():
    temporary=STARTUP_JOURNAL.with_suffix('.tmp');temporary.write_text(json.dumps(startup_history,indent=2)+'\\n');temporary.replace(STARTUP_JOURNAL)
save_startup()
try:
"""
    setup=journal_start+textwrap.indent(setup,'    ')+"""except BaseException:
    startup_event.update(status='setup_failed',error=traceback.format_exc());save_startup();raise
else:
    startup_event['status']='setup_ready';save_startup()
"""
    run="try:\n"+textwrap.indent(run,'    ')+"""except BaseException:
    startup_event.update(status='execution_failed',error=traceback.format_exc());save_startup();raise
else:
    startup_event['status']='execution_finished' if START_TRAINING else 'setup_ready';save_startup()
"""
    def cell(kind,s):return {'cell_type':kind,'metadata':{},'source':s.splitlines(True),**({'execution_count':None,'outputs':[]} if kind=='code' else {})}
    cells=[cell('markdown','# seed401 Spectron HS → LL\nPrivate input; T4 GPU0 FP32. Run pilot, save Output Successful, stop; new runtime restore then recovery; save Successful and stop; new runtime restore then main. Main requires four completed AdamW anchors. Set all inputs explicitly. No GPU validation is claimed by this notebook.\n'),cell('code',f'''START_TRAINING=False
PHASE='pilot' # next fresh runtime: recovery; then main
QUOTA_HOURS=None # observed remaining GPU quota
ALLOCATED_AT=None # actual current allocation start, including setup/idle
ADAMW_STATE='' # verified AdamW state, containing main/LR-*/ results
RESTORE_FROM=''
RESTORE_SNAPSHOT=''
STARTUP_HISTORY_FROM='' # previous Output startup journal, including failed setup sessions
EXPECTED_MANIFEST={manifest_hash!r}
'''),cell('code',setup),cell('code',run),cell('markdown','Quick Save + Save output → Successful → Stop session. Record printed SPECTRON_SNAPSHOT and exact version. Next runtime attaches that Output and fills RESTORE_FROM/RESTORE_SNAPSHOT. Failure reservations remain charged. Do not retry failed sessions automatically. Old AdamW ledgers remain preserved.\n')]
    (OUT/'seed401-spectron.ipynb').write_text(json.dumps({'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells},indent=2)+'\n')
    for c in cells:
        if c['cell_type']=='code':compile(''.join(c['source']),'spectron-notebook','exec')
    return OUT/'seed401-spectron.ipynb'

if __name__=='__main__':print(notebook())
