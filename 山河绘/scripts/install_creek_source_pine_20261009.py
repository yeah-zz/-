"""Install only the reviewed second artwork and restart an idle Web service."""
from pathlib import Path
import sys,copy,json,hashlib,shutil,datetime,urllib.request,time,subprocess,os,io,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.apply_reviewed_web_artworks_20261009 import validate,status_installed
from scripts.consolidate_selected_artworks import index_for,digest,stroke_data
from web.catalog import ARTWORKS
from painting_session import session_lock

BASE='http://127.0.0.1:8765'
SOURCE=ROOT/'artifacts/color_coverage_moderate_20261009/02_松岸曲水/任务资料/painting.json'
TARGET=ROOT/ARTWORKS[1][3]
STAMP=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
BACKUP=ROOT/'archive'/('第二幅原图松树_Web应用前_'+STAMP)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def get(endpoint):
    with urllib.request.urlopen(BASE+'/api/'+endpoint,timeout=8) as response:return json.load(response)
def post(endpoint,body,token):
    request=urllib.request.Request(BASE+'/api/'+endpoint,data=json.dumps(body).encode(),method='POST',
        headers={'Content-Type':'application/json','X-Workbench-Token':token})
    with urllib.request.urlopen(request,timeout=8) as response:return json.load(response)
def ready():
    state=get('state')
    assert not state.get('parametersLocked'), 'Hardware task still active'
    assert state.get('status') in ('idle','complete','stopped','saved','error') or state.get('mode')=='demo', 'Active task'
    assert not any(row.get('status')=='building' for row in get('library')['operations'])
    assert not any(row.get('busy') for row in get('studio/jobs')['jobs'])
    assert not any((ROOT/'robot_runs'/name).exists() for name in ('painting_session.lock','blue_wash_motion.lock'))
    return state
def launch():
    with (ROOT/'web/server.creek-source.stdout.log').open('a',encoding='utf-8') as out, (ROOT/'web/server.creek-source.stderr.log').open('a',encoding='utf-8') as err:
        subprocess.Popen([sys.executable,'-u',str(ROOT/'web/server.py')],cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),
            env=os.environ|{'PYTHONIOENCODING':'utf-8'})
    for _ in range(40):
        try:return get('library')
        except OSError:time.sleep(.25)
    raise RuntimeError('Web service startup failed; inspect web/server.creek-source.stderr.log')

def main():
    ready();config=get('config');assert Path(config['root']).resolve()==ROOT.resolve()
    source_sha=sha(SOURCE);before_sha=sha(TARGET);bundle=read(SOURCE)
    assert 'source_aligned_pine_revision' in bundle['workflow']
    assert len(bundle['workflow']['source_aligned_pine_revision']['crowns'])==4
    checked=validate(SOURCE)
    others=[ROOT/item[3] for i,item in enumerate(ARTWORKS) if i!=1]+[ROOT/'web/data/site_settings.json']
    others_sha={str(p):sha(p) for p in others}
    stroke_sha=digest(stroke_data(bundle));bundle['workflow']=status_installed(bundle['workflow'])
    bundle['workflow']['web_installation']=dict(status='installed',date=STAMP,candidate=str(SOURCE),
        candidate_sha256=source_sha,stroke_data_sha256=stroke_sha,revision='source-aligned-four-crowns')
    task=TARGET.parent;package=task.parent;writes={}
    def put(path,obj):writes[path]=json.dumps(obj,ensure_ascii=False,indent=2).encode('utf-8')
    put(TARGET,bundle);put(task/'black_only.json',bundle['ink_job'])
    for p in task.glob('*_黑线任务.json'):put(p,bundle['ink_job'])
    for p in task.glob('*_计划.json'):put(p,bundle['workflow'])
    order=index_for(bundle);counts={k:len(v) for k,v in order.items()}
    put(task/'绘画顺序索引.json',dict(format='shanhe-stroke-order-index-v1',stroke_data_sha256=stroke_sha,counts=counts,stages=order))
    put(task/'部位顺序.json',bundle['workflow']['execution_order'])
    put(task/'Web当前应用记录.json',bundle['workflow']['web_installation']|checked)
    table=io.StringIO(newline='');writer=csv.writer(table)
    writer.writerow(['阶段','当前顺序','笔触ID','部位或区域','笔触类别','原笔触ID','长度mm'])
    for stage,rows in order.items():
        for row in rows:writer.writerow([stage,row['order'],row['id'],row['region'],row['role'],row['parent_stroke_id'],row['length_mm']])
    writes[task/'逐笔顺序索引.csv']=table.getvalue().encode('utf-8-sig')
    preview=SOURCE.parent.parent/'候选轨迹预览.png';preview_bytes=preview.read_bytes()
    for name in ('web_preview.png','当前作画阶段预览.png','执行路径预览_非实测线宽.png','任务预览_线宽色彩为估计.png','黑线加蓝绿_实际路径.png'):
        writes[package/'预览与下压标注'/name]=preview_bytes
    writes[package/'预览与下压标注/纯勾线.png']=(SOURCE.parent.parent/'新黑线预览.png').read_bytes()
    writes[package/'预览与下压标注/原图松树修正对照.jpg']=(SOURCE.parent.parent/'原图松树修正对照.jpg').read_bytes()
    library_path=ROOT/'待画作品/画稿清单.json';library=read(library_path)
    library['artworks'][1].update(counts=counts,stroke_data_sha256=stroke_sha,web_applied_at=STAMP)
    put(library_path,library)
    BACKUP.mkdir(parents=True,exist_ok=False);manifest=[]
    for p,payload in writes.items():
        backup=BACKUP/'files'/p.relative_to(ROOT)
        if p.exists():backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,backup)
        manifest.append(dict(path=str(p),before_sha256=sha(p) if p.exists() else None,after_sha256=hashlib.sha256(payload).hexdigest()))
    report=dict(status='prepared',source_sha256=source_sha,stroke_data_sha256=stroke_sha,files=manifest,counts=counts)
    def save_report():(BACKUP/'应用记录.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    save_report();state=ready()
    assert sha(SOURCE)==source_sha and sha(TARGET)==before_sha
    # Finish only an in-memory preview; live tasks are rejected above.
    if state.get('mode')=='demo' and state['status'] not in ('complete','stopped'):
        post('action',{'command':'finish'},config['token'])
    ready();assert post('service/restart',{},config['token']).get('restarting')
    for _ in range(50):
        try:get('state')
        except OSError:break
        time.sleep(.2)
    else:raise RuntimeError('Web did not stop; formal materials unchanged')
    done=[]
    try:
        with session_lock(ROOT/'robot_runs'):
            assert not (ROOT/'robot_runs/blue_wash_motion.lock').exists()
            for entry in manifest:
                p=Path(entry['path']);assert (sha(p) if p.exists() else None)==entry['before_sha256']
            for p,payload in writes.items():
                p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.install.tmp')
                tmp.write_bytes(payload);tmp.replace(p);done.append(p)
            validate(TARGET);assert digest(stroke_data(read(TARGET)))==stroke_sha
            assert others_sha=={str(p):sha(p) for p in others}
        items=launch();item=next(x for x in items['items'] if x['id']=='painting-02')
        assert item['counts']==counts and not items['unavailable']
        geometry=get('artwork?id=painting-02')
        from web.server import artwork_data
        assert geometry==artwork_data(TARGET)
        with urllib.request.urlopen(BASE+'/api/image?id=painting-02',timeout=8) as response:assert response.read()==preview_bytes
        report.update(status='installed-and-web-verified',other_artworks_and_settings_unchanged=True,offline_validation=checked,
            web_geometry_matches=True,web_image_matches=True,hardware_started=False);save_report()
    except Exception:
        # Roll back the material set if installation or offline validation fails.
        for p in reversed(done):
            backup=BACKUP/'files'/p.relative_to(ROOT)
            if backup.exists():shutil.copy2(backup,p)
            else:p.unlink(missing_ok=True)
        report['status']='rolled-back';save_report()
        try:get('state')
        except OSError:launch()
        raise
    print(json.dumps(dict(status=report['status'],counts=counts,backup=str(BACKUP),web_image_matches=True,web_geometry_matches=True),ensure_ascii=False))

if __name__=='__main__':main()
