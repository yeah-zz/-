"""Back up and install the five reviewed candidates; never start robot motion."""
from pathlib import Path
import sys,json,copy,hashlib,shutil,datetime,urllib.request,time,socket
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from web.catalog import ARTWORKS
from painting_session import session_lock,save
from scripts.consolidate_selected_artworks import digest,stroke_data,index_for

CANDIDATES=ROOT/'artifacts/color_coverage_moderate_20261009'
STAMP=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
BACKUP=ROOT/'archive'/('Web正式应用前_'+STAMP)
BASE='http://127.0.0.1:8765'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def get(endpoint):
    with urllib.request.urlopen(BASE+'/api/'+endpoint,timeout=8) as r:return json.load(r)
def check_idle():
    s=get('state');assert not s.get('parametersLocked') and s['status'] in ('idle','complete','stopped','saved','error'),s
    assert not any(x.get('status')=='building' for x in get('library').get('operations',[]))
    assert not any(x.get('busy') for x in get('studio/jobs').get('jobs',[]))
    return s
def validate(path):
    from web.site_settings import prepared_artwork
    from run_color_experiment import module_lines,reload_stroke_limit
    from color_profile import ColorProfile
    from itertools import groupby
    with patch.object(socket.socket,'connect',side_effect=AssertionError('Offline check cannot connect to hardware')):
        prepared=prepared_artwork(path,ROOT/'web/data/site_settings.json')
        m=prepared[1]['modules']['color'];profile=ColorProfile(**m['motion_candidate']).to_dict()
        assert reload_stroke_limit(m)==8
        assert profile['accel_joint_percent']==profile['accel_linear_percent']==0
        for c in ('blue','green'):
            lines,skipped=module_lines(m,c,profile);assert not skipped,(path,c,skipped)
            strokes=[s for s in m['strokes'] if s['color']==c]
            regions=[k for k,_ in groupby(s['drawing_region'] for s in strokes)]
            assert len(regions)==len(set(regions))
            assert len(lines)==len(strokes)
        assert all(.2<=v<=.3 for s in m['strokes'] for v in s['press'])
        assert all(v==.15 for s in prepared[0]['meta']['ink_execution']['base_strokes'] if s['role']=='title' for v in s['press'])
        return {'counts':{k:len(v) for k,v in index_for({'ink_job':prepared[0],'workflow':prepared[1]}).items()},
                'reload_strokes_per_load':8,'effective_motion':profile,'offline_validated':True}
def status_installed(value):
    if isinstance(value,dict):
        return {k:('installed-to-web' if k=='status' and v=='preview-only-not-installed' else status_installed(v)) for k,v in value.items()}
    if isinstance(value,list):return [status_installed(x) for x in value]
    return value

def main():
    check_idle();sources=sorted(CANDIDATES.glob('*/任务资料/painting.json'));assert len(sources)==5
    settings=ROOT/'web/data/site_settings.json';settings_sha=sha(settings)
    specs=[]
    for source,item in zip(sources,ARTWORKS[:5]):
        bundle=read(source);h=sha(source);checked=validate(source)
        specs.append(dict(source=source,target=ROOT/item[3],data=bundle,source_sha=h,
                          old_sha=sha(ROOT/item[3]),stroke_sha=digest(stroke_data(bundle)),checked=checked))
    second=specs[1]['data'];assert 'horizontal_title_revision' in second['workflow']
    assert not any(s.get('drawing_region') in {'左侧远山','中间远山','右侧远山'} for s in second['ink_job']['meta']['ink_execution']['base_strokes'])
    # The third candidate already incorporates the older right-ridge pending fix.
    pending=specs[2]['target'].parent/'待应用_右峰黑线/应用记录.json'
    if pending.exists() and not read(pending).get('applied'):
        older=read(pending.parent/'painting.json')
        assert older['ink_job']['meta']['ink_execution']['base_strokes']==specs[2]['data']['ink_job']['meta']['ink_execution']['base_strokes']
    writes={};records=[]
    def putjson(path,data):writes[path]=json.dumps(data,ensure_ascii=False,separators=(',',':')).encode('utf-8')
    for s in specs:
        b=copy.deepcopy(s['data']);b['workflow']=status_installed(b['workflow'])
        b['workflow']['web_installation']={'status':'installed','date':STAMP,'candidate':str(s['source']),
            'candidate_sha256':s['source_sha'],'stroke_data_sha256':s['stroke_sha']}
        target=s['target'];task=target.parent;package=task.parent
        putjson(target,b);putjson(task/'black_only.json',b['ink_job'])
        for p in task.glob('*_黑线任务.json'):putjson(p,b['ink_job'])
        for p in task.glob('*_计划.json'):putjson(p,b['workflow'])
        order=index_for(b);counts={k:len(v) for k,v in order.items()}
        putjson(task/'绘画顺序索引.json',{'format':'shanhe-stroke-order-index-v1','stroke_data_sha256':s['stroke_sha'],'counts':counts,'stages':order})
        putjson(task/'部位顺序.json',b['workflow']['execution_order'])
        putjson(task/'Web当前应用记录.json',b['workflow']['web_installation']|s['checked'])
        preview=s['source'].parent.parent/'候选轨迹预览.png';assert preview.exists()
        for name in ('web_preview.png','当前作画阶段预览.png','执行路径预览_非实测线宽.png','任务预览_线宽色彩为估计.png','黑线加蓝绿_实际路径.png'):
            writes[package/'预览与下压标注'/name]=preview.read_bytes()
        black_preview=s['source'].parent.parent/'新黑线预览.png'
        if black_preview.exists():writes[package/'预览与下压标注/纯勾线.png']=black_preview.read_bytes()
        records.append({'title':package.name,'target':str(target),'candidate':str(s['source']),
                        'source_sha256':s['source_sha'],'old_sha256':s['old_sha'],
                        'stroke_data_sha256':s['stroke_sha'],'counts':counts})
    if pending.exists() and not read(pending).get('applied'):
        spec=read(pending);spec.update(applied=True,applied_as_part_of=STAMP,
            note='右峰修复已包含在本次正式应用的第三幅候选稿中。');putjson(pending,spec)
    library_path=ROOT/'待画作品/画稿清单.json';library=read(library_path)
    for row,record in zip(library['artworks'][:5],records):
        row.update(counts=record['counts'],stroke_data_sha256=record['stroke_data_sha256'],web_applied_at=STAMP)
    putjson(library_path,library)
    # Prefer an installed, trajectory-generated thumbnail without replacing the
    # user's original reference image. Other library items retain their fallback.
    server=ROOT/'web/server.py';server_bytes=server.read_bytes();server_text=server_bytes.decode('utf-8-sig')
    old="                self.add(ROOT/source, title, ROOT/'待画作品'/directory/'参考原图.png',"
    new="                preview = ROOT/'待画作品'/directory/'预览与下压标注'/'web_preview.png'\n                image = preview if preview.exists() else ROOT/'待画作品'/directory/'参考原图.png'\n                self.add(ROOT/source, title, image,"
    assert old in server_text,'Server changed; review before applying'
    writes[server]=server_text.replace(old,new,1).encode('utf-8')
    BACKUP.mkdir(parents=True,exist_ok=False)
    manifest=[]
    for p,payload in writes.items():
        rel=p.relative_to(ROOT);original=BACKUP/'files'/rel
        exists=p.exists()
        if exists:original.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,original)
        manifest.append({'path':str(p),'existed':exists,'before_sha256':sha(p) if exists else None,
                         'after_sha256':hashlib.sha256(payload).hexdigest()})
    save(BACKUP/'应用清单.json',{'status':'prepared','date':STAMP,'files':manifest,'artworks':records})
    check_idle()
    for s in specs:
        assert sha(s['source'])==s['source_sha'],'Candidate changed during validation'
        assert sha(s['target'])==s['old_sha'],'Formal artwork changed during validation'
    assert sha(settings)==settings_sha
    config=get('config');assert Path(config['root']).resolve()==ROOT.resolve() and config.get('serviceRestart')
    req=urllib.request.Request(BASE+'/api/service/restart',data=b'{}',method='POST',
        headers={'Content-Type':'application/json','X-Workbench-Token':config['token']})
    with urllib.request.urlopen(req,timeout=8) as r:assert json.load(r).get('restarting')
    for _ in range(60):
        try:get('state')
        except OSError:break
        time.sleep(.2)
    else:raise RuntimeError('Web did not exit; no material files installed')
    done=[]
    try:
        with session_lock(ROOT/'robot_runs'):
            assert not (ROOT/'robot_runs/blue_wash_motion.lock').exists()
            for entry in manifest:
                p=Path(entry['path'])
                assert (sha(p) if p.exists() else None)==entry['before_sha256'],'Destination changed'
            for p,payload in writes.items():
                p.parent.mkdir(parents=True,exist_ok=True)
                temp=p.with_name(p.name+'.web-install.tmp');temp.write_bytes(payload);temp.replace(p);done.append(p)
            for s in specs:
                checked=validate(s['target'])
                assert digest(stroke_data(read(s['target'])))==s['stroke_sha']
            for entry in manifest:assert sha(Path(entry['path']))==entry['after_sha256']
        save(BACKUP/'应用清单.json',{'status':'installed','date':STAMP,'files':manifest,'artworks':records,'settings_unchanged':sha(settings)==settings_sha})
    except Exception:
        for p in reversed(done):
            backup=BACKUP/'files'/p.relative_to(ROOT)
            if backup.exists():shutil.copy2(backup,p)
            else:p.unlink(missing_ok=True)
        raise
    print(json.dumps({'status':'installed','backup':str(BACKUP),'artworks':records,'web_restart_required':True},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
