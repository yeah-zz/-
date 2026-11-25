"""Use artwork 1's actual pine paths in the creek preview, with a fitted root."""
from pathlib import Path
import sys,copy,json,socket
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,cv2
from PIL import Image
from ink_control import refresh_stats
from scripts.consolidate_selected_artworks import digest,stroke_data
from scripts.creek_open_branches_20261009 import FOLDER,TARGET,read,save,sha
from scripts.preview_color_coverage_20261009 import narrow_render,make_board,aligned_reference,connect_region

BASELINE=FOLDER/'历史版本/松树对齐第一幅前/painting.json'
REFERENCE=FOLDER.parent/'01_松瀑亭舟/任务资料/painting.json'
REMOVE_PREFIXES=('song-an-qu-shui-ink-00278','song-an-qu-shui-ink-00334',
    'song-an-qu-shui-ink-00338','song-an-qu-shui-ink-00312',
    'song-an-qu-shui-ink-00280','song-an-qu-shui-ink-00311')

def transform(points):
    p=np.asarray(points,float)
    q=(p-[282.,685.])*.80+[312.,680.]
    # Fit only the low trunk/root to the creek's existing sloping rock edge.
    # The canopy and upper branch structure retain the source's proportions.
    t=np.clip((p[:,1]-600)/85,0,1);blend=t*t*(3-2*t)
    q[:,1]-=blend*(20+.12*(p[:,0]-282))
    q[:,1]+=18*(1-blend)  # separate the crown from the background mountain wash
    return q

def main():
    def offline(*a,**k):raise AssertionError('Preview only: hardware connection prohibited')
    socket.socket.connect=offline;socket.socket.connect_ex=offline
    if not BASELINE.exists():
        BASELINE.parent.mkdir(parents=True,exist_ok=True);BASELINE.write_bytes(TARGET.read_bytes())
    old=read(BASELINE);new=copy.deepcopy(old);ref=read(REFERENCE);refhash=sha(REFERENCE)
    canonical=ROOT/'待画作品/02_松岸曲水_竖题山河绘_绘字修复版/任务资料/painting.json';canonhash=sha(canonical)
    scale=old['ink_job']['meta']['ink_execution']['mm_per_px']
    ink=old['ink_job']['meta']['ink_execution']['base_strokes'];kept=[];removed=[];trimmed=None
    for s in ink:
        if s['id'].startswith(REMOVE_PREFIXES) or (s['id'].startswith('creek-clean-branch-') and s.get('drawing_region')=='左侧主松树'):
            removed.append(s['id']);continue
        st=copy.deepcopy(s)
        if s['id']=='song-an-qu-shui-ink-00346':
            # This path mixes the former trunk edge with the rock crest. Keep
            # the exact rock suffix from the junction, not a rebuilt rock line.
            q=np.asarray(s['points']);at=int(np.argmin(np.linalg.norm(q-[329.,652.],axis=1)))
            for key,value in s.items():
                if isinstance(value,list) and len(value)==len(q):st[key]=value[at:]
            st.update(id=s['id']+'-rock-only',parent_stroke_id=s['id'])
            refresh_stats(st,scale);trimmed=dict(id=s['id'],kept_from_point=at)
        kept.append(st)
    refink=[s for s in ref['ink_job']['meta']['ink_execution']['base_strokes'] if s.get('drawing_region')=='整棵松树']
    pine=[]
    for i,src in enumerate(refink):
        st=copy.deepcopy(src);st['points']=transform(src['points']).tolist()
        st.update(id=f'creek-first-pine-ink-{i:03d}',drawing_region='左侧主松树',
                  reference_tree_stroke_id=src['id'],subject_order=0)
        if 'parent_stroke_id' in st:st['parent_stroke_id']='creek-first-pine-'+str(st['parent_stroke_id'])
        pine.append(refresh_stats(st,scale))
    at=max(i+1 for i,s in enumerate(kept) if s.get('drawing_region')=='左侧主松树')
    newink=kept[:at]+pine+kept[at:]
    refcolors=[s for s in ref['workflow']['modules']['color']['strokes'] if s.get('drawing_region')=='整棵松树']
    copied=[]
    for i,src in enumerate(refcolors):
        st=copy.deepcopy(src);st['points']=transform(src['points']).tolist()
        st.update(id=f'creek-first-pine-green-{i:03d}',drawing_region='左侧主松树',
                  reference_tree_stroke_id=src['id'],subject_order=0)
        st.setdefault('canopy_index',int(src['region_id'].rsplit('-',1)[-1]))
        st['press']=np.clip(st['press'],.2,.3).tolist()
        copied.append(refresh_stats(st,scale))
    # Keep each source canopy together. Join only through already painted space.
    # Never connect separate canopies across the white openings between them.
    greens=[];joins=0;w,h=old['ink_job']['meta']['image_size']
    for cluster in dict.fromkeys(s['canopy_index'] for s in copied):
        ss=[s for s in copied if s['canopy_index']==cluster]
        paint=np.zeros((h,w),np.uint8)
        for s in ss:cv2.polylines(paint,[np.rint(s['points']).astype(np.int32)],False,1,max(1,round(.8/scale)))
        safe=cv2.erode(paint,np.ones((3,3),np.uint8))
        linked,n=connect_region(ss,safe,1.,np.zeros(2),scale);greens.extend(linked);joins+=n
    oldcolors=old['workflow']['modules']['color']['strokes'];newcolors=[];inserted=False
    for s in oldcolors:
        if s.get('drawing_region')=='左侧主松树' and s['color']=='green':
            if not inserted:newcolors.extend(greens);inserted=True
        else:newcolors.append(copy.deepcopy(s))
    assert inserted
    from web.execution import prepare
    from artwork_order import ink_payload
    prepare(TARGET);new['ink_job']=ink_payload(old['ink_job'],newink)
    plan=new['workflow'];plan['modules']['outline'].update(strokes=copy.deepcopy(newink),stroke_count=len(newink))
    plan['modules']['color'].update(strokes=newcolors,stroke_count=len(newcolors))
    plan['source_revision']['stroke_data_sha256']=digest(stroke_data(new))
    plan['pine_matched_to_first']=dict(date='2026-10-09',status='preview-only-not-installed',
        reference=str(REFERENCE),reference_sha256=refhash,source_ink_paths=len(refink),source_leaf_paths=len(refcolors),
        canvas_scale=.80,source_anchor=[282.,685.],destination_anchor=[312.,680.],upper_clearance_offset_px=18,root_fitted=True,
        removed_old_pine_ids=removed,retained_rock_suffix=trimmed,
        note='采用第一幅实际松树枝干和叶团轨迹，保持上部结构比例，仅将下段树根接到第二幅原岸石。')
    save(TARGET,new)
    from run_json_painting import load_artwork,manifest_for
    from web.motion_profile import apply_to_artwork,DEFAULTS
    from run_color_experiment import module_lines,reload_stroke_limit
    from color_profile import ColorProfile
    from itertools import groupby
    prepared=apply_to_artwork(load_artwork(TARGET),DEFAULTS);manifest_for(prepared[2],prepared[1],prepared[3])
    module=prepared[1]['modules']['color'];profile=ColorProfile(**module['motion_candidate']).to_dict()
    for c in ('blue','green'):
        _,skipped=module_lines(module,c,profile);assert not skipped
        regions=[key for key,_ in groupby(s['drawing_region'] for s in newcolors if s['color']==c)]
        assert len(regions)==len(set(regions))
    assert reload_stroke_limit(module)==8
    assert all(.2<=v<=.3 for s in newcolors for v in s['press'])
    assert [s for s in oldcolors if s.get('drawing_region')!='左侧主松树']==[s for s in newcolors if s.get('drawing_region')!='左侧主松树']
    assert [s for s in ink if s['role']=='title']==[s for s in newink if s['role']=='title']
    untouched=[s for s in ink if s['id'] not in removed and s['id']!=trimmed['id']]
    assert untouched==[s for s in newink if not s['id'].startswith('creek-first-pine-') and s['id']!=trimmed['id']+'-rock-only']
    assert sha(REFERENCE)==refhash and sha(canonical)==canonhash
    report=dict(offline_web_load_passed=True,hardware_connections=0,canonical_unchanged=True,reference_unchanged=True,
        source_ink_paths=len(refink),source_canopies=len({s['canopy_index'] for s in copied}),
        new_pine_color_contacts=len(greens),additional_safe_joins=joins,
        title_and_shrubs_preserved=True,remaining_scene_preserved_except_tree_root_junction=True,
        candidate_sha256=sha(TARGET),baseline_sha256=sha(BASELINE))
    save(FOLDER/'松树对齐第一幅校验.json',report)
    before=narrow_render(old);after=narrow_render(new);reference=narrow_render(ref)
    make_board([before,after],['上一版：四块平整松冠','修改后：沿用第一幅枝干与错落松冠'],
        '第二幅松岸曲水｜松树按第一幅重做',1800,730).save(FOLDER/'松树对齐第一幅前后对比.jpg',quality=96)
    make_board([reference.crop((205,355,785,710)),before.crop((205,410,730,712)),after.crop((205,410,730,712))],
        ['第一幅松树参考','第二幅上一版','第二幅修改后'],
        '松树风格对照｜使用第一幅实际绘制轨迹',2100,690).save(FOLDER/'第一幅与第二幅松树对照.jpg',quality=96)
    after.save(FOLDER/'候选轨迹预览.png')
    black=copy.deepcopy(new);black['workflow']['modules']['color']['strokes']=[]
    narrow_render(black).save(FOLDER/'新黑线预览.png')
    source=cv2.imdecode(np.fromfile(FOLDER/'参考原图.png',np.uint8),1)
    make_board([aligned_reference(source,new),before,after],['原参考图','上一版候选','松树统一风格候选'],
        '02 松岸曲水｜松树按第一幅调整').save(FOLDER/'原图与轨迹对比.jpg',quality=95)
    root=FOLDER.parent;folders=sorted(p for p in root.iterdir() if p.is_dir() and p.name[:2].isdigit())
    overview=Image.new('RGB',(1500,2465),'white')
    for i,p in enumerate(folders):overview.paste(Image.open(p/'原图与轨迹对比.jpg').resize((1500,493),Image.Resampling.LANCZOS),(0,i*493))
    overview.save(root/'五幅上色对比总览.jpg',quality=95)
    for path in (root/'检查汇总.json',FOLDER/'检查报告.json'):
        audit=read(path);row=audit['artworks'][1] if 'artworks' in audit else audit
        row.update(pine_matched_to_first=report,current_candidate_sha256=sha(TARGET),
            counts={c:sum(s['color']==c for s in newcolors) for c in ('blue','green')},
            prior_coverage_metrics_are_historical=True)
        save(path,audit)
    webaudit=read(root/'网页加载与补蘸校验.json')
    webaudit['artworks'][1].update(pine_matched_to_first=True,web_load=True,
        no_region_revisits=True,reload_limit=8,current_candidate_sha256=sha(TARGET))
    save(root/'网页加载与补蘸校验.json',webaudit)
    notes=root/'说明.md';notes.write_text(notes.read_text(encoding='utf-8').split('\n## 第二幅松树对齐第一幅')[0]+
        '\n## 第二幅松树对齐第一幅\n最新候选采用第一幅实际枝干及9组叶团路径，缩放后衔接第二幅岸石。保留现有横题、删顶排远山和三处灌木，按区域连画及补蘸设置保留。最新校验见松树对齐第一幅校验.json；旧的四松冠/27笔统计属于历史版本。尚未安装到Web正式任务。\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
