"""Preview-only creek composition: remove the upper distant band, horizontal title."""
from pathlib import Path
import sys,copy,json,socket
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from PIL import Image
from ink_control import refresh_stats
from scripts.consolidate_selected_artworks import digest,stroke_data
from scripts.creek_open_branches_20261009 import FOLDER,TARGET,read,save,sha
from scripts.preview_color_coverage_20261009 import narrow_render,make_board,aligned_reference

BASELINE=FOLDER/'历史版本/删顶排远山横题前/painting.json'
FAR={'左侧远山','中间远山','右侧远山'}

def main():
    def offline(*a,**k):raise AssertionError('Offline candidate only')
    socket.socket.connect=offline;socket.socket.connect_ex=offline
    if not BASELINE.exists():
        BASELINE.parent.mkdir(parents=True,exist_ok=True);BASELINE.write_bytes(TARGET.read_bytes())
    old=read(BASELINE);new=copy.deepcopy(old)
    canonical=ROOT/'待画作品/02_松岸曲水_竖题山河绘_绘字修复版/任务资料/painting.json'
    canonical_hash=sha(canonical)
    font_path=next((ROOT/'待画作品').glob('04*/任务资料/painting.json'))
    font=read(font_path);font_hash=sha(font_path)
    ink=old['ink_job']['meta']['ink_execution']['base_strokes']
    far_ink=[s for s in ink if s.get('drawing_region') in FAR]
    body=[copy.deepcopy(s) for s in ink if s.get('drawing_region') not in FAR and s.get('role')!='title']
    colors=old['workflow']['modules']['color']['strokes']
    far_color=[s for s in colors if s.get('drawing_region') in FAR]
    newcolor=[copy.deepcopy(s) for s in colors if s.get('drawing_region') not in FAR]
    source_titles=[s for s in font['ink_job']['meta']['ink_execution']['base_strokes'] if s['role']=='title']
    source_scale=font['ink_job']['meta']['ink_execution']['mm_per_px']
    scale=old['ink_job']['meta']['ink_execution']['mm_per_px'];factor=source_scale/scale
    allpoints=np.vstack([s['points'] for s in source_titles]);center=(allpoints.min(0)+allpoints.max(0))/2
    # The body starts at y=322; the title's lower edge is at y=292. Centered
    # within the paper, with the same physical size as artwork 4.
    destination=np.array([768.,226.]);titles=[]
    for i,prior in enumerate(source_titles):
        s=copy.deepcopy(prior)
        s['points']=((np.asarray(s['points'])-center)*factor+destination).tolist()
        s.update(id=f'creek-horizontal-title-{i:02d}',drawing_region='题字',
                 subject_order=999,reference_title_stroke_id=prior['id'])
        # Parent IDs must not point at the fourth artwork's active title manifest.
        if 'parent_stroke_id' in s:s['parent_stroke_id']='creek-horizontal-'+str(s['parent_stroke_id'])
        titles.append(refresh_stats(s,scale))
    output=body+titles
    from web.execution import prepare
    from artwork_order import ink_payload
    prepare(TARGET);new['ink_job']=ink_payload(old['ink_job'],output)
    plan=new['workflow'];plan['modules']['outline'].update(strokes=copy.deepcopy(output),stroke_count=len(output))
    plan['modules']['color'].update(strokes=newcolor,stroke_count=len(newcolor))
    plan['execution_order']['subjects']=[name for name in plan['execution_order']['subjects'] if name not in FAR]
    plan['execution_order']['title_last']=True
    plan['source_revision']['stroke_data_sha256']=digest(stroke_data(new))
    q=np.vstack([s['points'] for s in titles]);bodypoints=np.vstack([s['points'] for s in body])
    size_mm=(q.max(0)-q.min(0))*scale
    plan['horizontal_title_revision']=dict(date='2026-10-09',status='preview-only-not-installed',
        removed_ink_ids=[s['id'] for s in far_ink],removed_color_ids=[s['id'] for s in far_color],
        reference_title_file=str(font_path),reference_title_sha256=font_hash,
        title_size_mm=size_mm.tolist(),title_center_canvas=destination.tolist(),
        note='删除最上面一排远山及其上色；采用第四幅同等实际尺寸的横排山河绘，置于上方居中，仍最后题字。')
    save(TARGET,new)
    from run_json_painting import load_artwork,manifest_for
    from web.motion_profile import apply_to_artwork,DEFAULTS
    from run_color_experiment import module_lines,reload_stroke_limit
    from color_profile import ColorProfile
    from itertools import groupby
    prepared=apply_to_artwork(load_artwork(TARGET),DEFAULTS)
    manifest_for(prepared[2],prepared[1],prepared[3])
    module=prepared[1]['modules']['color'];profile=ColorProfile(**module['motion_candidate']).to_dict()
    for c in ('blue','green'):
        _,skipped=module_lines(module,c,profile);assert not skipped
        names=[name for name,_ in groupby(s['drawing_region'] for s in newcolor if s['color']==c)]
        assert len(names)==len(set(names))
    assert reload_stroke_limit(module)==8
    assert body==[s for s in ink if s.get('drawing_region') not in FAR and s.get('role')!='title']
    assert newcolor==[s for s in colors if s.get('drawing_region') not in FAR]
    assert all(s['role']=='title' for s in output[-len(titles):])
    assert np.allclose(size_mm,(allpoints.max(0)-allpoints.min(0))*source_scale)
    assert bodypoints[:,1].min()-q[:,1].max()>20
    assert len({s['id'] for s in output})==len(output)
    assert sha(canonical)==canonical_hash and sha(font_path)==font_hash
    report=dict(offline_web_load_passed=True,canonical_unchanged=True,hardware_connections=0,
        removed_top_band_ink=len(far_ink),removed_top_band_colors=len(far_color),
        title_reference='第四幅',title_size_mm=np.round(size_mm,2).tolist(),
        body_vertical_gap_mm=round((bodypoints[:,1].min()-q[:,1].max())*scale,2),
        title_last=True,remaining_body_and_colors_exactly_preserved=True,
        candidate_sha256=sha(TARGET),baseline_sha256=sha(BASELINE))
    save(FOLDER/'横题与删远山校验.json',report)
    before=narrow_render(old);after=narrow_render(new)
    make_board([before,after],['修改前：顶排远山＋右侧竖题','修改后：上方横题，保留下方山景'],
        '第二幅松岸曲水｜删顶排远山，横排题字',1800,730).save(FOLDER/'删远山横题前后对比.jpg',quality=96)
    after.save(FOLDER/'候选轨迹预览.png')
    black=copy.deepcopy(new);black['workflow']['modules']['color']['strokes']=[]
    narrow_render(black).save(FOLDER/'新黑线预览.png')
    source=cv2.imdecode(np.fromfile(FOLDER/'参考原图.png',np.uint8),1)
    make_board([aligned_reference(source,new),before,after],['原参考图','上一版候选','横题构图候选'],
        '02 松岸曲水｜上方横题构图').save(FOLDER/'原图与轨迹对比.jpg',quality=95)
    root=FOLDER.parent;folders=sorted(p for p in root.iterdir() if p.is_dir() and p.name[:2].isdigit())
    overview=Image.new('RGB',(1500,2465),'white')
    for i,p in enumerate(folders):overview.paste(Image.open(p/'原图与轨迹对比.jpg').resize((1500,493),Image.Resampling.LANCZOS),(0,i*493))
    overview.save(root/'五幅上色对比总览.jpg',quality=95)
    for path in (root/'检查汇总.json',FOLDER/'检查报告.json'):
        audit=read(path);row=audit['artworks'][1] if 'artworks' in audit else audit
        row.update(horizontal_title_revision=report,counts={c:sum(s['color']==c for s in newcolor) for c in ('blue','green')},
            title_exactly_preserved=False,current_candidate_sha256=sha(TARGET),prior_coverage_metrics_are_historical=True)
        save(path,audit)
    webaudit=read(root/'网页加载与补蘸校验.json')
    webaudit['artworks'][1].update(title_unchanged=False,horizontal_title_revision=True,web_load=True,
        no_region_revisits=True,reload_limit=8,current_candidate_sha256=sha(TARGET))
    save(root/'网页加载与补蘸校验.json',webaudit)
    notes=root/'说明.md'
    notes.write_text(notes.read_text(encoding='utf-8').split('\n## 第二幅横题构图')[0]+
        '\n## 第二幅横题构图\n删除顶部三组远山区域的23条黑线及21条上色路径；横排题字采用第四幅相同物理尺寸（约80.3×25.5mm），在画布上方居中。其余山景、最新整理的松树和灌木、补蘸设置保持不变，仍最后题字。最新校验见横题与删远山校验.json。尚未替换Web正式任务。\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
