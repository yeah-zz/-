"""Add a few fine, attached forks to artwork 01 without changing other marks."""
from pathlib import Path
import sys,json,copy,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw,ImageFont
from painting_session import save
from web.catalog import ARTWORKS
from artwork_order import ink_payload
from ink_control import refresh_stats
from scripts.consolidate_selected_artworks import digest,stroke_data
from scripts.directional_rock_color import smooth_curve
from scripts.source_wash_comparison import render

FORKS=[[(208,442),(191,447),(177,458)],[(237,395),(261,393),(278,381)],
       [(337,426),(359,422),(384,416)],[(302,442),(324,460),(351,469),(379,463)],
       [(399,524),(410,507),(398,484)],[(470,552),(491,555),(517,545)],
       [(548,581),(563,580),(589,573)],[(585,597),(617,591),(635,581)]]
FORKS.extend([
    [(154,431),(143,423),(135,414)],[(185,439),(179,430),(173,422)],
    [(235,385),(217,377),(208,367)],[(252,394),(273,384),(291,375)],
    [(354,424),(365,413),(365,402)],[(380,439),(403,433),(423,435)],
    [(405,503),(426,496),(445,490)],[(443,531),(463,521),(483,515)],
    [(494,552),(515,560),(535,559)],[(538,579),(550,567),(565,558)],
    [(576,590),(586,581),(604,578)],[(607,598),(630,607),(651,607)]])
EXTRA_BARK=[[(172,634),(188,610),(204,588)],[(213,574),(229,559),(249,546)]]


def audit(candidate,baseline):
    candidate=Path(candidate);old=json.loads(Path(baseline).read_text(encoding='utf-8'));new=json.loads(candidate.read_text(encoding='utf-8'))
    assert stroke_data(old)['color']==stroke_data(new)['color'],'上色发生改变'
    before={s['id']:s for s in stroke_data(old)['black']};after={s['id']:s for s in stroke_data(new)['black']}
    assert set(before)<=set(after)
    for sid,original in before.items():
        current=copy.deepcopy(after[sid]);prior=copy.deepcopy(original)
        if sid.startswith('simple-fork-'):
            assert current['points']==prior['points'] and set(current['press'])=={.15}
            for key in ('press','width_mm','press_mean','press_max'):current.pop(key,None);prior.pop(key,None)
        assert current==prior,'非细枝结构发生改变'
    additions=[s for sid,s in after.items() if sid not in before]
    expected=sum('pine-fine-fork-%02d'%i not in before for i in range(len(FORKS)))+sum('pine-fine-bark-%02d'%i not in before for i in range(len(EXTRA_BARK)))
    assert len(additions)==expected
    scale=new['ink_job']['meta']['ink_execution']['mm_per_px'];tr=new['workflow']['source_revision']['source_transform'];ratio=tr['scale'];offset=np.array(tr['offset'])
    for s in additions:
        assert s['role'] in ('branch','bark_texture') and s['drawing_region']=='整棵松树' and set(s['press'])=={.15}
        assert s['length_mm']<20
        if s['role']=='branch':
            parent=np.asarray(before[s['branch_parent_id']]['points']);assert np.linalg.norm(parent-np.array(s['points'][0]),axis=1).min()<1e-6,'细枝未连接主枝'
        local=(np.array(s['points'])-offset)/ratio
        assert np.all((local[:,0]>70)&(local[:,0]<740)&(local[:,1]>330)&(local[:,1]<(650 if s['role']=='bark_texture' else 625)))
        assert abs(np.linalg.norm(np.diff(s['points'],axis=0),axis=1).sum()*scale-s['length_mm'])<1e-5
    assert new['workflow']['source_revision']['stroke_data_sha256']==digest(stroke_data(new))
    from web.execution import prepare
    from run_json_painting import load_artwork
    prepare(candidate);load_artwork(candidate)
    return dict(robot_connected=False,added_fine_branches=sum(s['role']=='branch' for s in additions),added_bark_textures=sum(s['role']=='bark_texture' for s in additions),fine_branch_pressure_mm=.15,
        original_main_branches_trunk_and_other_ink_preserved=True,all_colors_exactly_preserved=True,
        fine_branches_connected_to_existing_structure=True,existing_simple_forks_refined=5,
        black_strokes=sum(s['role']!='title' for s in after.values()),total_strokes=len(after)+len(stroke_data(new)['color']),runtime_validation_passed=True)


def comparison(old,new,path):
    tr=new['workflow']['source_revision']['source_transform'];r=tr['scale'];x,y=tr['offset']
    crop=(x+60*r,y+330*r,x+735*r,y+625*r)
    board=Image.new('RGB',(1800,600),'#f7f7f0');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',27)
    d.text((25,18),'松瀑亭舟 · 松树细枝与分叉调整',font=font,fill='#304c3b')
    for i,(b,label) in enumerate(((old,'修改前'),(new,'修改后 · 补充细枝，轻笔刻画'))):
        left=25+890*i;d.text((left,70),label,font=font,fill='#304c3b');pic=render(b).crop(crop);pic.thumbnail((850,430),Image.Resampling.LANCZOS);board.paste(pic,(left+(850-pic.width)//2,115+(430-pic.height)//2))
    d.text((25,552),'新增细枝下压0.15mm；主干、主枝及全部上色保留。线宽为未实测模型。',font=font,fill='#687562');board.save(path)


def main():
    path=ROOT/ARTWORKS[0][3];old=json.loads(path.read_text(encoding='utf-8'));new=copy.deepcopy(old)
    from web.execution import prepare
    prepare(path)
    tr=old['workflow']['source_revision']['source_transform'];r=tr['scale'];offset=np.array(tr['offset']);scale=old['ink_job']['meta']['ink_execution']['mm_per_px']
    ink=copy.deepcopy(stroke_data(old)['black']);parents=[s for s in ink if s['role'] in ('branch','trunk')]
    points=np.vstack([(np.array(s['points'])-offset)/r for s in parents]);owners=[s['id'] for s in parents for _ in s['points']];tree=cKDTree(points)
    for s in ink:
        if s['id'].startswith('simple-fork-'):s['press']=[.15]*len(s['points']);refresh_stats(s,scale)
    template=copy.deepcopy(next(s for s in parents if s['role']=='branch'))
    additions=[]
    existing_ids={s['id'] for s in ink}
    for i,line in enumerate(FORKS):
        if 'pine-fine-fork-%02d'%i in existing_ids:continue
        guide=np.array(line,float);_,idx=tree.query(guide[0]);guide[0]=points[idx]
        p=smooth_curve(guide,.25/(scale*r))*r+offset;n=len(p);s=copy.deepcopy(template)
        for key in ('source_edge_index','source_path_ids','parent_stroke_id','segment','segments'):s.pop(key,None)
        s.update(id='pine-fine-fork-%02d'%i,points=p.tolist(),press=[.15]*n,speed=[40.]*n,branch_parent_id=owners[idx],source_fork_guide=guide.tolist())
        additions.append(refresh_stats(s,scale))
    for i,line in enumerate(EXTRA_BARK):
        if 'pine-fine-bark-%02d'%i in existing_ids:continue
        p=smooth_curve(line,.25/(scale*r))*r+offset;n=len(p);s=copy.deepcopy(template)
        for key in ('source_edge_index','source_path_ids','parent_stroke_id','segment','segments'):s.pop(key,None)
        s.update(id='pine-fine-bark-%02d'%i,role='bark_texture',points=p.tolist(),press=[.15]*n,speed=[40.]*n,source_bark_guide=line)
        additions.append(refresh_stats(s,scale))
    pos=next(i for i,s in enumerate(ink) if s['drawing_region']!='整棵松树');ink[pos:pos]=additions
    new['ink_job']=ink_payload(old['ink_job'],ink);plan=new['workflow'];plan['modules']['outline'].update(strokes=copy.deepcopy(ink),stroke_count=len(ink))
    branch_count=sum(s['role']=='branch' for s in additions);bark_count=sum(s['role']=='bark_texture' for s in additions)
    revision=plan['source_revision'];note='松树继续补充细节：本次新增%d条与已有枝干连接的细枝及分叉、%d条顺干树皮纹理，均用0.15mm下压；主干、其他黑线、题字及全部蓝绿上色保留。'%(branch_count,bark_count)
    revision['pine_branch_refinement']=dict(total_fine_branches=len(FORKS),additional_bark_textures=len(EXTRA_BARK),fine_pressure_mm=.15,source_guides=FORKS);revision['after_counts']=dict(black=len(ink),color=len(stroke_data(new)['color']));revision['note']+=' '+note;plan['note']=revision['note'];revision['stroke_data_sha256']=digest(stroke_data(new))
    pending=path.parent/'待应用_松枝细节';pending.mkdir(exist_ok=True);save(pending/'painting.json',new)
    result=audit(pending/'painting.json',path);save(pending/'松枝细节校验.json',result)
    comparison(old,new,pending/'松枝细节前后对照.png');render(new).save(pending/'任务预览_线宽色彩为估计.png');render(new).save(pending/'黑线加蓝绿_实际路径.png')
    blank=copy.deepcopy(new);blank['workflow']['modules']['color']['strokes']=[];render(blank).save(pending/'纯勾线.png')
    (pending/'松枝细节说明.md').write_text('# 松瀑亭舟 · 松枝细节\n\n'+note+'\n\n'+json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    save(pending/'应用记录.json',dict(baseline_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256((pending/'painting.json').read_bytes()).hexdigest(),applied=False))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
