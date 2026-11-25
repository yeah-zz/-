"""Thin distant outlines, sparse summit color, and uncolored pavilion."""
from pathlib import Path
import sys,json,copy,hashlib
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from painting_session import save
from web.catalog import ARTWORKS
from artwork_order import ink_payload
from ink_control import refresh_stats
from shuimo import press_to_width_mm
from scripts.consolidate_selected_artworks import digest,stroke_data
from scripts.refine_pavilion_mist_20261007 import scan,FAR
from scripts.source_aligned_washes import dense_path,points_inside
from scripts.source_wash_comparison import render
from scripts.order_selected_artworks import ORDERS

def audit(candidate,baseline):
    candidate=Path(candidate);old=json.loads(Path(baseline).read_text(encoding='utf-8'));new=json.loads(candidate.read_text(encoding='utf-8'))
    before=stroke_data(old);after=stroke_data(new)
    assert {s['id']:s for s in after['black'] if s['role']!='far_outline'}=={s['id']:s for s in before['black'] if s['role']!='far_outline'}
    keep=lambda s:s['role'] not in ('mist_distance_wash','roof_wash','summit_accent') and '远山' not in s['drawing_region']
    assert [s for s in after['color'] if s['role']!='summit_accent']==[s for s in before['color'] if keep(s)]
    assert not any(s['drawing_region']=='亭子' for s in after['color'])
    ridges=[s for s in after['black'] if s['role']=='far_outline'];assert len(ridges)==16 and all(set(s['press'])=={.15} for s in ridges)
    masks=np.load(candidate.parent/'山顶少色掩膜.npz');tr=new['workflow']['source_revision']['source_transform'];r=tr['scale'];off=np.array(tr['offset']);scale=new['ink_job']['meta']['ink_execution']['mm_per_px']
    accents=[s for s in after['color'] if s['role']=='summit_accent']
    for s in accents:
        assert set(s['press'])=={.15} and s['length_mm']<=42.001
        assert np.all(points_inside((dense_path(s['points'],.05/scale)-off)/r,masks['safe_%d'%s['mask_index']]))
    assert new['workflow']['source_revision']['stroke_data_sha256']==digest(after)
    from web.execution import prepare
    from run_json_painting import load_artwork
    prepare(candidate);load_artwork(candidate)
    counts=dict(black=sum(s['role']!='title' for s in after['black']),title=sum(s['role']=='title' for s in after['black']),**dict(Counter(s['color'] for s in after['color'])))
    return dict(counts=counts,total_strokes=sum(counts.values()),far_outline_count=len(ridges),summit_color_strokes=len(accents),pavilion_color_strokes=0,near_paths_preserved=True,runtime_validation_passed=True,robot_connected=False)

def main():
    path=ROOT/ARTWORKS[2][3];package=path.parent.parent;old=json.loads(path.read_text(encoding='utf-8'));new=copy.deepcopy(old)
    from web.execution import prepare
    prepare(path)
    prior=json.loads((package/'历史版本/20261007_远山云雾调整前/painting.json').read_text(encoding='utf-8'))
    scale=old['ink_job']['meta']['ink_execution']['mm_per_px'];tr=old['workflow']['source_revision']['source_transform'];r=tr['scale'];off=np.array(tr['offset']);pixel_mm=scale*r
    ink=copy.deepcopy([s for s in stroke_data(old)['black'] if s['role']!='far_outline']);ridges=[]
    for s in stroke_data(prior)['black']:
        if s['role']=='far_outline':
            s=copy.deepcopy(s);s['press']=[.15]*len(s['points']);refresh_stats(s,scale);ridges.append(s)
    ink.extend(ridges);ink.sort(key=lambda s:(s['role']=='title',ORDERS[2].index(s['drawing_region'])))
    new['ink_job']=ink_payload(old['ink_job'],ink);new['workflow']['modules']['outline'].update(strokes=copy.deepcopy(ink),stroke_count=len(ink))
    colors=[copy.deepcopy(s) for s in stroke_data(old)['color'] if s['role'] not in ('mist_distance_wash','roof_wash','summit_accent') and '远山' not in s['drawing_region']]
    masks=np.load(path.parent/'远山云雾覆盖掩膜.npz');safe_masks={};created=[]
    for i,(region,_) in enumerate(FAR):
        safe=masks['safe_%d'%i].copy();band=np.zeros_like(safe)
        for x in range(safe.shape[1]):
            ys=np.flatnonzero(safe[:,x])
            if len(ys):band[ys[0]:ys[0]+10,x]=safe[ys[0]:ys[0]+10,x]
        safe_masks['safe_%d'%i]=band
        candidates=[line for line in scan(band,pixel_mm,press=.15) if np.linalg.norm(line[-1]-line[0])*pixel_mm>=1.5]
        # A few accents near the upper ridge, rather than filling every row.
        ys=np.flatnonzero(band.any(axis=1));ceiling=float(np.quantile(ys,.45)) if len(ys) else 0
        candidates=[line for line in candidates if line[0,1]<=ceiling]
        candidates=sorted(candidates,key=lambda line:-np.linalg.norm(line[-1]-line[0]))[:3]
        for j,line in enumerate(candidates):
            q=dense_path(line,.25/pixel_mm)*r+off;n=len(q)
            s=dict(id='summit-accent-%d-%d'%(i,j),color='blue',role='summit_accent',drawing_region=region,subject_order=ORDERS[2].index(region),mask_index=i,region_id='summit-%d'%i,points=q.tolist(),press=[.15]*n,speed=[40.]*n,brush_width_mm_estimate=press_to_width_mm(.15),target_width_mm=[press_to_width_mm(.15)]*n,target_opacity=[.10]*n,pressure_mm=None,pigment_recipe_calibrated=False,recipe='sparse-summit-optical-target',layer=0,path_style='short-summit-accent')
            created.append(refresh_stats(s,scale))
    colors.extend(created);colors.sort(key=lambda s:(('blue','green').index(s['color']),ORDERS[2].index(s['drawing_region'])))
    plan=new['workflow'];plan['modules']['color'].update(strokes=colors,stroke_count=len(colors));plan['fill_report']={c:dict(strokes=sum(s['color']==c for s in colors),coverage_is_model_estimate=True) for c in ('blue','green')}
    rev=plan['source_revision'];note='按用户最新要求：远山恢复细勾线，下压0.15mm；取消整片远山铺色，仅山顶短色带下压0.15mm；亭子不涂色。近景既有路径及下压量保持。预览按现有笔宽模型绘制，下压量不等于纸上笔宽，色阶尚未实测。'
    rev.update(distance_mist_refinement=False,distance_summit_only=True,pavilion_uncolored=True,after_counts=dict(black=len(ink),color=len(colors)));rev['note']=note;plan['note']=note;rev['stroke_data_sha256']=digest(stroke_data(new))
    pending=path.parent/'待应用_远山细勾';pending.mkdir(exist_ok=True);save(pending/'painting.json',new);np.savez_compressed(pending/'山顶少色掩膜.npz',**safe_masks)
    result=audit(pending/'painting.json',path);save(pending/'远山细勾校验.json',result)
    groups=defaultdict(list)
    for s in ink+colors:groups[(s['drawing_region'],s['role'],s.get('color','black'))].append(s)
    lines=['# 云山亭阶当前下压量','',note,'','| 部位 | 类型 | 颜色 | 笔数 | 下压量 mm | 模型笔宽 mm |','|---|---|---|---:|---|---|']
    for (region,role,color),ss in groups.items():
        ps=sorted({round(v,3) for s in ss for v in s['press']});lines.append('| %s | %s | %s | %d | %s | %s |'%(region,role,color,len(ss),' / '.join('%.2f'%p for p in ps),' / '.join('%.2f'%press_to_width_mm(p) for p in ps)))
    (pending/'各部位下压量.md').write_text('\n'.join(lines),encoding='utf-8')
    for name in ('任务预览_线宽色彩为估计.png','黑线加蓝绿_实际路径.png'):render(new).save(pending/name)
    black=copy.deepcopy(new);black['workflow']['modules']['color']['strokes']=[];render(black).save(pending/'纯勾线.png')
    source=Image.open(package/'参考原图.png').convert('RGB');w,h=source.size;pic=render(new,True).crop((*off,off[0]+r*w,off[1]+r*h)).resize((w,h),Image.Resampling.LANCZOS)
    board=Image.new('RGB',(2000,1040),'#f7f7f0');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',27)
    for i,(im,label) in enumerate(((source,'参考原图'),(pic,'远山细勾 · 山顶少色 · 亭子留白'))):
        x=35+i*1000;draw.text((x,25),label,font=font,fill='#304c3b');im.thumbnail((930,650));board.paste(im,(x,130))
    labels=['远山勾线 0.15；山顶少量蓝色 0.15','近景外轮廓 0.30；右侧山石外轮廓 0.25','皴线、亭子、台阶、细枝 0.15；粗枝 0.30','山石原铺色 0.30 / 补色 0.20；叶团 0.19','水纹 0.10；题字 0.15；亭子上色 0 笔','以上单位 mm，均为下压量。笔宽和浓淡为模型估计，尚未纸上标定。']
    for j,label in enumerate(labels):draw.text((35,725+j*47),label,font=font,fill='#304c3b')
    board.save(pending/'远山细勾与下压量对照.png')
    save(pending/'应用记录.json',dict(baseline_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256((pending/'painting.json').read_bytes()).hexdigest(),applied=False))
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
