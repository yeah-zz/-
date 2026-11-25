"""Replace sparse leaf strips by overlapping source-mask fills, offline."""
from pathlib import Path
import sys,json,copy,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from web.catalog import ARTWORKS
from painting_session import save
from ink_control import refresh_stats
from shuimo import press_to_width_mm
from scripts.consolidate_selected_artworks import digest,stroke_data
from scripts.source_aligned_washes import dense_path,points_inside
from scripts.source_wash_comparison import render
from scripts.order_selected_artworks import ORDERS

PADS=[('左侧主松树',(88,442,243,500)),('左侧主松树',(205,393,410,466)),
      ('左侧主松树',(374,466,559,539)),('左侧主松树',(512,544,683,615)),
      ('左下小树',(32,641,151,717)),('右下岛上树',(978,494,1138,557)),
      ('近岸右侧小树',(799,776,979,836))]
PRESS=.19
PITCH_MM=.70


def source_comparison(bundle,source,destination):
    tr=bundle['workflow']['source_revision']['source_transform'];ratio=tr['scale'];x,y=tr['offset']
    original=Image.open(source).convert('RGB');w,h=original.size
    actual=render(bundle).crop((x,y,x+ratio*w,y+ratio*h)).resize((w,h),Image.Resampling.LANCZOS)
    board=Image.new('RGB',(2000,820),'#f7f7f0');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',30)
    for i,(pic,label) in enumerate(((original,'参考原图'),(actual,'调整后的实际路径预览'))):
        left=35+i*1000;draw.text((left,25),label,font=font,fill='#304c3b');pic.thumbnail((930,670));board.paste(pic,(left+(930-pic.width)//2,85+(670-pic.height)//2))
    pitch=bundle['workflow']['color_geometry'].get('canopy_pitch_mm',.6362)
    draw.text((35,770),'叶团填色下压0.19mm、笔距约%.2fmm；笔宽、颜色和纸上渗色为模型估计。'%pitch,font=font,fill='#687562')
    board.save(destination/'新版原图与标注下压对照.png')


def scan(mask,pixel_mm):
    step=max(1,int(np.floor(PITCH_MM/pixel_mm)));ys=np.where(mask)[0]
    turn_mask=cv2.erode(mask,np.ones((3,3),np.uint8))
    if not len(ys):return []
    rows=list(range(int(ys.min()),int(ys.max())+1,step))
    if rows[-1]!=int(ys.max()):rows.append(int(ys.max()))
    result=[];current=[]
    def length(p):return np.linalg.norm(np.diff(p,axis=0),axis=1).sum()*pixel_mm if len(p)>1 else 0
    def flush():
        nonlocal current
        if len(current)>1:result.append(np.array(current,float))
        current=[]
    for row,y in enumerate(rows):
        xs=np.where(mask[y])[0]
        if not len(xs):flush();continue
        runs=np.split(xs,np.where(np.diff(xs)>1)[0]+1)
        if row%2:runs=runs[::-1]
        for run in runs:
            if len(run)<2:continue
            a=np.array([run[0],y],float);b=np.array([run[-1],y],float)
            if row%2:a,b=b,a
            if current:
                connection=dense_path([current[-1],a],.05/pixel_mm)
                if not np.all(points_inside(connection,turn_mask)) or length(current+[a,b])>48:flush()
            current.extend([a,b])
    flush();return result


def area(strokes,shape,scale,ratio,offset):
    canvas=np.zeros(shape,np.uint8)
    for s in strokes:
        q=np.rint((np.asarray(s['points'])-offset)/ratio).astype(np.int32)
        cv2.polylines(canvas,[q],False,1,max(1,round(s['brush_width_mm_estimate']/(scale*ratio))))
    return canvas


def audit(candidate,baseline):
    candidate=Path(candidate);baseline=Path(baseline)
    new=json.loads(candidate.read_text(encoding='utf-8'));old=json.loads(baseline.read_text(encoding='utf-8'))
    assert new['ink_job']==old['ink_job'],'黑线、下压或执行点改变'
    nonleaf=lambda b:[s for s in stroke_data(b)['color'] if s.get('role') not in ('leaf_color','canopy_fill')]
    assert nonleaf(new)==nonleaf(old),'山石颜色改变'
    tr=new['workflow']['source_revision']['source_transform'];ratio=tr['scale'];offset=np.array(tr['offset']);scale=new['ink_job']['meta']['ink_execution']['mm_per_px']
    masks=np.load(candidate.parent/'叶团填色掩膜.npz');shape=masks['leaf_0'].shape
    before=[s for s in stroke_data(old)['color'] if s.get('role') in ('leaf_color','canopy_fill')]
    after=[s for s in stroke_data(new)['color'] if s.get('role')=='canopy_fill']
    # Black contours and the retained forks are part of the visible leaf mass.
    black=np.zeros(shape,np.uint8)
    for s in stroke_data(new)['black']:
        q=np.rint((np.asarray(s['points'])-offset)/ratio).astype(np.int32)
        cv2.polylines(black,[q],False,1,max(1,round(s['width_mm']/(scale*ratio))))
    samples=0;maximum=0;coverage=[]
    for s in after:
        assert set(s['press'])=={PRESS} and s['length_mm']<=48.001
        local=(dense_path(s['points'],.05/scale)-offset)/ratio
        assert np.all(points_inside(local,masks['safe_%d'%s['canopy_index']]))
        samples+=len(local);maximum=max(maximum,s['length_mm'])
    target=np.zeros(shape,np.uint8)
    for i in range(len(PADS)):
        leaf=masks['leaf_%d'%i];target|=leaf
        visible=area([s for s in after if s['canopy_index']==i],shape,scale,ratio,offset)
        paintable=leaf&(1-black)
        fraction=float((visible&paintable).sum()/max(1,paintable.sum()))
        assert fraction>.82,'叶团模型覆盖不足'
        coverage.append(dict(pad=i,region=PADS[i][0],estimated_coverage_excluding_black_structure=round(fraction,4)))
    old_visible=area(before,shape,scale,ratio,offset);new_visible=area(after,shape,scale,ratio,offset)
    target&=1-black
    before_ratio=float((old_visible&target).sum()/target.sum());after_ratio=float((new_visible&target).sum()/target.sum())
    assert after_ratio>before_ratio+.1
    assert new['workflow']['source_revision']['stroke_data_sha256']==digest(stroke_data(new))
    from web.execution import prepare
    from run_json_painting import load_artwork
    prepare(candidate);load_artwork(candidate)
    return dict(robot_connected=False,black_and_pressures_exactly_preserved=True,terrain_colors_exactly_preserved=True,
        source_leaf_groups=len(PADS),leaf_pressure_mm=PRESS,actual_pitch_mm=round(max(1,int(np.floor(PITCH_MM/(scale*ratio))))*scale*ratio,4),
        estimated_brush_width_mm=press_to_width_mm(PRESS),estimated_coverage_before=round(before_ratio,4),estimated_coverage_after=round(after_ratio,4),
        coverage_by_pad=coverage,contact_samples_checked=samples,max_contact_length_mm=round(maximum,4),leaf_strokes_before=len(before),leaf_strokes_after=len(after),
        dense_segments_and_connectors_in_masks=True,runtime_validation_passed=True,actual_brush_width_and_bleed_need_measurement=True)


def main():
    path=ROOT/ARTWORKS[1][3];old=json.loads(path.read_text(encoding='utf-8'));new=copy.deepcopy(old)
    tr=old['workflow']['source_revision']['source_transform'];ratio=tr['scale'];offset=np.array(tr['offset']);scale=old['ink_job']['meta']['ink_execution']['mm_per_px'];pixel_mm=scale*ratio
    source=path.parent.parent/'参考原图.png';image=cv2.imdecode(np.fromfile(str(source),np.uint8),1);shape=image.shape[:2]
    sm=cv2.GaussianBlur(image.astype(float),(0,0),1);b,g,r=sm.transpose(2,0,1)
    pigment=((g-r>12)&(b-r>8)&(b-g<22)&(g<195)).astype(np.uint8)
    radius=int(np.ceil((press_to_width_mm(PRESS)/2)/pixel_mm))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    colors=[copy.deepcopy(s) for s in stroke_data(old)['color'] if s.get('role') not in ('leaf_color','canopy_fill')]
    masks={};leaves=[]
    for i,(region,(x0,y0,x1,y1)) in enumerate(PADS):
        part=np.zeros(shape,np.uint8);part[y0:y1,x0:x1]=pigment[y0:y1,x0:x1]
        part=cv2.morphologyEx(part,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        contours,_=cv2.findContours(part,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);leaf=np.zeros(shape,np.uint8)
        for co in contours:
            if cv2.contourArea(co)>75:cv2.drawContours(leaf,[co],-1,1,-1)
        safe=cv2.erode(leaf,kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        # Retain the few original dark forks within each leaf mass.
        branch=((image.max(axis=2)<75)&(leaf>0)).astype(np.uint8)
        # Keep the lower structural forks visible, while letting leaf pigment
        # cover the numerous fine crown twigs rather than leaving broad holes.
        branch[:int(y0+(y1-y0)*.60)]=0
        safe[cv2.dilate(branch,np.ones((3,3),np.uint8))>0]=0
        masks['leaf_%d'%i]=leaf;masks['safe_%d'%i]=safe
        for j,line in enumerate(scan(safe,pixel_mm)):
            p=dense_path(line,.25/pixel_mm)*ratio+offset;n=len(p)
            s=dict(id='creek-solid-canopy-%02d-%03d'%(i,j),color='green',role='canopy_fill',drawing_region=region,
                subject_order=ORDERS[1].index(region),canopy_index=i,region_id='creek-canopy-%d'%i,
                points=p.tolist(),press=[PRESS]*n,speed=[40.]*n,recipe='source-mask-overlapping-canopy',layer=0,
                target_opacity=[.45]*n,target_width_mm=[press_to_width_mm(PRESS)]*n,brush_width_mm_estimate=press_to_width_mm(PRESS),
                pressure_mm=None,pigment_recipe_calibrated=False,path_style='in-mask-overlapping-serpentine')
            leaves.append(refresh_stats(s,scale))
    colors.extend(leaves);colors.sort(key=lambda s:(('blue','green').index(s['color']),ORDERS[1].index(s['drawing_region']),s.get('canopy_index',-1),s.get('source_ribbon_index',-1)))
    plan=new['workflow'];plan['modules']['color'].update(strokes=colors,stroke_count=len(colors))
    revision=plan['source_revision'];revision['canopy_gap_repair']=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        pads=PADS,pressure_mm=PRESS,pitch_target_mm=PITCH_MM,black_and_terrain_preserved=True)
    note='叶团补满修复：松冠与三组灌木按新版原图叶团外形重建重叠铺色；填色下压仍为0.19mm，中心笔距约0.64mm，保留少量枝干留白。黑线下压标注与山石蓝绿路径均保持不变。'
    revision['note']=revision['note']+' '+note;plan['note']=revision['note'];revision['stroke_data_sha256']=digest(stroke_data(new))
    plan['color_geometry'].update(canopy_pitch_mm=max(1,int(np.floor(PITCH_MM/pixel_mm)))*pixel_mm,leaf_pressure_mm=PRESS)
    plan['fill_report']={c:dict(strokes=sum(s['color']==c for s in colors),coverage_is_model_estimate=True) for c in ('blue','green')}
    pending=path.parent/'待应用_叶团补满';pending.mkdir(exist_ok=True);save(pending/'painting.json',new);np.savez_compressed(pending/'叶团填色掩膜.npz',**masks)
    result=audit(pending/'painting.json',path);save(pending/'叶团补满校验.json',result)
    from scripts.three_source_washes import previews
    previews(old,new,pending,'松岸曲水',canopy_fill=True)
    render(new).save(pending/'黑线加蓝绿_实际路径.png')
    source_comparison(new,source,pending)
    (pending/'叶团补满说明.md').write_text('# 松岸曲水 · 叶团补满\n\n'+note+'\n\n'+json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    save(pending/'应用记录.json',dict(baseline_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256((pending/'painting.json').read_bytes()).hexdigest(),applied=False))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
