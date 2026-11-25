"""User's eight annotations applied to edited creek source, wholly offline."""
from pathlib import Path
import sys,json,copy,hashlib
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from web.catalog import ARTWORKS
from painting_session import save
from artwork_order import ink_payload
from ink_control import refresh_stats
from scripts.consolidate_selected_artworks import digest,stroke_data
from scripts.source_aligned_washes import pigment_masks,ribbon_strokes,join_ribbon_sweeps,dense_path,points_inside
from scripts.three_source_washes import GUIDES,NEAR_RIBBON_WIDTH_FACTOR,FAR_RIBBON_WIDTH_FACTOR,previews
from scripts.order_selected_artworks import ORDERS
from shuimo import press_to_width_mm

SOURCE=ROOT/'参考图'/'松岸曲水_三组灌木_标注下压_20261006.png'
KEEP_TREES={'左侧主松树','左下小树','右下岛上树','近岸右侧小树'}
DROP_TREES={'近岸中间小树','中间岛上树','右上岸边树'}
ANNOTATIONS=[(54.8,13.6,.20,'远山'),(26.1,30.6,.25,'中景轮廓'),(50.5,33.3,.15,'中景皴线'),(24,66.8,.30,'近景轮廓'),(45,73.7,.15,'近景皴线'),(21.5,53,.30,'松树粗干；细枝0.15'),(59.2,78.6,.15,'灌木枝干'),(39.9,56.1,None,'上色贴合原图')]


def pressure_for(s,local,image,pixel_mm):
    role=s['role'];region=s['drawing_region'];n=len(local)
    if role=='title' or role=='water':return s['press']
    if role=='texture':return [.15]*n
    if role=='far_outline':return [.20]*n
    if role=='mountain_outline':return [.25]*n
    if region in KEEP_TREES and region!='左侧主松树':return [.15]*n
    if role=='foliage_outline':return [.15]*n
    if role=='branch' or region=='左侧主松树':
        # Source ink thickness separates the strong trunk from finer branches.
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        dt=cv2.distanceTransform((gray<100).astype(np.uint8),cv2.DIST_L2,5)
        dt=cv2.dilate(dt,np.ones((5,5),np.uint8))
        q=np.rint(local).astype(int);q[:,0]=np.clip(q[:,0],0,image.shape[1]-1);q[:,1]=np.clip(q[:,1],0,image.shape[0]-1)
        width=dt[q[:,1],q[:,0]]*2*pixel_mm
        return np.where(width>=1.6,.30,.15).tolist()
    return [.30]*n if role=='near_outline' else s['press']


def audit(candidate,baseline):
    candidate=Path(candidate);baseline=Path(baseline)
    new=json.loads(candidate.read_text(encoding='utf-8'));old=json.loads(baseline.read_text(encoding='utf-8'))
    revision=new['workflow']['source_revision'];tr=revision['source_transform'];scale=new['ink_job']['meta']['ink_execution']['mm_per_px']
    ratio=tr['scale'];offset=np.array(tr['offset']);image=cv2.imdecode(np.fromfile(str(SOURCE),np.uint8),1)
    originals={s['id']:s for s in stroke_data(old)['black']};ink=stroke_data(new)['black']
    for s in ink:
        prior=originals[s['id']];assert s['points']==prior['points'] and s['speed']==prior['speed']
        local=(np.array(s['points'])-offset)/ratio
        assert s['press']==pressure_for(prior,local,image,scale*ratio)
        assert s['drawing_region'] not in DROP_TREES
    assert {s['id']:s for s in ink if s['role']=='title'}=={s['id']:s for s in originals.values() if s['role']=='title'}
    masks=np.load(candidate.parent/'原图色带禁涂掩膜.npz');samples=0;maximum=0
    for s in stroke_data(new)['color']:
        if s.get('role')!='terrain_wash':continue
        local=(dense_path(s['points'],.05/scale)-offset)/ratio
        assert np.all(points_inside(local,masks['safe']))
        assert np.all(points_inside(local,masks['ribbon_%d'%s['source_ribbon_index']]))
        assert not np.any(points_inside(local,masks['protected']))
        assert s['length_mm']<=50.001;maximum=max(maximum,s['length_mm']);samples+=len(local)
    assert revision['stroke_data_sha256']==digest(stroke_data(new))
    from run_json_painting import load_artwork
    from web.execution import prepare
    prepare(candidate);load_artwork(candidate)
    return dict(robot_connected=False,annotations_applied=True,source=SOURCE.as_posix(),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        pressure_values_mm=sorted(set(v for s in ink if s['role'] not in ('title','water') for v in s['press'])),
        title_and_water_preserved=True,only_requested_tree_groups_retained=True,black_geometry_and_speeds_preserved=True,
        terrain_contact_samples_checked=samples,max_terrain_contact_length_mm=round(maximum,4),source_pigment_and_protected_masks_passed=True,
        counts=dict(black=sum(s['role']!='title' for s in ink),title=sum(s['role']=='title' for s in ink),**dict(Counter(s['color'] for s in stroke_data(new)['color']))),
        runtime_validation_passed=True,actual_brush_width_and_bleed_need_measurement=True)


def main():
    path=ROOT/ARTWORKS[1][3];old=json.loads(path.read_text(encoding='utf-8'));new=copy.deepcopy(old)
    from web.execution import prepare
    prepare(path)
    tr=old['workflow']['source_revision']['source_transform'];ratio=tr['scale'];offset=np.array(tr['offset'])
    scale=old['ink_job']['meta']['ink_execution']['mm_per_px'];pixel_mm=scale*ratio
    image=cv2.imdecode(np.fromfile(str(SOURCE),np.uint8),1);shape=image.shape[:2]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);chroma=np.ptp(image.astype(float),axis=2)
    source_ink=((gray<160)&(chroma<65)).astype(np.uint8)
    source_ink=cv2.dilate(source_ink,np.ones((9,9),np.uint8))
    ink=[]
    for prior in stroke_data(old)['black']:
        if prior['drawing_region'] in DROP_TREES:continue
        s=copy.deepcopy(prior);local=(np.asarray(s['points'])-offset)/ratio
        if s['role'] not in ('title','water') and np.mean(points_inside(local,source_ink))<.30:continue
        s['press']=pressure_for(s,local,image,pixel_mm);ink.append(refresh_stats(s,scale))
    new['ink_job']=ink_payload(old['ink_job'],ink)
    new['workflow']['modules']['outline'].update(strokes=copy.deepcopy(ink),stroke_count=len(ink))
    raw=pigment_masks(image);radius=int(np.ceil((.5+press_to_width_mm(.3)/2)/pixel_mm))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))
    protected=np.zeros(shape,np.uint8)
    for s in ink:
        if s['role'] in ('near_outline','mountain_outline','far_outline','branch','title','foliage_outline'):
            q=np.rint((np.array(s['points'])-offset)/ratio).astype(np.int32);cv2.polylines(protected,[q],False,1,1)
    colors=[copy.deepcopy(s) for s in stroke_data(old)['color'] if s.get('role') not in ('terrain_wash','mountain_color','distant_color') and s['drawing_region'] in KEEP_TREES]
    for s in colors:
        q=np.rint((np.array(s['points'])-offset)/ratio).astype(np.int32);cv2.polylines(protected,[q],False,1,max(1,round(s.get('brush_width_mm_estimate',1)/pixel_mm)))
    protected=cv2.dilate(protected,kernel)
    envelope=cv2.morphologyEx(raw['blue']|raw['green'],cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(41,41)))
    safe=envelope.copy();safe[protected>0]=0
    safe[:radius]=0;safe[-radius:]=0;safe[:,:radius]=0;safe[:,-radius:]=0
    masks=dict(safe=safe,protected=protected,pigment_envelope=envelope);contacts={'blue':np.zeros(shape,np.uint8),'green':np.zeros(shape,np.uint8)}
    rr=int(np.ceil((press_to_width_mm(.30)+.5)/pixel_mm));gap_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*rr+1,2*rr+1))
    constraints=[]
    for j,(region,color,guide,widths) in sorted(enumerate(GUIDES[1]),key=lambda item:('blue','green').index(item[1][1])):
        factor=FAR_RIBBON_WIDTH_FACTOR if '远山' in region else NEAR_RIBBON_WIDTH_FACTOR
        widths=[v*factor for v in widths];allowed=safe.copy()
        if color=='green':allowed[cv2.dilate(contacts['blue'],gap_kernel)>0]=0
        paths,mask=ribbon_strokes(guide,widths,allowed,raw[color],color,region,j,j,scale,ratio,offset)
        paths=join_ribbon_sweeps(paths,mask,scale,ratio,offset);masks['ribbon_%d'%j]=mask
        for k,s in enumerate(paths):
            s['id']='annotated-creek-%02d-%s-%03d'%(j,color,k);s['subject_order']=ORDERS[1].index(region)
            q=np.rint((np.array(s['points'])-offset)/ratio).astype(np.int32);cv2.polylines(contacts[color],[q],False,1,1)
        colors.extend(paths);constraints.append(dict(region=region,color=color,guide=guide,half_width_px=widths,strokes=len(paths)))
    colors.sort(key=lambda s:(('blue','green').index(s['color']),ORDERS[1].index(s['drawing_region']),s.get('source_ribbon_index',-1)))
    plan=new['workflow'];plan['modules']['color'].update(strokes=colors,stroke_count=len(colors))
    plan['fill_report']={c:dict(strokes=sum(s['color']==c for s in colors),coverage_is_model_estimate=True) for c in ('blue','green')}
    revision=plan['source_revision'];revision.pop('black_trace_cleanup',None)
    note='按用户八处标注重整：远山0.20mm、中景轮廓0.25mm、近景轮廓与松树粗干0.30mm，皴线、松树细枝与三组灌木枝干0.15mm；只保留主松树及指定三组灌木。蓝绿上色根据新版原图颜料掩膜和山势重建，中心笔距1.0mm；原题字与水纹保留，题字最后。数值为下压量，笔宽与渗色仍需实测。'
    revision.update(annotated_creek_pressures=True,annotations=ANNOTATIONS,reference_source_file=str(SOURCE),terrain_ribbon_guides=constraints,
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),note=note,stroke_data_sha256=digest(stroke_data(new)))
    plan['note']=note
    pending=path.parent/'待应用_标注下压';pending.mkdir(exist_ok=True);save(pending/'painting.json',new)
    np.savez_compressed(pending/'原图色带禁涂掩膜.npz',**masks)
    result=audit(pending/'painting.json',path);save(pending/'标注下压校验.json',result)
    previews(old,new,pending,'松岸曲水 · 标注下压',black_cleanup=True)
    from PIL import Image,ImageDraw,ImageFont
    from scripts.source_wash_comparison import render
    source=Image.open(SOURCE).convert('RGB');w,h=source.size;actual=render(new).crop((offset[0],offset[1],offset[0]+ratio*w,offset[1]+ratio*h)).resize((w,h),Image.Resampling.LANCZOS)
    board=Image.new('RGB',(2000,820),'#f7f7f0');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',30)
    for i,(pic,label) in enumerate(((source,'新版原图'),(actual,'按标注下压的真实路径预览'))):
        x=35+i*1000;d.text((x,25),label,font=font,fill='#304c3b');pic.thumbnail((930,670));board.paste(pic,(x+(930-pic.width)//2,85+(670-pic.height)//2))
    d.text((35,770),'0.20 / 0.25 / 0.30 / 0.15mm 均为下压量；显示笔宽、颜料浓淡为模型估计。',font=font,fill='#687562');board.save(pending/'新版原图与标注下压对照.png')
    render(new).save(pending/'黑线加蓝绿_实际路径.png')
    blank=copy.deepcopy(new);blank['workflow']['modules']['color']['strokes']=[];render(blank).save(pending/'纯勾线.png')
    (pending/'标注下压说明.md').write_text('# 松岸曲水 · 用户标注下压\n\n'+note+'\n\n'+json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    save(pending/'应用记录.json',dict(baseline_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256((pending/'painting.json').read_bytes()).hexdigest(),applied=False))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
