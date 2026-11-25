"""Apply the reviewed source-faithful tree and overlapping color style to 03."""
from pathlib import Path
import sys,json,copy,hashlib
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from original_d_tracer import LandscapeBrushGenerator
from scipy.ndimage import gaussian_filter
from painting_session import save
from artwork_order import ink_payload
from ink_control import refresh_stats
from shuimo import press_to_width_mm
from web.catalog import ARTWORKS
from scripts.consolidate_selected_artworks import digest,stroke_data
from scripts.source_aligned_washes import dense_path,points_inside,ribbon_strokes,join_ribbon_sweeps
from scripts.source_faithful_pine_branches import merge
from scripts.repair_creek_canopy_fill import scan
from scripts.three_source_washes import GUIDES,previews,box
from scripts.order_selected_artworks import ORDERS

TREES=[('亭左松树',(350,343,513,446),(352,344,512,427)),
       ('亭右松树',(709,370,824,455),(709,370,824,429)),
       ('近景松树',(282,647,480,744),(282,647,480,722))]


def scalar_press(s):
    if s['role']=='far_outline':return .20
    if s['role']=='mountain_outline':return .25
    if s['role']=='near_outline':return .30
    if s['role']=='texture':return .15
    return None


def audit(candidate,baseline):
    candidate=Path(candidate);old=json.loads(Path(baseline).read_text(encoding='utf-8'));new=json.loads(candidate.read_text(encoding='utf-8'))
    before={s['id']:s for s in stroke_data(old)['black']};trees={t[0] for t in TREES};ink=stroke_data(new)['black']
    for s in ink:
        if s['drawing_region'] in trees:continue
        prior=before[s['id']];assert s['points']==prior['points'] and s['speed']==prior['speed']
        pressure=scalar_press(s)
        if pressure is None:assert s==prior
        else:assert set(s['press'])=={pressure}
    assert {s['id'] for s in ink if s['drawing_region'] not in trees}=={s['id'] for s in before.values() if s['drawing_region'] not in trees}
    rev=new['workflow']['source_revision'];tr=rev['source_transform'];r=tr['scale'];off=np.array(tr['offset']);scale=new['ink_job']['meta']['ink_execution']['mm_per_px']
    masks=np.load(candidate.parent/'第三幅原图约束掩膜.npz');samples=0;max_length=0
    branches=[s for s in ink if s.get('recipe')=='pavilion-source-pine']
    assert branches
    for s in branches:
        local=(dense_path(s['points'],.05/scale)-off)/r
        assert np.all(points_inside(local,masks['tree_support']))
        assert set(s['press'])<={.15,.30}
        samples+=len(local)
    colors=stroke_data(new)['color'];leaves=[s for s in colors if s['role']=='canopy_fill'];assert all(any(s['canopy_index']==i for s in leaves) for i in range(3))
    for s in colors:
        local=(dense_path(s['points'],.05/scale)-off)/r;max_length=max(max_length,s['length_mm'])
        assert s['length_mm']<=48.001
        mask=masks['leaf_safe_%d'%s['canopy_index']] if s['role']=='canopy_fill' else masks['ribbon_%d'%s['source_ribbon_index']]
        assert np.all(points_inside(local,mask))
        samples+=len(local)
    assert rev['stroke_data_sha256']==digest(stroke_data(new))
    assert new['workflow']['modules']['outline']['strokes']==ink
    from web.execution import prepare
    from run_json_painting import load_artwork
    prepare(candidate);load_artwork(candidate)
    def modeled_area(b):
        w,h=b['ink_job']['meta']['image_size'];m=np.zeros((h,w),np.uint8)
        for s in stroke_data(b)['color']:
            cv2.polylines(m,[np.rint(s['points']).astype(np.int32)],False,1,max(1,round(s['brush_width_mm_estimate']/scale)))
        return int(m.sum())
    counts=dict(black=sum(s['role']!='title' for s in ink),title=sum(s['role']=='title' for s in ink),**dict(Counter(s['color'] for s in colors)))
    return dict(robot_connected=False,counts=counts,total_strokes=sum(counts.values()),source_tree_strokes=len(branches),canopy_fill_strokes=len(leaves),
        source_traced_trees=True,architecture_stairs_title_and_water_exactly_preserved=True,pressure_layers_mm=[.15,.20,.25,.30],
        color_footprint_ratio=round(modeled_area(new)/modeled_area(old),3),dense_contacts_checked=samples,max_color_contact_length_mm=round(max_length,4),
        original_ink_masks_and_color_masks_passed=True,runtime_validation_passed=True,actual_brush_width_and_bleed_need_measurement=True)


def main():
    path=ROOT/ARTWORKS[2][3];package=path.parent.parent;old=json.loads(path.read_text(encoding='utf-8'));new=copy.deepcopy(old)
    from web.execution import prepare
    prepare(path)
    tr=old['workflow']['source_revision']['source_transform'];r=tr['scale'];off=np.array(tr['offset']);scale=old['ink_job']['meta']['ink_execution']['mm_per_px'];pixel_mm=scale*r
    source=package/'参考原图.png';im=cv2.imdecode(np.fromfile(str(source),np.uint8),1);shape=im.shape[:2];gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);b,g,red=im.astype(float).transpose(2,0,1)
    trees={t[0] for t in TREES};ink=[]
    for prior in stroke_data(old)['black']:
        if prior['drawing_region'] in trees:continue
        s=copy.deepcopy(prior);pressure=scalar_press(s)
        if pressure is not None:s['press']=[pressure]*len(s['points']);refresh_stats(s,scale)
        ink.append(s)
    neutral=((gray<130)&(np.ptp(im.astype(float),axis=2)<75)&(b>=g-7)).astype(np.uint8)*255
    tree_support=np.zeros(shape,np.uint8);branches=[];template=copy.deepcopy(next(s for s in stroke_data(old)['black'] if s['role']=='branch'))
    dt=cv2.distanceTransform((gray<100).astype(np.uint8),cv2.DIST_L2,5)
    for i,(region,tree_box,_) in enumerate(TREES):
        roi=np.zeros(shape,np.uint8);box(roi,tree_box);mask=neutral*roi;mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8));support=cv2.dilate(mask,np.ones((5,5),np.uint8));tree_support|=support
        raw=[np.array(p,float) for p in LandscapeBrushGenerator._trace_skeleton_paths(LandscapeBrushGenerator._thin_binary(mask))]
        raw=[p for p in raw if np.linalg.norm(np.diff(p,axis=0),axis=1).sum()*pixel_mm>.1];paths=merge(raw,mask)
        for j,p in enumerate(paths):
            arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]*pixel_mm
            if arc[-1]<1.0:continue
            t=np.unique(np.r_[0,np.arange(0,arc[-1],.20),arc[-1]]);local=np.column_stack([np.interp(t,arc,p[:,k]) for k in (0,1)])
            if not np.all(points_inside(dense_path(local,.05/pixel_mm),support)):continue
            q=local*r+off;n=len(q);xy=np.rint(local).astype(int);width=dt[xy[:,1],xy[:,0]]*2*pixel_mm;press=np.where(width>1.6,.30,.15).tolist();s=copy.deepcopy(template)
            for key in ('source_path_ids','parent_stroke_id','segment','segments'):s.pop(key,None)
            s.update(id='pavilion-source-pine-%d-%03d'%(i,j),points=q.tolist(),press=press,speed=[40.]*n,role='branch',drawing_region=region,subject_order=ORDERS[2].index(region),recipe='pavilion-source-pine')
            branches.append(refresh_stats(s,scale))
    ink.extend(branches);ink.sort(key=lambda s:ORDERS[2].index(s['drawing_region']));new['ink_job']=ink_payload(old['ink_job'],ink)
    new['workflow']['modules']['outline'].update(strokes=copy.deepcopy(ink),stroke_count=len(ink))
    colors=[];masks=dict(tree_support=tree_support);leaf_pigment=((g-red>9)&(b-red>6)&(b-g<24)&(g<205)).astype(np.uint8)
    leaf_radius=int(np.ceil(press_to_width_mm(.19)/2/pixel_mm));leaf_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(leaf_radius*2+1,leaf_radius*2+1))
    canopy_union=np.zeros(shape,np.uint8)
    for i,(region,_,leaf_box) in enumerate(TREES):
        roi=np.zeros(shape,np.uint8);box(roi,leaf_box);part=cv2.morphologyEx(leaf_pigment&roi,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        contours,_=cv2.findContours(part,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);leaf=np.zeros(shape,np.uint8)
        for co in contours:
            if cv2.contourArea(co)>25:cv2.drawContours(leaf,[co],-1,1,-1)
        safe=cv2.erode(leaf,leaf_kernel);canopy_union|=leaf;masks['leaf_safe_%d'%i]=safe;masks['leaf_%d'%i]=leaf
        for j,p in enumerate(scan(safe,pixel_mm)):
            q=dense_path(p,.25/pixel_mm)*r+off;n=len(q);s=dict(id='pavilion-solid-canopy-%d-%03d'%(i,j),role='canopy_fill',color='green',canopy_index=i,drawing_region=region,subject_order=ORDERS[2].index(region),region_id='pavilion-canopy-%d'%i,
                points=q.tolist(),press=[.19]*n,speed=[40.]*n,brush_width_mm_estimate=press_to_width_mm(.19),target_width_mm=[press_to_width_mm(.19)]*n,target_opacity=[.45]*n,pressure_mm=None,pigment_recipe_calibrated=False,recipe='source-mask-overlapping-canopy',layer=0,path_style='in-mask-overlapping-serpentine')
            colors.append(refresh_stats(s,scale))
    blurred=cv2.GaussianBlur(im.astype(float),(0,0),4);bb,gg,rr=blurred.transpose(2,0,1)
    raw={'blue':((bb-rr>10)&(bb-gg>3)&(rr<245)).astype(np.uint8),'green':((gg-rr>8)&(bb-gg<15)&(rr<245)).astype(np.uint8)}
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9));raw={c:cv2.morphologyEx(m,cv2.MORPH_CLOSE,kernel) for c,m in raw.items()}
    protected=np.zeros(shape,np.uint8)
    for s in ink:
        if s['role'] in ('near_outline','mountain_outline','far_outline','pavilion','stairs','title','branch'):
            xy=np.rint((np.asarray(s['points'])-off)/r).astype(np.int32);cv2.polylines(protected,[xy],False,1,1)
    box(protected,(505,280,716,443));box(protected,(386,429,554,509));protected|=canopy_union
    rad=int(np.ceil((.5+press_to_width_mm(.30)/2)/pixel_mm));protected=cv2.dilate(protected,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*rad+1,2*rad+1)))
    safe=(raw['blue']|raw['green']).copy();safe[protected>0]=0;safe[:rad]=0;safe[-rad:]=0;safe[:,:rad]=0;safe[:,-rad:]=0
    masks['terrain_safe']=safe;guides=GUIDES[2]+[('左下山石','blue',[(225,840),(266,842),(298,852)],[5,15,3]),('左下山石','green',[(351,849),(374,852),(395,861)],[5,12,3]),('左下山石','green',[(533,802),(556,808),(579,814)],[5,13,3]),('右侧山石','blue',[(1690,347),(1720,362),(1759,377)],[5,14,3])]
    blue_contact=np.zeros(shape,np.uint8);gap=int(np.ceil((press_to_width_mm(.30)+.5)/pixel_mm));gap_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*gap+1,2*gap+1))
    for j,(region,color,guide,widths) in sorted(enumerate(guides),key=lambda x:('blue','green').index(x[1][1])):
        factor=2.0 if '远山' in region else 3.4;allowed=safe.copy()
        if color=='green':allowed[cv2.dilate(blue_contact,gap_kernel)>0]=0
        paths,mask=ribbon_strokes(guide,[v*factor for v in widths],allowed,raw[color],color,region,j,j,scale,r,off);paths=join_ribbon_sweeps(paths,mask,scale,r,off);masks['ribbon_%d'%j]=mask
        for k,s in enumerate(paths):
            s['id']='pavilion-expanded-wash-%02d-%s-%03d'%(j,color,k);s['subject_order']=ORDERS[2].index(region)
            if color=='blue':xy=np.rint((np.asarray(s['points'])-off)/r).astype(np.int32);cv2.polylines(blue_contact,[xy],False,1,1)
        colors.extend(paths)
    # The original roof is blue-green; add pigment only within its colored cap.
    roof_roi=np.zeros(shape,np.uint8);box(roof_roi,(514,286,713,361))
    roof=((b-red>12)&(g-red>8)&(gray>70)&(roof_roi>0)).astype(np.uint8)
    roof=cv2.morphologyEx(roof,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8));roof_safe=cv2.erode(roof,leaf_kernel)
    roof_index=len(guides);masks['ribbon_%d'%roof_index]=roof_safe
    for j,p in enumerate(scan(roof_safe,pixel_mm)):
        q=dense_path(p,.25/pixel_mm)*r+off;n=len(q)
        s=dict(id='pavilion-blue-roof-%03d'%j,color='blue',role='roof_wash',drawing_region='亭子',subject_order=ORDERS[2].index('亭子'),source_ribbon_index=roof_index,
            points=q.tolist(),press=[.19]*n,speed=[40.]*n,brush_width_mm_estimate=press_to_width_mm(.19),target_width_mm=[press_to_width_mm(.19)]*n,target_opacity=[.45]*n,
            pressure_mm=None,pigment_recipe_calibrated=False,recipe='source-blue-roof',layer=0,region_id='pavilion-roof',path_style='in-original-roof-mask')
        colors.append(refresh_stats(s,scale))
    colors.sort(key=lambda s:(('blue','green').index(s['color']),ORDERS[2].index(s['drawing_region']),s.get('canopy_index',-1),s.get('source_ribbon_index',-1)))
    plan=new['workflow'];plan['modules']['color'].update(strokes=colors,stroke_count=len(colors));plan['color_geometry'].update(canopy_pitch_mm=max(1,int(np.floor(.70/pixel_mm)))*pixel_mm,leaf_pressure_mm=.19);plan['fill_report']={c:dict(strokes=sum(s['color']==c for s in colors),coverage_is_model_estimate=True) for c in ('blue','green')}
    rev=plan['source_revision'];rev.update(third_landscape_refinement=True,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_tree_boxes=TREES,after_counts=dict(black=len(ink),color=len(colors)))
    note='第三幅参照前两幅调整：三棵松树从原图真实墨线重提，叶团按原图形状重建重叠铺色，中心笔距约0.6mm、填色下压0.19mm；远山0.20mm、中景轮廓0.25mm、近景轮廓0.30mm，皴线和细枝0.15mm；扩大蓝绿岩面覆盖与远山淡色区域，保留原图白色崖面和云雾。亭子、台阶、题字和水纹结构逐笔保留。墨晕、笔宽和颜色浓淡未实测。'
    rev['note']+=' '+note;plan['note']=rev['note'];rev['stroke_data_sha256']=digest(stroke_data(new))
    pending=path.parent/'待应用_第三幅调整';pending.mkdir(exist_ok=True);save(pending/'painting.json',new);np.savez_compressed(pending/'第三幅原图约束掩膜.npz',**masks)
    result=audit(pending/'painting.json',path);save(pending/'第三幅调整校验.json',result)
    previews(old,new,pending,'云山亭阶 · 枝干与上色重整')
    from scripts.source_wash_comparison import render
    from scripts.repair_creek_canopy_fill import source_comparison
    source_comparison(new,source,pending);(pending/'原图与第三幅调整对照.png').write_bytes((pending/'新版原图与标注下压对照.png').read_bytes())
    for name in ('黑线加蓝绿_实际路径.png','任务预览_线宽色彩为估计.png'):render(new).save(pending/name)
    blank=copy.deepcopy(new);blank['workflow']['modules']['color']['strokes']=[];render(blank).save(pending/'纯勾线.png')
    (pending/'第三幅调整说明.md').write_text('# 云山亭阶 · 枝干与上色重整\n\n'+note+'\n\n'+json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    save(pending/'应用记录.json',dict(baseline_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256((pending/'painting.json').read_bytes()).hexdigest(),applied=False))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
