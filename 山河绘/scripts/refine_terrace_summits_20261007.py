"""Replace distant comb-like hatching with source-clipped curved slope strokes."""
from pathlib import Path
import sys,json,copy,hashlib
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw,ImageFont
from painting_session import save
from web.catalog import ARTWORKS
from scripts.source_aligned_washes import ribbon_geometry,dense_path,points_inside
from scripts.source_wash_comparison import render
from scripts.consolidate_selected_artworks import digest,stroke_data
from ink_control import refresh_stats
from shuimo import press_to_width_mm

# Curves follow individual pictured slopes, not a fixed zone-wide angle.
GUIDES=[
 ('左侧远山',[(237,311),(216,321),(191,334)],[3,7,2]),
 ('左侧远山',[(414,283),(381,312),(343,340),(298,363)],[5,13,13,2]),
 ('左侧远山',[(437,282),(463,316),(492,349),(534,379)],[5,12,11,2]),
 ('左侧远山',[(520,326),(549,354),(581,379),(620,400)],[3,8,8,2]),
 ('左侧远山',[(423,436),(391,446),(351,454)],[3,7,2]),
 ('左侧远山',[(432,437),(473,453),(514,467)],[3,9,2]),
 ('中间远山',[(682,365),(716,385),(756,406),(800,426)],[3,9,8,2]),
 ('中间远山',[(686,453),(650,473),(610,488),(565,497)],[4,12,11,2]),
 ('中间远山',[(702,455),(745,480),(792,496),(849,507)],[4,12,11,2]),
 ('右侧远山',[(1251,457),(1211,489),(1170,520),(1127,550)],[5,14,13,2]),
 ('右侧远山',[(1273,463),(1310,486),(1354,513),(1400,544)],[4,12,11,2]),
 ('右侧远山',[(1373,452),(1410,473),(1448,495)],[3,8,2]),
 ('右侧远山',[(1215,618),(1177,642),(1137,659)],[3,9,2]),
 ('右侧远山',[(1228,619),(1272,644),(1323,662),(1385,681)],[3,9,8,2]),
 ('中间远山',[(1041,523),(1008,547),(970,565),(927,575)],[3,8,8,2]),
]

MOUNTAINS=[('左侧远山',(170,255,620,427),425),('中间远山',(585,350,900,440),683),
 ('左侧远山',(320,420,555,490),425),('中间远山',(500,440,1000,553),695),
 ('右侧远山',(985,420,1491,608),1260),('中间远山',(790,515,1100,622),1040),
 ('右侧远山',(1050,602,1491,738),1220)]

def full_slope_paths(region_box,peak_x,pixel_mm):
    x0,y0,x1,y1=region_box;pitch=press_to_width_mm(.19)*.68/pixel_mm
    yy=np.arange(y0,y1,.45);dy=yy-y0
    for start in np.arange(x0-100,x1+100,pitch):
        xx=peak_x+(start-peak_x)*(1+.003*dy)+4*np.sin(dy/75+start/150)
        yield np.column_stack((xx,yy))

def join_sweeps(strokes,masks,cache,scale,r,off):
    from collections import defaultdict
    groups=defaultdict(list)
    for s in strokes:groups[(s['color'],s['drawing_region'],s['region_id'])].append(s)
    result=[]
    for (color,region,_),ss in groups.items():
        remaining=list(ss)
        while remaining:
            current=copy.deepcopy(remaining.pop(0))
            while remaining:
                end=np.array(current['points'][-1]);options=[]
                for j,s in enumerate(remaining):
                    for reverse in (False,True):
                        start=np.array(s['points'][-1] if reverse else s['points'][0]);distance=np.linalg.norm(end-start)*scale
                        if distance<=2.5 and current['length_mm']+s['length_mm']+distance<=45:options.append((distance,j,reverse))
                found=None
                for distance,j,reverse in sorted(options):
                    s=remaining[j];q=np.array(s['points'][::-1] if reverse else s['points']);turn=np.linspace(end,q[0],max(2,int(np.ceil(distance/.15))+1))
                    if np.all(points_inside((dense_path(turn,.05/scale)-off)/r,masks[color])):found=(j,q,turn);break
                if found is None:break
                j,q,turn=found;remaining.pop(j);xy=np.vstack((current['points'],turn[1:],q));tree,arrays=cache[(region,color)];_,ix=tree.query(xy);press=arrays['press'][ix].tolist()
                current.update(points=xy.tolist(),press=press,speed=[40.]*len(xy),target_width_mm=[press_to_width_mm(v) for v in press],target_opacity=arrays['target_opacity'][ix].tolist(),path_style='continuous-source-shaped-curved-return');refresh_stats(current,scale)
            result.append(current)
    return result

def fields(bundle,region,color):
    ss=[s for s in stroke_data(bundle)['color'] if s['role']=='distant_color' and s['drawing_region']==region and s['color']==color]
    if not ss:
        previous=json.loads((ROOT/ARTWORKS[4][3]).parent.parent.joinpath('历史版本/20261007_第五幅顺坡上色前/painting.json').read_text(encoding='utf-8'))
        ss=[s for s in stroke_data(previous)['color'] if s['role']=='distant_color' and s['drawing_region']==region and s['color']==color]
    q=np.concatenate([s['points'] for s in ss]);tree=cKDTree(q)
    arrays={k:np.concatenate([s[k] for s in ss]) for k in ('press','target_opacity')}
    return tree,arrays

def audit(candidate,baseline):
    candidate=Path(candidate);old=json.loads(Path(baseline).read_text(encoding='utf-8'));new=json.loads(candidate.read_text(encoding='utf-8'))
    assert old['ink_job']==new['ink_job'] and old['workflow']['modules']['outline']==new['workflow']['modules']['outline']
    kept=lambda d:[s for s in stroke_data(d)['color'] if s['role']!='distant_color']
    assert kept(old)==kept(new)
    before=old['workflow']['modules']['color'];after=new['workflow']['modules']['color'];assert before['motion_candidate']==after['motion_candidate']
    masks=np.load(candidate.parent/'顺山坡上色掩膜.npz');tr=new['workflow']['source_revision']['source_transform'];r=tr['scale'];off=np.array(tr['offset']);scale=before['mm_per_px'];added=[s for s in after['strokes'] if s.get('recipe')=='terrace-curved-slope'];cache={}
    for s in added:
        assert s['length_mm']<=45.001
        assert np.all(points_inside((dense_path(s['points'],.05/scale)-off)/r,masks[s['color']]))
        key=(s['drawing_region'],s['color'])
        if key not in cache:cache[key]=fields(old,*key)
        tree,arrays=cache[key];_,ix=tree.query(s['points'])
        assert np.array_equal(s['press'],arrays['press'][ix])
    assert new['workflow']['source_revision']['stroke_data_sha256']==digest(stroke_data(new))
    from web.execution import prepare
    from run_json_painting import load_artwork,manifest_for
    prepare(candidate);ink,plan,strokes,meta=load_artwork(candidate);manifest_for(strokes,plan,meta)
    counts=dict(black=sum(s['role']!='title' for s in strokes),title=sum(s['role']=='title' for s in strokes),**dict(Counter(s['color'] for s in after['strokes'])))
    def footprint(bundle):
        out=np.zeros(masks['target'].shape,np.uint8)
        for s in stroke_data(bundle)['color']:
            if s['role']!='distant_color':continue
            q=np.rint((np.array(s['points'])-off)/r).astype(np.int32)
            for i,(a,b) in enumerate(zip(q[:-1],q[1:])):
                width=(press_to_width_mm(s['press'][i])+press_to_width_mm(s['press'][i+1]))/(2*scale*r)
                cv2.line(out,tuple(a),tuple(b),1,max(1,round(width)))
        return float((out&masks['target']).sum()/max(1,masks['target'].sum()))
    coverage_before=footprint(old);coverage_after=footprint(new)
    assert coverage_after>coverage_before+.3
    return dict(robot_connected=False,ink_exactly_preserved=True,terrace_water_and_near_colors_exactly_preserved=True,pressure_values_inherited_from_same_region_and_color=True,source_mask_contact_audit_passed=True,
        distant_color_before=sum(s['role']=='distant_color' for s in before['strokes']),distant_color_after=len(added),color_before=len(before['strokes']),color_after=len(after['strokes']),counts=counts,total_strokes=sum(counts.values()),runtime_validation_passed=True,
        modeled_coverage_of_selected_source_color_mask_before=round(coverage_before,4),modeled_coverage_of_selected_source_color_mask_after=round(coverage_after,4))

def main():
    path=ROOT/ARTWORKS[4][3];old=json.loads(path.read_text(encoding='utf-8'));new=copy.deepcopy(old);plan=new['workflow'];module=plan['modules']['color'];scale=module['mm_per_px'];tr=plan['source_revision']['source_transform'];r=tr['scale'];off=np.array(tr['offset']);pixel_mm=scale*r
    source=cv2.imdecode(np.fromfile(str(path.parent.parent/'参考原图.png'),np.uint8),1);b,g,red=cv2.split(source.astype(float));chroma=source.max(2).astype(float)-source.min(2);gray=cv2.cvtColor(source,cv2.COLOR_BGR2GRAY)
    pigment=(np.maximum(b,g)-red>6)&(gray<251)&(gray>65)&(chroma>8);green=pigment&(g>=b-8);blue=pigment&~green
    far=np.zeros(gray.shape,np.uint8)
    for polygon in [[[185,280],[475,230],[850,420],[900,495],[740,568],[470,510],[330,438],[220,380]],[[780,545],[1010,469],[1290,420],[1490,420],[1490,704],[1090,735],[865,654]]]:
        cv2.fillPoly(far,[np.array(polygon,np.int32)],1)
    green&=far>0;blue&=far>0
    radius=int(np.ceil(press_to_width_mm(.19)/2/pixel_mm))+1
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    pigment=cv2.morphologyEx((blue|green).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))&(far>0)
    safe=cv2.erode(pigment,kernel);masks={c:safe.copy() for c in ('blue','green')}
    created=[];cache={};assigned=np.zeros(gray.shape,np.uint8)
    for gi,(region,box,peak) in enumerate(MOUNTAINS):
        x0,y0,x1,y1=box;roi=np.zeros(gray.shape,np.uint8);roi[y0:y1,x0:x1]=1;roi&=1-assigned;assigned|=roi
        for color in ('blue','green'):
            if (region,color) not in cache:cache[(region,color)]=fields(old,region,color)
            tree,arrays=cache[(region,color)]
            for local in full_slope_paths(box,peak,pixel_mm):
                good=points_inside(local,masks[color]&roi)
                edges=np.diff(np.r_[False,good,False].astype(int))
                for a,z in zip(np.where(edges==1)[0],np.where(edges==-1)[0]):
                    q=local[a:z]
                    if len(q)<3:continue
                    qq=np.rint(q).astype(int);chosen='blue' if np.mean(b[qq[:,1],qq[:,0]]-g[qq[:,1],qq[:,0]])>8 else 'green'
                    if color!=chosen:continue
                    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]*pixel_mm
                    if arc[-1]<2.5:continue
                    for start in np.arange(0,arc[-1],44):
                        end=min(start+44,arc[-1]);t=np.unique(np.r_[np.arange(start,end,.2),end]);u=np.column_stack([np.interp(t,arc,q[:,axis]) for axis in (0,1)])
                        if len(u)<3 or not np.all(points_inside(dense_path(u,.05/pixel_mm),masks[color])):continue
                        xy=u*r+off;_,ix=tree.query(xy);press=arrays['press'][ix].tolist();n=len(xy)
                        s=dict(id='terrace-slope-%d-%s-%d'%(gi,color,len(created)),color=color,role='distant_color',drawing_region=region,subject_order=plan['execution_order']['subjects'].index(region),region_id='curved-slope-%d'%gi,points=xy.tolist(),press=press,speed=[40.]*n,target_width_mm=[press_to_width_mm(v) for v in press],target_opacity=arrays['target_opacity'][ix].tolist(),brush_width_mm_estimate=max(press_to_width_mm(v) for v in press),layer=0,pressure_mm=None,pigment_recipe_calibrated=False,recipe='terrace-curved-slope',path_style='source-shaped-curved-slope',source_mountain_box=box)
                        created.append(refresh_stats(s,scale))
    created=join_sweeps(created,masks,cache,scale,r,off)
    for s in created:
        q=np.rint((np.array(s['points'])-off)/r).astype(int);strength=255-red[q[:,1],q[:,0]]
        s['target_opacity']=np.clip(strength/(183 if s['color']=='blue' else 158),.04,.38).tolist()
    kept=[copy.deepcopy(s) for s in module['strokes'] if s['role']!='distant_color'];colors=kept+created;colors.sort(key=lambda s:(('blue','green').index(s['color']),plan['execution_order']['subjects'].index(s['drawing_region'])))
    module.update(strokes=colors,stroke_count=len(colors));rev=plan['source_revision'];note='第五幅远山按原图淡蓝青绿色区扩大顺坡色带至两侧山坡，不再仅在山顶点色；色带收尾错开，白色云雾保持留白。山顶沿用同部位同颜色原有下压值；勾线、梯田、水面和近景色笔逐项保持。'
    rev.update(summit_slope_refinement=True,note=note,stroke_data_sha256=digest(stroke_data(new)));plan['note']=note
    masks['target']=safe&assigned
    pending=path.parent/'待应用_第五幅顺坡上色';pending.mkdir(exist_ok=True);save(pending/'painting.json',new);np.savez_compressed(pending/'顺山坡上色掩膜.npz',**masks);report=audit(pending/'painting.json',path);save(pending/'顺山坡上色校验.json',report)
    (pending/'顺山坡上色说明.md').write_text('# 第五幅顺山坡上色\n\n'+note+'\n\n'+json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for name in ('任务预览_线宽色彩为估计.png','黑线加蓝绿_实际路径.png'):render(new).save(pending/name)
    board=Image.new('RGB',(2000,1050),'#f7f7f0');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',28)
    for i,(bundle,label) in enumerate(((old,'修改前：远山色区过窄'),(new,'修改后：顺山坡连续扫色'))):
        pic=render(bundle,True);pic.thumbnail((930,760));x=35+i*1000;draw.text((x,25),label,font=font,fill='#304c3b');board.paste(pic,(x,100))
    draw.text((35,865),'勾线、梯田、水面和近景色笔保持；山顶继承同区域原有下压值。',font=font,fill='#304c3b');draw.text((35,910),'当前上色 %d 笔，其中山顶 %d 笔；预览笔宽、色阶与渗色仍为估计。'%(len(colors),len(created)),font=font,fill='#687562');board.save(pending/'第五幅顺山坡上色前后对照.png')
    original=Image.open(path.parent.parent/'参考原图.png').convert('RGB');w,h=original.size
    actual=render(new,True).crop((*off,off[0]+r*w,off[1]+r*h)).resize((w,h),Image.Resampling.LANCZOS)
    comparison=Image.new('RGB',(2000,1120),'#f7f7f0');cd=ImageDraw.Draw(comparison)
    for i,(pic,label) in enumerate(((original,'参考原图'),(actual,'远山按原图色区扩展 · 实际路径预览'))):
        x=35+1000*i;cd.text((x,25),label,font=font,fill='#304c3b');pic=pic.copy();pic.thumbnail((930,850));comparison.paste(pic,(x,85))
    cd.text((35,980),'远山补回山坡淡染范围，白色云雾留白；下压沿用原值。',font=font,fill='#304c3b')
    cd.text((35,1030),'形状来自执行路径；浓淡、混色、笔宽和渗色尚未纸上标定。',font=font,fill='#687562');comparison.save(pending/'第五幅远山扩色与原图对照.png')
    save(pending/'应用记录.json',dict(baseline_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256((pending/'painting.json').read_bytes()).hexdigest(),applied=False));print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
