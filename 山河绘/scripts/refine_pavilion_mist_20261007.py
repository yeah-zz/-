"""Source-shaped pale distance and coverage supplements; optical tone is a target."""
from pathlib import Path
import sys,json,copy,hashlib
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from web.catalog import ARTWORKS
from painting_session import save
from artwork_order import ink_payload
from ink_control import refresh_stats
from shuimo import press_to_width_mm
from scripts.consolidate_selected_artworks import digest,stroke_data
from scripts.source_aligned_washes import dense_path,points_inside
from scripts.source_wash_comparison import render
from scripts.order_selected_artworks import ORDERS

FAR=[('左侧远山',(218,192,612,330)),('中间远山',(866,350,1483,478)),
     ('中间远山',(805,502,1166,637)),('右侧远山',(1185,445,1497,594)),('右侧远山',(1580,316,1774,393))]
NEAR=[('左侧高山',(0,105,215,388)),('左侧高山',(0,375,253,580)),
      ('左下山石',(0,584,291,783)),('亭下完整山石',(271,505,634,793)),
      ('亭下完整山石',(635,435,960,793)),('右侧山石',(1385,330,1774,669)),('左下山石',(145,736,715,882))]


def raster(strokes,shape,scale,r,off):
    m=np.zeros(shape,np.uint8)
    for s in strokes:
        width=s.get('brush_width_mm_estimate',s.get('width_mm',1.))
        q=np.rint((np.asarray(s['points'])-off)/r).astype(np.int32)
        cv2.polylines(m,[q],False,1,max(1,round(width/(scale*r))))
    return m


def scan(mask,pixel_mm,press=.20):
    stride=max(1,int(round(press_to_width_mm(press)/2/pixel_mm)));ys=np.where(mask)[0]
    if not len(ys):return []
    rows=list(range(int(ys.min()),int(ys.max())+1,stride))
    if rows[-1]!=int(ys.max()):rows.append(int(ys.max()))
    result=[];maximum=max(2,int(np.floor(42/pixel_mm)))
    for row,y in enumerate(rows):
        xs=np.where(mask[y])[0]
        if not len(xs):continue
        for run in np.split(xs,np.where(np.diff(xs)>1)[0]+1):
            if len(run)<3:continue
            for a in range(0,len(run)-1,maximum-1):
                part=run[a:a+maximum]
                if len(part)<2:continue
                p=np.array([[part[0],y],[part[-1],y]],float)
                if row%2:p=p[::-1]
                result.append(p)
    return result


def audit(candidate,baseline):
    candidate=Path(candidate);old=json.loads(Path(baseline).read_text(encoding='utf-8'));new=json.loads(candidate.read_text(encoding='utf-8'))
    expected=[s for s in stroke_data(old)['black'] if s['role']!='far_outline'];assert stroke_data(new)['black']==expected
    kept=lambda b:{s['id']:s for s in stroke_data(b)['color'] if s.get('role')!='terrain_wash' or '远山' not in s['drawing_region']}
    originals=kept(old);current={s['id']:s for s in stroke_data(new)['color']}
    assert all(current[k]==v for k,v in originals.items()),'近景或叶团原有路径发生改变'
    rev=new['workflow']['source_revision'];tr=rev['source_transform'];r=tr['scale'];off=np.array(tr['offset']);scale=new['ink_job']['meta']['ink_execution']['mm_per_px']
    masks=np.load(candidate.parent/'远山云雾覆盖掩膜.npz');shape=masks['far_target'].shape;additions=[s for s in current.values() if s['id'] not in originals]
    samples=0;maximum=0
    for s in additions:
        assert s['role'] in ('mist_distance_wash','coverage_supplement') and set(s['press'])=={.20}
        local=(dense_path(s['points'],.05/scale)-off)/r
        assert np.all(points_inside(local,masks['safe_%d'%s['mask_index']]))
        assert s['length_mm']<=42.001;maximum=max(maximum,s['length_mm']);samples+=len(local)
    far_before=[s for s in stroke_data(old)['color'] if '远山' in s['drawing_region']]
    far_after=[s for s in additions if s['role']=='mist_distance_wash']
    target=masks['far_target'];before_area=raster(far_before,shape,scale,r,off)&target;after_area=raster(far_after,shape,scale,r,off)&target
    before_ratio=float(before_area.sum()/max(1,target.sum()));after_ratio=float(after_area.sum()/max(1,target.sum()))
    assert after_ratio>before_ratio+.25
    assert rev['stroke_data_sha256']==digest(stroke_data(new))
    from web.execution import prepare
    from run_json_painting import load_artwork
    prepare(candidate);load_artwork(candidate)
    counts=dict(black=len(expected)-sum(s['role']=='title' for s in expected),title=sum(s['role']=='title' for s in expected),**dict(Counter(s['color'] for s in current.values())))
    return dict(robot_connected=False,counts=counts,total_strokes=sum(counts.values()),far_black_lines_removed=sum(s['role']=='far_outline' for s in stroke_data(old)['black']),
        original_near_ink_and_existing_near_colors_preserved=True,far_color_coverage_before=round(before_ratio,4),far_color_coverage_after=round(after_ratio,4),
        source_masked_new_strokes=len(additions),contact_samples_checked=samples,max_contact_length_mm=round(maximum,4),far_and_supplement_press_mm=.20,
        white_cloud_areas_not_filled=True,optical_tone_targets_not_physically_calibrated=True,runtime_validation_passed=True)


def main():
    path=ROOT/ARTWORKS[2][3];package=path.parent.parent;old=json.loads(path.read_text(encoding='utf-8'));new=copy.deepcopy(old)
    from web.execution import prepare
    prepare(path)
    tr=old['workflow']['source_revision']['source_transform'];r=tr['scale'];off=np.array(tr['offset']);scale=old['ink_job']['meta']['ink_execution']['mm_per_px'];pixel_mm=scale*r
    source=package/'参考原图.png';im=cv2.imdecode(np.fromfile(str(source),np.uint8),1);shape=im.shape[:2];sm=cv2.GaussianBlur(im.astype(float),(0,0),3);b,g,red=sm.transpose(2,0,1)
    ink=[copy.deepcopy(s) for s in stroke_data(old)['black'] if s['role']!='far_outline'];new['ink_job']=ink_payload(old['ink_job'],ink);new['workflow']['modules']['outline'].update(strokes=copy.deepcopy(ink),stroke_count=len(ink))
    colors=[copy.deepcopy(s) for s in stroke_data(old)['color'] if s['role']!='terrain_wash' or '远山' not in s['drawing_region']]
    radius=int(np.ceil(press_to_width_mm(.20)/2/pixel_mm));kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))
    structures=np.zeros(shape,np.uint8)
    for s in ink:
        if s['role'] in ('pavilion','stairs','title','branch'):
            q=np.rint((np.array(s['points'])-off)/r).astype(np.int32);cv2.polylines(structures,[q],False,1,max(1,round(s['width_mm']/pixel_mm)))
    cv2.rectangle(structures,(505,280),(716,443),1,-1);cv2.rectangle(structures,(386,429),(554,509),1,-1)
    protected=cv2.dilate(structures,kernel)
    existing=raster(colors,shape,scale,r,off);occupied=cv2.dilate(existing,np.ones((3,3),np.uint8))
    near_ink=raster([s for s in ink if s['role'] in ('near_outline','mountain_outline')],shape,scale,r,off)
    near_ink=cv2.dilate(near_ink,kernel)
    far_pigment=((b-red>4)&(b-g>1)&(red<250)).astype(np.uint8)
    hues={'blue':((b-red>6)&(b-g>2)&(red<247)).astype(np.uint8),'green':((g-red>5)&(b-g<14)&(red<247)).astype(np.uint8)}
    masks={'far_target':np.zeros(shape,np.uint8)};index=0;created=[]
    for far,regions in ((True,FAR),(False,NEAR)):
        for region,(x0,y0,x1,y1) in regions:
            for color in (('blue',) if far else ('blue','green')):
                roi=np.zeros(shape,np.uint8);roi[y0:y1,x0:x1]=1
                pigment=(far_pigment if far else hues[color])&roi
                pigment=cv2.morphologyEx(pigment,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))&roi
                safe=cv2.erode(pigment,kernel);safe[(protected>0)|(occupied>0)|(near_ink>0)]=0
                if far:masks['far_target']|=safe
                masks['safe_%d'%index]=safe
                for j,line in enumerate(scan(safe,pixel_mm)):
                    local=dense_path(line,.25/pixel_mm);q=local*r+off;n=len(q);xy=np.rint(local).astype(int)
                    target=np.clip((255-red[xy[:,1],xy[:,0]])/(255-72 if color=='blue' else 255-97),.04,.5 if far else .45)
                    pitch=max(1,int(round(press_to_width_mm(.20)/2/pixel_mm)))*pixel_mm;alpha=1-(1-target)**(pitch/press_to_width_mm(.20))
                    s=dict(id='pavilion-mist-%02d-%s-%03d'%(index,color,j),color=color,role='mist_distance_wash' if far else 'coverage_supplement',drawing_region=region,
                        subject_order=ORDERS[2].index(region),mask_index=index,region_id='pavilion-distance-%d'%index,points=q.tolist(),press=[.20]*n,speed=[40.]*n,
                        brush_width_mm_estimate=press_to_width_mm(.20),target_width_mm=[press_to_width_mm(.20)]*n,target_opacity=alpha.tolist(),pressure_mm=None,
                        pigment_recipe_calibrated=False,recipe='source-shaped-optical-wash-target',layer=0,path_style='source-mask-overlapping-wash')
                    s=refresh_stats(s,scale);created.append(s)
                    footprint=np.zeros(shape,np.uint8);cv2.polylines(footprint,[xy.astype(np.int32)],False,1,max(1,round(press_to_width_mm(.20)/pixel_mm)))
                    occupied|=footprint
                index+=1
    colors.extend(created);colors.sort(key=lambda s:(('blue','green').index(s['color']),ORDERS[2].index(s['drawing_region']),s.get('canopy_index',-1),s.get('mask_index',-1)))
    plan=new['workflow'];plan['modules']['color'].update(strokes=colors,stroke_count=len(colors));plan['fill_report']={c:dict(strokes=sum(s['color']==c for s in colors),coverage_is_model_estimate=True) for c in ('blue','green')}
    rev=plan['source_revision'];note='远山与云雾交界修订：移除远山黑色边线，依据原图淡蓝颜料区域生成整片远山铺色，山脚渐淡并保留白云形状；补足源图近景色区中的漏铺部分。新增铺色下压0.20mm，色阶与透明度为光学目标，未作纸上浓淡标定；原有亭子、台阶、松树、题字、水纹及近景铺色均保留。'
    rev.update(distance_mist_refinement=True,distance_regions=FAR,coverage_supplement_regions=NEAR,optical_tone_not_calibrated=True,after_counts=dict(black=len(ink),color=len(colors)))
    rev['note']+=' '+note;plan['note']=rev['note'];rev['stroke_data_sha256']=digest(stroke_data(new))
    pending=path.parent/'待应用_远山云雾';pending.mkdir(exist_ok=True);save(pending/'painting.json',new);np.savez_compressed(pending/'远山云雾覆盖掩膜.npz',**masks)
    result=audit(pending/'painting.json',path);save(pending/'远山云雾校验.json',result)
    for name in ('任务预览_线宽色彩为估计.png','黑线加蓝绿_实际路径.png'):render(new).save(pending/name)
    source_pic=Image.open(source).convert('RGB');w,h=source_pic.size;actual=render(new,honor_opacity=True).crop((off[0],off[1],off[0]+r*w,off[1]+r*h)).resize((w,h),Image.Resampling.LANCZOS)
    board=Image.new('RGB',(2000,820),'#f7f7f0');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',30)
    for i,(pic,label) in enumerate(((source_pic,'参考原图'),(actual,'远山与云雾交界 · 规划色阶预览'))):
        x=35+i*1000;draw.text((x,25),label,font=font,fill='#304c3b');pic.thumbnail((930,670),Image.Resampling.LANCZOS);board.paste(pic,(x+(930-pic.width)//2,85+(670-pic.height)//2))
    draw.text((35,770),'形状来自实际路径；透明度和淡染为规划目标，实物浓淡、笔宽和渗色尚未标定。',font=font,fill='#687562');board.save(pending/'远山云雾与原图对照.png')
    (pending/'远山云雾说明.md').write_text('# 云山亭阶 · 远山与云雾交界\n\n'+note+'\n\n'+json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    save(pending/'应用记录.json',dict(baseline_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256((pending/'painting.json').read_bytes()).hexdigest(),applied=False))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
