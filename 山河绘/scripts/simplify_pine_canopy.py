"""Replace needle trajectories with compact, in-mask green canopy fills; offline."""
from pathlib import Path
import sys,json,copy,shutil,hashlib
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from web.catalog import ARTWORKS
from painting_session import session_lock,save
from artwork_order import ink_payload
from scripts.revise_pine_waterfall import order
from scripts.consolidate_selected_artworks import digest,stroke_data
from shuimo import press_to_width_mm
from ink_control import refresh_stats

PADS=[(144,425,86,33),(243,359,96,35),(369,405,79,32),(173,468,36,16),
      (396,462,62,22),(488,476,64,26),(476,520,44,20),(574,528,77,32),(643,581,68,27)]


def fill_paths(mask,pixel_mm,max_length=48.):
    """Horizontal serpentine sweeps, joined only through the same leaf mask."""
    step=max(2,int(np.floor(press_to_width_mm(.20)*.78/pixel_mm)))
    occupied=np.where(mask);paths=[];current=[]
    if not len(occupied[0]):return paths
    def length(points):return float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())*pixel_mm if len(points)>1 else 0
    def inside(a,b):
        q=np.rint(np.linspace(a,b,max(2,int(np.linalg.norm(np.array(b)-a))*2+1))).astype(int)
        return bool(np.all(mask[q[:,1],q[:,0]]))
    rows=range(int(occupied[0].min())+step//2,int(occupied[0].max())+1,step)
    for row,y in enumerate(rows):
        xs=np.where(mask[y])[0]
        if not len(xs):continue
        runs=np.split(xs,np.where(np.diff(xs)>1)[0]+1)
        if row%2:runs=runs[::-1]
        for run in runs:
            if len(run)<3:continue
            a=(int(run[0]),y);b=(int(run[-1]),y)
            if row%2:a,b=b,a
            if current and (not inside(current[-1],a) or length(current+[a,b])>max_length):
                paths.append(current);current=[]
            current.extend([a,b])
    if current:paths.append(current)
    return paths


def main():
    path=ROOT/ARTWORKS[0][3];package=path.parent.parent;task=path.parent;preview=package/'预览与下压标注'
    with session_lock(ROOT/'robot_runs'):
        if (ROOT/'robot_runs'/'blue_wash_motion.lock').exists():raise ValueError('Painting is active')
        if json.loads(path.read_text(encoding='utf-8'))['workflow'].get('source_revision',{}).get('terrain_fill'):
            raise ValueError('山石已经改为顺皴刷色，不能用旧版叶团备份覆盖。')
        history=package/'历史版本'/'20261006_针叶改叶团前';history.mkdir(parents=True,exist_ok=True)
        backup=history/'painting.json'
        if not backup.exists():shutil.copy2(path,backup)
        prior=json.loads(backup.read_text(encoding='utf-8'));bundle=copy.deepcopy(prior)
        from web.execution import prepare
        prepare(backup)
        revision=bundle['workflow']['source_revision'];ratio=revision['source_transform']['scale'];offset=np.array(revision['source_transform']['offset'])
        scale=bundle['ink_job']['meta']['ink_execution']['mm_per_px'];pixel_mm=scale*ratio
        image=cv2.imdecode(np.fromfile(str(package/'松叶简化版原图.png'),np.uint8),1)
        b,g,r=image.astype(int).transpose(2,0,1);yy,xx=np.indices(g.shape)
        mask=((g>b+8)&(g>r+12)&(g<235)&(xx>50)&(xx<720)&(yy>310)&(yy<625)).astype(np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        filled=np.zeros_like(mask)
        for co in contours:
            if cv2.contourArea(co)>300:cv2.drawContours(filled,[co],-1,1,-1)
        # Partition the connected green pads into the intended canopy masses.
        distances=np.stack([((xx-cx)/rx)**2+((yy-cy)/ry)**2 for cx,cy,rx,ry in PADS])
        assignment=np.argmin(distances,axis=0)
        colors=[copy.deepcopy(s) for s in prior['workflow']['modules']['color']['strokes'] if s.get('role')!='leaf_color']
        leaf_strokes=[];leaf_masks=[]
        for pad in range(len(PADS)):
            part=(filled&(assignment==pad)).astype(np.uint8);leaf_masks.append(part)
            safe=cv2.erode(part,np.ones((7,7),np.uint8))
            for j,line in enumerate(fill_paths(safe,pixel_mm)):
                q=np.array(line,float)*ratio+offset
                arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]*scale
                t=np.unique(np.r_[arc,np.arange(0,arc[-1],.35)])
                points=np.column_stack([np.interp(t,arc,q[:,k]) for k in (0,1)]);n=len(points)
                st=dict(id='green-canopy-%02d-%02d'%(pad,j),color='green',role='canopy_fill',
                    region_id='canopy-%02d'%pad,canopy_index=pad,points=points.tolist(),press=[.20]*n,speed=[40.]*n,
                    recipe='solid-green-canopy',layer=0,target_opacity=[.45]*n,pressure_mm=None,
                    path_style='in-mask-continuous-serpentine',pigment_recipe_calibrated=False,
                    brush_width_mm_estimate=press_to_width_mm(.20),target_width_mm=[press_to_width_mm(.20)]*n)
                refresh_stats(st,scale);assert st['length_mm']<=48.00001
                source_points=np.rint((points-offset)/ratio).astype(int)
                assert np.all(part[source_points[:,1],source_points[:,0]])
                colors.append(st);leaf_strokes.append(st)
        ink=[];removed_branches=[]
        for s in prior['ink_job']['meta']['ink_execution']['base_strokes']:
            # Keep the large trunk and branch contours; remove small traced forks.
            if s.get('drawing_region')=='整棵松树' and s.get('source_edge_index') not in (15,16,17):
                removed_branches.append(s['id']);continue
            ink.append(copy.deepcopy(s))
        # A few forked connectors expose the structure beneath the broad pads.
        forks=[[(208,442),(166,432),(143,416)],[(243,409),(233,387),(243,365)],
               [(302,442),(339,426),(370,403)],[(399,524),(430,488),(457,472)],
               [(548,581),(577,562),(610,554)]]
        template=copy.deepcopy(next(s for s in ink if s.get('role')=='branch'))
        for i,line in enumerate(forks):
            q=np.array(line,float)*ratio+offset;arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]*scale
            t=np.unique(np.r_[arc,np.arange(0,arc[-1],.35)]);points=np.column_stack([np.interp(t,arc,q[:,k]) for k in (0,1)])
            st=copy.deepcopy(template)
            for key in ('source_edge_index','source_path_ids','parent_stroke_id','segment','segments'):st.pop(key,None)
            st.update(id='simple-fork-%02d'%i,role='branch',points=points.tolist(),press=[.20]*len(points),speed=[40.]*len(points))
            ink.append(refresh_stats(st,scale))
        ink=order(ink,ratio,offset,True)
        # Explicit tree annotation is needed for new canopy paths outside old semantic masks.
        for st in leaf_strokes:st['drawing_region']='整棵松树'
        ordered=[]
        for color in ('blue','green'):
            ss=[s for s in colors if s['color']==color]
            ss=order(ss,ratio,offset,False)
            # Canopy pads complete one at a time, including all fill strips.
            for s in ss:
                if s.get('role')=='canopy_fill':s['drawing_region']='整棵松树';s['subject_order']=0
            from scripts.order_selected_artworks import ORDERS
            ss.sort(key=lambda s:(ORDERS[0].index(s['drawing_region']),s.get('canopy_index',-1)))
            ordered.extend(ss)
        plan=bundle['workflow'];plan['modules']['outline'].update(strokes=copy.deepcopy(ink),stroke_count=len(ink))
        plan['modules']['color'].update(strokes=ordered,stroke_count=len(ordered))
        bundle['ink_job']=ink_payload(prior['ink_job'],ink)
        from ink_control import build_plan,InkPolicy
        from shuimo import ink_palette_index
        _,stats=build_plan(ink,ink_palette_index(),scale,InkPolicy(**bundle['ink_job']['meta']['ink_execution']['policy']))
        bundle['ink_job']['meta']['quality_stats']=stats
        revision.update(leaf_mode='solid_canopy_pads',canopy_count=len(PADS),canopy_fill_strokes=len(leaf_strokes),
            canopy_baseline='历史版本/20261006_针叶改叶团前/painting.json',
            canopy_image='松叶简化版原图.png',canopy_image_sha256=hashlib.sha256((package/'松叶简化版原图.png').read_bytes()).hexdigest(),
            canopy_pitch_mm=float(max(2,int(np.floor(press_to_width_mm(.20)*.78/pixel_mm)))*pixel_mm),
            canopy_removed_branch_ids=removed_branches,after_counts=dict(black=len(ink),color=len(ordered)),
            note='针状松叶改为9组绿色叶团，连续蛇形涂抹；保留主干、主枝和少量分叉，所有颜色后最后题字。')
        revision['stroke_data_sha256']=digest(stroke_data(bundle));plan['note']=revision['note']
        plan['fill_report']={c:dict(strokes=sum(s['color']==c for s in ordered),coverage_not_remeasured=True) for c in ('blue','green')}
        from run_color_from_job import validate_plan
        validate_plan(plan,bundle['ink_job'])
        for p in (package/'参考原图.png',preview/'任务预览_线宽色彩为估计.png',preview/'执行路径预览_非实测线宽.png'):
            if p.exists() and not (history/p.name).exists():shutil.copy2(p,history/p.name)
        # Same image frame and paper margins as the existing reviewed trajectories.
        w,h=bundle['ink_job']['meta']['image_size'];reference=np.full((h,w,3),255,np.uint8)
        resized=cv2.resize(image,(round(image.shape[1]*ratio),round(image.shape[0]*ratio)),interpolation=cv2.INTER_AREA)
        x,y=offset;reference[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
        cv2.imencode('.png',reference)[1].tofile(str(package/'参考原图.png'))
        save(path,bundle);save(task/'black_only.json',bundle['ink_job'])
        for p in task.glob('*_黑线任务.json'):save(p,bundle['ink_job'])
        for p in task.glob('*_计划.json'):save(p,plan)
        save(task/'原图修订与下压说明.json',revision)
        save(task/'叶团上色设计.json',dict(pads=PADS,press_mm=.20,pitch_mm=revision['canopy_pitch_mm'],
            counts=len(leaf_strokes),all_centerlines_inside_own_canopy=True,max_stroke_length_mm=48,robot_connected=False))
        audit=json.loads((task/'离线检查.json').read_text(encoding='utf-8'))
        from collections import Counter
        audit.update(ink_strokes=len(ink),source_revision=revision,colors=plan['fill_report'],
            role_counts=dict(Counter(s['role'] for s in ink)));save(task/'离线检查.json',audit)
        canvas=np.full((h,w,3),255,np.uint8)
        for st in ink+ordered:
            color=(45,45,45) if st in ink else (175,125,72) if st['color']=='blue' else (97,152,97)
            cv2.polylines(canvas,[np.rint(st['points']).astype(np.int32)],False,color,max(1,round(press_to_width_mm(st['press'][0])/scale)),cv2.LINE_AA)
        for name in ('任务预览_线宽色彩为估计.png','执行路径预览_非实测线宽.png'):
            cv2.imencode('.png',canvas)[1].tofile(str(preview/name))
        note='\n\n## 松叶简化\n\n针状松叶改成9组绿色叶团，采用0.20 mm连续蛇形涂抹，完整涂完一组叶团再画下一组；保留主干、主枝和5条简单分叉。新增参考图为松叶简化版原图.png，最初用户原图保留。\n'
        for p in (package/'阅读说明.md',task/'原图修订与下压说明.md'):
            text=p.read_text(encoding='utf-8');text=text.split('\n\n## 松叶简化')[0]+note
            text=text.replace('松叶原有扇形线条移到绿色上色阶段，采用0.20 mm；','松叶采用9组简化叶团连续绿色涂抹，下压0.20 mm；')
            if p.name=='阅读说明.md':
                import re
                text=re.sub(r'当前笔数：.*?。','当前笔数：山水勾勒 %d 笔，蓝色 %d 笔，绿色 %d 笔，最后题字 13 笔。'%(len(ink)-13,sum(s['color']=='blue' for s in ordered),sum(s['color']=='green' for s in ordered)),text)
            p.write_text(text,encoding='utf-8')
        print(json.dumps(dict(canopy_pads=len(PADS),canopy_strokes=len(leaf_strokes),black=len(ink)-13,
            blue=sum(s['color']=='blue' for s in ordered),green=sum(s['color']=='green' for s in ordered),title=13)))


if __name__=='__main__':main()
