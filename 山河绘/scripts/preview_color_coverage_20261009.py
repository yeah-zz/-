"""Offline, review-only color candidates. Never install tasks or connect to a robot."""
from pathlib import Path
import sys, json, copy, hashlib, shutil, socket
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw, ImageFont
from ink_control import refresh_stats
from shuimo import press_to_width_mm
from scripts.source_aligned_washes import dense_path, points_inside
from scripts.source_wash_comparison import render
from scripts.consolidate_selected_artworks import digest, stroke_data

OUT = ROOT / 'artifacts/color_coverage_moderate_20261009'
TITLES = ['松瀑亭舟', '松岸曲水', '云山亭阶', '柳岸渔舟', '青绿梯田云山']
PITCH_MM = .60
ASSUMED_WIDTH_MM = .80  # illustrative, NOT a measured/calibrated brush width


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def source_masks(img, pixel_mm):
    smooth = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 4.5)
    b, g, r = smooth.transpose(2, 0, 1)
    # Weakly colored washes count; neutral paper, ink and cloud gaps do not.
    pigment = ((np.maximum(b, g)-r > 10) & (r < 244) & (g > 55))
    raw = {'blue': pigment & (b-g > 7), 'green': pigment & (b-g <= 7)}
    masks = {}
    for c, rawmask in raw.items():
        mask = cv2.morphologyEx(rawmask.astype(np.uint8), cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for n in range(1, count):
            if stats[n, cv2.CC_STAT_AREA]*pixel_mm**2 < .8:
                mask[labels == n] = 0
        masks[c] = mask
    return masks


def width_mask(strokes, shape, ratio, off, pixel_mm, width=ASSUMED_WIDTH_MM):
    canvas = np.zeros(shape, np.uint8)
    for s in strokes:
        q = np.rint((np.asarray(s['points'])-off)/ratio).astype(np.int32)
        cv2.polylines(canvas, [q], False, 1, max(1, round(width/pixel_mm)), cv2.LINE_8)
    return canvas


def narrow_render(bundle):
    """Same assumed .8 mm color line width on BOTH versions; no opacity effects."""
    meta = bundle['ink_job']['meta']; w, h = meta['image_size']
    base = copy.deepcopy(bundle)
    base['workflow']['modules']['color']['strokes'] = []
    canvas = np.asarray(render(base)).copy()
    scale = meta['ink_execution']['mm_per_px']
    for c, rgb in [('blue', (72, 125, 175)), ('green', (97, 152, 97))]:
        for s in bundle['workflow']['modules']['color']['strokes']:
            if s['color'] != c: continue
            q = np.rint(s['points']).astype(np.int32)
            cv2.polylines(canvas, [q], False, rgb, max(1, round(ASSUMED_WIDTH_MM/scale)), cv2.LINE_AA)
    return Image.fromarray(canvas)


def samples_by_region(strokes, ratio, off, color=None):
    coords=[]; owners=[]; names=[]; tangents=defaultdict(list)
    for s in strokes:
        if color and s['color'] != color: continue
        region = s.get('drawing_region', s.get('region_id', '区域'))
        if region not in names: names.append(region)
        p = (np.asarray(s['points'])-off)/ratio
        q = dense_path(p, 3.)
        coords.extend(q); owners.extend([names.index(region)]*len(q))
        d=np.diff(p,axis=0); lens=np.linalg.norm(d,axis=1)
        tangents[region].extend(d[lens>.1])
    return cKDTree(coords), np.asarray(owners), names, tangents


def make_stroke(q, example, number, color, ratio, off, scale):
    s = {k: copy.deepcopy(example[k]) for k in
         ('role','drawing_region','subject_order','region_id','layer') if k in example}
    n=len(q)
    # Keep the local original depth, with the user's new lower/upper limits.
    depth = float(np.clip(np.median(example['press']), .20, .30))
    s.update(id=f'coverage-{number:05d}-{color}', color=color, points=(q*ratio+off).tolist(),
             press=[depth]*n, speed=[40.]*n, pressure_mm=None,
             recipe='source-pigment-coverage', path_style='source-clipped-parallel-sweep',
             target_width_mm=[press_to_width_mm(depth)]*n,
             brush_width_mm_estimate=press_to_width_mm(depth),
             pigment_recipe_calibrated=False, coverage_pitch_mm=PITCH_MM)
    return refresh_stats(s,scale)


def join_local_sweeps(strokes, safe, ratio, off, scale):
    """Round adjacent returns within one source mask; never bridge a cloud/ink gap."""
    remaining=list(strokes); output=[]
    while remaining:
        current=remaining.pop(0); legs=1
        while remaining and legs<4:
            p=np.asarray(current['points']); end=p[-1];options=[]
            for k,s in enumerate(remaining):
                for reverse in (False,True):
                    q=np.asarray(s['points'][::-1] if reverse else s['points'])
                    distance=np.linalg.norm(end-q[0])*scale
                    if distance<=1.8 and current['length_mm']+s['length_mm']+distance*2<45:
                        options.append((distance,k,reverse))
            found=None
            for distance,k,reverse in sorted(options):
                q=np.asarray(remaining[k]['points'][::-1] if reverse else remaining[k]['points'])
                # Raster sweep endpoints lie on the inset boundary; round the return
                # slightly inside it instead of demanding a turn beyond the boundary.
                for trim in (0,1,2,3):
                    if len(p)<=trim+3 or len(q)<=trim+3:continue
                    pp=p[:-trim] if trim else p;qq=q[trim:]
                    a=pp[-1];b=qq[0];ta=pp[-1]-pp[-2];tb=qq[1]-qq[0]
                    ta/=np.linalg.norm(ta);tb/=np.linalg.norm(tb)
                    t=np.linspace(0,1,33)[:,None];span=np.linalg.norm(b-a)
                    turn=(2*t**3-3*t**2+1)*a+(t**3-2*t**2+t)*ta*span*.5+(-2*t**3+3*t**2)*b+(t**3-t**2)*tb*span*.5
                    if np.all(points_inside((dense_path(turn,.2)-off)/ratio,safe)):
                        found=(k,pp,qq,turn);break
                if found is not None:break
            if found is None:break
            k,p,q,turn=found;remaining.pop(k)
            merged=np.vstack((p,turn[1:],q[1:]));n=len(merged);depth=current['press'][0]
            current.update(points=merged.tolist(),press=[depth]*n,speed=[40.]*n,
                           target_width_mm=[press_to_width_mm(depth)]*n,
                           path_style='source-mask-rounded-return',connected_passes=legs+1)
            refresh_stats(current,scale);legs+=1
        output.append(current)
    return output


def connect_region(strokes,safe,ratio,off,scale):
    """Visit nearby paths and join only within a single named region and color.

    Retain original point pressures/speeds when reversing/merging. Each contact
    stays under 80 mm, leaving room under the runtime's 100 mm loading budget.
    """
    if not strokes:return [],0
    remaining=copy.deepcopy(strokes);out=[];current=remaining.pop(0);joins=0
    while remaining:
        p=np.asarray(current['points']);end=p[-1];options=[]
        for k,s in enumerate(remaining):
            for reverse in (False,True):
                q=np.asarray(s['points'][-1] if reverse else s['points'][0])
                options.append((np.linalg.norm(end-q)*scale,k,reverse))
        distance,k,reverse=min(options)
        s=remaining.pop(k)
        if reverse:
            for key in ('points','press','speed','target_width_mm','target_opacity'):
                if isinstance(s.get(key),list):s[key]=s[key][::-1]
        q=np.asarray(s['points']);turn=None
        if distance<=2.0 and current['length_mm']+s['length_mm']+distance*2<=80:
            a=end;b=q[0];ta=p[-1]-p[-2];tb=q[1]-q[0]
            if np.linalg.norm(ta)>1e-8 and np.linalg.norm(tb)>1e-8:
                ta/=np.linalg.norm(ta);tb/=np.linalg.norm(tb)
                t=np.linspace(0,1,25)[:,None];span=np.linalg.norm(b-a)
                curve=(2*t**3-3*t**2+1)*a+(t**3-2*t**2+t)*ta*span*.5+(-2*t**3+3*t**2)*b+(t**3-t**2)*tb*span*.5
                if np.all(points_inside((dense_path(curve,.2)-off)/ratio,safe)):turn=curve
        if turn is None:
            out.append(current);current=s;continue
        previous=list(current.get('joined_source_ids',[current['id']]))
        following=list(s.get('joined_source_ids',[s['id']]))
        presses=current['press']+np.linspace(current['press'][-1],s['press'][0],len(turn)).tolist()[1:]+s['press'][1:]
        speeds=current['speed']+np.linspace(current['speed'][-1],s['speed'][0],len(turn)).tolist()[1:]+s['speed'][1:]
        merged=np.vstack((p,turn[1:],q[1:]))
        current.update(points=merged.tolist(),press=presses,speed=speeds,
                       target_width_mm=[press_to_width_mm(v) for v in presses],
                       joined_source_ids=previous+following,path_style='continuous-within-subject')
        current.pop('target_opacity',None)
        refresh_stats(current,scale);joins+=1
        assert current['length_mm']<=80+1e-6
    out.append(current)
    return out,joins


def supplement(old, img, number):
    new=copy.deepcopy(old); module=new['workflow']['modules']['color']
    oldst=old['workflow']['modules']['color']['strokes']
    trans=old['workflow']['source_revision']['source_transform']
    ratio=trans['scale'];off=np.asarray(trans['offset']);scale=module['mm_per_px'];pxmm=ratio*scale
    masks=source_masks(img,pxmm); shape=img.shape[:2]
    pigment_distance=cv2.distanceTransform(masks['blue']|masks['green'],cv2.DIST_L2,5)
    # Protect the executed ink footprint, not just the source photograph's dark
    # pixels: the reviewed outlines include deliberate geometry refinements.
    protected=np.zeros(shape,np.uint8)
    ink=old['ink_job']['meta']['ink_execution']['base_strokes']
    for s in ink:
        q=np.rint((np.asarray(s['points'])-off)/ratio).astype(np.int32)
        ink_width=max(press_to_width_mm(v) for v in s['press'])
        # Brush center clearance keeps the .30 mm width model outside the ink.
        width=ink_width+press_to_width_mm(.30)+.20
        cv2.polylines(protected,[q],False,1,max(1,int(np.ceil(width/pxmm))))
    for role in ('title','pavilion','boat','stairs'):
        ss=[s for s in ink if s.get('role')==role]
        if not ss:continue
        q=(np.vstack([s['points'] for s in ss])-off)/ratio
        lo=np.floor(q.min(0)-1/pxmm).astype(int);hi=np.ceil(q.max(0)+1/pxmm).astype(int)
        cv2.rectangle(protected,tuple(lo),tuple(hi),1,-1)
    for s in module['strokes']:
        s.setdefault('speed',[float(module['motion_candidate']['draw_velocity_percent'])]*len(s['points']))
        s['press']=np.clip(s['press'], .20, .30).tolist()
        s['target_width_mm']=[press_to_width_mm(v) for v in s['press']]
        s['brush_width_mm_estimate']=float(np.mean(s['target_width_mm']))
        refresh_stats(s,scale)
    module['motion_candidate']['press_mm']=.30
    # Web now inherits acceleration. Do not put the obsolete 20% override in new candidates.
    module['motion_candidate']['accel_joint_percent']=0
    module['motion_candidate']['accel_linear_percent']=0
    added=[]; reports={}; mask_export={}; region_safe={};join_count=0
    tree,owners,names,tangents=samples_by_region(oldst,ratio,off)
    examples={s.get('drawing_region',s.get('region_id','区域')):s for s in oldst}
    for c in ('blue','green'):
        mask=masks[c]
        yy,xx=np.nonzero(mask); coords=np.column_stack((xx,yy))
        distances,indices=tree.query(coords)
        assigned=np.full(shape,-1,np.int16);assigned[yy,xx]=owners[indices]
        # Restrict additions to existing named subjects, and avoid filling open lake,
        # boat detail or distant unrelated pigment from the source image.
        for j,name in enumerate(names):
            if any(v in name for v in ('水纹','渔舟')): mask[assigned==j]=0
        mask[protected>0]=0
        before=width_mask([s for s in oldst if s['color']==c],shape,ratio,off,pxmm)
        distance_to_existing=cv2.distanceTransform((1-before).astype(np.uint8),cv2.DIST_L2,5)*pxmm
        covered=width_mask([s for s in module['strokes'] if s['color']==c],shape,ratio,off,pxmm)
        for j, name in enumerate(names):
            area=((assigned==j)&(mask>0)).astype(np.uint8)
            if not area.any():continue
            # Radius covers the largest .30 mm model width, plus .1 mm geometric reserve.
            # This is an estimate; real spreading still needs a paper calibration.
            inset=(press_to_width_mm(.30)/2+.10)/pxmm
            # Leave room at paper/ink edges, not a white channel at every blue/green
            # boundary. Neighboring source hues may meet under the brush footprint.
            safe=((pigment_distance>inset)&(area>0)&(protected==0)).astype(np.uint8)
            region_safe[(c,name)]=safe.copy()
            # User rejected full-area fill. Work only beside the original bands;
            # distant mountains, foliage and terraces get especially little expansion.
            growth=.35 if '远' in name else .20 if '树' in name else .30 if '梯田' in name else .80
            safe &= (distance_to_existing<=growth).astype(np.uint8)
            if not safe.any():continue
            d=np.asarray(tangents[name]); lengths=np.linalg.norm(d,axis=1)
            # Principal segment direction keeps the established rock-face / leaf strokes.
            cov=(d.T@d) if len(d) else np.eye(2)
            _,vec=np.linalg.eigh(cov);v=vec[:,-1]
            if v[1]<0:v=-v
            normal=np.array([v[1],-v[0]])
            ys,xs=np.nonzero(safe); ps=np.column_stack((xs,ys));along=ps@v; across=ps@normal
            t=np.arange(along.min(),along.max()+.5,.45)
            scan=[]
            for k,u in enumerate(np.arange(across.min(),across.max()+1,PITCH_MM/pxmm)):
                line=t[:,None]*v+u*normal
                good=points_inside(line,safe)
                edges=np.diff(np.r_[False,good,False].astype(int))
                for a,b in zip(np.where(edges==1)[0],np.where(edges==-1)[0]):
                    q=line[a:b]
                    if len(q)<2:continue
                    length=np.linalg.norm(q[-1]-q[0])*pxmm
                    if length<1.5:continue
                    # Only add paths that actually close coverage gaps. Avoid repainting
                    # an already filled terrace just to raise the apparent stroke count.
                    ij=np.rint(q).astype(int)
                    if np.mean(covered[ij[:,1],ij[:,0]])>.80:continue
                    arc=np.linspace(0,length,len(q))
                    breaks=np.linspace(0,length,int(np.ceil(length/40))+1)
                    for lo,hi in zip(breaks[:-1],breaks[1:]):
                        ts=np.r_[np.arange(lo,hi,.25),hi]
                        segment=np.column_stack([np.interp(ts,arc,q[:,ax]) for ax in (0,1)])
                        if k%2:segment=segment[::-1]
                        if not np.all(points_inside(dense_path(segment,.25),safe)):continue
                        st=make_stroke(segment,examples[name],len(added)+len(scan),c,ratio,off,scale)
                        scan.append(st)
                        cv2.polylines(covered,[np.rint(segment).astype(np.int32)],False,1,
                                      max(1,round(ASSUMED_WIDTH_MM/pxmm)))
            # Do not connect arbitrary separate source patches through white gaps.
            fraction=.15 if '远' in name else .10 if '树' in name else .15 if '梯田' in name else .45
            original_length=sum(s['length_mm'] for s in oldst if s['color']==c and s.get('drawing_region',s.get('region_id','区域'))==name)
            budget=original_length*fraction
            # Distribute limited additions through the region instead of filling
            # one end completely. This is a contact-length cap, not an opacity trick.
            selected=[]
            order=[]
            for phase in range(4):order.extend(range(phase,len(scan),4))
            for k in order:
                if scan[k]['length_mm']<=budget:
                    selected.append(scan[k]);budget-=scan[k]['length_mm']
            added.extend(join_local_sweeps(selected,safe,ratio,off,scale))
        after=width_mask([s for s in module['strokes']+added if s['color']==c],shape,ratio,off,pxmm)
        reports[c]={'target_area_mm2':float(mask.sum()*pxmm**2),
                    'before_coverage':float((before&mask).sum()/max(1,mask.sum())),
                    'after_coverage':float((after&mask).sum()/max(1,mask.sum()))}
        mask_export[c]=mask
    # Keep region order stable. All original paths keep their internal order.
    for i,s in enumerate(added):s['id']=f'coverage-{number:02d}-{s["color"]}-{i:05d}'
    groups=defaultdict(list)
    for s in module['strokes']+added:groups[(s['color'],s.get('drawing_region',s.get('region_id','区域')))].append(s)
    combined=[]
    for c in ('blue','green'):
        for name in names:
            safe=region_safe.get((c,name),np.zeros(shape,np.uint8))
            linked,count=connect_region(groups[(c,name)],safe,ratio,off,scale)
            combined.extend(linked);join_count+=count
    module['strokes']=combined
    module['reload_strokes_per_load']=8
    module['stroke_count']=len(module['strokes'])
    for c in ('blue','green'):
        final_mask=width_mask([s for s in combined if s['color']==c],shape,ratio,off,pxmm)
        target=mask_export[c]
        reports[c]['after_coverage']=float((final_mask&target).sum()/max(1,target.sum()))
    new['workflow']['execution_order']['color_mode']='one_color_all_subjects'
    new['workflow']['execution_order']['subjects']=names
    new['workflow']['source_revision']['stroke_data_sha256']=digest(stroke_data(new))
    new['workflow']['coverage_review']={
        'date':'2026-10-09','status':'preview-only-not-installed',
        'pressure_range_mm':[.20,.30],'pitch_mm':PITCH_MM,
        'assumed_color_width_mm':ASSUMED_WIDTH_MM,'calibrated_on_paper':False,
        'source_pigment_coverage':reports,'added_strokes':len(added),
        'continuous_joins':join_count,'max_joined_contact_mm':80,
        'reload_stroke_limit':8,'reload_length_limit_mm':100,'reload_time_limit_s':60,
        'color_order':'one_color_all_subjects','style':'restrained-original-band-expansion',
        'note':'下压范围仅针对上色；黑线及题字保持基线。覆盖率是0.8mm假设笔宽的离线估计，不是实物保证。'}
    return new,reports,mask_export,added


def aligned_reference(img,bundle):
    trans=bundle['workflow']['source_revision']['source_transform']
    ratio=trans['scale'];off=trans['offset'];w,h=bundle['ink_job']['meta']['image_size']
    matrix=np.array([[ratio,0,off[0]],[0,ratio,off[1]]],np.float32)
    aligned=cv2.warpAffine(img,matrix,(w,h),borderValue=(255,255,255))
    return Image.fromarray(cv2.cvtColor(aligned,cv2.COLOR_BGR2RGB))


def make_board(pics,labels,title,width=2100,height=690):
    board=Image.new('RGB',(width,height),'#f6f4ed');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',27)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
    draw.text((24,12),title,font=font,fill='#243a35')
    cell=(width-48)//len(pics)
    for i,(pic,label) in enumerate(zip(pics,labels)):
        pic=pic.copy();pic.thumbnail((cell-15,height-140),Image.Resampling.LANCZOS)
        x=24+i*cell
        draw.text((x,56),label,font=small,fill='#243a35')
        board.paste(pic,(x+(cell-15-pic.width)//2,99+(height-140-pic.height)//2))
    draw.text((24,height-36),'轨迹预览：两版上色均按假设笔宽0.8mm显示；颜色及渗化未实测，不代表实物效果。',font=small,fill='#65736c')
    return board


def main():
    # Enforce offline execution even if an imported validator accidentally connects.
    def no_network(*args,**kwargs):raise RuntimeError('Offline preview cannot connect to a robot/network')
    socket.socket.connect=no_network;socket.socket.connect_ex=no_network
    OUT.mkdir(parents=True,exist_ok=True)
    paths=sorted((ROOT/'待画作品').glob('*/任务资料/painting.json'))[:5]
    protected=paths+list((ROOT/'web').glob('*settings*.json'))+[ROOT/'config.py']
    pending=paths[2].parent/'待应用_右峰黑线/painting.json'
    if pending.exists():protected+=[pending,pending.parent/'应用记录.json']
    hashes={str(p):sha(p) for p in protected if p.exists()}
    summaries=[];boards=[]
    for n,path in enumerate(paths,1):
        effective=path
        if n==3 and pending.exists() and not read(pending.parent/'应用记录.json').get('applied'):
            effective=pending
        old=read(effective)
        reference=path.parent.parent/('用户原图.png' if n==1 else '参考原图.png')
        revision=old['workflow']['source_revision']
        expected=revision.get('source_image_sha256',revision.get('source_sha256'))
        assert expected==sha(reference), 'Source image does not match the saved coordinate transform'
        img=cv2.imdecode(np.fromfile(reference,np.uint8),cv2.IMREAD_COLOR)
        new,coverage,masks,added=supplement(old,img,n)
        folder=OUT/f'{n:02d}_{TITLES[n-1]}'
        target=folder/'任务资料/painting.json'
        folder.mkdir(parents=True,exist_ok=True)
        shutil.copy2(reference,folder/reference.name)
        # Preserve relative assets used by the original loader, without altering ink data.
        source_asset=(path.parent/old['ink_job']['meta']['image']).resolve()
        dest_asset=(target.parent/old['ink_job']['meta']['image']).resolve()
        if source_asset.exists():
            assert dest_asset.is_relative_to(folder.resolve())
            dest_asset.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source_asset,dest_asset)
        save(target,new)
        assert old['ink_job']==new['ink_job'], 'Black/title changed'
        from run_json_painting import load_artwork
        from web.execution import prepare
        from run_color_experiment import module_lines
        from color_profile import ColorProfile
        prepare(target)
        load_artwork(target)
        module=new['workflow']['modules']['color'];profile=ColorProfile(**module['motion_candidate']).to_dict()
        counts={}
        for c in ('blue','green'):
            lines,skipped=module_lines(module,c,profile)
            assert not skipped,(n,c,skipped)
            counts[c]=len(lines)
        presses=np.concatenate([s['press'] for s in module['strokes']])
        assert np.isfinite(presses).all() and presses.min()>=.20 and presses.max()<=.30
        assert len({s['id'] for s in module['strokes']})==len(module['strokes'])
        oldlength=sum(s['length_mm'] for s in old['workflow']['modules']['color']['strokes'])
        newlength=sum(s['length_mm'] for s in module['strokes'])
        summary={'title':TITLES[n-1],'source':str(effective),'source_sha256':sha(effective),
                 'candidate':str(target),'pressure_mm':[float(presses.min()),float(presses.max())],
                 'counts':counts,'added_strokes':len(added),'contact_length_mm_before':oldlength,
                 'contact_length_mm_after':newlength,'coverage':coverage,
                 'continuous_joins':new['workflow']['coverage_review']['continuous_joins'],
                 'reload_strokes_per_load':module['reload_strokes_per_load'],
                 'black_and_title_exactly_preserved':True,'runtime_load_ok':True,'skipped_paths':0}
        save(folder/'检查报告.json',summary);summaries.append(summary)
        before=narrow_render(old);after=narrow_render(new);source=aligned_reference(img,old)
        board=make_board([source,before,after],['原图（对齐画布）','修改前 · 同一笔宽','适度补色版 · 保留留白'],f'{n:02d} {TITLES[n-1]}｜原图与可执行轨迹对比')
        board.save(folder/'原图与轨迹对比.jpg',quality=95)
        after.save(folder/'候选轨迹预览.png')
        if n==1:
            r=old['workflow']['source_revision']['source_transform']['scale'];o=old['workflow']['source_revision']['source_transform']['offset']
            box=(int(o[0]),int(610*r+o[1]),int(600*r+o[0]),int(1055*r+o[1]))
            zoom=make_board([p.crop(box) for p in (source,before,after)],['原图局部','修改前 · 同一笔宽','适度补色 · 不铺满'], '第一幅左前景山石｜照片所示稀疏上色区域',1800,680)
            zoom.save(OUT/'第一幅山石局部对比.jpg',quality=96)
        boards.append(board)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    assert hashes=={str(p):sha(Path(p)) for p in hashes},'Runtime source changed'
    save(OUT/'检查汇总.json',{'runtime_files_unchanged':True,'protected_hashes':hashes,'artworks':summaries})
    contact=Image.new('RGB',(1500,5*493),'white')
    for i,b in enumerate(boards):contact.paste(b.resize((1500,493),Image.Resampling.LANCZOS),(0,i*493))
    contact.save(OUT/'五幅上色对比总览.jpg',quality=95)
    rows=[]
    for n,title in enumerate(TITLES,1):
        name=f'{n:02d}_{title}'
        rows.append(f'<h2>{n}. {title}</h2><a href="{name}/原图与轨迹对比.jpg"><img src="{name}/原图与轨迹对比.jpg"></a>')
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>五幅适度补色预览</title><style>body{font-family:system-ui;background:#f6f4ed;color:#243a35;margin:24px}img{width:100%;max-width:2100px}p{max-width:1000px;line-height:1.8}a{color:inherit}</style><h1>五幅适度补色预览</h1><p>未替换 Web 正式画稿。所有上色逐点下压为 0.20–0.30 mm；黑线和题字保持各自基线。仅在原有色带附近适度补色，保留留白。每种颜色按部位依次完成，区域内能安全连接的笔画合并；新增连续接触不超过 80 mm，不能连接的留白处抬笔。每次蘸取上限为 8 笔，同时保留 100 mm 累计画程和 60 秒补蘸条件。预览直接绘制候选 JSON 轨迹，前后均假设笔宽 0.8 mm，颜色及渗化尚未实测。</p>'+''.join(rows),encoding='utf-8')


if __name__=='__main__':main()
