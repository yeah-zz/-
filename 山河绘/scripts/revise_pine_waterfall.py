"""Offline revision of painting 01 from the user's annotated source image."""
from pathlib import Path
import sys, json, copy, shutil, hashlib
from collections import Counter
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web.catalog import ARTWORKS
from artwork_order import ink_payload
from painting_session import save, session_lock
from ink_control import refresh_stats
from geometry import CanvasMap
from scripts.order_selected_artworks import ORDERS
from scripts.consolidate_selected_artworks import digest, stroke_data

RIDGES = [
 [[356,351],[399,333],[438,325],[476,304],[501,295],[528,277],[543,279],[568,301],[592,309]],
 [[580,282],[609,268],[634,241],[649,243],[671,219],[687,218],[704,234],[726,239]],
 [[1045,254],[1072,237],[1094,224],[1107,228],[1126,247],[1150,258],[1182,266]],
 [[1174,288],[1204,263],[1234,247],[1248,247],[1264,260],[1289,266],[1320,282]],
 [[425,413],[455,402],[478,380],[500,365],[516,364],[538,381]],
 [[597,477],[634,464],[661,444],[683,426],[702,420],[718,430],[735,447],[758,455]],
 [[785,578],[813,561],[835,542],[856,540],[876,523],[895,513],[918,518],[936,535]],
 [[656,644],[681,625],[706,613],[725,592],[745,585],[763,589],[782,603]],
 [[461,609],[483,596],[505,600],[525,615],[543,622]],
 [[1406,638],[1424,617],[1447,605]],
]
ANNOTATIONS = [(57.5,15.5,.25,'主峰轮廓'),(73.6,34.5,.25,'中景山体轮廓'),
 (43.6,24.8,.15,'远山轮廓'),(55.9,25.3,.15,'皴线'),(94.8,33,.25,'右侧山体轮廓'),
 (18.6,77.1,.30,'近景山石轮廓'),(16.8,53.2,.30,'树干'),(26.7,47.8,.20,'树枝'),
 (23.4,33.1,None,'松叶绿色')]


def subject(st, points):
    x,y=np.mean(points,axis=0);role=st.get('role');edge=st.get('source_edge_index')
    if role=='title':return '题字'
    if role in ('pine_needle','leaf_color','branch','trunk') or edge in (15,16,17):return '整棵松树'
    if st.get('distant_ridge_index') is not None:return '远山'
    if edge==0:return '左岸山石'
    if edge in (1,3,4,5):return '左前景山石'
    if edge==2:return '中前景山石'
    if st.get('designed_subject','').startswith('亭子') or 1210<x<1430 and 525<y<688:return '亭子'
    if st.get('designed_subject','').startswith('小船') or 753<x<977 and 783<y<851:return '小舟'
    if edge in (10,11):return '主峰'
    if edge in (9,12):return '主峰右侧山峰'
    if edge in (6,7,13,14) or x>970 and y<580:return '瀑布与右侧崖壁'
    if edge==8 or x>1100 and 660<y<780:return '亭下山岛'
    if 720<x<1150 and 540<y<710:return '远山'
    if y<490 and x>430:return '主峰' if x<900 else '主峰右侧山峰'
    if x<720 and y<760:return '左岸山石' if x<150 else '左前景山石'
    if x<720 and y<1000:return '左前景山石' if x<420 else '中前景山石'
    return '湖面水纹'


def order(strokes, ratio, offset, ink):
    groups={s:[] for s in ORDERS[0]}
    for st in strokes:
        points=(np.array(st['points'])-offset)/ratio
        region=subject(st,points);st['drawing_region']=region;st['subject_order']=ORDERS[0].index(region)
        if region=='题字' and 'title_character' not in st:
            x=np.mean(points[:,0]);st['title_character']='山河绘'[0 if x<300 else 1 if x<430 else 2]
        groups[region].append(st)
    result=[]
    for region in ORDERS[0]:
        pending=groups[region]
        # Parents stay contiguous; original segments are never reversed.
        parents={}
        for st in pending:parents.setdefault(st.get('parent_stroke_id',st['id']) if ink else st['id'],[]).append(st)
        chunks=list(parents.values())
        for chunk in chunks:chunk.sort(key=lambda s:s.get('segment',0))
        def priority(chunk):
            st=chunk[0]
            if region=='题字':return '山河绘'.index(st['title_character'])
            return 0 if st.get('role') in ('trunk','branch','coarse','far','far_outline') else 1
        end=None
        while chunks:
            best=min(range(len(chunks)),key=lambda i:(priority(chunks[i]),
                np.linalg.norm(np.array(chunks[i][0]['points'][0])-end) if end is not None else np.mean(chunks[i][0]['points'],axis=0)[1]))
            chunk=chunks.pop(best);result.extend(chunk);end=np.array(chunk[-1]['points'][-1])
    return result


def revise():
    folder,_,_,source=ARTWORKS[0];path=ROOT/source;package=path.parent.parent;task=path.parent
    with session_lock(ROOT/'robot_runs'):
        if (ROOT/'robot_runs'/'blue_wash_motion.lock').exists():raise ValueError('Painting is active')
        if json.loads(path.read_text(encoding='utf-8-sig'))['workflow'].get('source_revision',{}).get('leaf_mode'):
            raise ValueError('松叶已改为叶团，请用simplify_pine_canopy修订，不能覆盖回旧针叶。')
        history=package/'历史版本'/'20261006_原图下压重整前';history.mkdir(parents=True,exist_ok=True)
        backup=history/'painting.json'
        if not backup.exists():shutil.copy2(path,backup)
        original=json.loads(backup.read_text(encoding='utf-8-sig'));bundle=copy.deepcopy(original)
        from web.execution import prepare
        prepare(backup)
        w,h=original['ink_job']['meta']['image_size'];scale=CanvasMap([w,h]).mm_per_px
        # Exact fit used by the original builder, including its paper margin.
        source_image=cv2.imdecode(np.fromfile(str(package/'用户原图.png'),np.uint8),cv2.IMREAD_COLOR)
        sh,sw=source_image.shape[:2];margin=int(np.ceil(21.5/scale))
        ratio=min((w-2*margin)/sw,(h-2*margin)/sh)
        offset=np.array([(w-round(sw*ratio))//2,(h-round(sh*ratio))//2])
        old_ink=original['ink_job']['meta']['ink_execution']['base_strokes']
        old_colors=original['workflow']['modules']['color']['strokes']
        ink=[];colors=[];removed=[];transferred=[]
        needle_mask=np.zeros((h,w),np.uint8)
        for st in old_ink:
            if st.get('role')=='pine_needle':cv2.polylines(needle_mask,[np.rint(st['points']).astype(np.int32)],False,1,5)
        needle_mask=cv2.dilate(needle_mask,np.ones((9,9),np.uint8))
        for prior in old_ink:
            st=copy.deepcopy(prior);q=(np.array(st['points'])-offset)/ratio;x,y=q.mean(axis=0)
            role=st.get('role');edge=st.get('source_edge_index');n=len(q)
            if role=='pine_needle':
                # Keep the traced fan geometry, draw it once with green pigment.
                st.update(id='green-leaf-'+str(prior['id']),color='green',role='leaf_color',
                    recipe='pine-green',region_id='pine-needles',path_style='source-pine-fan',
                    press=[.20]*n,pressure_mm=None,target_opacity=[.40]*n,pigment_recipe_calibrated=False)
                st.pop('parent_stroke_id',None);st.pop('segment',None);st.pop('segments',None)
                refresh_stats(st,scale);st['target_width_mm']=[st['width_mm']]*n;st['brush_width_mm_estimate']=st['width_mm']
                colors.append(st);transferred.append(prior['id']);continue
            structure=role=='title' or 'designed_subject' in st or edge is not None
            if role=='far_texture' and x<680 and 330<y<650 and not structure:
                removed.append(st['id']);continue
            # Remove isolated flecks and accidental internal traces, retain structures.
            if not structure and ((role in ('texture','far_texture','detail') and st['length_mm']<5)
                                  or (role=='coarse' and x<680 and 330<y<650 and st['length_mm']<5)):
                removed.append(st['id']);continue
            if role=='title':ink.append(st);continue
            if edge==15 or (edge==16 and x<300 and y>510) or (edge is None and role=='coarse' and x<300 and 500<y<740):
                st['role']='trunk';press=[.30]*n
            elif edge==16:
                st['role']='branch';press=[.30 if px<300 and py>510 else .20 for px,py in q]
            elif edge==17 or edge is None and role=='coarse' and x<680 and 330<y<630:
                st['role']='branch';press=[.20]*n
            elif role in ('far_texture','texture'):
                press=[.15]*n
            elif role=='far' or edge in (9,10,6,7,13,14) or role=='coarse' and x>970 and y<580:
                press=[.25]*n
            elif role=='coarse':press=[.30]*n
            else:press=list(st['press'])
            st['press']=press;refresh_stats(st,scale);ink.append(st)
        # Add the missing misty mountain ridges as complete open contours.
        template=copy.deepcopy(next(st for st in ink if st['role']=='far'))
        for i,line in enumerate(RIDGES):
            q=np.array(line,float)*ratio+offset
            arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))]*scale
            t=np.linspace(0,arc[-1],max(2,int(np.ceil(arc[-1]/.35))+1))
            points=np.column_stack([np.interp(t,arc,q[:,k]) for k in (0,1)])
            st=copy.deepcopy(template)
            for key in ('source_edge_index','source_path_ids','parent_stroke_id','segment','segments','title_character'):st.pop(key,None)
            st.update(id='mist-ridge-%02d'%i,points=points.tolist(),press=[.15]*len(points),speed=[40.]*len(points),
                role='far_outline',name='远山完整山脊',distant_ridge_index=i,semantic_region=1)
            ink.append(refresh_stats(st,scale))
        # Existing long mountain/lake fills stay; replace leaf sweeps with traced green fans.
        for prior in old_colors:
            st=copy.deepcopy(prior);q=np.rint(st['points']).astype(int)
            on_leaf=np.mean(needle_mask[q[:,1],q[:,0]])>.35
            if 'pine-cluster' in str(st.get('region_id','')) or on_leaf:
                removed.append(st['id']);continue
            colors.append(st)
        ink=order(ink,ratio,offset,True)
        ordered=[]
        for color in ('blue','green'):ordered.extend(order([st for st in colors if st['color']==color],ratio,offset,False))
        plan=bundle['workflow'];plan['modules']['outline']['strokes']=copy.deepcopy(ink)
        plan['modules']['outline']['stroke_count']=len(ink)
        plan['modules']['color']['strokes']=ordered;plan['modules']['color']['stroke_count']=len(ordered)
        bundle['ink_job']=ink_payload(original['ink_job'],ink)
        from ink_control import build_plan,InkPolicy
        from shuimo import ink_palette_index
        _,stats=build_plan(ink,ink_palette_index(),scale,InkPolicy(**bundle['ink_job']['meta']['ink_execution']['policy']))
        bundle['ink_job']['meta']['quality_stats']=stats
        plan.pop('joining',None)
        plan['fill_report']={c:dict(strokes=sum(s['color']==c for s in ordered),
            coverage_not_remeasured=True) for c in ('blue','green')}
        revision=dict(format='shanhe-user-source-revision-v1',date='2026-10-06',
            baseline='历史版本/20261006_原图下压重整前/painting.json',
            source_image_sha256=hashlib.sha256((package/'用户原图.png').read_bytes()).hexdigest(),
            source_transform=dict(scale=ratio,offset=offset.tolist()),
            annotations=[dict(x_percent=x,y_percent=y,press_mm=p,part=part) for x,y,p,part in ANNOTATIONS],
            title_geometry_preserved=True,title_last=True,leaf_color='green',leaf_press_mm=.20,
            removed_ids=removed,black_leaf_ids_transferred_to_green=transferred,
            added_far_ridge_ids=['mist-ridge-%02d'%i for i in range(len(RIDGES))],
            before_counts=dict(black=len(old_ink),color=len(old_colors)),
            after_counts=dict(black=len(ink),color=len(ordered)),
            note='按用户原图坐标修订；保留长上色路径，减少零碎短笔；松叶由黑线阶段移至绿色阶段；墨色与线宽为模型估计。')
        plan['source_revision']=revision
        plan['note']=revision['note']
        calibration=dict(role_pressures_mm=dict(trunk=.30,branch=.20,coarse=.30,far=.25,far_outline=.15,texture=.15,far_texture=.15,title=.15),
            pressure_knots_mm=[.15,.20,.25,.30],fresh_press_factor=1.,extra_global_pressure_offset_mm=0.,
            split_seam_pressure_fade=False,preview_brush_width_model_is_estimate=True,
            note='树干转入树枝的同一长路径逐点使用0.30/0.20；松叶绿色0.20。')
        bundle['ink_job']['meta']['pressure_calibration']=plan['pressure_calibration']=calibration
        from run_color_from_job import validate_plan
        from run_json_painting import manifest_for
        validate_plan(plan,bundle['ink_job']);manifest_for(ink,plan,bundle['ink_job']['meta'])
        # Validate the pressure actually used by the color executor, without hardware.
        from run_color_experiment import module_lines
        from config import ArmConfig
        for color in ('blue','green'):
            lines,skipped=module_lines(plan['modules']['color'],color,plan['modules']['color']['motion_candidate'])
            assert not skipped
            for st,line in zip([s for s in ordered if s['color']==color],lines):
                contacts=[p for tag,p in line['sequence'] if tag in ('touch','draw')]
                assert np.allclose(np.array(contacts)[:,2]-ArmConfig.PAPER_Z,st['press'],atol=1e-8,rtol=0)
        revision['stroke_data_sha256']=digest(stroke_data(bundle))
        save(path,bundle);save(task/'black_only.json',bundle['ink_job'])
        for p in task.glob('*_黑线任务.json'):save(p,bundle['ink_job'])
        for p in task.glob('*_计划.json'):save(p,plan)
        save(task/'原图修订与下压说明.json',revision)
        audit=json.loads((task/'离线检查.json').read_text(encoding='utf-8-sig'))
        if not (history/'离线检查.json').exists():shutil.copy2(task/'离线检查.json',history/'离线检查.json')
        audit.update(ink_strokes=len(ink),source_transform=revision['source_transform'],
            role_counts=dict(Counter(s['role'] for s in ink)),source_revision=revision,
            black_runtime_pressures_unchanged=False,color_runtime_pressures_unchanged=False,
            user_requested_pressures_verified=True,robot_connected=False)
        audit['colors']={c:dict(strokes=sum(s['color']==c for s in ordered),
            source_leaf_strokes=sum(s['color']==c and s.get('role')=='leaf_color' for s in ordered)) for c in ('blue','green')}
        audit.pop('joining',None)
        save(task/'离线检查.json',audit)
        save(task/'笔序修改校验.json',dict(robot_connected=False,all_strokes_ids_geometry_press_speeds_preserved=False,
            user_requested_source_revision=True,title_after_all_colors=True,ink_count=len(ink),color_count=len(ordered)))
        note=['# 第一幅：原图与下压修订（2026-10-06）','',
              '主峰与中景、右侧山体外轮廓0.25 mm；远山完整山脊与山体内部皴线0.15 mm；近景山石轮廓、树干0.30 mm；树枝0.20 mm。',
              '松叶原有扇形线条移到绿色上色阶段，采用0.20 mm；题字保留原0.15 mm轮廓，所有颜色完成后最后写。',
              '删去孤立短碎笔，补出10条远山山脊；原有较长山体与湖面上色路径保留。',
              '按原图实际缩放与留白换算标记坐标，画布映射和PAPER_Z设置保持原项目规则。',
              '旧版在历史版本/20261006_原图下压重整前。预览为路径示意，实际笔宽、色浓需上纸验证。','']
        (task/'原图修订与下压说明.md').write_text('\n'.join(note),encoding='utf-8')
        readme=package/'阅读说明.md';text=readme.read_text(encoding='utf-8')
        start=text.index('笔数：');end=text.index('程序和现场配置',start)
        text=text[:start]+('当前笔数：山水勾勒 %d 笔，蓝色 %d 笔，绿色 %d 笔，最后题字 13 笔。\n\n本幅已按用户原图重新调整远山轮廓、下压与绿色松叶。详见任务资料/原图修订与下压说明.md。\n\n'%(len(ink)-13,sum(s['color']=='blue' for s in ordered),sum(s['color']=='green' for s in ordered)))+text[end:]
        text=text.replace('本次笔数和原有路径、下压量保持不变，题字仍使用原轮廓。','当前版本已按原图修订，题字仍使用原轮廓。')
        readme.write_text(text,encoding='utf-8')
        # Replace stale previews, retaining the old images alongside the old JSON.
        preview=package/'预览与下压标注'
        from PIL import Image,ImageDraw,ImageFont
        from shuimo import press_to_width_mm
        canvas=np.full((h,w,3),255,np.uint8)
        for st in ink+ordered:
            color=(45,45,45) if st in ink else (175,125,72) if st['color']=='blue' else (97,152,97)
            q=np.rint(st['points']).astype(int)
            for j,(a,b) in enumerate(zip(q,q[1:])):
                width=max(1,round(press_to_width_mm(st['press'][j])/scale))
                cv2.line(canvas,tuple(a),tuple(b),color,width,cv2.LINE_AA)
        for filename in ('任务预览_线宽色彩为估计.png','执行路径预览_非实测线宽.png','各部位下压mm标注.png'):
            p=preview/filename
            if p.exists() and not (history/filename).exists():shutil.copy2(p,history/filename)
        cv2.imencode('.png',canvas)[1].tofile(str(preview/'任务预览_线宽色彩为估计.png'))
        shutil.copy2(preview/'任务预览_线宽色彩为估计.png',preview/'执行路径预览_非实测线宽.png')
        board=Image.new('RGB',(1600,1300),'#f7f7f0');draw=ImageDraw.Draw(board)
        font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
        draw.text((30,16),'松瀑亭舟 · 按用户原图设置下压与绿色松叶',font=font,fill='#304c3b')
        picture=Image.fromarray(cv2.cvtColor(source_image,cv2.COLOR_BGR2RGB));picture.thumbnail((1540,1000))
        ox=(1600-picture.width)//2;oy=65;board.paste(picture,(ox,oy))
        for i,(x,y,p,part) in enumerate(ANNOTATIONS,1):
            px=ox+x/100*picture.width;py=oy+y/100*picture.height
            draw.ellipse((px-16,py-16,px+16,py+16),fill='#fdf9dc',outline='#b34532',width=3)
            draw.text((px-7,py-15),str(i),font=small,fill='#b34532')
        for i,(x,y,p,part) in enumerate(ANNOTATIONS):
            label='%d. %s：%s'%(i+1,part,'绿色上色（0.20 mm）' if p is None else '%.2f mm'%p)
            draw.text((35+(i%3)*525,1090+(i//3)*48),label,font=small,fill='#304c3b')
        draw.text((35,1255),'标记按原图百分比换算；实际执行以路径逐点下压为准。',font=small,fill='#687562')
        board.save(preview/'各部位下压mm标注.png')
        print(json.dumps(revision['after_counts']))


if __name__=='__main__':revise()
