# -*- coding: utf-8 -*-
"""Small, editable assistant-traced rock stroke supervision. Offline only."""
import sys,json,hashlib,math,time,base64,shutil
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import distance_transform_edt,gaussian_filter1d
from scipy.interpolate import PchipInterpolator
ROOT=LAB/'rock_strokes_v08_20261023';OLD=LAB/'rock_strokes_v07_20261023';INK=LAB/'runs'/'mobilenet_unet_rock_cunfa_v06_expanded'
if '--name' in sys.argv:
    import re
    name=sys.argv[sys.argv.index('--name')+1]
    if not re.fullmatch(r'rock_strokes_v08_[a-zA-Z0-9_]+',name):raise ValueError('Use a named v08 experiment under the E-drive lab')
    ROOT=LAB/name
for p in ('labels','review','runs','evaluation','source_snapshot'):(ROOT/p).mkdir(parents=True,exist_ok=True)
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)

def write(name,obj):
    (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def dense_path(points):
    p=np.asarray(points,float);s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];unique=np.r_[True,np.diff(s)>.01];p=p[unique];s=s[unique]
    u=np.linspace(0,s[-1],max(2,math.ceil(s[-1])+1))
    return np.stack([PchipInterpolator(s,p[:,k])(u) for k in range(2)],axis=1)

def targets_from_annotation(r):
    from gongbi_training.cunfa_prepare import ink_target
    rgb=np.asarray(Image.open(r['image']).convert('RGB'));h,w=rgb.shape[:2];ink,_=ink_target(rgb);density=ink[1].astype(np.float32)/255
    center=np.zeros((h,w),np.uint8);angle=np.zeros((2,h,w),np.float32);width=np.zeros((h,w),np.float32);role=np.zeros((2,h,w),np.float32)
    overlay=rgb.copy();dense=[]
    for index,stroke in enumerate(r['paths']):
        p=dense_path(stroke['points']);p[:,0]=np.clip(p[:,0],0,w-1);p[:,1]=np.clip(p[:,1],0,h-1)
        dp=np.gradient(p,axis=0);theta=np.arctan2(dp[:,1],dp[:,0]);xy=np.rint(p).astype(int);xs,ys=xy.T
        # Keep the visually selected centerline. No automatic snapping to a
        # ridge teacher or model output masquerades as an independent label.
        center[ys,xs]=1;angle[0,ys,xs]=np.cos(2*theta);angle[1,ys,xs]=np.sin(2*theta)
        which=0 if stroke['role']=='outline' else 1;role[which,ys,xs]=1
        widths=np.ones(len(p))
        normal=np.stack((-np.sin(theta),np.cos(theta)),axis=1)
        for sign in (-1,1):
            alive=np.ones(len(p),bool)
            for rad in range(1,5):
                q=np.rint(p+normal*rad*sign).astype(int);q[:,0]=np.clip(q[:,0],0,w-1);q[:,1]=np.clip(q[:,1],0,h-1)
                alive&=density[q[:,1],q[:,0]]>np.maximum(.035,density[ys,xs]*.3);widths+=alive
        width[ys,xs]=np.clip(widths,1,8)/12
        color=(25,115,240) if which==0 else (235,100,15)
        cv2.polylines(overlay,[xy],False,color,1,cv2.LINE_AA)
        cv2.putText(overlay,str(index),(int(xs[0]),int(ys[0])),cv2.FONT_HERSHEY_SIMPLEX,.32,color,1,cv2.LINE_AA)
        dense.append(dict(role=stroke['role'],points=p.round(2).tolist(),ink_evidence_mean=float(density[ys,xs].mean())))
    dist,nearest=distance_transform_edt(center==0,return_indices=True);idx=tuple(nearest)
    target=np.concatenate((np.exp(-dist[None]**2/(2*1.05**2)).astype(np.float32),angle[:,nearest[0],nearest[1]],width[idx][None],role[:,nearest[0],nearest[1]]))
    # Partial labels: annotated line and its close neighborhood only. Unknown
    # dark lines outside that tube are never background negatives.
    conf=np.zeros((h,w),np.float32);conf[dist<=2.5]=.7
    conf[(dist>2.5)&(dist<7)&(density<.10)]=.35
    conf[(dist>15)&(density<.008)]=.035
    roles_conf=((dist<2.5)*.65).astype(np.float32)
    path=ROOT/'labels'/(r['id']+'.npz');np.savez_compressed(path,target=target.astype(np.float16),confidence=conf.astype(np.float16),roles_confidence=roles_conf.astype(np.float16),line_mask=center,density=density.astype(np.float16))
    Image.fromarray(overlay).save(ROOT/'review'/(r['id']+'_overlay.png'))
    return dict(r,labels=str(path),dense_paths=dense,width=w,height=h,image_sha256=hashlib.sha256(Path(r['image']).read_bytes()).hexdigest())

def prepare():
    from gongbi_training.stroke_annotations_v08 import initial_annotations
    if any((ROOT/'runs').glob('*/training.jsonl')):raise FileExistsError('Preserve existing training inputs; use --name for a new experiment')
    rows=initial_annotations(OLD,INK);records=[]
    if '--reviewed' in sys.argv:
        path=Path(sys.argv[sys.argv.index('--reviewed')+1]).resolve()
        if path.drive.upper()!='E:':raise ValueError('Reviewed annotation input must stay on E drive')
        reviewed=json.loads(path.read_text(encoding='utf-8'));known={r['id']:r for r in rows}
        for new in reviewed['records']:
            if new['id'] not in known:raise ValueError('Unknown image identity')
            for p in new['paths']:
                if p['role'] not in ('outline','cun') or len(p['points'])<2 or not np.isfinite(np.asarray(p['points'],float)).all():raise ValueError('Invalid path')
            known[new['id']]['paths']=new['paths'];known[new['id']]['provenance']='user edited assistant initial traces; expert status not established'
        rows=list(known.values())
    for r in rows:
        records.append(targets_from_annotation(r));print('ANNOTATED',r['id'],len(r['paths']),flush=True)
    write('annotations.json',dict(provenance='assistant initial visual traces, NOT human/expert gold',human_expert_verified=False,records=rows))
    write('manifest.json',dict(records=records,complete_coverage=False,unmarked_is_unknown=True,original_pen_direction_known=False))
    from collections import Counter
    write('annotation_audit.json',dict(images=len(records),paths=sum(len(r['paths']) for r in records),images_by_split=dict(Counter(r['split'] for r in records)),paths_by_split={s:sum(len(r['paths']) for r in records if r['split']==s) for s in ('train','validation','test')},role_counts=dict(Counter(p['role'] for r in records for p in r['paths'])),source='assistant initial visual trace, not expert annotation',parent_models_have_seen_validation_image=True))
    for start in (0,4):
        batch=records[start:start+4];board=Image.new('RGB',(1300,len(batch)*285),'#f4f4ed');d=ImageDraw.Draw(board)
        for i,r in enumerate(batch):
            d.text((6,i*285+3),r['id']+' | '+r['split']+' | 蓝：结构轮廓；橙：内部线候选；均待人工复核',font=FONT,fill='black')
            for j,path in enumerate((r['image'],ROOT/'review'/(r['id']+'_overlay.png'))):
                im=Image.open(path).convert('RGB');im.thumbnail((625,245));board.paste(im,(j*650+10+(625-im.width)//2,i*285+32+(245-im.height)//2))
        board.save(ROOT/'review'/f'annotation_sheet_{start//4+1}.jpg',quality=94)
    make_editor(rows)
    print('PREPARE_COMPLETE',flush=True)

def make_editor(rows):
    data=[]
    for r in rows:
        p=dict(r);p['image_data']='data:image/png;base64,'+base64.b64encode(Path(r['image']).read_bytes()).decode();data.append(p)
    html='''<!doctype html><meta charset="utf-8"><title>山石逐笔标注复核</title><style>body{font:16px sans-serif;margin:20px;background:#f4f4ed}canvas{background:white;max-width:100%;border:1px solid #ccc;touch-action:none}button,select{font:inherit;margin:4px}textarea{display:block;width:95%;height:140px}p{max-width:1100px}</style><h2>山石逐笔标注复核</h2><p>这些是助手初始描线，未经画家确认。蓝色为结构轮廓，橙色为内部线候选；没有描到的地方视为未知。选择笔画后可拖动顶点，或“新增笔画”并逐点点击。复制JSON后请保存到E盘实验目录，不会自动下载或写入C盘。</p><select id="image"></select><select id="stroke"></select><select id="role"><option value="outline">结构轮廓</option><option value="cun">内部皴线候选</option></select><button id="new">新增笔画</button><button id="delete">删除笔画</button><button id="finish">结束新增</button><button id="copy">复制标注JSON</button><label>导入JSON <input type="file" id="load" accept=".json"></label><p id="status"></p><canvas id="canvas"></canvas><textarea id="json" placeholder="修改后点击复制标注JSON；不能把此操作视为专家确认"></textarea><script>
let data=EMBEDDED;const sel=document.getElementById('image'),st=document.getElementById('stroke'),role=document.getElementById('role'),canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),box=document.getElementById('json'),status=document.getElementById('status');let picture=new Image(),adding=false,drag=-1;
function record(){return data[+sel.value]}function current(){return record().paths[+st.value]}
function reload(){sel.replaceChildren();data.forEach((r,i)=>sel.add(new Option(r.id+' ['+r.split+']',i)));changeImage()}
function updateStrokes(){st.replaceChildren();record().paths.forEach((p,i)=>st.add(new Option(i+' '+p.role,i)));if(current())role.value=current().role;draw()}
function changeImage(){adding=false;picture.onload=()=>{canvas.width=picture.naturalWidth;canvas.height=picture.naturalHeight;updateStrokes()};picture.src=record().image_data}
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(picture,0,0);record().paths.forEach((p,i)=>{ctx.strokeStyle=p.role==='outline'?'#1973f0':'#eb640f';ctx.lineWidth=i===+st.value?2:1;ctx.beginPath();p.points.forEach(([x,y],k)=>k?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke();if(i===+st.value){p.points.forEach(([x,y])=>{ctx.fillStyle='white';ctx.fillRect(x-2,y-2,4,4);ctx.strokeRect(x-2,y-2,4,4)})}});status.textContent=adding?'新增模式：点击添加点，完成后点结束新增':'拖动所选笔画的顶点。原始笔顺与下压均未知。'}
function xy(e){let b=canvas.getBoundingClientRect();return[(e.clientX-b.left)*canvas.width/b.width,(e.clientY-b.top)*canvas.height/b.height]}
canvas.onpointerdown=e=>{let p=xy(e),s=current();if(!s)return;if(adding){s.points.push(p.map(Math.round));draw();return}let best=12;drag=-1;s.points.forEach((q,i)=>{let d=Math.hypot(q[0]-p[0],q[1]-p[1]);if(d<best){best=d;drag=i}});canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(drag<0)return;let p=xy(e);current().points[drag]=[Math.round(Math.max(0,Math.min(canvas.width-1,p[0]))),Math.round(Math.max(0,Math.min(canvas.height-1,p[1])))];draw()};canvas.onpointerup=()=>drag=-1;sel.onchange=changeImage;st.onchange=()=>{role.value=current().role;draw()};role.onchange=()=>{if(current())current().role=role.value;draw()};document.getElementById('new').onclick=()=>{record().paths.push({role:role.value,points:[],original_pen_direction_known:false});updateStrokes();st.value=record().paths.length-1;adding=true;draw()};document.getElementById('finish').onclick=()=>{adding=false;draw()};document.getElementById('delete').onclick=()=>{if(current())record().paths.splice(+st.value,1);updateStrokes()};document.getElementById('copy').onclick=async()=>{let rows=data.map(({image_data,...r})=>r);box.value=JSON.stringify({provenance:'user-edited initial traces; expert verification status must be separately recorded',human_expert_verified:false,records:rows},null,2);try{await navigator.clipboard.writeText(box.value);status.textContent='已复制。请保存到 E 盘 annotations_reviewed.json。'}catch{box.focus();box.select();status.textContent='请按Ctrl+C复制，再保存到E盘。'}};document.getElementById('load').onchange=async e=>{let obj=JSON.parse(await e.target.files[0].text()),old=new Map(data.map(r=>[r.id,r.image_data]));data=obj.records.map(r=>({...r,image_data:old.get(r.id)}));reload()};reload();</script>'''
    (ROOT/'annotation_editor.html').write_text(html.replace('EMBEDDED',json.dumps(data,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')

def reference_metrics(info,record):
    from scipy.spatial import cKDTree
    reference=record['dense_paths'];paths=[np.asarray(s['xy_pixels'],float) for s in info['strokes']]
    if not paths:return dict(annotated_point_coverage=0,mean_best_single_path_coverage=0,mean_fragments_per_reference=0)
    points=np.concatenate(paths);labels=np.concatenate([np.full(len(p),i) for i,p in enumerate(paths)]);tree=cKDTree(points);trees=[cKDTree(p) for p in paths];covered=total=0;best=[];fragments=[]
    for stroke in reference:
        p=np.asarray(stroke['points']);dist,idx=tree.query(p);good=dist<=4;covered+=int(good.sum());total+=len(p)
        best.append(max(float((t.query(p)[0]<=4).mean()) for t in trees))
        counts=np.bincount(labels[idx[good]],minlength=len(paths));fragments.append(int((counts>=max(3,len(p)*.04)).sum()))
    return dict(annotated_point_coverage=covered/max(1,total),mean_best_single_path_coverage=float(np.mean(best)),mean_fragments_per_reference=float(np.mean(fragments)),matching_tolerance_pixels=4,reference_status='assistant partial traces, not expert gold; unmarked regions ignored')

def geometry_checks():
    from gongbi_training.stroke_planner_v08 import repair_paths,plan
    d=np.zeros((90,150),np.float32);cv2.line(d,(10,30),(100,30),.7,3);prediction=np.zeros((6,90,150),np.float32);prediction[0]=1;prediction[1]=1;prediction[3]=.3
    a=np.stack((np.arange(10,44),np.full(34,30)),1);b=np.stack((np.arange(49,101),np.full(52,30)),1)
    paths,events=repair_paths([a,b],d,prediction);assert len(paths)==1 and len(events)==1
    d[:,44:49]=0;paths,events=repair_paths([a,b],d,prediction);assert len(paths)==2 and not events
    # Close parallel strokes must not become one sideways connecting stroke.
    d[:]=.7;c=a+np.array([0,6]);paths,events=repair_paths([a,c],d,prediction);assert len(paths)==2
    empty,view,replay,roles=plan(np.zeros((2,50,70),np.float32),np.zeros((6,50,70),np.float32));assert empty['planned_strokes']==0
    write('geometry_checks.json',dict(ink_supported_gap_joined=True,blank_gap_not_joined=True,nearby_parallel_strokes_not_joined=True,empty_input_valid=True,hardware_used=False))

def evaluate():
    import torch
    from gongbi_training.stroke_train_v08 import model_new,infer
    from gongbi_training.stroke_direction_v07 import model_new as old_model,infer as old_infer,plan as old_plan
    from gongbi_training.stroke_planner_v08 import plan
    from gongbi_training.model import GongbiLineNet
    from gongbi_training.infer import predict
    torch.set_num_threads(4);cv2.setNumThreads(2)
    new=model_new().cuda();new.load_state_dict(torch.load(ROOT/'best.pt',map_location='cuda',weights_only=False)['model']);new.eval()
    previous=old_model().cuda();previous.load_state_dict(torch.load(OLD/'best.pt',map_location='cuda',weights_only=False)['model']);previous.eval()
    ink_model=GongbiLineNet().cuda();ink_model.load_state_dict(torch.load(INK/'best.pt',map_location='cuda',weights_only=False)['model']);ink_model.eval()
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'));results=[];previews={};plans=[]
    for r in manifest['records']:
        if r['split']=='train':continue
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));h,w=rgb.shape[:2];folder=ROOT/'evaluation'/r['id'];folder.mkdir(exist_ok=True)
        before=old_infer(previous,rgb);after=infer(new,rgb);padded=np.pad(rgb,((32,32),(32,32),(0,0)),mode='edge');ink=predict(ink_model,padded,'cuda',tile=320)[:,32:32+h,32:32+w]
        baseline,baseline_view,_=old_plan(ink,before);model_only,_,_=old_plan(ink,after[:4]);final,view,replay,roles=plan(ink,after)
        np.savez_compressed(folder/'raw_predictions.npz',previous=before,continued=after,ink=ink)
        for name,image in [('original',rgb),('previous_plan',baseline_view),('continued_plan',view),('continuous_width_replay',replay),('role_candidates',roles),('centerline_probability',np.uint8(np.clip(after[0]*255,0,255)))]:Image.fromarray(image).save(folder/(name+'.png'))
        (folder/'planned_trajectory.json').write_text(json.dumps(final,ensure_ascii=False),encoding='utf-8');plans.append(dict(id=r['id'],**final))
        row=dict(id=r['id'],split=r['split'],previous=reference_metrics(baseline,r),model_only=reference_metrics(model_only,r),continued=reference_metrics(final,r),previous_paths=baseline['planned_strokes'],model_only_paths=model_only['planned_strokes'],continued_paths=final['planned_strokes'],ink_supported_joins=len(final['ink_supported_joins']))
        # Role evaluation is restricted to the explicitly traced lines. Its
        # target itself is provisional and is never called cunfa accuracy.
        role_correct=role_count=0
        for p in r['dense_paths']:
            xy=np.rint(p['points']).astype(int);xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1);score=after[4:6,xy[:,1],xy[:,0]].mean(1);role_correct+=int(int(np.argmax(score))==(0 if p['role']=='outline' else 1));role_count+=1
        row['assistant_role_agreement']=role_correct/max(1,role_count);results.append(row);previews[r['id']]=[rgb,Image.open(ROOT/'review'/(r['id']+'_overlay.png')).convert('RGB'),baseline_view,view,replay,roles]
        print('EVALUATED',json.dumps(row),flush=True)
    write('evaluation/metrics.json',dict(reference_type='assistant partial initial visual traces; not expert gold',test_images_not_used_in_v08_training=True,parent_model_saw_validation_image=True,per_image=results))
    test=[r for r in results if r['split']=='test'];aggregate={stage:{key:float(np.mean([r[stage][key] for r in test])) for key in ('annotated_point_coverage','mean_best_single_path_coverage','mean_fragments_per_reference')} for stage in ('previous','model_only','continued')};write('evaluation/aggregate.json',aggregate)
    labels=['原图','助手初始描线（待复核）','上一版轨迹','本轮模型＋断线修复','连续笔宽回放（非实画）']
    ids=[r['id'] for r in test];board=Image.new('RGB',(2000,350*len(ids)),'#f4f4ed');draw=ImageDraw.Draw(board)
    for i,identity in enumerate(ids):
        y=i*350;draw.text((8,y+3),identity+' | 未用于本轮训练；标注不完整，不代表全部正确线条',font=FONT,fill='black')
        for j,label in enumerate(labels):
            draw.text((j*400+8,y+30),label,font=FONT,fill='black');im=Image.fromarray(previews[identity][j]) if isinstance(previews[identity][j],np.ndarray) else previews[identity][j].copy();im=im.convert('RGB');im.thumbnail((382,280));board.paste(im,(j*400+9+(382-im.width)//2,y+62+(280-im.height)//2))
    board.save(ROOT/'evaluation'/'comparison.jpg',quality=95)
    roles_board=Image.new('RGB',(1200,330*len(ids)),'#f4f4ed');draw=ImageDraw.Draw(roles_board)
    for i,identity in enumerate(ids):
        draw.text((8,i*330+4),'原图 / 初始标注 / 模型角色候选：蓝轮廓、橙内部、灰不确定（未经专家确认）',font=FONT,fill='black')
        for j,index in enumerate((0,1,5)):
            a=previews[identity][index];im=Image.fromarray(a) if isinstance(a,np.ndarray) else a.copy();im=im.convert('RGB');im.thumbnail((380,280));roles_board.paste(im,(j*400+8+(380-im.width)//2,i*330+40+(280-im.height)//2))
    roles_board.save(ROOT/'evaluation'/'role_candidates.jpg',quality=94)
    geometry_checks();make_viewer(plans);report(results,aggregate)
    print('EVALUATION_COMPLETE',flush=True)

def make_viewer(plans):
    html='''<!doctype html><meta charset="utf-8"><title>v08 离线轨迹</title><style>body{font:16px sans-serif;margin:24px;background:#f4f4ed}canvas{max-width:100%;height:auto;background:white;border:1px solid #ccc}button,select{font:inherit}</style><h2>山石轨迹与断线修复：离线回放</h2><p>仅显示像素轨迹及估计宽度。相对下压没有毫米或力的标定，尚未连接机械臂。结构与内部线均为候选。</p><select id="case"></select><button id="play">播放/暂停</button><input type="range" min="0" max="1000" value="1000" id="progress"><label><input type="checkbox" id="width" checked>估计笔宽</label><label><input type="checkbox" id="joins">标出修复段</label><p id="state"></p><canvas id="canvas"></canvas><script>
const data=EMBEDDED;const sel=document.getElementById('case'),canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),slider=document.getElementById('progress'),wide=document.getElementById('width'),joins=document.getElementById('joins'),state=document.getElementById('state');let running=false;data.forEach((r,i)=>sel.add(new Option(r.id,i)));function draw(){const p=data[+sel.value];canvas.width=p.image_width;canvas.height=p.image_height;ctx.fillStyle='white';ctx.fillRect(0,0,canvas.width,canvas.height);let total=p.strokes.reduce((a,s)=>a+s.xy_pixels.length-1,0),limit=total*slider.value/1000,count=0;for(const s of p.strokes){ctx.lineCap='butt';for(let k=1;k<s.xy_pixels.length;k++){if(++count>limit)break;ctx.strokeStyle=wide.checked?'rgba(0,0,0,'+s.ink_density[k]+')':'hsl('+(s.order*47%360)+' 60% 40%)';ctx.lineWidth=wide.checked?s.width_pixels[k]:1;ctx.beginPath();ctx.moveTo(...s.xy_pixels[k-1]);ctx.lineTo(...s.xy_pixels[k]);ctx.stroke();}}if(joins.checked){ctx.strokeStyle='red';ctx.lineWidth=2;for(const j of p.ink_supported_joins){ctx.beginPath();j.curve.forEach(([x,y],k)=>k?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke()}}state.textContent=p.planned_strokes+'条候选笔画；'+p.ink_supported_joins.length+'处短间隙修复。真实笔画身份尚未确认。'}sel.onchange=draw;slider.oninput=draw;wide.onchange=draw;joins.onchange=draw;document.getElementById('play').onclick=()=>{running=!running;if(running&&slider.value>=1000)slider.value=0};setInterval(()=>{if(running){slider.value=Math.min(1000,+slider.value+4);draw();if(slider.value>=1000)running=false}},30);draw();</script>'''
    (ROOT/'trajectory_viewer.html').write_text(html.replace('EMBEDDED',json.dumps(plans,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')

def report(rows,aggregate):
    summary=json.loads((ROOT/'training_summary.json').read_text(encoding='utf-8'));audit=json.loads((ROOT/'annotation_audit.json').read_text(encoding='utf-8'))
    table='\n'.join(f"| {name} | {aggregate[k]['annotated_point_coverage']:.3f} | {aggregate[k]['mean_best_single_path_coverage']:.3f} | {aggregate[k]['mean_fragments_per_reference']:.2f} |" for k,name in [('previous','v07模型＋旧规划'),('model_only','v08模型＋旧规划'),('continued','v08模型＋间隙修复')])
    text=f'''# v08 部分逐笔监督与断线修复实验

8张图上逐条描绘了{audit['paths']}条初始线：训练49条/4图、验证11条/1图、对照30条/3图。所有描线来自助手视觉判断，不是人工画家或专家标注；可能存在坐标、角色、笔画拆分错误。未标注区域保持未知，不当作没有线。线角色只分结构轮廓和内部线候选，不含可靠的披麻、斧劈等皴法名称，也不知道原始笔顺。

从v07权重初始化新6通道模型：中心线、cos2θ、sin2θ、相对宽度、轮廓候选、内部线候选。训练三轮共{summary['successful_updates']}次有效更新，耗时{summary['elapsed_seconds']:.1f}秒。混合49条训练描线、180个已知山石几何样本（含断续/飞白增强）及原训练集的低权重自动标签。程序样本和增强均不计作新增真实画作。

检查点按固定验证损失选出：第{summary['selected']['round']}轮第{summary['selected']['step']}步。验证图曾被父模型训练使用，所以这不是严格的全流程独立验证。3张本轮对照图未用于v08训练或检查点选择，但以前已在开发中查看；本轮标注也尚未专家复核。

## 对照指标

仅在30条助手初始描线上，以4像素容差检查路径覆盖。未标注区域没有全量精确率可报告；这些值不是原图完整提线率、人工准确率或泛化能力。单条覆盖越高，表明较多参考线能由同一条计划路径解释。片段数减少必须与覆盖率一起看，不能删线冒充改善。

| 方法 | 已描线点覆盖 | 最佳单条路径覆盖 | 每条参考线对应片段数 |
|---|---|---|---|
{table}

## 规划与角色

保留方向脊线规划框架。短间隙修复需方向相合、原墨迹支持和模型方向支持，最大间隙9像素，使用端部切向一致的曲线；不会仅因模型分数高就跨空白补线。修复段在HTML回放中可标红检查。合并后重新计算宽度，避免旧片段端部的人为提笔导致接头变细。

结构/内部线角色来自少量助手初始描线与程序几何监督，分数未经校准，低置信显示灰色。不能把橙色线直接理解为准确识别出的皴法，更不能把角色图当作专家审定结果。

毛笔宽度仍由图像截面和模型候选估计，上限8像素是预览约束。relative_pressure只是0..1建议，非毫米/力。physical_z、physical_speed、paper_transform均为空；ready_for_robot=false、actually_drawn=false。未连接、使能或移动机械臂，回放不是实画或毛笔物理仿真。

## 可复核标注

annotation_editor.html可选图、选笔画、拖动顶点、增删笔画并切换角色。复制JSON后应保存到E盘annotations_reviewed.json。页面不会自动下载到C盘。新实验可用prepare --reviewed读取修改文件，并加--name rock_strokes_v08_reviewed_20261023写入新的E盘目录，再对同名实验train、evaluate；已有训练目录禁止重做prepare以免改写输入。human_expert_verified默认false，编辑页面不会自动变成专家确认。
'''
    (ROOT/'REPORT.md').write_text(text,encoding='utf-8')
    files=[Path(__file__),Path(__file__).with_suffix('.ps1')]+[Path(__file__).resolve().parents[1]/'gongbi_training'/name for name in ('stroke_annotations_v08.py','stroke_train_v08.py','stroke_planner_v08.py')]
    for path in files:shutil.copy2(path,ROOT/'source_snapshot'/path.name)

def verify():
    import torch
    planfiles=[]
    for folder in ('evaluation','streamline_evaluation','streamline_filtered'):planfiles+=list((ROOT/folder).glob('*/planned_trajectory.json'))
    for p in planfiles:
        r=json.loads(p.read_text(encoding='utf-8'));assert r['physical_z'] is None and not r['ready_for_robot'] and not r['actually_drawn']
        for s in r['strokes']:
            assert len(s['xy_pixels'])==len(s['width_pixels'])==len(s['relative_pressure'])==len(s['ink_density']);assert np.isfinite(s['xy_pixels']).all();assert 0<=min(s['relative_pressure'])<=max(s['relative_pressure'])<=1
    hashes={}
    for p in [ROOT/'best.pt']+[ROOT/'runs'/f'round_{ri}'/'best.pt' for ri in (1,2,3)]:
        ck=torch.load(p,map_location='cpu',weights_only=False);assert all(torch.isfinite(v).all() for v in ck['model'].values());hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    images=[]
    for folder in ('evaluation','streamline_evaluation','streamline_filtered'):images+=list((ROOT/folder).rglob('*.png'))+list((ROOT/folder).glob('*.jpg'))
    for p in images:
        with Image.open(p) as im:im.verify()
    for name in ('annotation_editor','trajectory_viewer','streamline_viewer','streamline_filtered_viewer'):
        if not (ROOT/(name+'.html')).exists():continue
        html=(ROOT/(name+'.html')).read_text(encoding='utf-8');script=html.split('<script>',1)[1].split('</script>',1)[0];(ROOT/(name+'_check.js')).write_text(script,encoding='utf-8')
    from gongbi_training.stroke_streamline_v08 import trace
    d=np.zeros((85,150),np.float32);cv2.line(d,(8,40),(65,40),.7,3);cv2.line(d,(83,40),(140,40),.7,3);pred=np.zeros((6,85,150),np.float32);pred[0]=1;pred[1]=1;pred[3]=.3
    paths,stats=trace(np.stack((np.minimum(d*2,1),d)),pred,strict=True);assert paths and all(not (p[:,0].min()<65 and p[:,0].max()>83) for p in paths)
    d[:]=0;cv2.line(d,(8,40),(140,40),.7,3);paths,stats=trace(np.stack((np.minimum(d*2,1),d)),pred,strict=True);assert max(np.ptp(p,axis=0)[0] for p in paths)>100
    (ROOT/'streamline_checks.json').write_text(json.dumps(dict(blank_paper_gap_not_crossed=True,long_horizontal_stroke_preserved=True,algorithm_settings_shared=True)),encoding='utf-8')
    files=[Path(__file__),Path(__file__).with_suffix('.ps1')]+[Path(__file__).resolve().parents[1]/'gongbi_training'/name for name in ('stroke_annotations_v08.py','stroke_train_v08.py','stroke_planner_v08.py','stroke_streamline_v08.py')]
    for p in files:shutil.copy2(p,ROOT/'source_snapshot'/p.name)
    write('annotation_hashes.json',dict(annotations_sha256=hashlib.sha256((ROOT/'annotations.json').read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest(),labels={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'labels').glob('*.npz')}))
    write('verification.json',dict(weights_finite=True,hashes=hashes,trajectory_arrays_finite=True,trajectory_files=len(planfiles),image_files=len(images),geometry_tests_passed=True,streamline_checks_passed=True,human_expert_annotations=0,robot_connected=False))
    print('VERIFIED',flush=True)

def streamline_evaluate():
    from gongbi_training.stroke_streamline_v08 import plan
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'));rows=[r for r in manifest['records'] if r['split']=='test'];results=[];plans=[]
    strict='--filtered' in sys.argv;outname='streamline_filtered' if strict else 'streamline_evaluation'
    out=ROOT/outname;out.mkdir(exist_ok=True);board=Image.new('RGB',(2000,350*len(rows)),'#f4f4ed');draw=ImageDraw.Draw(board)
    for i,r in enumerate(rows):
        folder=ROOT/'evaluation'/r['id'];z=np.load(folder/'raw_predictions.npz');p=z['continued'];ink=z['ink'];info,view,replay,roles=plan(ink,p,strict=strict)
        dest=out/r['id'];dest.mkdir(exist_ok=True);Image.fromarray(view).save(dest/'planned_trajectory.png');Image.fromarray(replay).save(dest/'continuous_replay.png');Image.fromarray(roles).save(dest/'role_candidates.png');(dest/'planned_trajectory.json').write_text(json.dumps(info,ensure_ascii=False),encoding='utf-8')
        stats=reference_metrics(info,r);results.append(dict(id=r['id'],metrics=stats,paths=info['planned_strokes'],diagnostics=info['streamline_diagnostics']));plans.append(dict(id=r['id'],**info))
        images=[Image.open(r['image']),Image.open(folder/'previous_plan.png'),Image.open(folder/'continued_plan.png'),Image.fromarray(view),Image.fromarray(replay)];labels=['原图','v07 像素邻接追踪','v08 局部断线修复','连续追踪＋去重（实验）' if strict else '方向场连续追踪（实验）','连续笔宽回放（非实画）']
        draw.text((8,i*350+3),r['id']+' | 开发回归图；完整正确性仍待专家标注',font=FONT,fill='black')
        for j,im in enumerate(images):
            draw.text((j*400+8,i*350+31),labels[j],font=FONT,fill='black');im=im.convert('RGB');im.thumbnail((382,280));board.paste(im,(j*400+9+(382-im.width)//2,i*350+62+(280-im.height)//2))
        print('STREAMLINE_RESULT',json.dumps(results[-1]),flush=True)
    board.save(out/'comparison.jpg',quality=95);write(outname+'/metrics.json',results)
    aggregate={k:float(np.mean([r['metrics'][k] for r in results])) for k in ('annotated_point_coverage','mean_best_single_path_coverage','mean_fragments_per_reference')};write(outname+'/aggregate.json',aggregate)
    # Preserve both viewers. The experimental viewer is explicitly named.
    prior=(ROOT/'trajectory_viewer.html').read_text(encoding='utf-8');make_viewer(plans);shutil.copy2(ROOT/'trajectory_viewer.html',ROOT/('streamline_filtered_viewer.html' if strict else 'streamline_viewer.html'));(ROOT/'trajectory_viewer.html').write_text(prior,encoding='utf-8')
    with (ROOT/'REPORT.md').open('a',encoding='utf-8') as f:f.write('\n\n## 方向场连续追踪开发实验\n\n局部拼接效果有限后，增加子像素方向场追踪：沿局部方向前进、横截面寻找中心，遇到明显空白、急转或重复路径即终止。该规划器在已经查看过的开发回归图上检查，不是独立盲测。相比只沿像素邻接图走线，它可以减少部分碎片；同时也可能偏离淡线或误接。因此保留为独立实验，没有替换旧方案。指标仍以同一批30条助手初始描线、4像素容差衡量，非人工准确率。\n\n'+json.dumps(aggregate,ensure_ascii=False,indent=2)+'\n')
    print('STREAMLINE_COMPLETE',json.dumps(aggregate),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='train':
        from gongbi_training.stroke_train_v08 import train
        train(ROOT,OLD/'best.pt',OLD)
    elif sys.argv[1]=='evaluate':evaluate()
    elif sys.argv[1]=='check':geometry_checks()
    elif sys.argv[1]=='verify':verify()
    elif sys.argv[1]=='streamline':streamline_evaluate()
