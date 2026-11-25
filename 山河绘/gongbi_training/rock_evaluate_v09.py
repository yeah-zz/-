# -*- coding: utf-8 -*-
"""Frozen-protocol evaluation; synthetic truth, real weak and partial traces separate."""
import json,math,hashlib,sys,shutil
from pathlib import Path
import torch,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial import cKDTree
from .model import GongbiLineNet
from .infer import predict
from .rock_multitask_v09 import model_new,infer,synthetic
from .stroke_train_v08 import model_new as old_model,infer as old_infer
from .stroke_streamline_v08 import plan as old_plan
from .rock_plan_v09 import plan
from .runtime import LAB
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
def write(root,name,obj):(root/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def load(path,kind):
    model=kind().cuda();model.load_state_dict(torch.load(path,map_location='cuda',weights_only=False)['model']);model.eval();return model
def weak_metric(ink,r):
    with np.load(r['labels']) as z:t=z['targets']/255;valid=z['weights'][0]>.1*255
    p=ink[0]>.5;y=t[0]>.3;tp=int((p&y&valid).sum());fp=int((p&~y&valid).sum());fn=int((~p&y&valid).sum());precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn)
    return dict(precision=precision,recall=recall,f1=2*tp/max(1,2*tp+fp+fn),density_mae=float(np.abs(ink[1]-t[1])[valid].mean()))
def trace_metric(info,r):
    paths=[np.asarray(p['xy_pixels']) for p in info['strokes']]
    if not paths:return dict(coverage=0,best_single_path=0,fragments=0)
    points=np.concatenate(paths);ids=np.concatenate([np.full(len(p),i) for i,p in enumerate(paths)]);tree=cKDTree(points);trees=[cKDTree(p) for p in paths];covered=total=0;best=[];frags=[]
    for stroke in r['dense_paths']:
        p=np.asarray(stroke['points']);dist,index=tree.query(p);good=dist<=4;covered+=int(good.sum());total+=len(p);best.append(max(float((t.query(p)[0]<=4).mean()) for t in trees));counts=np.bincount(ids[index[good]],minlength=len(paths));frags.append(int((counts>=max(3,len(p)*.04)).sum()))
    return dict(coverage=covered/max(1,total),best_single_path=float(np.mean(best)),fragments=float(np.mean(frags)))
def paste(board,arr,x,y,w,h):
    im=Image.fromarray(arr) if isinstance(arr,np.ndarray) else arr.copy();im=im.convert('RGB');im.thumbnail((w,h));board.paste(im,(x+(w-im.width)//2,y+(h-im.height)//2))
def activate(p):
    p=p.float();p[:,[0,3,4,5,6,7,8,9]]=p[:,[0,3,4,5,6,7,8,9]].sigmoid();p[:,1:3]=torch.nn.functional.normalize(p[:,1:3],dim=1);p[:,10:14]=torch.nn.functional.normalize(p[:,10:14],dim=1);return p
def known_benchmark(new,old,ink_model):
    rows=[]
    with torch.no_grad():
        for i in range(32):
            rgb,t,c,ids=synthetic(np.random.default_rng(500000+i));x=torch.from_numpy(rgb.transpose(2,0,1).copy())[None].cuda().float()/255;p=activate(new(x))[0].cpu().numpy();q=old(x)[0].float().cpu().numpy();q[[0,3,4,5]]=1/(1+np.exp(-q[[0,3,4,5]]));baseline_ink=ink_model(x).sigmoid()[0].cpu().numpy();mask=t[0]>.5
            pc=p[0]>.35;qc=q[0]>.35;truth=t[0]>.35
            def f1(v):return 2*int((v&truth).sum())/max(1,int(v.sum()+truth.sum()))
            dy=c[9]>0;v=dict(seed=500000+i,previous_center_f1=f1(qc),continued_center_f1=f1(pc),previous_width_mae_pixels=float(np.abs(q[3]-t[3])[mask].mean()*12),continued_width_mae_pixels=float(np.abs(p[3]-t[3])[mask].mean()*12),previous_density_mae=float(np.abs(baseline_ink[1]-t[6]).mean()),continued_density_mae=float(np.abs(p[6]-t[6]).mean()),continued_dry_appearance_mae=float(np.abs(p[9]-t[9])[dy].mean()) if dy.any() else None)
            within=[];between=[];means=[]
            for identity in np.unique(ids[ids>0]):
                e=p[10:14,ids==identity].T[::3];mean=e.mean(0);mean/=max(1e-5,np.linalg.norm(mean));within.append(float((e@mean).mean()));means.append(mean)
            if len(means)>1:
                m=np.array(means);s=m@m.T;between=s[~np.eye(len(m),dtype=bool)].tolist()
            v['within_stroke_embedding_cosine']=float(np.mean(within));v['between_stroke_embedding_cosine']=float(np.mean(between));rows.append(v)
    aggregate={key:float(np.mean([r[key] for r in rows if r[key] is not None])) for key in rows[0] if key!='seed'}
    return dict(examples=32,provenance='procedural rock geometry and alpha/holes, seeds disjoint from train/validation; NOT real painting accuracy',aggregate=aggregate,per_image=rows)

def evaluate(root,v08,ink_root):
    torch.set_num_threads(4);cv2.setNumThreads(2)
    frozen=json.loads((root/'frozen_manifest.json').read_text(encoding='utf-8'));protocol=json.loads((root/'evaluation_protocol.json').read_text(encoding='utf-8'));ck=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    assert ck['config']['frozen_manifest_sha256']==hashlib.sha256((root/'frozen_manifest.json').read_bytes()).hexdigest()
    new=load(root/'best.pt',model_new);old=load(v08/'best.pt',old_model);ink_model=load(ink_root/'best.pt',GongbiLineNet)
    known=known_benchmark(new,old,ink_model);write(root,'evaluation/known_synthetic_metrics.json',known)
    fresh=frozen['new_geometry_test'];metrics=[];preview={};plans=[]
    for r in fresh:
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));h,w=rgb.shape[:2];pad=np.pad(rgb,((32,32),(32,32),(0,0)),mode='edge');baseline_ink=predict(ink_model,pad,'cuda',tile=320)[:,32:32+h,32:32+w];p=infer(new,rgb);ink=p[[7,6]]
        folder=root/'evaluation'/r['id'];folder.mkdir(exist_ok=True);np.savez_compressed(folder/'probabilities.npz',multitask=p,baseline_ink=baseline_ink);Image.fromarray(np.uint8(np.clip(255*(1-ink[1]),0,255))).save(folder/'ink.png');Image.fromarray(np.uint8(np.clip(255*(1-baseline_ink[1]),0,255))).save(folder/'previous_ink.png')
        row=dict(id=r['id'],source_group=r['source_group'],previous=weak_metric(baseline_ink,r),continued=weak_metric(ink,r));metrics.append(row)
        if r['id'] in protocol['preview_ids']:
            q=old_infer(old,rgb);before,bv,br,brole=old_plan(baseline_ink,q,strict=True);same,sv,sr,srole=plan(baseline_ink,p);after,av,ar,arole=plan(ink,p)
            (folder/'planned_trajectory.json').write_text(json.dumps(after,ensure_ascii=False),encoding='utf-8');Image.fromarray(av).save(folder/'planned_trajectory.png');Image.fromarray(ar).save(folder/'geometric_replay.png');Image.fromarray(arole).save(folder/'role_candidates.png');plans.append(dict(id=r['id'],**after));preview[r['id']]=[rgb,bv,sv,av,ar]
            row['planned_paths']=dict(previous=before['planned_strokes'],new_model_old_ink=same['planned_strokes'],new_model_own_ink=after['planned_strokes'])
        print('NEW_SOURCE_TEST',r['id'],flush=True)
    groups=[]
    for g in sorted({r['source_group'] for r in metrics}):
        rows=[r for r in metrics if r['source_group']==g];groups.append(dict(source_group=g,examples=len(rows),**{stage:{k:float(np.mean([r[stage][k] for r in rows])) for k in ('precision','recall','f1','density_mae')} for stage in ('previous','continued')}))
    macro={stage:{k:float(np.mean([r[stage][k] for r in groups])) for k in ('precision','recall','f1','density_mae')} for stage in ('previous','continued')};write(root,'evaluation/new_source_weak_metrics.json',dict(images=len(fresh),source_groups=len(groups),source_macro=macro,groups=groups,per_image=metrics,interpretation='automatic ink mask agreement, NOT expert accuracy; book-level independence unverified'))
    board=Image.new('RGB',(2000,350*len(preview)),'#f4f4ed');draw=ImageDraw.Draw(board);labels=['新来源原图','v08＋旧墨迹模型','v09＋旧墨迹模型','v09＋自身墨迹输出','连续笔宽回放（非实画）']
    for i,identity in enumerate(protocol['preview_ids']):
        draw.text((8,i*350+3),identity+' | 训练前固定的新来源图',font=FONT,fill='black')
        for j,label in enumerate(labels):draw.text((j*400+8,i*350+31),label,font=FONT,fill='black');paste(board,preview[identity][j],j*400+9,i*350+62,382,280)
    board.save(root/'evaluation'/'new_source_pipeline.jpg',quality=95)
    # All retained new images are shown; selected pipeline previews were fixed
    # in a manifest before any model predictions or test metrics were read.
    for start in range(0,len(fresh),8):
        rows=fresh[start:start+8];board=Image.new('RGB',(1500,270*len(rows)),'#f4f4ed');d=ImageDraw.Draw(board)
        for i,r in enumerate(rows):
            d.text((8,i*270+3),r['id']+' | 原图 / v06墨迹 / v09墨迹',font=FONT,fill='black');folder=root/'evaluation'/r['id']
            for j,pth in enumerate((r['image'],folder/'previous_ink.png',folder/'ink.png')):paste(board,Image.open(pth),j*500+8,i*270+32,482,230)
        board.save(root/'evaluation'/f'all_new_ink_{start//8+1}.jpg',quality=92)
    # Legacy partial assistant traces are regression diagnostics, not a new
    # independent evaluation and never called expert ground truth.
    ref=json.loads((v08/'manifest.json').read_text(encoding='utf-8'));regress=[];legacy_images={}
    for r in ref['records']:
        if r['split']!='test':continue
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));h,w=rgb.shape[:2];p=infer(new,rgb);q=old_infer(old,rgb);pad=np.pad(rgb,((32,32),(32,32),(0,0)),mode='edge');base=predict(ink_model,pad,'cuda',tile=320)[:,32:32+h,32:32+w];b,bv,br,_=old_plan(base,q,strict=True);same,sv,sr,_=plan(base,p);a,av,ar,_=plan(p[[7,6]],p)
        regress.append(dict(id=r['id'],previous=trace_metric(b,r),new_model_old_ink=trace_metric(same,r),continued=trace_metric(a,r)));legacy_images[r['id']]=[rgb,bv,sv,av,ar]
    write(root,'evaluation/legacy_partial_trace_metrics.json',regress)
    board=Image.new('RGB',(2000,350*len(regress)),'#f4f4ed');d=ImageDraw.Draw(board)
    for i,r in enumerate(regress):
        d.text((8,i*350+3),r['id']+' | 旧开发回归图；仅30条助手初始描线',font=FONT,fill='black')
        for j,label in enumerate(labels):d.text((j*400+8,i*350+31),label if j else '旧开发图原图',font=FONT,fill='black');paste(board,legacy_images[r['id']][j],j*400+9,i*350+62,382,280)
    board.save(root/'evaluation'/'legacy_regression.jpg',quality=95)
    # Attribute previews have separate visible units and evidence limits.
    attribute_board=Image.new('RGB',(1800,320*len(protocol['preview_ids'])),'#f4f4ed');d=ImageDraw.Draw(attribute_board)
    attrlabels=['原图','中心线概率','像素宽度候选','墨色深浅','干笔外观代理（未实图验证）','角色候选：蓝轮廓/橙内部']
    for i,identity in enumerate(protocol['preview_ids']):
        r=next(r for r in fresh if r['id']==identity);rgb=np.asarray(Image.open(r['image']).convert('RGB'));p=np.load(root/'evaluation'/identity/'probabilities.npz')['multitask'];mask=(p[0]>.35)&(p[7]>.3);width=cv2.applyColorMap(np.uint8(np.clip(p[3]*12/8*255,0,255)),cv2.COLORMAP_TURBO);width[~mask]=255;dry=cv2.applyColorMap(np.uint8(np.clip(p[9]*255,0,255)),cv2.COLORMAP_TURBO);dry[~mask]=255;roles=np.full_like(rgb,255);roles[mask&(p[4]>p[5])]=[25,115,240];roles[mask&(p[4]<=p[5])]=[235,100,15];arrays=[rgb,np.uint8(255*(1-p[0])),width,np.uint8(255*(1-p[6])),dry,roles]
        for j,label in enumerate(attrlabels):d.text((j*300+5,i*320+4),label,font=FONT,fill='black');paste(attribute_board,arrays[j],j*300+7,i*320+38,286,270)
    attribute_board.save(root/'evaluation'/'appearance_attributes.jpg',quality=94)
    write(root,'evaluation/geometry_summary.json',dict(known_synthetic=known['aggregate'],new_source_weak=macro,legacy_partial=regress));make_viewer(root,plans);print('EVALUATION_COMPLETE',flush=True)

def make_viewer(root,plans):
    html='''<!doctype html><meta charset="utf-8"><title>v09 离线笔画回放</title><style>body{font:16px sans-serif;margin:24px;background:#f4f4ed}canvas{max-width:100%;height:auto;background:white;border:1px solid #ccc}</style><h2>新来源山石：离线轨迹回放</h2><p>路线来自v09中心线、方向、墨色与宽度输出。笔画身份及皴法未经专家确认；相对下压不是毫米或力。未连接机械臂，回放不是实画。</p><select id="case"></select><button id="play">播放/暂停</button><input id="slider" type="range" min="0" max="1000" value="1000"><label><input id="wide" type="checkbox" checked>估计笔宽</label><label><input id="join" type="checkbox">标红候选修复段</label><p id="status"></p><canvas id="canvas"></canvas><script>const data=EMBEDDED;let running=false;const sel=document.getElementById('case'),canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),slider=document.getElementById('slider'),wide=document.getElementById('wide'),join=document.getElementById('join');data.forEach((r,i)=>sel.add(new Option(r.id,i)));function draw(){let p=data[+sel.value];canvas.width=p.image_width;canvas.height=p.image_height;ctx.fillStyle='white';ctx.fillRect(0,0,canvas.width,canvas.height);let total=p.strokes.reduce((a,s)=>a+s.xy_pixels.length-1,0),limit=total*slider.value/1000,n=0;for(let s of p.strokes){ctx.lineCap='butt';for(let k=1;k<s.xy_pixels.length;k++){if(++n>limit)break;ctx.strokeStyle=wide.checked?'rgba(0,0,0,'+s.ink_density[k]+')':'hsl('+(s.order*47%360)+' 60% 40%)';ctx.lineWidth=wide.checked?s.width_pixels[k]:1;ctx.beginPath();ctx.moveTo(...s.xy_pixels[k-1]);ctx.lineTo(...s.xy_pixels[k]);ctx.stroke()}}if(join.checked){ctx.lineWidth=2;ctx.strokeStyle='red';for(let j of p.ink_supported_joins){ctx.beginPath();j.curve.forEach(([x,y],k)=>k?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke()}}document.getElementById('status').textContent=p.planned_strokes+'条候选笔画；修复'+p.ink_supported_joins.length+'处短间隙；相对下压未标定。'}sel.onchange=draw;slider.oninput=draw;wide.onchange=draw;join.onchange=draw;document.getElementById('play').onclick=()=>{running=!running;if(running&&slider.value>=1000)slider.value=0};setInterval(()=>{if(running){slider.value=Math.min(1000,+slider.value+4);draw();if(slider.value>=1000)running=false}},30);draw();</script>'''
    (root/'trajectory_viewer.html').write_text(html.replace('EMBEDDED',json.dumps(plans,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
