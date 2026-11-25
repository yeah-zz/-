# -*- coding: utf-8 -*-
"""High-confidence centerline fine-tuning and continuous vector-field tracing."""
from pathlib import Path
import json, math, time
import cv2, numpy as np, torch
from scipy.ndimage import gaussian_filter1d, distance_transform_edt
from PIL import Image, ImageDraw
import v24_sources as src
import v24_train as v24
import v24_eval as ev
import v21_stroke_learning as v21
from rock_v15_common import read, jsave, digest, FONT, SMALL

ROOT24=src.ROOT
ROOT=ROOT24.parent/'joint_center_curves_v25_20261026'

def force_cpu():
    # Installed torch lacks kernels for this host's RTX 5060 (sm_120).
    # Keep the experiment dependency-free and run it transparently on CPU.
    torch.nn.Module.cuda=lambda self,*args,**kwargs:self.to('cpu')
    torch.Tensor.cuda=lambda self,*args,**kwargs:self.to('cpu')

def sample(im,q):
    q=np.asarray(q,np.float32)
    return cv2.remap(im.astype(np.float32),q[:,0][None],q[:,1][None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[0]

def strict_center_target(rgb, label_path):
    h,w=rgb.shape[:2]; alpha=v21.v16.photometry(rgb)[1]
    accepted=[]; rejected=0
    for stroke in read(label_path)['strokes']:
        p=np.asarray(stroke['points'],np.float32)
        if len(p)<3: continue
        p=v21.dense(p).astype(np.float32)
        if len(p)>4: p=gaussian_filter1d(p,.35,axis=0)
        t=np.gradient(p,axis=0);t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-6);n=np.c_[-t[:,1],t[:,0]]
        offsets=np.arange(-10,10.01,.5,dtype=np.float32)
        for i in range(1,len(p)-1,2):
            xy=p[i]+offsets[:,None]*n[i]; val=sample(alpha,xy)
            peak=float(val.max()); imax=int(np.argmax(val))
            if peak<.025 or abs(float(offsets[imax]))>3.0:
                rejected+=1;continue
            cut=max(.012,.35*peak);on=val>=cut
            # Require a single local ink band around the strongest response.
            ids=np.flatnonzero(on)
            if not len(ids): rejected+=1;continue
            groups=np.split(ids,np.where(np.diff(ids)>1)[0]+1);band=next((g for g in groups if imax in g),None)
            if band is None or band[0]==0 or band[-1]==len(offsets)-1:
                rejected+=1;continue
            width=float((band[-1]-band[0]+1)*.5)
            if width<.75 or width>15.0:
                rejected+=1;continue
            weights=np.maximum(val[band]-.008,0)
            shift=float(np.sum(offsets[band]*weights)/max(float(weights.sum()),1e-6))
            if abs(shift)>3.0:
                rejected+=1;continue
            accepted.append((p[i]+n[i]*shift,peak,width))
    center=np.zeros((h,w),np.uint8); evidence=np.zeros((h,w),np.float32)
    for q,peak,width in accepted:
        x,y=np.rint(q).astype(int)
        if 0<=x<w and 0<=y<h:
            cv2.circle(center,(int(x),int(y)),1,1,-1)
            cv2.circle(evidence,(int(x),int(y)),1,float(peak),-1)
    if not center.any(): return np.zeros((h,w),np.float32),np.zeros((h,w),np.float32),accepted,rejected
    dist=distance_transform_edt(center==0).astype(np.float32)
    target=np.exp(-(dist**2)/(2*1.0**2)).astype(np.float32)
    mask=(dist<=3.0).astype(np.float32)
    return target,mask,accepted,rejected

def prepare():
    if (ROOT/'fine_tuned.pt').exists(): raise FileExistsError('Keep completed v25 experiment immutable')
    ROOT.mkdir(parents=True,exist_ok=True); (ROOT/'strict_targets').mkdir(exist_ok=True)
    rows=[]; boards=[]
    manifest=read(ROOT24/'training_manifest.json')
    for r in manifest['records']:
        if r['kind']!='new_visible_reference' or r['split']!='train': continue
        z=np.load(r['data']); rgb=z['rgb']; target,mask,accepted,rejected=strict_center_target(rgb,r['labels'])
        # Only narrow centerline neighborhoods are supervised. Earlier broad
        # paper masks may cross rock interiors, so they are intentionally omitted.
        dest=ROOT/'strict_targets'/(r['id']+'.npz'); np.savez_compressed(dest,target=target,mask=mask)
        rows.append(dict(id=r['id'],samples_accepted=len(accepted),samples_rejected_low_confidence=rejected,supervised_pixel_fraction=float(mask.mean()),target_sha256=digest(dest)))
        view=rgb.copy(); yy,xx=np.where(mask>0); keep=target[yy,xx]>.55; view[yy[keep],xx[keep]]=[230,35,35]; view[yy[~keep],xx[~keep]]=[80,150,220]
        boards.append((r['id'],rgb,view))
    board=Image.new('RGB',(1200,340*len(boards)+40),'#f5f3ec'); d=ImageDraw.Draw(board);d.text((5,5),'v25 中心线高置信参考：红=高置信中心，蓝=仅作窄带位置约束，非人工专家金标准',font=FONT,fill='black')
    for i,(key,a,b) in enumerate(boards):
        d.text((5,35+340*i),key+' 原图 / 严格中心参考',font=SMALL,fill='black');v21.paste(board,a,5,60+340*i,590,275);v21.paste(board,b,605,60+340*i,590,275)
    board.save(ROOT/'CENTER_REFERENCE_AUDIT.jpg',quality=95)
    jsave(ROOT/'reference_audit.json',dict(records=rows,rule='image-supported single ink band; peak>=.025; width .75..15px; center offset<=3px; unknown remains masked',expert_labels=False,physical_lifts_known=False))
    print('V25_PREPARED',json.dumps(rows),flush=True)

def train(steps=600):
    if not (ROOT/'strict_targets'/'U01.npz').exists(): prepare()
    if (ROOT/'fine_tuned.pt').exists(): raise FileExistsError('Keep completed checkpoint immutable')
    force_cpu()
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(252610);rng=np.random.default_rng(252610)
    manifest,pool=v24.load_pool(); model=v24.JointNet(); model.load_state_dict(torch.load(ROOT24/'joint'/'final.pt',map_location='cpu',weights_only=False)['model']); model.train()
    for p in model.node_head.parameters(): p.requires_grad_(False)
    for p in model.embedding_head.parameters(): p.requires_grad_(False)
    lr=[3e-6,1.2e-5]; enc=list(model.a.parameters())+list(model.b.parameters())+list(model.c.parameters()); dec=list(model.d.parameters())+list(model.e.parameters())+list(model.head.parameters())
    opt=torch.optim.AdamW([dict(params=enc,lr=lr[0]),dict(params=dec,lr=lr[1])],weight_decay=1e-4)
    rows=[]
    for group in pool.values():
        for row in group:
            if row['kind']=='new_visible_reference':
                z=np.load(ROOT/'strict_targets'/(row['id']+'.npz')); row['t'][0]=z['target']; row['m'][0]=z['mask']
            rows.append(row)
    groups={k:[r for r in rows if r['kind']==k] for k in ['new_visible_reference','v21_clear_replay','reviewed_replay']}
    logs=[];start=time.perf_counter()
    for step in range(1,steps+1):
        batch=[]
        for _ in range(6):
            u=rng.random(); kind='new_visible_reference' if u<.4 else 'v21_clear_replay' if u<.7 else 'reviewed_replay'; row=groups[kind][int(rng.integers(len(groups[kind])))]; pair=row['pairs'][int(rng.integers(len(row['pairs'])))] if row['pairs'] and rng.random()<.6 else None; batch.append(v24.training_crop(row,rng,pair))
        x,t,m,node,teacher,valid,points,support,weights,label,pw=[torch.from_numpy(np.stack([a[k] for a in batch])) for k in range(11)]
        factor=.25+.75*.5*(1+math.cos(math.pi*step/steps))
        for g,base in zip(opt.param_groups,lr):g['lr']=base*factor
        opt.zero_grad(set_to_none=True)
        out=model(x)[:,:5]; loss=v24.v21.train17.loss_fn(out,t,m)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite v25 loss')
        loss.backward(); grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5)
        if not torch.isfinite(grad): raise RuntimeError('Nonfinite v25 gradients')
        opt.step()
        if step%100==0:
            rec=dict(step=step,loss=float(loss.detach()),elapsed_seconds=time.perf_counter()-start); logs.append(rec);print('V25_TRAIN',json.dumps(rec),flush=True)
    model.eval(); torch.save(dict(model=model.state_dict(),steps=steps,init_checkpoint=str(ROOT24/'joint'/'final.pt'),init_sha256=digest(ROOT24/'joint'/'final.pt'),lr=lr,seed=252610,center_reference='strict image-supported subset',robot_connected=False),ROOT/'fine_tuned.pt')
    jsave(ROOT/'training_summary.json',dict(updates=steps,elapsed_seconds=time.perf_counter()-start,losses=logs,checkpoint_sha256=digest(ROOT/'fine_tuned.pt'),source='v24 joint checkpoint',robot_connected=False));print('V25_TRAINED',flush=True)

def direct_curves(pred,seed_th=.65,follow_th=.32,ink_th=.45,max_steps=180):
    """Integrate predicted center confidence and double-angle tangent field into polylines."""
    center,ink=pred[0],pred[1];h,w=center.shape; yy,xx=np.mgrid[:h,:w].astype(np.float32)
    theta=.5*np.arctan2(pred[3],pred[2]); tx=np.cos(theta).astype(np.float32);ty=np.sin(theta).astype(np.float32)
    # Candidate seeds are center ridges; enforce spacing to avoid one seed per pixel.
    normalx=-ty;normaly=tx; blur=cv2.GaussianBlur(center,(0,0),.7)
    a=cv2.remap(blur,xx+normalx,yy+normaly,cv2.INTER_LINEAR);b=cv2.remap(blur,xx-normalx,yy-normaly,cv2.INTER_LINEAR)
    ys,xs=np.where((blur>=a)&(blur>=b)&(center>=seed_th)&(ink>=ink_th))
    order=np.argsort(-center[ys,xs]); seeds=[]
    for j in order:
        p=np.array([xs[j],ys[j]],np.float32)
        if all(np.linalg.norm(p-s)>7 for s in seeds):seeds.append(p)
        if len(seeds)>=40:break
    def val(im,p):
        x=float(np.clip(p[0],0,w-1.001));y=float(np.clip(p[1],0,h-1.001));x0=int(x);y0=int(y);fx=x-x0;fy=y-y0
        return float((1-fy)*((1-fx)*im[y0,x0]+fx*im[y0,x0+1])+fy*((1-fx)*im[y0+1,x0]+fx*im[y0+1,x0+1]))
    def trace(seed,sign):
        pts=[seed.copy()];prev=None;stale=0
        for _ in range(max_steps):
            p=pts[-1];ang=.5*math.atan2(val(pred[3],p),val(pred[2],p));d=np.array([math.cos(ang),math.sin(ang)],np.float32)
            if prev is not None and d@prev<0:d=-d
            elif prev is None:d*=sign
            # Correct each step to the strongest nearby normal ridge, with parabolic subpixel offset.
            n=np.array([-d[1],d[0]],np.float32); offsets=np.array([-2.,-1.,0.,1.,2.],np.float32); cand=p+1.35*d+offsets[:,None]*n
            vals=np.array([val(center,q) for q in cand]); k=int(vals.argmax()); shift=float(offsets[k])
            if 0<k<len(vals)-1:
                den=vals[k-1]-2*vals[k]+vals[k+1]
                if abs(den)>1e-6:shift+=float(np.clip(.5*(vals[k-1]-vals[k+1])/den,-.5,.5))
            q=p+1.35*d+(shift*n)
            c=val(center,q); inkv=val(ink,q)
            if c<follow_th or inkv<ink_th:
                stale+=1
                if stale>=2:break
            else:stale=0
            if q[0]<1 or q[0]>=w-1 or q[1]<1 or q[1]>=h-1:break
            if len(pts)>12 and any(np.linalg.norm(q-r)<1.5 for r in pts[-12:-4]):break
            pts.append(q);prev=d
        return np.asarray(pts,np.float32)
    paths=[]
    for s in seeds:
        f=trace(s,1);bpath=trace(s,-1)[::-1];path=np.concatenate([bpath[:-1],f],axis=0)
        if len(path)<15:continue
        # Collapse repeated traces from seeds already explained by a prior curve.
        if any(np.mean(np.min(np.linalg.norm(path[:,None,:]-old[None,:,:],axis=2),axis=1)<2.0)>.72 for old in paths):continue
        paths.append(path)
    return paths

def metric(refs,paths,valid): return v21.path_metrics(refs,paths,valid)

def score(rows):
    return float(np.mean([.5*r['point_coverage_3px']+.5*r['mean_best_single_path_coverage']-.025*r['mixed_reference_paths_flagged'] for r in rows]))

def evaluate():
    force_cpu()
    torch.set_num_threads(4);cv2.setNumThreads(2);cal=ev.cases('calibration');dev=ev.cases('development')
    (ROOT/'evaluation').mkdir(exist_ok=True); (ROOT/'calibration_predictions').mkdir(exist_ok=True)
    model=v24.JointNet();model.load_state_dict(torch.load(ROOT24/'joint'/'final.pt',map_location='cpu',weights_only=False)['model']);model.eval();ck=torch.load(ROOT/'fine_tuned.pt',map_location='cpu',weights_only=False);ft=v24.JointNet();ft.load_state_dict(ck['model']);ft.eval()
    # Freeze curve-follow thresholds on calibration data only.
    configs=[(.55,.22),(.55,.38),(.65,.30),(.65,.46),(.75,.30),(.75,.46)]; results=[]
    calpred={}
    for name,m in [('v24',model),('v25',ft)]:
        for r,rgb,refs,valid,bg in cal:
            cache=ROOT/'calibration_predictions'/(name+'_'+r['id']+'.npz')
            p=np.load(cache)['prediction'] if cache.exists() else v24.predict(m,rgb)
            if not cache.exists():np.savez_compressed(cache,prediction=p)
            calpred[name,r['id']]=(p,refs,valid)
    for s,f in configs:
        rows=[]
        for r,rgb,refs,valid,bg in cal:
            p=calpred['v25',r['id']][0];paths=direct_curves(p,s,f);rows.append(metric(refs,paths,valid))
        results.append(dict(seed_threshold=s,follow_threshold=f,score=score(rows),single=float(np.mean([x['mean_best_single_path_coverage'] for x in rows])),fragments=float(np.mean([x['mean_reference_fragments'] for x in rows])),paths=float(np.mean([x['path_count'] for x in rows]))))
    best=max(results,key=lambda x:x['score']);s,f=best['seed_threshold'],best['follow_threshold'];jsave(ROOT/'curve_calibration.json',dict(selected=best,candidates=results,calibration_only=True,reference_labels_used_at_inference=False))
    rows=[]; imgs={}
    # Standard skeleton/graph decoder vs continuous-field integration, and v24 vs v25.
    for r,rgb,refs,valid,bg in dev:
        key=r['id']; views={}; metrics={}
        for name,m in [('v24',model),('v25',ft)]:
            cache=ROOT/'evaluation'/(name+'_'+key+'.npz')
            pred=np.load(cache)['prediction'] if cache.exists() else v24.predict(m,rgb)
            if not cache.exists():np.savez_compressed(cache,prediction=pred)
            pths,_,_=ev.decode(pred,dict(node=None,affinity=None)); direct=direct_curves(pred,s,f)
            metrics[name+'_graph']=metric(refs,pths,valid);metrics[name+'_curve']=metric(refs,direct,valid);views[name+'_graph']=pths;views[name+'_curve']=direct
        rows.append(dict(id=key,split=r['split'],metrics=metrics));imgs[key]=(rgb,views)
        print('V25_EVAL',key,{k:round(v['mean_best_single_path_coverage'],3) for k,v in metrics.items()},flush=True)
    groups={}
    for g in sorted(set(r['split'] for r in rows)):
        group=[r for r in rows if r['split']==g]
        groups[g]={variant:{metric:float(np.mean([r['metrics'][variant][metric] for r in group])) for metric in ['point_coverage_3px','mean_best_single_path_coverage','mean_reference_fragments','path_count','mixed_reference_paths_flagged']} for variant in ['v24_graph','v24_curve','v25_graph','v25_curve']}
    jsave(ROOT/'evaluation.json',dict(records=rows,groups=groups,curve_thresholds=best,development_not_blind=True,partial_references=True,physical_width_calibrated=False,robot_connected=False))
    keys=['H01','H02','stage_065','sina_6586334590_31','X04','R04'];board=Image.new('RGB',(1600,320*len(keys)+42),'#f5f3ec');d=ImageDraw.Draw(board);d.text((5,5),'v25 中心参考精修与连续曲线积分对照（离线像素轨迹，不是实机落笔）',font=FONT,fill='black')
    for i,key in enumerate(keys):
        rgb,views=imgs[key]
        for j,(title,a) in enumerate([('原图',rgb),('v24骨架追踪',v21.line_view(rgb.shape[:2],views['v24_graph'],True)),('v24连续曲线',v21.line_view(rgb.shape[:2],views['v24_curve'],True)),('v25中心精修+曲线',v21.line_view(rgb.shape[:2],views['v25_curve'],True))]):
            d.text((j*400+5,42+i*320),key+' '+title,font=SMALL,fill='black');v21.paste(board,a,j*400+5,68+i*320,390,245)
    board.save(ROOT/'RESULT_COMPARISON.jpg',quality=95)
    print('V25_GROUPS',json.dumps(groups),flush=True)

def summarize():
    data=read(ROOT/'evaluation.json');rows=data['records'];groups={}
    metrics=['point_coverage_3px','mean_best_single_path_coverage','mean_reference_fragments','path_count','mixed_reference_paths_flagged']
    for g in sorted(set(r['split'] for r in rows)):
        rr=[r for r in rows if r['split']==g]
        groups[g]={v:{m:float(np.mean([r['metrics'][v][m] for r in rr])) for m in metrics} for v in ['v24_graph','v24_curve','v25_graph','v25_curve']}
    data['groups']=groups;data['metrics_are_partial_reference_path_metrics_not_full_image_accuracy']=True;jsave(ROOT/'evaluation.json',data)
    new=groups.get('new_work_development',{});cfg=read(ROOT/'curve_calibration.json')['selected']
    lines=['# v25 中心线精修与连续曲线实验','',
      '本轮在 v24 联合候选上做 200 步 CPU 微调，只对有图像墨迹支撑的窄中心带施加监督。中心参考筛选阈值：墨迹峰值≥0.025、单墨带宽0.75–15像素、参考偏移≤3像素；其余区域保持未知。参考由非专家辅助描线生成，不是专家金标准。','',
      f"连续曲线阈值只用校准集选择：seed={cfg['seed_threshold']:.2f}, follow={cfg['follow_threshold']:.2f}。曲线通过对预测中心概率和方向场双向积分得到，尚未验证为真实人类落笔顺序。",'',
      '| 新作品留出组（H01/H02） | 点覆盖@3px | 单笔覆盖 | 参考碎片均值 | 输出轨迹数均值 | 混合参考笔画代理 |','|---|---:|---:|---:|---:|---:|']
    for v,title in [('v24_graph','v24 骨架解码'),('v24_curve','v24 连续积分'),('v25_graph','v25 精修+骨架'),('v25_curve','v25 精修+连续积分')]:
        if v in new:lines.append(f"| {title} | {new[v]['point_coverage_3px']:.3f} | {new[v]['mean_best_single_path_coverage']:.3f} | {new[v]['mean_reference_fragments']:.2f} | {new[v]['path_count']:.1f} | {new[v]['mixed_reference_paths_flagged']:.2f} |")
    lines+=['','## 固定留出组细分','', '| 组 | v24骨架 单笔覆盖 | v24连续 单笔覆盖 | v25骨架 单笔覆盖 | v25连续 单笔覆盖 |','|---|---:|---:|---:|---:|']
    for g in ['development_holdout','fixed_external_development','user_reviewed_old_development','new_work_development']:
        if g not in groups:continue
        x=groups[g];lines.append(f"| {g} | {x['v24_graph']['mean_best_single_path_coverage']:.3f} | {x['v24_curve']['mean_best_single_path_coverage']:.3f} | {x['v25_graph']['mean_best_single_path_coverage']:.3f} | {x['v25_curve']['mean_best_single_path_coverage']:.3f} |")
    lines+=['','## 结论','',
      '解释限制（2026-10-27补充）：本轮为控制CPU耗时，连续曲线解码仅取全图最高置信度的40个起点，每个方向最多积分180步。这些上限会限制复杂图覆盖，并使长线在达到步数上限时停止；因此不能将全部漏线归因于模型。本轮方案从预测的中心概率和方向场积分出曲线，没有训练直接回归整笔曲线控制点的网络。后续应先固定权重、只调整解码预算与空间覆盖，隔离此项影响。','',
      '在两张新作品留出图上，v25连续积分相对v24连续积分的单笔覆盖仅从0.496升至0.501；点覆盖从0.533升至0.597，参考碎片均值从1.72降到1.37，混合参考笔画代理从6降到3。提升幅度有限，而且只有两张新作。连续积分的单笔覆盖仍低于v24骨架解码的0.605，不能把少轨迹、少碎片直接解释为更完整，因为漏掉的线也会让碎片计数变少。v25中心微调配合骨架解码的新作单笔覆盖为0.582，低于v24的0.605；Sina31和X04上连续积分仍明显失效。因此本轮候选暂不晋升为默认模型。','',
      '图中颜色只区分输出曲线，不代表真实笔顺、笔宽或实际落笔。所有结果均为离线像素轨迹；未连接机械臂，未采集纸面效果。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print('V25_SUMMARIZED',json.dumps(groups),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','train','evaluate','summarize']);p.add_argument('--steps',type=int,default=600);a=p.parse_args()
    {'prepare':prepare,'train':lambda:train(a.steps),'evaluate':evaluate,'summarize':summarize}[a.mode]()
