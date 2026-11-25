# -*- coding: utf-8 -*-
"""Offline, pinned baseline extraction with bounded simplification and audit."""
import os,sys
from pathlib import Path
sys.dont_write_bytecode=True
_LAB=Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022')
for _key,_sub in {'TEMP':'temp','TMP':'temp','PIP_CACHE_DIR':'cache/pip','TORCH_HOME':'cache/torch','HF_HOME':'cache/huggingface','XDG_CACHE_HOME':'cache','CUDA_CACHE_PATH':'cache/cuda'}.items():os.environ[_key]=str(_LAB/_sub)
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from v33_replace_core import *
import argparse,datetime,html

STABLE=LAB/'stable_line_extractor_20261028'

def length(p):return float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())

def simplify(p,epsilon=.45):
    p=np.asarray(p,np.float32)
    if len(p)<3:return p
    q=cv2.approxPolyDP(p[:,None],epsilon,False).reshape(-1,2)
    if len(q)<2:return p
    q[0]=p[0];q[-1]=p[-1]
    # Verify the actual two-sided representation error, not only its vertex count.
    a=arc(p,.25);b=arc(q,.25);error=max(float(cKDTree(a).query(b)[0].max()),float(cKDTree(b).query(a)[0].max()))
    return q if error<=.60 else p

def duplicate_only(paths):
    # Remove only near-identical retracing; close parallel strokes stay separate.
    out=[];ids=[];audit=[]
    for i,p in sorted(enumerate(paths),key=lambda z:length(z[1]),reverse=True):
        p=np.asarray(p,np.float32);a=arc(p);t=tangent(a);redundant=False
        for j,q in enumerate(out):
            if np.any(a.max(0)<q.min(0)-.6) or np.any(a.min(0)>q.max(0)+.6):continue
            b=arc(q);d,k=cKDTree(b).query(a)
            if np.mean(d<=.55)<.985 or np.abs((t*tangent(b)[k]).sum(-1)).mean()<.98:continue
            order=k if k[-1]>=k[0] else -k
            if np.maximum(-np.diff(order),0).sum()/max(1,np.ptp(k))>.05:continue
            redundant=True;audit.append(dict(event='near_identical_retrace',input_path=i,retained_input_path=ids[j]));break
        if not redundant:out.append(p);ids.append(i)
    order=np.argsort(ids);return [out[i] for i in order],[ids[i] for i in order],audit

def ink_loop_candidate(p,rgb):
    p=arc(p);h,w=rgb.shape[:2]
    if len(p)<8 or np.linalg.norm(p[0]-p[-1])>1.5:return None
    lo=p.min(0);hi=p.max(0)
    if max(hi-lo)>14 or min(hi-lo)<2:return None
    mask=np.zeros((h,w),np.uint8);cv2.fillPoly(mask,[np.rint(p).astype(np.int32)],1)
    inside=cv2.erode(mask,np.ones((3,3),np.uint8))>0
    if inside.sum()<3:return None
    ring=(cv2.dilate(mask,np.ones((7,7),np.uint8))>0)&(mask==0)
    if not ring.any():return None
    a=observed(rgb);g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    fraction=float((a[inside]>.08).mean());contrast=float(np.median(g[ring])-np.median(g[inside]))
    if fraction<.85 or contrast<.03:return None
    yy,xx=np.where(inside);weight=a[inside]+.001;return dict(center=[float(np.average(xx,weights=weight)),float(np.average(yy,weights=weight))],apparent_radius_px=float(np.sqrt(mask.sum()/np.pi)),filled_ink_fraction=fraction,raw_gray_contrast=contrast,status='ink_mark_hypothesis_not_verified_pen_action')

def guarded_centers(rgb,paths):
    from scipy.ndimage import gaussian_filter1d
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255;gray=cv2.GaussianBlur(gray,(0,0),.5);alpha=observed(rgb);off=np.arange(-12,12.25,.5,dtype=np.float32);pp=[arc(p) for p in paths];trees=[cKDTree(p) for p in pp];out=[];audit=[];h,w=gray.shape
    for sid,p in enumerate(pp):
        if len(p)<15:out.append(np.asarray(paths[sid],np.float32));continue
        t=tangent(p);n=np.c_[-t[:,1],t[:,0]];q=p[:,None]+n[:,None]*off[None,:,None];prof=cv2.remap(alpha,q[:,:,0],q[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE);prof=gaussian_filter1d(prof,.7,axis=1);raw_prof=cv2.remap(gray,q[:,:,0],q[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE);shift=np.zeros(len(p),np.float32);known=np.zeros(len(p),np.float32)
        otherdist=np.full(len(p),999.)
        for j,tree in enumerate(trees):
            if j!=sid:otherdist=np.minimum(otherdist,tree.query(p)[0])
        for k,v in enumerate(prof):
            if k<4 or k>=len(p)-4 or otherdist[k]<6 or min(p[k][0],p[k][1],w-1-p[k][0],h-1-p[k][1])<13:continue
            if np.linalg.norm(t[max(0,k-3)]-t[min(len(p)-1,k+3)])>.65:continue
            peak=int(np.argmax(v[18:31]))+18;strength=float(v[peak])
            if strength<.10:continue
            threshold=max(.025,.45*strength);a=peak;b=peak
            while a>0 and v[a-1]>threshold:a-=1
            while b<48 and v[b+1]>threshold:b+=1
            if a<2 or b>46 or b-a<2 or b-a>24:continue
            if min(raw_prof[k,a-2],raw_prof[k,b+2])-raw_prof[k,a:b+1].min()<.004:continue
            # Adjacent independent peaks make the center ambiguous.
            outside=np.r_[v[:max(0,a-2)],v[min(49,b+3):]]
            if len(outside) and outside.max()>.4*strength:continue
            target=.5*(off[a]+off[b])
            if abs(target)>1.5:continue
            shift[k]=target;known[k]=1
        weight=gaussian_filter1d(known,2);smoothed=gaussian_filter1d(shift*known,2)/np.maximum(weight,1e-6);delta=np.where((known>0)&(weight>.7),np.clip(smoothed,-1.5,1.5),0);candidate=p+n*delta[:,None];active=np.abs(delta)>.15
        if active.sum()<5:out.append(np.asarray(paths[sid],np.float32));continue
        gain=float((sample(gray,p[active])-sample(gray,candidate[active])).mean())
        balance_gain=float((np.abs(shift[active])-np.abs(shift[active]-delta[active])).mean());strong_band=float(np.median(prof[active].max(1)))>=.2
        # A flat black band has no darker center pixel. Its two visible boundaries
        # can still establish a center; do not optimize darkness alone.
        if gain<-.01 or (gain<.004 and not (strong_band and balance_gain>=.5)):out.append(np.asarray(paths[sid],np.float32));continue
        # The correction cannot create a new close return/fold.
        pairs=cKDTree(candidate).query_pairs(1.2);fold=any(abs(a-b)>8 and np.linalg.norm(p[a]-p[b])>3 for a,b in pairs)
        if fold:out.append(np.asarray(paths[sid],np.float32));continue
        out.append(candidate);audit.append(dict(event='isolated_strong_ink_center_adjustment',input_path=sid,active_points=int(active.sum()),max_shift_px=float(np.abs(delta).max()),mean_raw_gray_darkening=gain,mean_band_balance_gain_px=balance_gain,endpoints_unchanged=bool(np.allclose(candidate[[0,-1]],p[[0,-1]]))))
    return out,audit

def process_paths(rgb,paths,mode='compact'):
    if mode=='baseline':return dict(paths=[np.asarray(p).tolist() for p in paths],input_membership=list(range(len(paths))),audit=[],ink_mark_hypotheses=[],new_strokes_added=0)
    if mode=='center_guarded':
        pp,audit=guarded_centers(rgb,paths);return dict(paths=[p.tolist() for p in pp],input_membership=list(range(len(paths))),audit=audit,ink_mark_hypotheses=[],new_strokes_added=0)
    kept,ids,audit=duplicate_only(paths);result=[];members=[];marks=[]
    for p,i in zip(kept,ids):
        mark=ink_loop_candidate(p,rgb) if mode=='ink_marks_trial' else None
        if mark is not None:marks.append(dict(mark,input_path=i));audit.append(dict(event='filled_small_closed_loop_to_ink_mark_hypothesis',input_path=i));continue
        result.append(simplify(p).tolist());members.append(i)
    return dict(paths=result,input_membership=members,audit=audit,ink_mark_hypotheses=marks,new_strokes_added=0)

def black_lines(shape,paths):
    out=np.full((*shape,3),255,np.uint8)
    for p in paths:
        if len(p)>1:cv2.polylines(out,[np.rint(p).astype(np.int32)],False,(20,20,20),1,cv2.LINE_AA)
    return out

def save_outputs(folder,rgb,result,meta):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);paths=result['paths'];Image.fromarray(rgb).save(folder/'input.png');Image.fromarray(black_lines(rgb.shape[:2],paths)).save(folder/'lines.png');overlay=rgb.copy()
    for p in paths:cv2.polylines(overlay,[np.rint(p).astype(np.int32)],False,(20,105,220),1,cv2.LINE_AA)
    Image.fromarray(overlay).save(folder/'overlay.png');doc=dict(result,meta=meta,coordinates='input_working_image_pixels',physical_z=None,physical_speed=None,ready_for_robot=False,actually_drawn=False,reference_labels_used=False);jsave(folder/'paths.json',doc)
    h,w=rgb.shape[:2];polys=['<polyline points="'+' '.join(f'{x:.3f},{y:.3f}' for x,y in p)+'"/>' for p in paths];svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}"><metadata>Offline image coordinates. No calibrated robot controls.</metadata><g fill="none" stroke="black" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">'+''.join(polys)+'</g></svg>';(folder/'lines.svg').write_text(svg,encoding='utf-8')
    page='''<!doctype html><meta charset="utf-8"><title>线条提取</title><style>body{font:16px system-ui;margin:24px;background:#f3f0e8}section{display:flex;flex-wrap:wrap;gap:18px}img{max-width:100%;background:white}div{max-width:600px}p{line-height:1.6}</style><h2>线条提取结果</h2><p>固定基线，无新增补线。像素轨迹未标定为机械臂坐标；SVG和路径文件供离线检查。</p><section><div><p>原图</p><img src="input.png"></div><div><p>黑白中心线</p><img src="lines.png"></div><div><p>原图叠加</p><img src="overlay.png"></div></section><p><a href="lines.svg">SVG</a> · <a href="paths.json">像素轨迹与处理记录</a></p>''';(folder/'preview.html').write_text(page,encoding='utf-8')
    for p in [folder/'lines.png',folder/'overlay.png']:
        with Image.open(p) as im:im.verify()

def predict(source,output=None,max_side=640,mode='auto',open_preview=False):
    import v28_stroke_relations as old
    source=Path(source).resolve()
    if not source.is_file():raise FileNotFoundError(str(source))
    lock=STABLE/'baseline_lock.json'
    if lock.exists():
        for item in read(lock)['files']:
            if digest(item['path'])!=item['sha256']:raise RuntimeError('Pinned baseline file changed: '+item['path'])
    if mode=='auto':mode=read(STABLE/'release_config.json')['mode'] if (STABLE/'release_config.json').exists() else 'baseline'
    if mode not in ['baseline','compact','center_guarded']:raise ValueError('Unvalidated ink-mark trials are not available in normal prediction')
    guard=dict(blocked=0)
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=os.fsdecode(args[0]).replace('\\','/').lower();name=path.rsplit('/',1)[-1]
            if '/labels/' in path or name in ['reference.json','partial_reference.json','reviewed.json','reference.npz','evaluation_masks.npz']:
                guard['blocked']+=1;raise RuntimeError('Reference labels are forbidden in prediction')
    sys.addaudithook(audit);im=Image.open(source).convert('RGB');original=list(im.size);im.thumbnail((max_side,max_side),Image.Resampling.LANCZOS);rgb=np.asarray(im);model=old.exp.load_model('base');pred=old.exp.prev.prior.predict(model,rgb)[:5];g=old.add_visual(old.build_graph(pred,rgb),model);choice=read(old.ROOT/'calibration.json')['selected'];pr=old.probabilities(g,old.models('visual'));accepted,_=old.select(g,pr,choice['threshold'],choice['margin']);paths=[p.tolist() for p in old.assemble(g,accepted)];del model,g,pred;torch.cuda.empty_cache();result=process_paths(rgb,paths,mode);stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');dest=Path(output) if output else STABLE/'outputs'/(source.stem+'_'+digest(source)[:8]+'_'+stamp)
    if not dest.is_absolute() or dest.drive.upper() not in ['E:','F:']:raise ValueError('Output directory must be absolute on E or F')
    meta=dict(input=str(source),input_sha256=digest(source),original_size=original,working_size=list(im.size),original_from_working_scale_xy=[original[0]/im.width,original[1]/im.height],mode=mode,candidate_model='frozen v31-era v28 baseline',cuda_device=torch.cuda.get_device_name(),new_training_steps=0,source_path_count=len(paths),retained_path_count=len(result['paths']),annotation_guard_active=True,blocked_annotation_opens=guard['blocked'],automatic_cunfa_classification=False,stable_baseline_not_claimed_new_accuracy_breakthrough=True);save_outputs(dest,rgb,result,meta);jsave(dest/'baseline_paths.json',dict(paths=paths,physical_z=None,ready_for_robot=False));Image.fromarray(black_lines(rgb.shape[:2],paths)).save(dest/'baseline_lines.png');page=(dest/'preview.html').read_text(encoding='utf-8');page=page.replace('<p>黑白中心线</p>','<p>当前输出（'+mode+'）</p>').replace('</section>','<div><p>未修正基线，供核对</p><img src="baseline_lines.png"></div></section>');(dest/'preview.html').write_text(page,encoding='utf-8');print('EXTRACTION_READY',str(dest/'preview.html'),json.dumps(meta,ensure_ascii=False),flush=True)
    if open_preview:
        import webbrowser
        webbrowser.open((dest/'preview.html').resolve().as_uri())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('image');ap.add_argument('--output');ap.add_argument('--max-side',type=int,default=640,choices=[512,640,960,1280]);ap.add_argument('--mode',choices=['auto','baseline','compact','center_guarded'],default='auto');ap.add_argument('--open-preview',action='store_true');a=ap.parse_args();predict(a.image,a.output,a.max_side,a.mode,a.open_preview)
