# -*- coding: utf-8 -*-
"""Image-only stroke proposals and matched calibration/development comparison."""
from v32_stroke_model import *
from scipy.ndimage import distance_transform_edt
import sys, base64, io, hashlib

def sample(a,p):
    p=np.asarray(p,np.float32).reshape(-1,2)
    if len(p)>30000:return np.concatenate([sample(a,p[i:i+30000]) for i in range(0,len(p),30000)])
    return cv2.remap(a,p[:,0][None],p[:,1][None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)[0]

def ridge_field(a):
    h,w=a.shape;best=np.zeros_like(a);direction=np.zeros((h,w,2),np.float32);yy,xx=np.mgrid[:h,:w]
    for sigma in [1.,2.3,4.5]:
        z=cv2.GaussianBlur(a,(0,0),sigma)
        aa=cv2.Sobel(z,cv2.CV_32F,2,0,ksize=3)/4;bb=cv2.Sobel(z,cv2.CV_32F,1,1,ksize=3)/4;cc=cv2.Sobel(z,cv2.CV_32F,0,2,ksize=3)/4
        lam=(aa+cc-np.sqrt((aa-cc)**2+4*bb**2))/2;theta=.5*np.arctan2(2*bb,aa-cc)
        u=np.stack([np.cos(theta),np.sin(theta)],-1);v=np.stack([-u[:,:,1],u[:,:,0]],-1)
        q=np.stack([xx,yy],-1).astype(np.float32)
        left=sample(z,(q+v).reshape(-1,2)).reshape(h,w);right=sample(z,(q-v).reshape(-1,2)).reshape(h,w)
        score=np.maximum(-lam,0)*sigma;score*=((z>=left)&(z>=right)&(a>.018))
        take=score>best;best[take]=score[take];direction[take]=u[take]
    # Fill direction away from exact nonmaximum centers from a smooth structure tensor.
    z=cv2.GaussianBlur(a,(0,0),1.3);gx=cv2.Sobel(z,cv2.CV_32F,1,0)/8;gy=cv2.Sobel(z,cv2.CV_32F,0,1)/8
    aa=cv2.GaussianBlur(gx*gx,(0,0),2);bb=cv2.GaussianBlur(gx*gy,(0,0),2);cc=cv2.GaussianBlur(gy*gy,(0,0),2);theta=.5*np.arctan2(2*bb,aa-cc)+math.pi/2
    field=np.stack([np.cos(theta),np.sin(theta)],-1).astype(np.float32);field[best>.001]=direction[best>.001]
    return best,direction,field

def seeds_from_image(a):
    score,u,field=ridge_field(a);flat=np.argsort(score.ravel())[::-1];covered=np.zeros(a.shape,np.uint8);out=[]
    for k in flat:
        y,x=np.unravel_index(k,a.shape)
        if score[y,x]<.002:break
        if x<6 or y<6 or x>=a.shape[1]-6 or y>=a.shape[0]-6 or covered[y,x]:continue
        out.append((np.array([x,y],np.float32),u[y,x]));cv2.circle(covered,(int(x),int(y)),6,1,-1)
        if len(out)>=280:break
    return out,field

@torch.no_grad()
def neural_walk(rgb,seeds,model):
    alpha=observed(rgb);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255;image=np.stack([alpha,gray],-1);support=cv2.dilate(alpha,np.ones((3,3),np.uint8));states=[]
    for i,(p,u) in enumerate(seeds):
        for sign in [1,-1]:states.append(dict(p=p.copy(),u=u*sign,path=[p.copy()],alive=True,bad=0,stop=[],seed=i,sign=sign))
    for chunk in range(12):
        active=[s for s in states if s['alive']]
        if not active:break
        for start in range(0,len(active),96):
            group=active[start:start+96];patches=np.stack([canonical(image,s['p'],s['u']).transpose(2,0,1) for s in group]);xx=torch.from_numpy(patches).cuda();points,logit,_=model(features(xx));points=points.cpu().numpy();probs=logit.sigmoid().cpu().numpy()
            for s,q,prob in zip(group,points,probs):
                v=np.array([-s['u'][1],s['u'][0]]);q=s['p']+(q-ANCHOR)[:,0:1]*s['u']+(q-ANCHOR)[:,1:2]*v
                for j in range(1,7):
                    p=q[j];d=p-s['path'][-1];dist=np.linalg.norm(d)
                    if prob[j-1]<.5 or not (2<=p[0]<rgb.shape[1]-2 and 2<=p[1]<rgb.shape[0]-2) or not (.5<dist<8):s['alive']=False;break
                    if d@s['u']/max(dist,1e-6)<-.35:s['alive']=False;break
                    if len(s['path'])>12 and np.linalg.norm(np.asarray(s['path'][:-10])-p,axis=1).min()<3:s['alive']=False;break
                    evidence=float(sample(support,p)[0]);s['bad']=s['bad']+1 if evidence<.018 else 0
                    if s['bad']>3:s['alive']=False;break
                    s['path'].append(p.copy());s['stop'].append(float(prob[j-1]))
                s['p']=s['path'][-1]
                if len(s['path'])>3:
                    direction=s['path'][-1]-s['path'][-4];s['u']=direction/max(1e-6,np.linalg.norm(direction))
    proposals=[]
    for i in range(len(seeds)):
        f,b=states[2*i:2*i+2];q=np.asarray(b['path'][::-1]+f['path'][1:],np.float32)
        if len(q)>4:proposals.append(dict(points=q.tolist(),seed=i,continuation_confidence=float(np.mean(f['stop']+b['stop'])) if f['stop']+b['stop'] else 0))
    return proposals

def classical_walk(rgb,seeds,field):
    a=observed(rgb);support=cv2.dilate(a,np.ones((3,3),np.uint8));out=[]
    for sid,(seed,axis) in enumerate(seeds):
        halves=[]
        for sign in [1,-1]:
            p=seed.copy();u=axis*sign;points=[p.copy()];bad=0
            for _ in range(72):
                v=np.array([-u[1],u[0]]);q=p+STEP*u;offset=np.linspace(-2,2,9);qq=q+offset[:,None]*v;strength=sample(a,qq)-.012*(offset**2);q=qq[int(np.argmax(strength))]
                if not (2<=q[0]<a.shape[1]-2 and 2<=q[1]<a.shape[0]-2):break
                ev=float(sample(support,q)[0]);bad=bad+1 if ev<.018 else 0
                if bad>3:break
                if len(points)>12 and np.linalg.norm(np.asarray(points[:-10])-q,axis=1).min()<3:break
                points.append(q.copy());t=np.array([sample(field[:,:,k],q)[0] for k in range(2)]);t*=1 if t@u>=0 else -1;u=.55*u+.45*t;u/=max(1e-6,np.linalg.norm(u));p=q
            halves.append(points)
        q=np.asarray(halves[1][::-1]+halves[0][1:],np.float32)
        if len(q)>4:out.append(dict(points=q.tolist(),seed=sid,continuation_confidence=1.))
    return out

def paths_mask(paths,shape):
    m=np.zeros(shape,np.uint8)
    for p in paths:
        if len(p)>1:cv2.polylines(m,[np.rint(p).astype(np.int32)],False,1,1)
    return m

def accept(rgb,base,proposals,cfg):
    alpha=observed(rgb);support=cv2.dilate(alpha,np.ones((3,3),np.uint8));seen=paths_mask(base,alpha.shape);accepted=[];audit=[]
    pp=sorted(proposals,key=lambda r:len(r['points']),reverse=True)
    for prop in pp:
        q=arc(prop['points']);length=float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum())
        if length<15:continue
        evidence=sample(support,q);supported=float((evidence>=cfg['ink']).mean())
        if supported<.85:continue
        dist=distance_transform_edt(seen==0);novel=float((sample(dist.astype(np.float32),q)>4).mean())
        if novel<cfg['novel']:continue
        accepted.append(q);cv2.polylines(seen,[np.rint(q).astype(np.int32)],False,1,1);audit.append(dict(seed=prop['seed'],length=length,supported_fraction=supported,novel_fraction=novel))
    return accepted,audit

def infer(partition):
    manifest=read(V31/'manifest.json');models={key:load(key) for key in ['procedural','adapted']};cv2.setNumThreads(2);torch.set_num_threads(4)
    for r in manifest[partition]:
        dest=ROOT/'proposals'/r['id'];dest.mkdir(parents=True,exist_ok=True)
        if (dest/'adapted.json').exists() and (dest/'classical.json').exists():continue
        # Inference consumes RGB only; no reference, valid-mask or background files opened.
        rgb=np.asarray(Image.open(r['raw']).convert('RGB'));a=observed(rgb);seeds,field=seeds_from_image(a);start=time.perf_counter();jsave(dest/'seeds.json',dict(points=[p.tolist() for p,u in seeds],axes=[u.tolist() for p,u in seeds],source_image_sha256=digest(r['raw']),reference_used=False))
        for key,model in models.items():
            proposals=neural_walk(rgb,seeds,model);jsave(dest/(key+'.json'),dict(proposals=proposals,seed_source='RGB_multiscale_ridges',reference_labels_used=False,physical_z=None,ready_for_robot=False,actually_drawn=False));print('V32_PROPOSE',r['id'],key,len(seeds),len(proposals),round(time.perf_counter()-start,1),flush=True)
        proposals=classical_walk(rgb,seeds,field);jsave(dest/'classical.json',dict(proposals=proposals,reference_labels_used=False));print('V32_CLASSICAL',r['id'],len(proposals),flush=True)

def metric(rgb,r,paths):
    import v26_tone_coverage as ev
    refs=read(r['reference_file'])['paths'];z=np.load(r['mask_file']);return ev.evaluate_paths(refs,[arc(p) for p in paths],z['valid'],z['bg'],rgb)

KEYS=['point_coverage_3px','mean_best_single_path_coverage','path_count','mixed_reference_paths_flagged','duplicate_point_fraction','paper_path_density','low_contrast_ref_coverage_3px']
def summary(rows,methods):
    result={}
    for name in methods:
        result[name]={}
        for k in KEYS:
            vals=[r['metrics'][name][k] for r in rows if r['metrics'][name].get(k) is not None];result[name][k]=float(np.mean(vals)) if vals else None
    return result

def gate(b,p):
    return bool(p['mean_best_single_path_coverage']>=b['mean_best_single_path_coverage']+.005 and p['point_coverage_3px']>=b['point_coverage_3px']-.005 and p['mixed_reference_paths_flagged']<=b['mixed_reference_paths_flagged'] and p['duplicate_point_fraction']<=b['duplicate_point_fraction']+.01)

def calibrate():
    if (ROOT/'calibration.json').exists():raise FileExistsError('Calibration choice frozen')
    options=[];rows=read(V31/'manifest.json')['calibration']
    for cfg in read(ROOT/'protocol.json')['addition_options']:
        records=[]
        for r in rows:
            rgb=np.asarray(Image.open(r['raw']).convert('RGB'));base=read(V31/'initial_paths'/(r['id']+'.json'))['paths'];pp=read(ROOT/'proposals'/r['id']/'adapted.json')['proposals'];added,audit=accept(rgb,base,pp,cfg);records.append(dict(id=r['id'],added=len(added),metrics=dict(initial=metric(rgb,r,base),adapted=metric(rgb,r,base+added))))
        s=summary(records,['initial','adapted']);b=s['initial'];p=s['adapted'];score=p['mean_best_single_path_coverage']+.5*p['point_coverage_3px']-.15*p['duplicate_point_fraction']-.025*p['mixed_reference_paths_flagged']-10*(p['paper_path_density'] or 0);options.append(dict(config=cfg,score=score,passed=gate(b,p),summary=s,records=records));print('V32_CALIBRATION',cfg,'gate',options[-1]['passed'],'summary',s,flush=True)
    selected=max(options,key=lambda a:a['score']);jsave(ROOT/'calibration.json',dict(selected=selected,options=options,development_used=False,default_replace_allowed=False))

def line_image(rgb,paths,base=None):
    out=np.full_like(rgb,255) if base is None else base.copy()
    for i,p in enumerate(paths):
        hsv=np.uint8([[[int(i*41%180),170,175]]]);color=tuple(int(v) for v in cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)[0,0]);cv2.polylines(out,[np.rint(p).astype(np.int32)],False,color,1,cv2.LINE_AA)
    return out

def paste(board,a,x,y,w,h):
    im=Image.fromarray(a);im.thumbnail((w,h),Image.Resampling.LANCZOS);board.paste(im,(x+(w-im.width)//2,y+(h-im.height)//2))

def evaluate(partition):
    cfg=read(ROOT/'calibration.json')['selected']['config'];records=[];views={};methods=['initial','classical','procedural','adapted']
    for r in read(V31/'manifest.json')[partition]:
        rgb=np.asarray(Image.open(r['raw']).convert('RGB'));base=read(V31/'initial_paths'/(r['id']+'.json'))['paths'];rec=dict(id=r['id'],group=r['group'],metrics=dict(initial=metric(rgb,r,base)),additions={});images={'original':rgb,'initial':line_image(rgb,base)}
        for name in methods[1:]:
            pp=read(ROOT/'proposals'/r['id']/(name+'.json'))['proposals'];added,audit=accept(rgb,base,pp,cfg);paths=base+[p.tolist() for p in added];rec['metrics'][name]=metric(rgb,r,paths);rec['additions'][name]=len(added);dest=ROOT/'trajectories'/name/r['id'];dest.mkdir(parents=True,exist_ok=True);jsave(dest/'trajectory.json',dict(paths=paths,added_paths=[p.tolist() for p in added],reference_labels_used_at_inference=False,coordinates='original_image_pixels',physical_z=None,ready_for_robot=False,actually_drawn=False,audit=audit));images[name]=line_image(rgb,paths)
            if name=='adapted':
                images['new']=line_image(rgb,added);ov=rgb.copy();cv2.polylines(ov,[np.rint(p).astype(np.int32) for p in added],False,(230,35,30),1,cv2.LINE_AA) if added else None;images['overlay']=ov
                for label,a in [('new',images['new']),('combined',images[name]),('overlay',ov)]:Image.fromarray(a).save(dest/(label+'.png'))
        records.append(rec);views[r['id']]=images
        board=Image.new('RGB',(1500,600),'#f3f0e8');d=ImageDraw.Draw(board);d.text((7,5),r['id']+' v32 RGB起点 → 有序曲线：离线候选，非实机落笔',font=FONT,fill='black')
        for i,(title,key) in enumerate([('原图','original'),('旧候选','initial'),('几何追踪补线','classical'),('程序监督网络补线','procedural'),('真实适配网络补线','adapted'),('新增线红色叠加','overlay')]):
            x=i%3*500;y=35+i//3*280;d.text((x+5,y),title,font=SMALL,fill='black');paste(board,images[key],x+5,y+25,490,245)
        (ROOT/'comparisons').mkdir(exist_ok=True);board.save(ROOT/'comparisons'/(r['id']+'.jpg'),quality=95);print('V32_EVAL',r['id'],rec['additions'],{k:round(v['mean_best_single_path_coverage'],4) for k,v in rec['metrics'].items()},flush=True)
    s=summary(records,methods);jsave(ROOT/(partition+'_evaluation.json'),dict(records=records,summary=s,config=cfg,partial_reference_not_whole_image_accuracy=True,reused_development_not_blind=True,all_paths_resampled_to_one_px=True))
    if partition=='development':
        board=Image.new('RGB',(1600,1195),'#f3f0e8');d=ImageDraw.Draw(board);d.text((7,5),'v32 实际输出：旧线灰底＋新增红线；没有连接机械臂',font=FONT,fill='black')
        for i,key in enumerate(['R04','H02','X03','N11']):
            for j,(title,name) in enumerate([('原图','original'),('旧候选中心线','initial'),('网络新增线','new'),('旧候选＋网络新增线','adapted')]):
                x=j*400;y=32+i*290;d.text((x+5,y),key+' '+title,font=SMALL,fill='black');paste(board,views[key][name],x+4,y+25,392,254)
        board.save(ROOT/'RESULT_COMPARISON.jpg',quality=97)
    print('V32_SUMMARY',partition,json.dumps(s),flush=True)

def diagnostics():
    sv=load_data('procedural_validation');rv=load_data('real_validation');out={}
    for key in ['procedural','adapted']:
        m=load(key);out[key]=dict(procedural=validation(m,sv),real_seed_conditioned=validation(m,rv));del m
    for key,data in [('procedural',sv),('real',rv)]:
        p=torch.tensor(ANCHOR,device='cuda')[None,None]+torch.stack([torch.arange(N+1,device='cuda')*STEP,torch.zeros(N+1,device='cuda')],-1)[None];d=torch.linalg.vector_norm(p-data[1],dim=-1);e=d[data[2]>.5].cpu().numpy();out['straight_'+key]=dict(mean_ordered_point_error_px=float(e.mean()),fraction_within_3px=float((e<=3).mean()))
    jsave(ROOT/'seed_conditioned_diagnostic.json',dict(results=out,uses_known_reference_seeds_for_diagnosis_only=True,not_full_image_extraction_accuracy=True,real_endpoints_not_validated=True))

def reference_free_check(source,dest):
    import os
    guard=dict(blocked_annotation_opens=0)
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=os.fsdecode(args[0]).replace('\\','/').lower();name=p.rsplit('/',1)[-1]
            if '/labels/' in p or name in ['reference.json','partial_reference.json','reviewed.json','reference.npz','evaluation_masks.npz','real_train.npz','real_validation.npz']:
                guard['blocked_annotation_opens']+=1;raise RuntimeError('Reference reads forbidden during image-only inference')
    sys.addaudithook(audit)
    dest=Path(dest)
    if not dest.is_absolute() or dest.drive.upper() not in ['E:','F:']:raise ValueError('E/F output required')
    rgb=np.asarray(Image.open(source).convert('RGB'));seeds,_=seeds_from_image(observed(rgb));pp=neural_walk(rgb,seeds,load());dest.mkdir(parents=True,exist_ok=True)
    jsave(dest/'proposals.json',dict(proposals=pp,physical_z=None,ready_for_robot=False,actually_drawn=False))
    jsave(dest/'interface_check.json',dict(source=str(source),source_sha256=digest(source),inference_annotation_read_guard_active=True,blocked_annotation_opens=guard['blocked_annotation_opens'],reference_labels_used=False,actual_device='cuda:0',seed_count=len(seeds),proposal_count=len(pp)))
    print('V32_REFERENCE_FREE_CHECK',len(pp),guard,flush=True)

def report():
    import v32_report
    v32_report.finish()

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='infer_calibration':infer('calibration')
    elif mode=='infer_development':infer('development')
    elif mode=='calibrate':calibrate()
    elif mode=='eval_calibration':evaluate('calibration')
    elif mode=='eval_development':evaluate('development')
    elif mode=='diagnostics':diagnostics()
    elif mode=='reference_free_check':reference_free_check(sys.argv[2],sys.argv[3])
    elif mode=='report':report()
    else:raise ValueError(mode)
