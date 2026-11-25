# -*- coding: utf-8 -*-
"""Learn local center judgement from frozen user review; abstain on ambiguity."""
from cross_section_v18 import *
from collections import Counter
ROOT=LAB/'autonomous_review_v19_20261025'
V18=LAB/'cross_sections_v18_20261025'


def profile_input(rgb,p,n):
    g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255;u=np.array([n[1],-n[0]],np.float32);coords=np.asarray(p)[None,None]+OFFSETS[None,:,None]*n+np.arange(-4,5,dtype=np.float32)[:,None,None]*u
    raw=cv2.remap(g,coords[:,:,0].astype(np.float32),coords[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101);av=np.sort(raw,axis=0)[1:-1].mean(0);l=float(np.quantile(av[OFFSETS<-15],.75));r=float(np.quantile(av[OFFSETS>15],.75));paper=np.interp(OFFSETS,[-20,20],[l,r]).clip(.15,1);contrast=np.maximum((paper-av)/paper,0);valid=((coords[:,:,0]>=0)&(coords[:,:,0]<g.shape[1])&(coords[:,:,1]>=0)&(coords[:,:,1]<g.shape[0])).mean(0)
    x=np.stack((np.clip(av/paper,0,1.2),contrast,raw.std(0)/paper,OFFSETS/24,valid)).astype(np.float32)
    prior,band=geometry_prior(x)
    return np.concatenate((x,prior[None],band[None]))


def geometry_prior(x):
    signal=gaussian_filter1d(x[1],.8);noise=max(.001,float(np.median(np.abs(np.diff(signal)))));prior=np.zeros_like(signal)+.01;band=np.zeros_like(signal);chosen=[]
    peaks=np.flatnonzero((signal>=np.r_[signal[0],signal[:-1]])&(signal>=np.r_[signal[1:],signal[-1]]))
    for peak in sorted(peaks,key=lambda i:-signal[i]):
        pv=float(signal[peak])
        if pv<max(.012,4*noise):continue
        on=(signal>max(2*noise,.27*pv)).astype(np.uint8)[None];on=cv2.morphologyEx(on,cv2.MORPH_CLOSE,np.ones((1,3),np.uint8))[0]>0;a=int(peak);b=int(peak)+1
        while a>0 and on[a-1]:a-=1
        while b<len(on) and on[b]:b+=1
        width=(b-a)*.5
        if a==0 or b==len(on) or not 1<=width<=32 or x[4,a:b].min()<.9:continue
        weight=np.maximum(signal[a:b]-pv*.1,0);centroid=float((OFFSETS[a:b]*weight).sum()/max(1e-6,weight.sum()));mid=float((OFFSETS[a]+OFFSETS[b-1])*.5);center=.7*mid+.3*centroid
        if any(abs(center-v)<2 for v in chosen):continue
        chosen.append(center);score=float(np.clip(pv/(noise*8),.5,1));prior=np.maximum(prior,score*np.exp(-(OFFSETS-center)**2/(2*1.3**2)));band[a:b]=1
    return prior.astype(np.float32),band.astype(np.float32)


class Reviewer(nn.Module):
    def __init__(self):
        super().__init__();self.body=nn.Sequential(nn.Conv1d(7,16,7,padding=3),nn.GroupNorm(4,16),nn.SiLU(),nn.Conv1d(16,32,5,padding=2),nn.GroupNorm(4,32),nn.SiLU(),nn.Conv1d(32,32,5,padding=2),nn.SiLU());self.center=nn.Conv1d(32,1,1);self.present=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,1))
    def forward(self,x):
        h=self.body(x);prior=(.02+.90*x[:,5]).clamp(.01,.99);logit=torch.logit(prior);return .7*self.center(h)[:,0].tanh()+logit,self.present(h)[:,0]


def inputs():
    audit=read(ROOT/'review_audit.json');candidate={r['id']:r for r in read(ROOT/'draft_candidates_snapshot.json')};records={r['id']:r for r in legacy.records()};photos={k:np.asarray(Image.open(r['image']).convert('RGB')) for k,r in records.items()};rows=[];rng=np.random.default_rng(1902510)
    for a in audit['rows']:
        if not a['reviewed'] or a['label']=='uncertain':continue
        r=candidate[a['id']];positive=a['label']!='background';n=np.asarray(r['normal'],np.float32);p=np.asarray(r['p'],np.float32);family='reviewed_heye' if r['sample_id'].startswith('R') else 'sina_tutorial';weight=1. if a['geometry_changed'] or not positive else .65
        for i in range(28):
            jitter=0. if i==0 else float(rng.uniform(-12,12));along=0. if i==0 else float(rng.uniform(-2.5,2.5));u=np.array([n[1],-n[0]]);q=p+n*jitter+u*along;offset=a['center']-jitter
            if positive and not (-23.5<=offset<=23.5 if i==0 else -21<=offset<=21):continue
            x=profile_input(photos[r['sample_id']],q,n);target=np.exp(-(OFFSETS-offset)**2/(2*1.15**2)) if positive else np.zeros_like(OFFSETS);rows.append(dict(x=x,target=target.astype(np.float32),present=float(positive),weight=weight,group=family,identity=a['id'],anchor=i==0,center=offset))
    # Real green regions are reviewed background, not unknown areas.
    for identity in ['R01','R02']:
        rgb,doc,bg,valid=legacy.source(records[identity]);ys,xs=np.nonzero(cv2.erode((bg>0).astype(np.uint8),np.ones((9,9),np.uint8)))
        for i in range(min(160,len(xs))):
            k=int(rng.integers(len(xs)));theta=float(rng.uniform(0,math.pi));n=np.array([math.cos(theta),math.sin(theta)],np.float32);x=profile_input(rgb,np.array([xs[k],ys[k]]),n);rows.append(dict(x=x,target=np.zeros(len(OFFSETS),np.float32),present=0.,weight=.6,group='reviewed_heye',identity=identity+'_green',anchor=False,center=0.))
    return rows


def fit(rows,steps,seed):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);m=Reviewer().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001);X=np.stack([r['x'] for r in rows]);T=np.stack([r['target'] for r in rows]);P=np.array([r['present'] for r in rows],np.float32);W=np.array([r['weight'] for r in rows],np.float32);positive=np.flatnonzero(P>0);negative=np.flatnonzero(P==0)
    for step in range(steps):
        indices=np.r_[rng.choice(positive,16),rng.choice(negative,16)];x=torch.from_numpy(X[indices].copy()).cuda();t=torch.from_numpy(T[indices].copy()).cuda();p=torch.from_numpy(P[indices]).cuda();w=torch.from_numpy(W[indices]).cuda();opt.zero_grad(set_to_none=True);center,pres=m(x);b=F.binary_cross_entropy_with_logits(center,t,reduction='none',pos_weight=center.new_tensor(8)).mean(1);prob=t/t.sum(1,keepdim=True).clamp_min(1);ce=-(F.log_softmax(center,1)*prob).sum(1)*p;loss=((b+.35*ce+.65*F.binary_cross_entropy_with_logits(pres,p,reduction='none'))*w).mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),3);opt.step()
    return m.eval(),float(loss.detach())


@torch.no_grad()
def assess(m,rows):
    x=torch.from_numpy(np.stack([r['x'] for r in rows])).cuda();c,p=m(x);c=c.sigmoid().cpu().numpy();p=p.sigmoid().cpu().numpy();out=[]
    for r,v,pr in zip(rows,c,p):
        i=int(np.argmax(v));estimate=float(OFFSETS[i]);prior=r['x'][5];peaks=np.flatnonzero((prior>.55)&(prior>=np.r_[prior[0],prior[:-1]])&(prior>=np.r_[prior[1:],prior[-1]]));near=[k for k in peaks if abs(OFFSETS[k]-estimate)<10];on=r['x'][6]>.5;a=i;b=i+1
        while a>0 and on[a-1]:a-=1
        while b<len(on) and on[b]:b+=1
        width=(b-a)*.5;signal=gaussian_filter1d(r['x'][1],.8);local_peaks=[k for k in range(max(a+1,1),min(b-1,len(signal)-1)) if signal[k]>=signal[k-1] and signal[k]>=signal[k+1] and signal[k]>.45*signal[a:b].max()];multimodal=any(abs(OFFSETS[k]-OFFSETS[j])>=4 and signal[min(k,j):max(k,j)+1].min()<.7*min(signal[k],signal[j]) for k in local_peaks for j in local_peaks if k<j);geometric_ok=len(near)==1 and width<=18 and on[i] and abs(estimate)<=8 and not multimodal;out.append(dict(id=r['identity'],positive=bool(r['present']),present_probability=float(pr),center_peak=float(v[i]),expected_offset=float(r['center']),estimated_offset=estimate,center_error=abs(estimate-r['center']) if r['present'] else None,auto_accept=bool(pr>=.9 and v[i]>=.55 and geometric_ok),geometric_support=bool(geometric_ok),visible_band_width=width))
    return out


def train_reviewer():
    torch.set_num_threads(4);cv2.setNumThreads(2);folder=ROOT/'reviewer';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'final.pt').exists():raise FileExistsError('Reviewer exists')
    rows=inputs();cv=[]
    for i,group in enumerate(['reviewed_heye','sina_tutorial']):
        train=[r for r in rows if r['group']!=group];test=[r for r in rows if r['group']==group and r['anchor']];model,loss=fit(train,400,190250+i);values=assess(model,test);torch.save(dict(model=model.state_dict(),heldout_group=group),folder/('fold_'+str(i)+'.pt'));cv.extend([dict(r,heldout_group=group) for r in values]);print('V19_REVIEW_CV',group,json.dumps(values),flush=True)
    model,loss=fit(rows,600,190260);torch.save(dict(model=model.state_dict(),steps=600,user_review_sha256=read(ROOT/'review_audit.json')['user_review_sha256'],groups=['reviewed_heye','sina_tutorial']),folder/'final.pt');positive=[r for r in cv if r['positive']];accepted=[r for r in cv if r['auto_accept']];summary=dict(cross_group_records=cv,positive_mean_center_error=float(np.mean([r['center_error'] for r in positive])),positive_2px_fraction=float(np.mean([r['center_error']<=2 for r in positive])),auto_accepted_count=len(accepted),auto_accepted_positive_2px_fraction=float(np.mean([r['center_error']<=2 for r in accepted if r['positive']])) if any(r['positive'] for r in accepted) else None,accepted_background_false_positives=sum(not r['positive'] for r in accepted),available_background_anchor_count=sum(not r['positive'] for r in cv),training_examples_are_augmentations_not_new_paintings=len(rows),full_updates=600,cross_group_updates=800,confidence_thresholds_are_not_accuracy_guarantees=True,uncertain_annotations_excluded=True)
    jsave(folder/'cv_report.json',summary);print('V19_REVIEWER_TRAINED',json.dumps({k:v for k,v in summary.items() if k!='cross_group_records'}),flush=True)


def load_reviewer():
    m=Reviewer().cuda();m.load_state_dict(torch.load(ROOT/'reviewer'/'final.pt',map_location='cuda',weights_only=False)['model']);return m.eval()

if __name__=='__main__':train_reviewer()


