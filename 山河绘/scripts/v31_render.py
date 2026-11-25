# -*- coding: utf-8 -*-
"""Differentiable swept ink ribbons with continuous cubic curves.

Own experimental renderer, not a reproduction of DiffVG or a physical brush.
Only image and machine-generated candidates are accepted by fit_image.
"""
from v31_profiles import ROOT,OFFSETS,observed,features,brush_profile,load
from rock_v15_common import read,jsave,digest
import numpy as np,cv2,torch,time,math
from torch import nn
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from stroke_link_v12 import dense

def resample(path,n):
    p=np.asarray(path,np.float32);a=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];a,idx=np.unique(a,return_index=True);p=p[idx]
    if len(p)<2:return np.repeat(p,n,axis=0)
    s=np.linspace(0,a[-1],n);return np.stack([np.interp(s,a,p[:,k]) for k in range(2)],1).astype(np.float32)

def profiles(alpha,p,normals):
    q=p[:,None]+normals[:,None]*OFFSETS[None,:,None]
    return cv2.remap(alpha,q[:,:,0].astype(np.float32),q[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

def classical(x):
    out=[]
    for v in x:
        v=gaussian_filter1d(v,.7);near=np.flatnonzero(np.abs(OFFSETS)<=2.5);peak=int(near[np.argmax(v[near])]);a=float(v[peak]);cut=max(.012,.45*a);left=peak;right=peak
        while left>0 and v[left-1]>cut:left-=1
        while right<48 and v[right+1]>cut:right+=1
        shift=np.clip(.5*(OFFSETS[left]+OFFSETS[right]),-2.5,2.5);out.append([shift,np.clip(.25*(right-left+1),.45,12),max(.01,a),min(1,a/.08)])
    return np.asarray(out,np.float32)

def logit(a):return np.log(np.clip(a,1e-4,1-1e-4)/np.clip(1-a,1e-4,1))

def adaptive_curve(path,scale,tolerance=.75):
    """Preserve machine candidates before optimizing image appearance.

    Knot placement uses source polyline error, never reference annotations.
    """
    p=np.asarray(path,np.float32);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];arc,ii=np.unique(arc,return_index=True);p=p[ii];length=float(arc[-1]);arc=arc/max(length,1e-6);grid=np.linspace(0,1,max(32,min(4000,int(math.ceil(length*2))+1)));source=np.stack([np.interp(grid,arc,p[:,k]) for k in range(2)],1);knots=np.linspace(0,1,4)
    for _ in range(92):
        controls=np.stack([np.interp(knots,arc,p[:,k]) for k in range(2)],1);spline=CubicSpline(knots,controls,bc_type='natural');error=np.linalg.norm(spline(grid)-source,axis=1);worst=int(np.argmax(error))
        if error[worst]<=tolerance:break
        where=float(grid[worst])
        if np.min(np.abs(knots-where))<1e-7:break
        knots=np.sort(np.r_[knots,where])
    controls=np.stack([np.interp(knots,arc,p[:,k]) for k in range(2)],1).astype(np.float32);samples=np.unique(np.r_[np.linspace(0,1,min(256,max(12,int(math.ceil(length*scale/2))+1))),knots]);basis=CubicSpline(knots,np.eye(len(knots)),bc_type='natural')(samples).astype(np.float32)
    error=np.linalg.norm(CubicSpline(knots,controls,bc_type='natural')(grid)-source,axis=1)
    return controls,basis,dict(knots=len(knots),source_parameter_aligned_max_error_px=float(error.max()),source_parameter_aligned_p95_error_px=float(np.percentile(error,95)),tolerance_px=tolerance,tolerance_met=bool(error.max()<=tolerance+.001))

class Ribbon(nn.Module):
    def __init__(self,path,alpha,scale,shape,model,variant):
        super().__init__();p=np.asarray(path,np.float32);controls,basis,self.geometry_check=adaptive_curve(p,scale);k=len(controls);t=np.gradient(controls,axis=0);t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-6);normals=np.c_[-t[:,1],t[:,0]];x=profiles(alpha,controls,normals)
        if variant=='learned':
            with torch.no_grad():params=model(features(x)).cpu().numpy()
        else:params=classical(x)
        self.register_buffer('base',torch.tensor(controls,device='cuda'));self.register_buffer('normal',torch.tensor(normals,device='cuda'));self.register_buffer('basis',torch.tensor(basis,device='cuda'));self.register_buffer('initial_params',torch.tensor(params,device='cuda'));self.shift=nn.Parameter(torch.tensor(np.arctanh(np.clip(params[:,0]/2.5,-.95,.95)),device='cuda'));self.radius=nn.Parameter(torch.tensor(logit((params[:,1]-.35)/11.65),device='cuda'));self.darkness=nn.Parameter(torch.tensor(logit(params[:,2]),device='cuda'));self.scale=scale
        h,w=shape;left,top=np.floor(p.min(0)*scale-14*scale-3).astype(int);right,bottom=np.ceil(p.max(0)*scale+14*scale+3).astype(int);left=max(0,left);top=max(0,top);right=min(w,right+1);bottom=min(h,bottom+1);yy,xx=np.mgrid[top:bottom,left:right];self.register_buffer('pixels',torch.tensor(np.c_[xx.ravel(),yy.ravel()],device='cuda',dtype=torch.float32));self.register_buffer('indices',torch.tensor((yy*w+xx).ravel(),device='cuda',dtype=torch.long));self.n_pixels=h*w
    def samples(self):
        normal=self.basis@self.normal;normal=normal/normal.norm(dim=1,keepdim=True).clamp_min(1e-6);p=self.basis@self.base+normal*(2.5*(self.basis@self.shift).tanh())[:,None];r=(self.basis@(.35+11.65*self.radius.sigmoid())).clamp(.35,12);a=(self.basis@self.darkness).sigmoid();return p,r,a
    def render(self):
        p,r,a=self.samples();p=p*self.scale;r=r*self.scale;start=p[:-1];delta=p[1:]-start;v=self.pixels[:,None]-start[None];along=(v*delta[None]).sum(2)/delta.square().sum(1).clamp_min(.01)[None];along=along.clamp(0,1);distance=(v-along[:,:,None]*delta[None]).square().sum(2);dist,idx=distance.min(1);t=along.gather(1,idx[:,None]).squeeze(1);radius=r[idx]*(1-t)+r[idx+1]*t;opacity=a[idx]*(1-t)+a[idx+1]*t;coverage=torch.sigmoid((radius-torch.sqrt(dist+1e-6))/.45);ink=(opacity*coverage).clamp(0,.995);return -torch.log1p(-ink),ink
    def regularization(self):
        s=2.5*self.shift.tanh();r=.35+11.65*self.radius.sigmoid();a=self.darkness.sigmoid();curvature=(s[2:]-2*s[1:-1]+s[:-2]).square().mean();smooth=(r[1:]-r[:-1]).square().mean()/36+(a[1:]-a[:-1]).square().mean();return .025*s.square().mean()/6.25+.012*curvature+.004*smooth

class PackedRibbons(nn.Module):
    """Same swept-ribbon equations, batched to avoid per-stroke GPU launches."""
    def __init__(self,ribbons):
        super().__init__();self.count=len(ribbons);self.counts=[r.basis.shape[0] for r in ribbons];self.scale=ribbons[0].scale;k=max(len(r.base) for r in ribbons);n=max(self.counts);s=len(ribbons)
        base=torch.zeros(s,k,2,device='cuda');normal=base.clone();basis=torch.zeros(s,n,k,device='cuda');mask=torch.zeros(s,k,device='cuda');segmask=torch.zeros(s,n-1,device='cuda',dtype=torch.bool);params={name:torch.zeros(s,k,device='cuda') for name in ['shift','radius','darkness']};pixels=[];indices=[];owner=[]
        for i,r in enumerate(ribbons):
            kk=len(r.base);nn=r.basis.shape[0];base[i,:kk]=r.base;normal[i,:kk]=r.normal;basis[i,:nn,:kk]=r.basis;basis[i,nn:,:kk]=r.basis[-1];mask[i,:kk]=1;segmask[i,:nn-1]=True
            for name in params:params[name][i,:kk]=getattr(r,name).detach()
            pixels.append(r.pixels);indices.append(r.indices);owner.append(torch.full((len(r.indices),),i,device='cuda',dtype=torch.long))
        for name,value in dict(base=base,normal=normal,basis=basis,mask=mask,segment_mask=segmask,pixels=torch.cat(pixels),indices=torch.cat(indices),owner=torch.cat(owner)).items():self.register_buffer(name,value)
        for name,value in params.items():setattr(self,name,nn_module_parameter(value))
    def __len__(self):return self.count
    def samples(self):
        normal=torch.bmm(self.basis,self.normal);normal=normal/normal.norm(dim=2,keepdim=True).clamp_min(1e-6);offset=2.5*torch.bmm(self.basis,self.shift[:,:,None]).tanh();p=torch.bmm(self.basis,self.base)+normal*offset;r=torch.bmm(self.basis,(.35+11.65*self.radius.sigmoid())[:,:,None]).squeeze(2).clamp(.35,12);a=torch.bmm(self.basis,self.darkness[:,:,None]).squeeze(2).sigmoid();return p,r,a
    def regularization(self):
        s=2.5*self.shift.tanh();r=.35+11.65*self.radius.sigmoid();a=self.darkness.sigmoid();count=self.mask.sum(1);m1=self.mask[:,1:];m2=self.mask[:,2:];curvature=((s[:,2:]-2*s[:,1:-1]+s[:,:-2]).square()*m2).sum(1)/(count-2);smooth=((r[:,1:]-r[:,:-1]).square()/36+(a[:,1:]-a[:,:-1]).square())*m1;return (.025*(s.square()*self.mask).sum(1)/count/6.25+.012*curvature+.004*smooth.sum(1)/(count-1)).mean()
    def raster(self,shape,return_layers=False):
        p,r,a=self.samples();p=p*self.scale;r=r*self.scale;taus=[];inks=[]
        for begin in range(0,len(self.pixels),20000):
            end=begin+20000;who=self.owner[begin:end];points=p[who];start=points[:,:-1];delta=points[:,1:]-start;v=self.pixels[begin:end,None]-start;along=(v*delta).sum(2)/delta.square().sum(2).clamp_min(.01);along=along.clamp(0,1);d2=(v-along[:,:,None]*delta).square().sum(2);d2=d2.masked_fill(~self.segment_mask[who],1e8);dist,idx=d2.min(1);u=along.gather(1,idx[:,None]).squeeze(1);rad=r[who,idx]*(1-u)+r[who,idx+1]*u;opacity=a[who,idx]*(1-u)+a[who,idx+1]*u;coverage=torch.sigmoid((rad-torch.sqrt(dist+1e-6))/.45);ink=(opacity*coverage).clamp(0,.995);inks.append(ink);taus.append(-torch.log1p(-ink))
        tau=torch.cat(taus);ink=torch.cat(inks);total=torch.zeros(shape[0]*shape[1],device='cuda').index_add(0,self.indices,tau);sumink=torch.zeros_like(total).index_add(0,self.indices,ink);image=-torch.expm1(-total);layers=[]
        if return_layers:
            for i in range(self.count):
                m=self.owner==i;layers.append((self.indices[m],tau[m],ink[m]))
        return image.reshape(shape),(sumink-image).mean(),layers

def nn_module_parameter(value):return nn.Parameter(value)

def render(ribbons,shape,return_layers=False):
    if isinstance(ribbons,PackedRibbons):return ribbons.raster(shape,return_layers)
    tau=torch.zeros(shape[0]*shape[1],device='cuda');sumink=torch.zeros_like(tau);layers=[]
    for ribbon in ribbons:
        t,a=ribbon.render();tau=tau.index_add(0,ribbon.indices,t);sumink=sumink.index_add(0,ribbon.indices,a)
        if return_layers:layers.append((ribbon.indices,t,a))
    image=-torch.expm1(-tau);overlap=(sumink-image).mean();return image.reshape(shape),overlap,layers

def fit_image(rgb,paths,variant='learned',steps=100,sparsity=.002):
    """Reference-free per-image parameter optimization, distinct from training."""
    torch.set_num_threads(4);cv2.setNumThreads(2);alpha=observed(rgb);h,w=alpha.shape;scale=min(1.,320/max(h,w));hh=max(8,round(h*scale));ww=max(8,round(w*scale));scale=min(ww/w,hh/h);hh=round(h*scale);ww=round(w*scale);target=torch.tensor(cv2.resize(alpha,(ww,hh),interpolation=cv2.INTER_AREA),device='cuda');chosen=[p for p in paths if len(p)>1 and np.linalg.norm(np.diff(p,axis=0),axis=1).sum()>=6];model=load() if variant=='learned' else None;start=time.perf_counter();ribbons=nn.ModuleList([Ribbon(p,alpha,scale,(hh,ww),model,variant) for p in chosen]);del model
    if not len(ribbons):return dict(paths=[],apparent_width_px=[],apparent_ink_alpha=[],rendered=np.zeros((h,w),np.float32),before_rendered=np.zeros((h,w),np.float32),log=[],meta=dict(no_candidates=True))
    geometry_checks=[r.geometry_check for r in ribbons];ribbons=PackedRibbons(ribbons);opt=torch.optim.Adam(ribbons.parameters(),lr=.045);weight=.35+.65*(target>.035).float();log=[];torch.cuda.reset_peak_memory_stats();best=1e9;best_state=None
    with torch.no_grad():before,_,_=render(ribbons,(hh,ww));before=before.cpu().numpy()
    for step in range(steps+1):
        image,overlap,_=render(ribbons,(hh,ww));photometric=((image-target).abs()*weight).mean();reg=ribbons.regularization();loss=photometric+reg+sparsity*overlap
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite inverse-rendering loss')
        if float(loss.detach())<best:best=float(loss.detach());best_state={k:v.detach().clone() for k,v in ribbons.named_parameters()}
        if step%25==0 or step==steps:log.append(dict(step=step,total=float(loss.detach()),photometric=float(photometric.detach()),regularization=float(reg.detach()),overlap=float(overlap.detach()),elapsed_seconds=time.perf_counter()-start))
        if step==steps:break
        opt.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(ribbons.parameters(),2)
        if not torch.isfinite(grad):raise RuntimeError('Nonfinite curve gradient')
        opt.step()
    with torch.no_grad():
        for name,param in ribbons.named_parameters():param.copy_(best_state[name])
    # Delete only a highly redundant stroke whose removal leaves observed ink
    # reconstruction essentially unchanged. Ambiguous overlaps are retained.
    with torch.no_grad():
        image,_,layers=render(ribbons,(hh,ww),return_layers=True);tau=-torch.log1p(-image.clamp(max=.999999)).flatten();current_error=((image-target).abs()*weight).mean();keep=list(range(len(ribbons)));pruning=[]
        for i in sorted(keep,key=lambda i:float(layers[i][2].sum())):
            index,t,a=layers[i];other=-torch.expm1(-(tau[index]-t).clamp_min(0));redundancy=float(((other>.6*a)&(a>.03)).float().sum()/(a>.03).sum().clamp_min(1))
            if redundancy<.90:continue
            proposed=tau.clone();proposed[index]=(proposed[index]-t).clamp_min(0);img=-torch.expm1(-proposed).reshape(hh,ww);err=((img-target).abs()*weight).mean();increase=float(err-current_error);local=increase*hh*ww/max(1,int((a>.03).sum()))
            if local>.003:continue
            tau=proposed;current_error=err;keep.remove(i);pruning.append(dict(source_curve=i,overlap_fraction=redundancy,local_error_increase=local))
        result=[];widths=[];tones=[];allp,allr,alla=ribbons.samples()
        for i in keep:
            n=ribbons.counts[i];p=allp[i,:n].cpu().numpy();r=allr[i,:n];a=alla[i,:n];p[:,0]=np.clip(p[:,0],0,w-1);p[:,1]=np.clip(p[:,1],0,h-1);result.append(p.tolist());widths.append((2*r).cpu().tolist());tones.append(a.cpu().tolist())
        fitted=-torch.expm1(-tau).reshape(hh,ww).cpu().numpy();final_error=float(((torch.tensor(fitted,device='cuda')-target).abs()*weight).mean())
    return dict(paths=result,apparent_width_px=widths,apparent_ink_alpha=tones,rendered=cv2.resize(fitted,(w,h)),before_rendered=cv2.resize(before,(w,h)),log=log,meta=dict(actual_device='cuda:0',gpu=torch.cuda.get_device_name(),steps=steps,variant=variant,working_shape=[hh,ww],normal_shift_limit_original_px=2.5,new_strokes_added=0,source_curves=len(chosen),retained_curves=len(keep),pruning=pruning,final_photometric_error=final_error,geometry_checks=geometry_checks,peak_gpu_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.perf_counter()-start,reference_labels_used=False,per_image_optimization_not_neural_training=True,renderer='batched_differentiable_round_capped_segment_sweep_on_adaptive_continuous_cubic_spline',renderer_revision=3,width_is_apparent_not_calibrated_pressure=True))
