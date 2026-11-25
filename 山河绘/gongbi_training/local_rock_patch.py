"""Bounded image-only fitting of existing paths. No learned semantic model."""
import numpy as np,cv2,torch
import torch.nn.functional as F

def resample(points,step=.4):
    p=np.asarray(points,dtype='float64');keep=np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-7];p=p[keep]
    a=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];s=np.linspace(0,a[-1],max(3,int(np.ceil(a[-1]/step))+1))
    return np.stack([np.interp(s,a,p[:,j]) for j in [0,1]],1)

def contrast_field(rgb):
    g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype('float32')/255
    return cv2.GaussianBlur(g,(0,0),2)-g

def sample_field(field,p):
    q=np.asarray(p,dtype='float32');return cv2.remap(field,q[:,0,None],q[:,1,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101).ravel()

def fit_existing_path(rgb,points,max_shift=.65,steps=160):
    torch.set_num_threads(4);p=resample(points);t=np.gradient(p,axis=0);t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-8);n=np.stack([-t[:,1],t[:,0]],1)
    field=contrast_field(rgb);image=torch.from_numpy(field)[None,None];base=torch.from_numpy(p.astype('float32'));normal=torch.from_numpy(n.astype('float32'));w,h=rgb.shape[1],rgb.shape[0]
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];k=min(10,max(3,int(arc[-1]/7)+2));control=torch.zeros((1,1,k),requires_grad=True);opt=torch.optim.Adam([control],lr=.05);ramp=torch.sin(torch.linspace(0,torch.pi,len(p)));ramp[0]=ramp[-1]=0
    history=[]
    for step in range(steps):
        opt.zero_grad();shift=F.interpolate(control,size=len(p),mode='linear',align_corners=True).flatten().tanh()*max_shift*ramp;q=base+normal*shift[:,None];grid=torch.stack([q[:,0]/(w-1)*2-1,q[:,1]/(h-1)*2-1],1)[None,None]
        score=F.grid_sample(image,grid,mode='bilinear',padding_mode='border',align_corners=True).mean();regular=.008*shift.square().mean()+.08*(shift[1:]-shift[:-1]).square().mean();loss=-score+regular;loss.backward();opt.step()
        if step%20==0 or step==steps-1:history.append(dict(step=step,loss=float(loss.detach()),contrast=float(score.detach())))
    with torch.no_grad():
        shift=F.interpolate(control,size=len(p),mode='linear',align_corners=True).flatten().tanh()*max_shift*ramp;q=(base+normal*shift[:,None]).numpy().astype('float64')
    q[0]=np.asarray(points)[0];q[-1]=np.asarray(points)[-1]
    before=float(sample_field(field,p).mean());after=float(sample_field(field,q).mean());used=after>=before+1e-5
    return (q if used else np.asarray(points,dtype='float64')),dict(used=used,mean_contrast_before=before,mean_contrast_after=after,max_normal_shift=float(abs(shift.numpy()).max()),endpoint_preserved=True,history=history,optimizer_not_recognition_training=True)
