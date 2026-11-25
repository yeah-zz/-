from faint_partial_references_20261101 import *
from scipy.ndimage import gaussian_filter1d
def refine_reference(rgb,points):
    p=np.asarray(points,np.float32);length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];t=np.linspace(0,length[-1],max(3,int(length[-1])+1));p=np.column_stack([np.interp(t,length,p[:,i]) for i in [0,1]]).astype(np.float32);tangent=np.gradient(gaussian_filter1d(p,2,axis=0),axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-6);normal=np.c_[-tangent[:,1],tangent[:,0]];offset=np.arange(-7,7.01,.5,dtype=np.float32);xy=p[:,None,:]+normal[:,None,:]*offset[None,:,None];gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255;gray=cv2.GaussianBlur(gray,(0,0),.55)
    def sample(q):return cv2.remap(gray,q[:,:,0],q[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT101)
    center=sample(xy);contrast=np.zeros(center.shape,np.float32)
    for radius in [2.,3.5,5.]:contrast=np.maximum(contrast,np.minimum(sample(xy+normal[:,None,:]*radius),sample(xy-normal[:,None,:]*radius))-center)
    cost=-np.clip(contrast,0,.15)/.02+.018*offset[None,:]**2;total=cost[0].copy();parents=[];transition=.10*(offset[:,None]-offset[None,:])**2
    for row in cost[1:]:
        c=total[:,None]+transition;parent=np.argmin(c,axis=0);parents.append(parent);total=row+c[parent,np.arange(len(offset))]
    idx=[int(total.argmin())]
    for parent in parents[::-1]:idx.append(int(parent[idx[-1]]))
    idx=np.array(idx[::-1]);q=xy[np.arange(len(p)),idx];con=contrast[np.arange(len(p)),idx];valid=(con>.009)&(np.abs(offset[idx])<6.75);return p,q,valid,con

def main():
    manifest=read(ROOT/'partial_reference_manifest.json');folder=ROOT/'audited_references';folder.mkdir(exist_ok=True);records=[]
    for row in manifest['records']:
        rgb=np.asarray(Image.open(row['path']).convert('RGB'));target=np.zeros((5,256,256),np.float32);known=np.zeros((4,256,256),np.float32);reference=np.zeros((256,256),np.uint8);boundary=reference.copy();drawim=Image.fromarray(rgb).resize((768,768),Image.Resampling.NEAREST);draw=ImageDraw.Draw(drawim);segments=[];rejected=total=0
        for anchors in row['references'].get('ink',[]):
            original,p,valid,contrast=refine_reference(rgb,anchors);total+=len(valid);rejected+=int((~valid).sum());draw.line([tuple(q*3) for q in original],fill=(170,170,170),width=1)
            # Visible supported sections only; no interpolation over rejected samples.
            starts=np.flatnonzero(valid&~np.r_[False,valid[:-1]]);ends=np.flatnonzero(valid&~np.r_[valid[1:],False])+1
            for a,b in zip(starts,ends):
                if b-a<6:continue
                q=p[a:b];c,d,di=path_data(q);reference[c]=1;core=d<=1.25;target[0,core]=1;target[1,core]=1;target[2:4,core]=di[:,core];known[0,core]=.4;known[1,core]=.5;known[2,core]=.3;known[3,core]=.4;draw.line([tuple(z*3) for z in q],fill=(30,210,70),width=2);segments.append(q.tolist())
        for anchors in row['references'].get('boundary',[]):
            c,d,di=path_data(anchors);boundary[c]=1;core=(d<=2)&~(known[0]>0);target[4,core]=1;known[0,core]=1;known[1,core]=1;known[3,core]=1;draw.line([tuple(np.array(z)*3) for z in anchors],fill=(230,50,175),width=2)
        for x,y,xx,yy in row['references'].get('negative_boxes',[]):known[0,y:yy,x:xx]=1;known[1,y:yy,x:xx]=1;known[3,y:yy,x:xx]=1;draw.rectangle((x*3,y*3,xx*3,yy*3),outline=(50,120,240),width=2)
        file=folder/(row['id']+'.npz');np.savez_compressed(file,target=target,known=known,ink_reference=reference>0,boundary_reference=boundary>0);drawim.save(folder/(row['id']+'_overlay.jpg'),quality=96);records.append(dict(**{k:v for k,v in row.items() if k not in ['label_file','label_file_sha256']},label_file=str(file),label_file_sha256=digest(file),visible_sections=segments,raw_anchor_samples=total,unsupported_or_search_boundary_samples=rejected,label_kind='Assistant RGB-selected anchors + bounded local photometric center refinement; soft analytical partial supervision, not expert gold or model-output distillation',reference_source_detector_is_not_independent_ground_truth=True))
    jsave(ROOT/'audited_reference_manifest.json',dict(records=records,uncertainty_native_px=3.,limitations='Photometric localization remains biased; do not report heldout coverage as true extraction accuracy. Unsupported intervals unknown, not negative or connected.',automatic_full_extractions_not_used_as_truth=True));print('AUDITED_REFERENCES',len(records),sum(len(r['visible_sections']) for r in records),sum(r['unsupported_or_search_boundary_samples'] for r in records),flush=True)
if __name__=='__main__':main()
