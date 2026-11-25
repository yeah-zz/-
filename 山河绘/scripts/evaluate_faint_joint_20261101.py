# -*- coding: utf-8 -*-
from faint_joint_model_20261101 import *
import split_detail_v4_frozen as graph
from mountain_structure_v4_20261101 import whole_candidates
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import binary_propagation,distance_transform_edt
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree
import argparse,math
CFG=dict(weak=.25,strong=.55,ink=.25,edge=.70,source_valley=.006,min_length=12.,min_direction_alignment=.30)

def trace(rgb,o,cfg,allowed=None):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255;gray=cv2.GaussianBlur(gray,(0,0),.35);valley=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)));score=o[0]*np.sqrt(np.clip(o[1],0,1))*(1-.65*o[4]);weak=(score>=cfg['weak'])&(o[1]>=cfg['ink'])&(o[4]<=cfg['edge'])&(valley>=cfg['source_valley']);strong=(score>=cfg['strong'])&weak;ridge=skeletonize(binary_propagation(strong,mask=weak));rows,_=whole_candidates(ridge,1.);mask=np.zeros(gray.shape,bool);selected=[]
    for r in rows:
        p=r['points'];length=graph.plen(p)
        if length<cfg['min_length']:continue
        tan=np.gradient(p,axis=0);angle=np.arctan2(tan[:,1],tan[:,0]);q=np.rint(p).astype(int);x,y=q[:,0],q[:,1];alignment=float(np.mean(np.cos(2*angle)*o[2,y,x]+np.sin(2*angle)*o[3,y,x]))
        if alignment<cfg['min_direction_alignment']:continue
        if allowed is not None and float(allowed[y,x].mean())<.90:continue
        mask[y,x]=True;selected.append(dict(points=p.tolist(),length=length,mean_score=float(score[y,x].mean()),direction_alignment=alignment,mean_color_boundary_probability=float(o[4,y,x].mean()),stroke_identity_verified=False))
    return mask,selected,score

def partial_metrics(mask,row):
    z=np.load(row['label_file']);reference=z['ink_reference'];edge=z['boundary_reference'];d=distance_transform_edt(~mask);coverage=float((d[reference]<=3).mean()) if reference.any() else None;edgehit=int((d[edge]<=3).sum());paper=0;area=0
    for x,y,xx,yy in row['references'].get('negative_boxes',[]):paper+=int(mask[y:yy,x:xx].sum());area+=(yy-y)*(xx-x)
    return dict(partial_reference_coverage_3px=coverage,reference_pixels=int(reference.sum()),color_boundary_ref_hits_3px=edgehit,color_boundary_reference_pixels=int(edge.sum()),line_pixels_in_checked_negative_boxes=paper,checked_negative_pixels=area,photometric_reference_bias=True,not_true_extraction_accuracy=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--predict-only',action='store_true');p.add_argument('--ids');p.add_argument('--revision',default='r2',choices=['r1','r2']);a=p.parse_args();torch.set_num_threads(4);cv2.setNumThreads(2);artifact=torch.load(ROOT/('fit_r2' if a.revision=='r2' else 'fit')/'model.pt',map_location='cpu',weights_only=False);model=FaintNet().cuda();model.load_state_dict(artifact['model']);folder=ROOT/('local_predictions_r2' if a.revision=='r2' else 'local_predictions');folder.mkdir(exist_ok=True);records=read(ROOT/'audited_reference_manifest.json')['records'];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for row in records:
        if a.ids and row['id'] not in a.ids.split(','):continue
        out=folder/row['id'];out.mkdir(exist_ok=True);rgb=np.asarray(Image.open(row['path']).convert('RGB'));file=out/'fields.npz'
        if file.exists():o=np.load(file)['fields']
        else:o=predict(rgb,model);np.savez_compressed(file,fields=o)
        if a.predict_only:continue
        mask,paths,score=trace(rgb,o,CFG);old=np.asarray(Image.open(PRIOR/'outputs_r3'/row['image']/'lines.png'))<128;x,y,xx,yy=row['box'];baseline=old[y:yy,x:xx];Image.fromarray(np.where(mask,0,255).astype(np.uint8)).save(out/'model_only.png');Image.fromarray(np.where(baseline,0,255).astype(np.uint8)).save(out/'previous.png');heat=cv2.applyColorMap(np.rint(o[0]*255).astype(np.uint8),cv2.COLORMAP_VIRIDIS)[:,:,::-1];Image.fromarray(heat).save(out/'center_probability.png');jsave(out/'metrics.json',dict(id=row['id'],split=row['split'],model=partial_metrics(mask,row),previous=partial_metrics(baseline,row)));jsave(out/'paths.json',dict(paths=paths,config=CFG,ready_for_robot=False));board=Image.new('RGB',(1536,430),'#f3f1eb');d=ImageDraw.Draw(board)
        for j,(im,title) in enumerate([(Image.fromarray(rgb),'原图'),(Image.fromarray(np.where(baseline,0,255).astype(np.uint8)),'上轮结果'),(Image.fromarray(np.where(mask,0,255).astype(np.uint8)),'模型单独候选（未合并）'),(Image.fromarray(heat),'中心概率')]):im=im.convert('RGB').resize((384,384),Image.Resampling.NEAREST);d.text((j*384+4,5),row['id']+' '+title,font=font,fill='black');board.paste(im,(j*384,40))
        board.save(out/'comparison.jpg',quality=97)
    print('FAINT_LOCAL_PREDICTIONS_READY',flush=True)
if __name__=='__main__':main()
