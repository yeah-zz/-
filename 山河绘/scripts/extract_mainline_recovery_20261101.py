# -*- coding: utf-8 -*-
"""Independent image inference. No annotation files or sample IDs are inputs."""
import sys
sys.dont_write_bytecode=True
from mountain_mainline_recovery_20261101 import *
from extract_mountain_structure_v4 import extract as prepare_v4
import datetime

def describe(rgb,base,previous,masks,doc,old_decisions):
    """Same frozen feature definition as catalog, for an arbitrary image."""
    scale=max(base.shape)/496.;joined,_=whole_candidates(base,scale)
    candidates=[dict(points=r['points'] if isinstance(r,dict) else r,origin='original_v4_candidate',role=r.get('role','unknown') if isinstance(r,dict) else 'unknown') for r in doc['paths']]
    for r in joined:
        p=r['points'];q=np.rint(p).astype(int)
        if v4.plen(p)>=40*scale and masks['interior'][q[:,1],q[:,0]].all():candidates.append(dict(points=p.tolist(),origin='junction_supported_candidate',role='long_structure'))
    assert len(candidates)==len(old_decisions)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255.;density=cv2.boxFilter(base.astype(np.float32),-1,(15,15));out=[]
    for i,(row,d) in enumerate(zip(candidates,old_decisions)):
        p=np.asarray(row['points'],np.float32);q=np.rint(p).astype(int);length=v4.plen(p);span=float(np.linalg.norm(p[-1]-p[0]));contrast,center=evidence(gray,p,scale)
        tang=p[2:]-p[:-2] if len(p)>2 else p[-1:]-p[:1];tang=tang/np.maximum(np.linalg.norm(tang,axis=1,keepdims=True),1e-6);turns=np.arccos(np.clip((tang[1:]*tang[:-1]).sum(1),-1,1)) if len(tang)>1 else np.zeros(1)
        y,x=q[:,1],q[:,0];vals=[length/scale,span/max(length,1e-6),float((contrast>=.012).mean()),float((contrast>=.006).mean()),float(np.median(contrast)),float(np.quantile(contrast,.2)),float(center.mean()),float(center.std()),float(turns.mean()),float(np.quantile(turns,.9)),float(np.linalg.norm(np.ptp(p,axis=0)))/scale,float((~previous[y,x]).mean()),float(density[y,x].mean()),float(masks['interior'][y,x].mean()),float(row['role']=='long_structure')]
        out.append(dict(id=i,points=p.tolist(),origin=row['origin'],role=row['role'],old_decision=d,features=dict(zip(FEATURES,vals)),protected=bool(masks['protect'][y,x].any()),fully_interior=bool(masks['interior'][y,x].all()),old_keep=d['decision']=='keep'))
    return out

def extract(image,output):
    image=Path(image).resolve();output=Path(output)
    if not output.is_absolute() or output.drive.upper() not in ['E:','F:']:raise ValueError('Use absolute E/F output')
    if (output/'recovery/paths.json').exists():raise FileExistsError('Preserve completed extraction')
    prepare_v4(image,output/'upstream');folder=output/'upstream/structure';rgb=np.asarray(Image.open(folder/'input.png').convert('RGB'));base=np.asarray(Image.open(folder/'baseline.png'))<128;previous=np.asarray(Image.open(folder/'lines.png'))<128;z=np.load(folder/'routing_masks.npz');masks={k:z[k] for k in z.files}
    rows=describe(rgb,base,previous,masks,read(output/'upstream/detail_baseline/paths.json'),read(folder/'decisions.json'));artifact=torch.load(ROOT/'fit/model.pt',map_location='cpu',weights_only=False);data=select_recovery(rgb,base,previous,masks,rows,artifact);out=output/'recovery';out.mkdir(exist_ok=True);bw=lambda m:np.where(m,0,255).astype(np.uint8)
    for name,key in [('lines','result'),('structure','core'),('adjustable_cun','optional'),('unknown','unknown'),('restored','restored'),('bridges','bridges')]:Image.fromarray(bw(data[key])).save(out/(name+'.png'))
    canonical=whole_candidates(data['result'],max(base.shape)/496.)[0];paths=[dict(points=r['points'].tolist(),stroke_identity_verified=False) for r in canonical]
    layers={name:[dict(points=r['points'].tolist(),stroke_identity_verified=False) for r in whole_candidates(data[key],max(base.shape)/496.)[0]] for name,key in [('structure','core'),('adjustable_cun','optional'),('unknown','unknown')]}
    jsave(out/'paths.json',dict(paths=paths,layers=layers,source=str(image),source_sha256=digest(image),model_sha256=digest(ROOT/'fit/model.pt'),policy=POLICY,coordinate_system='input_image_pixels',physical_z=None,physical_speed=None,ready_for_robot=False,actually_drawn=False,role_labels_are_provisional=True));jsave(out/'decisions.json',data['decisions']);jsave(out/'bridge_events.json',data['bridge_events'])
    poly=''.join('<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in r['points'])+'"/>' for r in paths);(out/'lines.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {base.shape[1]} {base.shape[0]}"><g fill="none" stroke="black" stroke-width="1">{poly}</g></svg>',encoding='utf-8')
    jsave(output/'entry.json',dict(source=str(image),model_sha256=digest(ROOT/'fit/model.pt'),inference_code_sha256=digest(Path(__file__).with_name('mountain_mainline_recovery_20261101.py')),entry_code_sha256=digest(__file__),annotation_files_read=False,manual_region_masks_used=False,robot_connected=False,new_training_steps=0,restored_candidates=sum(r['decision']=='restore' for r in data['decisions']),bridge_pixels=int(data['bridges'].sum())))
    print('MAINLINE_IMAGE_READY',str(out/'lines.png'),flush=True)
    return data,rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--output');a=p.parse_args();out=Path(a.output) if a.output else ROOT/'manual_outputs'/(Path(a.image).stem+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));extract(a.image,out)
