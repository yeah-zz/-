# -*- coding: utf-8 -*-
"""Reference-free automatic QC on extracted paths; abstention retains baseline."""
from review_guided_train_v19 import *
from datetime import datetime,timezone,timedelta


@torch.no_grad()
def review_paths(rgb,paths,model):
    if not v16.monochrome_route(rgb):return dict(scope='monochrome_rock_only',color_domain_not_validated=True,paths_changed=False,expert_truth=False,user_review_required=False,records=[])
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255;rows=[];samples=[]
    for index,path in enumerate(paths):
        q=dense(path)
        for k in range(5,len(q)-5,12):
            delta=q[min(len(q)-1,k+4)]-q[max(0,k-4)];norm=np.linalg.norm(delta)
            if norm<1:continue
            u=delta/norm;n=np.array([-u[1],u[0]],np.float32);x=features_with_gray(gray,q[k],n);samples.append(dict(path=index,point=q[k].tolist(),x=x,present=1,identity=str(index),center=0.))
    for start in range(0,len(samples),128):
        group=samples[start:start+128];decisions=reviewer.assess(model,group)
        for row,d in zip(group,decisions):
            if d['auto_accept'] and abs(d['estimated_offset'])<=2:status='automatic_center_consistent'
            elif d['present_probability']<.05 and row['x'][1].max()<.012:status='probable_background'
            else:status='uncertain_kept_with_baseline'
            rows.append(dict(path=row['path'],point=row['point'],status=status,**d))
    return dict(records=rows,counts=dict(Counter(r['status'] for r in rows)),scope='monochrome_rock_local_cross_sections',learned_from_user_review=True,expert_truth=False,paths_changed=False,uncertain_policy='retain baseline, do not auto-label as background',user_review_required=False,model_sha256=digest(ROOT/'reviewer'/'final.pt'),robot_connected=False)


def overlay(rgb,qc):
    image=Image.fromarray(rgb.copy());d=ImageDraw.Draw(image)
    for r in qc['records']:
        x,y=r['point'];color={'automatic_center_consistent':'#289653','probable_background':'#ba4150','uncertain_kept_with_baseline':'#cd8c15'}[r['status']];d.ellipse((x-1.8,y-1.8,x+1.8,y+1.8),fill=color)
    return np.asarray(image)


def run(input_path,output_path=None):
    import ink_predict_v16 as stable
    output=Path(output_path) if output_path else ROOT/'automatic_batches'/datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    if output.drive.upper()=='C:':raise ValueError('Output must stay on E or F')
    stable.run(input_path,str(output));summary=read(output/'summary.json');model=reviewer.load_reviewer()
    for row in summary['records']:
        dest=Path(row['comparison']).parent;rgb=np.asarray(Image.open(row['source']).convert('RGB'));trajectory=read(dest/'candidate_trajectory.json');qc=review_paths(rgb,trajectory['paths'],model);jsave(dest/'v19_automatic_review.json',qc);Image.fromarray(overlay(rgb,qc)).save(dest/'v19_review_overlay.png');row.update(automatic_review=str(dest/'v19_automatic_review.json'),uncertain_policy='keep baseline',user_review_required=False)
    summary.update(automatic_reviewer='v19 learned + geometry',automatic_review_is_not_expert_truth=True,user_review_required=False,weights_for_extraction='protected v16 baseline',robot_connected=False);jsave(output/'summary.json',summary);print('V19_AUTOMATIC_BATCH',str(output),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output');a=p.parse_args();run(a.input,a.output)
