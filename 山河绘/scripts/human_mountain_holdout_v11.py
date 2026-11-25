# -*- coding: utf-8 -*-
"""A06 user-label check: same-work internal holdout, not independent artwork."""
import sys,json,shutil
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
from human_mountain_v11 import REVIEW,ROOT,PARENT,fields,digest,jsave,FONT,SMALL,paste
from human_mountain_eval_v11 import extract,metrics,draw_result
from gongbi_training.review_server_v11 import validate_document
from gongbi_training.stroke_train_v08 import model_new,infer

def run():
    root=ROOT/'refinement_sparse';folder=root/'evaluation'/'A06_same_work_holdout';folder.mkdir(exist_ok=True)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'));assert 'A06' not in [r['id'] for r in manifest['records']]
    review=json.loads((REVIEW/'package_manifest.json').read_text(encoding='utf-8'));r=next(x for x in review['samples'] if x['id']=='A06');source=REVIEW/'labels'/'A06.json';doc=json.loads(source.read_text(encoding='utf-8'));validate_document(doc,r,review['package_id'])
    if doc['review_state']!='checked_partial':raise ValueError('A06 is not user checked')
    shutil.copy2(source,folder/'user_labels_snapshot.json');shutil.copy2(REVIEW/r['image'],folder/'original.png');rgb=np.asarray(Image.open(folder/'original.png').convert('RGB'));center,_,mask,_=fields(doc);overlap=np.zeros(center.shape,np.uint8);x0,y0,x1,y1=r['source_box'];overlap_boxes=[]
    for t in manifest['records']:
        if t['source_id']!=r['source_id']:continue
        a,b,c,d=t['source_box'];left=max(a,x0);top=max(b,y0);right=min(c,x1);bottom=min(d,y1)
        if left<right and top<bottom:
            box=[max(0,left-x0-16),max(0,top-y0-16),min(x1-x0,right-x0+16),min(y1-y0,bottom-y0+16)];overlap[box[1]:box[3],box[0]:box[2]]=1;overlap_boxes.append(dict(training_id=t['id'],excluded_local_box=box))
    mask[:,overlap>0]=0;center=center*(mask[3]>0)
    torch.set_num_threads(4);cv2.setNumThreads(2);model=model_new().cuda();model.load_state_dict(torch.load(PARENT,map_location='cuda',weights_only=False)['model']);model.eval();before=infer(model,rgb);model.load_state_dict(torch.load(root/'final.pt',map_location='cuda',weights_only=False)['model']);after=infer(model,rgb)
    oldsk,oldpaths=extract(before);newsk,newpaths=extract(after);oldline,_=draw_result(rgb,oldsk);newline,overlay=draw_result(rgb,newsk);reference=np.full_like(rgb,255);reference[cv2.dilate(center,np.ones((2,2),np.uint8))>0]=[235,75,20]
    for name,a in [('user_reference_active',reference),('before',oldline),('after',newline),('after_overlay',overlay)]:Image.fromarray(a).save(folder/(name+'.png'))
    np.savez_compressed(folder/'probabilities.npz',before=before,after=after,scoring_valid_mask=mask[3],known_overlap=overlap)
    board=Image.new('RGB',(2000,500),'#f5f3ec');dd=ImageDraw.Draw(board);dd.text((10,5),'A06：未参与本轮训练的同画作内部对照；评分排除已知重叠和紫色区，非独立画作测试',font=FONT,fill='black')
    for j,(label,a) in enumerate([('原图',rgb),('你的有效参考线',reference),('训练前候选线',oldline),('训练后候选线',newline),('训练后叠加',overlay)]):dd.text((j*400+8,40),label,font=SMALL,fill='black');paste(board,a,j*400+8,75,384,410)
    board.save(folder/'comparison.jpg',quality=95)
    report=dict(id='A06',training_sample=False,checkpoint_selected_using_this=False,independent_test=False,source_group=r['source_group'],user_strokes=len(doc['strokes']),scoring_valid_pixels=int(mask[3].sum()),known_overlap_exclusions=overlap_boxes,unknown_cross_reference_overlap_possible=True,
      before=metrics(doc,before,oldsk,mask),after=metrics(doc,after,newsk,mask),label_sha256=digest(folder/'user_labels_snapshot.json'),raw_predictions_not_masked_using_annotations=True,robot_connected=False)
    jsave(folder/'metrics.json',report);print('A06_INTERNAL_CHECK',json.dumps(dict(before=report['before']['partial_reference_coverage_3px'],after=report['after']['partial_reference_coverage_3px'],excluded_overlap=overlap_boxes)),flush=True)
    # Specific regression checks for the filled-region-to-medial-axis failure.
    uniform=np.zeros((6,80,80),np.float32);uniform[0]=.7;uniform[1]=1;sk,_=extract(uniform);assert not sk.any()
    yy,xx=np.mgrid[:80,:80];single=uniform.copy();single[0]=np.exp(-(yy-40.)**2/(2*1.2**2))*((xx>=5)&(xx<=74));sk,_=extract(single);assert sk[40,10:70].mean()>.9
    jsave(root/'evaluation_checks.json',dict(uniform_response_does_not_create_medial_axis=True,known_straight_ridge_retained=True,purple_excluded_from_all_losses_verified=True,both_models_same_postprocessing=True,raw_predictions_not_ground_truth_masked=True,independent_artwork_validation=False))
if __name__=='__main__':run()
