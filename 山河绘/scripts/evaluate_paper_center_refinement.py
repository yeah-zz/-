"""Fixed-threshold before/after test, with paper labels scored after inference."""
import sys
sys.dont_write_bytecode=True
from paper_center_refinement_data import ROOT,PARENT,LOCAL,LAB
from joint_ink_structure_data import hard_fixture,write_record
from evaluate_joint_ink_structure import predict,load_joint,vectorize,save_output,reference_center,pixel_metrics,relation_counts,summarize_relation,partial_metric
from calibrate_joint_ink_gate import gated_center
from train_joint_ink_structure import JointModel,prior
from rock_v15_common import read,jsave,digest
from pathlib import Path
from PIL import Image
import numpy as np,cv2,torch,json
from scipy.ndimage import distance_transform_edt

def evaluate(output_root=ROOT,checkpoint_path=None,renderer=hard_fixture,seed_start=910000):
    ROOT=Path(output_root)
    if (ROOT/'evaluation_results.json').exists():raise FileExistsError('Preserve completed evaluation')
    torch.set_num_threads(4);cv2.setNumThreads(2);checkpoint=Path(checkpoint_path) if checkpoint_path else ROOT/'faint_preservation_stage2/refined.pt';after=JointModel().cuda();after.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']);after.eval();before=load_joint();settings=dict(center_threshold=.35,ink_threshold=.25,method='nms',relation_threshold=.5)
    jsave(ROOT/'evaluation_protocol.json',dict(settings=settings,thresholds_copied_from_parent=True,after_sha256=digest(checkpoint),before_sha256=digest(PARENT/'joint.pt'),fresh_seeds=[seed_start,seed_start+47],fresh_test_family='same procedural family, corrected rendering when supplied; independent seeds only',actual_paper_regression_source='G05; historical reused artworks, no full-image gold',test_used_for_threshold_or_checkpoint_selection=False,robot_connected=False))
    fresh=[];kinds=['parallel','crossing','junction','dry_continuation','needles','blank']
    for i in range(48):
        rgb,t,m,role,rel,rm,meta=renderer(seed_start+i,kinds[i%6]);meta.update(split='test',kind='new_declared',source_group='new_seed_'+str(seed_start+i));fresh.append(write_record(ROOT/'fresh_declared',f'fresh_{i:03}',rgb,t,m,role,rel,rm,meta))
    jsave(ROOT/'fresh_test_manifest.json',dict(records=fresh,used_for_training_or_calibration=False))
    actual_labels={r['id']:r for r in read(ROOT/'manifest.json')['records'] if r['kind']=='actual_refined'};papers=[r for r in read(ROOT/'manifest.json')['records'] if r['kind']=='paper_stress' and r['split']=='regression']
    art=[dict(id=r['id'],image=str(Path(r['baseline'])/'input.png'),kind='actual_artwork') for r in read(LOCAL/'baseline_manifest.json')['records']]
    for ident in ['R04','N11']:art.append(dict(id=ident,image=actual_labels[ident]['image'],kind='actual_artwork'))
    rows=[dict(r,eval_kind='declared') for r in fresh]+[dict(r,eval_kind='paper_stress') for r in papers]+[dict(r,eval_kind='actual') for r in art]
    for i,r in enumerate(rows):
        rgb=np.asarray(Image.open(r['image']).convert('RGB'))
        for tag,net in [('before',before),('after',after)]:
            cached=PARENT/'evaluation'/r['id']/'joint/prediction.npz'
            if tag=='before' and r['eval_kind']=='actual' and cached.exists():
                expected=np.asarray(Image.open(LOCAL/'baseline'/r['id']/'input.png').convert('RGB'));assert np.array_equal(expected,rgb);pred=np.load(cached)['prediction'].astype(np.float32)
            else:pred=predict(net,rgb).astype(np.float16).astype(np.float32)
            mask=gated_center(pred,settings['center_threshold'],settings['ink_threshold']);paths,meta=vectorize(mask,pred,rgb,dict(relation=dict(threshold=.5)))
            meta.update(settings=settings,checkpoint_sha256=digest(checkpoint if tag=='after' else PARENT/'joint.pt'))
            save_output(ROOT/'evaluation'/r['id']/tag/'joint',rgb,pred,mask,paths,meta,persist_prediction=r['eval_kind']=='declared')
        if i%16==0:print('PAPER_CENTER_EVALUATE',i,len(rows),flush=True)
    jsave(ROOT/'prediction_stage_complete.json',dict(count=len(rows),test_labels_opened_during_prediction=False))
    results=[]
    for r in rows:
        row=dict(id=r['id'],kind=r['eval_kind'])
        truth=reference_center(r) if r['eval_kind']=='declared' else None
        z=np.load(r['data']) if r['eval_kind'] in ['declared','paper_stress'] else (np.load(actual_labels[r['id']]['data']) if r['id'] in actual_labels else None)
        if r['eval_kind']=='declared':row['case']=read(r['reference'])['case']
        for tag in ['before','after']:
            folder=ROOT/'evaluation'/r['id']/tag/'joint';mask=np.asarray(Image.open(folder/'raw_center.png'))<128;info={}
            if r['eval_kind']=='declared':
                info['center']=pixel_metrics(mask,truth) if truth.any() else dict(f1=None,blank_fp_fraction=float(mask.mean()))
                if truth.any():
                    rgb=np.asarray(Image.open(r['image']).convert('RGB'));contrast=prior.photometry(rgb)[1];faint=truth&(contrast<.12);distance=distance_transform_edt(~mask);info['faint_reference_pixels']=int(faint.sum());info['faint_matched_pixels']=int((distance[faint]<=1.5).sum())
                pred=np.load(folder/'prediction.npz')['prediction'].astype(np.float32);info['relation']=relation_counts(pred,z,.5)
            elif r['eval_kind']=='paper_stress':info['paper_fp_pixels']=int(mask.sum());info['paper_pixels']=int(mask.size)
            elif z is not None:
                paper=z['paper_mask']>0;info['paper_fp_pixels']=int(mask[paper].sum());info['paper_pixels']=int(paper.sum())
                if z['target'][0].max()>.3:
                    info['partial_nonexpert_center']=partial_metric(mask,z);ref=(z['target'][0]>.3)&(z['weight'][0]>0);distance=distance_transform_edt(~ref);band=(z['target'][1]>.5)&(z['weight'][0]>0);info['off_center_band_pixels']=int((mask&band&(distance>2.)).sum());info['evaluated_band_line_pixels']=int((mask&band).sum())
            row[tag]=info
        results.append(row)
    jsave(ROOT/'evaluation_results.json',dict(records=results,actual_full_image_accuracy=None,new_actual_artworks=False,fresh_program_test=True,full_image_default_v4_unchanged=True))
    print('PAPER_CENTER_EVALUATED',len(results),flush=True)
if __name__=='__main__':
    from corrected_stroke_fixture import fixture
    stage=ROOT/'corrected_renderer_stage3'
    evaluate(output_root=stage,checkpoint_path=stage/'fit/refined.pt',renderer=fixture,seed_start=950000)
