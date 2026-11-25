"""Stage two: calibrate both learned heads on calibration only, fresh-seed retest.

Stage-one tests motivated this experiment and are no longer called blind. Old
stage-one outputs and training weights stay unchanged. No v4 mask is substituted.
"""
import sys
sys.dont_write_bytecode=True
from joint_ink_structure_data import ROOT,R2,LOCAL,OFFSETS,hard_fixture,write_record
from evaluate_joint_ink_structure import predict,load_joint,initialize,center_mask,pixel_metrics,reference_center,partial_metric,vectorize,save_output,relation_counts,summarize_relation
from evaluate_simple_landscape_specialist_r2 import single_path_score
from rock_v15_common import read,jsave,digest
from pathlib import Path
import numpy as np,cv2,torch,json
from PIL import Image
STAGE=ROOT/'ink_gate_stage2'

def gated_center(pred,center_threshold,ink_threshold,method='nms'):
    q=pred.copy();q[1]=(pred[1]>=ink_threshold).astype(np.float32)
    return center_mask(q,center_threshold,method)

def calibration_metrics(cache,threshold,ink_threshold=None):
    positives=[];partial=[];background=[];blank=[]
    for r,pred in cache:
        mask=center_mask(pred,threshold['threshold'],threshold['method']) if ink_threshold is None else gated_center(pred,threshold['threshold'],ink_threshold,threshold['method'])
        if r['kind']=='actual_partial':
            value=partial_metric(mask,np.load(r['data']))
            if value.get('f1') is not None:partial.append(value['f1'])
            else:background.append(value['background_fp_fraction'])
        else:
            truth=reference_center(r)
            if truth.any():positives.append(pixel_metrics(mask,truth)['f1'])
            else:blank.append(float(mask.mean()))
    return dict(declared_nonblank_f1=float(np.mean(positives)),actual_partial_f1=float(np.mean(partial)),actual_background_fp_fraction=float(np.mean(background)),declared_blank_fp_fraction=float(np.mean(blank)))

def main():
    if (STAGE/'summary.json').exists():raise FileExistsError('Preserve completed gate calibration experiment')
    STAGE.mkdir(exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(2);model=load_joint();old=initialize().eval();base_selection=read(R2/'selection.json')['selected'];selection=read(ROOT/'selection.json')
    cal=[r for r in read(ROOT/'manifest.json')['records'] if r['split']=='calibration'];new_cache=[];old_cache=[]
    for r in cal:
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));new_cache.append((r,predict(model,rgb)));old_cache.append((r,predict(old,rgb)))
    baseline=calibration_metrics(old_cache,base_selection);trials=[]
    # Safety caps are defined against the old neural model on calibration data.
    caps=dict(actual_background_fp_fraction=baseline['actual_background_fp_fraction']+.0002,declared_blank_fp_fraction=baseline['declared_blank_fp_fraction']+.0002,minimum_declared_nonblank_f1=baseline['declared_nonblank_f1']-.02)
    for center in [.25,.35,.45,.55,.65]:
        for ink in [.12,.25,.4,.55,.7,.85]:
            values=calibration_metrics(new_cache,dict(threshold=center,method='nms'),ink)
            valid=values['actual_background_fp_fraction']<=caps['actual_background_fp_fraction'] and values['declared_blank_fp_fraction']<=caps['declared_blank_fp_fraction'] and values['declared_nonblank_f1']>=caps['minimum_declared_nonblank_f1']
            trials.append(dict(center_threshold=center,ink_threshold=ink,method='nms',**values,passes_safety=bool(valid),score=values['declared_nonblank_f1']+.1*values['actual_partial_f1']))
    eligible=[r for r in trials if r['passes_safety']]
    chosen=max(eligible,key=lambda r:r['score']) if eligible else min(trials,key=lambda r:r['actual_background_fp_fraction']+r['declared_blank_fp_fraction'])
    protocol=dict(selected=chosen,baseline_calibration=baseline,caps=caps,trials=trials,calibration_ids=[r['id'] for r in cal],checkpoint_sha256=digest(ROOT/'joint.pt'),new_training_steps=0,stage1_test_consumed_for_diagnosis=True,test_used_to_select_thresholds=False,actual_test_has_no_full_gold=True)
    jsave(STAGE/'selection.json',protocol);print('INK_GATE_SELECTED',json.dumps(chosen),flush=True)
    kinds=['parallel','crossing','junction','dry_continuation','needles','blank'];fresh=[]
    for i in range(36):
        rgb,t,m,role,rel,rm,meta=hard_fixture(760000+i,kinds[i%6]);meta.update(split='test',kind='fresh_declared_hard',source_group='hard_seed_'+str(760000+i))
        fresh.append(write_record(STAGE/'fresh_test',f'fresh_{i:03}',rgb,t,m,role,rel,rm,meta))
    jsave(STAGE/'fresh_test_manifest.json',dict(records=fresh,seed_interval=[760000,760035],training_or_calibration_used=False,new_seeds_not_new_style=True))
    # Save all fresh inference before reading test references for scoring.
    for i,r in enumerate(fresh):
        rgb=np.asarray(Image.open(r['image']).convert('RGB'))
        for tag,net in [('previous_neural',old),('joint',model)]:
            pred=predict(net,rgb)
            mask=center_mask(pred,base_selection['threshold'],base_selection['method']) if tag=='previous_neural' else gated_center(pred,chosen['center_threshold'],chosen['ink_threshold'])
            paths,meta=vectorize(mask,pred,rgb,selection,use_relations=tag=='joint');save_output(STAGE/'evaluation'/r['id']/tag,rgb,pred,mask,paths,meta)
        if i%12==0:print('INK_GATE_PREDICT',i,36,flush=True)
    jsave(STAGE/'prediction_stage_complete.json',dict(count=36,test_reference_read_in_inference=False))
    scores=[]
    for r in fresh:
        truth=reference_center(r);doc=read(r['reference']);z=np.load(r['data']);row=dict(id=r['id'],case=doc['case'])
        for tag in ['previous_neural','joint']:
            folder=STAGE/'evaluation'/r['id']/tag;mask=np.asarray(Image.open(folder/'raw_center.png'))<128;pred=np.load(folder/'prediction.npz')['prediction'].astype(np.float32);paths=read(folder/'paths.json')['paths']
            values=pixel_metrics(mask,truth) if truth.any() else dict(precision=None,recall=None,f1=None,blank_fp_fraction=float(mask.mean()))
            row[tag]=dict(center=values,single_path_coverage=single_path_score([np.asarray(p,np.float32) for p in paths],doc) if doc['strokes'] else None,relation=relation_counts(pred,z,selection['relation']['threshold']))
        scores.append(row)
    # Existing artwork remains a qualitative regression set. Reuse cached logits.
    for r in read(LOCAL/'baseline_manifest.json')['records']:
        ident=r['id'];rgb=np.asarray(Image.open(Path(r['baseline'])/'input.png').convert('RGB'));source=ROOT/'evaluation'/ident/'joint/prediction.npz';pred=np.load(source)['prediction'].astype(np.float32);mask=gated_center(pred,chosen['center_threshold'],chosen['ink_threshold']);paths,meta=vectorize(mask,pred,rgb,selection)
        meta.update(ink_threshold=chosen['ink_threshold'],center_threshold=chosen['center_threshold'],prediction_source=str(source),prediction_sha256=digest(source));save_output(STAGE/'evaluation'/ident/'joint',rgb,pred,mask,paths,meta,persist_prediction=False)
    metrics={}
    for tag in ['previous_neural','joint']:
        positive=[r[tag]['center'] for r in scores if r[tag]['center']['f1'] is not None];blank=[r[tag]['center']['blank_fp_fraction'] for r in scores if r[tag]['center'].get('blank_fp_fraction') is not None]
        metrics[tag]=dict(center={k:float(np.mean([r[k] for r in positive])) for k in ['precision','recall','f1']},blank_fp_fraction=float(np.mean(blank)),mean_single_path_coverage=float(np.mean([r[tag]['single_path_coverage'] for r in scores if r[tag]['single_path_coverage'] is not None])),relation=summarize_relation([r[tag]['relation'] for r in scores]))
    safety=metrics['joint']['blank_fp_fraction']<=caps['declared_blank_fp_fraction']
    jsave(STAGE/'summary.json',dict(metrics=metrics,scores=scores,selection=chosen,new_training_steps=0,actual_full_image_accuracy=None,default_v4_unchanged=True,new_test_is_procedural_only=True,fresh_blank_safety_pass=bool(safety),adoption='research only; not promoted'))
    print('INK_GATE_EVALUATED',json.dumps(metrics),flush=True)
if __name__=='__main__':main()
