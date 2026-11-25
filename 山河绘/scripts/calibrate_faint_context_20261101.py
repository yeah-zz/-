from evaluate_faint_joint_20261101 import *
from faint_context_routing_20261101 import mountain_context
def main():
    previous=read(ROOT/'threshold_calibration.json');records=[r for r in read(ROOT/'audited_reference_manifest.json')['records'] if r['split']=='calibration'];cache={};candidates=[]
    for r in records:
        image=r['image']
        if image not in cache:cache[image]=mountain_context(np.asarray(Image.open(BASE/'originals'/(image+'.png')).convert('RGB')))
    for item in previous['all_candidates']:
        paper=edge=edgen=0;coverage=[];oldcov=[]
        for r in records:
            x,y,xx,yy=r['box'];rgb=np.asarray(Image.open(r['path']).convert('RGB'));o=np.load(ROOT/'local_predictions_r2'/r['id']/'fields.npz')['fields'];mask,_,_=trace(rgb,o,item['config'],cache[r['image']][y:yy,x:xx]);m=partial_metrics(mask,r);paper+=m['line_pixels_in_checked_negative_boxes'];edge+=m['color_boundary_ref_hits_3px'];edgen+=m['color_boundary_reference_pixels'];old=np.asarray(Image.open(PRIOR/'outputs_r3'/r['image']/'lines.png'))<128;old=old[y:yy,x:xx];u=partial_metrics(mask|old,r);b=partial_metrics(old,r)
            if u['partial_reference_coverage_3px'] is not None:coverage.append(u['partial_reference_coverage_3px']);oldcov.append(b['partial_reference_coverage_3px'])
        feasible=paper<=3 and edge/max(edgen,1)<=.10 and item['synthetic_tolerant_precision']>=.80;candidates.append(dict(**{k:v for k,v in item.items() if k!='feasible'},real_checked_negative_pixels_after_context=paper,real_boundary_hit_fraction_after_context=edge/max(edgen,1),potential_union_partial_coverage=float(np.mean(coverage)),previous_partial_coverage=float(np.mean(oldcov)),feasible_after_context=feasible))
    feasible=[r for r in candidates if r['feasible_after_context']];chosen=max(feasible or candidates,key=lambda r:r['synthetic_tolerant_f1']+.1*r['potential_union_partial_coverage']-.005*r['real_checked_negative_pixels_after_context']);jsave(ROOT/'context_calibration.json',dict(selected=chosen,feasible_found=bool(feasible),all_candidates=candidates,context_rule='Automatic whole-image blue-green mountain/material context, no manual inference mask',raw_model_negative_pixel_failure_retained=True,synthetic_metrics_are_before_real_image_context=True,potential_union_is_not_a_final_repair_or_accuracy=True,test_labels_used=False));print('FAINT_CONTEXT_CALIBRATION',chosen,flush=True)
if __name__=='__main__':main()
