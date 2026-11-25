from evaluate_faint_joint_20261101 import *
import itertools
def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);model=FaintNet().cuda();artifact=torch.load(ROOT/'fit_r2/model.pt',map_location='cpu',weights_only=False);model.load_state_dict(artifact['model']);manifest=read(ROOT/'paired_manifest_r2.json');synthetic=[r for r in manifest['synthetic'] if r['split']=='calibration'][:16];folder=ROOT/'calibration';folder.mkdir(exist_ok=True);fixtures=[]
    for r in synthetic:
        z=np.load(r['file']);rgb=z['rgb'];p=folder/(str(r['seed'])+'_fields.npz')
        if p.exists():o=np.load(p)['fields']
        else:o=predict(rgb,model);np.savez_compressed(p,fields=o)
        fixtures.append((r,rgb,z['target'].astype(np.float32),o))
    real=[r for r in read(ROOT/'audited_reference_manifest.json')['records'] if r['split']=='calibration'];cache=[(r,np.asarray(Image.open(r['path']).convert('RGB')),np.load(ROOT/'local_predictions_r2'/r['id']/'fields.npz')['fields']) for r in real];scores=[]
    for weak,strong,edge in itertools.product([.25,.35,.45],[.55,.70,.82],[.4,.7]):
        cfg=dict(CFG,weak=weak,strong=strong,edge=edge);pn=pt=gn=gt=0;edge_total=edge_hit=0;cover=[];paper=0;real_edge=real_edgen=0
        for r,rgb,t,o in fixtures:
            mask,_,_=trace(rgb,o,cfg);ref=t[0]>.4;dr=distance_transform_edt(~ref);dp=distance_transform_edt(~mask);pn+=int(mask.sum());pt+=int((mask&(dr<=2)).sum());gn+=int(ref.sum());gt+=int((ref&(dp<=2)).sum());edge_total+=int((t[4]>.5).sum());edge_hit+=int(((t[4]>.5)&(dp<=2)).sum())
        for r,rgb,o in cache:
            mask,_,_=trace(rgb,o,cfg);m=partial_metrics(mask,r);paper+=m['line_pixels_in_checked_negative_boxes'];real_edge+=m['color_boundary_ref_hits_3px'];real_edgen+=m['color_boundary_reference_pixels']
            if m['partial_reference_coverage_3px'] is not None:cover.append(m['partial_reference_coverage_3px'])
        precision=pt/max(pn,1);recall=gt/max(gn,1);f1=2*precision*recall/max(precision+recall,1e-8);feasible=paper<=3 and real_edge/max(real_edgen,1)<=.10 and precision>=.80;scores.append(dict(config=cfg,synthetic_tolerant_precision=precision,synthetic_tolerant_recall=recall,synthetic_tolerant_f1=f1,synthetic_boundary_false_hit_fraction=edge_hit/max(edge_total,1),real_checked_negative_pixels=paper,real_partial_boundary_hit_fraction=real_edge/max(real_edgen,1),real_photometric_partial_coverage=float(np.mean(cover)),feasible=feasible))
    feasible=[r for r in scores if r['feasible']];chosen=max(feasible or scores,key=lambda r:r['synthetic_tolerant_f1']-.3*r['real_partial_boundary_hit_fraction']-.005*r['real_checked_negative_pixels']);jsave(ROOT/'threshold_calibration.json',dict(selected=chosen,feasible_found=bool(feasible),all_candidates=scores,calibration_synthetic_seeds=[r['seed'] for r in synthetic],calibration_real_parents=['T03','T12'],test_labels_used=False,real_partial_coverage_not_accuracy=True,model_sha256=digest(ROOT/'fit_r2/model.pt')));print('FAINT_CALIBRATION',chosen,flush=True)
if __name__=='__main__':main()
