from extract_faint_joint_20261101 import *
import xml.etree.ElementTree as ET
checks=[]
def check(name,v):checks.append(dict(name=name,passed=bool(v)));assert v,name
def main():
    verify_round();check('accepted_and_round_weights_code_unchanged',True);paired=read(ROOT/'paired_manifest_r2.json');parents={s:{r['parent'] for r in paired['real'] if r['split']==s} for s in ['train','calibration','old_regression_test']};check('whole_image_parents_do_not_cross_splits',all(not parents[a]&parents[b] for a,b in [('train','calibration'),('train','old_regression_test'),('calibration','old_regression_test')]))
    seeds={s:{r['seed'] for r in paired['synthetic'] if r['split']==s} for s in ['train','calibration','test']};check('synthetic_seeds_do_not_cross_splits',all(not seeds[a]&seeds[b] for a,b in [('train','calibration'),('train','test'),('calibration','test')]))
    for r in paired['synthetic']:
        check(str(r['seed'])+'_data_hash',digest(r['file'])==r['sha256']);z=np.load(r['file']);check(str(r['seed'])+'_ink_target_binary',np.isin(z['target'][1],[0,1]).all())
    for r in paired['real']:
        z=np.load(r['file']);check(r['id']+'_uncertain_real_centers_not_precise_positive_gold',not ((z['target'][0]>.5)&(z['known'][0]>0)).any());check(r['id']+'_partial_data_hash',digest(r['file'])==r['sha256'])
    # Complete candidate replacement, no novelty clipping; protected areas are immutable.
    shape=(128,128);p1=np.c_[np.arange(10,32),np.full(22,60)];p2=np.c_[np.arange(55,85),np.full(30,60)];base=graph.raster(shape,[p1,p2]);base[100,100]=True;rgb=np.full((*shape,3),240,np.uint8);rgb[60,10:85]=70;protect=np.zeros(shape,bool);candidate=dict(points=np.c_[np.arange(10,85),np.full(75,60)].tolist(),length=74.,mean_score=.95,direction_alignment=1.,mean_color_boundary_probability=0.)
    result,dec=fuse(rgb,base,protect,[candidate]);check('two_old_fragments_replaced_by_one_complete_candidate',len(dec['accepted'])==1 and len(dec['replaced_old_path_ids'])==2 and result[60,10:85].all());check('isolated_original_evidence_preserved',result[100,100]);protect[60,50]=True;result,dec=fuse(rgb,base,protect,[candidate]);check('candidate_touching_protected_detail_deferred_whole',not dec['accepted'] and np.array_equal(result,base))
    for r in read(ROOT/'audited_reference_manifest.json')['records']:
        folder=ROOT/'local_repair'/r['id'];old=np.asarray(Image.open(folder/'previous.png'))<128;new=np.asarray(Image.open(folder/'repaired.png'))<128;protected=np.asarray(Image.open(folder/'protected.png'))<128;d=read(folder/'decisions.json');check(r['id']+'_tree_protection_exact',np.array_equal(new[protected],old[protected]));check(r['id']+'_complete_new_candidates_retained',all(new[np.rint(np.array(a['points'])[:,1]).astype(int),np.rint(np.array(a['points'])[:,0]).astype(int)].all() for a in d['accepted']));oldp,_=graph.pixel_chains(old);removed=graph.raster(old.shape,[oldp[i] for i in d['replaced_old_path_ids']]);check(r['id']+'_deletions_only_complete_matched_old_fragments',not (old&~new&~removed).any())
    fresh=[]
    for ident in ['N01','N02','N03']:
        folder=ROOT/'fresh_outputs'/ident
        if not (folder/'entry.json').exists():continue
        e=read(folder/'entry.json');doc=read(folder/'paths.json');new=np.asarray(Image.open(folder/'repaired.png'))<128;old=np.asarray(Image.open(folder/'previous.png'))<128;protected=np.asarray(Image.open(folder/'protected.png'))<128;isolated=np.asarray(Image.open(folder/'isolated_evidence.png'))<128;paths=[np.asarray(p['points']) for p in doc['paths']];check(ident+'_source_and_model_hashes',digest(e['image'])==e['source_sha256'] and digest(ROOT/'fit_r2/model.pt')==e['model_sha256']);check(ident+'_protected_exact',np.array_equal(old[protected],new[protected]));check(ident+'_export_matches_except_explicit_isolated_evidence',np.array_equal(graph.raster(new.shape,paths)|isolated,new));check(ident+'_svg_path_count',len(ET.parse(folder/'lines.svg').findall('.//{http://www.w3.org/2000/svg}polyline'))==len(paths));check(ident+'_robot_not_enabled',not doc['ready_for_robot'] and not doc['actually_drawn'] and doc['physical_z'] is None and doc['physical_speed'] is None)
        seen=set();duplicates=0
        for p in paths:
            for a,b in zip(p[:-1],p[1:]):key=tuple(sorted((tuple(a),tuple(b))));duplicates+=key in seen;seen.add(key)
        check(ident+'_no_duplicate_export_edges',duplicates==0);fresh.append(ident)
    train=read(ROOT/'fit_r2/training_config.json');check('r2_training_source_hash_matches',digest(Path(__file__).parent/'train_faint_joint_20261101.py')==train['source_code_sha256']);check('r2_initial_checkpoint_unchanged',digest(ROOT/'fit/model.pt')==train['init_sha256']);verify_round();jsave(ROOT/'integrity_checks.json',dict(all_passed=True,count=len(checks),checks=checks,fresh_outputs_checked=fresh,not_real_accuracy_or_hardware_validation=True));print('FAINT_INTEGRITY_CHECKS',len(checks),fresh,flush=True)
if __name__=='__main__':main()
