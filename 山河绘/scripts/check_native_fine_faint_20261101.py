from refine_native_fine_faint_20261101 import *
from run_stability_brush_20261101 import verify_lock
checks=[]
def check(name,v):
    checks.append(dict(name=name,passed=bool(v)))
    assert v,name

def main():
    verify_lock();check('accepted_18_baseline_files_unchanged',True)
    # Does native-pixel evidence preserve known narrow needles, while blank grain stays empty?
    rng=np.random.default_rng(110105);n=256;paper=np.full((n,n,3),240,np.float32);paper+=rng.normal(0,.35,(n,n,1));rgb=np.clip(paper,0,255).astype(np.uint8);truth=np.zeros((n,n),np.uint8)
    for theta in np.linspace(-2.85,-.3,17):
        a=np.array([128,145]);b=a+np.array([math.cos(theta),math.sin(theta)])*45;cv2.line(rgb,tuple(a),tuple(np.rint(b).astype(int)),(95,100,90),1,cv2.LINE_AA);cv2.line(truth,tuple(a),tuple(np.rint(b).astype(int)),1,1)
    f=field(rgb);distance=distance_transform_edt(~f['fine']);yy,xx=np.mgrid[:n,:n];visible=(truth>0)&(((xx-128)**2+(yy-145)**2)>18**2);coverage=float((distance[visible]<=1.5).mean());print('VISIBLE_SYNTHETIC_NEEDLE_COVERAGE',coverage,flush=True);check('narrow_fan_visible_tips_known_geometry_coverage',coverage>.80)
    blank=np.clip(paper,0,255).astype(np.uint8);bf=field(blank);check('mild_blank_grain_not_fine_ink',bf['fine'].sum()==0)
    gray=np.ones((50,50),np.float32)*.9;gray[:,18:26]=.25;a=np.array([[19.,10.],[19.,20.]]);b=np.array([[24.,10.],[24.,20.]]);check('same_broad_ink_band_can_match',same_band(gray,a,b).all());gray[:,20:24]=.9;check('brighter_gap_keeps_parallel_strokes_separate',not same_band(gray,a,b).any())
    # Bridging must not cross unmarked blank paper, even with aligned endpoints.
    m=np.zeros((80,100),bool);m[40,15:40]=True;m[40,46:75]=True;contrast=np.zeros(m.shape,np.float32);contrast[m]=.1;score=contrast.copy();ff=dict(ridge=m,contrast=contrast,threshold=np.full(m.shape,.01,np.float32),score=score);out,events=bridge_ridges(ff,np.ones(m.shape,bool));check('aligned_gap_in_blank_paper_stays_open',len(events)==0 and np.array_equal(out,m))
    contrast[40,40:46]=.1;score[40,40:46]=.1;out,events=bridge_ridges(ff,np.ones(m.shape,bool));check('same_path_gap_requires_source_evidence',len(events)==1 and out[40,40:46].all())
    # With no candidate evidence or predicted fan, refinement is exactly conservative.
    p=np.array([[10.,10.],[50.,50.]],np.float32);rgb=np.full((128,128,3),240,np.uint8);f=field(rgb);base=graph.raster((128,128),[p]);result,masks,d=refine(rgb,[p],np.zeros((128,128),np.float32),f);check('no_source_candidate_keeps_baseline_exact',np.array_equal(result,base))
    rows=[]
    for folder in sorted(OUTPUTS.glob('T*')):
        if not (folder/'entry.json').exists():continue
        e=read(folder/'entry.json');doc=read(folder/'paths.json');dec=read(folder/'decisions.json');result=np.asarray(Image.open(folder/'lines.png'))<128;export=np.asarray(Image.open(folder/'exported_lines.png'))<128;isolated=np.asarray(Image.open(folder/'isolated_evidence.png'))<128;fan=np.asarray(Image.open(folder/'fan.png'))<128;base=np.asarray(Image.open(folder/'baseline.png'))<128;p=[np.array(r['points']) for r in doc['paths']];check(folder.name+'_raster_export_identity',np.array_equal(graph.raster(result.shape,p),export));check(folder.name+'_isolated_evidence_separate',np.array_equal(export|isolated,result));check(folder.name+'_robot_disabled',not doc['ready_for_robot'] and not doc['actually_drawn'] and doc['physical_z'] is None);check(folder.name+'_source_and_model_hashes',digest(e['source'])==e['source_sha256'] and digest(ROOT/'router/router.pt')==e['router_model_sha256']);check(folder.name+'_faint_candidates_not_chopped',all(result[np.rint(np.array(r['points'])[:,1]).astype(int),np.rint(np.array(r['points'])[:,0]).astype(int)].all() for r in dec['faint']))
        # Retained old pixels outside the learned fan may only be removed as complete matching old paths.
        old=read(e['baseline_paths']);h,w=result.shape
        with Image.open(Path(e['baseline_paths']).parent/'input.png') as im:bw,bh=im.size
        oldpaths=[np.array(r['points'])*[w/bw,h/bh] for r in old['paths']];removedmask=graph.raster(result.shape,[oldpaths[i] for i in dec['replaced_old_path_ids']]);check(folder.name+'_outside_fan_deletions_explained',(base&~result&~fan&~removedmask).sum()==0)
        seen=set();duplicates=0
        for path in p:
            for a,b in zip(path[:-1],path[1:]):
                key=tuple(sorted((tuple(a),tuple(b))));duplicates+=key in seen;seen.add(key)
        check(folder.name+'_no_duplicate_export_edges',duplicates==0);rows.append(dict(id=folder.name,export_paths=len(p),isolated_pixels=int(isolated.sum()),faint_candidates=len(dec['faint'])))
    jsave(ROOT/'behavior_checks.json',dict(all_passed=True,checks=checks,count=len(checks),known_synthetic_fan_coverage_at_1_5px=coverage,observed_output_records=rows,not_an_accuracy_or_robot_validation=True));print('NATIVE_BEHAVIOR_CHECKS',len(checks),flush=True)
if __name__=='__main__':main()
