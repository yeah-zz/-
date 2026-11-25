"""Behavior and synthetic checks; not real-artwork accuracy estimates."""
from stroke_completeness_v5 import *
import stroke_completeness_v5 as implementation
import check_split_detail_structure as prior

def check():
    prior.stitch=implementation.stitch;prior.CFG=implementation.CFG;result=prior.check()
    shape=(100,100);rgb=np.full((*shape,3),245,np.uint8)
    ink=np.zeros(shape,bool);ink[10:90,45:55]=True;ink[48:50,49:51]=False
    original=ink.copy();coherence=np.ones(shape,np.float32);dense=np.zeros(shape,bool)
    fixed,events=fill_tiny_ink_voids(ink,rgb,dense,coherence)
    assert len(events)==1 and fixed[48:50,49:51].all() and np.array_equal(ink,original)
    result['tiny_void_corrected_only_in_derived_topology_not_original_ink']=True
    fixed,events=fill_tiny_ink_voids(ink,rgb,np.ones(shape,bool),coherence)
    assert not events and np.array_equal(fixed,ink)
    result['dense_detail_voids_preserved']=True
    ring=np.zeros(shape,np.uint8);cv2.rectangle(ring,(35,35),(45,45),1,1)
    fixed,events=fill_tiny_ink_voids(ring.astype(bool),rgb,dense,coherence)
    assert not events and np.array_equal(fixed,ring>0)
    result['genuine_closed_thin_contour_not_filled']=True
    fixed,events=fill_tiny_ink_voids(ink,rgb,dense,np.zeros(shape,np.float32))
    assert not events
    result['unoriented_crossing_void_not_automatically_filled']=True
    white=np.full((*shape,3),250,np.uint8);sk,_=weak_ridges(white);assert not sk.any()
    result['weak_recovery_does_not_create_uniform_background_lines']=True
    original_bridge=implementation.bridge_curve
    def outside_bridge(a,b,ta,tb,gray):
        q,c=original_bridge(a,b,ta,tb,gray);q[1:-1,0]=gray.shape[1];return q,np.ones(len(q),np.float32)
    implementation.bridge_curve=outside_bridge
    try:
        edge_paths=[prior.chain([99,10],[99,35]),prior.chain([99,40],[99,78])]
        output,events,_=implementation.stitch(edge_paths,white,np.zeros(shape,bool),np.zeros(shape,bool),CFG)
        assert not events and len(output)==2
    finally:implementation.bridge_curve=original_bridge
    result['out_of_image_bridge_abstains_without_array_error']=True
    # Measure known synthetic faint-line response over fixed contrast conditions.
    responses=[]
    for delta in [4,6,10,30]:
        test=white.copy();cv2.line(test,(45,15),(45,85),(250-delta,)*3,2,cv2.LINE_AA)
        a,_=ink_ridges(test);b,_=weak_ridges(test)
        responses.append(dict(gray_delta=delta,v4_center_pixels=int(a[20:80,44:47].sum()),v5_weak_candidate_center_pixels=int(b[20:80,44:47].sum())))
    assert sum(r['v5_weak_candidate_center_pixels'] for r in responses)>=sum(r['v4_center_pixels'] for r in responses)
    assert any(r['v5_weak_candidate_center_pixels']>r['v4_center_pixels'] for r in responses)
    result['weak_ridge_candidates_recover_known_faint_line_response']=True
    jsave(ROOT/'behavior_checks.json',dict(checks=result,all_passed=all(result.values()),synthetic_faint_responses=responses,synthetic_response_not_final_real_accuracy=True))
    print('V5_BEHAVIOR_CHECKS',json.dumps(dict(checks=result,responses=responses)),flush=True)

if __name__=='__main__':check()
