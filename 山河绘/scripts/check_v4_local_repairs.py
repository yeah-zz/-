"""Meaningful invariants for the small-image evidence-recovery stage."""
import sys,json
sys.dont_write_bytecode=True
import numpy as np,cv2
from PIL import Image
from audit_v4_local_repairs import OUT,R2,dump,sha
import v4_local_repair_candidates as repair
import split_detail_v4_frozen as v4

def main():
    checks=[]
    def check(name,ok,**info):
        checks.append(dict(name=name,passed=bool(ok),**info));assert ok,name
    blank=np.full((496,496,3),245,np.uint8);empty=np.zeros((496,496),bool)
    a,_,_=repair.recover(blank,[],empty);check('blank_image_adds_no_lines',not a)
    raw=empty.copy();cv2.line(raw.view(np.uint8),(70,100),(85,120),1,1);cv2.line(raw.view(np.uint8),(74,100),(89,120),1,1)
    rgb=blank.copy();rgb[raw]=60
    a,_,_=repair.recover(rgb,[],raw)
    check('parallel_lines_not_connected',len(a)==2 and all(np.linalg.norm(np.asarray(p['points'])[0]-np.asarray(p['points'])[-1])>10 for p in a),paths=len(a))
    raw=empty.copy();cv2.circle(raw.view(np.uint8),(200,200),4,1,1);rgb=blank.copy();rgb[raw]=50
    a,_,_=repair.recover(rgb,[],raw);check('closed_ring_not_added_as_new_centerline',not a)
    raw=empty.copy();cv2.line(raw.view(np.uint8),(100,100),(115,115),1,1)
    a,_,_=repair.recover(blank,[],raw);check('unsupported_skeleton_rejected',not a)
    u=np.asarray([[100,100],[100,140],[140,140],[140,100]],np.float32)
    left=np.asarray([[100,80],[100,100]],np.float32);right=left+np.array([40,0],np.float32)
    original=[dict(points=p.tolist()) for p in [u,left,right]]
    top=np.asarray([[100,100],[140,100]],np.float32)
    old=v4.raster((496,496),[u,left,right]);raw=old|v4.raster((496,496),[top]);rgb=blank.copy();rgb[raw]=50
    ungated,_,_=repair.recover(rgb,original,raw,dict(repair.CFG,forbid_new_pixel_graph_cycles=False))
    gated,_,decisions=repair.recover(rgb,original,raw)
    check('supported_open_segment_would_close_ink_cycle_without_guard',bool(ungated))
    check('cycle_guard_defers_whole_candidate',not gated and any(d.get('reason')=='would_create_new_pixel_graph_cycle_keep_pending' for d in decisions))
    source=R2/'native_outputs/N02/v4';doc=json.loads((source/'paths.json').read_text(encoding='utf-8'))
    rgb=np.asarray(Image.open(source/'input.png').convert('RGB'));raw=np.asarray(Image.open(source/'raw_skeleton.png').convert('L'))<128
    a,_,why=repair.recover(rgb,doc['paths'],raw);check('accepted_native_N02_kept_exactly',not a and why[0]['reason']=='outside_evaluated_size_range_keep_v4')
    manifest=json.loads((OUT/'baseline_manifest.json').read_text(encoding='utf-8'))
    for r in manifest['records']:
        folder=OUT/'candidate'/r['id']
        if not folder.exists():continue
        base=json.loads((Path(r['baseline'])/'paths.json').read_text(encoding='utf-8'))['paths']
        new=json.loads((folder/'paths.json').read_text(encoding='utf-8'))['paths']
        check(r['id']+'_baseline_path_prefix_preserved',new[:len(base)]==base)
        raw=np.asarray(Image.open(Path(r['baseline'])/'raw_skeleton.png').convert('L'))<128
        added=v4.raster(raw.shape,[np.asarray(p['points'],np.float32) for p in new[len(base):]])
        check(r['id']+'_new_geometry_only_on_original_source_skeleton',not (added&~raw).any())
        before=v4.raster(raw.shape,[np.asarray(p['points'],np.float32) for p in base])
        check(r['id']+'_new_pixel_graph_cycles_do_not_increase',repair.graph_cycles(before|added)<=repair.graph_cycles(before))
        for name,expected in r['hashes'].items():
            check(r['id']+'_frozen_'+name,sha(Path(r['baseline'])/name)==expected)
    dump(OUT/'behavior_checks.json',dict(checks=checks,new_training_steps=0))
    print(json.dumps(dict(passed=len(checks),output=str(OUT/'behavior_checks.json'))))
from pathlib import Path
if __name__=='__main__':main()
