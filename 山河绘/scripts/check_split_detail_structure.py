# -*- coding: utf-8 -*-
"""Behavior checks for evidence-supported joining and native pixel geometry."""
from split_detail_structure import *

def chain(a,b,n=30):return np.linspace(a,b,n).astype(np.float32)

def check():
    report={};shape=(96,96);dense=np.zeros(shape,bool);ink=np.zeros(shape,bool)
    white=np.full((*shape,3),255,np.uint8)
    paths=[chain([10,32],[35,32]),chain([40,32],[78,32])]
    rgb=white.copy();cv2.line(rgb,(10,32),(78,32),(20,20,20),2)
    out,events,_=stitch(paths,rgb,ink,dense,CFG)
    assert len(out)==1 and len(events)==1 and events[0]['gap']<=CFG['max_bridge']
    report['supported_short_gap_joined']=True
    out,events,_=stitch(paths,white,ink,dense,CFG)
    assert not events and len(out)==2
    report['blank_gap_not_invented']=True
    parallel=[chain([10,32],[35,32]),chain([40,36],[78,36])]
    out,events,_=stitch(parallel,rgb,ink,dense,CFG)
    assert not events and len(out)==2
    report['adjacent_parallel_strokes_not_merged']=True
    crossing=paths+[chain([37,15],[37,60])]
    out,events,_=stitch(crossing,rgb,ink,dense,CFG)
    assert not events
    report['bridge_crossing_unrelated_branch_rejected']=True
    ambiguous=[chain([10,32],[35,32]),chain([40,31],[78,25]),chain([40,33],[78,39])]
    amb=rgb.copy();cv2.line(amb,(35,32),(40,31),(20,20,20),2);cv2.line(amb,(35,32),(40,33),(20,20,20),2)
    out,events,_=stitch(ambiguous,amb,ink,dense,CFG)
    assert not events
    report['ambiguous_fork_not_forced']=True
    out,events,_=stitch(paths,rgb,ink,np.ones(shape,bool),CFG)
    assert not events
    report['dense_detail_regions_not_force_joined']=True
    sk=np.zeros(shape,np.uint8);cv2.line(sk,(10,20),(70,20),1,1);cv2.line(sk,(40,20),(40,60),1,1);cv2.circle(sk,(60,70),9,1,1)
    sk=skeletonize(sk>0);pp,neighbors=pixel_chains(sk);covered=raster(shape,pp)
    assert np.array_equal(covered,sk)
    report['pixel_graph_preserves_t_junction_and_closed_curve']=True
    gray=np.full((*shape,3),245,np.uint8);cv2.line(gray,(26,12),(26,80),(20,20,20),5);cv2.line(gray,(36,12),(36,80),(20,20,20),3)
    ridge,response=ink_ridges(gray)
    assert ridge[20:70,25:28].sum()>35 and ridge[20:70,35:38].sum()>35
    assert ridge[20:70,30:33].sum()<5
    report['two_ink_bands_have_two_centers_not_a_middle_join']=True
    pure,_=ink_ridges(white)
    assert not pure.any()
    report['uniform_blank_has_no_ridge']=True
    source=[chain([10,32],[78,32],69)]
    fragments=[chain([10,32],[35,32]),chain([40,32],[78,32])]
    out,n=consolidate_source_lines(source,fragments,dense)
    assert n==1 and len(out)==1 and np.array_equal(out[0],source[0])
    report['intact_source_replaces_overlapping_fragments_whole']=True
    source=[chain([10,30],[78,30],69),chain([10,33],[78,33],69)]
    out,n=consolidate_source_lines(source,[],dense)
    assert n==2 and len(out)==2
    report['intact_close_parallel_source_lines_kept_separately']=True
    return report

if __name__=='__main__':
    result=check();jsave(LAB/'split_detail_structure_20261028'/'v4'/'behavior_checks.json',dict(checks=result,all_passed=all(result.values()),reference_accuracy_not_measured=True))
    print('BEHAVIOR_CHECKS',json.dumps(result),flush=True)
