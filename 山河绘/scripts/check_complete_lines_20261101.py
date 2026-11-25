# -*- coding: utf-8 -*-
"""Meaningful image-evidence and stroke-neighbor behavior checks."""
import sys
sys.dont_write_bytecode=True
from mountain_complete_lines_20261101 import *
checks=[]
def ck(name,ok):checks.append(dict(name=name,passed=bool(ok)));assert ok,name

def fixture(points,width=3,faint=False):
    rgb=np.full((496,496,3),240,np.uint8);p=np.rint(points).astype(np.int32);cv2.polylines(rgb,[p],False,(222,222,222) if faint else (35,35,35),width,cv2.LINE_AA);field=evidence_field(rgb);masks=dict(area=np.ones((496,496),bool),protect=np.zeros((496,496),bool),mass=np.ones((496,496),bool));return rgb,field,masks

def main():
    # A semicircle is a true curved centerline; no endpoint-span straightness
    # requirement may reject it merely for bending.
    t=np.linspace(-math.pi*.8,math.pi*.8,130);p=np.column_stack([180+50*np.cos(t),180+50*np.sin(t)]);rgb,field,masks=fixture(p);f=arc_profile(rgb,p,field);row=dict(key='v4_curve',features=f);ck('large_true_curve_eligible',geometry_gate(row));ck('curve_is_not_straight',np.linalg.norm(p[-1]-p[0])/v4.plen(p)<.65)
    p=np.column_stack([np.arange(70,180),np.full(110,90)]);rgb,field,masks=fixture(p,3,True);f=arc_profile(rgb,p,field);ck('faint_line_detected',float(field['ridge'][87:94,80:170].sum())>60);ck('faint_profile_eligible',geometry_gate(dict(key='ridge_faint',features=f)))
    # A source curve remains painted while the detector is made to miss its
    # middle: this verifies recovery, rather than drawing a line across blank.
    t=np.linspace(-.6,.6,100);p=np.rint(np.column_stack([220+50*np.sin(t),140+50*(1-np.cos(t))])).astype(np.float32);rgb,field,masks=fixture(p);allmask=v4.raster((496,496),[p]);broken=allmask.copy();broken[:,218:223]=False;res,added,events=bridge_curves(rgb,broken,masks,field);ck('curved_source_gap_joined',added.any() and any(e['accepted'] for e in events));ck('gap_follows_source_evidence',not (added&~cv2.dilate(field['weak'].astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)).any())
    blank=rgb.copy();blank[:,218:223]=240;bf=evidence_field(blank);_,a,_=bridge_curves(blank,broken,masks,bf);ck('blank_gap_not_bridged',not a.any())
    protected={k:v.copy() for k,v in masks.items()};protected['protect'][:,218:223]=True;protected['area']&=~protected['protect'];_,a,_=bridge_curves(rgb,broken,protected,field);ck('protected_detail_unchanged',not a.any())
    p1=np.array([[80,80],[80,150]],np.float32);p2=p1+[4,0];rgb,field,masks=fixture(p1);cv2.polylines(rgb,[p2.astype(np.int32)],False,(35,35,35),3);field=evidence_field(rgb);parallel=v4.raster((496,496),[p1,p2]);_,a,_=bridge_curves(rgb,parallel,masks,field);ck('near_parallel_not_joined',not a.any())
    ck('complete_stage_has_no_count_budget',POLICY['complete_count_limit'] is None and CFG['complete_candidate_count_limit'] is None)
    jsave(ROOT/'behavior_checks.json',dict(checks=checks,count=len(checks),all_passed=True));print('COMPLETE_BEHAVIOR_CHECKS',len(checks),flush=True)
if __name__=='__main__':main()
