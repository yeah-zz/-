"""Behavior checks for simplification, including real contours that must survive."""
import sys
sys.dont_write_bytecode=True
from mountain_structure_v4_20261101 import *
checks=[]
def check(name,ok):
    checks.append(dict(name=name,passed=bool(ok)));assert ok,name
def paint(rgb,paths):
    for p in paths:cv2.polylines(rgb,[np.rint(p).astype(np.int32)],False,(20,25,25),1)
def row(p,role='long_structure'):return dict(points=np.asarray(p,np.float32).tolist(),role=role)
# Known connected blue mass, two distinct long strokes, one tiny closed ring,
# and a separate fan. No short-texture fixture from the invalid renderer used.
rgb=np.full((256,256,3),245,np.uint8);cv2.rectangle(rgb,(20,95),(160,245),(120,174,180),-1)
long1=np.c_[np.full(81,65),np.arange(125,206)];long2=long1+[6,0]
t=np.linspace(0,2*np.pi,25);ring=np.c_[105+3*np.cos(t),155+3*np.sin(t)]
needles=[np.array([[200,65],[200+25*np.cos(a),65+25*np.sin(a)]]) for a in np.linspace(-3,-.1,12)]
paths=[long1,long2,ring]+needles;paint(rgb,paths);base=v4.raster(rgb.shape[:2],paths);records=[row(long1),row(long2),row(ring,'dense_detail')]+[row(p,'dense_detail') for p in needles]
selected,kept,pending,masks,decisions,events,counts=select(rgb,base,source_records=records)
check('no_invented_pixels',not (selected&~base).any())
check('two_true_parallel_long_strokes_preserved',bool(selected[150,65]) and bool(selected[150,71]))
check('parallel_strokes_not_joined_across_gap',not selected[150,66:71].any())
tree=v4.raster(base.shape,needles);check('separate_tree_fan_preserved',not (tree&~selected).any())
circle=v4.raster(base.shape,[ring]);check('small_internal_closed_ring_deferred',not (selected&circle).any())
check('known_main_strokes_not_deleted_for_lower_line_count',not (v4.raster(base.shape,[long1,long2])&~selected).any())
# A genuinely closed outer rock contour is retained, even though a cycle.
rgb=np.full((256,256,3),245,np.uint8);cv2.ellipse(rgb,(128,175),(72,62),0,0,360,(110,172,185),-1)
t=np.linspace(0,2*np.pi,250);contour=np.c_[128+72*np.cos(t),175+62*np.sin(t)];paint(rgb,[contour]);base=v4.raster(rgb.shape[:2],[contour]);selected,*_=select(rgb,base,source_records=[row(contour)])
check('legitimate_closed_outer_contour_preserved',np.array_equal(selected,base))
# No blue pigment means abstain from mountain simplification.
rgb=np.full((256,256,3),245,np.uint8);paint(rgb,[long1,ring]);base=v4.raster(rgb.shape[:2],[long1,ring]);selected,*_=select(rgb,base,source_records=[row(long1),row(ring)])
check('monochrome_without_applicable_color_routes_keeps_v4',np.array_equal(selected,base))
jsave(ROOT/'behavior_checks.json',dict(checks=checks,all_passed=True,new_neural_training_steps=0))
print('MOUNTAIN_STRUCTURE_CHECKS',len(checks))
