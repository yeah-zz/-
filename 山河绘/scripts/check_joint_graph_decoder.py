import sys
sys.dont_write_bytecode=True
from evaluate_joint_ink_structure import join_supported_endpoints,v4,ROOT
from rock_v15_common import jsave
import numpy as np,cv2,json
checks=[]
def run(kind):
    mask=np.zeros((128,128),np.uint8);cv2.line(mask,(20,70),(60,70),1,1);cv2.line(mask,(63,70),(100,70),1,1)
    rgb=np.full((128,128,3),245,np.uint8);cv2.line(rgb,(20,70),(100,70),(30,30,30),1)
    pred=np.full((18,128,128),.99,np.float32)
    if kind=='blank_gap':rgb[70,61:63]=245
    if kind=='low_affinity':pred[10:]=.01
    if kind=='parallel':mask[:]=0;cv2.line(mask,(20,70),(100,70),1,1);cv2.line(mask,(20,73),(100,73),1,1)
    if kind=='crossing':cv2.line(mask,(61,60),(61,80),1,1)
    _,neighbors=v4.pixel_chains(mask>0);_,joins=join_supported_endpoints(neighbors,pred,rgb,.5)
    json.dumps(joins)
    return len(joins)
for kind,expected in [('supported',1),('blank_gap',0),('low_affinity',0),('parallel',0),('crossing',0)]:
    actual=run(kind);checks.append(dict(case=kind,joins=actual,expected=expected,passed=actual==expected));assert actual==expected,(kind,actual)
jsave(ROOT/'graph_decoder_checks.json',dict(checks=checks));print('JOINT_GRAPH_CHECKS',len(checks))
