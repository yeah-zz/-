# -*- coding: utf-8 -*-
"""Single-image research inference: no labels or reference trajectories supplied."""
from v24_train import ROOT,load_model,predict
from v24_eval import decode
import v21_stroke_learning as v21
from rock_v15_common import read,jsave,digest,FONT,SMALL
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np,cv2,argparse,sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--grayscale',action='store_true');a=p.parse_args();source=Path(a.input);out=Path(a.output)
    if not out.is_absolute() or out.drive.upper() not in ['E:','F:']:raise ValueError('Use an absolute E:/F: output path')
    rgb=np.asarray(Image.open(source).convert('RGB'))
    if a.grayscale:rgb=np.repeat(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)[:,:,None],3,2)
    if not v21.v16.monochrome_route(rgb):raise ValueError('This candidate is for monochrome rock research; color landscapes are not validated')
    decision=read(ROOT/'joint'/'calibration.json');cfg=decision['selected']['config'];model=load_model('joint');pred=predict(model,rgb);paths,sk,info=decode(pred,cfg);out.mkdir(parents=True,exist_ok=True);jsave(out/'trajectory.json',dict(paths=[p.round(3).tolist() for p in paths],coordinates='pixels',physical_z=None,ready_for_robot=False,actually_drawn=False,reference_labels_used_at_inference=False,decoder_config=cfg,info=info))
    for channel,name in [(0,'center_probability'),(1,'ink_probability'),(5,'turn_probability')]:Image.fromarray(np.uint8(255*(1-pred[channel]))).save(out/(name+'.png'))
    board=Image.new('RGB',(1600,440),'#f5f3ec');d=ImageDraw.Draw(board)
    for j,(title,im) in enumerate([('只输入原图',rgb),('预测墨迹概率',np.uint8(255*(1-pred[1]))),('预测中心概率',np.uint8(255*(1-pred[0]))),('联合训练候选路径',v21.line_view(rgb.shape[:2],paths,True))]):d.text((400*j+5,8),title,font=FONT,fill='black');v21.paste(board,im,400*j+5,40,390,390)
    board.save(out/'comparison.jpg',quality=95);jsave(out/'summary.json',dict(source=str(source),source_sha256=digest(source),checkpoint_sha256=digest(ROOT/'joint'/'final.pt'),reference_labels_used=False,new_reference_module_loaded='v24_visible_references' in sys.modules,robot_connected=False,official_model_replaced=False,input_shape=list(rgb.shape[:2]),path_count=len(paths),auxiliary_decoding_enabled=decision['auxiliary_decoding_enabled']));print('V24_REFERENCE_FREE',str(out),flush=True)

if __name__=='__main__':main()
