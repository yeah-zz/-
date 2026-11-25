# -*- coding: utf-8 -*-
"""Batch offline inference with explicit model-review uncertainty and SVG export."""
from rock_v15_common import *
import argparse,html
from datetime import datetime,timezone,timedelta
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
from gongbi_training.stroke_train_v08 import model_new,infer
from rock_pairs_v15 import load,connect,frozen_paths
from rock_eval_v15 import quality_audit
from human_mountain_v11 import paste
from evaluate_clean_strokes_v12 import line_view

def run(input_path,output_path=None,round_index=4):
    source=Path(input_path);source.mkdir(parents=True,exist_ok=True) if not source.exists() and not source.suffix else None
    files=[source] if source.is_file() else sorted(p for p in source.iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'])
    if not files:raise ValueError('No images in '+str(source))
    dest=Path(output_path) if output_path else ROOT/'batch_inference'/datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    if dest.drive.upper()=='C:':raise ValueError('Project policy: outputs cannot be on C:')
    dest.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(2);raster=model_new().cuda();raster.load_state_dict(torch.load(PARENT,map_location='cuda',weights_only=False)['model']);raster.eval();reviewer,threshold=load(round_index);rows=[]
    for i,path in enumerate(files):
        rgb=np.asarray(Image.open(path).convert('RGB'));h,w=rgb.shape[:2];p=infer(raster,rgb);raw=frozen_paths(p);paths,events,uncertain=connect(raw,p[0],rgb,reviewer,threshold);paths=[q for q in paths if np.linalg.norm(np.diff(q,axis=0),axis=1).sum()>=12];quality=quality_audit(paths,p[0],rgb,uncertain);out=dest/(str(i+1).zfill(3)+'_'+path.stem);out.mkdir(exist_ok=True)
        jsave(out/'trajectory.json',dict(paths=[q.round(2).tolist() for q in paths],connection_decisions=events,uncertain_endpoint_pairs=uncertain,coordinates='pixels',physical_z=None,ready_for_robot=False,actually_drawn=False,experimental_round=round_index));jsave(out/'automatic_review.json',quality);Image.fromarray(line_view((h,w),paths,True)).save(out/'prediction.png')
        board=Image.new('RGB',(1400,650),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,5),'离线实验结果：模型判接线＋程序质检，非专家保证；颜色分笔',font=FONT,fill='black');paste(board,rgb,10,45,680,590);paste(board,line_view((h,w),paths,True),710,45,680,590);board.save(out/'comparison.jpg',quality=95)
        svg=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">','<title>Experimental pixel paths; not robot commands</title>']
        for j,q in enumerate(paths):svg.append('<polyline id="stroke_'+str(j)+'" points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in q)+'" fill="none" stroke="black" stroke-width="1"/>')
        svg.append('</svg>');(out/'trajectory.svg').write_text('\n'.join(svg),encoding='utf-8');rows.append(dict(source=str(path),source_sha256=digest(path),paths=len(paths),joins=len(events),quality=quality,result=str(out/'comparison.jpg')));print('AUTO_INFERENCE',path.name,len(paths),quality['status'],flush=True)
    jsave(dest/'summary.json',dict(records=rows,round=round_index,raster_sha256=digest(PARENT),reviewer_sha256=digest(ROOT/f'round_{round_index:02d}'/'pair_net.pt'),per_image_user_review_required=False,expert_correctness_not_guaranteed=True,robot_connected=False))
    print('BATCH_RESULTS',str(dest),flush=True);return dest

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',default=str(ROOT/'new_inputs'));parser.add_argument('--output');parser.add_argument('--round',type=int,default=4);args=parser.parse_args();run(args.input,args.output,args.round)
