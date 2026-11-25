# -*- coding: utf-8 -*-
import sys,json
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure
configure()
import torch,numpy as np,cv2
from PIL import Image,ImageDraw
from scipy.ndimage import distance_transform_edt,maximum_filter
from stroke_topology_v13 import ROOT,PARENT,procedural,model_new,jsave,digest,infer
from topology_trace_v13 import extract
from evaluate_clean_strokes_v12 import path_metrics,line_view
from human_mountain_v11 import FONT,paste

def check():
    torch.set_num_threads(4);cv2.setNumThreads(2);folder=ROOT/'small_fit';model=model_new().cuda();model.load_state_dict(torch.load(folder/'final.pt',map_location='cuda',weights_only=False)['model']);model.eval();rows=[];board=Image.new('RGB',(1800,450*3+40),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,5),'三组已知轨迹小样：只检查训练拟合与追踪，不是实际山石提取准确率',font=FONT,fill='black')
    for i,seed in enumerate((710002,710001,710003)):
        rgb,target,mask,reference=procedural(seed)
        with torch.no_grad():p=model(torch.from_numpy(rgb.transpose(2,0,1).copy())[None].cuda().float()/255).float();p[:,[0,5,6,7]]=p[:,[0,5,6,7]].sigmoid();p[:,1:3]=torch.nn.functional.normalize(p[:,1:3],dim=1);p[:,3:5]=torch.nn.functional.normalize(p[:,3:5],dim=1);pred=p[0].cpu().numpy()
        raw,_,sk=extract(pred,False);paths,info,_=extract(pred,True);truth=target[0]>.95;detected=pred[0]>=.5;distance=distance_transform_edt(~detected);to_ref=distance_transform_edt(~truth);recall=float(np.mean(distance[truth]<=3));precision=float(np.mean(to_ref[detected]<=3));scores=path_metrics(reference,paths,np.ones(truth.shape,np.uint8))
        keys={}
        for name,ch in [('endpoints',5),('junctions',6)]:
            t=target[ch]>.99;est=(pred[ch]>=.3)&(pred[ch]>=maximum_filter(pred[ch],size=5));dist=distance_transform_edt(~est) if est.any() else np.full(t.shape,1000.);keys[name+'_recall_5px']=float(np.mean(dist[t]<=5)) if t.any() else None
        rows.append(dict(seed=seed,center_recall_3px=recall,center_precision_3px=precision,raw_paths=len(raw),**scores,**keys));np.savez_compressed(folder/(f'known_{i}_prediction.npz'),prediction=pred)
        heat=np.uint8(255*(1-pred[0]));key=rgb.copy();key[pred[5]>.3]=[235,40,30];key[pred[6]>.3]=[20,90,240]
        for j,(label,a) in enumerate([('程序输入图',rgb),('网络主线概率',np.repeat(heat[:,:,None],3,2)),('起止点红 / 交叉蓝',key),('预测分支',line_view(truth.shape,raw,True)),('交叉配对后',line_view(truth.shape,paths,True))]):d.text((j*360+7,45+i*450),label,font=FONT,fill='black');paste(board,a,j*360+7,85+i*450,346,350)
    passed=all(r['center_recall_3px']>=.95 and r['center_precision_3px']>=.9 for r in rows);jsave(folder/'diagnostic.json',dict(localization_gate_passed=passed,baseline_sha256=digest(PARENT),rows=rows,gate_is_not_generalization=True,all_inputs_known_procedural=True,identity_pairing_fully_solved=False));board.save(folder/'comparison.jpg',quality=95)
    print('SMALL_GATE',json.dumps(dict(passed=passed,rows=rows)),flush=True)
if __name__=='__main__':check()
