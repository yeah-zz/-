# -*- coding: utf-8 -*-
"""Image-only stage ablation; no training, reference labels, or robot access."""
import os, sys
from pathlib import Path
sys.dont_write_bytecode=True
LAB=Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022')
for k,v in {'TEMP':'temp','TMP':'temp','PIP_CACHE_DIR':'cache/pip','TORCH_HOME':'cache/torch','HF_HOME':'cache/huggingface','XDG_CACHE_HOME':'cache','CUDA_CACHE_PATH':'cache/cuda'}.items():os.environ[k]=str(LAB/v)
from rock_v15_common import read, jsave
import v28_stroke_relations as old
import numpy as np, cv2, torch, json
from PIL import Image,ImageDraw,ImageFont
from skimage.morphology import skeletonize
from topology_trace_v13 import graph_paths

SOURCE=LAB/'generated_simple_landscapes_20261028'
OUT=SOURCE/'loss_diagnosis'
SCENES=['01_rocks','02_pine','03_pavilion']

def black(shape,paths):
    im=np.full((*shape,3),255,np.uint8)
    for p in paths:
        if len(p)>1:cv2.polylines(im,[np.rint(p).astype(np.int32)],False,(0,0,0),1,cv2.LINE_AA)
    return im

def native_graph(pred,rgb):
    # Same frozen backbone, original trace policy and learned relation model.
    g=old.add_visual(old.build_graph(pred,rgb),model)
    choice=read(old.ROOT/'calibration.json')['selected']
    probabilities=old.probabilities(g,ensemble)
    accepted,_=old.select(g,probabilities,choice['threshold'],choice['margin'])
    return old.assemble(g,accepted),g['trace'],len(accepted),g['chains']

def panels(folder,views):
    # Report layout only; extraction images are neither retouched nor filtered.
    width=410;rowh=468
    board=Image.new('RGB',(width*3,rowh*2+75),'#f3f1eb');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',19)
    for i,(title,im) in enumerate(views):
        x=i%3*width;y=i//3*rowh
        draw.text((x+6,y+5),title,font=font,fill='#222222')
        pil=Image.fromarray(im);pil.thumbnail((400,400),Image.Resampling.LANCZOS);board.paste(pil,(x+6,y+40))
    draw.text((6,rowh*2+6),'固定权重消融：更高分辨率不代表识别正确；墨迹图不是单笔轨迹。',font=font,fill='#333333')
    board.save(folder/'stages.jpg',quality=95)

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);cv2.setNumThreads(2)
    model=old.exp.load_model('base');ensemble=old.models('visual')
    summary=[]
    for sid in SCENES:
        folder=OUT/sid;folder.mkdir(exist_ok=True)
        native=np.asarray(Image.open(SOURCE/'originals'/(sid+'.png')).convert('RGB'))
        per=[];predictions={};outputs={}
        for size in [640,1254]:
            im=Image.fromarray(native);im.thumbnail((size,size),Image.Resampling.LANCZOS);rgb=np.asarray(im)
            pred=old.exp.prev.prior.predict(model,rgb)[:5]
            np.savez_compressed(folder/f'prediction_{size}.npz',prediction=pred)
            paths,trace,joined,chains=native_graph(pred,rgb)
            early,_=old.exp.prev.adaptive_curves(pred,read(old.exp.prev.ROOT/'calibration.json')['selected']['config'])
            raw_ink=np.clip((1-pred[1])*255,0,255).astype(np.uint8)
            Image.fromarray(raw_ink).save(folder/f'model_ink_{size}.png')
            Image.fromarray(black(rgb.shape[:2],early)).save(folder/f'field_traces_{size}.png')
            Image.fromarray(black(rgb.shape[:2],paths)).save(folder/f'final_{size}.png')
            Image.fromarray(black(rgb.shape[:2],chains)).save(folder/f'unjoined_{size}.png')
            jsave(folder/f'paths_{size}.json',{'paths':[p.tolist() for p in paths],'reference_used':False,'robot_connected':False})
            per.append({'input_size':list(im.size),'paths':len(paths),'traces':len(early),'unjoined_chains':len(chains),'accepted_pairs':joined,'trace':trace,'center_quantiles':np.quantile(pred[0],[.5,.9,.99]).tolist(),'ink_quantiles':np.quantile(pred[1],[.5,.9,.99]).tolist()})
            predictions[size]=pred;outputs[size]=paths
            print('DIAGNOSED',sid,size,json.dumps(per[-1]),flush=True)
        rgb=native;pred=predictions[1254]
        panels(folder,[('原图',rgb),('旧输出：640 输入',black((640,640),outputs[640])),('同权重：原分辨率输入',black(rgb.shape[:2],outputs[1254])),('原分辨率：墨迹概率图',np.repeat(np.clip((1-pred[1])*255,0,255).astype(np.uint8)[:,:,None],3,2)),('原分辨率：方向场追踪',np.asarray(Image.open(folder/'field_traces_1254.png'))),('原分辨率：不连接的图边',np.asarray(Image.open(folder/'unjoined_1254.png')))])
        if sid=='02_pine':
            # Fixed image-coordinate crop chosen by subject location, no labels.
            box=(130,240,705,565)
            detail=Image.new('RGB',(1150,730),'#f3f1eb');d=ImageDraw.Draw(detail);font=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',20)
            candidates=[('原图：松冠局部',rgb),('模型墨迹概率：松冠局部',np.repeat(np.clip((1-pred[1])*255,0,255).astype(np.uint8)[:,:,None],3,2)),('原分辨率方向场轨迹',np.asarray(Image.open(folder/'field_traces_1254.png'))),('原分辨率最终轨迹',black(rgb.shape[:2],outputs[1254]))]
            for i,(label,a) in enumerate(candidates):
                x=(i%2)*575;y=(i//2)*365;d.text((x+5,y+5),label,font=font,fill='black');detail.paste(Image.fromarray(a).crop(box),(x,y+35))
            detail.save(OUT/'pine_detail.jpg',quality=95)
        summary.append({'id':sid,'runs':per})
    jsave(OUT/'ablation.json',{'backbone_unchanged':True,'references_used':False,'robot_connected':False,'new_training_steps':0,'gpu':torch.cuda.get_device_name(),'samples':summary,'limitations':'No pixel or stroke gold; path counts are diagnostics, not accuracy.'})
    print('DIAGNOSIS_COMPLETE',str(OUT),flush=True)
