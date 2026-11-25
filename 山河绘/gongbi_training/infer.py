"""Overlapping learned inference; no painting-specific path deletion."""
from pathlib import Path
import argparse
import json
import time
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from .runtime import LAB
from .model import GongbiLineNet


@torch.no_grad()
def predict(model,rgb,device,tile=320,stride=224):
    h,w=rgb.shape[:2]
    padded=cv2.copyMakeBorder(rgb,0,max(0,tile-h),0,max(0,tile-w),cv2.BORDER_REFLECT_101)
    ph,pw=padded.shape[:2]
    def starts(n):return sorted(set(list(range(0,n-tile+1,stride))+[n-tile]))
    window=np.maximum(np.outer(np.hanning(tile),np.hanning(tile)),.05).astype(np.float32)
    total=np.zeros((2,ph,pw),np.float32);den=np.zeros((ph,pw),np.float32)
    locations=[(y,x) for y in starts(ph) for x in starts(pw)]
    for begin in range(0,len(locations),8):
        group=locations[begin:begin+8]
        arr=np.stack([padded[y:y+tile,x:x+tile].transpose(2,0,1) for y,x in group]).astype(np.float32)/255.
        probs=model(torch.from_numpy(arr).to(device)).sigmoid().cpu().numpy()
        for p,(y,x) in zip(probs,group):
            total[:,y:y+tile,x:x+tile]+=p*window
            den[y:y+tile,x:x+tile]+=window
    return (total/den[None])[:,:h,:w]


def paths_from_probability(prob,low=.35,high=.55,min_length=12):
    from original_d_tracer import LandscapeBrushGenerator as Trace
    n,components,stats,_=cv2.connectedComponentsWithStats((prob>=low).astype(np.uint8),8)
    selected=np.zeros(prob.shape,np.uint8)
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]<min_length:continue
        box=components==i
        if np.any(prob[box]>=high):selected[box]=255
    selected[[0,-1],:]=0;selected[:,[0,-1]]=0
    skeleton=Trace._thin_binary(selected)
    paths=[]
    for p in Trace._trace_skeleton_paths(skeleton):
        points=np.asarray(p,float)
        length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
        if length>=min_length:paths.append(p)
    canvas=np.full((*prob.shape,3),255,np.uint8)
    for p in paths:cv2.polylines(canvas,[np.asarray(p,np.int32)],False,(25,30,26),1,cv2.LINE_AA)
    return paths,canvas


def panel(rgb,line,prob,subtitle):
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    overlay=np.rint(rgb*.75+255*.25).astype(np.uint8)
    overlay[prob>.5]=[225,40,50]
    result=Image.new('RGB',(1800,880),(246,245,239));draw=ImageDraw.Draw(result)
    draw.text((18,12),subtitle,font=font,fill=(35,43,38))
    for i,(label,array) in enumerate([('原图',rgb),('模型结构线预测',line),('预测叠加（红色）',overlay)]):
        x=15+i*600;draw.text((x,55),label,font=font,fill=(35,43,38))
        im=Image.fromarray(array);im.thumbnail((570,750))
        result.paste(im,(x+(570-im.width)//2,100+(750-im.height)//2))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='mobilenet_unet_v01')
    parser.add_argument('--checkpoint',default='best.pt')
    parser.add_argument('--ids',nargs='*')
    parser.add_argument('--low',type=float,default=.35)
    parser.add_argument('--high',type=float,default=.55)
    parser.add_argument('--output-label',default='inference',help='Separate previews for best, last or intermediate checkpoints')
    args=parser.parse_args(argv)
    torch.set_num_threads(4);cv2.setNumThreads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run=LAB/'runs'/args.run
    ckpt=torch.load(run/args.checkpoint,map_location=device,weights_only=False)
    model=GongbiLineNet(pretrained=False).to(device);model.load_state_dict(ckpt['model']);model.eval()
    manifest_file=run/'dataset_manifest_snapshot.json'
    manifest=json.loads((manifest_file if manifest_file.exists() else LAB/'datasets'/'manifest.json').read_text(encoding='utf-8'))
    pool=manifest.get('inference_records',manifest['real'])
    records=[r for r in pool if (r['id'] in args.ids if args.ids else True if 'inference_records' in manifest else r['id'].startswith('user_') or r['split']=='validation')]
    if not args.output_label.replace('_','').replace('-','').isalnum():raise ValueError('Output label must be a simple folder name')
    out=run/args.output_label;out.mkdir(exist_ok=True)
    rows=[]
    for r in records:
        start=time.perf_counter();folder=out/r['id'];folder.mkdir(exist_ok=True)
        rgb=np.asarray(Image.open(r['image']).convert('RGB'))
        probability=predict(model,rgb,device,tile=ckpt['config']['tile'])
        paths,line=paths_from_probability(probability[0],args.low,args.high)
        for name,arr in [('structure_probability',probability[0]),('detail_probability',probability[1])]:
            Image.fromarray(np.rint(arr*255).astype(np.uint8)).save(folder/(name+'.png'))
        Image.fromarray(line).save(folder/'structure.png')
        h,w=rgb.shape[:2]
        svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
             '<g fill="none" stroke="#19201a" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">']
        for p in paths:svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in p)+'"/>')
        svg.extend(['</g>','</svg>']);(folder/'structure.svg').write_text('\n'.join(svg),encoding='utf-8')
        role='训练参考图（不代表泛化验证）' if r['split']=='train' else ('独立留出测试（暂无人工真值）' if r['split']=='test' else '独立留出验证（暂无人工真值）')
        panel(rgb,line,probability[0],r['id']+' | '+role).save(folder/'comparison.png')
        info=dict(id=r['id'],split=r['split'],group=r['group'],width=w,height=h,paths=paths,
                  path_count=len(paths),checkpoint_step=ckpt['step'],low=args.low,high=args.high,
                  checkpoint_file=args.checkpoint,
                  min_length_pixels=12,seconds=time.perf_counter()-start,ready_for_robot=False,
                  method='learned probability + shared hysteresis + skeleton tracing; no per-image manual cleanup')
        (folder/'paths.json').write_text(json.dumps(info,ensure_ascii=False),encoding='utf-8')
        rows.append({k:v for k,v in info.items() if k!='paths'})
        print('INFERENCE',r['id'],len(paths),round(info['seconds'],2),flush=True)
    (out/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print('INFERENCE_COMPLETE',str(out),flush=True)
