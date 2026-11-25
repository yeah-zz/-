"""New procedural diagnostic, independent seeds; never real-art accuracy."""
import argparse
import json
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from .runtime import LAB
from .model import GongbiLineNet
from .synthetic import make_sample


def make_challenge(seed):
    rgb,target=make_sample(seed,320)
    rng=np.random.default_rng(seed+230000)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    # Test a palette unseen in this exact synthetic seed, not just blue-green.
    rgb=np.clip(gray[:,:,None]*np.array([1.08,1.,.82])[None,None,:],0,255).astype(np.uint8)
    frame=np.zeros((320,320),np.uint8)
    for _ in range(3):
        x=int(rng.integers(15,305));width=int(rng.integers(2,5))
        cv2.line(rgb,(x,0),(x,319),(26,23,18),width)
        cv2.line(frame,(x,0),(x,319),255,width)
    cv2.rectangle(rgb,(0,0),(319,319),(28,25,20),5)
    cv2.rectangle(frame,(0,0),(319,319),255,5)
    target[:,frame>0]=0
    return rgb,target,frame


@torch.no_grad()
def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='mobilenet_unet_v02');parser.add_argument('--baseline',default='mobilenet_unet_v01');args=parser.parse_args(argv)
    torch.set_num_threads(4);cv2.setNumThreads(2)
    output=LAB/'runs'/args.run
    cases=[make_challenge(900000+i) for i in range(16)]
    scores={};pictures={}
    for name in [args.baseline,args.run]:
        ckpt=torch.load(LAB/'runs'/name/'best.pt',map_location='cuda',weights_only=False)
        model=GongbiLineNet().cuda().eval();model.load_state_dict(ckpt['model'])
        tp=fp=fn=frame_positive=frame_pixels=0
        for begin in range(0,len(cases),4):
            batch=cases[begin:begin+4]
            x=np.stack([r.transpose(2,0,1) for r,_,_ in batch]).astype(np.float32)/255.
            probs=model(torch.from_numpy(x).cuda()).sigmoid().cpu().numpy()[:,0]
            for j,(prob,(_,target,frame)) in enumerate(zip(probs,batch)):
                pred=prob>.55;truth=target[0]>.3
                tp+=int((pred&truth).sum());fp+=int((pred&~truth).sum());fn+=int((~pred&truth).sum())
                frame_positive+=int((pred&(frame>0)).sum());frame_pixels+=int((frame>0).sum())
                if begin==0:pictures[(name,j)]=255-np.rint(prob*255).astype(np.uint8)
        precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
        scores[name]=dict(checkpoint_step=ckpt['step'],f1=2*precision*recall/max(precision+recall,1e-8),
                          precision=precision,recall=recall,frame_false_positive_fraction=frame_positive/max(frame_pixels,1))
        del model
    info=dict(interpretation='16 new procedural geometry samples with faded palette and synthetic frame distractors; NOT real painting accuracy',
              seeds=list(range(900000,900016)),used_for_training=False,threshold=.55,metrics=scores)
    (output/'synthetic_challenge.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    canvas=Image.new('RGB',(1440,1550),(246,245,239));draw=ImageDraw.Draw(canvas)
    draw.text((15,10),'全新合成诊断样本：褪色 + 已知分隔线；不是实际国画准确率',font=font,fill=(30,45,35))
    for i in range(4):
        arrays=[cases[i][0],255-np.rint(cases[i][1][0]*255).astype(np.uint8),pictures[(args.baseline,i)],pictures[(args.run,i)]]
        for j,(label,array) in enumerate(zip(['输入','已知结构标签','第一版概率','第二版概率'],arrays)):
            draw.text((j*360+10,i*375+50),label,font=font,fill=(30,45,35))
            canvas.paste(Image.fromarray(array),(j*360+10,i*375+85))
    canvas.save(output/'synthetic_challenge.png')
    print(json.dumps(info,ensure_ascii=True,indent=2),flush=True)
