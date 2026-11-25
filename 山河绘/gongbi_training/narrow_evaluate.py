"""All heldout real crops: weak agreement and previews, never gold accuracy."""
import argparse
import hashlib
import json
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from .runtime import LAB
from .model import GongbiLineNet
from .narrow_collect import save_json


@torch.no_grad()
def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='mobilenet_unet_narrow_v03');args=parser.parse_args(argv)
    torch.set_num_threads(4);cv2.setNumThreads(2)
    run=LAB/'runs'/args.run
    if not (run/'training_summary.json').exists():raise RuntimeError('Final heldout evaluation is only allowed after the training run completes')
    manifest=json.loads((run/'dataset_manifest_snapshot.json').read_text(encoding='utf-8'))
    records=[r for r in manifest['real'] if r['split']=='test']
    checkpoint=run/'best.pt';weight_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    ckpt=torch.load(checkpoint,map_location='cuda',weights_only=False)
    model=GongbiLineNet().cuda().eval();model.load_state_dict(ckpt['model'])
    out=run/'heldout_test_all';out.mkdir(exist_ok=True)
    tp=fp=fn=0;rows=[]
    for begin in range(0,len(records),12):
        batch=records[begin:begin+12]
        images=[np.asarray(Image.open(r['image']).convert('RGB')) for r in batch]
        x=torch.from_numpy(np.stack([im.transpose(2,0,1) for im in images]).astype(np.float32)/255.).cuda()
        probabilities=model(x).sigmoid().cpu().numpy()[:,0]
        for r,rgb,prob in zip(batch,images,probabilities):
            with np.load(r['labels']) as labels:
                target=labels['targets'][0]/255.;weight=labels['weights'][0]/255.
            valid=weight>0;pred=prob>.5;truth=target>.3
            a=int((pred&truth&valid).sum());b=int((pred&~truth&valid).sum());c=int((~pred&truth&valid).sum())
            tp+=a;fp+=b;fn+=c
            Image.fromarray(np.rint(prob*255).astype(np.uint8)).save(out/(r['id']+'_probability.png'))
            rows.append(dict(id=r['id'],group=r['group'],tp=a,fp=b,fn=c,
                             predicted_primary_fraction=float(pred.mean()),expert_ground_truth=False))
        print('FINAL_TEST',min(begin+12,len(records)),'/',len(records),flush=True)
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
    result=dict(real_test_crops=len(records),groups=sorted({r['group'] for r in records}),checkpoint_step=ckpt['step'],checkpoint_sha256=weight_hash,
                threshold=.5,weak_label_agreement=dict(precision=precision,recall=recall,f1=2*precision*recall/max(precision+recall,1e-8)),
                actual_expert_painting_accuracy_available=False,
                interpretation='Agreement with automatic contour proposals, not expert extraction accuracy. All test crops evaluated after training; does not select checkpoints.',images=rows)
    save_json(run/'heldout_test_metrics.json',result)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    for page,start in enumerate(range(0,len(records),12)):
        chosen=records[start:start+12]
        canvas=Image.new('RGB',(1800,1400),(246,244,237));draw=ImageDraw.Draw(canvas)
        draw.text((15,12),'全部留出山石测试：原图 / 模型概率；顺序固定，无人工挑选',font=font,fill=(30,45,35))
        for i,r in enumerate(chosen):
            x=(i%3)*600;y=60+(i//3)*335
            draw.text((x+6,y),r['id'][-20:],font=font,fill=(30,45,35))
            original=Image.open(r['image']);original.thumbnail((280,280));canvas.paste(original,(x+8,y+30))
            p=np.asarray(Image.open(out/(r['id']+'_probability.png')))
            preview=Image.fromarray(255-p);preview.thumbnail((280,280));canvas.paste(preview,(x+305,y+30))
        canvas.save(out/f'test_sheet_{page:02d}.jpg',quality=92)
    print(json.dumps({k:v for k,v in result.items() if k!='images'},ensure_ascii=False,indent=2),flush=True)
