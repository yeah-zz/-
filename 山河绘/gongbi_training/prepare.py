"""Prepare honest weak labels, paired synthetic data, and grouped splits."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from .runtime import LAB,ROOT
from .labels import weak_targets,annotation_regions
from .synthetic import make_sample
from .guidance import apply_guidance


def save(path,rgb):
    Image.fromarray(rgb).save(path)


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--synthetic',type=int,default=640)
    parser.add_argument('--grids-only',action='store_true')
    parser.add_argument('--manual-only',action='store_true')
    args=parser.parse_args(argv)
    if args.manual_only:
        manifest=json.loads((LAB/'datasets'/'manifest.json').read_text(encoding='utf-8'))
        changed=0
        for r in manifest['real']:
            manual=LAB/'data'/'manual'/(r['id']+'.png')
            if not manual.exists():continue
            labels=np.asarray(Image.open(manual))
            with np.load(r.get('original_weak_labels',r['labels'])) as a:
                targets=a['targets'].copy();weights=a['weights'].copy()
            if labels.shape!=targets.shape[1:] or labels.max()>3:raise ValueError('Invalid manual labels: '+r['id'])
            known=labels>0
            targets[0,known]=(labels[known]==2).astype(np.uint8)*255
            targets[1,known]=(labels[known]==3).astype(np.uint8)*255
            weights[:,known]=255
            # Original weak proposal remains intact; review can be revised.
            target_file=manual.with_name(r['id']+'_merged.npz')
            np.savez_compressed(target_file,targets=targets,weights=weights)
            r.setdefault('original_weak_labels',r['labels'])
            r['labels']=str(target_file);r['manual_pixels']=int(known.sum())
            r['supervision']='weak_plus_user_reviewed_sparse_pixels_NOT_dense_gold'
            changed+=1
        (LAB/'datasets'/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        print('MANUAL_MERGE',changed,'images; original painting splits preserved')
        return
    sources=json.loads((LAB/'data'/'sources.json').read_text(encoding='utf-8'))
    if args.grids_only:
        for r in sources['records']:
            if r['id'] not in ('user_qianli_02','user_qianli_05'):continue
            im=Image.open(r['path']).convert('RGB');draw=ImageDraw.Draw(im)
            font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
            for x in range(0,im.width,100):
                draw.line([(x,0),(x,im.height)],fill=(255,80,120),width=1)
                for y in range(0,im.height,200):draw.text((x+2,y+2),str(x),font=font,fill=(255,255,255))
            for y in range(0,im.height,100):
                draw.line([(0,y),(im.width,y)],fill=(255,80,120),width=1)
                draw.text((2,y+15),str(y),font=font,fill=(255,255,255))
            im.save(LAB/'previews'/(r['id']+'_coordinate_grid.png'))
        return
    processed=[]
    for r in sources['records']:
        folder=LAB/'data'/'processed'/r['id'];folder.mkdir(exist_ok=True)
        im=Image.open(r['path']).convert('RGB')
        ratio=min(1.,1600/max(im.size));im=im.resize((round(im.width*ratio),round(im.height*ratio)),Image.Resampling.LANCZOS)
        rgb=np.asarray(im).copy()
        targets,weights,colors=weak_targets(rgb)
        targets,weights=annotation_regions(r['id'],rgb.shape,targets,weights)
        targets,weights,guide_overlay,guide_info=apply_guidance(r['id'],rgb,targets,weights)
        if guide_info:
            save(folder/'sparse_guidance_overlay.png',guide_overlay)
            (folder/'sparse_guidance.json').write_text(json.dumps(guide_info,ensure_ascii=False,indent=2),encoding='utf-8')
        # Very dark mount/frame edges are not verified content. Ignore a narrow
        # outer border rather than teach black frames as mountain contours.
        weights[:,:5,:]=0;weights[:,-5:,:]=0;weights[:,:,:5]=0;weights[:,:,-5:]=0
        save(folder/'image.png',rgb)
        np.savez_compressed(folder/'weak_labels.npz',targets=(targets*255).astype(np.uint8),weights=(weights*255).astype(np.uint8))
        structure=np.repeat((255-np.rint(targets[0]*255).astype(np.uint8))[:,:,None],3,axis=2)
        save(folder/'weak_structure.png',structure)
        panel=Image.new('RGB',(1200,440),(245,244,238))
        font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
        draw=ImageDraw.Draw(panel)
        for col,(label,view) in enumerate([('原图',Image.fromarray(rgb)),('自动弱标签（不是人工真值）',Image.fromarray(structure))]):
            draw.text((20+col*600,15),label,font=font,fill=(35,45,40));view.thumbnail((560,370));panel.paste(view,(20+col*600+(560-view.width)//2,55+(370-view.height)//2))
        panel.save(folder/'label_preview.png')
        processed.append(dict(id=r['id'],group=r['group'],split=r['split'],image=str(folder/'image.png'),
                              labels=str(folder/'weak_labels.npz'),size=list(im.size),
                              supervision='automatic_weak_proposal_NOT_gold',sparse_guidance=guide_info,source=r))
        print('WEAK_LABELS',r['id'],list(im.size),flush=True)
    synthetic=LAB/'datasets'/'synthetic';synthetic.mkdir(exist_ok=True)
    synth_records=[]
    for i in range(args.synthetic+64):
        split='train' if i<args.synthetic else 'validation'
        seed=i if split=='train' else 100000+i
        rgb,target=make_sample(seed,320)
        prefix=synthetic/f'{split}_{seed:06d}'
        save(str(prefix)+'.png',rgb)
        np.savez_compressed(str(prefix)+'.npz',targets=(target*255).astype(np.uint8))
        synth_records.append(dict(id=prefix.name,seed=seed,split=split,image=str(prefix)+'.png',
                                  labels=str(prefix)+'.npz',supervision='known_procedural_geometry'))
    train_groups={r['group'] for r in processed if r['split']=='train'}
    val_groups={r['group'] for r in processed if r['split']=='validation'}
    assert not(train_groups&val_groups)
    manifest=dict(real=processed,synthetic=synth_records,policy='structure-first',
                  training_groups=sorted(train_groups),validation_groups=sorted(val_groups),
                  split_by_painting_or_album=True,real_dense_human_labels=0,
                  note='Synthetic metrics only measure procedural distribution. Weak-label agreement is not real artwork accuracy.')
    (LAB/'datasets'/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    # Show paired data rather than present generated paintings as real samples.
    gallery=Image.new('RGB',(1600,800),(244,243,236));draw=ImageDraw.Draw(gallery)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
    for i in range(4):
        rgb,target=make_sample(i,320)
        x=i*400
        draw.text((x+10,10),'合成输入 / 已知几何标签',font=font,fill=(30,40,35))
        gallery.paste(Image.fromarray(rgb),(x+20,55))
        line=np.repeat((255-(target[0]*255).astype(np.uint8))[:,:,None],3,axis=2)
        gallery.paste(Image.fromarray(line),(x+20,405))
    gallery.save(LAB/'previews'/'synthetic_pairs.png')
    print('DATA_READY',len(processed),'real images',len(synth_records),'paired synthetic patches',flush=True)
