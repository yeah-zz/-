"""Rank the new real-image corpus by style, color and simple structure."""
from pathlib import Path
import argparse
import json
import sys
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from .runtime import LAB
from .narrow_collect import CORPUS,save_json


def image_statistics(rgb):
    rgb=cv2.resize(rgb,(min(800,rgb.shape[1]),round(rgb.shape[0]*min(1,800/rgb.shape[1])))) if rgb.shape[1]>800 else rgb
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    green=((hsv[:,:,0]>34)&(hsv[:,:,0]<135)&(hsv[:,:,1]>45)&(hsv[:,:,2]>45))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    edge=cv2.Canny(cv2.bilateralFilter(gray,7,35,5),55,135)>0
    d=cv2.resize(gray,(17,16));bits=(d[:,1:]>d[:,:-1]).flatten()
    return dict(blue_green_fraction=float(green.mean()),edge_density=float(edge.mean()),
                sharpness=float(cv2.Laplacian(gray,cv2.CV_32F).var()),
                dhash=''.join(f'{int(x):02x}' for x in np.packbits(bits)))


def load_clip():
    sys.path.insert(0,r'E:\Codex_Workspace_Data\CLIPascene_20261021\source\assets\CLIP-main')
    sys.path.append(r'E:\Codex_Workspace_Data\CLIPascene_20261021\environment\deps')
    import clip
    model,transform=clip.load(r'E:\Codex_Workspace_Data\CLIPascene_20261021\source\assets\ViT-B-32.pt',device='cuda',jit=False)
    return clip,model.eval(),transform


@torch.no_grad()
def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=0);parser.add_argument('--from-progress',action='store_true');args=parser.parse_args(argv)
    torch.set_num_threads(4);cv2.setNumThreads(2)
    payload=json.loads((CORPUS/('download_progress.json' if args.from_progress else 'downloaded_sources.json')).read_text(encoding='utf-8'))
    records=payload['records']
    if args.limit:records=records[:args.limit]
    clip,model,transform=load_clip()
    positive=[
        'an ancient Chinese blue and green gongbi landscape painting with clear mountain outlines and turquoise rocks',
        'a detail of A Thousand Li of Rivers and Mountains by Wang Ximeng, Chinese blue green landscape painting',
        'a traditional Chinese blue green painting of a mountain or a rock with simple clear contours',
    ]
    negative=[
        'a dense black and white Chinese ink landscape painting with countless fine texture strokes',
        'a traditional Chinese painting of flowers and birds',
        'a Chinese portrait painting, people in a palace, human figures and clothing',
        'Chinese calligraphy writing, a seal, a signature, text on silk',
        'a photograph of a mountain, rock or a modern building',
        'a map of a Chinese imperial garden, numerous buildings, pavilions, trees and roads',
    ]
    text=model.encode_text(clip.tokenize(positive+negative).cuda()).float();text/=text.norm(dim=1,keepdim=True)
    references=json.loads((LAB/'data'/'user_sources.json').read_text(encoding='utf-8'))
    ref_batch=torch.stack([transform(Image.open(r['path']).convert('RGB')) for r in references if r['id'] in ['user_qianli_02','user_qianli_05','user_qianli_06']]).cuda()
    ref=model.encode_image(ref_batch).float();ref/=ref.norm(dim=1,keepdim=True)
    scored=[]
    for start in range(0,len(records),24):
        batch=records[start:start+24]
        images=[]
        for r in batch:
            with Image.open(r['path']) as im:
                im.thumbnail((1800,1800));images.append(im.convert('RGB'))
        x=torch.stack([transform(im) for im in images]).cuda()
        features=model.encode_image(x).float();features/=features.norm(dim=1,keepdim=True)
        similarities=(features@text.T).cpu().numpy();refs=(features@ref.T).max(1).values.cpu().numpy()
        for r,im,sim,refsim in zip(batch,images,similarities,refs):
            stats=image_statistics(np.asarray(im));pos=float(sim[:len(positive)].max());neg=float(sim[len(positive):].max())
            margin=pos-neg
            reasons=[]
            if stats['blue_green_fraction']<.025:reasons.append('insufficient_blue_green_mountain_color')
            if pos<.235 or margin<-.025:reasons.append('style_or_subject_mismatch')
            if stats['edge_density']>.17:reasons.append('too_many_fine_edges')
            if stats['sharpness']<12:reasons.append('blurred')
            if min(im.size)<250:reasons.append('insufficient_detail_resolution')
            score=.65*refsim+.35*pos-.5*max(0,stats['edge_density']-.065)
            scored.append(dict(r,statistics=stats,clip_positive=pos,clip_negative=neg,clip_margin=margin,
                               reference_similarity=float(refsim),selection_score=float(score),
                               selected=not reasons,rejection_reasons=reasons,
                               review_status='automatically_screened_NOT_expert_curated'))
        if start%120==0:print('STYLE_SCREEN',min(start+24,len(records)),'/',len(records),flush=True)
    # Conservative image-near-duplicate removal before any split.
    accepted=[];hashes=[]
    for r in sorted(scored,key=lambda r:-r['selection_score']):
        if not r['selected']:continue
        ph=int(r['statistics']['dhash'],16)
        duplicate=None
        for old_hash,old in hashes:
            if bin(ph^old_hash).count('1')<=8:duplicate=old;break
        if duplicate:
            r['selected']=False;r['rejection_reasons'].append('near_duplicate');r['duplicate_of']=duplicate
        else:accepted.append(r);hashes.append((ph,r['id']))
    prefix='partial_' if args.from_progress else ''
    save_json(CORPUS/(prefix+'screening.json'),dict(records=scored,selected_files=len(accepted),selected_page_groups=len({r['group'] for r in accepted}),
              clip_weights='existing local ViT-B-32; only style screening, not stroke extraction',
              counting_policy='Downloaded images include whole works and same-work details; never called independent paintings.'))
    save_json(CORPUS/(prefix+'selected_sources.json'),dict(records=accepted,selected_files=len(accepted),page_groups=len({r['group'] for r in accepted})))
    previews=CORPUS/'previews';previews.mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    for page,start in enumerate(range(0,min(len(accepted),240),36)):
        chosen=accepted[start:start+36];canvas=Image.new('RGB',(1800,1500),(245,243,235));draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(chosen):
            x=(i%6)*300;y=(i//6)*250
            im=Image.open(r['path']).convert('RGB');im.thumbnail((280,205));canvas.paste(im,(x+10+(280-im.width)//2,y+5+(205-im.height)//2))
            draw.text((x+6,y+212),r['id'][-12:],font=font,fill=(25,40,30));draw.text((x+6,y+233),r['title'][:15],font=font,fill=(25,40,30))
        canvas.save(previews/(prefix+f'selected_{page:02d}.jpg'),quality=92)
    print('SCREENING_COMPLETE',len(accepted),'selected photos',len({r['group'] for r in accepted}),'parent page groups',flush=True)
