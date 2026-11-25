"""Make auditable real mountain/rock crops, grouped before splitting."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter,defaultdict
import argparse
import hashlib
import html
import json
import re
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .runtime import LAB
from .narrow_collect import CORPUS,save_json
from .narrow_select import image_statistics
from .simple_rocks import make_simple_sample,simple_weak_targets


def group_information(record):
    page=CORPUS/'pages'/(hashlib.sha256(record['page_url'].encode()).hexdigest()+'.html')
    body=page.read_text(encoding='utf-8')
    found=re.search(r'作品作者[：:]</span>(.*?)</p>',body,re.S)
    artist=html.unescape(re.sub('<[^>]+>','',found.group(1))).strip() if found else ''
    title=record['title']
    if '千里江山' in title:return 'wang_ximeng_qianli_jiangshan','王希孟'
    if '圆明园四十景' in title:return 'yuanmingyuan_40_views_album','唐岱等'
    # All album leaves by a known artist remain together. This is more
    # conservative than guessing that different public pages mean new works.
    if artist and artist not in ['佚名','不详','未知','无款','宋人','明人','清人']:
        return 'artist_'+hashlib.sha256(artist.encode()).hexdigest()[:16],artist
    name=re.search('《([^》]+)》',title)
    if name and len(name.group(1))>4:
        return 'work_'+hashlib.sha256(name.group(1).encode()).hexdigest()[:16],artist
    return record['group'],artist


def crop_proposals(rgb):
    h,w=rgb.shape[:2];proposals=[]
    for size in (320,440,560):
        if min(h,w)<size:continue
        step=round(size*.6)
        xs=sorted(set(list(range(0,w-size+1,step))+[w-size]))
        ys=sorted(set(list(range(0,h-size+1,step))+[h-size]))
        for y in ys:
            for x in xs:
                crop=rgb[y:y+size,x:x+size]
                stat=image_statistics(crop)
                fraction=stat['blue_green_fraction'];edge=stat['edge_density']
                if not .10<=fraction<=.88 or not .006<=edge<=.105 or stat['sharpness']<18:continue
                hsv=cv2.cvtColor(cv2.bilateralFilter(crop,7,38,5),cv2.COLOR_RGB2HSV)
                mask=(((hsv[:,:,0]>34)&(hsv[:,:,0]<135)&(hsv[:,:,1]>45))).astype(np.uint8)
                mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
                # A first-stage rock example should include its crest and a
                # broad lower body, not just a magnified slope or floating leaves.
                if mask[:max(12,size//12)].mean()>.18 or mask[-max(12,size//12):].mean()<.20:continue
                n,_,stats,_=cv2.connectedComponentsWithStats(mask,8)
                largest=stats[1:,cv2.CC_STAT_AREA].max()/size**2 if n>1 else 0.
                if largest<.09:continue
                score=largest-.8*abs(fraction-.4)-2*max(0,edge-.05)
                proposals.append((float(score),[x,y,x+size,y+size],stat))
    chosen=[]
    for score,box,stat in sorted(proposals,key=lambda p:-p[0]):
        x,y,x1,y1=box;size=x1-x
        duplicate=False
        for _,previous,_ in chosen:
            a,b,c,d=previous
            overlap=max(0,min(x1,c)-max(x,a))*max(0,min(y1,d)-max(y,b))
            union=size**2+(c-a)*(d-b)-overlap
            if overlap/max(union,1)>.65:duplicate=True;break
        if not duplicate:chosen.append((score,box,stat))
        if len(chosen)>=7:break
    return chosen


def training_item(job):
    source,group,split,index,box,out=job
    x,y,x1,y1=box
    with Image.open(source['path']) as im:crop=im.convert('RGB').crop((x,y,x1,y1)).resize((320,320),Image.Resampling.LANCZOS)
    rgb=np.asarray(crop)
    target,weight=simple_weak_targets(rgb)
    coverage=float((target[0]>.3).mean())
    if not .001<=coverage<=.075:return None
    key=source['id']+f'_rock_{index:02d}'
    folder=out/key;folder.mkdir(parents=True,exist_ok=True)
    crop.save(folder/'image.png')
    np.savez_compressed(folder/'labels.npz',targets=np.rint(target*255).astype(np.uint8),weights=np.rint(weight*255).astype(np.uint8))
    Image.fromarray(np.rint((1-target[0])*255).astype(np.uint8)).save(folder/'weak_structure.png')
    return dict(id=key,group=group,split=split,image=str(folder/'image.png'),labels=str(folder/'labels.npz'),size=[320,320],
                supervision='automatic_supported_outer_contours_plus_at_most_three_long_folds_NOT_gold',
                crop_box_original_pixels=box,parent_image=source['id'],parent_page_group=source['group'],source=source,
                primary_coverage=coverage,manual_expert_verified=False)


def source_proposals(record):
    with Image.open(record['path']) as im:
        original_size=im.size;im.thumbnail((1200,1200));rgb=np.asarray(im.convert('RGB'))
    scale_x=original_size[0]/rgb.shape[1];scale_y=original_size[1]/rgb.shape[0]
    candidates=[]
    for score,box,stat in crop_proposals(rgb):
        box=[round(box[0]*scale_x),round(box[1]*scale_y),round(box[2]*scale_x),round(box[3]*scale_y)]
        candidates.append((score,box,stat))
    return record,candidates


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--preview-only',action='store_true');parser.add_argument('--max-crops',type=int,default=3000);args=parser.parse_args(argv)
    cv2.setNumThreads(1)
    file=CORPUS/('partial_selected_sources.json' if args.preview_only else 'selected_sources.json')
    selected=json.loads(file.read_text(encoding='utf-8'))['records']
    groups={};artists={}
    for r in selected:
        group,artist=group_information(r);groups[r['group']]=group;artists[group]=artist
    # Merge groups joined by near-duplicate images before deciding splits.
    screening_file=CORPUS/('partial_screening.json' if args.preview_only else 'screening.json')
    screening=json.loads(screening_file.read_text(encoding='utf-8'))['records'];lookup={r['id']:r for r in screening}
    parent={v:v for v in groups.values()}
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:
            # Preserve the known Qianli training group name.
            if b=='wang_ximeng_qianli_jiangshan':a,b=b,a
            parent[b]=a
    for r in screening:
        if r.get('duplicate_of') in lookup:
            previous=lookup[r['duplicate_of']]
            if r['group'] in groups and previous['group'] in groups:union(groups[r['group']],groups[previous['group']])
    groups={key:find(value) for key,value in groups.items()}
    counts=Counter(groups[r['group']] for r in selected)
    eligible=sorted([g for g,n in counts.items() if g!='wang_ximeng_qianli_jiangshan' and n>=2],key=lambda g:hashlib.sha256(('narrow_v03_'+g).encode()).hexdigest())
    nval=max(1,min(5,len(eligible)//6));ntest=max(1,min(3,len(eligible)//10))
    validation=set(eligible[:nval]);test=set(eligible[nval:nval+ntest])
    assert not(validation&test) and 'wang_ximeng_qianli_jiangshan' not in validation|test
    split={g:'validation' if g in validation else 'test' if g in test else 'train' for g in counts}
    out=CORPUS/('preview_crops' if args.preview_only else 'prepared_crops');out.mkdir(exist_ok=True)
    maxcrops=120 if args.preview_only else args.max_crops*3
    # Round-robin across source pages prevents a long scroll with many details
    # from filling the entire crop budget before other paintings are visited.
    by_page=defaultdict(list)
    for r in selected:by_page[r['group']].append(r)
    ordered=[]
    for index in range(max(map(len,by_page.values()),default=0)):
        for page in sorted(by_page):
            if index<len(by_page[page]):ordered.append(by_page[page][index])
    jobs=[];per_page=Counter();source_examples={}
    if args.preview_only:ordered=ordered[:80]
    with ThreadPoolExecutor(max_workers=6) as proposal_pool:
        for number,(r,candidates) in enumerate(proposal_pool.map(source_proposals,ordered)):
            if per_page[r['group']]<45:
                group=groups[r['group']]
                for i,(_,box,_) in enumerate(candidates):
                    if per_page[r['group']]>=45:break
                    jobs.append((r,group,split[group],i,box,out));per_page[r['group']]+=1
                    source_examples.setdefault(group,r)
                    if len(jobs)>=maxcrops:break
            if (number+1)%100==0:print('CROP_CANDIDATES',number+1,'/',len(ordered),'sources',len(jobs),'candidate crops',flush=True)
            if len(jobs)>=maxcrops:break
    # Whole-image style screening does not guarantee a crop contains a rock.
    # Reject tree crowns, buildings, words and texture-only patches explicitly.
    import torch
    from .narrow_select import load_clip
    clip,clip_model,transform=load_clip()
    prompts=[
        'a close-up of a mountain peak or rock in a Chinese blue-green landscape painting',
        'a single Chinese gongbi mountain or stone, clear outlines, blue green mineral colors',
        'trees and leaves in a Chinese painting, foliage and branches',
        'a close-up of blue-green tree crowns with many leaves in a Chinese gongbi painting',
        'Chinese buildings and palace architecture, pavilions, a map of an imperial garden',
        'Chinese calligraphy, writing, seals and blank paper',
        'only rough paper texture, a close-up of silk grain, no mountain shape',
        'flat water with ripples in a Chinese painting, no mountain or rock',
    ]
    with torch.no_grad():
        text=clip_model.encode_text(clip.tokenize(prompts).cuda()).float();text/=text.norm(dim=1,keepdim=True)
        accepted_jobs=[];crop_decisions=[]
        for begin in range(0,len(jobs),48):
            batch=jobs[begin:begin+48];images=[];last_path=None;last_image=None
            for r,_,_,_,box,_ in batch:
                if r['path']!=last_path:
                    with Image.open(r['path']) as im:last_image=im.convert('RGB')
                    last_path=r['path']
                images.append(transform(last_image.crop(tuple(box))))
            features=clip_model.encode_image(torch.stack(images).cuda()).float();features/=features.norm(dim=1,keepdim=True)
            sim=(features@text.T).cpu().numpy()
            for job,scores in zip(batch,sim):
                positive=float(scores[:2].max());negative=float(scores[2:].max())
                keep=positive>.245 and positive-negative>.005
                crop_decisions.append(dict(parent_image=job[0]['id'],box=job[4],rock_score=positive,negative_score=negative,selected=keep))
                if keep:accepted_jobs.append(job)
            if begin%480==0:print('ROCK_CONTENT_SCREEN',min(begin+48,len(jobs)),'/',len(jobs),'kept',len(accepted_jobs),flush=True)
    save_json(CORPUS/('partial_crop_screening.json' if args.preview_only else 'crop_screening.json'),crop_decisions)
    jobs=accepted_jobs
    del clip_model,text,features,last_image
    torch.cuda.empty_cache()
    records=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,result in enumerate(pool.map(training_item,jobs)):
            if result:records.append(result)
            if (i+1)%200==0:print('CROPS_PREPARED',i+1,'/',len(jobs),'valid',len(records),flush=True)
    if not args.preview_only:records=records[:args.max_crops]
    if not args.preview_only:
        surviving=Counter(r['group'] for r in records)
        actual_eligible=sorted([g for g,n in surviving.items() if g!='wang_ximeng_qianli_jiangshan' and n>=2],key=lambda g:hashlib.sha256(('narrow_v03_'+g).encode()).hexdigest())
        if len(actual_eligible)<3:raise RuntimeError('Too few independent surviving groups for training/validation/test')
        nval=max(1,min(5,len(actual_eligible)//6));ntest=max(1,min(3,len(actual_eligible)//10))
        validation=set(actual_eligible[:nval]);test=set(actual_eligible[nval:nval+ntest])
        split={g:'validation' if g in validation else 'test' if g in test else 'train' for g in surviving}
        for r in records:r['split']=split[r['group']]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    for page,start in enumerate(range(0,min(len(records),120),12)):
        canvas=Image.new('RGB',(1920,1260),(246,244,237));draw=ImageDraw.Draw(canvas)
        for i,r in enumerate(records[start:start+12]):
            col=i%3;row=i//3;x=col*640;y=row*315
            draw.text((x+8,y+4),'真实山石裁块 / 自动轮廓弱标签',font=font,fill=(30,40,35))
            for j,key in enumerate(['image.png','weak_structure.png']):
                im=Image.open(Path(r['image']).parent/key);im.thumbnail((280,280));canvas.paste(im,(x+j*310+10,y+32))
        canvas.save(CORPUS/'previews'/f'{"partial_" if args.preview_only else ""}crop_pairs_{page:02d}.jpg',quality=92)
    if args.preview_only:
        print('CROP_PREVIEW_READY',len(records));return
    synthetic=[];synth_folder=CORPUS/'synthetic_simple_rocks';synth_folder.mkdir(exist_ok=True)
    for index in range(1152):
        ss='train' if index<1024 else 'validation';seed=300000+index if ss=='train' else 800000+index
        rgb,target=make_simple_sample(seed)
        prefix=synth_folder/f'{ss}_{seed}'
        Image.fromarray(rgb).save(str(prefix)+'.png');np.savez_compressed(str(prefix)+'.npz',targets=np.rint(target*255).astype(np.uint8))
        synthetic.append(dict(id=prefix.name,image=str(prefix)+'.png',labels=str(prefix)+'.npz',split=ss,seed=seed,supervision='known_simple_rock_procedural_geometry'))
    actual_groups={r['group'] for r in records}
    # Fixed small inference panel, independent of crop count.
    inference=[]
    old=json.loads((LAB/'datasets'/'manifest.json').read_text(encoding='utf-8'))
    for r in old['real']:
        if r['id'] in ['user_qianli_02','user_qianli_05','user_qianli_06']:inference.append(r)
    for ss,limit in [('train',3),('validation',4),('test',3)]:
        chosen_groups=[]
        for r in records:
            if r['split']!=ss or r['group'] in chosen_groups:continue
            inference.append(r);chosen_groups.append(r['group'])
            if len(chosen_groups)>=limit:break
    manifest=dict(real=records,synthetic=synthetic,inference_records=inference,
        training_groups=sorted(g for g in actual_groups if split[g]=='train'),
        validation_groups=sorted(g for g in actual_groups if split[g]=='validation'),
        test_groups=sorted(g for g in actual_groups if split[g]=='test'),split_by_painting_or_album=True,
        grouping_policy='known artist groups + known albums + near-duplicate links; user Qianli all training',
        real_dense_human_labels=0,policy='simple_qinglu_mountain_and_rock_outer_contours_first',
        no_previous_dense_ink_corpus=True,source_files_selected=len(selected),source_page_groups=len(by_page),
        source_photos_used=len({r['parent_image'] for r in records}),crop_counts=dict(Counter(r['split'] for r in records)),
        artists=artists,note='Real photos and same-painting crops are not independent artworks. Automatic targets are not expert ground truth.')
    for a,b in [('training_groups','validation_groups'),('training_groups','test_groups'),('validation_groups','test_groups')]:assert not(set(manifest[a])&set(manifest[b]))
    dest=LAB/'datasets'/'manifest_narrow_v03.json';save_json(dest,manifest)
    save_json(CORPUS/'prepared_summary.json',{k:v for k,v in manifest.items() if k not in ['real','synthetic','inference_records','artists']})
    print('NARROW_DATA_READY',len(records),'real crops',len(synthetic),'simple synthetic pairs',manifest['crop_counts'],flush=True)
