"""Curated rock ROIs, paper-normalized weak ink masks and paired degradation.

All variants inherit their parent tutorial group and split. Masks are automatic
weak targets, NEVER expert outlines or semantic annotations of a cunfa type.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .runtime import LAB
from .cunfa_collect import CORPUS,sheet

IDS=['43d531f6-6d46-4c55-bb58-dff7d0c5ea8f','737ddb08-88f9-475f-9afa-62741f0dea83',
 '9b1e34a9-fb08-4fda-af82-cd47335ffa0d','d8f6b6bc-036a-4b96-a268-3928b21a196f',
 '8e9e007b-1fe0-4958-b964-29c6447ea2b5','0446d35b-4315-49b6-83a4-2f7243b71213',
 '470ba285-9644-4954-b2ae-d009de9a738d','664da1fa-19c7-42e3-a98d-6bb7669affe7',
 '99550c74-af76-4cc8-98b0-22582f438586','e277f395-93f3-4859-b529-cee755986bdb',
 '9470ec32-7e2f-43c0-b011-1a564c484188','660c82f0-2062-4b0a-8e07-d49322e2c67a',
 'e5de3d57-78a7-4266-b623-37d9d10e43f1','a5c67d0e-6e70-4c00-9a7f-23d8d53f9c66',
 '97cd57ac-3e46-4491-aec3-32b5abb7738d','5ac08e0e-fa7d-4af7-a21e-d4d966d58d6c',
 '799ad7b9-00f0-4e56-8813-fcd7524b38b8']
# Rectangles are reviewed drawing regions only. Explanatory prose is excluded.
ROIS={
 1:[(165,530,585,864)],2:[(90,510,552,870)],3:[(120,515,530,825)],
 4:[(163,510,598,852)],5:[(165,485,616,825)],
 6:[(40,32,222,144),(259,25,438,140),(35,154,228,290),(267,147,451,285),
    (39,316,226,448),(278,300,443,445),(32,466,226,609),(271,466,450,626)],
 7:[(5,5,485,717),(487,250,638,710),(202,754,515,851)],
 8:[(78,90,565,379)],
 9:[(10,0,390,235),(421,0,788,250),(15,269,416,531),(375,316,794,634),
    (12,603,386,829),(486,724,771,1052),(5,884,425,1150),(8,1180,352,1464),(401,1116,798,1402)],
 10:[(5,15,480,399)],
 11:[(62,92,146,160),(163,127,258,194),(319,23,544,153),(10,203,178,294),
     (384,166,623,331),(110,289,288,376),(13,389,224,479),(355,328,629,464),
     (92,492,270,593),(311,464,628,596)],
 12:[(4,129,281,354),(331,143,652,354),(5,420,308,603),(346,423,652,617),
     (4,667,317,869),(333,707,654,868)],
 13:[(15,51,285,328)],
 14:[(87,186,362,347),(328,116,602,264),(5,419,282,596),(309,441,541,627)],
 15:[(25,4,144,85),(163,3,297,85),(301,2,468,98),(30,119,244,224),(246,115,471,233),
     (9,245,449,420)],
 16:[(30,24,455,332),(32,357,456,664)],
 17:[(40,12,578,458),(43,487,572,789)]}

def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def ink_target(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    gray=cv2.bilateralFilter(gray,5,13,3)
    # Large closing estimates local paper, retaining dark ink inside broad marks.
    k=max(21,min(71,(min(gray.shape)//7)|1))
    bg=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
    bg=cv2.GaussianBlur(bg,(0,0),1.3)
    contrast=np.maximum(bg.astype(np.float32)-gray,0)/np.maximum(bg,50)
    hi=gray.astype(np.float32)-cv2.GaussianBlur(gray,(0,0),.7).astype(np.float32)
    paper=gray>=np.percentile(gray,60)
    noise=float(np.median(np.abs(hi[paper])))/max(float(np.median(gray[paper])),50)
    floor=max(.035,min(.12,3.2*noise+.018))
    density=np.clip((contrast-floor)/(.76-floor),0,1)
    # Colored labels/highlights are not ink annotations. Ignore occlusions.
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    colored=((hsv[:,:,1]>95)&(hsv[:,:,2]>105)).astype(np.uint8)
    colored=cv2.dilate(colored,np.ones((5,5),np.uint8))>0
    density[colored]=0
    # Reject paper-grain specks, but retain small real strokes and drybrush gaps.
    n,cc,stats,_=cv2.connectedComponentsWithStats((density>.12).astype(np.uint8),8)
    valid=np.zeros(gray.shape,np.uint8)
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if area>=9 and max(w,h)>=6:
            local=density[y:y+h,x:x+w][cc[y:y+h,x:x+w]==i]
            if local.max()>.33 or max(w,h)>=24:valid[cc==i]=1
    valid=cv2.dilate(valid,np.ones((3,3),np.uint8))
    density*=valid
    # Reject disconnected paper blotches outside coherent ink groups; preserve
    # the groups' interior grayscale strokes without filling their white holes.
    seed=cv2.dilate((density>.25).astype(np.uint8),np.ones((7,7),np.uint8))
    n,cc,stats,_=cv2.connectedComponentsWithStats(seed,8)
    if n>1:
        main_area=max(stats[1:,cv2.CC_STAT_AREA]);region=np.zeros(gray.shape,np.uint8)
        for i in range(1,n):
            if stats[i,cv2.CC_STAT_AREA]>=max(90,main_area*.06):
                ys,xs=np.where(cc==i)
                hull=cv2.convexHull(np.column_stack((xs,ys)).astype(np.int32))
                cv2.fillConvexPoly(region,hull,1)
        region=cv2.dilate(region,np.ones((11,11),np.uint8))
        density*=region
    coverage=np.clip(density*3,0,1)
    targets=np.stack([coverage,density])
    weights=np.full_like(targets,.8)
    # Extremely faint ink vs paper is ambiguous and receives lower confidence.
    ambiguous=(contrast>floor*.6)&(contrast<floor*1.2)
    weights[:,ambiguous]=.18;weights[:,colored]=0
    return np.rint(targets*255).astype(np.uint8),np.rint(weights*255).astype(np.uint8)

def phash(rgb):
    gray=cv2.cvtColor(cv2.resize(rgb,(32,32)),cv2.COLOR_RGB2GRAY).astype(np.float32)
    d=cv2.dct(gray)[:8,:8].flatten()[1:];return d>np.median(d)

def web_rois(r):
    """Reviewed source selection. No advertisements, avatars, full landscapes."""
    pi=int(r['id'].split('_')[1]);ii=int(r['id'].split('_')[2])
    if pi==13:return [(0.,0.,1.,1.)]
    if pi==14:
        if ii not in (8,9,10,11,12,13,14,15,16):return []
        return [(0.,0.,1.,1.)]
    if 15<=pi<=24:
        # Drop tiny embedded heading / corner footer, leaving the actual drawing.
        return [(.02,.10,.98,.93)]
    if pi==27 and ii in (14,23):return [(.02,.02,.98,.95)]
    if pi==28:return [(.02,.09,.98,.91)]
    if pi==29:
        # Explicit ROIs keep written calligraphy and broad colored washes out.
        boxes={
          0:[(.10,.02,.48,.63),(.47,.36,.96,.94)],1:[(.60,.01,.96,.60),(.02,.42,.60,.98)],
          2:[(.23,.05,.98,.96)],3:[(.0,.04,.53,.68),(.50,.50,.98,.96)],
          5:[(.22,.02,.98,.96)],6:[(.20,.02,.96,.96)],7:[(.08,.02,.94,.95)],
          8:[(.10,.03,.47,.47),(.50,.02,.98,.49),(.0,.46,.55,.98)],
          9:[(.25,.03,.98,.96)],11:[(.20,.02,.98,.96)],
          12:[(.12,.03,.50,.45),(.50,.02,.96,.50),(.33,.43,.95,.98)],
          13:[(.20,.05,.83,.96)],14:[(.20,.04,.94,.96)],
          17:[(.10,.01,.51,.51),(.60,.44,.99,.95),(.03,.50,.50,.96)],
          18:[(.05,.55,.48,.96),(.60,.05,.98,.55)],
          25:[(.01,.01,.48,.48),(.35,.47,.98,.98)],
          29:[(.08,.04,.55,.48),(.30,.54,.94,.96)],
          31:[(.49,.02,.99,.48),(.28,.50,.98,.96)],
          32:[(.55,.02,.99,.48),(.02,.38,.94,.96)],
          34:[(.48,.03,.97,.49),(.22,.46,.95,.94)],
          36:[(.31,.06,.97,.95)],37:[(.18,.04,.69,.47),(.25,.46,.97,.96)],
          38:[(.08,.08,.98,.84)],39:[(.08,.08,.98,.84)],40:[(.08,.08,.98,.84)],
          41:[(.08,.08,.98,.84)],42:[(.08,.08,.98,.84)],
          43:[(.23,.01,.95,.92)],46:[(.05,.06,.97,.92)],51:[(.10,.03,.97,.94)],
        }
        return boxes.get(ii,[])
    return []

def group_for_user(n):
    if n<=5:return 'tutorial_five_stages'
    if n in (6,10):return 'cunfa_eight_methods'
    if n==7:return 'axe_rock_faces'
    if n==8:return 'annotated_angular_rocks'
    if n==9:return 'rock_line_album'
    if n==11:return 'isolated_gray_rock_album'
    if n in (12,14,15,16):return 'basic_rock_combinations'
    if n==13:return 'simple_tall_rock'
    return 'two_rocks_paper_photo'

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--variants',type=int,default=6);args=parser.parse_args(argv)
    for name in ('crops','labels','pairs','references','review'):(CORPUS/name).mkdir(exist_ok=True)
    candidates=[];reference_audit=[]
    for n,identity in enumerate(IDS,1):
        src=Path('C:/Users/Lenovo/AppData/Local/Temp')/f'codex-clipboard-{identity}.png'
        dest=CORPUS/'references'/f'user_{n:02d}.png';shutil.copy2(src,dest)
        reference_audit.append(dict(number=n,source=str(src),copy=str(dest),sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
        rgb=np.asarray(Image.open(dest).convert('RGB'))
        for ri,box in enumerate(ROIS[n]):
            x1,y1,x2,y2=box
            candidates.append(dict(id=f'ref_{n:02d}_{ri:02d}',rgb=rgb[y1:y2,x1:x2].copy(),group=group_for_user(n),
                                   parent_image=str(dest),source_type='user_reference',roi=list(box),source_page=None))
    web=json.loads((CORPUS/'web_sources.json').read_text(encoding='utf-8'))
    for r in web:
        for ri,box in enumerate(web_rois(r)):
            im=Image.open(r['image']).convert('RGB');w,h=im.size
            crop=im.crop(tuple(round(x*(w if i%2==0 else h)) for i,x in enumerate(box)))
            crop.thumbnail((900,900));rgb=np.asarray(crop)
            pi=int(r['id'].split('_')[1])
            group='tutorial_cunfa_step_book' if pi in list(range(13,14))+list(range(15,25))+[28] else 'historic_stone_methods_book' if pi==29 else 'tutorial_five_stages' if pi==14 else 'historic_cunfa_examples'
            candidates.append(dict(id=r['id']+f'_c{ri}',rgb=rgb,group=group,parent_image=r['image'],
                                   source_type='public_web_tutorial',roi=list(box),source_page=r['source_page']))
    # Exact and conservative perceptual near duplicates merge source groups.
    parents=list(range(len(candidates)))
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    group_first={};hashes=[]
    for i,r in enumerate(candidates):
        if r['group'] in group_first:parents[find(i)]=find(group_first[r['group']])
        else:group_first[r['group']]=i
        hashes.append(phash(r['rgb']))
    duplicate_pairs=[]
    for i in range(len(candidates)):
        for j in range(i):
            if np.count_nonzero(hashes[i]!=hashes[j])<=4:
                parents[find(i)]=find(j);duplicate_pairs.append((candidates[i]['id'],candidates[j]['id']))
    # Known cross-posts are conservatively one tutorial family, even at different resolutions.
    components={}
    for i,r in enumerate(candidates):components.setdefault(find(i),set()).add(r['group'])
    held_test={'isolated_gray_rock_album','simple_tall_rock','two_rocks_paper_photo'}
    held_val={'axe_rock_faces','annotated_angular_rocks'}
    real=[];seen=set();rng=np.random.default_rng(462031)
    for i,r in enumerate(candidates):
        rgb=r.pop('rgb');digest=hashlib.sha256(rgb.tobytes()).hexdigest()
        if digest in seen:continue
        seen.add(digest)
        g=components[find(i)];r['group']='+'.join(sorted(g))
        r['split']='test' if g&held_test else 'validation' if g&held_val else 'train'
        path=CORPUS/'crops'/(r['id']+'.png');Image.fromarray(rgb).save(path)
        targets,weights=ink_target(rgb)
        labels=CORPUS/'labels'/(r['id']+'.npz');np.savez_compressed(labels,targets=targets,weights=weights)
        r.update(image=str(path),labels=str(labels),sha256=digest,label_type='automatic paper-normalized ink weak supervision; includes cunfa')
        real.append(r)
    # This photographed six-panel sheet retains visible paper blotches after
    # normalization. Preserve the original, but do not teach those as ink.
    quality_rejected=[r for r in real if r['id'].startswith('ref_12_')]
    real=[r for r in real if not r['id'].startswith('ref_12_')]
    # Paired photometric restorations use a known target derived from each weak
    # mask. These are variants of the SAME examples, not independent artworks.
    pairs=[]
    for r in real:
        if r['split']=='test':continue
        with np.load(r['labels']) as f:target=f['targets'].copy()
        density=target[1].astype(np.float32)/255.;h,w=density.shape
        for variant in range(args.variants if r['split']=='train' else 2):
            paper=np.ones((h,w,3),np.float32)*rng.uniform(207,254,(1,1,3))
            coarse=cv2.resize(rng.normal(0,1,(8,8)).astype(np.float32),(w,h),interpolation=cv2.INTER_CUBIC)*rng.uniform(2,11)
            paper+=coarse[:,:,None]
            paper+=rng.normal(0,rng.uniform(.3,4),(h,w,1))
            if variant%3==1:
                # Realistic paper fibers, absent from ink targets.
                fibers=rng.normal(0,1,(h,max(5,w//12))).astype(np.float32)
                paper+=cv2.resize(fibers,(w,h),interpolation=cv2.INTER_LINEAR)[:,:,None]*4
            opacity=density*rng.uniform(.65,1.)
            ink_color=rng.uniform(0,40,(1,1,3))
            image=paper*(1-opacity[:,:,None])+ink_color*opacity[:,:,None]
            image=np.clip(image,0,255).astype(np.uint8)
            identity=r['id']+f'_pair{variant:02d}';image_path=CORPUS/'pairs'/(identity+'.jpg')
            Image.fromarray(image).save(image_path,quality=int(rng.integers(75,97)))
            pairs.append(dict(id=identity,image=str(image_path),labels=r['labels'],group=r['group'],split=r['split'],parent_id=r['id'],
                              provenance='Degraded render from the parent automatic weak ink mask; not a new artwork'))
    manifest=dict(version=4,task='rock_cunfa_ink_restoration',real=real,synthetic=pairs,
                  channel_names=['all_ink_coverage_including_cunfa','continuous_ink_density'],
                  human_dense_annotation_count=0,inference_records=[r for r in real if r['split']!='train' or r['id'].startswith('ref_')],
                  notes=['All descendants inherit tutorial-group split.','No semantic cunfa-category accuracy available.',
                         'Automatic weak masks are a baseline limitation; additional images alone cannot repair their mistakes.'])
    manifest_path=LAB/'datasets'/'manifest_rock_cunfa_v04.json';dump(manifest_path,manifest)
    audit=dict(input_user_sheets=len(IDS),downloaded_unique_web_images=len(web),curated_examples=len(real),
               curated_web_examples=sum(r['source_type']=='public_web_tutorial' for r in real),
               curated_user_examples=sum(r['source_type']=='user_reference' for r in real),
               used_web_parent_images=len({r['parent_image'] for r in real if r['source_type']=='public_web_tutorial'}),
               examples_per_split=dict(Counter(r['split'] for r in real)),
               groups_per_split={s:len({r['group'] for r in real if r['split']==s}) for s in ('train','validation','test')},
               paired_variants_per_split=dict(Counter(r['split'] for r in pairs)),near_duplicate_pairs=duplicate_pairs,
               label_provenance='Automatically normalized ink masks, NOT expert traced contours/cunfa labels',
               quality_excluded_examples=[dict(id=r['id'],reason='Paper blotches remain in automatic target') for r in quality_rejected],
               references=reference_audit,manifest=str(manifest_path))
    for a in real:
        for b in real:
            assert a['group']!=b['group'] or a['split']==b['split'],'Source-group leakage'
    dump(CORPUS/'dataset_audit.json',audit);sheet(real,'curated')
    # Actual target previews: never substitute these for learned predictions.
    examples=[r for r in real if r['id'] in ('ref_02_00','ref_06_00','ref_11_04','ref_12_00','ref_16_01','ref_17_00')]
    canvas=Image.new('RGB',(1200,len(examples)*275),'#f3f3ef');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    for i,r in enumerate(examples):
        with np.load(r['labels']) as f:alpha=f['targets'][1]
        for j,im in enumerate((Image.open(r['image']).convert('RGB'),Image.fromarray(255-alpha).convert('RGB'))):
            im.thumbnail((565,235));canvas.paste(im,(j*600+(565-im.width)//2,i*275+35))
        draw.text((15,i*275+4),r['id']+' | '+r['split']+' | 输入 / 自动弱标签（非模型预测）',font=font,fill='black')
    canvas.save(CORPUS/'review'/'weak_targets.jpg',quality=94)
    print('PREPARED',json.dumps({k:v for k,v in audit.items() if k not in ('references','near_duplicate_pairs')},ensure_ascii=True),flush=True)
