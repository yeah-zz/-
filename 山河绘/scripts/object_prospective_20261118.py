# -*- coding: utf-8 -*-
"""Prospective fixed-model work-group check; coarse source labels are not expert truth."""
import os,sys,json,hashlib,re,time,concurrent.futures
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import requests,cv2,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from gongbi_training.source_scene_mapping import source_to_scene
from gongbi_training.scene_objects import SceneObjects
from gongbi_training.global_detail_gate import GlobalDetailGate
from global_detail_gate_20261118 import make_input,restore
ROOT=LAB/'object_prospective_20261118';BASE=LAB/'guohua_scene4_mapping_fix_20261116';GATE=LAB/'global_detail_gate_20261118'
ARMS=['global256','detail512','fixed_mean','learned_gate']
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pixels(path):
    im=Image.open(path).convert('RGB');return hashlib.sha256(str(im.size).encode()+np.array(im).tobytes()).hexdigest()
def write(name,v):(ROOT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def prepare():
    ROOT.mkdir(exist_ok=True);assert not (ROOT/'protocol.json').exists(),'Protocol already frozen'
    listing=read(LAB/'guohua_objects_v2_20261113/full_file_listings.json');allrows=[r for v in listing['listings'].values() for r in v if r.get('path','').endswith('.jpg')]
    # Names chosen only from metadata, before retrieving pixels/masks or predicting.
    chosen=[]
    for artist in ['wangximen','zouzhe']:
        rows=sorted([r for r in allrows if Path(r['path']).name.startswith(artist+'_')],key=lambda r:r['path'])
        groups={re.sub(r'_\d+$','',Path(r['path']).stem) for r in rows}
        # This prevents silently mixing different named works into one group.
        group=sorted(groups)[0];rows=[r for r in rows if re.sub(r'_\d+$','',Path(r['path']).stem)==group]
        assert len(rows)>=4
        for i in [0,round((len(rows)-1)/3),round(2*(len(rows)-1)/3),len(rows)-1]:
            r=rows[i];chosen.append(dict(id=Path(r['path']).stem,work_group=group,filename_author_candidate=artist,
                dataset_path=r['path'],expected_sha256=r.get('lfs',{}).get('oid'),file_size=r['size']))
    # Conservative local scan: previously downloaded material with either named prefix excludes selection.
    seen=[]
    for folder,dirs,files in os.walk(LAB):
        dirs[:]=[d for d in dirs if not d.startswith('browser_profile') and d not in {'cache','temp','.git'}]
        for name in files:
            if name.startswith(('wangximen_','zouzhe_')) and name.endswith(('.jpg','.png','.npz')):seen.append(str(Path(folder)/name))
    assert not seen,'Selected named group already present locally'
    write('selection_before_download.json',dict(records=chosen,revision=listing['revision'],
        selection='Two named metadata candidates, lexicographically first work per candidate, four evenly spaced entries per work',
        preexisting_matching_files=seen,images_previewed=False,predictions_run=False,
        author_identity_verified=False,same_public_source_as_training=True))
    write('protocol.json',dict(date='2026-11-18',selection_sha256=sha(ROOT/'selection_before_download.json'),
        checkpoint=str(BASE/'mapping_fixed.pt'),checkpoint_sha256=sha(BASE/'mapping_fixed.pt'),gate=str(GATE/'gate.pt'),gate_sha256=sha(GATE/'gate.pt'),
        script_sha256=sha(__file__),arms=ARMS,training=False,thresholds_or_model_selection=False,
        evaluate='All8selected crops; source coarse4class masks; aggregate by work plus overall; neither8independent works nor expert acceptance',
        labels='Original source labels remapped by existing frozen mapping; unknown0/255 ignored',
        reference_freeze_required_before_inference=True,artwork_attribution_and_cross_collection_lineage_unverified=True,
        robot_connected=False))
    print('FROZEN',[(r['id'],r['work_group']) for r in chosen],flush=True)
def download():
    selection=read(ROOT/'selection_before_download.json');assert sha(ROOT/'selection_before_download.json')==read(ROOT/'protocol.json')['selection_sha256']
    (ROOT/'raw').mkdir(exist_ok=True)
    def fetch(r):
        row=dict(r)
        for kind,path in [('image',r['dataset_path']),('mask',r['dataset_path'].replace('/images/','/annotations/').replace('.jpg','.png'))]:
            dest=ROOT/'raw'/(r['id']+('_mask.png' if kind=='mask' else '.jpg'))
            url=f'https://huggingface.co/datasets/Rane7/CLP_dataset/resolve/{selection["revision"]}/{path}'
            if not dest.exists():
                response=requests.get(url,timeout=40);response.raise_for_status();dest.write_bytes(response.content)
            row[kind]=str(dest);row[kind+'_url']=url;row[kind+'_sha256']=sha(dest)
        if r['expected_sha256']:assert row['image_sha256']==r['expected_sha256']
        im=Image.open(row['image']).convert('RGB');mask=np.array(Image.open(row['mask']));assert mask.shape==np.array(im).shape[:2]
        assert set(np.unique(mask))<=set(range(15))|{255}
        row['size']=list(im.size);row['decoded_pixel_sha256']=pixels(row['image']);return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,selection['records']))
    oldpaths={r['image'] for r in read(BASE/'dataset_manifest.json')['records']}
    refs=read(BASE/'references.json');oldpaths|={r['image'] for r in refs['photo_cases']+refs['video_cases']}
    oldpaths|={r['image'] for r in read(LAB/'object_scale_pairs_20261118/results.json')['records']}
    oldhash={pixels(p):p for p in oldpaths};duplicates=[r['id'] for r in rows if r['decoded_pixel_sha256'] in oldhash]
    assert not duplicates,'Duplicate with known object-task material'
    assert len({r['decoded_pixel_sha256'] for r in rows})==len(rows)
    write('dataset_manifest.json',dict(records=rows,duplicate_pixels_with_known_object_material=duplicates,
        compared_known_images=len(oldpaths),full_work_or_author_independence_externally_verified=False))
    board=Image.new('RGB',(1600,850),'#f5f2e9');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    for i,r in enumerate(rows):
        im=Image.open(r['image']).convert('RGB');im.thumbnail((390,370));x=i%4*400;y=i//4*425;board.paste(im,(x+5,y+40));d.text((x+6,y+6),r['id'],font=font,fill='#243f35')
    board.save(ROOT/'source_contact.jpg',quality=95);print('DOWNLOADED',len(rows),flush=True)
def evaluate():
    p=read(ROOT/'protocol.json');assert not (ROOT/'results.json').exists(),'No reevaluation/tuning';assert sha(__file__)==p['script_sha256']
    assert sha(p['checkpoint'])==p['checkpoint_sha256'] and sha(p['gate'])==p['gate_sha256']
    freeze=read(ROOT/'reference_freeze.json');assert sha(ROOT/'local_references.json')==freeze['reference_sha256']
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    net=SceneObjects().cuda();net.load_state_dict(torch.load(p['checkpoint'],map_location='cuda',weights_only=False)['model']);net.eval();net.requires_grad_(False)
    gate=GlobalDetailGate().cuda();gate.load_state_dict(torch.load(p['gate'],map_location='cuda',weights_only=False)['model']);gate.eval();gate.requires_grad_(False)
    (ROOT/'predictions').mkdir(exist_ok=True);results=[]
    for r in read(ROOT/'dataset_manifest.json')['records']:
        image=np.array(Image.open(r['image']).convert('RGB'));target=source_to_scene(np.array(Image.open(r['mask'])));features,geometry,native=make_input(net,image)
        with torch.inference_mode():logp,_=gate(torch.from_numpy(features)[None].cuda())
        probs=[native[0],native[1],restore((features[3:7]+features[7:11])*.5,geometry),restore(logp.exp()[0].cpu().numpy(),geometry)]
        confusions={};local={}
        for arm,prob in zip(ARMS,probs):
            pred=prob.argmax(0).astype('uint8');np.savez_compressed(ROOT/f'predictions/{r["id"]}_{arm}.npz',prediction=pred)
            valid=target!=255;c=np.bincount((target[valid].astype('int64')*4+pred[valid]).ravel(),minlength=16).reshape(4,4);confusions[arm]=c.tolist()
            local[arm]=[]
            for ref in read(ROOT/'local_references.json')['records']:
                if ref['image_id']!=r['id'] or ref['object_type']=='unknown':continue
                x0,y0,x1,y1=ref['box'];m=pred[y0:y1,x0:x1];cls=1 if ref['object_type']=='rock' else 2
                local[arm].append(dict(id=ref['id'],target=ref['object_type'],correct_fraction=float((m==cls).mean()),regional_context_only=True))
        results.append(dict(id=r['id'],image=r['image'],work_group=r['work_group'],confusions=confusions,local_references=local));print('EVAL',r['id'],flush=True)
    aggregates={}
    for group in ['all']+sorted({r['work_group'] for r in results}):
        aggregates[group]={}
        for arm in ARMS:
            c=sum(np.array(r['confusions'][arm]) for r in results if group=='all' or r['work_group']==group);u=c.sum(0)+c.sum(1)-c.diagonal();iou=[float(c[k,k]/u[k]) if u[k] else None for k in range(4)]
            aggregates[group][arm]=dict(confusion=c.tolist(),support=c.sum(1).tolist(),iou=iou,mean_iou=float(np.mean([v for v in iou if v is not None])))
    write('results.json',dict(records=results,aggregates=aggregates,new_work_candidate_groups=2,new_images=8,
        prospective_after_model_freeze=True,expert_accuracy=False,independent_source=False,author_independence_verified=False,
        acceptance_passed=False,adopted=False,new_model_trained=False,robot_connected=False));print(json.dumps(aggregates),flush=True)
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--download' in sys.argv:download()
    else:evaluate()
