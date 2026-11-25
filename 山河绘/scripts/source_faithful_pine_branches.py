"""Replace invented upper forks with the original image's actual ink network."""
from pathlib import Path
import sys,json,copy,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw,ImageFont
from original_d_tracer import LandscapeBrushGenerator
from painting_session import save
from web.catalog import ARTWORKS
from artwork_order import ink_payload
from ink_control import refresh_stats
from scripts.consolidate_selected_artworks import digest,stroke_data
from scripts.source_aligned_washes import dense_path,points_inside
from scripts.source_wash_comparison import render

TREE=[(115,360),(235,333),(346,369),(450,418),(574,468),(730,565),(738,625),(624,638),(477,587),(359,550),(270,554),(205,599),(173,630),(150,625),(180,566),(227,511),(208,477),(102,462),(60,455),(60,405)]


def merge(paths,mask):
    """Join only near collinear ends supported by the same original ink."""
    support=cv2.dilate(mask,np.ones((3,3),np.uint8))
    while len(paths)>1:
        endpoints=np.vstack([p[[0,-1]] for p in paths]);candidates=[]
        for u,v in cKDTree(endpoints).query_pairs(2.5):
            i,j=u//2,v//2
            if i==j:continue
            a=paths[i][::-1] if u%2==0 else paths[i];b=paths[j][::-1] if v%2 else paths[j]
            ta=a[-1]-a[max(0,len(a)-3)];tb=b[min(2,len(b)-1)]-b[0]
            cosine=np.dot(ta,tb)/max(1e-9,np.linalg.norm(ta)*np.linalg.norm(tb))
            if cosine<.65:continue
            connector=dense_path([a[-1],b[0]],.25)
            if not np.all(points_inside(connector,support)):continue
            candidates.append((np.linalg.norm(a[-1]-b[0])+2*(1-cosine),i,j,a,b))
        used=set();added=[]
        for _,i,j,a,b in sorted(candidates,key=lambda x:x[0]):
            if i in used or j in used:continue
            added.append(np.vstack([a,b[1:] if np.linalg.norm(a[-1]-b[0])<1e-6 else b]));used.update((i,j))
        if not used:break
        paths=[p for i,p in enumerate(paths) if i not in used]+added
    return paths


def audit(candidate,baseline):
    candidate=Path(candidate);old=json.loads(Path(baseline).read_text(encoding='utf-8'));new=json.loads(candidate.read_text(encoding='utf-8'))
    assert stroke_data(new)['color']==stroke_data(old)['color']
    outside=lambda b:[s for s in stroke_data(b)['black'] if s['drawing_region']!='整棵松树']
    assert outside(new)==outside(old),'松树之外的黑线改变'
    ink=stroke_data(new)['black'];assert not any(s['id'].startswith(('pine-fine-fork','simple-fork','pine-fine-bark')) for s in ink)
    tr=new['workflow']['source_revision']['source_transform'];r=tr['scale'];off=np.array(tr['offset']);scale=new['ink_job']['meta']['ink_execution']['mm_per_px']
    masks=np.load(candidate.parent/'原图松枝提线掩膜.npz');support=masks['source_support'];samples=0
    branches=[s for s in ink if s.get('recipe')=='original-pine-ink-network']
    assert branches
    for s in branches:
        local=(dense_path(s['points'],.05/scale)-off)/r
        assert np.all(points_inside(local,support)),'新枝条偏离原图墨线'
        assert set(s['press'])=={.15} and s['length_mm']<=45.001
        samples+=len(local)
    assert new['workflow']['source_revision']['stroke_data_sha256']==digest(stroke_data(new))
    from web.execution import prepare
    from run_json_painting import load_artwork
    prepare(candidate);load_artwork(candidate)
    return dict(robot_connected=False,source_traced_branch_strokes=len(branches),no_invented_forks=True,
        source_ink_support_tolerance_px=2,dense_source_contact_samples_checked=samples,
        branch_pressure_mm=.15,other_black_and_all_color_strokes_preserved=True,
        black_strokes=sum(s['role']!='title' for s in ink),total_strokes=len(ink)+len(stroke_data(new)['color']),runtime_validation_passed=True)


def main():
    path=ROOT/ARTWORKS[0][3];package=path.parent.parent;source=package/'松叶简化版原图.png'
    old=json.loads(path.read_text(encoding='utf-8'));new=copy.deepcopy(old)
    from web.execution import prepare
    prepare(path)
    tr=old['workflow']['source_revision']['source_transform'];r=tr['scale'];off=np.array(tr['offset']);scale=old['ink_job']['meta']['ink_execution']['mm_per_px'];pixel_mm=scale*r
    image=cv2.imdecode(np.fromfile(str(source),np.uint8),1);gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);chroma=np.ptp(image.astype(float),axis=2)
    roi=np.zeros(gray.shape,np.uint8);cv2.fillPoly(roi,[np.array(TREE,np.int32)],1)
    b,g,_=image.astype(float).transpose(2,0,1)
    mask=((gray<130)&(chroma<75)&(b>=g-7)&(roi>0)).astype(np.uint8)*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    skeleton=LandscapeBrushGenerator._thin_binary(mask)
    paths=[]
    for line in LandscapeBrushGenerator._trace_skeleton_paths(skeleton):
        p=np.array(line,float);length=np.linalg.norm(np.diff(p,axis=0),axis=1).sum()*pixel_mm
        if length<.1:continue
        if np.linalg.norm(p[-1]-p[0])<2 and np.linalg.norm(np.ptp(p,axis=0))*pixel_mm<1.2:continue
        paths.append(p)
    paths=merge(paths,mask)
    paths=[p for p in paths if np.linalg.norm(np.diff(p,axis=0),axis=1).sum()*pixel_mm>=1.5]
    template=copy.deepcopy(next(s for s in stroke_data(old)['black'] if s['role']=='branch'))
    root=[]
    for prior in stroke_data(old)['black']:
        if prior['drawing_region']!='整棵松树':continue
        if prior['role']=='bark_texture' and not prior['id'].startswith('pine-fine-bark'):root.append(copy.deepcopy(prior));continue
        if prior['role']!='trunk':continue
        local=(np.array(prior['points'])-off)/r;valid=local[:,1]>=625
        transitions=np.diff(np.r_[False,valid,False].astype(int));starts=np.where(transitions==1)[0];ends=np.where(transitions==-1)[0]
        for j,(a,b) in enumerate(zip(starts,ends)):
            if b-a<2:continue
            s=copy.deepcopy(prior);s.update(id=prior['id']+'-root-%d'%j,points=prior['points'][a:b],press=prior['press'][a:b],speed=prior['speed'][a:b])
            root.append(refresh_stats(s,scale))
    branches=[]
    support=cv2.dilate(mask,np.ones((5,5),np.uint8))
    for i,p in enumerate(paths):
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]*pixel_mm
        breaks=np.linspace(0,arc[-1],int(np.ceil(arc[-1]/40))+1)
        for j,(a,b) in enumerate(zip(breaks[:-1],breaks[1:])):
            t=np.unique(np.r_[a,np.arange(a,b,.20),b]);local=np.column_stack([np.interp(t,arc,p[:,k]) for k in (0,1)])
            if not np.all(points_inside(dense_path(local,.05/pixel_mm),support)):continue
            q=local*r+off;n=len(q);s=copy.deepcopy(template)
            for key in ('parent_stroke_id','source_edge_index','source_path_ids','segment','segments','branch_parent_id','source_fork_guide'):s.pop(key,None)
            s.update(id='original-pine-network-%03d-%02d'%(i,j),points=q.tolist(),press=[.15]*n,speed=[40.]*n,recipe='original-pine-ink-network',source_path=p.tolist())
            branches.append(refresh_stats(s,scale))
    tree_ink=root+branches;other=[copy.deepcopy(s) for s in stroke_data(old)['black'] if s['drawing_region']!='整棵松树'];ink=tree_ink+other
    new['ink_job']=ink_payload(old['ink_job'],ink);plan=new['workflow'];plan['modules']['outline'].update(strokes=copy.deepcopy(ink),stroke_count=len(ink))
    rev=plan['source_revision'];rev.pop('pine_branch_refinement',None)
    note='松枝按原图重提：撤掉全部人工小分叉，直接从原图真实墨线提取上部枝干网络，保留自然弯曲和分叉位置；上部枝条下压0.15mm，保留根部主干及原3条树皮纹理；其他黑线及所有上色路径保持不变。'
    rev.update(source_faithful_pine_network=True,source_pine_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),after_counts=dict(black=len(ink),color=len(stroke_data(new)['color'])))
    rev['note']+=' '+note;plan['note']=rev['note'];rev['stroke_data_sha256']=digest(stroke_data(new))
    pending=path.parent/'待应用_原图松枝';pending.mkdir(exist_ok=True);save(pending/'painting.json',new);np.savez_compressed(pending/'原图松枝提线掩膜.npz',source_ink=mask,source_support=support,skeleton=skeleton)
    result=audit(pending/'painting.json',path);save(pending/'原图松枝校验.json',result)
    # Same source framing, with color removed so branches remain inspectable.
    crop=(60,330,740,740);actualcrop=(off[0]+crop[0]*r,off[1]+crop[1]*r,off[0]+crop[2]*r,off[1]+crop[3]*r)
    board=Image.new('RGB',(2000,790),'#f7f7f0');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',30)
    blank=copy.deepcopy(new);blank['workflow']['modules']['color']['strokes']=[]
    for i,(pic,label) in enumerate(((Image.open(source).convert('RGB').crop(crop),'原图枝干局部'),(render(blank).crop(actualcrop),'沿原图提取的黑线路径'))):
        left=35+i*1000;draw.text((left,30),label,font=font,fill='#304c3b');pic.thumbnail((930,650),Image.Resampling.LANCZOS);board.paste(pic,(left+(930-pic.width)//2,95+(650-pic.height)//2))
    draw.text((35,745),'只按原图真实墨线生成枝干；右侧线宽为下压模型估计。',font=font,fill='#687562');board.save(pending/'原图松枝与黑线对照.png')
    for name in ('任务预览_线宽色彩为估计.png','黑线加蓝绿_实际路径.png'):render(new).save(pending/name)
    render(blank).save(pending/'纯勾线.png')
    (pending/'原图松枝说明.md').write_text('# 松瀑亭舟 · 原图松枝重提\n\n'+note+'\n\n'+json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    save(pending/'应用记录.json',dict(baseline_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256((pending/'painting.json').read_bytes()).hexdigest(),applied=False))
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
