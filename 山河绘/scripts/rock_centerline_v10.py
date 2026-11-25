# -*- coding: utf-8 -*-
"""Ink-conditioned centerline-only experiment. Offline, no robot imports."""
import sys,json,hashlib,math,shutil,time
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=LAB/'rock_centerline_v10_20261024';V07=LAB/'rock_strokes_v07_20261023';V08=LAB/'rock_strokes_v08_20261023';V09=LAB/'rock_multitask_v09_20261024';INK=LAB/'runs'/'mobilenet_unet_rock_cunfa_v06_expanded'
for p in ('review','labels','ink_cache','runs','evaluation','source_snapshot'):(ROOT/p).mkdir(parents=True,exist_ok=True)
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
def write(name,obj):(ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def candidates():
    rows=json.loads((V07/'manifest.json').read_text(encoding='utf-8'))['real'];groups={}
    for r in rows:
        if r['split']!='train' or not r.get('phase'):continue
        im=Image.open(r['image']).convert('RGB');a=np.asarray(im)
        if min(im.size)<170 or max(im.size)>950:continue
        hsv=cv2.cvtColor(a,cv2.COLOR_RGB2HSV)
        if np.median(hsv[:,:,1])>45:continue
        with np.load(r['labels']) as z:fraction=float((z['targets'][0]>80).mean())
        if not .025<fraction<.30:continue
        groups.setdefault(r['group'],[]).append(r)
    picks=[]
    for group in sorted(groups):picks+=groups[group][:3]
    picks=picks[:72];write('candidate_records.json',picks)
    for start in range(0,len(picks),36):
        batch=picks[start:start+36];board=Image.new('RGB',(1500,math.ceil(len(batch)/6)*195),'#f4f4ed');d=ImageDraw.Draw(board)
        for i,r in enumerate(batch):
            p=Image.open(r['image']).convert('RGB');p.thumbnail((240,160));x=i%6*250;y=i//6*195;board.paste(p,(x+(240-p.width)//2,y));d.text((x+4,y+164),str(start+i)+' '+r['id'][-12:],font=FONT,fill='black')
        board.save(ROOT/'review'/f'candidates_{start//36+1}.jpg',quality=93)
    print('CANDIDATES',len(picks),flush=True)

def prepare():
    import torch
    from scipy.interpolate import PchipInterpolator
    from gongbi_training.centerline_annotations_v10 import additions
    from gongbi_training.centerline_only_v10 import geometry_target
    from gongbi_training.cunfa_prepare import ink_target
    from gongbi_training.model import GongbiLineNet
    from gongbi_training.infer import predict
    if any((ROOT/'runs').glob('*/training.jsonl')):raise FileExistsError('Preserve training inputs')
    old=json.loads((V08/'manifest.json').read_text(encoding='utf-8'))['records'];rows=[]
    for r in old:rows.append(dict(id=r['id'],image=r['image'],split=r['split'],source_group='original_'+r['split'] if r['split']!='test' else ('old_book_2022' if r['id']=='test_new_book' else 'old_book_196559761'),paths=[dict(points=p['points'],role='line') for p in r['paths']],provenance='assistant initial visual trace, not expert',complete=False))
    additional=additions(json.loads((ROOT/'candidate_records.json').read_text(encoding='utf-8')),V09)
    new_groups={r['id']:r['source_group'] for r in json.loads((V09/'frozen_manifest.json').read_text(encoding='utf-8'))['new_geometry_test']}
    for r in additional:
        if r['id'] in ('extra_57','extra_59'):r['split']='validation'
        if r['id'] in new_groups:r['source_group']=new_groups[r['id']]
        rows.append(r)
    assert len(rows)==20
    torch.set_num_threads(4);cv2.setNumThreads(2);teacher=GongbiLineNet().cuda();teacher.load_state_dict(torch.load(INK/'best.pt',map_location='cuda',weights_only=False)['model']);teacher.eval();records=[]
    for r in rows:
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));h,w=rgb.shape[:2];center=np.zeros((h,w),np.uint8);angles=np.zeros((2,h,w),np.float32);ids=np.zeros((h,w),np.int32);collision=np.zeros((h,w),np.uint8);dense=[];overlay=rgb.copy()
        for j,stroke in enumerate(r['paths'],1):
            p=np.asarray(stroke['points'],float);s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];unique=np.r_[True,np.diff(s)>.01];p=p[unique];s=s[unique];u=np.linspace(0,s[-1],max(2,math.ceil(s[-1])+1));p=np.stack([PchipInterpolator(s,p[:,k])(u) for k in range(2)],1);p[:,0]=np.clip(p[:,0],0,w-1);p[:,1]=np.clip(p[:,1],0,h-1);xy=np.rint(p).astype(int);x,y=xy.T;v=np.gradient(p,axis=0);theta=np.arctan2(v[:,1],v[:,0]);collision[y,x]|=((ids[y,x]>0)&(ids[y,x]!=j)).astype(np.uint8);ids[y,x]=j;center[y,x]=1;angles[0,y,x]=np.cos(2*theta);angles[1,y,x]=np.sin(2*theta);dense.append(dict(points=p.round(2).tolist()))
            cv2.polylines(overlay,[xy],False,(25,110,240),1,cv2.LINE_AA)
        target,dist=geometry_target(center,angles);auto,_=ink_target(rgb);confidence=np.zeros_like(target);confidence[0,dist<=2.5]=.65;confidence[0,(dist>2.5)&(dist<7)&(auto[0]<35)]=.35;confidence[0,(dist>9)&(auto[0]<5)]=.045;confidence[1:3,dist<2.5]=.6;confidence[3:5,(dist<5)&((auto[0]>25)|(dist<1.5))]=.4;amb=cv2.dilate(collision,np.ones((5,5),np.uint8))>0;confidence[1:5,amb]=0
        labels=ROOT/'labels'/(r['id']+'.npz');np.savez_compressed(labels,target=target.astype(np.float16),confidence=confidence.astype(np.float16));Image.fromarray(overlay).save(ROOT/'review'/(r['id']+'_trace.png'))
        cache=ROOT/'ink_cache'/(r['id']+'.npz')
        if not cache.exists():
            pad=np.pad(rgb,((32,32),(32,32),(0,0)),mode='edge');ink=predict(teacher,pad,'cuda',tile=320)[:,32:32+h,32:32+w];np.savez_compressed(cache,ink=ink.astype(np.float16))
        records.append(dict(r,width=w,height=h,dense_paths=dense,labels=str(labels),ink_cache=str(cache),image_sha256=hashlib.sha256(Path(r['image']).read_bytes()).hexdigest(),labels_sha256=hashlib.sha256(labels.read_bytes()).hexdigest(),human_expert_verified=False,unmarked_is_unknown=True));print('PREPARED',r['id'],len(dense),flush=True)
    del teacher;torch.cuda.empty_cache()
    from collections import Counter
    groups={}
    for r in records:groups.setdefault(r['source_group'],set()).add(r['split'])
    assert all(len(v)==1 for v in groups.values())
    write('manifest.json',dict(records=records,annotation_provenance='assistant visually traced initial labels, no human expert verification',created_before_training=True,book_family_independence_fully_verified=False))
    audit=dict(images=len(records),images_by_split=dict(Counter(r['split'] for r in records)),paths_by_split={s:sum(len(r['paths']) for r in records if r['split']==s) for s in ('train','validation','test')},expert_annotations=0,known_source_groups=len(groups),known_source_groups_do_not_cross_splits=True,parent_models_may_have_seen_validation_images=True)
    write('data_audit.json',audit)
    for start in range(0,len(records),5):
        batch=records[start:start+5];board=Image.new('RGB',(1300,270*len(batch)),'#f4f4ed');d=ImageDraw.Draw(board)
        for i,r in enumerate(batch):
            d.text((8,i*270+3),r['id']+' | '+r['split']+' | 仅助手初始描线，待人工复核',font=FONT,fill='black')
            for j,p in enumerate((r['image'],ROOT/'review'/(r['id']+'_trace.png'))):
                im=Image.open(p).convert('RGB');im.thumbnail((625,232));board.paste(im,(j*650+8+(625-im.width)//2,i*270+32+(232-im.height)//2))
        board.save(ROOT/'review'/f'annotation_pack_{start//5+1}.jpg',quality=94)
    write('evaluation_protocol.json',dict(created_before_training=True,trace_tolerance_pixels=4,center_heatmap_threshold=.35,preview_ids=[r['id'] for r in records if r['split']=='test' and r['id'].startswith('v09_')],ink_model='v06 frozen',planner='stroke_streamline_v08 strict=True, unchanged',new_test_groups=2,expert_accuracy_available=False,offset_predictions_auxiliary_only=True));print('DATA_READY',json.dumps(audit),flush=True)

def checks():
    import torch
    from gongbi_training.centerline_only_v10 import model_new,input_from_ink,loss_fn
    torch.set_num_threads(4);rows=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))['records'];r=next(r for r in rows if r['split']=='train');ink=np.load(r['ink_cache'])['ink'].astype(np.float32);z=np.load(r['labels']);features=input_from_ink(ink);t=z['target'].astype(np.float32);c=z['confidence'].astype(np.float32)
    x=np.stack([cv2.resize(a,(256,256)) for a in features]);target=np.stack([cv2.resize(a,(256,256)) for a in t]);weight=np.stack([cv2.resize(a,(256,256)) for a in c]);model=model_new(V08/'best.pt').cuda();out=model(torch.from_numpy(x)[None].cuda());value=loss_fn(out,torch.from_numpy(target)[None].cuda(),torch.from_numpy(weight)[None].cuda());assert out.shape==(1,5,256,256) and torch.isfinite(value);value.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(norm)
    groups={}
    for record in rows:groups.setdefault(record['source_group'],set()).add(record['split'])
    assert all(len(v)==1 for v in groups.values());assert not {r['image'] for r in rows if r['split']=='train'}&{r['image'] for r in rows if r['split']=='test'}
    write('training_checks.json',dict(finite_forward_backward=True,output_shape=list(out.shape),gradient_norm=float(norm),known_source_groups_disjoint=True,test_images_excluded=True,expert_annotations=0,hardware_used=False));print('CHECKS_PASS',flush=True)

def finalize():
    import torch,base64
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'));records=manifest['records'];config=json.loads((ROOT/'config.json').read_text(encoding='utf-8'));assert config['manifest_sha256']==hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest();assert config['frozen_ink_sha256']==hashlib.sha256((INK/'best.pt').read_bytes()).hexdigest()
    for r in records:
        assert r['image_sha256']==hashlib.sha256(Path(r['image']).read_bytes()).hexdigest();assert r['labels_sha256']==hashlib.sha256(Path(r['labels']).read_bytes()).hexdigest()
    hashes={}
    for p in [ROOT/'best.pt',ROOT/'initial.pt']+[ROOT/'runs'/f'phase_{phase}'/'best.pt' for phase in (1,2)]:
        ck=torch.load(p,map_location='cpu',weights_only=False);assert all(torch.isfinite(t).all() for t in ck['model'].values());hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    plans=list((ROOT/'evaluation').glob('*/planned_trajectory.json'))
    for p in plans:
        a=json.loads(p.read_text(encoding='utf-8'));assert not a['ready_for_robot'] and not a['actually_drawn'] and a['physical_z'] is None
        for s in a['strokes']:
            assert len(s['xy_pixels'])==len(s['width_pixels'])==len(s['ink_density'])==len(s['relative_pressure']);assert np.isfinite(s['xy_pixels']).all()
    images=list((ROOT/'evaluation').rglob('*.png'))+list((ROOT/'evaluation').glob('*.jpg'))
    for p in images:
        with Image.open(p) as im:im.verify()
    # Embed native-size before/after pictures for zoom and overlay review.
    data=[]
    for r in records:
        if r['split']!='test':continue
        folder=ROOT/'evaluation'/r['id'];item=dict(id=r['id'])
        for key,name in [('original','original.png'),('previous','before_centerlines.png'),('trained','after_centerlines.png')]:item[key]='data:image/png;base64,'+base64.b64encode((folder/name).read_bytes()).decode()
        data.append(item)
    html='''<!doctype html><meta charset="utf-8"><title>中心线新旧对照</title><style>body{font:16px sans-serif;margin:20px;background:#f4f4ed}button,select{font:inherit}#panels{display:flex;gap:12px;overflow:auto}figure{margin:4px;flex:0 0 auto}img,canvas{background:white;border:1px solid #ccc}p{max-width:1100px}</style><h2>中心线新旧对照</h2><p>原图、旧版与本轮使用相同墨迹和规划器。描线标签尚未专家确认，本轮未证明整体优于旧版。可放大图片或调整红色叠加，检查漏线、误接与重复线。所有图均为离线结果。</p><select id="case"></select><button id="plus">放大</button><button id="minus">缩小</button><label>红线叠加 <input id="alpha" type="range" min="0" max="100" value="65"></label><div id="panels"><figure><figcaption>原图</figcaption><img id="original"></figure><figure><figcaption>旧版中心线</figcaption><img id="previous"></figure><figure><figcaption>本轮中心线</figcaption><img id="trained"></figure><figure><figcaption>本轮叠加原图</figcaption><canvas id="overlay"></canvas></figure></div><script>
const data=EMBEDDED,sel=document.getElementById('case'),alpha=document.getElementById('alpha'),canvas=document.getElementById('overlay'),ctx=canvas.getContext('2d'),original=document.getElementById('original'),previous=document.getElementById('previous'),trained=document.getElementById('trained');let zoom=.7;data.forEach((r,i)=>sel.add(new Option(r.id,i)));function resize(){for(let im of [original,previous,trained]){im.style.width=im.naturalWidth*zoom+'px';im.style.height=im.naturalHeight*zoom+'px'}canvas.style.width=canvas.width*zoom+'px';canvas.style.height=canvas.height*zoom+'px'}function draw(){if(!original.complete||!trained.complete||!original.naturalWidth||!trained.naturalWidth)return;canvas.width=original.naturalWidth;canvas.height=original.naturalHeight;ctx.drawImage(original,0,0);let out=ctx.getImageData(0,0,canvas.width,canvas.height),tmp=document.createElement('canvas');tmp.width=canvas.width;tmp.height=canvas.height;let tc=tmp.getContext('2d');tc.drawImage(trained,0,0);let line=tc.getImageData(0,0,tmp.width,tmp.height).data,a=+alpha.value/100;for(let i=0;i<line.length;i+=4){let amount=(1-line[i]/255)*a;out.data[i]=out.data[i]*(1-amount)+245*amount;out.data[i+1]=out.data[i+1]*(1-amount)+25*amount;out.data[i+2]=out.data[i+2]*(1-amount)+25*amount}ctx.putImageData(out,0,0);resize()}function load(){let r=data[+sel.value];original.onload=draw;trained.onload=draw;previous.onload=resize;original.src=r.original;previous.src=r.previous;trained.src=r.trained}sel.onchange=load;alpha.oninput=draw;document.getElementById('plus').onclick=()=>{zoom=Math.min(3,zoom+.2);resize()};document.getElementById('minus').onclick=()=>{zoom=Math.max(.2,zoom-.2);resize()};load();</script>'''
    (ROOT/'comparison_viewer.html').write_text(html.replace('EMBEDDED',json.dumps(data).replace('</','<\\/')),encoding='utf-8')
    for name in ('comparison_viewer','trajectory_viewer'):
        text=(ROOT/(name+'.html')).read_text(encoding='utf-8');(ROOT/(name+'_check.js')).write_text(text.split('<script>',1)[1].split('</script>',1)[0],encoding='utf-8')
    report=(ROOT/'REPORT.md').read_text(encoding='utf-8').replace('纸面误检只在明确标为纸面的局部区域统计，未标注暗线不计入。','纸面误检只在自动判作纸面的局部区域统计（仍可能误判），未标注暗线不计入。')
    report+='''

## 本轮观察与采用结论

四张新增参考图上，部分参考点覆盖从0.7599略降至0.7533，最佳单路径覆盖从0.4318降至0.3928，每条参考线对应片段数基本不变（4.27→4.25）。原三张开发图覆盖从0.7194降至0.6831。已自动标作纸面的局部误检降低，但不能把它当作全图精确率提高，未标注暗线仍未评分。

目视检查显示一些细碎线条收敛，但漏线、断裂、宽墨块的重复中心线和错误弧线仍存在。不能用画面更干净或片段更少宣称完整性改善。中心线模块的输入更换及训练没有在本轮证据下证明整体优于v08，因此本轮标为experimental_not_promoted，旧模型和实际绘画流程均保持原样。

comparison_viewer.html可选7张图、按原始像素放大并调整红色叠加透明度。trajectory_viewer.html默认细线回放。所有预测图直接来自相同提取/规划管线，没有手工修改预测结果。
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8')
    for p in [Path(__file__),Path(__file__).with_suffix('.ps1')]+[Path(__file__).resolve().parents[1]/'gongbi_training'/f for f in ('centerline_annotations_v10.py','centerline_only_v10.py','centerline_eval_v10.py')]:shutil.copy2(p,ROOT/'source_snapshot'/p.name)
    write('verification.json',dict(status='experimental_not_promoted',frozen_ink_hash_unchanged=True,manifest_hash_unchanged=True,reference_images_and_labels_unchanged=True,weights_finite=True,hashes=hashes,verified_trajectory_files=len(plans),verified_images=len(images),same_planner_for_before_after=True,frozen_width_and_role_predictions=True,expert_annotations=0,robot_connected=False,browser_interaction_tested=False));print('FINAL_VERIFIED',flush=True)

if __name__=='__main__':
    if sys.argv[1]=='candidates':candidates()
    elif sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='train':
        from gongbi_training.centerline_only_v10 import train
        train(ROOT,V08/'best.pt',INK/'best.pt')
    elif sys.argv[1]=='check':checks()
    elif sys.argv[1]=='evaluate':
        from gongbi_training.centerline_eval_v10 import evaluate
        evaluate(ROOT,V08,V09,INK)
    elif sys.argv[1]=='finalize':finalize()
