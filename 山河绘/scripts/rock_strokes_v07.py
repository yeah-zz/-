"""Directional rock brushstroke experiment. Offline only; no device imports."""
import sys, json, hashlib, re, io, time, math
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure, LAB
configure()
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
ROOT=LAB/'rock_strokes_v07_20261023'
for folder in ('raw','html','review','labels','crops','runs','evaluation'):
    (ROOT/folder).mkdir(parents=True,exist_ok=True)
OLD=LAB/'rock_cunfa_v06_20261023'
PARENT=LAB/'runs'/'mobilenet_unet_rock_cunfa_v06_expanded'
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)

def write(name,obj):
    (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def sheet(rows,name):
    for start in range(0,len(rows),48):
        batch=rows[start:start+48];im=Image.new('RGB',(1440,math.ceil(len(batch)/6)*190),'#f4f4ed');d=ImageDraw.Draw(im)
        for k,r in enumerate(batch):
            x=k%6*240;y=k//6*190;p=Image.open(r['image']).convert('RGB');p.thumbnail((232,155))
            im.paste(p,(x+(232-p.width)//2,y));d.text((x+3,y+158),f'{start+k}: '+r['id'][-8:],font=FONT,fill='black')
        im.save(ROOT/'review'/f'{name}_{start//48+1:02d}.jpg',quality=91)

def collect():
    from gongbi_training.cunfa_collect import fetch,Images
    from urllib.parse import urljoin
    from concurrent.futures import ThreadPoolExecutor,as_completed
    previous=[]
    for path in (OLD/'new_sources.json',OLD/'new_sources_phase2.json',LAB/'rock_cunfa_v04'/'web_sources.json'):
        previous+=json.loads(path.read_text(encoding='utf-8-sig'))
    old_urls={r.get('url',r.get('image_url')) for r in previous}
    old_pages={r.get('page',r.get('source_page')) for r in previous}
    rgb_hash={hashlib.sha256(np.asarray(Image.open(r['image']).convert('RGB')).tobytes()).hexdigest() for r in previous}
    queue=sorted(p for p in old_pages if p and 'kuhw.com' in p)
    queue+=['https://www.kuhw.com/hhjc/sshjc/','https://www.kuhw.com/hhjc/sshjc/index_2.html','https://www.kuhw.com/hhjc/sshjc/index_3.html']
    visited=set();tasks={};pages=[];errors=[]
    while queue and len(visited)<95:
        page=queue.pop(0)
        if page in visited:continue
        visited.add(page)
        path=ROOT/'html'/(hashlib.sha256(page.encode()).hexdigest()[:14]+'.html')
        try:
            raw=path.read_bytes() if path.exists() else fetch(page);path.write_bytes(raw)
            html=re.sub(r'<!--.*?-->','',raw.decode('utf-8','replace'),flags=re.S)
            parser=Images();parser.feed(html)
            for url,title in re.findall(r'<a[^>]*href=[\"\']([^\"\']+)[\"\'][^>]*>(.*?)</a>',html,re.S):
                title=re.sub('<[^>]+>','',title);u=urljoin(page,url)
                if re.search('皴|山石|石头|石法|叠石|画石',title) and '/hhjc/' in u and u.endswith('.html') and u not in visited and u not in queue:queue.append(u)
            # Already collected articles are discovery seeds only.
            if page not in old_pages and 'class="entry"' in html:
                body=html.split('class="entry"',1)[1].split('class="page5"',1)[0]
                p=Images();p.feed(body)
                for url in p.images:
                    u=urljoin(page,url)
                    if '/d/file/' in u and u not in old_urls:tasks.setdefault(u,dict(page=page,url=u,title=parser.title))
                pages.append(dict(page=page,title=parser.title))
            if len(visited)%10==0:print('DISCOVERY',len(visited),'new urls',len(tasks),flush=True)
        except Exception as exc:errors.append(dict(page=page,error=str(exc)))
    def job(item):
        key=hashlib.sha256(item['url'].encode()).hexdigest()[:14];path=ROOT/'raw'/(key+'.png')
        if not path.exists():
            p=Image.open(io.BytesIO(fetch(item['url'],item['page']))).convert('RGB')
            if min(p.size)<100 or max(p.size)<240:raise ValueError('small')
            p.thumbnail((1600,1600));p.save(path)
        p=Image.open(path).convert('RGB');digest=hashlib.sha256(np.asarray(p).tobytes()).hexdigest()
        return dict(item,id='v07_'+key,image=str(path),rgb_sha256=digest,width=p.width,height=p.height)
    results=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs={pool.submit(job,r):r for r in list(tasks.values())[:550]}
        for f in as_completed(jobs):
            try:results.append(f.result())
            except Exception as exc:errors.append(dict(url=jobs[f]['url'],error=str(exc)))
    new=[]
    for r in sorted(results,key=lambda r:(r['page'],r['id'])):
        if r['rgb_sha256'] not in rgb_hash:new.append(r);rgb_hash.add(r['rgb_sha256'])
    write('new_sources.json',new);write('collection_audit.json',dict(pages=pages,visited=len(visited),attempted=len(tasks),downloaded=len(results),new_exact_unique=len(new),errors=errors))
    sheet(new,'new');print('COLLECTION_COMPLETE',len(new),flush=True)

def prepare():
    from gongbi_training.cunfa_prepare import ink_target,phash
    from gongbi_training.stroke_direction_v07 import weak_targets,procedural
    cv2.setNumThreads(2)
    old=json.loads((PARENT/'dataset_manifest_snapshot.json').read_text(encoding='utf-8'))
    rows=json.loads((ROOT/'new_sources.json').read_text(encoding='utf-8'))
    boxes={0:(.04,.40,.96,.93),1:(.23,.20,.97,.95),2:(.28,.28,.97,.94),3:(.14,.20,.96,.95),6:(.03,.43,.97,.96),7:(.24,.22,.98,.97),9:(.01,.32,.98,.96),11:(.20,.32,.98,.95),13:(.02,.57,.98,.97)}
    accepted=[];rejected=[];old_hashes=[(r,phash(np.asarray(Image.open(r['image']).convert('RGB')))) for r in old['real']]
    hashes=[v for r,v in old_hashes]
    for i,r in enumerate(rows):
        if i not in boxes and i<20:rejected.append(dict(index=i,reason='complex painting, text-heavy sheet or photograph'));continue
        im=Image.open(r['image']).convert('RGB');w,h=im.size;box=boxes.get(i,(.01,.01,.99,.99));im=im.crop(tuple(round(v*(w if k%2==0 else h)) for k,v in enumerate(box)));im.thumbnail((850,850));rgb=np.asarray(im);ph=phash(rgb)
        if any(np.count_nonzero(ph!=p)<=5 for p in hashes):rejected.append(dict(index=i,reason='perceptual near duplicate'));continue
        hashes.append(ph);identity=r['id']+'_c0';image=ROOT/'crops'/(identity+'.png');im.save(image)
        targets,weights=ink_target(rgb);labels=ROOT/'labels'/(identity+'_ink.npz');np.savez_compressed(labels,targets=targets,weights=weights)
        # One teaching page is always one split; known Li Xiongcai book pages
        # join the existing book family when present.
        family='v07_page_'+hashlib.sha256(r['page'].encode()).hexdigest()[:10]
        role='test' if '/2022.html' in r['page'] else 'train'
        if '黎雄才' in r['title']:
            matches=[o for o in old['real'] if any(t in ((o.get('page') or '')+(o.get('source_page') or '')) for t in ('1914.html','1963.html'))]
            roles={o['split'] for o in matches}
            if roles:role='test' if 'test' in roles else 'validation' if 'validation' in roles else 'train'
        accepted.append(dict(id=identity,image=str(image),labels=str(labels),group=family,split=role,page=r['page'],source_page=r['page'],review_index=i,new_v07=True,target_provenance='automatic weak ink and ridge labels; not human stroke annotation'))
    real=[]
    for i,r in enumerate(old['real']+accepted):
        out=dict(r);path=ROOT/'labels'/(r['id']+'_direction.npz')
        if not path.exists():
            rgb=np.asarray(Image.open(r['image']).convert('RGB'))
            with np.load(r['labels']) as z:density=z['targets'][1].astype(np.float32)/255
            t,c=weak_targets(rgb,density);np.savez_compressed(path,target=t.astype(np.float16),confidence=c.astype(np.float16))
        out['direction_labels']=str(path);real.append(out)
        if i%100==0:print('DIRECTION_LABELS',i,flush=True)
    assert accepted and any(r['split']=='train' for r in accepted)
    manifest=dict(real=real,previous_manifest=str(PARENT/'dataset_manifest_snapshot.json'),new_accepted=accepted,rejected=rejected,expert_stroke_labels=0)
    write('manifest.json',manifest)
    from collections import Counter
    write('dataset_audit.json',dict(new_downloaded=len(rows),new_accepted_images=len(accepted),new_split=dict(Counter(r['split'] for r in accepted)),total_real=len(real),total_split=dict(Counter(r['split'] for r in real)),rejected=rejected,all_data_on_E=True))
    sheet(accepted,'accepted')
    rgb,t,confidence,paths=procedural(np.random.default_rng(97001),320)
    preview=Image.new('RGB',(960,320),'white');preview.paste(Image.fromarray(rgb),(0,0));preview.paste(Image.fromarray(np.uint8(255*(1-t[0]))).convert('RGB'),(320,0))
    display=np.full((320,320,3),255,np.uint8)
    for j,p in enumerate(paths):cv2.polylines(display,[p.astype(np.int32)],False,(int(j*29%200),90,int(j*57%200)),1,cv2.LINE_AA)
    preview.paste(Image.fromarray(display),(640,0));preview.save(ROOT/'review'/'procedural_targets.jpg',quality=94)
    print('PREPARE_COMPLETE',len(accepted),len(real),flush=True)

def evaluate():
    import torch,shutil
    from gongbi_training.stroke_direction_v07 import model_new,infer,plan,procedural
    sys.path.insert(1,str(OLD))
    from trajectory_offline import make_plan,save_plan
    from gongbi_training.model import GongbiLineNet
    from gongbi_training.infer import predict
    torch.set_num_threads(4);cv2.setNumThreads(2)
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'));training=json.loads((ROOT/'training_summary.json').read_text(encoding='utf-8'))
    choices=[]
    for ri in (1,2,3):
        path=ROOT/'runs'/f'round_{ri}'/'best.pt';ck=torch.load(path,map_location='cpu',weights_only=False);choices.append((ck['validation']['loss'],ri,path));del ck
    _,selected,path=min(choices);shutil.copy2(path,ROOT/'best.pt')
    direction_models=[]
    for _,ri,p in sorted(choices,key=lambda v:v[1]):
        m=model_new().cuda();m.load_state_dict(torch.load(p,map_location='cuda',weights_only=False)['model']);m.eval();direction_models.append(m)
    ink_model=GongbiLineNet().cuda();ink_model.load_state_dict(torch.load(PARENT/'best.pt',map_location='cuda',weights_only=False)['model']);ink_model.eval()
    tests=[r for r in manifest['real'] if r['split']=='test'];records={r['id']:r for r in manifest['real']}
    combos=['new_8d098ee60236de_p1_c0','new_b54b8467007896_p1_c0','new_d91ef970ece326_p1_c0']
    extra=[r['id'] for r in tests if r.get('new_v07')];combos+=extra[:3]
    metrics=[];outputs={};plans=[]
    for r in tests:
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));h,w=rgb.shape[:2];folder=ROOT/'evaluation'/r['id'];folder.mkdir(exist_ok=True)
        stats=[];predictions=[]
        z=np.load(r['direction_labels']);t=z['target'].astype(np.float32);confidence=z['confidence'];valid=confidence>.2;truth=t[0]>.35
        for ri,m in enumerate(direction_models,1):
            pred=infer(m,rgb);predictions.append(pred);p=pred[0]>.35
            tp=int((p&truth&valid).sum());fp=int((p&~truth&valid).sum());fn=int((~p&truth&valid).sum())
            cosine=(pred[1]*t[1]+pred[2]*t[2])/np.maximum(np.hypot(pred[1],pred[2])*np.hypot(t[1],t[2]),1e-5)
            angle=float((.5*np.arccos(np.clip(cosine,-1,1)))[truth&valid].mean()*180/np.pi) if (truth&valid).any() else None
            stats.append(dict(round=ri,tp=tp,fp=fp,fn=fn,weak_centerline_f1=2*tp/max(1,2*tp+fp+fn),weak_direction_error_degrees=angle))
        metrics.append(dict(id=r['id'],new_v07=bool(r.get('new_v07')),rounds=stats))
        if r['id'] in combos:
            padded=np.pad(rgb,((32,32),(32,32),(0,0)),mode='edge');ink=predict(ink_model,padded,'cuda',tile=320)[:,32:32+h,32:32+w]
            selected_pred=predictions[selected-1]
            old_info,old_view,old_replay=make_plan(ink);analytic_info,analytic_view,analytic_replay=plan(ink);info,view,replay=plan(ink,selected_pred)
            save_plan(folder,info,w,h)
            for name,a in [('original',rgb),('old_plan',old_view),('planner_only',analytic_view),('new_plan',view),('new_replay',replay),('old_replay',old_replay),('learned_center_probability',np.uint8(np.clip(selected_pred[0]*255,0,255))),('ink_density',np.uint8(255*(1-ink[1])))]:Image.fromarray(a).save(folder/(name+'.png'))
            np.savez_compressed(folder/'raw_predictions.npz',ink=ink,direction=selected_pred)
            plans.append(dict(id=r['id'],previous_strokes=old_info['planned_strokes'],planner_only_strokes=analytic_info['planned_strokes'],new_strokes=info['planned_strokes'],small_closed_candidates_rejected=info['rejected_small_closed_candidates']))
            outputs[r['id']]=[rgb,old_view,analytic_view,view,replay]
        print('EVALUATED',r['id'],flush=True)
    write('evaluation/metrics.json',dict(selected_round=selected,selection_rule='minimum fixed validation loss; test results not used',per_image=metrics,interpretation='automatic ridge teacher agreement, not expert painting accuracy'))
    write('evaluation/planner_comparison.json',plans)
    labels=['原图','上一版：墨块骨架轨迹','仅更换方向规划器','本轮模型＋方向规划器','连续笔宽回放（非实画）']
    def board(ids,name):
        canvas=Image.new('RGB',(2000,len(ids)*350),'#f5f5ee');draw=ImageDraw.Draw(canvas)
        for i,identity in enumerate(ids):
            y=i*350;draw.text((8,y+3),identity+(' | 新增来源留出图' if records[identity].get('new_v07') else ' | 固定旧留出图'),font=FONT,fill='black')
            for j,label in enumerate(labels):
                draw.text((j*400+6,y+31),label,font=FONT,fill='black');im=Image.fromarray(outputs[identity][j]).convert('RGB');im.thumbnail((382,280));canvas.paste(im,(j*400+9+(382-im.width)//2,y+62+(280-im.height)//2))
        canvas.save(ROOT/'evaluation'/name,quality=95)
    board(combos[:3],'fixed_combinations.jpg')
    if len(combos)>3:board(combos[3:],'new_heldout_rocks.jpg')
    # Every round is preserved for visual review on the three fixed inputs.
    rounds_board=Image.new('RGB',(1600,340*3),'#f5f5ee');draw=ImageDraw.Draw(rounds_board)
    for i,identity in enumerate(combos[:3]):
        rgb=outputs[identity][0];data=np.load(ROOT/'evaluation'/identity/'raw_predictions.npz');ink=data['ink'];y=i*340
        draw.text((8,y+3),'原图 / 第1轮 / 第2轮 / 第3轮；固定图，未参与训练',font=FONT,fill='black')
        arrays=[rgb]+[plan(ink,infer(m,rgb))[1] for m in direction_models]
        for j,a in enumerate(arrays):
            im=Image.fromarray(a).convert('RGB');im.thumbnail((380,290));rounds_board.paste(im,(j*400+8,y+40+(290-im.height)//2))
    rounds_board.save(ROOT/'evaluation'/'three_rounds.jpg',quality=94)
    verify_geometry(ROOT)
    report(ROOT,selected,choices,metrics,plans,combos)
    print('EVALUATION_COMPLETE_SELECTED_ROUND',selected,flush=True)

def verify_geometry(root):
    from gongbi_training.stroke_direction_v07 import plan
    # Physical meaning tests: tapering open strokes must not become loops,
    # and two separated strokes must not be joined across blank paper.
    density=np.zeros((110,170),np.float32)
    cv2.line(density,(15,25),(70,45),.75,3);cv2.line(density,(100,65),(150,85),.75,4)
    coverage=np.minimum(density*3,1);info,_,replay=plan(np.stack((coverage,density)))
    assert info['planned_strokes']>=2
    for r in info['strokes']:
        p=np.asarray(r['xy_pixels']);assert np.isfinite(p).all();assert not (p[:,0].min()<75 and p[:,0].max()>95)
        assert len(p)==len(r['relative_pressure'])==len(r['width_pixels']);assert max(r['width_pixels'])<=8.01
    empty,_,_=plan(np.zeros((2,80,80),np.float32));assert empty['planned_strokes']==0
    write('geometry_checks.json',dict(separated_strokes_not_merged=True,empty_input_valid=True,arrays_finite=True,uncalibrated_pressure_explicit=True,robot_connected=False))

def report(root,selected,choices,metrics,plans,combos):
    import shutil
    audit=json.loads((root/'dataset_audit.json').read_text(encoding='utf-8'));summary=json.loads((root/'training_summary.json').read_text(encoding='utf-8'))
    aggregate=[]
    for ri in (1,2,3):
        rows=[r['rounds'][ri-1] for r in metrics];tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
        aggregate.append(dict(round=ri,weak_f1=2*tp/max(1,2*tp+fp+fn),weak_direction_error=float(np.mean([r['weak_direction_error_degrees'] for r in rows if r['weak_direction_error_degrees'] is not None]))))
    write('evaluation/aggregate.json',aggregate)
    sources=json.loads((root/'new_sources.json').read_text(encoding='utf-8'));text='# 本轮公开教程来源\n\n'+'\n'.join(f"- [{r['title']}]({r['page']})" for r in {r['page']:r for r in sources}.values())
    (root/'SOURCES.md').write_text(text,encoding='utf-8')
    table='\n'.join(f"| {r['round']} | {r['weak_f1']:.4f} | {r['weak_direction_error']:.2f}° |" for r in aggregate)
    text=f'''# 山石皴线方向模型 v07（离线实验）

新增下载 {audit['new_downloaded']} 张精确去重图片；人工查看拼图、裁剪并剔除近似重复后保留 {audit['new_accepted_images']} 张。新增划分 {audit['new_split']}。合计真实裁图 {audit['total_real']}，划分 {audit['total_split']}。源图、裁图、增强样本不混为独立画作。

训练三轮，每轮800次有效更新，总计{summary['total_successful_updates']}次；耗时{summary['elapsed_seconds']:.1f}秒。每轮混合真实弱监督与160个已知开放笔画的程序生成训练样本；另有不同种子的程序生成验证集。新网络复用v06的骨干，重新训练四通道笔画头：中心线、cos(2θ)、sin(2θ)、相对宽度。不是8类皴法分类器。墨迹图仍来自冻结的v06模型，不把规划改善宣称为墨迹模型改善。

模型检查点根据固定验证损失选择第{selected}轮；测试图不参与训练或检查点选择。本次新增25张全部用于训练，评估仍使用原有34张固定测试图；没有新增独立盲测集。固定测试图已被多轮查看，属于开发回归集。规划器曾在其中一张图上作视觉检查，因此不能把整条流水线的结果称为独立盲测准确率。同书不同页的潜在重复仍需更完整排查。

| 轮次 | 自动脊线弱标签F1 | 与弱方向标签的角误差 |
|---|---|---|
{table}

这些数值只衡量自动脊线教师一致性，不是人工准确率、皴法识别率、原始笔顺恢复率或实机精度。没有人工逐笔标注，不能承诺完整提线。单幅成画的真实走笔方向通常不能唯一恢复。

## 规划变化与对照

上一版细化整片墨迹，会沿墨块孔洞绕圈。新规划器在局部墨色脊线上找中心，按方向在交点配对，并剔除短小近闭合候选。长闭合轮廓保留；因此仍可能留有错误弧线。三栏消融对照分别是旧规划器、仅换方向规划器、新方向模型加规划器。圆圈减少不能全归功于训练。

宽度改用横截面墨色与模型的宽度候选，做连续平滑；本次像素回放宽度上限为8px，宽墨块不会被当作单根无限粗笔画。8px是离线预览约束，不是纸上毫米宽度。起收笔接触渐变是人为先验，并非恢复出的真实压力。

机械臂将来执行的是经过纸面映射和软毛笔标定后的轨迹点、Z或压力、速度、抬落笔。彩色计划图是路线可视化；连续笔宽回放是同一路线按估计宽度渲染的检查图，不是给机械臂执行的另一条路线，也不是毛笔物理仿真。`relative_pressure`仅是0..1相对建议，尚无毫米/力映射。所有计划ready_for_robot=false、physical_z=null、robot_connected=false。未连接设备，未生成实画记录。

## 已知限制

剔除小闭合候选可能同时删掉真实的小石轮廓；单条脊线可能分段。密集皴擦的真实笔画身份和笔顺仍有歧义。当前没有可靠的轮廓/皴/擦/染逐像素语义标签，不能把每条内部线都认定为正确皴法。结果必须配合原图检查，下一步需真实逐笔标注与实机标定。
'''
    (root/'REPORT.md').write_text(text,encoding='utf-8')
    snap=root/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in [Path(__file__),Path(__file__).resolve().parents[1]/'gongbi_training'/'stroke_direction_v07.py',Path(__file__).with_suffix('.ps1')]:shutil.copy2(p,snap/p.name)
    write('verification.json',dict(best_sha256=hashlib.sha256((root/'best.pt').read_bytes()).hexdigest(),best_selected_round=selected,actual_drawing=False,geometry_tests_passed=True,checkpoints_preserved=3))
    # Offline visualizer reads embedded JSON only. No network or device endpoint.
    embedded=[]
    for identity in combos:
        folder=root/'evaluation'/identity;embedded.append(dict(id=identity,plan=json.loads((folder/'planned_trajectory.json').read_text(encoding='utf-8'))))
    viewer='''<!doctype html><meta charset="utf-8"><title>离线山石笔画回放</title><style>body{font:16px sans-serif;background:#f4f4ed;margin:24px}canvas{background:white;border:1px solid #ccc;max-width:100%;height:auto}label{margin:15px}p{max-width:1000px}</style><h2>方向笔画与连续笔宽：离线回放</h2><p>颜色图显示中心线路径；宽度图显示同一条路径的估计落墨。相对下压未标定，不是毫米或力；没有设备控制功能。实际落笔尚未采集。</p><select id="cases"></select><button id="play">播放/暂停</button><input id="progress" type="range" min="0" max="1000" value="1000"><label><input type="checkbox" id="wide" checked>连续笔宽</label><p id="state"></p><canvas id="canvas"></canvas><script>
const data=EMBEDDED;const sel=document.getElementById('cases'),slider=document.getElementById('progress'),wide=document.getElementById('wide'),canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),state=document.getElementById('state');let active=false;
data.forEach((r,i)=>sel.add(new Option(r.id,i)));
function draw(){let p=data[+sel.value].plan;canvas.width=p.image_width;canvas.height=p.image_height;ctx.fillStyle='white';ctx.fillRect(0,0,canvas.width,canvas.height);let total=p.strokes.reduce((a,s)=>a+s.xy_pixels.length,0),limit=+slider.value/1000*total,count=0;for(const s of p.strokes){let q=s.xy_pixels;ctx.lineCap='butt';for(let k=1;k<q.length;k++){if(++count>limit)break;ctx.strokeStyle=wide.checked?'rgba(0,0,0,'+s.ink_density[k]+')':'hsl('+(s.order*47%360)+' 60% 40%)';ctx.lineWidth=wide.checked?s.width_pixels[k]:1;ctx.beginPath();ctx.moveTo(...q[k-1]);ctx.lineTo(...q[k]);ctx.stroke();}}state.textContent=p.planned_strokes+'条候选笔画；剔除'+p.rejected_small_closed_candidates+'条短小近闭合候选。未连接机械臂。';}
sel.onchange=draw;slider.oninput=draw;wide.onchange=draw;document.getElementById('play').onclick=()=>{active=!active;if(active&&+slider.value>=1000)slider.value=0;};setInterval(()=>{if(active){slider.value=Math.min(1000,+slider.value+4);draw();if(+slider.value>=1000)active=false;}},30);draw();</script>'''
    (root/'trajectory_viewer.html').write_text(viewer.replace('EMBEDDED',json.dumps(embedded,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')

def extra_examples():
    """New page examples, not used in training; no quantitative gold claims."""
    import torch
    from gongbi_training.stroke_direction_v07 import model_new,infer,plan
    from gongbi_training.cunfa_prepare import phash
    from gongbi_training.model import GongbiLineNet
    from gongbi_training.infer import predict
    sys.path.insert(1,str(OLD))
    from trajectory_offline import make_plan,save_plan
    torch.set_num_threads(4);cv2.setNumThreads(2)
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'));sources=json.loads((ROOT/'new_sources.json').read_text(encoding='utf-8'))
    training_hashes=[phash(np.asarray(Image.open(r['image']).convert('RGB'))) for r in manifest['real'] if r['split']=='train']
    model=model_new().cuda();model.load_state_dict(torch.load(ROOT/'best.pt',map_location='cuda',weights_only=False)['model']);model.eval()
    ink_model=GongbiLineNet().cuda();ink_model.load_state_dict(torch.load(PARENT/'best.pt',map_location='cuda',weights_only=False)['model']);ink_model.eval()
    samples=[(14,(.015,.343,.56,.535)),(14,(.005,.65,.56,.875)),(15,(.16,.36,.88,.64))]
    board=Image.new('RGB',(2000,350*len(samples)),'#f5f5ee');draw=ImageDraw.Draw(board);rows=[]
    labels=['新来源原图裁剪','上一版骨架轨迹','仅换方向规划器','本轮模型＋方向规划器','连续笔宽回放（非实画）']
    for i,(index,box) in enumerate(samples):
        r=sources[index];im=Image.open(r['image']).convert('RGB');w,h=im.size;im=im.crop(tuple(round(v*(w if k%2==0 else h)) for k,v in enumerate(box)));rgb=np.asarray(im);h,w=rgb.shape[:2]
        nearest=min(int(np.count_nonzero(phash(rgb)!=p)) for p in training_hashes)
        if nearest<=5:raise RuntimeError('Supplementary example too similar to training crop')
        identity=f'extra_untrained_{i+1}';folder=ROOT/'evaluation'/identity;folder.mkdir(exist_ok=True);im.save(folder/'original.png')
        padded=np.pad(rgb,((32,32),(32,32),(0,0)),mode='edge');ink=predict(ink_model,padded,'cuda',tile=320)[:,32:32+h,32:32+w];p=infer(model,rgb)
        oi,ov,orr=make_plan(ink);ai,av,ar=plan(ink);info,view,replay=plan(ink,p);save_plan(folder,info,w,h)
        for name,a in [('old_plan',ov),('planner_only',av),('new_plan',view),('new_replay',replay)]:Image.fromarray(a).save(folder/(name+'.png'))
        draw.text((8,i*350+3),f'新增未训练样例 {i+1} | 仅视觉检查；无人工真值，不计入F1',font=FONT,fill='black')
        for j,a in enumerate([rgb,ov,av,view,replay]):
            draw.text((j*400+8,i*350+31),labels[j],font=FONT,fill='black');piece=Image.fromarray(a).convert('RGB');piece.thumbnail((382,280));board.paste(piece,(j*400+9+(382-piece.width)//2,i*350+62+(280-piece.height)//2))
        rows.append(dict(id=identity,page=r['page'],source_image=r['image'],roi=box,nearest_training_phash_distance=nearest,used_for_training=False,quantitative_gold=False,stroke_count=info['planned_strokes']))
    board.save(ROOT/'evaluation'/'new_untrained_examples.jpg',quality=95);write('evaluation/new_untrained_examples.json',rows)
    with (ROOT/'REPORT.md').open('a',encoding='utf-8') as f:f.write('\n\n另增加3个新页面未训练裁图作为视觉检查，均来自本轮未用于训练的2022.html教程页；与全部训练裁图的pHash距离大于5。它们没有人工标注，不纳入34张回归图的量化指标，也不宣称已排除所有同书关系。\n')
    print('EXTRA_EXAMPLES_COMPLETE',json.dumps(rows,ensure_ascii=True),flush=True)

def finalize():
    import shutil,torch
    summary=json.loads((ROOT/'training_summary.json').read_text(encoding='utf-8'))
    assert summary['total_successful_updates']==2400
    hashes={}
    for ri in (1,2,3):
        p=ROOT/'runs'/f'round_{ri}'/'best.pt';ck=torch.load(p,map_location='cpu',weights_only=False)
        assert all(torch.isfinite(t).all() for t in ck['model'].values());hashes[str(ri)]=hashlib.sha256(p.read_bytes()).hexdigest()
    plans=list((ROOT/'evaluation').glob('*/planned_trajectory.json'))
    for p in plans:
        info=json.loads(p.read_text(encoding='utf-8'))
        assert info['physical_z'] is None and info['physical_speed'] is None and info['ready_for_robot'] is False and info['actually_drawn'] is False
        for s in info['strokes']:
            assert len(s['xy_pixels'])==len(s['width_pixels'])==len(s['relative_pressure'])
            assert np.isfinite(np.asarray(s['xy_pixels'])).all()
            assert 0<=min(s['relative_pressure'])<=max(s['relative_pressure'])<=1
    images=list((ROOT/'evaluation').rglob('*.png'))+list((ROOT/'evaluation').glob('*.jpg'))
    for p in images:
        with Image.open(p) as im:im.verify()
    html=(ROOT/'trajectory_viewer.html').read_text(encoding='utf-8');script=html.split('<script>',1)[1].split('</script>',1)[0]
    (ROOT/'viewer_syntax_check.js').write_text(script,encoding='utf-8')
    for p in [Path(__file__),Path(__file__).resolve().parents[1]/'gongbi_training'/'stroke_direction_v07.py',Path(__file__).with_suffix('.ps1')]:shutil.copy2(p,ROOT/'source_snapshot'/p.name)
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    fingerprints=[]
    for r in manifest['new_accepted']:
        fingerprints.append(dict(id=r['id'],image_sha256=hashlib.sha256(Path(r['image']).read_bytes()).hexdigest(),label_sha256=hashlib.sha256(Path(next(o['direction_labels'] for o in manifest['real'] if o['id']==r['id'])).read_bytes()).hexdigest()))
    write('new_training_hashes.json',fingerprints)
    with (ROOT/'REPORT.md').open('a',encoding='utf-8') as f:f.write('''

## 本轮视觉结论

固定三图中，绕孔洞形成的圆圈明显减少，大圆墨团被连续细条回放替代；改善主要来自方向脊线规划器。加入学习模型与只换规划器的差异较小，三轮之间也没有显著视觉跃升。第3图的笔画段数从旧190段增加为216段，说明碎片问题尚未解决。自动中心线弱标签F1从第1轮0.551降到第3轮0.534，而自动方向角误差从18.39°降到15.65°，不能只引用角度改善而宣称提线更完整。

三张新页面裁图可见黑墨区仍会产生额外短线，部分淡线仍丢失；曲线的自然粗细和真实笔画身份尚未恢复。当前方案是离线研究结果，不是已验收的机械臂绘画程序。
''')
    v=json.loads((ROOT/'verification.json').read_text(encoding='utf-8'));v.update(round_checkpoint_sha256=hashes,verified_image_files=len(images),verified_trajectory_files=len(plans),all_stroke_arrays_finite=True,new_training_fingerprints=len(fingerprints));write('verification.json',v)
    print('FINAL_VERIFIED',json.dumps(v),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='collect':collect()
    elif sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='train':
        from gongbi_training.stroke_direction_v07 import train
        train(ROOT,PARENT)
    elif sys.argv[1]=='evaluate':evaluate()
    elif sys.argv[1]=='extra':extra_examples()
    elif sys.argv[1]=='finalize':finalize()
    elif sys.argv[1]=='check':
        verify_geometry(ROOT)
        from gongbi_training.stroke_direction_v07 import plan
        identity='new_d91ef970ece326_p1_c0'
        folder=PARENT/'evaluation'/identity
        ink=np.stack([np.asarray(Image.open(folder/(name+'.png'))).astype(np.float32)/255 for name in ('ink_coverage_probability','ink_density_probability')])
        info,view,replay=plan(ink)
        Image.fromarray(view).save(ROOT/'review'/'planner_probe.png');Image.fromarray(replay).save(ROOT/'review'/'planner_probe_replay.png')
        print('GEOMETRY_CHECKS_PASS',info['planned_strokes'],info['rejected_small_closed_candidates'])
