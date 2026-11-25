# -*- coding: utf-8 -*-
"""Frozen-model scale diagnostic on known cases, not a new acceptance test."""
import os,sys,json,hashlib,time
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
from gongbi_training.runtime import configure,LAB
configure()
import torch,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from gongbi_training.scene_objects import SceneObjects
from guohua_objects_train_20261112 import pack
ROOT=LAB/'object_scale_pairs_20261118';BASE=LAB/'guohua_scene4_mapping_fix_20261116'
COLORS=np.array([[241,237,222],[196,140,62],[70,143,105],[110,112,163]],'uint8')
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,x):(ROOT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def mask(case,region):
    m=Image.new('L',tuple(case['size']),0);ImageDraw.Draw(m).polygon([tuple(q) for q in region['points']],fill=1)
    return np.array(m)>0
def prepare():
    ROOT.mkdir(exist_ok=True);assert not (ROOT/'protocol.json').exists(),'Already frozen'
    originals=read(LAB/'guohua_boundary_reference_20261116/references.json')['cases']
    checked=read(LAB/'guohua_objects_v4_sparse_20261116/weak_reference_manifest.json')['cases']
    specs=[('H_14','foreground_rock','foliage_core'),('qiu-ying_fairyland-of-peach-blossoms_8','rock_right','crown_top'),('wang-shimin_clearing-of-rivers-and-mountains-after-snow_6','rock_upper_right','pine_crown_right')]
    pairs=[]
    for ident,rname,tname in specs:
        c=next(c for c in originals+checked if c['id']==ident) if ident=='H_14' else next(c for c in checked if c['id']==ident)
        regs=[next(r for r in c['regions'] if r['name']==n) for n in [rname,tname]]
        assert [r['object'] for r in regs]==['rock','tree'] and all(r['kind']=='core_polygon' for r in regs)
        assert not (mask(c,regs[0])&mask(c,regs[1])).any()
        pairs.append(dict(id=ident,image=c['image'],image_sha256=sha(c['image']),size=c['size'],regions=regs,
            unknown_between_cores=True,exact_boundary_verified=False,expert_verified=False,
            reference_reused=True,new_work=False,usage='known_diagnostic_only_not_training'))
    write('pairs.json',dict(cases=pairs,new_pixel_labels=0,new_independent_works=0,ink_labels=False,cunfa_labels=False))
    write('protocol.json',dict(date='2026-11-18',scales=[256,512,768],
        checkpoint=str(BASE/'mapping_fixed.pt'),checkpoint_sha256=sha(BASE/'mapping_fixed.pt'),
        script_sha256=sha(__file__),pairs_sha256=sha(ROOT/'pairs.json'),
        selection='all3fixed sizes reported, no crop, threshold, retraining or result-selected fusion',
        evaluate='same known14 corrected coarse targets and3 known paired cores; not expert or independent test',
        hypothesis='Check scale sensitivity before collecting dense boundary supervision',
        no_reference_used_at_inference=True,independent_validation=False,robot_connected=False))
    print('PREPARED',flush=True)
def run():
    p=read(ROOT/'protocol.json');assert not (ROOT/'results.json').exists(),'Frozen evaluation'
    assert sha(__file__)==p['script_sha256'] and sha(p['checkpoint'])==p['checkpoint_sha256']
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    net=SceneObjects().cuda();net.load_state_dict(torch.load(p['checkpoint'],map_location='cuda',weights_only=False)['model']);net.eval();net.requires_grad_(False)
    pairs=read(ROOT/'pairs.json')['cases'];known=[r for r in read(BASE/'results.json')['records'] if r['split']=='new_test']
    ds={r['id']:r for r in read(BASE/'dataset_manifest.json')['records']}
    cases={r['id']:r for r in known+pairs};(ROOT/'predictions').mkdir(exist_ok=True);rows=[];start=time.time()
    for ident,c in cases.items():
        im=np.array(Image.open(c['image']).convert('RGB'));scores={}
        for size in p['scales']:
            x,_,(left,top,nw,nh,w,h)=pack(im,size=size)
            with torch.inference_mode():prob=net(torch.from_numpy(x)[None].cuda()).softmax(1)[0].cpu().numpy()
            prob=prob[:,top:top+nh,left:left+nw];prob=np.stack([cv2.resize(a,(w,h),interpolation=cv2.INTER_LINEAR) for a in prob]);pred=prob.argmax(0).astype('uint8')
            np.savez_compressed(ROOT/f'predictions/{ident}_{size}.npz',prediction=pred)
            score={}
            if ident in {r['id'] for r in known}:
                target=np.array(Image.open(ds[ident]['target']));valid=target!=255
                confusion=np.bincount((target[valid].astype('int64')*4+pred[valid]).ravel(),minlength=16).reshape(4,4)
                score['confusion']=confusion.tolist()
            if ident in {r['id'] for r in pairs}:
                pair=next(r for r in pairs if r['id']==ident);regions=[]
                for reg in pair['regions']:
                    m=mask(pair,reg);cls=1 if reg['object']=='rock' else 2
                    regions.append(dict(name=reg['name'],object=reg['object'],pixels=int(m.sum()),
                        correct_fraction=float((pred[m]==cls).mean()),wrong_tree_fraction=float((pred[m]==2).mean()) if cls==1 else None))
                score['regions']=regions
            scores[str(size)]=score
            if size==256:
                old=np.load(BASE/f'predictions/{ident}_mapping_fixed.npz')['prediction'] if (BASE/f'predictions/{ident}_mapping_fixed.npz').exists() else None
                if old is not None:assert np.array_equal(old,pred),'256baseline changed'
        rows.append(dict(id=ident,image=c['image'],scores=scores,scope='known_regression'));print('CASE',ident,flush=True)
    aggregate={}
    for size in p['scales']:
        c=sum(np.array(r['scores'][str(size)]['confusion']) for r in rows if 'confusion' in r['scores'][str(size)])
        union=c.sum(0)+c.sum(1)-c.diagonal();iou=[float(c[k,k]/union[k]) if union[k] else None for k in range(4)]
        aggregate[str(size)]=dict(confusion=c.tolist(),iou=iou,mean_iou=float(np.mean([v for v in iou if v is not None])))
    assert sha(p['checkpoint'])==p['checkpoint_sha256']
    write('results.json',dict(records=rows,known14=aggregate,duration_seconds=time.time()-start,
        new_model_trained=False,new_boundary_labels=0,adopted=False,independent_validation=False,robot_connected=False))
    render()
    print(json.dumps(aggregate),flush=True)
def render():
    result=read(ROOT/'results.json');pairs=read(ROOT/'pairs.json')['cases'];(ROOT/'views').mkdir(exist_ok=True)
    board=Image.new('RGB',(1600,1140),'#f5f2e9');d=ImageDraw.Draw(board);font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
    for i,pair in enumerate(pairs):
        im=Image.open(pair['image']).convert('RGB');marked=im.copy();draw=ImageDraw.Draw(marked)
        for reg in pair['regions']:draw.line([tuple(q) for q in reg['points']]+[tuple(reg['points'][0])],fill='#b67d28' if reg['object']=='rock' else '#237b54',width=2)
        marked.save(ROOT/f'views/{pair["id"]}_reference.png')
        images=[marked]+[Image.fromarray(COLORS[np.load(ROOT/f'predictions/{pair["id"]}_{s}.npz')['prediction']]) for s in [256,512,768]]
        for j,(pic,title) in enumerate(zip(images,['来源与复用核心参考','冻结模型256','冻结模型512','冻结模型768'])):
            factor=min(380/pic.width,305/pic.height);pic=pic.resize((round(pic.width*factor),round(pic.height*factor)),Image.Resampling.NEAREST)
            board.paste(pic,(j*400+8,i*380+40));d.text((j*400+8,i*380+7),title,font=font(20),fill='#243d35')
        d.text((8,i*380+351),pair['id'],font=font(17),fill='#765139')
    board.save(ROOT/'comparison.jpg',quality=94)
    tables='<table><tr><th>输入尺寸</th><th>已知14图四类平均IoU</th></tr>'+''.join(f'<tr><td>{s}</td><td>{result["known14"][str(s)]["mean_iou"]*100:.1f}%</td></tr>' for s in [256,512,768])+'</table>'
    regional=[]
    for pair in pairs:
        row=next(r for r in result['records'] if r['id']==pair['id'])
        for idx,reg in enumerate(pair['regions']):
            regional.append(dict(id=pair['id'],region=reg['name'],object=reg['object'],correct_fraction={str(s):row['scores'][str(s)]['regions'][idx]['correct_fraction'] for s in [256,512,768]}))
    write('regional_summary.json',regional)
    tables+='<h2>既有核心参考对应比例（非专家准确率）</h2><div class="scroll"><table><tr><th>作品／区域</th><th>256</th><th>512</th><th>768</th></tr>'+''.join('<tr><td>'+html_escape(r['id']+'/'+r['region'])+'</td>'+''.join(f'<td>{r["correct_fraction"][str(s)]*100:.1f}%</td>' for s in [256,512,768])+'</tr>' for r in regional)+'</table></div>'
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>树石核心配对与输入尺寸诊断</title><style>body{font:17px/1.7 system-ui;max-width:1600px;margin:auto;padding:24px;background:#f5f2e9;color:#243d35}img{max-width:100%;height:auto}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #bbb}.scroll{overflow-x:auto}.notice{padding:16px;background:#e8dec7}h1{font-size:30px}</style><h1>树石核心配对与输入尺寸诊断</h1><p class="notice">复用三组助手核心参考；两者之间仍未知，不是假定的精确边界。无新增专家标签、训练权重或独立测试作品。下图是冻结模型原始对象分区，不是笔触重绘。</p><p>棕为山石，绿为树木，米白为空景，紫为其他。所有尺寸直接输入完整原图，未使用参考裁剪、融合或人工遮罩。三个尺寸全部展示，不以局部最好结果替换旧版。</p>TABLE<img src="comparison.jpg"><p><a href="pairs.json">核心配对及来源</a> · <a href="protocol.json">冻结协议</a> · <a href="results.json">全部逐图结果</a> · <a href="verification.json">核查记录</a></p></html>'''
    (ROOT/'index.html').write_text(doc.replace('TABLE',tables),encoding='utf8')
def html_escape(x):
    import html
    return html.escape(x)
if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:run()
