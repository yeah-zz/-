# -*- coding: utf-8 -*-
"""Independent GPU entry with calibrated retry and explicit v4 fallback."""
from evaluate_simple_landscape_specialist_r2 import *
from pathlib import Path
import argparse,datetime,html

def choose(model,rgb,selection):
    pred=predict(model,rgb).astype(np.float16).astype(np.float32);mask=center_mask(pred,selection['selected']['threshold'],selection['selected']['method']);paths,_=vectorize(mask,pred,rgb);mask=baseline.raster(rgb.shape[:2],paths);q=qa(mask,rgb)
    chosen_mask,chosen_paths,chosen_q=mask,paths,q;retried=False
    if not passes(q,selection['qa_gates']):
        retried=True;low=center_mask(pred,selection['retry_threshold'],selection['selected']['method']);pp,_=vectorize(low,pred,rgb);low=baseline.raster(rgb.shape[:2],pp);qq=qa(low,rgb)
        score=lambda v:v['image_support']+.6*v['source_pixel_coverage']-.03*v['small_closed_regions']
        if passes(qq,selection['qa_gates']) or score(qq)>score(q):chosen_mask,chosen_paths,chosen_q=low,pp,qq
    return dict(raw_mask=mask,raw_paths=paths,raw_qa=q,mask=chosen_mask,paths=chosen_paths,qa=chosen_q,retried=retried,passed=passes(chosen_q,selection['qa_gates']))

def run(source,output):
    source=Path(source).resolve();output=Path(output)
    if not output.is_absolute() or output.drive.upper() not in ['E:','F:']:raise ValueError('Use E/F absolute output')
    selection=read(ROOT/'selection.json')
    if digest(ROOT/'specialist.pt')!=selection['checkpoint_sha256']:raise RuntimeError('Specialist checkpoint changed')
    rgb=np.asarray(Image.open(source).convert('RGB'))
    if max(rgb.shape[:2])>2600:raise ValueError('Use image <=2600px long edge; no silent resize')
    torch.set_num_threads(4);cv2.setNumThreads(2);model=load_model();candidate=choose(model,rgb,selection)
    baseline.run(source,output/'v4');doc=read(output/'v4'/'paths.json');bp=[np.asarray(r['points'],np.float32) for r in doc['paths']];bm=baseline.raster(rgb.shape[:2],bp)
    finalmask=candidate['mask'] if candidate['passed'] else bm;finalpaths=candidate['paths'] if candidate['passed'] else bp
    write_lines(output/'raw_neural',rgb,candidate['raw_mask'],candidate['raw_paths'],candidate['raw_qa'],'trained specialist raw')
    write_lines(output/'retried_neural',rgb,candidate['mask'],candidate['paths'],candidate['qa'],'trained specialist fixed retry')
    write_lines(output/'selected',rgb,finalmask,finalpaths,candidate['qa'],'specialist' if candidate['passed'] else 'v4 fallback, QA flag unresolved')
    audit=dict(source=str(source),source_sha256=digest(source),checkpoint_sha256=digest(ROOT/'specialist.pt'),selection_sha256=digest(ROOT/'selection.json'),automatic_rule_pass=candidate['passed'],fallback_to_v4=not candidate['passed'],retried=candidate['retried'],qa=candidate['qa'],qa_not_expert_acceptance=True,reference_labels_used=False,robot_connected=False,ready_for_robot=False)
    jsave(output/'run.json',audit)
    panels=''.join(f'<figure><figcaption>{label}</figcaption><a href="{p}"><img src="{p}"></a></figure>' for p,label in [('selected/input.png','原图'),('v4/lines.png','v4 对照'),('raw_neural/lines.png','专用网络原始输出'),('selected/lines.png','质检与回退后的选择')])
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>简洁山水专用模型试验</title><style>body{font:16px system-ui;background:#f3f1eb;margin:24px}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}figure{margin:0}img{width:100%}</style><h1>简洁山水专用模型试验</h1><p>自动规则通过：'+str(candidate['passed'])+'；回退到 v4：'+str(not candidate['passed'])+'。规则通过不等于专家确认；所有结果为离线像素中心线。</p><main>'+panels+'</main><p><a href="selected/lines.svg">选择结果 SVG</a> · <a href="run.json">运行记录</a></p>',encoding='utf-8')
    print('SPECIALIST_ENTRY_READY',str(output/'index.html'),json.dumps(audit),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--output');a=p.parse_args();out=Path(a.output) if a.output else ROOT/'manual_outputs'/(Path(a.image).stem+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));run(a.image,out)
