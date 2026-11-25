# -*- coding: utf-8 -*-
"""Local pavilion reconstruction + pine/rock controls, source-derived widths.
Explicit development ROIs, no semantic generalization or physical rendering claim.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,hashlib,colorsys,time
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
from source_stroke_width_20261104 import CONFIG,reconstruct_lines,width_profile,render_profiles
from module_stroke_core_20261104 import samples,length

ROOT=LAB/'pavilion_source_strokes_20261104'
BASE=LAB/'module_stroke_planner_20261104/outputs/G03_01'
CASES=[dict(id='pavilion',title='亭子',box=[108,198,231,299],architecture=True,
 polygon=[[171,205],[181,219],[196,232],[227,239],[225,251],[215,255],[214,289],[135,291],[132,256],[120,251],[119,239],[150,230],[166,219]]),
 dict(id='pine',title='松针局部',box=[278,121,488,240],architecture=False),
 dict(id='rock',title='山石局部',box=[178,285,326,389],architecture=False)]
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def convert(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(str(type(v)))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ROOT.mkdir(exist_ok=True);cv2.setNumThreads(2)
    image=np.array(Image.open(BASE/'source.png').convert('RGB'));document=json.loads((BASE/'offline_plan.json').read_text('utf8'))
    results=[]
    for case in CASES:
        start=time.monotonic();folder=ROOT/case['id'];folder.mkdir(exist_ok=True)
        x0,y0,x1,y1=case['box'];roi=np.zeros(image.shape[:2],np.uint8)
        if 'polygon' in case:cv2.fillPoly(roi,[np.array(case['polygon'],np.int32)],1)
        else:roi[y0:y1,x0:x1]=1
        ids=[];old=[]
        for i,row in enumerate(document['curves']):
            p=samples(row['points'],.5);q=np.rint(p).astype(int);q=np.clip(q,0,495)
            if roi[q[:,1],q[:,0]].mean()>.85:
                ids.append(i);old.append(np.asarray(row['points'],float))
        print(case['id'],'input',len(old),flush=True)
        if case['architecture']:new,diagnostics=reconstruct_lines(image,old)
        else:
            new=[dict(points=p,source_curve_ids=[i],artist_stroke_identity_verified=False) for i,p in enumerate(old)]
            diagnostics=dict(input_curves=len(old),output_curves=len(old),architecture_rules_applied=False,
                all_input_curve_ids_accounted=True,reason='control: do not impose architectural straightness on pine/rock')
        profiles_old=[width_profile(image,p) for p in old]
        # Architecture paths already underwent bounded recentering in grouping.
        # Do not silently shift them again while estimating their widths.
        profiles=[width_profile(image,r['points'],recenter=not case['architecture']) for r in new]
        assert sorted(i for r in new for i in r['source_curve_ids'])==list(range(len(old)))
        # Measure in original pixels. Translate only after width inference.
        origin=np.array([x0,y0]);shape=(y1-y0,x1-x0)
        def local(profiles):return [{**p,'points':p['points']-origin} for p in profiles]
        old_local=local(profiles_old);new_local=local(profiles)
        exact_old=[{**p,'points':p['original_sample_points']-origin} for p in profiles_old]
        fixed,fixed_count,_,_=render_profiles(shape,exact_old,flatten_width_px=496/186.8)
        same_old,same_old_count,_,_=render_profiles(shape,old_local)
        old_centers,_,_,_=render_profiles(shape,exact_old,flatten_width_px=.4)
        units=np.full((shape[0]*4,shape[1]*4,3),255,np.uint8)
        for i,p in enumerate(new_local):
            color=tuple(int(v*200) for v in colorsys.hsv_to_rgb(i*.618%1,.8,1))
            cv2.polylines(units,[np.rint(p['points']*4).astype(np.int32)],False,color,2,cv2.LINE_AA)
        relative,count,_,uncertain=render_profiles(shape,new_local)
        candidate,new_count,events,_=render_profiles(shape,new_local,relieve_junctions=True)
        native=Image.fromarray(image[y0:y1,x0:x1]);native.save(folder/'source_native.png')
        for name,im in [('previous_1mm.png',fixed),('old_source_width.png',same_old),('old_centers.png',old_centers),('units.png',units),
                        ('source_width.png',relative),('junction_candidate.png',candidate),('uncertain_width.png',uncertain)]:
            Image.fromarray(im).save(folder/name)
        all_widths=np.concatenate([p['width_px'][p['observed']] for p in profiles if p['observed'].any()])
        valid=sum(int(p['observed'].sum()) for p in profiles);total=sum(len(p['observed']) for p in profiles)
        def overlap(c):return float((c>=2).sum()/max(1,(c>0).sum()))
        stats=dict(id=case['id'],title=case['title'],source_size=[496,496],source_crop_size=list(native.size),
            input_curves=len(old),new_curve_candidates=len(new),old_curve_ids_accounted=len(old),
            no_short_details_deferred=True,source_width_observed_samples=valid,source_width_total_samples=total,
            original_detail_completeness_not_established=True,
            contact_count_reduction_fraction=1-len(new)/max(1,len(old)),
            centerline_recentering_applied=True,architecture_width_pass_recenters_again=False,
            observed_width_fraction=valid/max(1,total),fully_unobserved_curves=sum(not p['observed'].any() for p in profiles),
            source_width_px_quantiles=np.quantile(all_widths,[.1,.5,.9]).tolist(),
            image_target_width_mm_quantiles=(np.quantile(all_widths,[.1,.5,.9])*186.8/496).tolist(),
            image_target_mm_is_only_comparison_scale=True,subpixel_width_precision_not_established=True,
            geometric_overlap=dict(previous_1mm=overlap(fixed_count),old_paths_source_width=overlap(same_old_count),source_width=overlap(count),junction_candidate=overlap(new_count)),
            junction_relief_prior_factor=.70,physical_pressure_mm=None,current_brush_calibrated=False,
            robot_connected=False,ready_for_robot=False,physical_blot_reduction=None,artist_stroke_accuracy=None,
            manual_nonexpert_development_roi=True,new_image_test=False,architecture_only_for_pavilion=case['architecture'],
            inference_seconds=time.monotonic()-start)
        save(folder/'statistics.json',stats)
        (folder/'candidate_strokes.json').write_text(json.dumps(dict(case=case,source=str(BASE/'source.png'),source_sha256=sha(BASE/'source.png'),
            original_curve_indices=ids,original_paths=old,new_candidates=new,width_profiles=profiles,
            junction_targets=events,diagnostics=diagnostics,parameter_config=CONFIG,physical_pressure_mm=None,
            no_physical_coordinates=True,ready_for_robot=False),ensure_ascii=False,indent=2,default=convert),encoding='utf8')
        labels=['原图','旧中心线（统一细线显示）','结构续接候选（颜色分笔）','原图相对笔宽目标','交汇处理候选（未标定）']
        ims=[native,Image.fromarray(old_centers),Image.fromarray(units),Image.fromarray(relative),Image.fromarray(candidate)]
        board=Image.new('RGB',(2050,415),'#f4f1e8');d=ImageDraw.Draw(board)
        for i,(im,label) in enumerate(zip(ims,labels)):
            factor=min(400/im.width,305/im.height);im=im.resize((max(1,round(im.width*factor)),max(1,round(im.height*factor))),Image.Resampling.LANCZOS)
            board.paste(im,(i*410,42));d.text((i*410+5,9),label,font=FONT,fill='black')
        d.text((8,355),f'{case["title"]}：{len(old)} → {len(new)}条候选；可测横截面{valid/max(1,total):.1%}（不是准确率）；原输入均可追溯。',font=FONT,fill='black')
        d.text((8,383),'图像目标与几何预览，不是毛笔实画、墨水模拟或训练准确率；当前纸笔墨的下压映射仍为空。',font=FONT,fill='black')
        board.save(folder/'comparison.jpg',quality=96);results.append(stats)
        print(case['id'],'output',len(new),'width_valid',round(valid/max(1,total),3),flush=True)
    save(ROOT/'results.json',dict(records=results,model_weights_trained=False,robot_connected=False,
        no_new_artist_ground_truth=True,source_file_sha256=sha(BASE/'source.png')))
    cards=''.join(f'<h2>{r["title"]} · {r["input_curves"]}→{r["new_curve_candidates"]}条候选</h2><img src="{r["id"]}/comparison.jpg"><p><a href="{r["id"]}/previous_1mm.png">旧中心线＋统一1mm假设</a> · <a href="{r["id"]}/old_source_width.png">旧线段＋原图笔宽（对照）</a> · <a href="{r["id"]}/source_width.png">续接线段＋原图笔宽</a> · <a href="{r["id"]}/uncertain_width.png">橙色显示不确定横截面</a> · <a href="{r["id"]}/statistics.json">统计</a> · <a href="{r["id"]}/candidate_strokes.json">候选与追溯</a></p>' for r in results)
    page='<meta charset="utf-8"><title>亭子完整笔画与相对粗细</title><style>body{font:17px/1.65 system-ui;max-width:2000px;padding:22px;margin:auto;background:#f4f1e8;color:#253b30}img{width:100%;background:white}p{max-width:1500px}.note{padding:15px;background:#e0e6da}</style><h1>亭子完整笔画、相对粗细与交汇 · 离线局部对照</h1><p class="note">本轮使用显式开发区域，亭子允许同轴结构合并，松针和山石只作笔宽处理对照。不采用一刀切的固定笔宽，不删除短细节。图像横截面可见宽度不等于毛笔实测宽度；无下压量或可执行机械臂指令。交汇收窄70%是待验证设计。</p>'+cards
    page+='<p class="note">原图只有496×496，亭子约123×101像素，不能精确恢复更细瓦线。橙色位置为没有可靠横截面测量的区域；纯黑预览中部分宽度由同笔邻域插值或1像素显示占位获得。候选分笔尚未经画家验证，不能作为完整艺术家笔画真值。</p>'
    page+='<h2>本轮判断</h2><p>结构续接减少了部分起落笔；屋顶仍有碎段，松针和山石保留了原候选，没有证明细节完整。相对宽度预览仍存在密集覆盖，不能认定能消除实画墨团。交汇收窄只改变几何目标，没有消除重复落墨的实测证据。本轮不替换上机任务，不生成下压量，不宣称模型完成了新训练。</p><p>下一步先获取能分清瓦线的高分辨率输入，再对同一条墨带的两侧候选作去重；保留有亮缝的相邻真线。上机前仍须用当前毛笔、纸墨完成细线、交汇和连续曲线局部验证。</p>'
    (ROOT/'index.html').write_text(page,encoding='utf8');print('REPORT',ROOT/'index.html',flush=True)

if __name__=='__main__':main()
