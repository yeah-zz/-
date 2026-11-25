"""Actual learned predictions, uncurated heldout gallery, audit and model card."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from .runtime import LAB
from .model import GongbiLineNet
from .infer import predict,paths_from_probability
from .cunfa_collect import CORPUS

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--run',default='mobilenet_unet_rock_cunfa_v04_clean');p.add_argument('--checkpoint',default='best.pt');p.add_argument('--validation-only',action='store_true');args=p.parse_args(argv)
    torch.set_num_threads(4);cv2.setNumThreads(2)
    run=LAB/'runs'/args.run;out=run/('validation_preview' if args.validation_only else 'evaluation');out.mkdir(exist_ok=True)
    ckpt=torch.load(run/args.checkpoint,map_location='cuda',weights_only=False)
    model=GongbiLineNet().cuda();model.load_state_dict(ckpt['model']);model.eval()
    assert ckpt['config']['channels']==['all_ink_coverage_including_cunfa','continuous_ink_density']
    manifest=json.loads((run/'dataset_manifest_snapshot.json').read_text(encoding='utf-8'))
    chosen_train={'ref_02_00','ref_06_00','ref_12_00','ref_16_00','ref_16_01','web_29_043_c0','web_28_022_c0'}
    records=[r for r in manifest['real'] if r['split']!='train' or r['id'] in chosen_train]
    if args.validation_only:records=[r for r in records if r['split']=='validation']
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    rows=[];metric=dict(tp=0,fp=0,fn=0,density_mae_sum=0.,valid_pixels=0)
    held_panels=[]
    for r in records:
        rgb=np.asarray(Image.open(r['image']).convert('RGB'));h,w=rgb.shape[:2]
        # A real paper margin protects image borders; never delete learned lines
        # by hand for particular examples. The same padding is used everywhere.
        padded=np.pad(rgb,((32,32),(32,32),(0,0)),mode='edge')
        probability=predict(model,padded,'cuda',tile=ckpt['config']['tile'])[:,32:32+h,32:32+w]
        folder=out/r['id'];folder.mkdir(exist_ok=True)
        Image.fromarray(np.rint(probability[0]*255).astype(np.uint8)).save(folder/'ink_coverage_probability.png')
        Image.fromarray(np.rint(probability[1]*255).astype(np.uint8)).save(folder/'ink_density_probability.png')
        # Shared global floor for display only; probability files stay unaltered.
        opacity=np.maximum(probability[1]-.035,0)/.965
        opacity*=np.clip((probability[0]-.12)/.2,0,1)
        ink=np.rint(255*(1-opacity)).astype(np.uint8)
        Image.fromarray(ink).save(folder/'ink.png')
        paths,line=paths_from_probability(probability[0],low=.38,high=.60,min_length=9)
        Image.fromarray(line).save(folder/'centerlines.png')
        radius=cv2.distanceTransform((probability[0]>.5).astype(np.uint8),cv2.DIST_L2,5)
        path_rows=[]
        for path in paths:
            points=np.asarray(path,np.int32);xs=np.clip(points[:,0],0,w-1);ys=np.clip(points[:,1],0,h-1)
            path_rows.append(dict(xy=path,width_pixels=(radius[ys,xs]*2).round(2).tolist(),density=probability[1,ys,xs].round(3).tolist()))
        payload=dict(id=r['id'],split=r['split'],source_group=r['group'],checkpoint_step=ckpt['step'],paths=path_rows,
                     units='image pixels; density normalized 0..1',stroke_order_known=False,ready_for_robot=False,
                     notes='Skeleton branches are geometry fragments, NOT recovered human brush strokes; no Z/force calibration exists.')
        (folder/'paths.json').write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
        svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><g stroke="black" fill="none" stroke-width="1">']
        for path in paths:svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in path)+'"/>')
        svg.append('</g></svg>');(folder/'centerlines.svg').write_text('\n'.join(svg),encoding='utf-8')
        panel=Image.new('RGB',(1500,620),'#f5f5f0');d=ImageDraw.Draw(panel)
        role='训练示例，不能证明泛化' if r['split']=='train' else '整组留出测试，未参与训练' if r['split']=='test' else '整组留出验证，用于选权重'
        d.text((15,8),r['id']+' | '+role+' | 模型第 '+str(ckpt['step'])+' 步',font=font,fill='black')
        for j,(label,array) in enumerate((('输入山石示范',rgb),('模型墨线：保留宽度和浓淡',ink),('中心线辅助预览：不等于人类笔顺',line))):
            d.text((j*500+12,43),label,font=small,fill='black')
            im=Image.fromarray(array).convert('RGB');im.thumbnail((474,535));panel.paste(im,(j*500+12+(474-im.width)//2,80+(535-im.height)//2))
        panel.save(folder/'comparison.jpg',quality=94)
        if r['split']=='test':
            with np.load(r['labels']) as f:target=f['targets']/255.;weight=f['weights']/255.
            valid=weight[0]>.1;pred=probability[0]>.5;truth=target[0]>.3
            metric['tp']+=int((pred&truth&valid).sum());metric['fp']+=int((pred&~truth&valid).sum());metric['fn']+=int((~pred&truth&valid).sum())
            metric['density_mae_sum']+=float((np.abs(probability[1]-target[1])*valid).sum());metric['valid_pixels']+=int(valid.sum())
            held_panels.append((r,rgb,ink))
        rows.append(dict(id=r['id'],split=r['split'],group=r['group'],path_fragments=len(paths),comparison=str(folder/'comparison.jpg')))
        print('CUNFA_PREDICTION',r['id'],r['split'],len(paths),flush=True)
    if args.validation_only:
        (out/'summary.json').write_text(json.dumps(dict(checkpoint_step=ckpt['step'],records=rows),ensure_ascii=False,indent=2),encoding='utf-8')
        print('VALIDATION_PREVIEW_COMPLETE',str(out),flush=True)
        return
    # Every heldout test appears, sorted in manifest order; no best-case selection.
    canvas=Image.new('RGB',(1200,len(held_panels)*245),'#f5f5f0');d=ImageDraw.Draw(canvas)
    for i,(r,rgb,ink) in enumerate(held_panels):
        d.text((12,i*245+3),r['id']+' | 留出测试 | 原图 / 模型墨线',font=font,fill='black')
        for j,arr in enumerate((rgb,ink)):
            im=Image.fromarray(arr).convert('RGB');im.thumbnail((570,208));canvas.paste(im,(j*600+(570-im.width)//2,i*245+32))
    canvas.save(out/'all_heldout_tests.jpg',quality=94)
    # Compact representative contact sheet, fixed IDs selected before training.
    selected=['ref_11_04','ref_17_00','ref_17_01','ref_13_00']
    compact=Image.new('RGB',(1200,len(selected)*295),'#f5f5f0');d=ImageDraw.Draw(compact)
    for i,name in enumerate(selected):
        r,rgb,ink=next(item for item in held_panels if item[0]['id']==name)
        d.text((12,i*295+3),name+' | 未参与训练 | 原图 / 模型墨线',font=font,fill='black')
        for j,arr in enumerate((rgb,ink)):
            im=Image.fromarray(arr).convert('RGB');im.thumbnail((570,254));compact.paste(im,(j*600+(570-im.width)//2,i*295+32))
    compact.save(out/'heldout_preview.jpg',quality=95)
    precision=metric['tp']/max(1,metric['tp']+metric['fp']);recall=metric['tp']/max(1,metric['tp']+metric['fn'])
    metric.update(weak_mask_precision=precision,weak_mask_recall=recall,weak_mask_f1=2*precision*recall/max(1e-9,precision+recall),
                  weak_density_mae=metric['density_mae_sum']/max(1,metric['valid_pixels']),
                  evaluated_test_examples=len(held_panels),test_source_groups=len({r['group'] for r,_,_ in held_panels}),
                  interpretation='Agreement with automatic paper-normalized targets ONLY; NOT expert line accuracy or robot readiness',human_dense_labels=0)
    (out/'metrics.json').write_text(json.dumps(metric,indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    audit=json.loads((CORPUS/'dataset_audit.json').read_text(encoding='utf-8'))
    summary=json.loads((run/'training_summary.json').read_text(encoding='utf-8'))
    sources=sorted({r['source_page'] for r in manifest['real'] if r.get('source_page')})
    (run/'SOURCES.md').write_text('# Public tutorial source pages\n\n'+ '\n'.join('- '+s for s in sources)+'\n\nPublic visibility is not an open license. Metadata preserves original pages; no blanket redistribution license is claimed.\n',encoding='utf-8')
    card=f'''# 单石与皴法墨线恢复 v04

任务：从已裁好的山石教学示范中提取墨迹，保留外轮廓及皴线的宽度、深浅。不是从整幅山水画中自动理解所有语义轮廓。

## 数据与训练

- 网页下载去重图片 {audit['downloaded_unique_web_images']} 张；其中含筛掉的无关图片，不等于训练画作数量。
- 实际使用 {audit['curated_examples']} 个裁图示例：网页 {audit['curated_web_examples']}，用户参考 {audit['curated_user_examples']}。
- 原始示例分布：{audit['examples_per_split']}；来源组数：{audit['groups_per_split']}。
- 配对退化版本：{audit['paired_variants_per_split']}；是同一示例的不同纸色/噪声版本，不是新增画作。
- 真实完成优化更新 {summary['successful_optimizer_updates']} 次，跳过 {summary['skipped_optimizer_updates']} 次；最终步数 {summary['completed_steps']}；选用权重第 {ckpt['step']} 步。
- 初始化：ImageNet MobileNetV3-small 编码器；新建解码器和墨线/墨色输出头。未续用前一轮青绿色块标签。

## 标签和评估限制

自动估计纸底、纸纹噪声、墨迹覆盖和灰度墨色作为弱标签，没有人工逐笔精描，也没有皴法类别的语义真值。淡皴漏标、纸斑残留和局部宽度偏差仍可能被模型继承。

验证预览已经发现：部分灰色/蓝色箭头和小字仍会进入墨线输出，不能声称自动识别并排除了全部文字；宽笔画及飞白转为骨架后存在分叉、环和碎段。教学页需要先裁出示范区。这一轮改善的是墨迹恢复，尚未解决可靠的人类笔触分解和笔顺恢复。

验证来源整组保留，连续步骤、近重复图和派生版本不跨组拆分。最终测试 {len(held_panels)} 个裁图、{metric['test_source_groups']} 个来源组；全部测试图展示在 evaluation/all_heldout_tests.jpg。弱标签 F1 {metric['weak_mask_f1']:.4f} 仅表示与自动标签的一致程度，不能作为实际绘画准确率。

## 输出

- ink_coverage_probability.png / ink_density_probability.png：未经人工修补的模型输出。
- ink.png：所有图统一应用显示阈值后的灰度墨线，保留笔画宽度和墨色。
- centerlines.svg / paths.json：辅助骨架碎段及像素宽度/墨色；分叉不是人类笔顺，无法唯一恢复真实笔触过程。
- 未标定毛笔宽度与 Z/压力、含墨量的关系，不可直接用于机械臂执行。整个流程未连接机械臂。

## 重现

项目 scripts/gongbi_lab.ps1 cunfa_collect --additional-only；cunfa_prepare --variants 6；train --name NEW_NAME --manifest "{LAB/'datasets'/'manifest_rock_cunfa_v04.json'}" --augmentation ink --precision bf16 --steps 3000 --batch 6 --tile 320 --real-probability .55 --warmup 120 --validation-real-patches 24；cunfa_evaluate --run NEW_NAME。

代码与清单快照、逐步训练日志、权重、全部原始概率图均保存在本次运行目录；数据/缓存/权重均在 E 盘。
'''
    (run/'MODEL_CARD.md').write_text(card,encoding='utf-8')
    shutil_audit=run/'dataset_audit.json';shutil_audit.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print('CUNFA_EVALUATION_COMPLETE',json.dumps(metric),str(out),flush=True)
