"""Truthful data-count audit and report for the simple qinglu/rock run."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .runtime import LAB
from .narrow_collect import CORPUS,save_json


def audit_data(manifest):
    records=manifest['real'];splits=['train','validation','test']
    group_sets={s:{r['group'] for r in records if r['split']==s} for s in splits}
    for i,a in enumerate(splits):
        for b in splits[i+1:]:assert not(group_sets[a]&group_sets[b])
    parent_image={};parent_page={}
    for r in records:
        assert parent_image.setdefault(r['parent_image'],r['split'])==r['split']
        assert parent_page.setdefault(r['parent_page_group'],r['split'])==r['split']
        x,y,x1,y1=r['crop_box_original_pixels'];w,h=r['source']['dimensions']
        assert 0<=x<x1<=w and 0<=y<y1<=h
        with Image.open(r['image']) as im:assert list(im.size)==r['size']==[320,320]
        with np.load(r['labels']) as data:
            assert data['targets'].shape==data['weights'].shape==(2,320,320)
    return dict(painting_artist_album_groups_disjoint=True,parent_photos_disjoint=True,parent_pages_disjoint=True,
                all_crop_coordinates_in_bounds=True,label_shapes_correct=True,real_crops=len(records),
                real_split_counts=dict(Counter(r['split'] for r in records)),
                independent_group_counts={s:len(group_sets[s]) for s in splits},
                source_photos_used=len(parent_image),parent_catalogue_pages_used=len(parent_page),
                human_expert_dense_labels=0)


def preflight(manifest_path):
    import torch
    from .model import GongbiLineNet,segmentation_loss
    from .data import TileSampler
    torch.set_num_threads(4)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    checks=audit_data(manifest)
    assert manifest['validation_groups'] and manifest['test_groups']
    sampler=TileSampler('train',320,manifest_path,'v1')
    (x,y,w),_=sampler.batch(np.random.default_rng(3022),4,.8,True)
    model=GongbiLineNet(pretrained=True).cuda()
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=segmentation_loss(model(x.cuda()),y.cuda(),w.cuda())
    loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
    assert torch.isfinite(loss) and torch.isfinite(gradient)
    checks.update(bf16_real_sample_forward_backward_finite=True,loss=float(loss.detach()),gradient_norm=float(gradient))
    save_json(CORPUS/'preflight_checks.json',checks)
    print(json.dumps(checks,ensure_ascii=False,indent=2),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='mobilenet_unet_narrow_v03');parser.add_argument('--preflight',type=Path);parser.add_argument('--review-note',default='Pending visual review');args=parser.parse_args(argv)
    if args.preflight:return preflight(args.preflight)
    run=LAB/'runs'/args.run
    manifest=json.loads((run/'dataset_manifest_snapshot.json').read_text(encoding='utf-8'))
    config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    summary=json.loads((run/'training_summary.json').read_text(encoding='utf-8'))
    collection=json.loads((CORPUS/'downloaded_sources.json').read_text(encoding='utf-8'))
    selected=json.loads((CORPUS/'selected_sources.json').read_text(encoding='utf-8'))
    inference=json.loads((run/'inference'/'summary.json').read_text(encoding='utf-8'))
    test_metrics_path=run/'heldout_test_metrics.json'
    test_metrics=json.loads(test_metrics_path.read_text(encoding='utf-8')) if test_metrics_path.exists() else None
    checks=audit_data(manifest)
    checks.update(completed_iterations=summary['completed_steps'],successful_updates=summary['successful_optimizer_updates'],
                  skipped_updates=summary['skipped_optimizer_updates'],
                  model_weight_files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob('*.pt')},
                  code_snapshot_matches=all(hashlib.sha256((run/'source_snapshot'/name).read_bytes()).hexdigest()==sha for name,sha in config['source_sha256'].items()),
                  expert_accuracy_available=False,ready_for_robot=False)
    checks['all_heldout_crops_evaluated']=test_metrics['real_test_crops'] if test_metrics else 0
    for r in inference:
        export=json.loads((run/'inference'/r['id']/'paths.json').read_text(encoding='utf-8'))
        assert all(0<=x<export['width'] and 0<=y<export['height'] for path in export['paths'] for x,y in path)
        with Image.open(run/'inference'/r['id']/'structure.png') as im:assert im.size==(export['width'],export['height'])
    checks['all_exported_path_coordinates_valid']=True
    assert summary['head_mean_absolute_parameter_update']>1e-6
    import torch
    checks['weights_finite']={}
    for path in run.glob('*.pt'):
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        checks['weights_finite'][path.name]=all(bool(torch.isfinite(t).all()) for t in checkpoint['model'].values())
    assert checks['code_snapshot_matches'] and all(checks['weights_finite'].values())
    save_json(run/'audit.json',checks)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    for split,name in [('train','training_examples'),('validation','validation_examples'),('test','test_examples')]:
        chosen=[r for r in inference if r['split']==split]
        if not chosen:continue
        canvas=Image.new('RGB',(1500,80+len(chosen)*735),(246,244,237));draw=ImageDraw.Draw(canvas)
        draw.text((15,15),{'train':'训练组参考，不能作为泛化证明','validation':'独立分组验证，只有自动弱标签','test':'未参与本轮训练和选权重的留出组，暂无专家真值'}[split],font=font,fill=(30,45,35))
        for i,r in enumerate(chosen):
            im=Image.open(run/'inference'/r['id']/'comparison.png');im.thumbnail((1500,735));canvas.paste(im,(0,75+i*735))
        canvas.save(run/(name+'.jpg'),quality=93)
    rock_examples=[r for r in inference if '_rock_' in r['id']]
    if rock_examples:
        examples=rock_examples[:8]
        gallery=Image.new('RGB',(1280,100+((len(examples)+1)//2)*350),(246,244,237));draw=ImageDraw.Draw(gallery)
        draw.text((15,12),'真实山石输入 / 模型预测；分组见每项标题',font=font,fill=(30,45,35))
        all_records={r['id']:r for r in manifest['real']}
        for index,r in enumerate(examples):
            x=(index%2)*640;y=75+(index//2)*350
            draw.text((x+10,y),{'train':'训练组','validation':'验证组','test':'留出测试组'}[r['split']],font=font,fill=(30,45,35))
            paths=[Path(all_records[r['id']]['image']),run/'inference'/r['id']/'structure.png']
            for col,path in enumerate(paths):
                im=Image.open(path);im.thumbnail((300,300));gallery.paste(im,(x+10+col*315,y+35))
        gallery.save(run/'rock_predictions.jpg',quality=94)
    reference_lookup={r['id']:r for r in manifest['inference_records']}
    for reference in ['user_qianli_02','user_qianli_05','user_qianli_06']:
        previous=LAB/'runs'/'mobilenet_unet_v02'/'inference'/reference/'structure.png'
        current=run/'inference'/reference/'structure.png'
        if not previous.exists() or not current.exists():continue
        source=Image.open(reference_lookup[reference]['image']).convert('RGB')
        old=Image.open(previous).convert('RGB');new=Image.open(current).convert('RGB')
        assert old.size==new.size==source.size
        panel=Image.new('RGB',(1800,900),(246,244,237));draw=ImageDraw.Draw(panel)
        draw.text((15,12),'同一《千里江山图》训练参考：原图 / 上一版 / 简单山石训练版',font=font,fill=(30,45,35))
        for i,(title,im) in enumerate([('原图',source),('上一版：细节较多',old),('本轮：主轮廓为主',new)]):
            draw.text((i*600+15,55),title,font=font,fill=(30,45,35));im.thumbnail((570,770));panel.paste(im,(i*600+15+(570-im.width)//2,100+(770-im.height)//2))
        panel.save(run/(reference+'_previous_comparison.png'))
    sources='\n'.join('- '+r['id']+' | '+r['title']+' | '+r['page_url'] for r in selected['records'])
    (run/'SOURCES.md').write_text('# Selected real-image sources\n\n'+sources,encoding='utf-8')
    card=f'''# 简单青绿山石轮廓：第三轮训练

## 这次收窄了什么
只使用新采集、自动筛选的青绿山水照片中的山体/山石局部，以及简单单山石合成配对。未混入上一轮那些细密水墨图。目标是外轮廓与少量主要转折线，辅助皴线单独保存。

## 实际数量（不同口径不能混用）
- 新发现203个作品目录页面，2208个公开预览图地址。
- 成功下载{collection['downloaded_files']}个文件；完全去重后{collection['unique_files']}张，来自{collection['painting_page_groups']}个页面分组。
- 风格、色彩、清晰度、纹理密度与近重复筛选后保留{selected['selected_files']}张照片，来自{selected['page_groups']}个页面分组。
- 实际用于切块的照片{checks['source_photos_used']}张、页面{checks['parent_catalogue_pages_used']}个，生成真实山石裁块{len(manifest['real'])}个：{checks['real_split_counts']}。
- 这些不是{len(manifest['real'])}幅独立画作。整图、长卷分段与局部存在同一作品关系；按已知作者、册页和近重复链接做保守分组。分组数量：{checks['independent_group_counts']}。
- 简单几何合成配对：1024训练、128验证，单独统计，不能称作真实作品。
- 专家完整线稿数量仍为0。真实标签为自动支持边界及最多3条长皴线候选，不是人工标准答案。CLIP仅用于样本内容筛选，不用于生成所展示的最终预测。

## 已实际训练
- {summary['completed_steps']}次迭代；成功更新{summary['successful_optimizer_updates']}次；跳过{summary['skipped_optimizer_updates']}次。
- 从 {config['initialization_checkpoint']} 初始化；{config['gpu']}；BF16；耗时{summary['elapsed_seconds']:.1f}秒。
- 数据清单与代码快照保存在本运行目录；原有两版权重与数据未覆盖。数据、缓存和权重全部在E盘，源码在F盘。
- 原始概率图、结构PNG、SVG、路径JSON在inference/。本轮选权重仅使用训练/验证组，test组没有用于本轮训练或本轮选权重。初始化继承第一版已有训练知识，见父权重和前轮数据记录。
- 预览best.pt来自本轮迭代 {sorted({r['checkpoint_step'] for r in inference})}，last.pt为最后迭代。

## 效果与限制
{args.review_note}

全部留出裁块评估数量：{test_metrics['real_test_crops'] if test_metrics else 0}；预测与自动标签的一致度：{test_metrics['weak_label_agreement'] if test_metrics else '尚未评估'}。这不是专家判定的提线准确率。全部留出图的原图/预测概率预览按固定顺序保存于heldout_test_all/，没有按观感挑图，也不据此改选权重。

自动标签有漏提、折线化和色块边界误差；增加图像数量不等于获得精确真值。验证指标只能表示与这些标签的一致度，合成指标只表示简单几何分布。尚不能宣称真实工笔提线准确率，也没有毛笔压力、浓淡干湿标定，不接入机器人生产流程。

## 查看与复现
- training_examples.jpg / validation_examples.jpg / test_examples.jpg：参考、验证、测试分别展示。
- audit.json：来源分组、裁切范围、标签尺寸、成功更新、权重有限性、哈希检查。
- SOURCES.md：每个选入原始照片的来源；完整候选及排除理由在narrow_corpus_v03/。
- narrow_corpus_v03/previews/：真实照片样本、山石裁块及自动标签对照。
- 来源保留公开页面与图片链接，只采集页面已公开展示的预览图，不访问付费原图下载。

在项目根目录：
```powershell
.\\scripts\\gongbi_lab.ps1 infer --run {run.name}
.\\scripts\\gongbi_lab.ps1 narrow_report --run {run.name}
```
训练参数以config.json为准，复现使用新运行名。
'''
    (run/'MODEL_CARD.md').write_text(card,encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False,indent=2),flush=True)
