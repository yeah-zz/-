"""Auditable model/data card and inference contact sheets."""
import argparse
import hashlib
import json
from collections import Counter
from PIL import Image,ImageDraw,ImageFont
from .runtime import LAB,ROOT


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='mobilenet_unet_v01');parser.add_argument('--review-note',default='Pending visual review');args=parser.parse_args(argv)
    run=LAB/'runs'/args.run
    if (run/'comparison_audit.json').exists():
        from .evaluation import write_report
        return write_report(run,args.review_note)
    manifest=json.loads((LAB/'datasets'/'manifest.json').read_text(encoding='utf-8'))
    summary=json.loads((run/'training_summary.json').read_text(encoding='utf-8'))
    config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    inference=json.loads((run/'inference'/'summary.json').read_text(encoding='utf-8'))
    real=manifest['real'];groups=sorted({r['group'] for r in real})
    assert not(set(manifest['training_groups'])&set(manifest['validation_groups']))
    inventory=[]
    for r in real:
        source=r['source'];inventory.append(dict(id=r['id'],group=r['group'],split=r['split'],
            processed_size=r['size'],source=source,
            sparse_guidance=json.loads((LAB/'data'/'processed'/r['id']/'sparse_guidance.json').read_text(encoding='utf-8')) if (LAB/'data'/'processed'/r['id']/'sparse_guidance.json').exists() else None))
    (run/'data_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob('*.pt')}
    checks=dict(painting_group_split_disjoint=True,groups=len(groups),real_images=len(real),
        real_split_counts=dict(Counter(r['split'] for r in real)),synthetic_counts=dict(Counter(r['split'] for r in manifest['synthetic'])),
        expert_dense_labels=0,checkpoint_sha256=hashes,
        train_source_hashes_match={name:hashlib.sha256((ROOT/'gongbi_training'/name).read_bytes()).hexdigest()==sha for name,sha in config['source_sha256'].items()},
        training_snapshot_matches_config={name:hashlib.sha256((run/'source_snapshot'/name).read_bytes()).hexdigest()==sha for name,sha in config['source_sha256'].items()},
        user_crops_all_one_training_group=len({r['group'] for r in real if r['id'].startswith('user_')})==1)
    (run/'audit.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
    for split,name in [('train','reference_predictions'),('validation','heldout_predictions')]:
        chosen=[r for r in inference if r['split']==split]
        canvas=Image.new('RGB',(1800,90+len(chosen)*660),(246,245,239));draw=ImageDraw.Draw(canvas)
        draw.text((20,18),'参考训练图预测，不能用于证明泛化能力' if split=='train' else '独立画作验证预测，暂无人工真值；仅供目视审查',font=font,fill=(30,45,35))
        for i,r in enumerate(chosen):
            view=Image.open(run/'inference'/r['id']/'comparison.png');view.thumbnail((1800,660))
            canvas.paste(view,((1800-view.width)//2,80+i*660))
        canvas.save(run/(name+'.jpg'),quality=92)
    sources='\n'.join('- '+r['id']+' | '+str(r['source'].get('title','用户提供'))+' | '+str(r['source'].get('page_url','用户提供参考图')) for r in real)
    card=f'''# 工笔山水结构线模型：首轮实验

方向：山脊、岩石主线、树干、屋宇、桥梁为主，皴线/叶簇/水纹另存细节通道。

## 实际完成
- 模型：MobileNetV3 Small ImageNet 编码器 + U-Net 解码器，两通道，{summary['parameter_count']:,} 参数。
- CUDA 训练 {summary['completed_steps']} 步；耗时 {summary['elapsed_seconds']:.1f} 秒；GPU {config['gpu']}。
- 主输出权重 best.pt；最近一步 last.pt。两者可含不同步骤。模型原始概率图、SVG 和路径 JSON 在 inference/。
- 真实数据23张：用户7张同一作品裁图 + 新下载16张馆藏文件。按画作/册页组分组，共14组，20张训练、3张验证。
- 合成几何配对640张训练、64张验证。它们不是704幅真实画作。
- 真实监督：自动弱标签 + 2张参考图上的11条助手草拟稀疏结构线。没有经过专家审核的完整标准线稿。

## 如何解读效果
这是已实际训练的初始模型，不是已证实高精度的成熟模型。合成样本指标只说明能否识别合成分布；真实验证指标只表示与自动弱标签的一致程度，不能称为真实提线准确率。真实图还存在绢纹、裂痕、深色设色区和细树枝引起的混淆。用户7张参考图都参与了训练，其预览必须视作训练样本表现。3张独立馆藏组用于选模型，尚无第三方最终测试集。

实测目视审查：user_qianli_02 能保留主山脊并抑制大面积天空/水纹，但树干与山体内部线条碎裂；user_qianli_05 仍有许多短碎线。三组留出作品普遍严重漏线，met_49227 主要识别出了屏风黑色分隔线，cma_1997_95 几乎只剩画幅边缘。当前模型的换图能力未通过验证，不能用于替换现有生产提线流程。

弱标签本身会漏线和误提，模型可能学到这些错误。先补充少量高质量、逐像素复核的结构线/背景标签，再扩展同类作品，比继续堆弱标签或训练步数更可靠。辅助水墨/设色样本不是全部严格同类工笔。部分长卷缩放后高度较小，后续应下载更高分辨率或分区原始扫描。应对画框、装裱、文字、印章添加背景负样本；增加尺度/褪色/色相变化；补充多幅独立青绿作品标签并单独留出最终测试集。

统一后处理：概率低阈值0.35/高阈值0.55，连通域保留、骨架细化、过滤短于12像素的路径。没有逐图手动美化预测。结构图只表示平面线条，尚无毛笔压力、速度、含墨量、含水量标定，不能直接当作机械臂作画程序。

## 文件与复现
- config.json / training.jsonl / training_summary.json：训练设置、全过程、结果。
- data_inventory.json / audit.json：每图来源、分组、草拟标签、权重哈希和数据审计。
- source_snapshot/：代码快照。F盘源码包 gongbi_training/，入口 scripts/gongbi_lab.py。
- 全部数据、缓存、权重与输出均保存E盘；不连接或移动机械臂。
- 在 F:\\山河绘\\山河绘 运行 powershell -ExecutionPolicy Bypass -File scripts\\gongbi_lab.ps1 infer --run {args.run}
- 打开标注工具：powershell -ExecutionPolicy Bypass -File scripts\\gongbi_lab.ps1 annotate，再浏览器访问 http://127.0.0.1:8766 。标签直接保存E盘，不经过浏览器下载目录。
- 标注0未知、1明确背景、2结构、3细节。保留整幅作品分组；验证图标注仍保留验证用途。
- 当前首轮训练没有使用用户复核标签。下一轮 prepare --manual-only 合并已复核标注后，使用新运行名 train --name mobilenet_unet_v02。

## 原始来源索引
{sources}
'''
    (run/'MODEL_CARD.md').write_text(card,encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=True,indent=2))
