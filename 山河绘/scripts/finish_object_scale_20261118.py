# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import numpy as np
from PIL import Image
from gongbi_training.guohua_hierarchy import CapabilityState,allowed_phases
ROOT=LAB/'object_scale_pairs_20261118';PROJECT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def main():
    p=read(ROOT/'protocol.json');result=read(ROOT/'results.json');pairs=read(ROOT/'pairs.json')
    assert sha(p['checkpoint'])==p['checkpoint_sha256'] and sha(ROOT/'pairs.json')==p['pairs_sha256']
    assert not result['new_model_trained'] and len(result['records'])==17
    for size in p['scales']:
        c=sum(np.array(r['scores'][str(size)]['confusion']) for r in result['records'] if 'confusion' in r['scores'][str(size)])
        union=c.sum(0)+c.sum(1)-c.diagonal();mean=np.mean([c[k,k]/union[k] for k in range(4) if union[k]])
        assert abs(mean-result['known14'][str(size)]['mean_iou'])<1e-12
    for case in pairs['cases']:
        assert sha(case['image'])==case['image_sha256'] and case['unknown_between_cores'] and not case['exact_boundary_verified']
    lock=read(LAB/'local_rock_patch_20261117/baseline_lock.json');baselines={k:sha(lock[k])==lock[k+'_sha256'] for k in ['approved500','working489']};assert all(baselines.values())
    checks=dict(frozen_checkpoint_hash_unchanged=True,paired_reference_hashes_unchanged=True,
        all_three_sizes_reported=True,known14_metrics_recomputed=True,known_cases=17,
        pairs_are_reused_cores_not_new_boundaries=True,new_model_trained=False,
        independent_validation=False,adopted=False,baseline_hashes=baselines,robot_connected=False)
    write(ROOT/'verification.json',checks)
    extra=read(ROOT/'extended_H14_diagnostics.json')['records']
    table='\n'.join('|'+r['region']+'|'+ '|'.join(f'{r["correct_fraction"][str(s)]*100:.1f}%' for s in [256,512,768])+'|' for r in extra)
    report=PROJECT/'reports/20261118_object_scale_pairs.md'
    report.write_text('''# 树石配对核心与冻结模型输入尺寸诊断

## 1. 推进哪项目标

推进G2山石／树木区分，支持G1防止树叶进入皴笔。针对上轮岩面被扩大标成树木的问题，先检查输入细节和全局对象判断的取舍，不追加弱区域训练。

## 2. 实际新增什么能力

完成同一冻结模型在256、512、768三种输入尺寸上的完整原图推理和同口径诊断。整理教程、仇英、王时敏三组既有山石／树木核心配对，核心之间保留未知；新增精确边界标签为0，独立作品为0。没有训练新模型，也没有新增通过独立验收的识别能力。

已知14图四类平均IoU：256为53.5%，512为60.1%，768为59.6%。三种配置固定后一次性检查，全部结果保留；不能把已知图上的最高值称为独立提升。

## 3. 还有哪些失败

512输入改善部分局部，但破坏另一些对象判断：王时敏岩面核心由89.9%降至28.1%，768进一步降至2.5%。同图松树核心在512仍为100%，说明不能只看树木保留或平均指标。

另对H_14原有全部核心补做已知案例诊断（看过输出后的检查，未参与训练）：

|既有助手核心|256|512|768|
|---|---:|---:|---:|
'''+table+'''

一处淡山体内部由100%降至0%，反映大尺寸下全局山体归属丢失。核心标签是既有助手参考，尚非专家精确边界；未标区域也不能自动作为负例。

本轮不能认定已经取得可靠的树石边界，也未解决皴法辨识、完整笔触、树木技能或实画。所有色块图是对象分区，非笔迹重绘。

## 4. 是否通过独立验证

未进行独立验证。17图都是已知回归或旧训练资料，14图来自既有粗对象标签；三组配对均复用助手核心。已核查冻结权重、原图及配对哈希、指标重算、三尺寸完整结果和500/489笔基线完整性。256输出在推理脚本中与已有缓存逐像素核对。没有把三组裁剪当成新作品。

## 采用决定及下一步

不替换默认输入尺寸、模型或轨迹。识别能力验收状态保持不变，树木技能训练未解锁，机械臂离线。下载、缓存及实验输出遵循E盘约定。

本轮提供的下一步假设是：全局山体归属和局部树冠细节需要分别约束后联合验证；不能只放大输入或继续增加区域在场损失。先补明确的边界支持参考，再验证是否能同时保住淡山体、岩面和树叶。

实验目录：`E:\\Codex_Workspace_Data\\Gongbi_Line_Model_20261022\\object_scale_pairs_20261118`。
''',encoding='utf8')
    text=(ROOT/'index.html').read_text('utf8')
    note='<h2>补充的同图退步检查</h2><p>512下，一处既有淡山体内部核心由100%降至0%；王时敏岩面核心由89.9%降至28.1%。这些是看过输出后的已知诊断。平均指标改善不足以采用配置。</p><p><a href="extended_H14_diagnostics.json">全部H_14既有核心检查</a></p>'
    if note not in text:text=text.replace('<img src="comparison.jpg">',note+'<img src="comparison.jpg">')
    (ROOT/'index.html').write_text(text,encoding='utf8')
    statepath=LAB/'guohua_hierarchy_20261112/current_milestones.json';state=read(statepath);caps=state['capability_state'].copy()
    assert not any(caps.values()) and not allowed_phases(CapabilityState(**caps))['tree_skill_training']
    state.update(latest_round_report=str(report),latest_scale_diagnostic=str(ROOT/'results.json'),
        latest_scale_diagnostic_status='known_scale_sensitivity_only; no input-size adoption',new_model_trained_this_round=False,
        new_object_presence_models_trained_this_round=0,new_cunfa_model_trained_this_round=False,
        production_replaced=False,robot_connected=False)
    assert state['capability_state']==caps;write(statepath,state)
    for name in ['research/project_plan_20261117.json','research/guohua_four_priorities_execution_20261116.json']:
        path=PROJECT/name;v=read(path);v.update(latest_round_report=str(report),latest_scale_diagnostic=str(ROOT/'results.json'));write(path,v)
    print(json.dumps(dict(report=str(report),checks=checks),ensure_ascii=False))
if __name__=='__main__':main()
