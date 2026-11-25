import sys
sys.dont_write_bytecode=True
from paper_center_refinement_data import ROOT,PARENT
from corrected_stroke_fixture import fixture
from joint_ink_structure_data import hard_fixture
from rock_v15_common import jsave,digest
from pathlib import Path
import numpy as np,shutil
STAGE=ROOT/'corrected_renderer_stage3';checks=[]
for seed in range(920000,920008):
    new=fixture(seed,'needles');rgb=new[0];old=hard_fixture(seed,'needles')[0]
    for i,row in enumerate(new[-1]['strokes']):
        x,y=np.rint(np.mean(row['points'],axis=0)).astype(int);difference=float(old[y,x].astype(float).mean()-rgb[y,x].astype(float).mean());ok=difference>2
        checks.append(dict(seed=seed,stroke=i,rgb_midpoint_darkening=difference,passed=bool(ok)));assert ok,(seed,i,difference)
jsave(STAGE/'renderer_rgb_checks.json',dict(checks=checks,all_passed=True,count=len(checks),purpose='RGB stroke midpoints are actually painted, not merely marked positive in labels'))
notice=dict(affected_renderer='joint_ink_structure_data.hard_fixture v1',problem='two-point needle paths painted endpoints only, but labels declared the full segment; other sparse paths could also be undersampled',affected_metrics=['joint_ink_structure_20261029 hard_declared and ink_gate_stage2 aggregate procedural metrics','paper_center_refinement_20261029 original seed910000 test, particularly needle recall'],unaffected='original RGB artworks and paper-region false-positive counts; original full-scene renderer uses segment interpolation',replacement_report=str(ROOT/'index.html'),legacy_weights_preserved=True,legacy_aggregate_scores_invalidated=True)
jsave(ROOT/'LEGACY_RENDERER_METRICS_INVALID.json',notice);jsave(PARENT/'RENDERER_AUDIT_NOTICE.json',notice)
page=PARENT/'index.html';backup=PARENT/'index_before_renderer_audit.html'
if not backup.exists():shutil.copy2(page,backup);assert digest(page)==digest(backup)
text=page.read_text(encoding='utf-8')
if 'id="renderer-audit-notice"' not in text:
    warning='<aside id="renderer-audit-notice"><strong>程序样本审计更正：</strong>本页部分程序难例的两点线段曾只渲染端点，受影响的松针与整体程序聚合指标不再作为结论。实际画面对照保留；<a href="../paper_center_refinement_20261029/index.html">查看修正样本、补训和新测试</a>。</aside>'
    text=text.replace('<h1>',warning+'<h1>',1);page.write_text(text,encoding='utf-8')
print('RENDERER_RGB_AUDIT',len(checks),flush=True)
