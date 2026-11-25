# -*- coding: utf-8 -*-
import json
from pathlib import Path
ROOT=Path(r'E:\Codex_Workspace_Data\Gongbi_Line_Model_20261022\paper_center_refinement_20261029')
path=ROOT/'mountain_diagnostics/audit.json'
audit=json.loads((ROOT/'mountain_diagnostics/G05_01_roi_geometry.json').read_text(encoding='utf-8'))
audit.update(image_id='G05_01',assessment='mountain region is denser and more fragmented; reject as overall extraction improvement',overall_extraction_accepted=False,pixel_counts_are_not_accuracy=True)
path.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
summary_path=ROOT/'summary.json';summary=json.loads(summary_path.read_text(encoding='utf-8'));summary.update(mountain_regression_audit=audit,overall_extraction_accepted=False);summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
page=ROOT/'index.html';text=page.read_text(encoding='utf-8')
warning='<aside id="mountain-regression-audit"><strong>山体质量复核：整体提取不通过。</strong>G05_01同一山体局部，线条像素5313→5861，连通块313→373，像素图闭环24→26。纸面专项改善未解决山体内部变密、变碎的问题。<a href="mountain_diagnostics/G05_01_mountain_before_after.jpg">查看同一区域放大对照</a>；默认继续保留v4。</aside>'
if 'mountain-regression-audit' not in text:text=text.replace('<h1>',warning+'<h1>',1)
page.write_text(text,encoding='utf-8')
print('Mountain regression recorded; checkpoints unchanged')
