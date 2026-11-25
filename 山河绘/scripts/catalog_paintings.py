from pathlib import Path
import sys,json,io,contextlib
from collections import Counter
from unittest.mock import patch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from run_mountain_balanced import load_reviewed
from run_color_from_job import load_plan
paths=[];warnings=[]
def scan(folder,depth):
 try: entries=list(folder.iterdir())
 except OSError as e: warnings.append(str(folder)+': '+str(e));return
 for p in entries:
  if p.is_file() and p.name.endswith('_黑线任务.json'):paths.append(p)
  elif depth and p.is_dir() and not p.name.endswith('_runs') and not p.name.startswith('资料'):scan(p,depth-1)
scan(root/'输出',1)
for folder in root.glob('展示选画_*'):scan(folder,2)
rows=[]
for black in sorted(paths):
 try:
  meta=json.loads(black.read_text(encoding='utf-8-sig'))['meta']
  plan=black.with_name(black.stem[:-5]+'_计划.json')
  counts={}
  if plan.exists():counts=Counter(s['color'] for s in json.loads(plan.read_text(encoding='utf-8-sig'))['modules']['color']['strokes'])
  with patch('robot.E6Robot.__init__',side_effect=RuntimeError('Offline inventory only')),contextlib.redirect_stdout(io.StringIO()):
   try:
    load_reviewed(black)
    if plan.exists():load_plan(plan,black)
    status='通过（黑线＋上色）' if plan.exists() else '黑线通过；缺同名上色计划'
   except Exception as e:status='未通过：'+str(e).replace('|','/').replace('\n',' ')[:120]
  rows.append(dict(path=str(black.relative_to(root)),black=meta.get('strokes'),colors=dict(counts),paper_z=meta.get('paper_z'),rest=meta.get('ink_execution',{}).get('policy',{}).get('rest_every'),status=status))
 except Exception as e:warnings.append(str(black)+': '+str(e))
lines=['# 画稿清单','', '整理日期：2026-09-28。下列检查只读取文件、核对当前配置，不连接机械臂；通过不表示已实画或效果验收。文件名含“续画”“剩余”的只适用于对应中断现场。','', '展示目录仍保留旧完整配色；松树专项、簇叶、疏淡等方案在输出目录分别保存，未替换旧入口。','', '| 黑线任务（点击打开） | 黑线笔数 | 上色笔数 | PAPER_Z | 每N笔固定休息（0关闭） | 整理时检查 |','|---|---:|---|---:|---:|---|']
for r in rows:
 path=r['path'].replace('\\','/');label=Path(r['path']).stem
 colors='、'.join({'blue':'蓝','yellow':'黄','green':'绿','red':'红'}.get(k,k)+str(v) for k,v in r['colors'].items()) or '无同名计划'
 lines.append(f"| [{label}](<../{path}>) | {r['black']} | {colors} | {r['paper_z']} | {r['rest']} | {r['status']} |")
lines+=['','## 使用方法','','从表中确认画稿后，把其路径作为 `run_mountain_balanced.py --job` 的参数。含数据子目录的展示画稿也可以使用对应bat入口。详情见[使用说明](../使用说明.md)。','', '本次清单只枚举输出目录顶层及直接子目录、展示目录的任务文件，跳过运行日志和备份，不把“未扫描”当作没有记录。']
if warnings:lines+=['','## 读取限制','']+['- '+s for s in warnings]
(root/'docs/画稿清单.md').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
(root/'docs/画稿离线检查.json').write_text(json.dumps(dict(rows=rows,warnings=warnings),ensure_ascii=False,indent=2),encoding='utf8')
print('Jobs:',len(rows),'Passed:',sum(r['status'].startswith(('通过','黑线通过')) for r in rows),'Warnings:',len(warnings))
