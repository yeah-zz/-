from pathlib import Path
import json, shutil, re, hashlib
root=Path.cwd()
backup=root/'archive/改名山河绘_路径备份_20260928'
backup.mkdir(exist_ok=False)
old='圆幅山水_预览与现场标定_20260926'
changes=[]
def relocate(v):
    if isinstance(v,str) and old in v:
        normalized=v.replace('\\','/')
        tail=normalized.split(old+'/',1)
        if len(tail)==2:
            candidate=root/tail[1]
            if candidate.exists(): return str(candidate)
        return v
    if isinstance(v,list): return [relocate(x) for x in v]
    if isinstance(v,dict): return {k:relocate(x) for k,x in v.items()}
    return v
for parent in [root/'输出', *root.glob('展示选画_*')]:
    for path in parent.rglob('*.json'):
        if any(p.endswith('_runs') or p.startswith('资料') for p in path.relative_to(parent).parts): continue
        if not path.name.endswith(('_计划.json','_黑线任务.json','_任务.json','调整记录.json')): continue
        before=json.loads(path.read_text(encoding='utf-8-sig'))
        after=relocate(before)
        if before==after: continue
        saved=backup/path.relative_to(root);saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,saved)
        path.write_text(json.dumps(after,ensure_ascii=False,indent=2),encoding='utf-8')
        changes.append(str(path.relative_to(root)))
# Keep user-facing launchers on the repaired default uv environment.
for path in root.glob('展示选画_*/山亭松瀑_完整画作/launch.ps1'):
    saved=backup/path.relative_to(root);saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,saved)
    path.write_text(path.read_text(encoding='utf-8-sig').replace('.展示运行环境','.venv'),encoding='utf-8-sig')
# Batch file stays ASCII, avoiding Windows codepage-dependent environment names.
p=root/'放弃旧任务.bat';shutil.copy2(p,backup/p.name)
p.write_bytes(b'@echo off\r\ncd /d "%~dp0"\r\n".venv\\Scripts\\python.exe" -B abandon_old_task.py\r\npause\r\n')
for show in root.glob('展示选画_*'):
    for p in show.rglob('*.txt'):
        if any(x.startswith('资料') for x in p.relative_to(show).parts):continue
        s=p.read_text(encoding='utf-8-sig')
        t=s.replace('.展示运行环境','.venv')
        if s!=t:
            saved=backup/p.relative_to(root);saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,saved)
            p.write_text(t,encoding='utf-8-sig')
    art=show/'山亭松瀑_完整画作'
    if art.is_dir():
        rows=[dict(file=str(p.relative_to(show)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in art.rglob('*') if p.is_file() and not any(x.endswith('_runs') for x in p.parts)]
        (show/'完整画作文件校验.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8-sig')
(backup/'修改清单.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf-8')
print('Updated plan/reference files:',len(changes))
