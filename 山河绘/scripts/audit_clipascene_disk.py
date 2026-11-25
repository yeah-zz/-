"""Read-only audit of disk space created by this experiment."""
from pathlib import Path
import datetime
import json
import os

ROOT=Path(__file__).resolve().parents[1]
START=datetime.datetime.fromisoformat("2026-10-21T21:26:00+08:00").timestamp()


def scan(path):
    total=0
    files=0
    for directory,_,names in os.walk(path):
        for name in names:
            p=Path(directory)/name
            try:
                total+=p.stat().st_size
                files+=1
            except OSError:
                pass
    return dict(path=str(path),bytes=total,gib=round(total/2**30,3),files=files)


temp=Path(os.environ["TEMP"])
folders=[p for p in temp.glob("pip-*") if p.is_dir() and p.stat().st_ctime>=START]
temp_rows=[scan(p) for p in folders]
cache=Path(r"C:\Users\Lenovo\AppData\Local\pip\cache")
recent=[]
for directory,_,names in os.walk(cache):
    for name in names:
        p=Path(directory)/name
        try:
            stat=p.stat()
            if stat.st_mtime>=START:
                recent.append(dict(path=str(p),bytes=stat.st_size,gib=round(stat.st_size/2**30,3)))
        except OSError:
            pass
build=scan(Path(r"C:\Users\Lenovo\.codex\visualizations\2026\10\21\01a15e88-26fa-7a12-9622-3be075148c5d\clipascene_build_tools"))
result=dict(start="2026-10-21T21:26:00+08:00",temporary_folders=temp_rows,
            recent_cache_files=recent,build_tools=build,
            temp_total_gib=round(sum(r["bytes"] for r in temp_rows)/2**30,3),
            recent_cache_gib=round(sum(r["bytes"] for r in recent)/2**30,3))
out=ROOT/"offline_previews"/"clipascene_20261021"/"disk_audit.json"
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(dict(temp_total_gib=result["temp_total_gib"],
                     temporary_folders=[r for r in temp_rows if r["bytes"]>10**7],
                     recent_cache_gib=result["recent_cache_gib"],
                     large_cache_files=[r for r in recent if r["bytes"]>10**8],
                     build_tools=build,audit_file=str(out)),ensure_ascii=True))
