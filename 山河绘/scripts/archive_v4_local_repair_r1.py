import sys,shutil
sys.dont_write_bytecode=True
from pathlib import Path
from audit_v4_local_repairs import OUT,sha,dump
archive=OUT/'iterations/r1_before_cycle_guard';archive.mkdir(parents=True,exist_ok=True)
files=[]
for folder in [OUT/'candidate',OUT/'source_snapshot']:
    for p in folder.rglob('*'):
        if p.is_file():files.append(p)
files += [p for p in OUT.iterdir() if p.is_file()]
for p in files:
    target=archive/p.relative_to(OUT);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert sha(target)==sha(p),'archive already differs: '+str(p)
    else:shutil.copy2(p,target)
    assert sha(p)==sha(target)
dump(archive/'archive_manifest.json',dict(files=[dict(relative=str(p.relative_to(OUT)),sha256=sha(p)) for p in files],reason='All 30 initial candidates increased pixel graph cycle count; retained for transparency, not adopted. Group 5 is now consumed for repair diagnosis and must not be called blind validation.'))
print('Archived',len(files),'files; originals not removed')
