# -*- coding: utf-8 -*-
"""按人工拼版编号，建立精选 patch 集。"""
import json, shutil
from pathlib import Path
from PIL import Image
import numpy as np, math

ROOT=Path(__file__).resolve().parent
CUR=ROOT/"data"/"curation2"
OUT=ROOT/"corpus"/"curated"
KEEP={
 "pima": list(range(0,10))+list(range(12,22))+[23]+[34,40,41,44,46,48]+[51,55,56,57,60,61,65,66,67,70],
 "zhedai":[13,15,16]+[31,34,36,39,41,45,46]+[50],
 "fupi":[5,7,9,10,12,13,15,16],
 "midian":[5,6,7,8,10,12,15,16,18,20,21,24]+[25,28,33,34,40,41,42,43,44,45,46,47]+[56,58,59,60,61,62,63,64,66,68,69,71,72,74]+[75,76,77,78,79],
}
for cls,ids in KEEP.items():
 d=OUT/cls; d.mkdir(parents=True,exist_ok=True)
 rows=json.loads((CUR/cls/"manifest.json").read_text(encoding="utf-8"))
 for i in ids:
  if i>=len(rows): continue
  src=Path(rows[i]["path"]); 
  if not src.exists(): continue
  dst=d/("%s_%s"%(src.parent.parent.name if src.parent.name=="patches" else "x",src.name))
  # 文件名可能重名，补索引
  if dst.exists(): dst=d/("%03d_%s"%(i,src.name))
  shutil.copy2(src,dst)
 (OUT/(cls+".json")).write_text(json.dumps([{"file":p.name} for p in sorted(d.glob("*.png"))],ensure_ascii=False,indent=2),encoding="utf-8")
 print(cls,len(list(d.glob("*.png"))))