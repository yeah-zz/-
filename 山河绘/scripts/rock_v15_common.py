# -*- coding: utf-8 -*-
import sys,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from PIL import ImageFont
ROOT=LAB/'autonomous_strokes_v15_20261025'
PREVIOUS=LAB/'reviewed_rocks_v14_20261024'/'corrected'
PARENT=LAB/'clean_strokes_v12_20261024'/'final.pt'
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
SMALL=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
def jsave(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
