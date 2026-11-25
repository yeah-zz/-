# -*- coding: utf-8 -*-
"""在人工精选 patch 上训练，输出干净模型。"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from master_studio import corpus

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"data"/"texture_model_clean.json"
CLASSES=["pima","zhedai","fupi","midian"]

def main():
 xs=[]; ys=[]
 for c in CLASSES:
  for p in sorted((ROOT/"corpus"/"curated"/c).glob("*.png")):
   try:
    f=corpus._features(Image.open(p).convert("RGB"))
    if f: xs.append(corpus._vector(f)); ys.append(c)
   except Exception: pass
 X=np.stack(xs); counts={c:ys.count(c) for c in CLASSES}
 mean=X.mean(0); std=X.std(0)+1e-6; Xn=(X-mean)/std
 cents={}; raw_cents={}
 for c in CLASSES:
  idx=[i for i,y in enumerate(ys) if y==c]
  if idx:
   cents[c]=Xn[idx].mean(0)
   raw_cents[c]=X[idx].mean(0)
 # LOO kNN
 best=(0,1)
 for k in (1,3,5):
  hit=0
  for i in range(len(Xn)):
   d=np.linalg.norm(Xn-Xn[i],axis=1); d[i]=1e9
   idx=np.argsort(d)[:k]; votes={}
   for j in idx: votes[ys[j]]=votes.get(ys[j],0)+1
   if max(votes,key=votes.get)==ys[i]: hit+=1
  acc=hit/len(Xn)
  if acc>best[0]: best=(acc,k)
 model=dict(format="texture-curated-v1",classes=CLASSES,counts=counts,
            centroids={c:v.tolist() for c,v in cents.items()},
            raw_centroids={c:v.tolist() for c,v in raw_cents.items()},
            mean=mean.tolist(),std=std.tolist(),
            train_vectors=Xn.tolist(),train_labels=ys,
            knn_k=best[1],samples=len(ys),loo_accuracy=round(best[0],3),
            note="人工从山水拼版中挑选：排除印章、题字、树冠、建筑。")
 OUT.write_text(json.dumps(model,ensure_ascii=False),encoding="utf-8")
 print(counts,len(ys),model["loo_accuracy"],model["knn_k"])

if __name__=="__main__": main()