"""Source-versus-proposal cards: a proposal is NEVER automatically a label."""
import sys,json
sys.dont_write_bytecode=True
from PIL import Image,ImageDraw,ImageFont
import numpy as np
from audit_v4_local_repairs import OUT,dump
f=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
records=[]
for ident,region in [('G01_01',(65,75,315,205)),('G02_01',(80,35,300,230))]:
    base=OUT/'baseline'/ident;new=OUT/'candidate'/ident
    rgb=Image.open(base/'input.png').convert('RGB');old=json.loads((base/'paths.json').read_text(encoding='utf-8'))['paths'];paths=json.loads((new/'paths.json').read_text(encoding='utf-8'))['paths'][len(old):]
    eligible=[]
    for i,path in enumerate(paths):
        p=np.asarray(path['points']);x,y=p.mean(0);x0,y0,x1,y1=region
        length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
        if x0<x<x1 and y0<y<y1 and 5<=length<=17:eligible.append((i,path))
    chosen=[]
    for i,path in sorted(eligible,key=lambda t:-len(t[1]['points'])):
        center=np.asarray(path['points']).mean(0)
        if all(np.linalg.norm(center-np.asarray(q[1]['points']).mean(0))>24 for q in chosen):chosen.append((i,path))
        if len(chosen)==8:break
    for i,path in chosen:
        p=np.asarray(path['points']);x,y=np.rint(p.mean(0)).astype(int);box=(int(x-20),int(y-20),int(x+21),int(y+21));key=f'L{len(records)+1:02}'
        records.append(dict(id=key,image_id=ident,proposal_index=i,box=box,points=path['points'],status='unreviewed_proposal',label_provenance='candidate from original source skeleton; requires explicit visual decision',scope='partial line segment, does not identify artist stroke or semantic cun type'))
    print(ident,len(chosen))
canvas=Image.new('RGB',(1200,4*240),'#f3f1eb');d=ImageDraw.Draw(canvas)
for k,r in enumerate(records):
    rgb=Image.open(OUT/'baseline'/r['image_id']/'input.png').convert('RGB');patch=rgb.crop(r['box']);over=patch.copy();od=ImageDraw.Draw(over)
    x0,y0,_,_=r['box'];od.line([(x-x0,y-y0) for x,y in r['points']],fill=(230,30,50),width=1)
    x=(k%4)*300;y=(k//4)*240;d.text((x+5,y+3),r['id']+' '+r['image_id']+' 原图 / 候选',font=f,fill='black')
    canvas.paste(patch.resize((145,190)),(x,y+30));canvas.paste(over.resize((145,190)),(x+150,y+30))
canvas.save(OUT/'local_reference_review_cards.jpg',quality=97)
dump(OUT/'local_reference_proposals.json',dict(records=records,not_training_labels_yet=True))
