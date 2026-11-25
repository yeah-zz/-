# -*- coding: utf-8 -*-
from cross_section_v18 import *
from collections import Counter
DEST=LAB/'autonomous_review_v19_20261025'
DEST.mkdir(parents=True,exist_ok=True)
review=read(ROOT/'expert_cross_sections'/'reviewed.json');draft={r['id']:r for r in read(ROOT/'review_candidates.json')};rows=[]
board=Image.new('RGB',(1320,210*math.ceil(len(review['annotations'])/3)+45),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,4),'v19 用户横截面校正审计：红色中心、橙色两侧；仅展示原始意见，未改写',font=FONT,fill='black')
for i,a in enumerate(review['annotations']):
    r=draft[a['id']];changes={k:float(a[k]-r[k]) for k in ['left','center','right']};row=dict(a,geometry_changed=any(abs(v)>.02 for v in changes.values()),changes=changes,original_kind=r['kind']);rows.append(row);x=(i%3)*440;y=45+(i//3)*210;im=Image.open(io.BytesIO(base64.b64decode(r['clip']))).convert('RGB').resize((384,192));draw=ImageDraw.Draw(im)
    for k in ['left','right','center']:
        px=(95.5+a[k]*3)*2;draw.line((px,0,px,192),fill='#df283d' if k=='center' else '#de9d1b',width=2)
    board.paste(im,(x+4,y+22));d.text((x+6,y),a['id']+' '+a['label']+(' 已改坐标' if row['geometry_changed'] else ' 草稿坐标已检查'),font=SMALL,fill='black')
board.save(DEST/'USER_REVIEW_AUDIT.jpg',quality=94)
shutil.copy2(ROOT/'expert_cross_sections'/'reviewed.json',DEST/'user_review_snapshot.json');shutil.copy2(ROOT/'review_candidates.json',DEST/'draft_candidates_snapshot.json');jsave(DEST/'review_audit.json',dict(counts=dict(Counter(a['label'] for a in rows)),geometry_modified=sum(a['geometry_changed'] for a in rows),reviewed_count=review['reviewed_count'],rows=rows,user_review_sha256=digest(ROOT/'expert_cross_sections'/'reviewed.json'),uncertain_not_background=True))
print(json.dumps(dict(counts=dict(Counter(a['label'] for a in rows)),geometry_modified=sum(a['geometry_changed'] for a in rows),out=str(DEST)),ensure_ascii=False))
