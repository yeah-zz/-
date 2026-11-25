"""Remove small green accents embedded in painting 01's blue rock faces."""
import copy
import csv
from datetime import datetime
import hashlib
import io
from pathlib import Path
import socket
import sys
from unittest.mock import patch
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.first_color_edge_gap_20261012 import h,read,encode,TARGET,TASK,PACKAGE
from scripts.consolidate_selected_artworks import digest,stroke_data,index_for
from web.catalog import ARTWORKS
from painting_session import session_lock

OUT=ROOT/'artifacts/第一幅去蓝面绿点_20261012'


def main():
    baseline=TARGET.read_bytes();old=read(TARGET);bundle=copy.deepcopy(old)
    assert not bundle['workflow'].get('green_accent_removal_revision'),'Already revised'
    module=bundle['workflow']['modules']['color'];colors=module['strokes'];scale=module['mm_per_px'];w,height=module['image_size'];shape=(height,w)
    green=[s for s in colors if s['color']=='green' and '树' not in s.get('drawing_region','')]
    blue=h.footprint([s for s in colors if s['color']=='blue'],shape,scale)
    mask=h.footprint(green,shape,scale);count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    selected=[];details=[]
    for i in range(1,count):
        x,y,ww,hh,area=stats[i];area_mm2=float(area*scale**2)
        overlap=float(np.count_nonzero((labels==i)&(blue>0))/area)
        if area_mm2<=80 and overlap>=.85:
            selected.append(i);details.append(dict(component=i,box=[int(x),int(y),int(ww),int(hh)],area_mm2=area_mm2,existing_blue_coverage=overlap))
    assert set(selected)=={2,3,4,5,6,8,9,11,13,14,15,16,17}, 'Material changed; inspect accents again'
    removed=[]
    for s in green:
        q=np.rint(s['points']).astype(int)
        if np.any(np.isin(labels[q[:,1],q[:,0]],selected)):removed.append(s['id'])
    keep=[copy.deepcopy(s) for s in colors if s['id'] not in removed]
    assert len(removed)==37
    assert [s for s in keep if s['color']=='blue']==[s for s in colors if s['color']=='blue']
    assert [s for s in keep if '树' in s.get('drawing_region','')]==[s for s in colors if '树' in s.get('drawing_region','')]
    assert bundle['ink_job']==old['ink_job']
    module.update(strokes=keep,stroke_count=len(keep))
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    revision=dict(date=stamp,source_sha256=hashlib.sha256(baseline).hexdigest(),removed_ids=removed,
                  removed_green_strokes=len(removed),components=details,blue_strokes_unchanged=True,
                  large_green_slopes_preserved=True,tree_colors_preserved=True,
                  note='删除第一幅蓝色岩面里圈选类型的13处零碎绿色点缀，共37笔。保留松树及三片成块绿色坡面。已有蓝色铺色不增加遍数，0.85mm下压和1mm中心线笔距保持。')
    plan=bundle['workflow'];plan['green_accent_removal_revision']=revision
    hashed=digest(stroke_data(bundle))
    for key in ('source_revision','web_installation'):plan[key]['stroke_data_sha256']=hashed
    OUT.mkdir(parents=True,exist_ok=True);candidate=TASK/'painting.green-dots-candidate.json';candidate.write_bytes(encode(bundle))
    try:
        from web.site_settings import prepared_artwork
        from run_json_painting import manifest_for
        with patch.object(socket.socket,'connect',side_effect=AssertionError('Offline only')):
            prepared=prepared_artwork(candidate,ROOT/'web/data/site_settings.json');manifest_for(prepared[2],prepared[1],prepared[3])
        order=index_for(bundle);counts={k:len(v) for k,v in order.items()}
        report=dict(offline_check='passed',hardware_started=False,effective_from='next-start',counts=counts,revision=revision,stroke_data_sha256=hashed)
        writes={TARGET:encode(bundle),TASK/'去蓝面绿点校验.json':encode(report)}
        for p in TASK.glob('*_计划.json'):writes[p]=encode(plan)
        for name in ('绘画顺序索引.json','Web当前应用记录.json'):
            p=TASK/name;v=read(p);v.update(counts=counts,stroke_data_sha256=hashed,green_accent_removal_revision=revision)
            if name=='绘画顺序索引.json':v['stages']=order
            writes[p]=encode(v)
        table=io.StringIO(newline='');writer=csv.writer(table);writer.writerow(['阶段','阶段内笔序','笔触ID','完整部位','类别','原笔触ID','长度mm'])
        for stage,rows in order.items():
            for row in rows:writer.writerow([stage,row['order'],row['id'],row['region'],row['role'],row['parent_stroke_id'],row['length_mm']])
        writes[TASK/'逐笔顺序索引.csv']=table.getvalue().encode('utf-8-sig')
        p=ROOT/'待画作品/画稿清单.json';catalog=read(p);row=next(row for row in catalog['artworks'] if row['folder']==PACKAGE.name)
        row.update(counts=counts,stroke_data_sha256=hashed,green_accent_removal_revision=revision);writes[p]=encode(catalog)
        png=h.render(bundle);preview=PACKAGE/'预览与下压标注'
        for name in ('web_preview.png','执行路径预览_非实测线宽.png','任务预览_线宽色彩为估计.png','黑线加蓝绿_实际路径.png','当前作画阶段预览.png','完整色面黑轮廓.png','去蓝面绿点后.png'):writes[preview/name]=png
        board=Image.new('RGB',(1600,660),'#faf8f0');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24)
        for x,b,label in [(20,old,'修改前：蓝色面中的绿色点缀'),(820,bundle,'修改后：去掉小绿点，保留成片绿色')]:
            picture=Image.open(io.BytesIO(h.render(b)));picture.thumbnail((760,545));board.paste(picture,(x,60));draw.text((x,15),label,font=font,fill='#34463a')
        buf=io.BytesIO();board.save(buf,format='PNG');writes[preview/'去蓝面绿点前后对照.png']=buf.getvalue()
        p=PACKAGE/'阅读说明.md';writes[p]=(p.read_text(encoding='utf-8')+'\n\n## 去掉蓝面小绿点（2026-10-12）\n\n'+revision['note']+f'\n当前蓝色{counts["blue"]}笔、绿色{counts["green"]}笔；已通过离线校验，下次启动使用。\n').encode('utf-8')
        backup=PACKAGE/'历史版本'/(stamp+'_去蓝面绿点前');backup.mkdir(parents=True,exist_ok=False)
        originals={p:p.read_bytes() if p.exists() else None for p in writes}
        protected={ROOT/a[3]:hashlib.sha256((ROOT/a[3]).read_bytes()).hexdigest() for a in ARTWORKS[1:]}
        for p,data in originals.items():
            if data is not None:
                q=backup/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data)
        (backup/'painting.json').write_bytes(baseline)
        with session_lock(ROOT/'robot_runs'):
            assert not (ROOT/'robot_runs/blue_wash_motion.lock').exists()
            assert TARGET.read_bytes()==baseline
            done=[]
            try:
                for p,data in writes.items():
                    assert (p.read_bytes() if p.exists() else None)==originals[p]
                    tmp=p.with_name(p.name+'.green-dots.tmp');tmp.write_bytes(data);tmp.replace(p);done.append(p)
                assert read(TARGET)==bundle
                assert all(hashlib.sha256(p.read_bytes()).hexdigest()==digest for p,digest in protected.items())
            except Exception:
                for p in reversed(done):
                    if originals[p] is None:p.unlink(missing_ok=True)
                    else:p.write_bytes(originals[p])
                raise
        (OUT/'report.json').write_bytes(encode(report))
        print(encode(dict(removed_green_strokes=len(removed),removed_components=len(selected),counts=counts,offline_check='passed',backup=str(backup))).decode('utf-8'))
    finally:candidate.unlink(missing_ok=True)


if __name__=='__main__':main()
