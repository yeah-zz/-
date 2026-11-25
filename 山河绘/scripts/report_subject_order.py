"""Material order tables and path previews derived from the actual execution stages."""
from collections import Counter
from itertools import groupby
from pathlib import Path
import csv
import hashlib
import io
import json
import sys
import shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from web.catalog import ARTWORKS
from artwork_order import ink_stages
from painting_session import save
from scripts.consolidate_selected_artworks import index_for,stroke_data,digest
from scripts.order_selected_artworks import signature

LABELS={'black':'山水勾勒','blue':'蓝色上色','green':'绿色上色','title':'最后题字'}


def main():
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    library=[]
    for folder,title,_,source in ARTWORKS:
        package=ROOT/'待画作品'/folder;task=package/'任务资料';preview=package/'预览与下压标注'
        bundle=json.loads((ROOT/source).read_text(encoding='utf-8'))
        old=json.loads((package/'历史版本'/'20261006_笔序调整前'/'painting.json').read_text(encoding='utf-8'))
        revision=bundle['workflow'].get('source_revision')
        if revision:
            assert revision['stroke_data_sha256']==digest(stroke_data(bundle))
        else:
            for current,prior in [(bundle['ink_job']['meta']['ink_execution']['base_strokes'],old['ink_job']['meta']['ink_execution']['base_strokes']), (bundle['workflow']['modules']['color']['strokes'],old['workflow']['modules']['color']['strokes'])]:
                assert Counter(map(signature,current))==Counter(map(signature,prior))
        index=index_for(bundle);counts={stage:len(ss) for stage,ss in index.items()}
        save(task/'绘画顺序索引.json',{'format':'shanhe-stroke-order-index-v1','stroke_data_sha256':digest(stroke_data(bundle)),'counts':counts,'stages':index,'title_after_all_colors':True})
        detailed=io.StringIO(newline='');writer=csv.writer(detailed);writer.writerow(['阶段','阶段内笔序','笔触ID','完整部位','类别','原笔触ID','长度mm'])
        rows=[]
        for stage,ss in index.items():
            for row in ss:writer.writerow([LABELS[stage],row['order'],row['id'],row['region'],row['role'],row['parent_stroke_id'],row['length_mm']])
            start=1
            for region,group in groupby(ss,key=lambda s:s['region']):
                count=len(list(group));rows.append([LABELS[stage],region,start,start+count-1,count]);start+=count
        (task/'逐笔顺序索引.csv').write_text(detailed.getvalue(),encoding='utf-8-sig')
        table=io.StringIO(newline='');w=csv.writer(table);w.writerow(['阶段','完整部位','阶段内起始笔','阶段内结束笔','笔数']);w.writerows(rows)
        (task/'绘画顺序清单.csv').write_text(table.getvalue(),encoding='utf-8-sig')
        lines=['# '+title+' · 当前作画顺序','','山水勾勒 → 一种颜色按部位画完整幅 → 洗笔换另一种颜色按部位画完整幅 → 洗笔准备黑墨 → 最后山、河、绘题字。','','部位内部先主轮廓、树干树枝与结构，再补细节；整部位完成后再进入下一个部位。同一原笔触的长笔分段连着画，不倒置压力或点列。','当前部位顺序：'+' → '.join(bundle['workflow']['execution_order']['subjects'])+'。','','题字保持原来轮廓画法，本次只改先后顺序，不改为填黑或新的书法笔法。','','| 阶段 | 完整部位 | 阶段内笔序 | 笔数 |','|---|---|---|---|']
        lines += ['| %s | %s | %d—%d | %d |'%tuple(row) for row in rows]
        lines += ['', '逐笔 ID 和部位见逐笔顺序索引.csv。修改前完整画稿在本幅“历史版本/20261006_笔序调整前/painting.json”。所有点列、下压量、速度、颜色和笔触ID均逐笔保留。','']
        if revision:
            lines[-2]='逐笔 ID 和部位见逐笔顺序索引.csv。本幅已按用户原图重整下压、远山和绿色松叶，详见原图修订与下压说明.md；笔数与部分路径已更新。'
        (task/'绘画顺序.md').write_text('\n'.join(lines),encoding='utf-8')
        readme=package/'阅读说明.md';text=readme.read_text(encoding='utf-8')
        text += '\n## 当前笔序（2026-10-06）\n\n已按完整部位重排山水勾勒与各颜色内部笔触；所有上色完成后，再洗笔准备黑墨最后写山、河、绘。详见任务资料/绘画顺序.md。本次笔数和原有路径、下压量保持不变，题字仍使用原轮廓。\n'
        if '## 当前笔序' not in readme.read_text(encoding='utf-8'):readme.write_text(text,encoding='utf-8')
        parts=ink_stages(bundle['ink_job'],bundle['workflow']);stages={'black':parts['black']}
        for color in ('blue','green'):stages[color]=[s for s in bundle['workflow']['modules']['color']['strokes'] if s['color']==color]
        stages['title']=parts['title']
        width,height=bundle['ink_job']['meta']['image_size'];canvas=np.full((height,width,3),255,np.uint8)
        board=Image.new('RGB',(1440,1080),'#f7f7f0');draw=ImageDraw.Draw(board)
        draw.text((25,15),title+' · 实际任务分阶段路径预览',font=font,fill='#304c3b')
        scale=bundle['ink_job']['meta']['ink_execution']['mm_per_px']
        for i,(stage,ss) in enumerate(stages.items()):
            for s in ss:
                q=np.rint(s['points']).astype(np.int32)
                color={'black':(35,40,35),'title':(35,40,35),'blue':(175,125,72),'green':(97,152,97)}[stage]
                stroke_width=max(1,round(s.get('brush_width_mm_estimate',s.get('width_mm',.32))/scale))
                cv2.polylines(canvas,[q],False,color,stroke_width,cv2.LINE_AA)
            picture=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB));picture.thumbnail((690,425))
            x=25+(i%2)*710;y=70+(i//2)*500
            draw.text((x,y),'%d · %s (%d笔)'%(i+1,LABELS[stage],len(ss)),font=font,fill='#304c3b')
            board.paste(picture,(x+(690-picture.width)//2,y+45))
        draw.text((25,1043),'题字只出现在最后一格；示意展示路径与顺序，不代表实物墨晕和颜色浓淡。',font=small,fill='#687562')
        board.save(preview/'当前作画阶段预览.png')
        library.append({'folder':folder,'title':title,'painting_json':source,'reference_image':'待画作品/'+folder+'/参考原图.png','counts':counts,'stroke_data_sha256':digest(stroke_data(bundle)),'preserved_order':False,'complete_subject_order':True,'title_after_all_colors':True})
        current_checks=[];checksum=task/'归档文件校验_SHA256.json'
        for p in sorted(package.rglob('*')):
            if p.is_file() and p!=checksum:current_checks.append({'file':p.relative_to(package).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
        save(checksum,{'format':'shanhe-material-checksums-v1','date':'2026-10-06','files':current_checks})
    save(ROOT/'待画作品'/'画稿清单.json',{'format':'shanhe-material-library-v1','date':'2026-10-06','artworks':library})
    report=ROOT/'待画作品'/'运行校验.json';old_report=report.with_name('材料统一校验_历史.json')
    if report.exists() and not old_report.exists():shutil.copy2(report,old_report)
    save(report,{'date':'2026-10-06','robot_connected':False,'complete_subject_order':True,'title_after_all_colors':True,'all_strokes_geometry_press_speeds_and_colors_preserved':False,'user_requested_source_revisions':[a[0] for a in ARTWORKS if json.loads((ROOT/a[3]).read_text(encoding='utf-8'))['workflow'].get('source_revision')],'validation':'Offline geometry and stage checks; original permutation checks apply to artworks without source revision','checks':library})
    sums=[];master=ROOT/'待画作品'/'整理清单_SHA256.json'
    for p in sorted((ROOT/'待画作品').rglob('*')):
        if p.is_file() and p!=master:sums.append({'file':p.relative_to(ROOT/'待画作品').as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
    save(master,{'format':'shanhe-material-library-checksums-v1','date':'2026-10-06','files':sums})
    print('Five order tables, detailed indexes, four-phase previews and checksums updated.')


if __name__=='__main__':main()
