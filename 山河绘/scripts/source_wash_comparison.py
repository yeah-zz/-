"""Before/after preview from executable JSONs with the same brush-width model."""
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from shuimo import press_to_width_mm


def render(bundle,honor_opacity=False):
    meta=bundle['ink_job']['meta'];w,h=meta['image_size'];scale=meta['ink_execution']['mm_per_px']
    canvas=np.full((h,w,3),255,np.uint8)
    ink=meta['ink_execution']['base_strokes'];colors=bundle['workflow']['modules']['color']['strokes']
    stages=[([s for s in ink if s.get('role')!='title'],(40,40,40)),
            ([s for s in colors if s['color']=='blue'],(175,125,72)),
            ([s for s in colors if s['color']=='green'],(97,152,97)),
            ([s for s in ink if s.get('role')=='title'],(40,40,40))]
    for strokes,color in stages:
        for s in strokes:
            target=canvas
            alpha=1.
            if honor_opacity and 'color' in s:
                opacity=s.get('target_opacity',.4)
                alpha=float(np.mean(opacity)) if isinstance(opacity,list) else float(opacity)
                alpha=float(np.clip(alpha,0,1));target=canvas.copy()
            q=np.rint(s['points']).astype(np.int32)
            if honor_opacity and 'color' in s and isinstance(s.get('target_opacity'),list):
                opacity=np.clip(np.asarray(s['target_opacity'],float),0,1)
                pad=int(np.ceil(max(press_to_width_mm(v) for v in s['press'])/scale))+2
                x0,y0=np.maximum(q.min(0)-pad,0);x1,y1=np.minimum(q.max(0)+pad+1,[w,h])
                alpha_map=np.zeros((y1-y0,x1-x0),np.uint8);local=q-[x0,y0]
                for i,(a,b) in enumerate(zip(local[:-1],local[1:])):
                    width=(press_to_width_mm(s['press'][i])+press_to_width_mm(s['press'][i+1]))/(2*scale)
                    value=int(round(255*(opacity[i]+opacity[i+1])/2))
                    cv2.line(alpha_map,tuple(a),tuple(b),value,max(1,round(width)),cv2.LINE_AA)
                weight=alpha_map.astype(np.float32)[:,:,None]/255
                roi=canvas[y0:y1,x0:x1]
                canvas[y0:y1,x0:x1]=np.rint(roi*(1-weight)+np.asarray(color)*weight).astype(np.uint8)
                continue
            if max(s['press'])-min(s['press'])<1e-8:
                cv2.polylines(target,[q],False,color,max(1,round(press_to_width_mm(s['press'][0])/scale)),cv2.LINE_AA)
            else:
                for i,(a,b) in enumerate(zip(q[:-1],q[1:])):
                    width=(press_to_width_mm(s['press'][i])+press_to_width_mm(s['press'][i+1]))/(2*scale)
                    cv2.line(target,tuple(a),tuple(b),color,max(1,round(width)),cv2.LINE_AA)
            if target is not canvas:canvas=cv2.addWeighted(target,alpha,canvas,1-alpha,0)
    return Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))


def comparison(old,new,path):
    board=Image.new('RGB',(1800,840),'#f7f7f0');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',27)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
    all_points=np.vstack([s['points'] for b in (old,new) for s in
        b['ink_job']['meta']['ink_execution']['base_strokes']+b['workflow']['modules']['color']['strokes']])
    lo=np.floor(all_points.min(0)-20).astype(int);hi=np.ceil(all_points.max(0)+20).astype(int)
    draw.text((30,16),'松瀑亭舟 · 上色轨迹修改前后（两图均按真实 JSON 绘制）',font=font,fill='#304c3b')
    for i,(bundle,label) in enumerate(((old,'修改前 · 色带较窄'),(new,'修改后 · 扩大岩面上色'))):
        count=len(bundle['ink_job']['meta']['ink_execution']['base_strokes'])+len(bundle['workflow']['modules']['color']['strokes'])
        pic=render(bundle).crop((*lo,*hi));pic.thumbnail((855,650),Image.Resampling.LANCZOS)
        x=30+i*890
        draw.text((x,65),label+'（%d笔）'%count,font=small,fill='#304c3b')
        board.paste(pic,(x+(855-pic.width)//2,110+(650-pic.height)//2))
    draw.text((30,785),'保留绿色叶团、树干留白和三条细纹；山石0.30mm，题字最后。线宽与颜色为估计，实物渗色需测量。',font=small,fill='#687562')
    board.save(path)
