# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode=True
from freeze_stability_brush_20261101 import ROOT,read,jsave
from stroke_geometry_brush_20261101 import render_parameters,resample
from PIL import Image,ImageDraw,ImageFont
import numpy as np,math,argparse,cv2
FONT=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)

def contacts():
    records=[read(f) for f in sorted((ROOT/'input_records').glob('T*.json'))];available=[r for r in records if (ROOT/'outputs'/r['id']/'complete/complete.png').exists()];folder=ROOT/'review_contacts';folder.mkdir(exist_ok=True)
    for start in range(0,len(available),4):
        board=Image.new('RGB',(1400,762),'#f3f1eb');d=ImageDraw.Draw(board)
        for j,r in enumerate(available[start:start+4]):
            ident=r['id'];ox=(j%2)*700;oy=(j//2)*381;d.text((ox+5,oy+4),ident+' 原图（496评估输入） / 冻结提取结果',font=FONT,fill='black')
            for k,path in enumerate([r['input'],ROOT/'outputs'/ident/'complete/complete.png']):
                im=Image.open(path).convert('RGB').resize((345,345),Image.Resampling.LANCZOS);board.paste(im,(ox+k*350,oy+32))
        board.save(folder/f'contact_{start//4+1:02}.jpg',quality=96)
    print('REVIEW_CONTACTS',len(available),flush=True)

def preview(ident):
    folder=ROOT/'outputs'/ident;input=ROOT/'inputs'/(ident+'.png');fields=[(input,'原图（496评估输入）'),(folder/'complete/complete.png','冻结版本提取'),(folder/'brush/smooth_centerlines.png','平滑曲线（保留分笔）'),(folder/'brush/image_width_tone_preview.png','笔宽浓淡离线预览（非实画）')];board=Image.new('RGB',(1984,550),'#f3f1eb');d=ImageDraw.Draw(board)
    for j,(path,title) in enumerate(fields):d.text((j*496+4,8),title,font=FONT,fill='black');board.paste(Image.open(path).convert('RGB'),(j*496,48))
    board.save(folder/'comparison.jpg',quality=96)

def one_stroke():
    ident='T01';folder=ROOT/'outputs'/ident;rows=read(folder/'brush/image_brush_parameters.json')['paths'];rank=[]
    for r in rows:
        p=np.asarray(r['points']);length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum());valid=np.array(r['valid']);v=np.array([w for w in r['width_px'] if w is not None]);variation=float(np.quantile(v,.9)-np.quantile(v,.1)) if len(v)>2 else 0.
        if length>=20 and valid.mean()>=.5:rank.append((length*valid.mean()*(.3+variation),r))
    assert rank;row=max(rank,key=lambda r:r[0])[1];p=np.asarray(row['points']);rgb=np.asarray(Image.open(ROOT/'inputs/T01.png').convert('RGB'));a=np.maximum(0,np.floor(p.min(0)-12)).astype(int);b=np.minimum([496,496],np.ceil(p.max(0)+13)).astype(int);x,y=a;xx,yy=b;original=Image.fromarray(rgb).crop((x,y,xx,yy));overlay=original.copy();d=ImageDraw.Draw(overlay);d.line([tuple(q-a) for q in p],fill=(35,125,220),width=1);rendered=Image.fromarray(render_parameters((496,496),[row])).convert('RGB').crop((x,y,xx,yy));board=Image.new('RGB',(1200,430),'#f3f1eb');d=ImageDraw.Draw(board)
    for j,(im,title) in enumerate([(original,'原图局部'),(overlay,'蓝：所示轨迹（方向未验证）'),(rendered,'该轨迹笔宽浓淡回放（非实画）')]):
        d.text((j*400+4,8),title,font=FONT,fill='black');factor=min(390/im.width,350/im.height,5);im=im.resize((round(im.width*factor),round(im.height*factor)),Image.Resampling.LANCZOS);board.paste(im,(j*400+4,48))
    board.save(ROOT/'ONE_STROKE_PREVIEW.jpg',quality=96);jsave(ROOT/'one_stroke_selection.json',dict(image=ident,path_id=row['path_id'],selection='Long measured path with width variation, chosen to illustrate parameters; not a representative accuracy claim',direction_verified=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];width=np.array([v if v is not None else np.nan for v in row['width_px']]);ink=np.array([v if v is not None else np.nan for v in row['optical_density']]);fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True,constrained_layout=True)
    for ax,values,title,color in [(axes[0],width,'图像笔宽 FWHM（像素）','#187ca0'),(axes[1],ink,'局部表观墨色强弱（相对光密度）','#765136')]:ax.plot(arc,values,color=color,lw=1.8);ax.set_ylabel(title,fontproperties=font);ax.grid(alpha=.2)
    axes[1].set_xlabel('沿轨迹的距离（输入图像像素）',fontproperties=font);axes[0].set_title('图像测量示例：空缺为未知，不对应机械臂下压量或含墨量',fontproperties=font);fig.savefig(ROOT/'ONE_STROKE_PARAMETERS.png',dpi=170);plt.close(fig)

def native_contacts():
    ids=read(ROOT/'evaluation_protocol.json')['native_resolution_substudy'];available=[i for i in ids if (ROOT/'native_outputs'/i/'complete/complete.png').exists()];folder=ROOT/'native_comparisons';folder.mkdir(exist_ok=True)
    for ident in available:
        fields=[(ROOT/'originals'/(ident+'.png'),'原生输入（1254像素）'),(ROOT/'outputs'/ident/'complete/complete.png','496像素输入提取'),(ROOT/'native_outputs'/ident/'complete/complete.png','原生分辨率输入提取')];board=Image.new('RGB',(1488,550),'#f3f1eb');d=ImageDraw.Draw(board)
        for j,(path,title) in enumerate(fields):im=Image.open(path).convert('RGB');im.thumbnail((496,496),Image.Resampling.LANCZOS);d.text((j*496+4,8),title,font=FONT,fill='black');board.paste(im,(j*496,48))
        board.save(folder/(ident+'.jpg'),quality=96)
    if 'T01' in available:
        box=np.array([.30,.02,.80,.25]);board=Image.new('RGB',(1881,345),'#f3f1eb');d=ImageDraw.Draw(board)
        for j,(path,title) in enumerate([(ROOT/'originals/T01.png','原图松树局部'),(ROOT/'outputs/T01/complete/complete.png','496输入：同区域放大'),(ROOT/'native_outputs/T01/complete/complete.png','1254输入：同区域原尺度')]):
            im=Image.open(path).convert('RGB');r=np.rint(box*np.array([im.width,im.height,im.width,im.height])).astype(int);im=im.crop(tuple(r));im=im.resize((627,288),Image.Resampling.NEAREST);d.text((j*627+4,8),title,font=FONT,fill='black');board.paste(im,(j*627,48))
        board.save(ROOT/'NATIVE_PINE_DETAIL.jpg',quality=96)

def main():
    p=argparse.ArgumentParser();p.add_argument('--contacts-only',action='store_true');a=p.parse_args();contacts()
    if a.contacts_only:return
    for f in sorted((ROOT/'input_records').glob('T*.json')):
        r=read(f)
        if (ROOT/'outputs'/r['id']/'brush/image_width_tone_preview.png').exists():preview(r['id'])
    one_stroke();native_contacts()
if __name__=='__main__':main()
