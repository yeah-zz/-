from native_fine_faint_20261101 import *
from refine_native_fine_faint_20261101 import OUTPUTS
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)

def main():
    available=sorted(p.name for p in OUTPUTS.iterdir() if (p/'entry.json').exists())
    for ident in available:
        out=OUTPUTS/ident;base=np.asarray(Image.open(out/'baseline.png'))<128;fan=np.asarray(Image.open(out/'fan.png'))<128;fine=np.asarray(Image.open(out/'fine.png'))<128;result=np.asarray(Image.open(out/'lines.png'))<128;fine_only=(base&~fan)|fine;faint_only=np.where(fan,base,result)
        Image.fromarray(np.where(fine_only,0,255).astype(np.uint8)).save(out/'fine_only.png');Image.fromarray(np.where(faint_only,0,255).astype(np.uint8)).save(out/'faint_only.png')
        fields=[(BASE/'originals'/(ident+'.png'),'原图'),(out/'baseline.png','上轮提取（轨迹缩放）'),(out/'lines.png','松针与淡线修正候选'),(out/'changes.png','蓝新增 / 红移除')];board=Image.new('RGB',(1984,550),'#f3f1eb');d=ImageDraw.Draw(board)
        for j,(path,title) in enumerate(fields):im=Image.open(path).convert('RGB');im.thumbnail((496,496),Image.Resampling.LANCZOS);d.text((j*496+4,8),ident+' '+title,font=font,fill='black');board.paste(im,(j*496,48))
        board.save(out/'comparison.jpg',quality=96)
        overlay=Image.open(BASE/'originals'/(ident+'.png')).convert('RGB');prob=np.asarray(Image.open(out/'fan_probability.png'))/255;arr=np.array(overlay);arr[prob>.65]=(arr[prob>.65]*.55+np.array([40,190,90])*.45).astype(np.uint8);Image.fromarray(arr).save(out/'routing_overlay.jpg',quality=95)
        n,cc,stats,_=cv2.connectedComponentsWithStats(fan.astype(np.uint8),8)
        if n>1:
            x,y,w,h,area=stats[1+np.argmax(stats[1:,cv2.CC_STAT_AREA])];box=(max(0,x-12),max(0,y-12),min(1254,x+w+12),min(1254,y+h+12));ww=box[2]-box[0];hh=box[3]-box[1];detail=Image.new('RGB',(ww*3,hh+50),'#f3f1eb');dd=ImageDraw.Draw(detail)
            for j,(path,title) in enumerate([(BASE/'originals'/(ident+'.png'),'原图局部'),(out/'baseline.png','旧轨迹同尺度'),(out/'fine_only.png','仅细节修正')]):dd.text((j*ww+4,8),title,font=font,fill='black');detail.paste(Image.open(path).convert('RGB').crop(box),(j*ww,46))
            detail.save(out/'detail_tree.jpg',quality=96);jsave(out/'detail_roi.json',dict(box=list(map(int,box)),selection='Largest model-routed connected region; not manual inference mask or representative success selection'))
    regions=[('T01',(350,25,970,335),'松针：原生1254像素同尺度局部'),('T19',(210,300,570,630),'淡山皴线：原生同尺度局部'),('T06',(570,180,1170,580),'大弯曲山面：原生同尺度局部')]
    for ident,box,title in regions:
        if ident not in available:continue
        out=OUTPUTS/ident;ww=box[2]-box[0];hh=box[3]-box[1];board=Image.new('RGB',(ww*3,hh+80),'#f3f1eb');d=ImageDraw.Draw(board)
        for j,(path,label) in enumerate([(BASE/'originals'/(ident+'.png'),'原图'),(out/'baseline.png','上轮轨迹'),(out/'lines.png','本轮候选')]):d.text((j*ww+4,7),label,font=font,fill='black');board.paste(Image.open(path).convert('RGB').crop(box),(j*ww,44))
        d.text((4,hh+49),title+'；仅为局部对照，不是准确率验证',font=font,fill='#444444');board.save(ROOT/(ident+'_DETAIL.jpg'),quality=96)
    for start in range(0,len(available),4):
        board=Image.new('RGB',(1488,600),'#f3f1eb');d=ImageDraw.Draw(board)
        for j,ident in enumerate(available[start:start+4]):
            ox=(j%2)*744;oy=(j//2)*300;d.text((ox+5,oy+4),ident+' 原图 / 旧结果 / 本轮候选',font=font,fill='black')
            for k,path in enumerate([BASE/'originals'/(ident+'.png'),OUTPUTS/ident/'baseline.png',OUTPUTS/ident/'lines.png']):im=Image.open(path).convert('RGB').resize((248,248),Image.Resampling.LANCZOS);board.paste(im,(ox+k*248,oy+35))
        board.save(ROOT/(f'CONTACT_{start//4+1:02}.jpg'),quality=96)
    print('NATIVE_VISUALS',len(available),flush=True)
if __name__=='__main__':main()
