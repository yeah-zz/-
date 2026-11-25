from v20_sources import *
from human_mountain_v11 import paste
CURR=ROOT/'paired_curriculum'
GROUPS={'basic':list(range(1,7)),'fupi_large':list(range(8,13)),'fupi_small':list(range(14,17)),'pima':list(range(18,22)),'guatie':list(range(23,28)),'heye':list(range(29,34)),'jiesuo':list(range(35,40)),'douban':[41],'zhedai':list(range(43,48)),'yuntou':list(range(49,54)),'midian':list(range(55,60)),'dingtou':list(range(61,64)),'yudian':list(range(65,69)),'mountains':list(range(69,73))}
PAIRS=[('basic',1,2,'cun'),('basic',2,3,'texture'),('fupi_large',8,9,'cun'),('fupi_large',9,10,'contour'),('fupi_large',10,11,'cun'),('fupi_small',14,15,'cun'),('guatie',23,24,'contour'),('guatie',24,25,'cun'),('guatie',25,26,'cun'),('jiesuo',35,36,'contour'),('jiesuo',36,37,'contour'),('zhedai',43,44,'mixed'),('zhedai',44,45,'texture'),('yuntou',49,50,'contour'),('yuntou',50,51,'mixed'),('yuntou',51,52,'contour'),('midian',55,56,'contour'),('midian',56,57,'contour'),('yudian',65,66,'contour'),('yudian',66,67,'contour'),('yudian',67,68,'cun'),('mountains',69,70,'mixed'),('mountains',70,71,'mixed')]

def main():
    (CURR/'images').mkdir(exist_ok=True);assets=read(CURR/'public_assets.json');records=[];lookup={n:g for g,ids in GROUPS.items() for n in ids}
    for asset in assets:
        p=CURR/'raw'/('stage_'+str(asset['index']).zfill(3)+'.bin')
        try:
            im=Image.open(p).convert('RGB');out=CURR/'images'/('stage_'+str(asset['index']).zfill(3)+'.png');im.save(out);group=lookup.get(asset['index'],'reference_artwork');records.append(dict(index=asset['index'],image=str(out),width=im.width,height=im.height,work_group=group,source_page='https://www.sohu.com/a/277347183_100275234',image_url=asset['url'],sha256=digest(out),raw_sha256=digest(p),new_source_images_not_independent_works=True))
        except Exception as e:print('SKIP',asset['index'],str(e),flush=True)
    cases={r['id']:r for r in read(ROOT/'external_manifest.json')['records']};hold=[('external_'+k,Image.open(r['image']).convert('L')) for k,r in cases.items() if k.startswith('X')];old=read(LAB/'reviewed_rocks_v14_20261024'/'corrected'/'manifest.json')['records']
    for r in old:
        if r['id'] in ['R03','R04']:
            im=Image.open(r['image']).convert('L');x,y,w,h=r['native_content'];hold.append((r['id'],im.crop((x,y,x+w,y+h))))
    sift=cv2.SIFT_create(nfeatures=600);matcher=cv2.BFMatcher();features=[]
    for key,im in hold:
        im.thumbnail((600,600));a=np.asarray(im);kp,des=sift.detectAndCompute(a,None);features.append((key,kp,des,a.shape))
    duplicates=[]
    for r in records:
        if r['work_group']=='reference_artwork':continue
        im=Image.open(r['image']).convert('L');im.thumbnail((600,600));kp,des=sift.detectAndCompute(np.asarray(im),None)
        if des is None:continue
        for key,kp2,des2,shape in features:
            if des2 is None:continue
            pairs=matcher.knnMatch(des,des2,k=2);good=[a for a,b in pairs if a.distance<.68*b.distance]
            if len(good)<10:continue
            src=np.float32([kp[a.queryIdx].pt for a in good]);dst=np.float32([kp2[a.trainIdx].pt for a in good]);H,inliers=cv2.findHomography(src,dst,cv2.RANSAC,3)
            if H is None:continue
            count=int(inliers.sum());coverage=(np.ptp(dst[inliers[:,0]>0,0])*np.ptp(dst[inliers[:,0]>0,1]))/max(1,shape[0]*shape[1])
            if count>=12 and coverage>.12:duplicates.append(dict(stage=r['index'],group=r['work_group'],heldout=key,inliers=count,heldout_match_box_fraction=float(coverage)))
    excluded=set(['pima','heye'])|{r['group'] for r in duplicates};pairs=[dict(group=g,before=a,after=b,added_role=role) for g,a,b,role in PAIRS if g not in excluded];jsave(CURR/'manifest.json',dict(records=records,groups=GROUPS,pairs=pairs,excluded_groups=sorted(excluded),holdout_duplicate_checks=duplicates,source='artist tutorial account with multiple works and progressive stages',pair_labels_are_temporal_weak_supervision_not_expert_pen_lifts=True))
    board=Image.new('RGB',(1600,260*len(GROUPS)+35),'#f5f3ec');d=ImageDraw.Draw(board);d.text((5,4),'新增教学序列：同一作品不同步骤不能计作独立画作；披麻/荷叶组保留，不用于本轮线条训练',font=FONT,fill='black');index={r['index']:r for r in records}
    for i,(group,ids) in enumerate(GROUPS.items()):
        selected=ids if len(ids)<=4 else [ids[0],ids[1],ids[-2],ids[-1]]
        for j,k in enumerate(selected):
            r=index[k];x=j*400;y=35+i*260;d.text((x+5,y),str(k)+' '+group+(' EXCLUDED' if group in excluded else ''),font=SMALL,fill='black');paste(board,np.asarray(Image.open(r['image'])),x+5,y+26,390,228)
    board.save(CURR/'SEQUENCE_CONTACT.jpg',quality=93);print('PAIRED_DATASET',json.dumps(dict(images=len(records),groups=len(GROUPS),pairs=len(pairs),excluded=sorted(excluded),matches=duplicates),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
