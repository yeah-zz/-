from structure_train_v20 import *
from skimage.morphology import skeletonize
from topology_trace_v13 import graph_paths
CURR=ROOT/'paired_curriculum'
HOLDOUT={'yuntou','midian','yudian'}


def image_and_mask(path):
    im=Image.open(path).convert('RGB');im.thumbnail((512,512));rgb=np.asarray(im);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);alpha=v16.photometry(rgb)[1];h,w=gray.shape;binary=(alpha>.055).astype(np.uint8);combined=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((3,11),np.uint8));n,cc,stats,_=cv2.connectedComponentsWithStats(combined,8);ignore=np.zeros((h,w),np.uint8)
    for k in range(1,n):
        x,y,bw,bh,area=stats[k];caption=bh<max(13,h*.045) and bw>bh*4 and (y<h*.22 or y+bh>h*.8);frame=bw<w*.035 and bh>h*.65 and (x<w*.06 or x+bw>w*.94)
        if caption or frame:cv2.rectangle(ignore,(max(0,x-3),max(0,y-3)),(min(w-1,x+bw+3),min(h-1,y+bh+3)),1,-1)
    ignore[:max(3,int(h*.22))]=1;ignore[int(h*.88):]=1;ignore[:,:max(3,int(w*.025))]=1;ignore[:,-max(3,int(w*.035)):]=1
    binary[ignore>0]=0;n,cc,stats,_=cv2.connectedComponentsWithStats(binary,8);keep=np.zeros(n,bool);keep[stats[:,cv2.CC_STAT_AREA]>=8]=True;keep[0]=False;binary=keep[cc].astype(np.uint8);mask=cv2.dilate(binary,np.ones((15,15),np.uint8));mask[ignore>0]=0
    return rgb,gray,alpha,binary,ignore,mask


def prepare():
    torch.set_num_threads(4);cv2.setNumThreads(2);manifest=read(CURR/'manifest.json');lookup={r['index']:r for r in manifest['records']};folder=CURR/'pairs';folder.mkdir(exist_ok=True);sift=cv2.SIFT_create(nfeatures=1200);matcher=cv2.BFMatcher();accepted=[];rejected=[];figures=[];teacher=load_model('control')
    for index,pair in enumerate(manifest['pairs']):
        key=pair['group']+'_'+str(pair['before'])+'_'+str(pair['after']);ra=lookup[pair['before']];rb=lookup[pair['after']];before,g0,a0,ink0,ignore0,mask0=image_and_mask(ra['image']);after,g1,a1,ink1,ignore1,mask1=image_and_mask(rb['image']);h,w=g1.shape;kp0,d0=sift.detectAndCompute(g0,mask0*255);kp1,d1=sift.detectAndCompute(g1,mask1*255);reason=None
        if d0 is None or d1 is None:reason='insufficient_image_features'
        else:
            good=[a for a,b in matcher.knnMatch(d0,d1,k=2) if a.distance<.7*b.distance]
            if len(good)<8:reason='insufficient_matches'
            else:
                src=np.float32([kp0[m.queryIdx].pt for m in good]);dst=np.float32([kp1[m.trainIdx].pt for m in good]);H,inside=cv2.findHomography(src,dst,cv2.RANSAC,2.5)
                if H is None or int(inside.sum())<8:reason='unstable_registration'
        if reason:rejected.append(dict(pair=key,reason=reason));print('PAIR_REJECT',key,reason,flush=True);continue
        inlier=inside[:,0]>0;warped_points=cv2.perspectiveTransform(src[inlier,None],H)[:,0];rms=float(np.sqrt(np.mean(np.sum((warped_points-dst[inlier])**2,1))));warped=cv2.warpPerspective(before,H,(w,h),borderValue=(255,255,255));prev=cv2.warpPerspective(a0,H,(w,h));prior_ink=cv2.warpPerspective(ink0,H,(w,h),flags=cv2.INTER_NEAREST)>0;valid=cv2.warpPerspective(np.ones(g0.shape,np.uint8),H,(w,h),flags=cv2.INTER_NEAREST);old_ignore=cv2.warpPerspective(ignore0,H,(w,h),flags=cv2.INTER_NEAREST);valid=cv2.erode(valid,np.ones((7,7),np.uint8));valid[(ignore1>0)|(old_ignore>0)]=0;dist_after=distance_transform_edt(ink1==0);oldvalid=prior_ink&(valid>0);preservation=float((dist_after[oldvalid]<=2.5).mean()) if oldvalid.any() else 0.;old_dist=distance_transform_edt(~prior_ink);added=(a1>.055)&(prev<.025)&(old_dist>max(3.,2+rms))&(valid>0);n,cc,stats,_=cv2.connectedComponentsWithStats(added.astype(np.uint8),8);keep=np.zeros(n,bool);keep[stats[:,cv2.CC_STAT_AREA]>=10]=True;keep[0]=False;added=keep[cc];newarea=int(added.sum());existing=prior_ink&(a1>.055)&(valid>0)
        if rms>1.8 or preservation<.72 or oldvalid.sum()<20:reason='registration_or_old_ink_inconsistent'
        elif newarea<max(80,int(oldvalid.sum()*.015)):reason='too_few_new_ink_pixels'
        if reason:rejected.append(dict(pair=key,reason=reason,rms=rms,old_ink_preservation=preservation,newarea=newarea));print('PAIR_REJECT',key,reason,round(preservation,2),newarea,flush=True);continue
        ink=(existing|added).astype(np.uint8);holes=(1-ink).astype(np.uint8);n,cc,stats,_=cv2.connectedComponentsWithStats(holes,8)
        for k in range(1,n):
            x,y,bw,bh,area=stats[k]
            if area<=8 and x>0 and y>0 and x+bw<w and y+bh<h:ink[cc==k]=1
        sk=skeletonize(ink>0);degree=cv2.filter2D(sk.astype(np.float32),-1,np.ones((3,3),np.float32))-sk;pad=np.pad(sk,1)
        for dy in [-1,1]:
            for dx in [-1,1]:degree-=(pad[1+dy:1+dy+h,1+dx:1+dx+w]&(pad[1:1+h,1+dx:1+dx+w]|pad[1+dy:1+dy+h,1:1+w]))
        width=distance_transform_edt(ink>0)*2;stable=sk&(degree<=2)&(width<=18)&(valid>0);field=np.zeros((8,h,w),np.float32);field[0]=a1;paths,_=graph_paths(stable,field,False,min_length=10);paths=[p for p in paths if not (np.linalg.norm(p[-1]-p[0])<2 and np.linalg.norm(np.diff(p,axis=0),axis=1).sum()<50)]
        if not paths:rejected.append(dict(pair=key,reason='no_unambiguous_center_chains'));continue
        c,angle,dist=v16.geometry(paths,(h,w));d,idx=distance_transform_edt(c==0,return_indices=True);wmap=np.minimum(width[tuple(idx)]/48,1);target=np.concatenate((np.exp(-d[None]**2/(2*1.1**2))*ink[None],ink[None],angle,wmap[None])).astype(np.float32);reliable=(d<3)&(ink>0);bg=(a1<.006)&(prev<.006)&(valid>0)&(distance_transform_edt(ink==0)>5);m0=np.maximum(reliable.astype(np.float32)*.4,bg*.18);m1=np.maximum(ink*.5,bg*.18);ma=(d<1.8)&(ink>0);m=np.stack((m0,m1,ma*.3,ma*.2)).astype(np.float32);m*=valid[None];roles=np.zeros((h,w),np.float32);role_known=np.zeros((h,w),np.float32)
        if pair['added_role']=='cun':
            points=np.column_stack(np.nonzero(existing))[:,::-1].astype(np.int32);interior=np.zeros((h,w),np.uint8)
            if len(points)>=3:cv2.fillConvexPoly(interior,cv2.convexHull(points),1);interior=cv2.erode(interior,np.ones((7,7),np.uint8))
            internal=added&(interior>0);roles[internal]=1;role_known[internal]=.30
        elif pair['added_role']=='contour':role_known[added]=.45
        if pair['before']==1:role_known[existing]=.45
        # Crop only the drawing body after text/edge removal. Registration remains in full-image audit.
        yy,xx=np.nonzero(ink);x0=max(0,int(xx.min())-15);x1=min(w,int(xx.max())+16);y0=max(0,int(yy.min())-15);y1=min(h,int(yy.max())+16);crop=after[y0:y1,x0:x1];t=target[:,y0:y1,x0:x1];m=m[:,y0:y1,x0:x1];v=valid[y0:y1,x0:x1];curve=c[y0:y1,x0:x1];photoalpha=a1[y0:y1,x0:x1];teacher_center=v16.predict(teacher,crop)[0];role=roles[y0:y1,x0:x1];rm=role_known[y0:y1,x0:x1];extra=np.stack((teacher_center,1-v,curve,np.zeros_like(v),((photoalpha<.18)&(curve>0)).astype(np.float32),np.ones_like(v)*.35,role,rm)).astype(np.float32);t=np.concatenate((t,extra));file=folder/(key+'.npz');np.savez_compressed(file,rgb=crop,target=t,mask=m);split='heldout_work' if pair['group'] in HOLDOUT else 'train';record=dict(pair,id=key,data=str(file),data_sha256=digest(file),before_image=ra['image'],after_image=rb['image'],before_sha256=ra['sha256'],after_sha256=rb['sha256'],homography=H.tolist(),registration_rms=rms,old_ink_preservation=preservation,new_ink_pixels=newarea,crop_box=[x0,y0,x1,y1],split=split,source='temporal_image_difference_not_expert',role_positive_pixels=int((role*rm>0).sum()),role_negative_pixels=int(((1-role)*rm>0).sum()));accepted.append(record)
        overlay=after.copy();overlay[added]=[220,35,45];overlay[existing]=[30,135,220];page=Image.new('RGB',(1600,370),'#f5f3ec');dd=ImageDraw.Draw(page)
        for j,(title,img) in enumerate([('前一步配准到后一步',warped),('后一步原图',after),('红新增／蓝已有墨迹',overlay),('派生中心监督（非预测）',np.uint8(255*(1-target[0])))]):dd.text((j*400+5,5),key+' '+title,font=SMALL,fill='black');paste(page,img,j*400+5,35,390,325)
        page.save(folder/(key+'_audit.jpg'),quality=94);figures.append((key,page));print('PAIR_ACCEPT',key,split,'rms',round(rms,2),'preserve',round(preservation,2),'new',newarea,flush=True)
    jsave(CURR/'paired_targets_manifest.json',dict(records=accepted,rejected=rejected,heldout_work_groups=sorted(HOLDOUT),external_references_excluded=True,source_stage_counts_not_independent_artworks=True,derived_temporal_masks_not_expert_gold=True))
    board=Image.new('RGB',(1600,370*len(figures)+35),'#f5f3ec');dd=ImageDraw.Draw(board);dd.text((5,4),'配对监督审计：来自分步图的配准/差分，不是模型提取结果；失败配对不用于训练',font=FONT,fill='black')
    for i,(key,page) in enumerate(figures):board.paste(page,(0,35+370*i))
    board.save(CURR/'PAIRED_LABEL_AUDIT.jpg',quality=92);print('PAIRED_LABELS_COMPLETE',len(accepted),'rejected',len(rejected),'train',sum(r['split']=='train' for r in accepted),'heldout',sum(r['split']!='train' for r in accepted),flush=True)
if __name__=='__main__':prepare()


