from structure_train_v20 import *
CURR=ROOT/'paired_curriculum'
manifest=read(CURR/'paired_targets_manifest.json')
for r in manifest['records']:
    z=np.load(r['data']);rgb=z['rgb'];t=z['target'].copy();m=z['mask'].copy();changes={}
    if r['group']=='fupi_large':
        raw=Image.open(r['after_image']).convert('RGB');raw.thumbnail((512,512));full=np.asarray(raw);gray=cv2.cvtColor(full,cv2.COLOR_RGB2GRAY);a=v16.photometry(full)[1];h,w=a.shape;x0,y0,x1,y1=r['crop_box'];yy,xx=np.mgrid[y0:y1,x0:x1];columns=(a[int(h*.23):int(h*.75),int(w*.85):]>.05).sum(0);col=int(columns.argmax())+int(w*.85);support=float(columns.max()/max(1,int(h*.75)-int(h*.23)));fringe=xx>w*.88;m[:,fringe]=0;t[6,fringe]=1;t[7:10,fringe]=0;t[12,fringe]=0
        if support>.5:
            negative=(np.abs(xx-col)<=2)&(yy>h*.23)&(yy<h*.70);t[:2,negative]=0;m[:2,negative]=.45;t[6,negative]=0;t[7:10,negative]=0;t[12,negative]=0;changes=dict(known_nonpainting_page_line_column=col,negative_pixels=int(negative.sum()),annotation_provenance='assistant_visual_pageframe_not_user_gold',vertical_support=support)
    file=Path(r['data']);np.savez_compressed(file,rgb=rgb,target=t,mask=m);r.update(data_sha256=digest(file),final_visual_filter=changes)
jsave(CURR/'paired_targets_manifest.json',manifest)
print('PAIR_FINAL_FILTER',[(r['id'],r['final_visual_filter']) for r in manifest['records'] if r['final_visual_filter']],flush=True)
