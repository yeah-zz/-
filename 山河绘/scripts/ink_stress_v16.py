# -*- coding: utf-8 -*-
"""Blank-paper controls, adjacent strokes and faded-real-image diagnostics."""
from ink_eval_v16 import *

def stress(stage,fusion=False):
    torch.set_num_threads(4);cv2.setNumThreads(2);model=load(stage);run_root=ROOT/f'stage_{stage:02d}'/('observed_fusion' if fusion else '');settings=read(run_root/'calibration.json')['selected'];old=old_model().cuda();old.load_state_dict(torch.load(PARENT,map_location='cuda',weights_only=False)['model']);old.eval();link,threshold=load_links(4);folder=run_root/'stress';folder.mkdir(exist_ok=True);rows=[];pictures=[]
    def plans(rgb):
        p=old_infer(old,rgb);before=frozen_paths(p);before,_,_=connect(before,p[0],rgb,link,threshold);before=[q for q in before if np.linalg.norm(np.diff(q,axis=0),axis=1).sum()>=12];pred=predict(model,rgb)
        if fusion:
            from ribbon_fusion_v16 import fusion_decode
            after,sk,info=fusion_decode(pred,rgb,**settings['config'])
        else:after,sk,info=decode(pred,settings['high'],settings['low'],settings['kind'])
        return before,after,sk,pred
    for i,noise in enumerate([0.,.5,1.,2.,3.,4.]):
        rng=np.random.default_rng(166000+i);g=np.ones((256,256,3),np.float32)*225;g+=cv2.resize(rng.normal(0,1,(5,5)).astype(np.float32),(256,256))[:,:,None]*4;g+=rng.normal(0,noise,(256,256,1));rgb=np.clip(g,0,255).astype(np.uint8);before,after,sk,pred=plans(rgb);row=dict(id='blank_'+str(i),kind='known_blank_control',noise_std=noise,before_path_count=len(before),after_path_count=len(after),before_line_pixel_fraction=float(raster(before,rgb.shape[:2]).mean()),after_line_pixel_fraction=float(sk.mean()));rows.append(row)
        if i in [2,4]:pictures.append((row['id'],rgb,before,after))
    for i,(width,gap,alpha) in enumerate([(2,2,.35),(2,4,.10),(6,2,.55),(10,4,.35)]):
        rgb=np.full((256,256,3),235,np.uint8);truth=[]
        for x in [105,105+width+gap]:
            u=np.linspace(0,1,180);q=np.stack((x+np.sin(u*2.5)*4,25+u*205),1);truth.append(q);canvas=np.zeros((256,256),np.float32);cv2.polylines(canvas,[np.rint(q).astype(np.int32)],False,alpha,width);rgb=np.clip(rgb*(1-canvas[:,:,None]),0,255).astype(np.uint8)
        before,after,sk,pred=plans(rgb);bsk=raster(before,rgb.shape[:2]).astype(bool)
        def separation(binary):
            dist=distance_transform_edt(~binary) if binary.any() else np.full(binary.shape,1e5);coverage=[]
            for q in truth:
                xy=np.rint(q[20:-20]).astype(int);coverage.append(float((dist[xy[:,1],xy[:,0]]<=1.5).mean()))
            n,labels,stats,_=cv2.connectedComponentsWithStats(binary[65:190,85:155].astype(np.uint8),8);components=int((stats[1:,cv2.CC_STAT_AREA]>=25).sum());return dict(stroke_coverage_1p5px=coverage,central_components=components,both_separate=bool(min(coverage)>=.85 and components==2))
        row=dict(id='parallel_'+str(i),kind='known_two_stroke_control',width=width,white_gap=gap,opacity=alpha,before=separation(bsk),after=separation(sk));rows.append(row);pictures.append((row['id'],rgb,before,after))
    jsave(folder/'known_controls.json',dict(records=rows,known_blank_and_stroke_count=True,not_real_artwork_accuracy=True));board=Image.new('RGB',(1500,340*len(pictures)+40),'#f5f3ec');d=ImageDraw.Draw(board);d.text((7,5),'空白纸／相邻双线控制：程序图，不是新画作',font=FONT,fill='black')
    for i,(identity,rgb,before,after) in enumerate(pictures):
        for j,(title,a) in enumerate([('程序输入',rgb),('v15 实际输出',line_view(rgb.shape[:2],before,True)),('v16 实际输出',line_view(rgb.shape[:2],after,True))]):d.text((j*500+6,40+i*340),identity+' '+title,font=SMALL,fill='black');paste(board,a,j*500+8,70+i*340,484,300)
    board.save(folder/'known_controls.jpg',quality=95)
    refs=reference_records()+read(V15/'new_labels'/'manifest.json')['records'];faded=[];board=Image.new('RGB',(1600,430*2+40),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,5),'已有真实图人工减淡至 30% 的诊断：不是天然淡墨新图，参考保持不变',font=FONT,fill='black')
    for i,identity in enumerate(['R03','N10']):
        r=next(r for r in refs if r['id']==identity);full=np.asarray(Image.open(r['image']).convert('RGB'));h,w=full.shape[:2];_,_,paper=photometry(full);paper_rgb=np.maximum(full.astype(np.float32),paper[:,:,None]*255);rgb=np.clip(paper_rgb+(full-paper_rgb)*.3,0,255).astype(np.uint8);left,top,cw,ch=r.get('native',[0,0,w,h]);source=rgb[top:top+ch,left:left+cw];before,after,sk,pred=plans(source);before=[q+np.array([left,top]) for q in before];after=[q+np.array([left,top]) for q in after];doc=read(r['labels']);_,_,mask,_=fields(doc);native=np.zeros((h,w),np.float32);native[top:top+ch,left:left+cw]=1;mask*=native[None];reference=[s['points'] for s in doc['strokes']];row=dict(id=identity,input_type='existing_real_image_artificially_faded',contrast_multiplier=.3,before=path_metrics(reference,before,mask[3]),after=path_metrics(reference,after,mask[3]),real_natural_faint_ink_accuracy=False);faded.append(row);out=folder/identity;out.mkdir(exist_ok=True);Image.fromarray(rgb).save(out/'faded_input.png');jsave(out/'metrics.json',row);jsave(out/'trajectory.json',dict(paths=[q.round(2).tolist() for q in after],ready_for_robot=False,actually_drawn=False))
        for j,(title,a) in enumerate([('人工减淡输入',rgb),('局部参考（非预测）',line_view((h,w),reference)),('v15 输出',line_view((h,w),before,True)),('v16 输出',line_view((h,w),after,True))]):d.text((j*400+5,40+i*430),identity+' '+title,font=SMALL,fill='black');paste(board,a,j*400+7,75+i*430,386,385)
        print('FADED_REAL_CHECK',identity,round(row['before']['point_coverage_3px'],3),round(row['after']['point_coverage_3px'],3),flush=True)
    board.save(folder/'faded_real_comparison.jpg',quality=95);jsave(folder/'faded_real_summary.json',dict(records=faded,diagnostic_not_natural_artwork_accuracy=True));print('V16_STRESS_COMPLETE',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',type=int);p.add_argument('--fusion',action='store_true');a=p.parse_args();stress(a.stage,a.fusion)
