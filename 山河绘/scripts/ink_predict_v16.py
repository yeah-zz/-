# -*- coding: utf-8 -*-
"""Offline experimental comparison; does not change the existing painting flow."""
from ink_eval_v16 import *
from datetime import datetime,timezone,timedelta

def run(input_path,output_path=None,stage=3,neural_only=False):
    source=Path(input_path);files=[source] if source.is_file() else sorted(p for p in source.iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.webp','.tif'])
    if not files:raise ValueError('No supported images')
    dest=Path(output_path) if output_path else ROOT/'batch_comparisons'/datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    if dest.drive.upper()=='C:':raise ValueError('Outputs must stay on E: or F:')
    dest.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(2);model=load(stage);settings=read(ROOT/f'stage_{stage:02d}'/'calibration.json')['selected'] if neural_only else read(ROOT/'protected_refinement'/'calibration.json')['selected'];old=old_model().cuda();old.load_state_dict(torch.load(PARENT,map_location='cuda',weights_only=False)['model']);old.eval();link,threshold=load_links(4);rows=[]
    for i,path in enumerate(files):
        rgb=np.asarray(Image.open(path).convert('RGB'));p=old_infer(old,rgb);before=frozen_paths(p);before,_,_=connect(before,p[0],rgb,link,threshold);before=[q for q in before if np.linalg.norm(np.diff(q,axis=0),axis=1).sum()>=12];route=monochrome_route(rgb);out=dest/(str(i+1).zfill(3)+'_'+path.stem);out.mkdir(exist_ok=True)
        if route:
            pred=predict(model,rgb)
            if neural_only:
                after,sk,info=decode(pred,settings['high'],settings['low'],settings['kind']);after,_,uncertain=connect(after,np.maximum(pred[0],pred[1]*.8),rgb,link,threshold,max_gap=8)
            else:
                from protected_refine_v16 import center_ribbons,recover_faint
                after,info=center_ribbons(before,rgb,pred,settings['strength']);after,weak=recover_faint(after,rgb,p,pred,old,link,threshold);info['faint_recovery']=weak;uncertain=[]
            np.savez_compressed(out/'prediction.npz',prediction=pred);Image.fromarray(np.uint8(255*(1-pred[1]))).save(out/'predicted_ink_ribbon.png')
        else:after=before;info=dict(original_color_branch_preserved=True)
        qc=quality_audit(after,np.maximum(pred[0],pred[1]*.8) if route else p[0],rgb,uncertain if route else []);jsave(out/'automatic_review.json',qc);jsave(out/'candidate_trajectory.json',dict(paths=[q.round(2).tolist() for q in after],coordinates='pixels',physical_z=None,ready_for_robot=False,actually_drawn=False,experimental=True,monochrome_specialist_used=route,trace_info=info));board=Image.new('RGB',(1500,600),'#f5f3ec');d=ImageDraw.Draw(board);d.text((8,5),'v16 离线候选对照：未替换正式模型，颜色仅表示分笔',font=FONT,fill='black')
        for j,(title,a) in enumerate([('原图',rgb),('v15 候选轨迹',line_view(rgb.shape[:2],before,True)),('v16 候选轨迹',line_view(rgb.shape[:2],after,True))]):d.text((j*500+7,38),title,font=SMALL,fill='black');paste(board,a,j*500+8,70,484,520)
        board.save(out/'comparison.jpg',quality=95);rows.append(dict(source=str(path),source_sha256=digest(path),monochrome_specialist_used=route,comparison=str(out/'comparison.jpg'),ready_for_robot=False));print('V16_OFFLINE_IMAGE',path.name,route,flush=True)
    jsave(dest/'summary.json',dict(records=rows,stage=stage,mode='neural_only' if neural_only else 'protected_refinement',official_model_unchanged=True,robot_connected=False,expert_review_not_replaced=True));print('V16_OFFLINE_OUTPUT',str(dest),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output');p.add_argument('--stage',type=int,default=3);p.add_argument('--neural-only',action='store_true');a=p.parse_args();run(a.input,a.output,a.stage,a.neural_only)
