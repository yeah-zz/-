# -*- coding: utf-8 -*-
"""Arbitrary image entry. No reference/diagnostic window is used at inference."""
from faint_candidate_fusion_20261101 import *
from pathlib import Path
from run_stability_brush_20261101 import verify_lock
import datetime

def verify_round():
    verify_lock();p=ROOT/'frozen_experiment.json'
    if p.exists():
        for row in read(p)['files']:assert digest(row['path'])==row['sha256'],'Frozen round file changed: '+row['path']

def extract(image,output,prior_output=None):
    verify_round();image=Path(image).resolve();output=Path(output)
    if not output.is_absolute() or output.drive.upper() not in ['E:','F:']:raise ValueError('Use absolute E/F output')
    if (output/'entry.json').exists():raise FileExistsError('Preserve completed extraction')
    output.mkdir(parents=True,exist_ok=True);rgb=np.asarray(Image.open(image).convert('RGB'));torch.set_num_threads(4);cv2.setNumThreads(2)
    if prior_output is None:
        from extract_complete_lines_20261101 import extract as complete
        from refine_native_fine_faint_20261101 import save_result
        im=Image.fromarray(rgb);im.thumbnail((496,496),Image.Resampling.LANCZOS);im.save(output/'evaluation_input.png');complete(output/'evaluation_input.png',output/'upstream_complete');prior_output=output/'upstream_r3';save_result(image,output/'upstream_complete/complete/complete_paths.json',prior_output)
    prior_output=Path(prior_output);parent=read(prior_output/'entry.json');assert parent['source_sha256']==digest(image),'Baseline input mismatch';base=np.asarray(Image.open(prior_output/'lines.png'))<128;protect=np.asarray(Image.open(prior_output/'fan.png'))<128;model=FaintNet().cuda();artifact=torch.load(ROOT/'fit_r2/model.pt',map_location='cpu',weights_only=False);model.load_state_dict(artifact['model']);o=predict(rgb,model);area=mountain_context(rgb);cfg=read(ROOT/'context_calibration.json')['selected']['config'];neural,rows,score=trace(rgb,o,cfg,area&~protect);result,decisions=fuse(rgb,base,protect,rows);assert np.array_equal(result[protect],base[protect]);np.savez_compressed(output/'fields.npz',fields=o)
    for name,m in [('previous',base),('model_candidates',neural),('repaired',result),('protected',protect),('mountain_context',area)]:Image.fromarray(np.where(m,0,255).astype(np.uint8)).save(output/(name+'.png'))
    Image.fromarray(rgb).save(output/'input.png');heat=cv2.applyColorMap(np.rint(o[4]*255).astype(np.uint8),cv2.COLORMAP_VIRIDIS)[:,:,::-1];Image.fromarray(heat).save(output/'color_boundary_probability.png');change=np.repeat(np.where(base,0,255).astype(np.uint8)[:,:,None],3,2);change[result&~base]=[20,135,210];change[base&~result]=[215,70,65];Image.fromarray(change).save(output/'changes.png');jsave(output/'decisions.json',{k:v for k,v in decisions.items() if k!='old_paths'});paths,_=graph.pixel_chains(result);exported=graph.raster(result.shape,paths);Image.fromarray(np.where(result&~exported,0,255).astype(np.uint8)).save(output/'isolated_evidence.png')
    jsave(output/'paths.json',dict(paths=[dict(points=p.tolist(),stroke_identity_verified=False) for p in paths],width=rgb.shape[1],height=rgb.shape[0],coordinate_system='native_image_pixels',ready_for_robot=False,actually_drawn=False,physical_z=None,physical_speed=None));svg=''.join('<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in p)+'"/>' for p in paths);(output/'lines.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {rgb.shape[1]} {rgb.shape[0]}"><g fill="none" stroke="black" stroke-width="1">'+svg+'</g></svg>',encoding='utf-8');stats=dict(previous_pixels=int(base.sum()),repaired_pixels=int(result.sum()),added_pixels=int((result&~base).sum()),removed_pixels=int((base&~result).sum()),whole_candidates_accepted=len(decisions['accepted']),whole_old_fragments_replaced=len(decisions['replaced_old_path_ids']),pending_candidates=len(decisions['pending']),protected_pixels_exact=True,exported_paths=len(paths),isolated_evidence_pixels=int((result&~exported).sum()),robot_connected=False)
    jsave(output/'entry.json',dict(image=str(image),source_sha256=digest(image),parent_entry_sha256=digest(prior_output/'entry.json'),parent_output=str(prior_output),model_sha256=digest(ROOT/'fit_r2/model.pt'),threshold_sha256=digest(ROOT/'context_calibration.json'),fusion_config=CFG_FUSION,manual_regions_or_reference_files_used=False,new_training_updates=0,statistics=stats));font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17);board=Image.new('RGB',(1984,550),'#f3f1eb');d=ImageDraw.Draw(board)
    for j,(path,title) in enumerate([(output/'input.png','原图'),(output/'previous.png','上轮完整流程（含松针）'),(output/'repaired.png','本轮淡线补提候选'),(output/'changes.png','蓝新增 / 红替换')]):im=Image.open(path).convert('RGB');im.thumbnail((496,496),Image.Resampling.LANCZOS);d.text((j*496+4,8),title,font=font,fill='black');board.paste(im,(j*496,48))
    board.save(output/'comparison.jpg',quality=96);verify_round();print('FAINT_NEW_IMAGE_READY',str(output),stats,flush=True);return stats

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--output');p.add_argument('--prior-output');a=p.parse_args();out=Path(a.output) if a.output else ROOT/'manual_outputs'/(Path(a.image).stem+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));extract(a.image,out,a.prior_output)
