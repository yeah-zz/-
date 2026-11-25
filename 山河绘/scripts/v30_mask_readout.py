# -*- coding: utf-8 -*-
"""Separate post-hoc mask-to-centerline readout diagnostic; no training."""
from rock_v15_common import LAB,read,jsave,FONT,SMALL
from pathlib import Path
import sys,types,importlib,gzip,pickle,time
import numpy as np,cv2
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from PIL import Image,ImageDraw
import v30_public_benchmark as exp
from topology_trace_v13 import graph_paths
import v26_tone_coverage as metrics

def run():
    root=exp.ROOT/'mask_readout_diagnostic';package=types.ModuleType('v30_cnnve');package.__path__=[str(exp.ROOT/'sources'/'SketchKit'/'sketchkit'/'vectorization'/'LineDrawer'/'CNNVE')];sys.modules['v30_cnnve']=package;importlib.import_module('v30_cnnve.utils.graph_utils');records=[]
    exp.PAD_MODE='edge'
    # Only these two development examples, predeclared before this readout.
    for row in [r for r in exp.all_rows() if r['id'] in ['R04','H02']]:
        for view in ['raw','normalized']:
            source=exp.ROOT/'results_edge_padding'/'linedrawer'/view/row['id'];doc=read(source/'trajectory.json');started=time.perf_counter()
            with gzip.open(source/'model_input.graph_merge','rb') as f:dic=pickle.load(f)
            inp,geom=exp.model_input(row,view);h,w=inp.shape;union=np.zeros((h,w),np.uint8);paths=[];widths=[];group_ids=[]
            for gid,ids in enumerate(dic['MergeList']):
                mask=np.zeros((h,w),np.uint8)
                for idx in ids:mask=np.maximum(mask,dic['SplitMaskDic'][idx]['update_mask'][:,:,0])
                union=np.maximum(union,mask);yy,xx=np.where(mask>0)
                if len(xx)<4:continue
                x0,x1=int(xx.min()),int(xx.max())+1;y0,y1=int(yy.min()),int(yy.max())+1;small=np.pad(mask[y0:y1,x0:x1]>0,3);sk=skeletonize(small);a=small.astype(np.float32);sm=cv2.GaussianBlur(a,(0,0),1);gx=cv2.Sobel(sm,cv2.CV_32F,1,0);gy=cv2.Sobel(sm,cv2.CV_32F,0,1);jxx=cv2.GaussianBlur(gx*gx,(0,0),3);jyy=cv2.GaussianBlur(gy*gy,(0,0),3);jxy=cv2.GaussianBlur(gx*gy,(0,0),3);theta=.5*np.arctan2(2*jxy,jxx-jyy)+np.pi/2;field=np.zeros((8,*small.shape),np.float32);field[0]=a;field[1]=np.cos(2*theta);field[2]=np.sin(2*theta);field[3:5]=field[1:3];traces,_=graph_paths(sk,field,True,min_length=6);dist=distance_transform_edt(small)
                for p in traces:
                    xy=np.rint(p).astype(int);paths.append(p+[x0-3,y0-3]);widths.append(np.maximum(1,2*dist[xy[:,1],xy[:,0]]));group_ids.append(gid)
            dest=root/view/row['id'];dest.mkdir(parents=True,exist_ok=True);meta=dict(doc['metadata']);meta.update(readout='individual_native_merged_stroke_masks_skeletonized_separately',not_original_paper_centerline_export=True,posthoc_development_diagnostic=True,not_new_blind_test=True,native_stroke_group_ids=group_ids,minimum_curve_length=6,additional_cpu_seconds=time.perf_counter()-started)
            exp.save_result(dest,paths,widths,union,meta,geom,doc['runtime_seconds']+meta['additional_cpu_seconds']);out=read(dest/'trajectory.json');z=np.load(row['mask_file']);refs=read(row['reference_file'])['paths'];rgb=z['rgb'];mm=metrics.evaluate_paths(refs,[np.asarray(p) for p in out['paths']],z['valid'],z['bg'],rgb);records.append(dict(id=row['id'],view=view,metrics=mm,posthoc_development_not_blind=True));Image.fromarray(exp.line_image(rgb.shape[:2],out['paths'])).save(dest/'centerlines.png');Image.fromarray(exp.line_image(rgb.shape[:2],out['paths'],rgb)).save(dest/'overlay.png');print('MASK_READOUT',row['id'],view,mm,flush=True)
    jsave(root/'evaluation.json',dict(records=records,no_finetuning=True,posthoc_diagnostic=True,not_original_line_drawer_export=True,not_used_to_select_parameters=True))
    board=Image.new('RGB',(1600,660),'#f3f0e8');d=ImageDraw.Draw(board);d.text((7,5),'额外诊断：同一LineDrawer预测，原生图边读取 / 逐笔墨迹掩膜读取（非新训练、非盲测）',font=FONT,fill='black')
    for i,key in enumerate(['R04','H02']):
        row=next(r for r in exp.all_rows() if r['id']==key)
        for j,(title,file) in enumerate([('原图',Path(row['raw'])),('旧v28',exp.ROOT/'results_edge_padding'/'v28'/'raw'/key/'centerlines.png'),('LineDrawer图边读取',exp.ROOT/'results_edge_padding'/'linedrawer'/'raw'/key/'centerlines.png'),('LineDrawer掩膜读取',root/'raw'/key/'centerlines.png')]):
            if not file.exists():Image.fromarray(exp.line_image(tuple(row['shape']),read(file.parent/'trajectory.json')['paths'])).save(file)
            d.text((j*400+6,35+i*310),key+' '+title,font=SMALL,fill='black');metrics.v21.paste(board,np.asarray(Image.open(file).convert('RGB')),j*400+5,65+i*310,390,265)
    board.save(root/'MASK_READOUT_COMPARISON.jpg',quality=97)

if __name__=='__main__':run()
