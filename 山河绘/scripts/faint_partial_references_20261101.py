# -*- coding: utf-8 -*-
"""Assistant partial references drawn after RGB inspection, not model targets copied."""
from faint_diagnostics_20261101 import *
from scipy.ndimage import distance_transform_edt

# Local native-pixel anchors. Only explicit visible segments are supervised.
# A segment ending before a dry gap is not an annotation to connect that gap.
REFS={
'T01_1':dict(kind='faint_ink',ink=[[[106,12],[110,45],[112,72],[116,102],[119,128]],[[145,13],[145,48],[143,83],[145,113],[143,145],[142,177],[142,208]]]),
'T01_2':dict(kind='one_sided_ink_and_paper',ink=[[[58,8],[57,33],[55,60],[60,88],[60,111],[53,145],[55,172],[58,202],[64,238]]],negative_boxes=[[102,15,240,240]]),
'T01_3':dict(kind='dry_ink',ink=[[[174,75],[175,107],[178,138],[178,164],[181,185]],[[134,12],[139,39],[142,67],[146,95],[148,123]]]),
'T06_1':dict(kind='faint_ink',ink=[[[64,120],[60,146],[59,170],[61,194],[62,219]],[[203,20],[200,44],[202,72],[208,94],[211,121],[218,145]]]),
'T06_2':dict(kind='wash_boundary_and_ink',ink=[[[161,26],[164,52],[166,77],[164,103],[165,130]]],boundary=[[[5,79],[20,73],[34,69],[42,66],[55,49]]],negative_boxes=[[5,5,65,35]]),
'T06_3':dict(kind='dry_curved_ink',ink=[[[56,106],[78,103],[103,92],[125,77],[145,57]],[[86,179],[109,177],[130,170],[153,164],[176,155]]]),
'T17_1':dict(kind='parallel_faint_ink',ink=[[[106,49],[113,77],[121,104],[129,127],[140,158],[149,188],[159,220]],[[158,10],[166,39],[170,67],[174,94],[183,124],[186,148]]]),
'T17_2':dict(kind='dry_ink',ink=[[[93,68],[83,89],[73,110],[60,137],[49,163]],[[145,76],[136,102],[125,126],[111,148],[101,174]]]),
'T17_3':dict(kind='faint_ink',ink=[[[95,37],[102,66],[108,92],[117,119],[128,147],[143,176]],[[137,16],[145,45],[150,72],[159,103],[171,132],[183,159]]]),
'T19_1':dict(kind='faint_ink',ink=[[[97,38],[112,57],[126,78],[140,100],[154,124],[162,149],[173,174]],[[172,38],[179,64],[186,83],[203,99],[210,124],[217,148],[226,177],[233,206]]],negative_boxes=[[60,84,102,138]]),
'T19_2':dict(kind='wash_boundary_and_ink',boundary=[[[12,67],[31,60],[50,56],[61,45],[76,29],[90,18]]],ink=[[[95,93],[115,76],[132,58],[146,36],[152,12]]],negative_boxes=[[5,5,60,34]]),
'T19_3':dict(kind='faint_ink',ink=[[[115,34],[110,58],[102,83],[95,108],[89,131]],[[182,12],[191,39],[197,65],[202,88],[211,115]]]),
'T03_1':dict(kind='calibration_faint_ink',ink=[[[84,72],[93,102],[101,131],[108,158],[117,184],[127,211]],[[176,80],[177,107],[182,132],[194,156],[201,181],[205,207]]]),
'T03_2':dict(kind='paper_negative',negative_boxes=[[12,12,244,244]]),
'T03_3':dict(kind='calibration_dry_ink',ink=[[[127,24],[125,53],[122,78],[117,100],[115,123],[108,150],[98,176],[91,195]],[[162,27],[158,51],[154,76],[150,99],[145,123],[140,147]]]),
'T12_1':dict(kind='calibration_faint_ink',ink=[[[79,110],[79,135],[79,159],[80,186],[81,205]],[[115,103],[111,125],[108,148],[103,167],[101,185]]]),
'T12_2':dict(kind='wash_boundary',boundary=[[[16,53],[38,50],[58,44],[75,43],[94,48],[113,51],[135,57],[153,53]],[[188,35],[199,20],[211,14],[223,18],[237,14],[250,22]]]),
'T12_3':dict(kind='calibration_faint_ink',ink=[[[119,146],[116,173],[111,199],[105,225],[98,245]],[[193,142],[182,158],[169,176],[157,199],[145,222]]]),
'T09_1':dict(kind='old_test_faint_ink',ink=[[[137,116],[146,140],[153,165],[153,188],[150,213]],[[95,126],[90,146],[82,169],[74,190],[69,211]]]),
'T09_2':dict(kind='old_test_wash_boundary',boundary=[[[89,66],[102,56],[118,55],[134,48],[146,37],[152,31],[162,36],[173,55]]]),
'T09_3':dict(kind='old_test_faint_ink',ink=[[[51,83],[49,105],[48,131],[50,155],[48,175]],[[90,103],[90,127],[92,150],[93,174]]]),
'T16_1':dict(kind='old_test_faint_ink',ink=[[[211,17],[218,44],[223,71],[225,98],[228,120]]]),
'T16_2':dict(kind='old_test_weak_boundary',boundary=[[[5,99],[17,93],[31,84]]],ink=[[[105,71],[98,91],[89,113],[81,136],[74,155]]]),
'T16_3':dict(kind='old_test_dry_ink',ink=[[[139,79],[154,80],[171,85],[186,93]],[[196,101],[207,111],[217,126],[225,145]]])}

def path_data(points,shape=(256,256)):
    p=np.asarray(points,np.float32);dense=[]
    for a,b in zip(p[:-1],p[1:]):dense.append(np.linspace(a,b,max(2,int(np.linalg.norm(b-a)*3)+1),endpoint=False))
    p=np.r_[np.concatenate(dense),p[-1:]];mask=np.zeros(shape,np.uint8);cv2.polylines(mask,[np.rint(p).astype(np.int32)],False,1,1);distance,indices=distance_transform_edt(mask==0,return_indices=True);tan=np.gradient(p,axis=0);angle=np.arctan2(tan[:,1],tan[:,0]);direction=np.zeros((2,*shape),np.float32);xy=np.rint(p).astype(int);direction[:,xy[:,1],xy[:,0]]=np.array([np.cos(2*angle),np.sin(2*angle)]);direction=direction[:,indices[0],indices[1]];return mask>0,distance,direction

def prepare():
    folder=ROOT/'partial_references';folder.mkdir(parents=True,exist_ok=True);manifest=read(ROOT/'diagnostic_manifest.json');records=[]
    for row in manifest['records']:
        key=row['id'];labels=REFS[key];rgb=np.asarray(Image.open(row['path']).convert('RGB'));target=np.zeros((5,256,256),np.float32);known=np.zeros((4,256,256),np.float32);inkref=np.zeros((256,256),bool);boundref=inkref.copy();overlay=Image.fromarray(rgb);draw=ImageDraw.Draw(overlay)
        for p in labels.get('ink',[]):
            c,d,di=path_data(p);inkref|=c;core=d<=2.;target[0,core]=1.;target[1,core]=1.;target[2:4,core]=di[:,core];known[0,core]=1;known[1,core]=.7;known[2,core]=.7;known[3,core]=.5;draw.line([tuple(q) for q in p],fill=(30,220,60),width=1)
        for p in labels.get('boundary',[]):
            c,d,di=path_data(p);boundref|=c;core=(d<=2)&~(known[0]>0);target[4,core]=1;known[0,core]=1;known[1,core]=1;known[3,core]=1;draw.line([tuple(q) for q in p],fill=(230,55,180),width=1)
        for x,y,xx,yy in labels.get('negative_boxes',[]):known[0,y:yy,x:xx]=1;known[1,y:yy,x:xx]=1;known[3,y:yy,x:xx]=1;draw.rectangle((x,y,xx,yy),outline=(50,120,240),width=1)
        output=folder/(key+'.npz');np.savez_compressed(output,target=target,known=known,ink_reference=inkref,boundary_reference=boundref);overlay.resize((512,512),Image.Resampling.NEAREST).save(folder/(key+'_overlay.png'));records.append(dict(**row,label_file=str(output),label_file_sha256=digest(output),reference_kind=labels['kind'],ink_paths=len(labels.get('ink',[])),boundary_paths=len(labels.get('boundary',[])),references=labels,center_uncertainty_native_px=3.,expert_verified=False,unmarked_not_negative=True))
    jsave(ROOT/'partial_reference_manifest.json',dict(records=records,annotation_kind='Assistant manually chosen visible RGB line segments and color-only boundaries; approximate partial centers, not artist expert or complete line art',raw_anchors_retained=True,automatic_extracted_paths_used_as_truth=False,training_parents=['T01','T06','T17','T19'],calibration_parents=['T03','T12'],old_regression_test_parents=['T09','T16'],test_is_not_new_blind_images=True,uncertain_dry_gaps_not_forced_connected=True));print('PARTIAL_REFERENCES',len(records),sum(r['ink_paths'] for r in records),sum(r['boundary_paths'] for r in records),flush=True)
if __name__=='__main__':prepare()
