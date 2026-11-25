# -*- coding: utf-8 -*-
from cross_section_v18 import *
path=ROOT/'review_candidates.json';rows=read(path);rows=[r for r in rows if r.get('kind')!='background'];cases={r['id']:r for r in legacy.records()}
for i,(identity,kind) in enumerate([('R01','known_paper'),('R02','known_paper'),('N06','known_paper'),('N06','photo_edge'),('N09','photo_edge'),('N05','photo_edge')]):
    rgb,doc,bg,valid=legacy.source(cases[identity]);h,w=rgb.shape[:2]
    if kind=='known_paper':
        mask=(bg>0).astype(np.uint8);mask[:8]=0;mask[-8:]=0;mask[:,:8]=0;mask[:,-8:]=0;y,x=np.unravel_index(distance_transform_edt(mask).argmax(),mask.shape);p=np.array([x,y],np.float32)
    else:p=np.array([w-6,h*.5],np.float32)
    u=np.array([0,1],np.float32);n=np.array([-1,0],np.float32);xx=(np.arange(192,dtype=np.float32)[None]-95.5)/3;yy=(np.arange(96,dtype=np.float32)[:,None]-47.5)/3;mx=(p[0]+xx*n[0]+yy*u[0]).astype(np.float32);my=(p[1]+xx*n[1]+yy*u[1]).astype(np.float32);clip=cv2.remap(rgb,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101);encoded=base64.b64encode(cv2.imencode('.png',cv2.cvtColor(clip,cv2.COLOR_RGB2BGR))[1]).decode();rows.append(dict(id=identity+'_bg_'+str(i),sample_id=identity,kind='background',p=p.tolist(),normal=n.tolist(),center=0.,left=-1.,right=1.,contrast=None,snr=None,derived_not_expert=True,background_candidate_type=kind,clip=encoded))
jsave(path,rows);print('BACKGROUND_REVIEW_CANDIDATES',6,flush=True)
