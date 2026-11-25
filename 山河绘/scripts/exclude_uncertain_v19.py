from review_guided_train_v19 import *
manifest=read(ROOT/'manifest.json');rows=read(ROOT/'review_audit.json')['rows'];draft={r['id']:r for r in read(ROOT/'draft_candidates_snapshot.json')};folder=ROOT/'data_v2';folder.mkdir(exist_ok=True);out=[]
for r in manifest['records']:
    z=np.load(r['data']);arrays={k:z[k].copy() for k in z.files};h,w=arrays['rgb'].shape[:2];yy,xx=np.mgrid[:h,:w];exclude=np.zeros((h,w),np.float32)
    for a in rows:
        if a['sample_id']!=r['id'] or a['label']!='uncertain':continue
        q=draft[a['id']];p=np.array(q['p']);n=np.array(q['normal']);u=np.array([n[1],-n[0]]);along=(xx-p[0])*u[0]+(yy-p[1])*u[1];normal=(xx-p[0])*n[0]+(yy-p[1])*n[1];exclude[(np.abs(along)<=3.5)&(np.abs(normal)<=10)&(arrays['valid']>0)]=1
    arrays['uncertain_exclusion']=exclude;arrays['mask'][:,exclude>0]=0;arrays['control_mask'][:,exclude>0]=0;path=folder/(r['id']+'.npz');np.savez_compressed(path,**arrays);out.append(dict(r,data=str(path),source_v17_data_sha256=r.get('data_sha256'),data_sha256=digest(path),uncertain_excluded_pixels=int(exclude.sum())))
manifest.update(records=out,uncertain_excluded_from_supervision_and_distillation=True,previous_training_trial_not_promoted=True);jsave(ROOT/'manifest_v2.json',manifest);print('UNCERTAIN_EXCLUSION_PIXELS',sum(r['uncertain_excluded_pixels'] for r in out),flush=True)
