# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
import json,hashlib
import numpy as np,cv2
from scipy.ndimage import distance_transform_edt
from module_stroke_core_20261104 import samples

def read(p):return json.loads(p.read_text('utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def verify(root):
    catalog=read(root/'catalog.json');records=[]
    for row in catalog['scenes']:
        p=read(root/row['id']/'plan.json');assert not p['ready_for_robot'] and not p['adopted']
        assert sha(Path(p['source']))==p['source_sha256']
        acts=p['actions'];curves=p['curves']
        assert [a['sequence'] for a in acts]==list(range(1,len(curves)+1))
        assert sorted(a['curve'] for a in acts)==list(range(len(curves)))
        done=set();last=None;phase=-1
        for a in acts:
            if a['module']!=last:
                assert a['module'] not in done;done.add(a['module']);last=a['module'];phase=-1
            assert a['phase']>=phase;phase=a['phase']
        assert p['stats']['hierarchy_order']['module_reentries']==0
        assert all(np.isfinite(np.asarray(r['points'])).all() and len(r['points'])>=2 for r in curves)
        assert len({i for c in curves for i in c['source_curve_ids']})==sum(len(c['source_curve_ids']) for c in curves)
        records.append(dict(id=row['id'],curves=len(curves),phase_order=True,locked_modules=True))
    assert read(root/'execution_binding.json')['journal'] is None
    # Frozen previous-round digests, not hashes collected after the new run.
    unchanged=[]
    for ident in ('G03_01','G05_03','T02'):
        for b in (500,700):
            key=f'{ident}_{b}';prior=read(LAB/'ink_mask_centerline_20261105'/key/'plan.json')
            assert sha(Path(prior['parent']))==prior['parent_sha256'];unchanged.append(key)
    result=dict(passed=True,plans=records,unchanged_approved_baselines=unchanged,robot_binding_absent=True,
        artistic_accuracy_passed=False,new_training=False)
    (root/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('PLAN_VERIFIED',root.name,len(records),'plans',len(unchanged),'unchanged baselines',flush=True)
    return result

def graph_audit():
    root=LAB/'autonomous_full_image_20261105';records=[]
    for summary in read(root/'results.json'):
        ident=summary['id'];p=read(root/ident/'plan.json');pending=read(root/ident/'pending.json')['curves']
        ids=[i for r in p['curves']+pending for i in r['source_curve_ids']]
        assert sorted(ids)==list(range(p['raw_graph_chain_count']))
        for d in read(root/ident/'junction_diagnostics.json'):
            for a in d['accepted']:
                assert a['mutual_owner_score']>=.54 and a['margin']>=.12
        # No source-gap synthesis: vector samples must stay on the native mask
        # within simplification/interpolation tolerance (not just endpoints).
        mask=np.asarray(__import__('PIL.Image',fromlist=['Image']).open(root/ident/'native_ink.png'))>0
        dist=distance_transform_edt(~mask);factor=p['native_to_viewer_scale'];max_distance=0.
        for r in p['curves']:
            q=samples(np.asarray(r['points'])*factor,.4);xy=np.rint(q).astype(int)
            xy[:,0]=np.clip(xy[:,0],0,dist.shape[1]-1);xy[:,1]=np.clip(xy[:,1],0,dist.shape[0]-1)
            max_distance=max(max_distance,float(dist[xy[:,1],xy[:,0]].max()))
        assert max_distance<=1.5,(ident,max_distance)
        records.append(dict(id=ident,source_chain_partition=True,max_native_distance_to_source_ink=max_distance))
    (root/'graph_verification.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    print('SOURCE_GRAPH_VERIFIED',len(records),flush=True)

if __name__=='__main__':
    verify(LAB/'autonomous_full_image_20261105');verify(LAB/'autonomous_full_image_screened_20261105');graph_audit()
