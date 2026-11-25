"""Artifact integrity and split leakage audit; no robot imports or connections."""
import sys,json,hashlib,ast
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB,ROOT
configure()
import numpy as np
from PIL import Image
manifest_file=LAB/'datasets'/'manifest_rock_cunfa_v04.json'
m=json.loads(manifest_file.read_text(encoding='utf-8'))
by_id={r['id']:r for r in m['real']};groups={};real_shapes={}
for r in m['real']:
    assert Path(r['image']).drive.upper()=='E:'
    assert Path(r['labels']).drive.upper()=='E:'
    groups.setdefault(r['group'],set()).add(r['split'])
    im=Image.open(r['image']);w,h=im.size
    with np.load(r['labels']) as z:
        assert z['targets'].shape==(2,h,w)
        assert z['weights'].shape==(2,h,w)
        assert z['targets'].dtype==np.uint8 and z['weights'].dtype==np.uint8
        assert np.any(z['targets'][0]>100),r['id']
    real_shapes[r['id']]=(h,w)
for r in m['synthetic']:
    parent=by_id[r['parent_id']]
    assert r['group']==parent['group'] and r['split']==parent['split']
    assert r['split']!='test'
    im=Image.open(r['image']);assert im.size==(real_shapes[parent['id']][1],real_shapes[parent['id']][0])
assert all(len(v)==1 for v in groups.values()),'Source-group split leakage'
assert not any(r['id'].startswith('ref_12_') for r in m['real'])
for p in (ROOT/'gongbi_training').glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
for p in (ROOT/'scripts').glob('cunfa_*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
report=dict(real=len(m['real']),paired=len(m['synthetic']),source_groups=len(groups),group_leakage=False,
            labels_and_images_aligned=True,all_generated_artifacts_on_E=True,
            no_test_source_variants_in_training=True,manifest_sha256=hashlib.sha256(manifest_file.read_bytes()).hexdigest())
(LAB/'rock_cunfa_v04'/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
if '--run' in sys.argv:
    import torch
    run=LAB/'runs'/sys.argv[sys.argv.index('--run')+1]
    config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    summary=json.loads((run/'training_summary.json').read_text(encoding='utf-8'))
    assert config['data_manifest_sha256']==report['manifest_sha256']
    assert summary['successful_optimizer_updates']==3000 and summary['skipped_optimizer_updates']==0
    checkpoints={}
    for name in ('best.pt','last.pt'):
        ckpt=torch.load(run/name,map_location='cpu',weights_only=False)
        assert all(torch.isfinite(t).all() for t in ckpt['model'].values())
        checkpoints[name]=dict(step=ckpt['step'],sha256=hashlib.sha256((run/name).read_bytes()).hexdigest())
    outputs=json.loads((run/'evaluation'/'summary.json').read_text(encoding='utf-8'))
    tests=[r for r in outputs if r['split']=='test']
    assert {r['id'] for r in tests}=={r['id'] for r in m['real'] if r['split']=='test'}
    for r in outputs:
        path=run/'evaluation'/r['id']
        for name in ('ink.png','ink_coverage_probability.png','ink_density_probability.png','centerlines.png','comparison.jpg'):
            im=Image.open(path/name);im.verify()
        payload=json.loads((path/'paths.json').read_text(encoding='utf-8'))
        assert payload['ready_for_robot'] is False
    result=dict(**report,checkpoints=checkpoints,successful_optimizer_updates=summary['successful_optimizer_updates'],
                skipped_optimizer_updates=summary['skipped_optimizer_updates'],all_test_examples_rendered=len(tests),
                all_weight_tensors_finite=True,all_prediction_images_valid=True,ready_for_robot=False)
    (run/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('TRAINED_RESULT_VERIFIED',json.dumps(result))
