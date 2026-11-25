import json,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gongbi_training.runtime import configure,LAB
configure()
from gongbi_training.cunfa_collect import CORPUS,sheet
manifest_path=LAB/'datasets'/'manifest_rock_cunfa_v04.json'
m=json.loads(manifest_path.read_text(encoding='utf-8'));a=json.loads((CORPUS/'dataset_audit.json').read_text(encoding='utf-8'))
rejected={r['id'] for r in m['real'] if r['id'].startswith('ref_12_')}
m['real']=[r for r in m['real'] if r['id'] not in rejected]
m['synthetic']=[r for r in m['synthetic'] if r['parent_id'] not in rejected]
m['inference_records']=[r for r in m['inference_records'] if r['id'] not in rejected]
a.update(curated_examples=len(m['real']),curated_web_examples=sum(r['source_type']=='public_web_tutorial' for r in m['real']),
         curated_user_examples=sum(r['source_type']=='user_reference' for r in m['real']),
         examples_per_split=dict(Counter(r['split'] for r in m['real'])),
         paired_variants_per_split=dict(Counter(r['split'] for r in m['synthetic'])),
         used_web_parent_images=len({r['parent_image'] for r in m['real'] if r['source_type']=='public_web_tutorial'}),
         quality_excluded_examples=[dict(id=i,reason='Paper blotches remain in automatic target') for i in sorted(rejected)])
manifest_path.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
(CORPUS/'dataset_audit.json').write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf-8')
sheet(m['real'],'quality_accepted')
aborted=LAB/'runs'/'mobilenet_unet_rock_cunfa_v04'
(aborted/'ABORTED.md').write_text('Stopped during initial training to exclude six noisy-paper weak targets. Not the deliverable model. Final run: mobilenet_unet_rock_cunfa_v04_clean.\n',encoding='utf-8')
print('QUALITY_ACCEPTED',a['curated_examples'],a['examples_per_split'],'paired',a['paired_variants_per_split'],'web originals',a['used_web_parent_images'])
