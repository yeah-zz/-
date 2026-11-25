import sys
sys.dont_write_bytecode=True
from joint_ink_structure_data import ROOT,partial_reference
from rock_v15_common import read,jsave,digest
rows=read(ROOT/'additional_local_proposals.json');manifest=read(ROOT/'manifest.json')
if any(r['id'].startswith('C') for r in manifest['records']):raise FileExistsError('Additional references already recorded')
accepted={'C02','C05','C06','C08','C11'}
for r in rows:
    r.update(status='assistant_partial_positive' if r['id'] in accepted else 'unknown',expert_verified=False,review_method='assistant visual review of original and overlaid proposal on ADDITIONAL_ROCK_REVIEW.jpg',note='visible isolated fine ink segment' if r['id'] in accepted else 'ambiguous wash, broad junction or foliage; keep unknown')
    if r['id'] in accepted:manifest['records'].append(partial_reference(r,'train'))
manifest['actual_train_source_groups']=['G01','G03'];manifest['unknown_proposals_ignored'] += [r['id'] for r in rows if r['status']=='unknown'];manifest['additional_local_review_sha256']=digest(ROOT/'ADDITIONAL_ROCK_REVIEW.jpg')
jsave(ROOT/'additional_local_reviewed.json',rows);jsave(ROOT/'manifest.json',manifest);print('Added 5 nonexpert partial rock lines; 6 proposals remain unknown')
