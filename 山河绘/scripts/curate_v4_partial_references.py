"""Record explicit assistant visual decisions; never convert all proposals to gold."""
import sys,json
sys.dont_write_bytecode=True
from audit_v4_local_repairs import OUT,dump,sha
doc=json.loads((OUT/'local_reference_proposals.json').read_text(encoding='utf-8'))
# Read side-by-side source/proposal cards: only isolated visible ink segments.
accepted={'L02','L05','L06','L09','L10','L11','L12','L14'}
reasons={'L01':'broad branch edge: physical stroke center uncertain','L03':'multiple adjacent needles and a bend: identity ambiguous','L04':'junction and multiple needle directions','L07':'small U-shaped arc: may be boundary of ink rather than stroke center','L08':'broad branch underside: ambiguous center','L13':'wide pale band mixed with adjacent dark ink','L15':'crossing of faint lines: stroke assignment uncertain','L16':'outer broad band: center uncertain'}
for r in doc['records']:
    r['status']='assistant_partial_positive' if r['id'] in accepted else 'unknown'
    r['review_method']='assistant inspected original RGB and proposed line side by side on local_reference_review_cards.jpg'
    r['review_reason']='visible isolated thin ink segment; approximate image center only' if r['id'] in accepted else reasons[r['id']]
    r['expert_verified']=False;r['pixel_tolerance']=1.5
    r['source_sha256']=sha(OUT/'baseline'/r['image_id']/'input.png')
background=[('B01','G01_01',[25,25,80,60]),('B02','G01_01',[330,30,470,180]),('B03','G01_01',[330,190,475,215]),('B04','G02_01',[20,20,80,75]),('B05','G02_01',[380,35,465,165])]
doc.update(background_regions=[dict(id=k,image_id=i,box=b,status='assistant_background',expert_verified=False) for k,i,b in background],not_training_labels_yet=False,neural_training_used=False,reference_level='nonexpert partial source-image review; not exact artist stroke or full-image gold',selection_bias='positive proposals selected from new additions; this set diagnoses recovery cases only and cannot estimate overall accuracy',unknown_policy='unknown and unlabeled pixels are not negatives',review_cards_sha256=sha(OUT/'local_reference_review_cards.jpg'))
dump(OUT/'actual_local_partial_references.json',doc)
print('8 partial positive segments; 8 unknown proposals; 5 background regions; zero expert gold labels')
