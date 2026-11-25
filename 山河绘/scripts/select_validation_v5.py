"""Select imagery before any new-image inference; retain provenance and exclusions."""
from collect_validation_v5 import *
import hashlib,numpy as np,cv2
SELECT=[132908,132835,132837,132840,132841,132926,149613,142642,130940,132838,149614,144286,149413,132839,132836,149791,132834,144829,160947,156183]
MET=[36055,39548,40508,63207]
def process(row):
    sid=row['id'];url=row.get('highres_url') or row['image_url'];dest=ROOT/'selected_originals'/(sid+'.jpg');dest.parent.mkdir(exist_ok=True)
    try:
        if not dest.exists():dest.write_bytes(get(url))
    except Exception:
        url=row['image_url'];dest.write_bytes(get(url))
    im=Image.open(dest).convert('RGB');native=list(im.size);im.thumbnail((1280,1280),Image.Resampling.LANCZOS)
    target=ROOT/'validation_inputs'/(sid+'.png');target.parent.mkdir(exist_ok=True);im.save(target)
    return dict(row,selected_image=str(target),original_image=str(dest),download_url=url,original_size=native,working_size=list(im.size),working_sha256=digest(target),sha256_original=digest(dest),used_for_training=False)
def phash(path):
    a=np.asarray(Image.open(path).convert('L').resize((32,32)),np.float32);v=cv2.dct(a)[:8,:8];return (v>np.median(v[1:])).ravel()
if __name__=='__main__':
    manifest=read(ROOT/'candidates_manifest.json');lookup={r['id']:r for r in manifest['records']};rows=[lookup['cma_'+str(i)] for i in SELECT]
    for mid in MET:
        obj=json.loads(get('https://collectionapi.metmuseum.org/public/collection/v1/objects/'+str(mid)));jsave(ROOT/'source_metadata'/('met_'+str(mid)+'.json'),obj)
        if not obj.get('isPublicDomain') or not obj.get('primaryImage'):raise RuntimeError('No public image for '+str(mid))
        rows.append(dict(id='met_'+str(mid),image_url=obj['primaryImage'],source_page=obj['objectURL'],source_group='The Metropolitan Museum of Art',title=obj['title'],license='Public Domain',kind='painting' if mid!=63207 else 'manual_page',artwork_group=str(mid)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(process,rows))
    old=[]
    historical=Path(r'E:\Codex_Workspace_Data\Landscape_Web_Test_20261022')
    for p in historical.rglob('*'):
        if p.suffix.lower() in ['.png','.jpg','.jpeg']:
            try:old.append((str(p),phash(p)))
            except Exception:pass
    for r in records:
        h=phash(r['selected_image']);near=min([(int(np.count_nonzero(h!=hh)),p) for p,hh in old],default=(64,None));r['nearest_known_old_full_image_phash_hamming']=near[0];r['nearest_known_old_image']=near[1];r['possible_duplicate_old_full_image']=near[0]<=5
    # Freeze all full images: no artwork-only manual crop or extraction-dependent selection.
    duplicates=[r['id'] for r in records if r['possible_duplicate_old_full_image']]
    jsave(ROOT/'validation_manifest.json',dict(records=records,selection='image content and clarity only, before v4/v5 inference',old_CMA_accessions_excluded=['1955.37.1','1954.583','1997.94','1997.95'],possible_old_duplicates=duplicates,known_old_scan_scope='Landscape_Web_Test_20261022 full images plus previously logged collection IDs; historical training crop provenance not exhaustive',same_album_leaves_correlated=True,independent_artworks_not_equal_to_image_count=True,manual_stroke_gold=False,expert_verified=False,work_long_edge_limit=1280,selected_before_new_image_inference=True))
    contact([dict(r,image=r['selected_image'],status='ok') for r in records])
    print('SELECTED_VALIDATION',len(records),'possible_duplicates',duplicates,flush=True)
