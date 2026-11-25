import sys
sys.dont_write_bytecode=True
from mountain_structure_v4_20261101 import *
from html.parser import HTMLParser
checks=[]
def check(name,ok):
    checks.append(dict(name=name,passed=bool(ok)));assert ok,name
records=read(ROOT/'results_all.json')['records'];protocol=read(ROOT/('frozen_protocol_'+CFG['version']+'.json'))
check('frozen_selection_code_unchanged',digest(Path(__file__).with_name('mountain_structure_v4_20261101.py'))==protocol['source_code_sha256'])
check('31_fixed_images_present',len(records)==31)
for r in records:
    ident=r['id'];folder=ROOT/'outputs'/ident;old=np.asarray(Image.open(folder/'baseline.png'))<128;new=np.asarray(Image.open(folder/'lines.png'))<128;z=np.load(folder/'routing_masks.npz');doc=read(folder/'paths.json');paths=[np.asarray(p['points'],np.float32) for p in doc['paths']]
    check(ident+'_no_added_pixels',not (new&~old).any())
    check(ident+'_outside_interior_exact',np.array_equal(new&~z['interior'],old&~z['interior']))
    check(ident+'_protected_detail_exact',np.array_equal(new&z['protect'],old&z['protect']))
    check(ident+'_export_matches_preview',np.array_equal(v4.raster(new.shape,paths),new))
    check(ident+'_source_unchanged',digest(doc['source'])==doc['source_sha256'])
    check(ident+'_offline_only',doc['ready_for_robot'] is False and doc['actually_drawn'] is False and doc['physical_z'] is None and doc['physical_speed'] is None)
    edges=set();duplicate=0
    for p in paths:
        for a,b in zip(p[:-1],p[1:]):
            if np.array_equal(a,b):continue
            e=tuple(sorted((tuple(a),tuple(b))));duplicate+=int(e in edges);edges.add(e)
    check(ident+'_no_duplicate_export_edges',duplicate==0)
    # Check original long-anchor hypotheses against the baseline source records.
    basefolder=LAB/'v4_local_repairs_20261029/candidate'/ident if ident!='N02' else LAB/'simple_landscape_specialist_20261029/round2/native_outputs/N02/v4'
    source=read(basefolder/'paths.json')['paths'];check(ident+'_baseline_not_changed',digest(basefolder/'paths.json')==doc['baseline_paths_sha256'])
    for d in read(folder/'decisions.json'):
        if d['reason']!='original_v4_long_structure_anchor':continue
        p=np.asarray(source[d['id']]['points'],np.float32);m=v4.raster(new.shape,[p]);check(ident+'_whole_anchor_'+str(d['id']),not (m&~new).any())
summary=read(ROOT/'summary.json')
for r in summary['tree_roi_checks']:check(r['id']+'_audited_tree_roi_exact',r['exactly_preserved'])
entry=np.asarray(Image.open(ROOT/'entry_check_r7/structure/lines.png'));batch=np.asarray(Image.open(ROOT/'outputs/G05_01/lines.png'));check('independent_image_entry_matches_batch',np.array_equal(entry,batch))
for r in read(LAB/'stable_line_extractor_20261028/baseline_lock.json')['files']:check('stable_'+Path(r['path']).name,digest(r['path'])==r['sha256'])
check('original_frozen_v4_code_unchanged',digest(Path(__file__).with_name('split_detail_v4_frozen.py'))==read(LAB/'v4_local_repairs_20261029/baseline_manifest.json')['baseline_code_sha256'])
class Links(HTMLParser):
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ['href','src'] and value and not value.startswith(('http:','https:','#','data:')):check('report_'+value,(ROOT/value).exists())
Links().feed((ROOT/'index.html').read_text(encoding='utf-8'))
jsave(ROOT/'FINAL_VERIFICATION.json',dict(checks=checks,count=len(checks),all_passed=True,robot_connected=False,accuracy_established=False))
print('MOUNTAIN_STRUCTURE_VERIFIED',len(checks))
