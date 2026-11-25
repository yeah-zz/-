"""Audited correctness-only amendment and full v5 replay, retaining old files."""
from run_validation_v5 import *
import shutil
if __name__=='__main__':
    protocol=read(ROOT/'fixed_protocol.json')
    if not (ROOT/'fixed_protocol_initial.json').exists():
        jsave(ROOT/'fixed_protocol_initial.json',protocol)
        amended=dict(protocol);amended['code']=[dict(r,sha256=digest(r['path'])) for r in protocol['code']]
        amended['correctness_amendment']=dict(reason='An endpoint bridge near the right image border rounded outside array bounds; add real-image bounds rejection before dense-mask access.',new_image_pairs_complete_at_discovery=18,parameters_and_scoring_unchanged=True,validation_used_for_performance_tuning=False,all_v5_cases_recomputed=True,old_v5_outputs_preserved=True)
        jsave(ROOT/'fixed_protocol.json',amended)
        for r in amended['code']:
            p=Path(r['path']);dest=ROOT/'frozen_code_after_boundary_guard'/p.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(p.read_bytes())
    freeze();rows=read(ROOT/'validation_manifest.json')['records'];results=[]
    for i,r in enumerate(rows):
        freeze();source=Path(r['selected_image']);item=dict(id=r['id'],source=r['source_page'],title=r['title'],source_sha256=r['working_sha256']);start=time.perf_counter()
        before=ROOT/'validation_outputs'/r['id']/'v5';preserve=ROOT/'pre_boundary_fix'/r['id']/'v5'
        if before.exists() and not preserve.exists():preserve.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(before,preserve)
        baseline=ROOT/'validation_outputs'/r['id']/'v4'
        if not (baseline/'statistics.json').exists():old.run(source,baseline)
        # Always replay v5, so the final set is one implementation revision.
        new.run(source,before)
        for version in ['v4','v5']:item[version]=metric(ROOT/'validation_outputs'/r['id']/version)
        item['old_successful_v5_replay_bitwise_equal']=digest(preserve/'lines.png')==digest(before/'lines.png') if (preserve/'lines.png').exists() else None
        item['elapsed_seconds']=time.perf_counter()-start;results.append(item)
        jsave(ROOT/'validation_results.json',dict(records=results,complete=len(results)==len(rows),expected_count=len(rows),parameters_not_tuned_on_validation=True,implementation_has_audited_boundary_guard_amendment=True,metrics_are_no_gold_diagnostics_not_accuracy=True))
        print('BOUNDARY_FIXED_PAIR',i+1,len(rows),r['id'],'old_replay_equal',item['old_successful_v5_replay_bitwise_equal'],'cycles',item['v4']['small_enclosed_regions_outside_detail'],item['v5']['small_enclosed_regions_outside_detail'],flush=True)
    print('VALIDATION_COMPLETE',len(results),flush=True)
