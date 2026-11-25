from extract_faint_joint_20261101 import *
def main():
    records=[read(ROOT/'fresh_inputs'/(i+'.json')) for i in ['N01','N02','N03']];assert len({r['sha256'] for r in records})==3
    for r in records:
        out=ROOT/'fresh_outputs'/r['id']
        if not (out/'entry.json').exists():extract(r['path'],out)
    jsave(ROOT/'fresh_results.json',dict(records=[read(ROOT/'fresh_outputs'/r['id']/'entry.json') for r in records],all_fresh_results_retained=True,new_training_steps=0));print('FRESH_THREE_COMPLETE',flush=True)
if __name__=='__main__':main()
