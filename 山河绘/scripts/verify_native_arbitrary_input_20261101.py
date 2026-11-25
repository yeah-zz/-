from refine_native_fine_faint_20261101 import *
import shutil
def main():
    folder=ROOT/'independent_entry_check';folder.mkdir(exist_ok=True);source=BASE/'originals/T01.png';renamed=folder/'unseen_filename.png';shutil.copy2(source,renamed);reads=[]
    def hook(event,args):
        if event=='open' and args and isinstance(args[0],str):reads.append(args[0])
    sys.addaudithook(hook);torch.set_num_threads(4);cv2.setNumThreads(2);out=folder/'result'
    if not (out/'entry.json').exists():save_result(renamed,BASE/'outputs/T01/complete/complete_paths.json',out)
    a=np.asarray(Image.open(out/'lines.png'));b=np.asarray(Image.open(OUTPUTS/'T01/lines.png'));assert np.array_equal(a,b),'Renaming source changed inference'
    forbidden=[p for p in reads if str(ROOT/'router').lower() in p.lower() and ('manifest.json' in p or '_partial.npz' in p or '_labels.png' in p)];assert not forbidden,forbidden
    jsave(folder/'verification.json',dict(pixel_identical_after_source_rename=True,training_annotation_files_read=forbidden,source_sha256=digest(renamed),baseline_result_sha256=digest(OUTPUTS/'T01/lines.png'),independent_result_sha256=digest(out/'lines.png'),model_sha256=digest(ROOT/'router/router.pt'),inference_uses_sample_ids=False,actually_drawn=False));print('INDEPENDENT_ENTRY_IDENTICAL',flush=True)
if __name__=='__main__':main()
