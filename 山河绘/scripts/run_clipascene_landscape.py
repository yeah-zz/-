"""Run official SceneSketch with isolated dependencies and explicit adaptations.

Whole-scene option: skip foreground segmentation/inpainting; CPU DiffVG and
CPU location MLP retain autograd through GPU CLIP losses. No robot calls.
"""
from pathlib import Path
import argparse
import copy
import json
import os
import sys
import time
import faulthandler
import re
faulthandler.enable()

ROOT = Path(__file__).resolve().parents[1]
STORAGE = Path(r"E:\Codex_Workspace_Data\CLIPascene_20261021")
EXP = STORAGE / "source"
REPO = EXP / "SceneSketch-master"
DEPS = STORAGE / "environment" / "deps"
GPU = STORAGE / "recovered_from_C" / "Temp" / "pip-target-m7b0uqsu" / "lib" / "python"
OUT = ROOT / "offline_previews" / "clipascene_20261021"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["WANDB_MODE"] = "disabled"
os.environ["MPLBACKEND"] = "Agg"
sys.dont_write_bytecode = True
for key, value in {"TEMP":STORAGE/"temp", "TMP":STORAGE/"temp",
                   "PIP_CACHE_DIR":STORAGE/"cache"/"pip", "TORCH_HOME":STORAGE/"cache"/"torch",
                   "HF_HOME":STORAGE/"cache"/"huggingface", "XDG_CACHE_HOME":STORAGE/"cache",
                   "CUDA_CACHE_PATH":STORAGE/"cache"/"cuda", "MPLCONFIGDIR":STORAGE/"cache"/"matplotlib"}.items():
    os.environ[key] = str(value)
sys.path[:0] = [str(GPU), str(DEPS), str(REPO), str(ROOT)]
if hasattr(os, "add_dll_directory") and (GPU / "torch" / "lib").exists():
    os.add_dll_directory(str(GPU / "torch" / "lib"))


def prepare():
    sys.path.insert(0,str(ROOT))
    from PIL import Image, ImageDraw
    import numpy as np
    import cv2
    original = ROOT / "待画作品" / "01_松瀑亭舟_横题山河绘" / "参考原图.png"
    im = Image.open(original).convert("RGB")
    crop = im.crop((195, 190, 1305, 940))
    ImageDraw.Draw(crop).rectangle((0, 0, 530, 124), fill="white")
    crop.save(OUT / "input_landscape.png")
    square = Image.new("RGB", (1110, 1110), "white")
    square.paste(crop, (0, 180))
    square.resize((448, 448), Image.Resampling.LANCZOS).save(OUT / "input_square.png")
    # Exact existing extraction function; no calls to task execution entrypoints.
    # Load the exact two pure extraction functions without importing unrelated
    # execution/UI modules (some require a newer Python than upstream's runtime).
    import ast
    import ink_control, shuimo
    from config import ShuimoConfig
    from original_d_tracer import LandscapeBrushGenerator
    source = ROOT / "scripts" / "preview_contrast_landscape.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node,ast.FunctionDef)
                 and node.name in ("join_source_paths","width_strokes")]
    scope=dict(np=np,cv2=cv2,ink_control=ink_control,shuimo=shuimo,
               ShuimoConfig=ShuimoConfig,Trace=LandscapeBrushGenerator)
    exec(compile(tree,str(source),"exec"),scope)
    width_strokes=scope["width_strokes"]
    bgr = np.asarray(crop)[:, :, ::-1].copy()
    strokes, stats = width_strokes(bgr, .15)
    canvas = np.full_like(bgr, 255)
    for s in strokes:
        pts = np.rint(s["points"]).astype(np.int32)
        cv2.polylines(canvas, [pts], False, (0,0,0), 2, cv2.LINE_AA)
    Image.fromarray(canvas).save(OUT / "existing_extractor.png")
    (OUT / "existing_strokes.json").write_text(json.dumps(strokes,ensure_ascii=False),encoding="utf-8")
    (OUT / "input_preprocessing.json").write_text(json.dumps(dict(
        source=str(original),crop_box=[195,190,1305,940],title_whiteout=[0,0,530,124],
        square_size=1110,vertical_padding=180,model_input_size=224,
        foreground_segmentation=False,inpainting=False,baseline_stats=stats),ensure_ascii=False,indent=2),encoding="utf-8")
    print("INPUT_READY",len(strokes),flush=True)


def train(layer, iterations, paths, seed, simplify_from=None, ratio=0, inference_only=False):
    import numpy as np
    # Compatibility aliases for upstream's old NumPy API use.
    np.int = int
    np.float = float
    import torch
    torch.set_num_threads(4)
    import torch.nn.functional as functional
    if not hasattr(functional,"_pad"):
        functional._pad=functional.pad
    print("TORCH",torch.__version__,"GPU",torch.cuda.get_device_name(0),flush=True)
    print("CUDA_TEST",(torch.ones(2,device="cuda")*2).tolist(),flush=True)
    import clip
    import CLIP_.clip as saliency_clip
    for mod in (clip, saliency_clip):
        original_load = mod.load
        def cached_load(name, device="cpu", jit=False, _load=original_load, **kw):
            if name == "ViT-B/32":
                name = str(EXP / "assets" / "ViT-B-32.pt")
            return _load(name, device=device, jit=jit, **kw)
        mod.load = cached_load
    import pydiffvg
    import sketch_utils
    import config
    import painterly_rendering as official
    # Whole scene path advertised by the authors: do not segment or inpaint.
    sketch_utils.get_mask_u2net = lambda args, im: (im, np.ones((im.height,im.width),dtype=np.float32))
    original_canvas_check=sketch_utils.is_in_canvas
    sketch_utils.is_in_canvas=lambda w,h,path,device: original_canvas_check(w,h,path,torch.device("cpu"))
    weights = [0.]*12
    weights[layer] = 1.
    name = f"layer{layer}_paths{paths}_seed{seed}" if not simplify_from else f"layer{layer}_simplified_ratio{ratio}"
    argv = ["scene",str(OUT/"input_square.png"),"--output_dir",str(OUT),
            "--wandb_name",name,"--num_iter",str(iterations),"--num_paths",str(paths),
            "--seed",str(seed),"--use_gpu","0","--mlp_train","1","--lr","0.0001",
            "--clip_model_name","ViT-B/32","--clip_conv_layer_weights",','.join(map(str,weights)),
            "--clip_fc_loss_weight","0","--save_interval","100","--eval_interval","50",
            "--min_eval_iter","50","--num_aug_clip","4","--use_wandb","0"]
    if simplify_from:
        base=OUT/simplify_from
        argv += ["--width_optim","1","--width_lr","0.00005","--ratio_loss",str(ratio),
                 "--path_svg",str(base/"svg_logs"/"init_svg.svg"),
                 "--mlp_points_weights_path",str(base/"points_mlp.pt"),"--gradnorm","1"]
    sys.argv = argv
    args=config.parse_arguments()
    args.device=torch.device("cuda")
    pydiffvg.set_use_gpu(False)
    pydiffvg.set_device(torch.device("cpu"))
    factory=official.load_renderer
    def cpu_renderer(args,target_im=None,mask=None):
        print("RENDERER_INIT_BEGIN",flush=True)
        local=copy.copy(args)
        local.device=torch.device("cpu")
        result=factory(local,target_im.cpu(),mask.cpu())
        print("RENDERER_INIT_COMPLETE",flush=True)
        return result
    official.load_renderer=cpu_renderer
    class Progress:
        def __init__(self, seq):self.seq=seq;self.start=time.monotonic()
        def __iter__(self):
            for i in self.seq:
                if i%25==0:print("ITER",i,"elapsed_s",round(time.monotonic()-self.start,1),flush=True)
                yield i
        def refresh(self):pass
    official.tqdm=Progress
    os.chdir(REPO)
    start=time.monotonic()
    if inference_only:
        sketch_utils.inference_sketch(args)
        result={}
    else:
        result=official.main(args)
    elapsed=time.monotonic()-start
    timing_scope="full official main including initialization and inference"
    if inference_only:
        log=STORAGE/f"run_layer{layer}.log"
        entries=re.findall(r"ITER (\d+) elapsed_s ([\d.]+)",log.read_text(encoding="utf-8",errors="replace"))
        if not entries or int(entries[-1][0])+1!=iterations:
            raise RuntimeError("Training log does not confirm requested iteration count")
        elapsed=float(entries[-1][1])
        timing_scope="iteration log at final step; excludes setup and export"
    np.save(str(Path(args.output_dir)/"config.npy"),dict(vars(args),**result))
    manifest=dict(official_repository="https://github.com/yael-vinker/SceneSketch",
                  layer=layer,iterations=iterations,initial_paths=paths,seed=seed,
                  elapsed_seconds=elapsed,foreground_segmentation=False,
                  timing_scope=timing_scope,inference_recovered_from_checkpoint=inference_only,
                  rasterizer="official DiffVG CPU",loss_device="GPU",location_mlp="CPU",
                  simplification_ratio=ratio if simplify_from else None,
                  implementation_adaptations=["NumPy deprecated aliases", "local CLIP weight cache",
                                             "whole-scene mask", "CPU-renderer/GPU-loss device split"])
    (Path(args.output_dir)/"experiment_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("RUN_COMPLETE",name,round(elapsed,1),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--prepare",action="store_true")
    ap.add_argument("--layer",type=int,default=4)
    ap.add_argument("--iterations",type=int,default=501)
    ap.add_argument("--paths",type=int,default=128)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--simplify-from")
    ap.add_argument("--ratio",type=float,default=0)
    ap.add_argument("--inference-only",action="store_true")
    a=ap.parse_args()
    if a.prepare:prepare()
    else:train(a.layer,a.iterations,a.paths,a.seed,a.simplify_from,a.ratio,a.inference_only)
