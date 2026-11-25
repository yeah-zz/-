"""Actual CUDA training, checkpoints, separate synthetic/weak validation metrics."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import random
import time
import shutil

import cv2
import numpy as np
import torch

from .runtime import LAB,ROOT
from .model import GongbiLineNet,segmentation_loss,ink_restoration_loss
from .data import TileSampler


@torch.no_grad()
def validate(model,sampler,device,nsynthetic=24,max_real_patches=None):
    model.eval()
    rng=np.random.default_rng(80808)
    rows={}
    for kind in ('synthetic','real_weak'):
        tp=fp=fn=0.
        losses=[]
        n=nsynthetic if kind=='synthetic' else max(3,len(sampler.real)*3)
        if kind=='real_weak' and max_real_patches is not None:n=min(n,max_real_patches)
        for index in range(n):
            forced=sampler.real[index%len(sampler.real)] if kind=='real_weak' else None
            if kind=='real_weak' and max_real_patches is not None:
                group=sampler.group_names[index%len(sampler.group_names)]
                pool=sampler.groups[group];forced=pool[(index//len(sampler.group_names))%len(pool)]
            x,y,w,_=sampler.sample(rng,real_probability=0.,augment=False,forced_record=forced)
            x=x[None].to(device);y=y[None].to(device);w=w[None].to(device)
            logits=model(x)
            loss,_=segmentation_loss(logits,y,w);losses.append(float(loss))
            pred=logits[0,0].sigmoid()>.5
            truth=y[0,0]>.3
            valid=w[0,0]>0
            tp+=float((pred&truth&valid).sum());fp+=float((pred&~truth&valid).sum());fn+=float((~pred&truth&valid).sum())
        precision=tp/max(tp+fp,1.);recall=tp/max(tp+fn,1.)
        rows[kind]=dict(loss=float(np.mean(losses)),precision=precision,recall=recall,
                        f1=2*precision*recall/max(precision+recall,1e-8),patches=n,
                        interpretation='procedural geometry only' if kind=='synthetic' else 'agreement with automatic weak labels; NOT actual painting accuracy')
        if sampler.manifest.get('task')=='rock_cunfa_ink_restoration':
            rows[kind]['interpretation']='Agreement with automatically normalized ink masks, NOT human accuracy; '+('paired degradation of heldout source drawings' if kind=='synthetic' else 'original heldout drawings')
    model.train()
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=1200)
    parser.add_argument('--batch',type=int,default=6)
    parser.add_argument('--tile',type=int,default=320)
    parser.add_argument('--name',default='mobilenet_unet_v01')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--init-from',type=Path,help='Start a new fine-tuning run from weights; reset optimizer and step')
    parser.add_argument('--manifest',type=Path,default=LAB/'datasets'/'manifest.json')
    parser.add_argument('--augmentation',choices=['v1','v2','ink'],default='v1')
    parser.add_argument('--real-probability',type=float,default=.4)
    parser.add_argument('--warmup',type=int,default=180)
    parser.add_argument('--encoder-lr',type=float,default=3e-5)
    parser.add_argument('--decoder-lr',type=float,default=3e-4)
    parser.add_argument('--precision',choices=['fp16','bf16','fp32'],default='fp16')
    parser.add_argument('--validation-real-patches',type=int)
    parser.add_argument('--no-pretrained',action='store_true')
    args=parser.parse_args(argv)
    manifest_metadata=json.loads(args.manifest.read_text(encoding='utf-8'))
    ink_task=manifest_metadata.get('task')=='rock_cunfa_ink_restoration'
    if ink_task:
        # validate() uses the same module-level loss as the optimization loop.
        global segmentation_loss
        segmentation_loss=ink_restoration_loss
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; refusing accidental slow CPU training')
    torch.set_num_threads(4);cv2.setNumThreads(2)
    torch.manual_seed(20261022);np.random.seed(20261022);random.seed(20261022)
    torch.backends.cudnn.benchmark=True
    device=torch.device('cuda')
    if args.steps<=0 or args.batch<=0 or args.tile<64:raise ValueError('Invalid training dimensions/steps')
    if args.resume and args.init_from:raise ValueError('Use --resume or --init-from, not both')
    if not 0<=args.real_probability<=1 or args.warmup<0:raise ValueError('Invalid sample probability/warmup')
    run=LAB/'runs'/args.name
    if (run/'config.json').exists() and not args.resume:raise FileExistsError('Run already exists; choose a new --name to preserve checkpoints')
    run.mkdir(exist_ok=True)
    config=dict(architecture='MobileNetV3-small encoder + U-Net decoder; 2 output channels',
                task='structure-first gongbi landscape contour segmentation',
                channels=['structure','optional_details'],tile=args.tile,batch=args.batch,
                requested_steps=args.steps,seed=20261022,gpu=torch.cuda.get_device_name(0),
                torch=torch.__version__,pretrained_encoder=not args.no_pretrained,
                learning_rates=dict(encoder=args.encoder_lr,decoder=args.decoder_lr),
                real_probability_after_warmup=args.real_probability,synthetic_only_warmup_steps=args.warmup,
                augmentation=args.augmentation,data_manifest_path=str(args.manifest),
                precision=args.precision,validation_real_patch_limit=args.validation_real_patches,
                data_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                initialization_checkpoint=str(args.init_from) if args.init_from else None,
                initialization_sha256=hashlib.sha256(args.init_from.read_bytes()).hexdigest() if args.init_from else None,
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'gongbi_training').glob('*.py')},
                real_gold_accuracy_available=False,ready_for_robot=False)
    if ink_task:
        config.update(task='Isolated rocks: restore ink coverage AND continuous density, including cunfa',
                      channels=['all_ink_coverage_including_cunfa','continuous_ink_density'],
                      target_provenance='Paper-normalized automatic weak masks; paired degradations, no expert dense annotations')
    (run/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    snap=run/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in (ROOT/'gongbi_training').glob('*.py'):shutil.copy2(p,snap/p.name)
    shutil.copy2(args.manifest,run/'dataset_manifest_snapshot.json')
    model=GongbiLineNet(pretrained=not args.no_pretrained and not args.init_from and not args.resume).to(device)
    parent_step=0
    if args.init_from:
        parent=torch.load(args.init_from,map_location=device,weights_only=False)
        model.load_state_dict(parent['model']);parent_step=parent['step']
    parameters=sum(p.numel() for p in model.parameters())
    encoder_ids={id(p) for p in model.encoder.parameters()}
    opt=torch.optim.AdamW([dict(params=list(model.encoder.parameters()),lr=args.encoder_lr),
                           dict(params=[p for p in model.parameters() if id(p) not in encoder_ids],lr=args.decoder_lr)],weight_decay=1e-4)
    if args.precision=='bf16' and not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 unsupported on this device')
    scaler=torch.amp.GradScaler('cuda',enabled=args.precision=='fp16')
    initial_head=model.head.weight.detach().clone()
    start_step=0;best=float('inf')
    if args.resume:
        ckpt=torch.load(args.resume,map_location=device,weights_only=False)
        model.load_state_dict(ckpt['model']);opt.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler']);start_step=ckpt['step'];best=ckpt.get('best_validation_loss',best)
    training=TileSampler('train',args.tile,args.manifest,args.augmentation);validation=TileSampler('validation',args.tile,args.manifest)
    rng=np.random.default_rng(20261022+start_step)
    logs=run/'training.jsonl'
    start=time.perf_counter();model.train()
    used=dict(real_weak=0,synthetic=0);last_validation=None
    successful_updates=0;skipped_updates=0

    def checkpoint(name,step):
        payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),
                     step=step,best_validation_loss=best,config=config,validation=last_validation)
        torch.save(payload,run/name)

    print('TRAINING_START',parameters,'parameters',args.steps,'steps',flush=True)
    for step in range(start_step,args.steps):
        probability=0. if step<args.warmup else args.real_probability
        (x,y,w),kinds=training.batch(rng,args.batch,probability,True)
        for kind in kinds:used[kind]+=1
        x=x.to(device);y=y.to(device);w=w.to(device)
        fraction=(step+1)/max(args.steps,1)
        factor=.15+.85*.5*(1+math.cos(math.pi*fraction))
        for group,base in zip(opt.param_groups,(args.encoder_lr,args.decoder_lr)):group['lr']=base*factor
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16 if args.precision=='bf16' else torch.float16,enabled=args.precision!='fp32'):
            logits=model(x)
            loss,parts=segmentation_loss(logits,y,w)
        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at step {step}')
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        update_finite=bool(torch.isfinite(grad))
        if update_finite:
            scaler.step(opt);scaler.update();successful_updates+=1
        else:
            skipped_updates+=1
            # A nonfinite norm must never be applied to the parameters, even
            # when individual gradient values passed the scaler's inf check.
            if scaler.is_enabled():scaler.update(new_scale=max(1.,scaler.get_scale()/2))
        if (step+1)%25==0 or step==start_step:
            row=dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(grad) if update_finite else None,
                     update_applied=update_finite,successful_optimizer_updates=successful_updates,skipped_optimizer_updates=skipped_updates,
                     elapsed_seconds=time.perf_counter()-start,lr=opt.param_groups[1]['lr'],**parts)
            with logs.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if (step+1)%200==0 or step+1==args.steps:
            last_validation=validate(model,validation,device,max_real_patches=args.validation_real_patches)
            score=.75*last_validation['synthetic']['loss']+.25*last_validation['real_weak']['loss']
            if score<best:
                best=score;checkpoint('best.pt',step+1)
            checkpoint('last.pt',step+1)
            event=dict(step=step+1,validation=last_validation,best_validation_loss=best)
            with logs.open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
    delta=float((model.head.weight.detach()-initial_head).abs().mean())
    assert delta>1e-7,'No measurable weight update'
    report=dict(completed_steps=args.steps,updates_this_invocation=args.steps-start_step,parent_checkpoint_step=parent_step,parameter_count=parameters,
                successful_optimizer_updates=successful_updates,skipped_optimizer_updates=skipped_updates,
                elapsed_seconds=time.perf_counter()-start,head_mean_absolute_parameter_update=delta,
                consumed_training_samples=used,max_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                validation=last_validation,actual_human_dense_labels=0,
                checkpoint_selection='75% synthetic loss +25% heldout weak-label loss; not human-judged painting accuracy',
                status='trained_initial_prototype_requires_real_annotation_and_review',ready_for_robot=False)
    if ink_task:
        report.update(checkpoint_selection='75% heldout paired-restoration loss +25% heldout original weak-mask loss; not expert accuracy',
                      status='isolated_rock_cunfa_ink_restoration_prototype',task=config['task'])
    (run/'training_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('TRAINING_COMPLETE',json.dumps(report,ensure_ascii=True),flush=True)
