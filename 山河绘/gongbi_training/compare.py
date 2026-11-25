"""Same-image, same-threshold baseline comparison and training audits."""
import argparse
import hashlib
import json
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .runtime import LAB
from .data import TileSampler


def preflight(manifest_path):
    old=json.loads((LAB/'datasets'/'manifest.json').read_text(encoding='utf-8'))
    new=json.loads(manifest_path.read_text(encoding='utf-8'))
    assert not(set(new['training_groups'])&set(new['validation_groups']))
    old_val={r['id']:r for r in old['real'] if r['split']=='validation'}
    for r in new['real']:
        if r['split']=='validation':
            assert r==old_val[r['id']],'Validation record changed'
    sampler=TileSampler('train',320,manifest_path,'v2')
    rng=np.random.default_rng(8901)
    samples=[sampler.sample(rng,.6,True) for _ in range(24)]
    for x,y,w,_ in samples:
        assert x.shape==(3,320,320) and y.shape==w.shape==(2,320,320)
        assert torch.isfinite(x).all() and torch.isfinite(y).all() and torch.isfinite(w).all()
        assert 0<=x.min() and x.max()<=1 and 0<=y.min() and y.max()<=1 and 0<=w.min() and w.max()<=1
    gallery=Image.new('RGB',(1600,800),(244,242,235));draw=ImageDraw.Draw(gallery)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    for i,(x,y,_,kind) in enumerate(samples[:4]):
        draw.text((i*400+10,10),kind+' / 增强配对',font=font,fill=(30,45,35))
        gallery.paste(Image.fromarray(np.rint(x.numpy().transpose(1,2,0)*255).astype(np.uint8)),(i*400+20,60))
        gallery.paste(Image.fromarray(np.rint((1-y[0].numpy())*255).astype(np.uint8)),(i*400+20,410))
    gallery.save(LAB/'previews'/'v02_augmented_training_pairs.png')
    print('PREFLIGHT_OK: 24 finite aligned samples; validation records unchanged; painting split disjoint',flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',default='mobilenet_unet_v01');parser.add_argument('--run',default='mobilenet_unet_v02');parser.add_argument('--preflight',type=__import__('pathlib').Path);args=parser.parse_args(argv)
    if args.preflight:return preflight(args.preflight)
    baseline=LAB/'runs'/args.baseline;current=LAB/'runs'/args.run
    first={r['id']:r for r in json.loads((baseline/'inference'/'summary.json').read_text(encoding='utf-8'))}
    second=json.loads((current/'inference'/'summary.json').read_text(encoding='utf-8'))
    folder=current/'baseline_comparison';folder.mkdir(exist_ok=True)
    snapshot=json.loads((current/'dataset_manifest_snapshot.json').read_text(encoding='utf-8'))
    records={r['id']:r for r in snapshot['real']}
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    rows=[]
    for r in second:
        previous=first[r['id']];assert (r['low'],r['high'],r['min_length_pixels'])==(previous['low'],previous['high'],previous['min_length_pixels'])
        rgb=Image.open(records[r['id']]['image']).convert('RGB')
        a=Image.open(baseline/'inference'/r['id']/'structure.png');b=Image.open(current/'inference'/r['id']/'structure.png')
        assert a.size==b.size==rgb.size
        role='参与训练的参考图' if r['split']=='train' else '独立留出验证图（已用于调参，非最终测试）'
        canvas=Image.new('RGB',(1800,900),(246,245,239));draw=ImageDraw.Draw(canvas)
        draw.text((15,12),r['id']+' | '+role,font=font,fill=(30,45,35))
        for i,(title,im) in enumerate([('原图',rgb),('第一版：1,200 步',a.copy()),('第二版：续训结果',b.copy())]):
            draw.text((i*600+15,53),title,font=font,fill=(30,45,35));im.thumbnail((570,770))
            canvas.paste(im,(i*600+15+(570-im.width)//2,95+(770-im.height)//2))
        canvas.save(folder/(r['id']+'.png'))
        p1=np.asarray(a.convert('L'))<180;p2=np.asarray(b.convert('L'))<180
        old_prob=np.asarray(Image.open(baseline/'inference'/r['id']/'structure_probability.png')).astype(float)/255.
        new_prob=np.asarray(Image.open(current/'inference'/r['id']/'structure_probability.png')).astype(float)/255.
        rows.append(dict(id=r['id'],split=r['split'],v1_path_count=previous['path_count'],v2_path_count=r['path_count'],
                         v1_ink_pixels=int(p1.sum()),v2_ink_pixels=int(p2.sum()),
                         probability_mean_absolute_change=float(np.abs(old_prob-new_prob).mean()),
                         actual_human_accuracy_available=False))
    for split,name in [('train','reference_comparison'),('validation','heldout_comparison')]:
        chosen=[r for r in second if r['split']==split]
        gallery=Image.new('RGB',(1500,90+len(chosen)*750),(246,245,239));draw=ImageDraw.Draw(gallery)
        draw.text((15,15),'原图 / 第一版 / 第二版；相同阈值；训练图与留出图分别展示',font=font,fill=(30,45,35))
        for i,r in enumerate(chosen):
            im=Image.open(folder/(r['id']+'.png'));im.thumbnail((1500,750));gallery.paste(im,(0,80+i*750))
        gallery.save(current/(name+'.jpg'),quality=93)
    config=json.loads((current/'config.json').read_text(encoding='utf-8'))
    source_matches={name:hashlib.sha256((current/'source_snapshot'/name).read_bytes()).hexdigest()==sha for name,sha in config['source_sha256'].items()}
    assert all(source_matches.values())
    finite_checkpoints={}
    scaler_states={}
    for path in current.glob('*.pt'):
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        finite_checkpoints[path.name]=all(bool(torch.isfinite(t).all()) for t in checkpoint['model'].values())
        scaler_states[path.name]=checkpoint['scaler']
    assert all(finite_checkpoints.values()),'Nonfinite model parameters or running buffers'
    log_rows=[json.loads(line) for line in (current/'training.jsonl').read_text(encoding='utf-8').splitlines()]
    nonfinite_gradient_logs=[r['step'] for r in log_rows if 'gradient_norm' in r and not np.isfinite(r['gradient_norm'])]
    assert all(np.isfinite(r['loss']) for r in log_rows if 'loss' in r)
    audit=dict(same_postprocessing=True,all_output_shapes_match=True,validation_records_unchanged=True,
               parameter_initialization_sha256=config.get('initialization_sha256'),training_code_snapshot_matches=True,
               no_new_expert_labels=True,model_ready_for_robot=False,baseline=args.baseline,run=args.run,images=rows,
               all_checkpoint_model_tensors_finite=finite_checkpoints,amp_scaler_states=scaler_states,
               logged_nonfinite_gradient_iterations=nonfinite_gradient_logs,
               successful_optimizer_updates_exact_count_available=False,
               amp_note='FP16 GradScaler skips updates with nonfinite gradients; step counts denote iterations, not guaranteed successful updates. Logs sample every 25 iterations; nonfinite log count is a lower bound.')
    (current/'comparison_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=True,indent=2),flush=True)
