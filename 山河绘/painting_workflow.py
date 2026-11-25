"""Selectable, ordered color/outline design modules in shared image coordinates."""
from pathlib import Path
import numpy as np
import color_pass
import imgio
import vision
import watercolor
import sparse_color
import outline_pass
from geometry import CanvasMap
from color_profile import ColorProfile


def parse_stages(value):
    stages = tuple(s.strip() for s in value.split(',')) if isinstance(value, str) else tuple(value)
    if not stages or len(set(stages)) != len(stages) or any(s not in ('color', 'outline') for s in stages):
        raise ValueError('Stages must be color, outline, color,outline or outline,color')
    return stages


def build_plan(reference, stages='color,outline', width=1., gap=.6,
               max_strokes=2, profile=None, existing_ink=False, work_size=1000):
    stages = parse_stages(stages)
    profile = (profile or ColorProfile()).to_dict()
    work, info = vision.prepare_image(vision.load_image(str(reference)), work_size=work_size)
    mapper = CanvasMap(info['prepared_size'], info['image_corners'])
    geometry = dict(image_size=list(info['prepared_size']), mm_per_px=mapper.mm_per_px,
                    image_corners=mapper.image_corners)
    modules = {}
    sequence = []
    for i, stage in enumerate(stages):
        if i:
            sequence.extend([dict(kind='observe_previous_stage_dry', automatic_seconds=None),
                             dict(kind='manual_wash_wipe_and_load', next_stage=stage)])
        if stage == 'color':
            respect_ink = existing_ink or 'outline' in stages[:i]
            strokes, regions = sparse_color.plan_paths(work, mapper.mm_per_px, width, gap,
                                                       max_strokes, respect_ink=respect_ink)
            modules[stage] = dict(format='sparse-color-design-v1', ready_for_execution=False,
                **geometry, strokes=strokes, regions=regions, stroke_count=len(strokes),
                max_strokes_per_region=max_strokes, respect_existing_ink=respect_ink,
                ink_gap_mm=gap, palette_bgr=watercolor.PALETTE,
                motion_candidate=profile, sequence=sparse_color.schedule(strokes),
                preview_model='Path footprints only; paper diffusion is not simulated')
        else:
            modules[stage] = outline_pass.build_plan(work, mapper.mm_per_px)
            modules[stage].update(geometry)
        sequence.append(dict(kind='run_module', module=stage))
    return dict(format='painting-workflow-design-v1', ready_for_execution=False,
                reference_image=str(Path(reference).resolve()),
                reference_sha256=color_pass._digest(reference), **geometry,
                stages=list(stages), modules=modules, sequence=sequence,
                requires=['Matching paper registration', 'Measured color brush width and light-pressure swatch',
                          'Verified color loading and drying before changing stages'],
                note='Offline design; no robot trajectory or inferred paper calibration'), work


def save_outputs(plan, work, plan_path, preview_path=None, split_stages=False):
    """Export the workflow design plus ONE preview image, refusing overwrites.

    split_stages=False（默认）只写：计划 JSON + 一张拼版预览图。
    split_stages=True 才额外按 stage 另存模块 JSON / 模块预览（便于单独比对或单独执行）。
    """
    plan_path = Path(plan_path)
    stage_files = {s: plan_path.with_name(plan_path.stem+'_'+s+'.json') for s in plan['stages']}
    previews = {} if (preview_path is None or not split_stages) else {
        s: Path(preview_path).with_name(Path(preview_path).stem+'_'+s+'.png') for s in plan['stages']}
    outputs = [plan_path, *(stage_files.values() if split_stages else []), *previews.values()]
    if preview_path is not None:
        preview_path = Path(preview_path)
        outputs.append(preview_path)
    if len({p.resolve() for p in outputs}) != len(outputs) or any(p.exists() for p in outputs):
        raise ValueError('Output paths duplicate or already exist; choose new output names')
    for p in outputs: p.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.full_like(work, 255)
    for stage in plan['stages']:
        module = plan['modules'][stage]
        if split_stages:
            color_pass.save_plan(stage_files[stage], module)
        if preview_path is None: continue
        if stage == 'color':
            colored, _ = watercolor.render_image(work, module, include_ink=False)
            if stage in previews and not imgio.imwrite(str(previews[stage]), colored):
                raise OSError('Cannot save color preview')
            # Multiply the deposited pigment over whatever earlier stages drew.
            canvas = np.rint(canvas.astype(float)*colored/255.).astype(np.uint8)
        else:
            outline = outline_pass.draw(np.full_like(work,255), module)
            if stage in previews and not imgio.imwrite(str(previews[stage]), outline):
                raise OSError('Cannot save outline preview')
            canvas = outline_pass.draw(canvas, module)
    if preview_path is not None and not imgio.imwrite(str(preview_path), canvas):
        raise OSError('Cannot save workflow preview')
    if split_stages:
        plan['module_files'] = {s: str(p.resolve()) for s,p in stage_files.items()}
    color_pass.save_plan(plan_path, plan)
    return outputs
