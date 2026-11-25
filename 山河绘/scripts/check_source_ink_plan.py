"""Validate the real reference through the planner without robot access."""
from pathlib import Path
import os
import sys
import json
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
STORAGE = Path(r'E:\Codex_Workspace_Data\CLIPascene_20261021')
for key, value in {'TEMP': STORAGE/'temp', 'TMP': STORAGE/'temp',
                   'PIP_CACHE_DIR': STORAGE/'cache'/'pip', 'TORCH_HOME': STORAGE/'cache'/'torch',
                   'HF_HOME': STORAGE/'cache'/'huggingface', 'XDG_CACHE_HOME': STORAGE/'cache',
                   'CUDA_CACHE_PATH': STORAGE/'cache'/'cuda'}.items():
    os.environ[key] = str(value)
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

import cv2
import unified_painting


def main():
    out = ROOT/'offline_previews'/'source_ink_v2_20261022'
    reference = ROOT/'offline_previews'/'clipascene_20261021'/'input_landscape.png'
    started = time.perf_counter()
    with patch('robot.E6Robot.__init__', side_effect=AssertionError('Hardware forbidden')):
        plan, work, lineart, waypoints, meta = unified_painting.build_plan(reference)
    assert not plan['ready_for_execution']
    strokes = plan['modules']['outline']['strokes']
    for s in strokes:
        assert len(s['points']) == len(s['press']) == len(s['speed']) == len(s['source_width_mm'])
    assert plan['modules']['outline']['image_corners'] == plan['modules']['color']['image_corners']
    report = dict(reference=str(reference), elapsed_seconds=time.perf_counter()-started,
                  work_size=[work.shape[1], work.shape[0]],
                  mm_per_px=plan['mm_per_px'], outline_strokes=len(strokes),
                  color_strokes=len(plan['modules']['color']['strokes']),
                  waypoint_count_in_memory=len(waypoints),
                  ready_for_execution=plan['ready_for_execution'],
                  robot_instantiation_forbidden=True,
                  robot_task_saved=False, quality_stats=meta['quality_stats'])
    out.mkdir(parents=True, exist_ok=True)
    cv2.imencode('.png', lineart)[1].tofile(str(out/'planner_centerlines.png'))
    (out/'planner_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'quality_stats'}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
