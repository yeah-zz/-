import json
import tempfile
import unittest
from pathlib import Path
from config import ArmConfig, CanvasConfig
from run_color_experiment import plan_lines
from experiment_resume import resume_suffix


class ResumeTests(unittest.TestCase):
    def test_only_confirmed_draw_endpoint_can_skip_unlifted_stroke(self):
        lines=plan_lines(100,100)
        r=dict(lines=lines,paper_z=ArmConfig.PAPER_Z,user=ArmConfig.USER_INDEX,
               tool=ArmConfig.TOOL_INDEX,canvas_corners=CanvasConfig.ROBOT_CORNERS,rpy=ArmConfig.RPY,
               contacts=[dict(id=lines[0]['id'],xy=lines[0]['xy'],completed=False)],
               steps=[dict(name=lines[0]['id']+':draw-path',targets=[p for n,p in lines[0]['sequence'] if n=='draw'])])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'record.json'
            def save(): p.write_text(json.dumps(r),encoding='utf8')
            save(); rest,audit=resume_suffix(p,lines)
            self.assertEqual(len(rest),1); self.assertEqual(audit['skipped_ids'],['trial-0'])
            before=p.read_bytes(); resume_suffix(p,lines); self.assertEqual(p.read_bytes(),before)
            r['pending_move']={'name':'draw'}; save()
            with self.assertRaises(ValueError): resume_suffix(p,lines)
            r.pop('pending_move'); r['steps']=[]; save()
            with self.assertRaises(ValueError): resume_suffix(p,lines)


if __name__=='__main__': unittest.main()
