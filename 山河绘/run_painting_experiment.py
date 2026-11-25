"""VS Code entry: real sparse color first, optional contours AFTER coloring."""
from datetime import datetime
from pathlib import Path
import os
from color_profile import ColorProfile
from painting_workflow import build_plan, save_outputs
from run_color_experiment import run, module_lines
from park_color_loading import target_pose
from config import ArmConfig, CanvasConfig
from project_paths import PROJECT_DIR, REFERENCE_IMAGE, OUTPUT_DIR, ROBOT_LOG_DIR


def calibration_snapshot():
    return repr((ArmConfig.PAPER_Z,ArmConfig.RPY,ArmConfig.USER_INDEX,ArmConfig.TOOL_INDEX,
                 CanvasConfig.ROBOT_CORNERS,ArmConfig.Z_INCREASES_DOWN,target_pose()))


def main():
    os.chdir(PROJECT_DIR)
    print('真机实验：先上色，上色全部完成后再选择是否勾线。')
    try:
        source=input('参考图路径 [%s]: ' % REFERENCE_IMAGE).strip().strip('"') or str(REFERENCE_IMAGE)
        profile=ColorProfile()
        calibration=calibration_snapshot()
        plan,work=build_plan(source,'color,outline',profile=profile)
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output=OUTPUT_DIR/('real_experiment_'+stamp+'.json')
        save_outputs(plan,work,output,output.with_suffix('.png'))
        print('本次构图预览:',output.with_suffix('.png').resolve())
        print('纸外起始/换色位置:',target_pose())
        print('当前 User/Tool:',ArmConfig.USER_INDEX,ArmConfig.TOOL_INDEX,'纸面 Z:',ArmConfig.PAPER_Z)
        print('下压 0.22 mm；行笔 v=15；全局 50%；只对空白纸做本次新实验。')
        print('请确认新纸与已标定四角对齐、纸面高度有效、笔未重新安装，或已经重新标定。')
        print('预览方向与纸面一致；高位路径清空；在场看护；机械臂已在纸外高位。')
        if input('满足上述条件输入 READY，其他输入取消: ').strip()!='READY': return 0
        color_module=plan['modules']['color']
        for color,label in [('blue','淡蓝'),('yellow','淡黄'),('red','红色')]:
            lines,skipped=module_lines(color_module,color,profile.to_dict())
            print(label,'本次实际笔数:',len(lines),'边缘留白跳过:',skipped)
            if not lines: continue
            print('请在纸外洗净、擦笔并补少量'+label+'，确认不滴液，移开手及接色物。')
            if input('输入 RUN 连续完成该颜色全部笔触；其他输入结束: ').strip()!='RUN': return 0
            if calibration_snapshot()!=calibration: raise ValueError('Calibration changed; rebuild experiment')
            if run(profile=profile,confirm_site=True,module=color_module,color=color,log_dir=ROBOT_LOG_DIR)!=0: return 4
        print('上色已完成，机械臂已回纸外高位。')
        if input('现在是否继续勾线？输入 y 继续，其他输入结束: ').strip().lower()!='y': return 0
        print('等颜色干燥后洗净笔、蘸墨并擦去余液，保持纸/笔安装及标定不变，清空路径。')
        lines,skipped=module_lines(plan['modules']['outline'],'ink',profile.to_dict())
        print('勾线:',len(lines),'笔；边缘留白跳过:',skipped)
        if input('已干燥且已备墨，输入 RUN 开始勾线: ').strip()!='RUN': return 0
        if calibration_snapshot()!=calibration: raise ValueError('Calibration changed; rebuild experiment')
        return run(profile=profile,confirm_site=True,module=plan['modules']['outline'],color='ink',log_dir=ROBOT_LOG_DIR)
    except (ValueError,OSError,EOFError,KeyboardInterrupt) as exc:
        print('停止:',exc); return 2


if __name__=='__main__': raise SystemExit(main())
