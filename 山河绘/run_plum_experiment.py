"""Real plum painting: automatic black reload, wait for brush head, red reload."""
from pathlib import Path
from datetime import datetime
import os
import json
from config import ArmConfig
from project_paths import PROJECT_DIR, PAINTING_DIR, PLUM_REFERENCE_IMAGE, OUTPUT_DIR, ROBOT_LOG_DIR
from experiment_resume import unresolved_records, resume_suffix
from run_color_experiment import module_lines
from robot import RobotError
from color_profile import ColorProfile
from pigment_station import PigmentStation
from plum_blossom import build_plan
from painting_workflow import save_outputs
from run_color_experiment import run
from park_color_loading import target_pose


def station_from_json(key):
    """非交互：直接按 pigment_measurements.json 里保存的数值建一个 PigmentStation。

    蘸色参数（dip_mm / dwell_s / drip_s / strokes_per_load）写进 json 就生效，
    不写就用 pigment_station.py 的默认值。墨和颜料可以不一样。
    """
    measured=json.loads((PAINTING_DIR/'pigment_measurements.json').read_text(encoding='utf-8-sig'))
    if (measured['user']!=ArmConfig.USER_INDEX or measured['tool']!=ArmConfig.TOOL_INDEX
            or measured['rpy']!=list(ArmConfig.RPY)):
        raise ValueError('测量使用的坐标系或姿态已改变，请重新测量')
    saved=measured[key]
    values=[saved['x'],saved['y'],saved['surface_z'],saved['rim_z']]
    tuning={k:float(saved[k]) for k in ('dip_mm','dwell_s','drip_s') if saved.get(k) is not None}
    if saved.get('strokes_per_load') is not None:
        tuning['strokes_per_load']=int(saved['strokes_per_load'])
    return PigmentStation(*values,**tuning).validate()


def read_station(label, key):
    print(label+'：输入实测的杯中心 X Y、液面 Z、杯沿最高点 Z（当前 User/Tool、当前笔头）。')
    print('杯沿 Z 为当前笔尖刚接触杯沿高度时的工具坐标；不要输入杯的物理高度。')
    measured=json.loads((PAINTING_DIR/'pigment_measurements.json').read_text(encoding='utf-8-sig'))
    if (measured['user']!=ArmConfig.USER_INDEX or measured['tool']!=ArmConfig.TOOL_INDEX
            or measured['rpy']!=list(ArmConfig.RPY)):
        raise ValueError('测量使用的坐标系或姿态已改变，请重新测量')
    saved=measured[key]
    defaults=[saved['x'],saved['y'],saved['surface_z']]
    print('已保存本次实测 X/Y/液面 Z:',defaults,'杯沿 Z:',saved.get('rim_z'))
    raw=input('回车使用已保存四项；或只输入新杯沿 Z；或输入完整 X Y 液面Z 杯沿Z: ').strip()
    if not raw:
        if saved.get('rim_z') is None: raise ValueError('尚未测量杯沿 Z，不能留空')
        values=defaults+[saved['rim_z']]
    else:
        values=list(map(float,raw.split()))
    if len(values)==1: values=defaults+values
    if len(values)!=4: raise ValueError('需要四个实测数值')
    # 蘸色参数（每杯独立）：写进 json 就生效，不写就用 PigmentStation 默认值
    tuning={k:float(saved[k]) for k in ('dip_mm','dwell_s','drip_s') if saved.get(k) is not None}
    if saved.get('strokes_per_load') is not None:
        tuning['strokes_per_load']=int(saved['strokes_per_load'])
    if tuning:
        print('  蘸色参数（来自 json）:',tuning)
    return PigmentStation(*values,**tuning).validate()


def main():
    os.chdir(PROJECT_DIR)
    try:
        reference=PLUM_REFERENCE_IMAGE
        plan,work=build_plan(reference)
        output=OUTPUT_DIR/('plum_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
        save_outputs(plan,work,output,output.with_suffix('.png'))
        print('梅花轨迹预览:',output.with_suffix('.png'))
        print('先自动蘸黑画枝干；换笔头后自动反复蘸红画花瓣；省略黄色花蕊。')
        print('User/Tool:',ArmConfig.USER_INDEX,ArmConfig.TOOL_INDEX,'纸面 Z:',ArmConfig.PAPER_Z)
        print('起始和换笔高位:',target_pose())
        resume_record=None
        blocked=unresolved_records(ROBOT_LOG_DIR)
        if blocked:
            if len(blocked)!=1: raise ValueError('存在多个中断记录，需要逐一核查')
            resume_record=blocked[0][0]
            lines,_=module_lines(plan['modules']['outline'],'ink',ColorProfile().to_dict())
            remaining,audit=resume_suffix(resume_record,lines)
            print('中断记录:',resume_record)
            print('跳过已行笔的 %d 笔，剩余 %d 笔。'%(len(audit['skipped_ids']),len(remaining)))
            print('必须已排除急停原因，笔尖已人工抬离纸面并回起始高位；原纸/笔/垫板没有移位。')
            if input('现场符合且继续原纸，输入 RESUME；其他输入取消: ').strip()!='RESUME': return 0
        black=read_station('黑色墨盘','black')
        print('确认：原纸配准与黑笔高度有效，杯内深度足够且路径清空，现场有人看护。')
        if input('输入 BLACK 实际开始黑色阶段: ').strip()!='BLACK': return 0
        if run(confirm_site=True,module=plan['modules']['outline'],color='ink',station=black,resume_record=resume_record)!=0: return 4
        print('黑色完成，机械臂已在纸外高位停止；不会按倒计时自行继续。')
        print('现在更换笔头，等待黑墨干燥，并测量另一个红色杯的坐标。')
        print('保留现有参数；必须确认新笔尖触纸高度仍一致，否则先重标并停止本次流程。')
        if input('换笔头完成、纸面高度一致且墨已干，输入 CHANGED；其他输入结束: ').strip()!='CHANGED': return 0
        red=read_station('红色颜料杯','red')
        print('确认红杯内有足够液深、笔头不滴液、双手及工具移开运动路径。')
        if input('输入 RED 实际开始红色阶段: ').strip()!='RED': return 0
        return run(profile=ColorProfile(),confirm_site=True,module=plan['modules']['color'],color='red',station=red)
    except (ValueError,OSError,EOFError,KeyboardInterrupt,RobotError) as exc:
        print('停止:',exc); return 2


if __name__=='__main__': raise SystemExit(main())
