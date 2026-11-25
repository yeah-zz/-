# -*- coding: utf-8 -*-
"""
config.py - 机械臂绘画 现场参数中心
====================================================================

【项目做什么】
  输入一张图 -> 自动分成若干色块 -> 每个色块用宽笔触铺满 ->
  按颜色分组，换色时自动洗笔、擦笔、蘸色、试色 -> 逐点 MovL 画到纸上。
  目标是"颜色分区大致正确 + 明显的大笔触感"，不追求精细皴法。

【上机前必须完成的 4 步标定】
  1. 量纸张四角在用户坐标系下的 (x, y)，填 CanvasConfig.ROBOT_CORNERS
  2. 量水盂、擦笔布、调色盘各色格、试色区的位置，填 StationConfig
  3. 把 StationConfig.STATIONS_ARE_PLACEHOLDER 改成 False（否则程序拒绝执行）
  4. 运行 `python main.py --test-strokes` 标定各档笔宽对应的下压量

【坐标与 Z 约定】
  - 所有 (x, y, z) 都是用户坐标系（USER_INDEX）下的绝对毫米值
  - Z_INCREASES_DOWN = True：Z 增大 = 毛笔下压（现场实测结论）
  - 下压量 press 一律为正值，含义是"比纸面再往下多少毫米"
  - 需要抬笔时用 z_for_lift(高度)，内部会自动处理符号

【安全】
  本文件站位坐标全部是占位值。STATIONS_ARE_PLACEHOLDER=True 时只允许
  --check / --probe / 预览，不允许执行完整绘制。
"""
import json
from pathlib import Path


# =====================================================================
# 1. 机械臂与坐标系
# =====================================================================
class ArmConfig:
    IP = "192.168.5.1"          # 控制器 LAN1 口默认地址
    PORT = 29999                # Dashboard 端口

    # ★★ 坐标系索引必须和"你量坐标时 DobotStudio Pro 点动面板上选的那个"一致 ★★
    # 同一组数字在不同用户坐标系下指向完全不同的物理位置。
    # 本项目原来的 (20,50)...(143,173) 和 PAPER_Z=-46.8 是从旧项目
    # DobotE6Calligraphy 继承来的，那是在【用户坐标系 2】里量的。
    # 如果你在别的坐标系里标定，这里必须跟着改，否则坐标全是错的。
    USER_INDEX = 2              # ← 本次标定用的是 User 2(2026)
    TOOL_INDEX = 0              # 工具坐标系索引（0~9，毛笔的 TCP）

    # 纸面高度：笔尖刚好轻触纸面时的【绝对 Z】。
    # ⚠ 注意 PAPER_Z 不是"下压深度"！它是纸面在哪（基准面）。
    #   真正表示深度的是 press_mm：绝对 Z = PAPER_Z + press（见 z_for_press()）。
    #
    # ★ 当前值来源：2026-09-23 用 `python main.py --calibrate-z`
    #   在画布中心 (108.3, 130.2) 实测 —— 笔尖刚好触纸、笔毛还没压弯。
    #
    # 【这个值为什么老是错的 —— 两个已踩过的坑】
    #   坑1：项目自带的继承值是旧项目 DobotE6Calligraphy 的 BASE_Z=-46.8，
    #        那是别的机器/别的用户坐标系下的数字，直接搬过来会扎进台面。
    #   坑2：拿"笔尖悬空时的读数"当纸面。悬空读数比纸面【更负】（=更高），
    #        填进去机器人会以为纸面在高处，笔永远碰不到纸（不撞机，但一笔画不出）。
    #        实测踩过：-80.88 是上一张纸留下的悬空值，比真实纸面高 28mm。
    #
    # ⚠ 遗留问题：四角坐标是在 Z=-94.6444（纸面上方 41.76mm）时量的，
    #   悬空 42mm 会产生视差误差。用 `python main.py --probe` 空跑四角肉眼检查：
    #   如果笔尖明显偏出纸角，就把笔降到离纸 1~2mm（Z≈-54.9）重新量四角。
    PAPER_Z = -24.7818        # 2026-11-02 用户截图实测首次触纸；下压量独立设置
                                #   历史：-80.88(悬停误读) -> -52.88(宽笔) -> -40.08(尖锋笔)
                                #        -> -39.1238(新纸) -> -12.4464(2026-09-25 换笔后)
                                #
                                #   ⚠ 这次比上一版整体抬高了 26.68 mm，说明新笔尖比旧笔
                                #     短了约 26.7 mm（工具原点更早碰到纸）。因此连带改了
                                #     ReachConfig.Z_MAX（见那边注释），其余相对纸面的高度
                                #     （TRANSIT_LIFT_MM / APPROACH_LIFT_MM / z_for_press）
                                #     会自动跟着平移，不用动。
                                #   ⚠ 前提：黑笔和红笔两支笔头的露出长度必须一致 ——
                                #     整幅流程只用这一个 PAPER_Z。换笔头画红色之前请确认
                                #     新笔头触纸高度和这一支相同；差得多就得给红阶段单独一套
                                #     纸面高度（流程改动），或重新夹笔到一致。
                                #
                                #   ★ 它是"笔尖刚好触纸时【工具原点】的 Z"。
                                #     不同毛笔笔尖到法兰的距离不同 -> 换笔后这个值会整体平移。
                                #     换笔/换纸/挪垫板之后必须重跑 --calibrate-z。
                                #
                                #   ★ 换笔后要连带检查的：
                                #       ReachConfig.Z_MAX 必须 >= PAPER_Z + PRESS_MAX_MM
                                #       当前红瓣：-13.65 + 1.30 = -12.35；Z_MAX=-10.0
                                #     TRANSIT_LIFT_MM / APPROACH_LIFT_MM 是【相对纸面】的，
                                #     会自动跟着平移，不用改。
    Z_INCREASES_DOWN = True     # 若你的机器相反，改成 False，全项目符号自动反转
    # ↑ 这个开关是"你机器用户坐标系 Z 轴朝哪边"决定的，不是物理定律。
    #   换了坐标系要重新确认：命令一个很小的 press，看笔是往下还是往上。

    LIFT_MM = 8.0               # 行笔时的抬笔高度（离纸多少毫米）
    # 站点之间移动的安全高度。★ 必须大于"障碍物高出纸面多少"。
    # 2026-09-23 实测：
    #   纸面 -52.88 / 墨面 -65.59（比纸面【高 12.7mm】）/ 盘沿目测比墨面高 22mm+
    #   -> 盘沿大约在 -88 或更高
    # 原来设 35mm 时 transit_z = -87.88，正好贴着盘沿 -> 会撞。改成 60mm：
    #   transit_z = -52.88 - 60 = -112.88
    #     比墨面(-65.59)高 47.3mm
    #     比目测盘沿(-88) 高 24.9mm  -> 有余量
    # 可达性实测：
    #   画布四角在 -105.88 已走通（更低的 25mm 版本）
    #   墨盘位置机械臂能抬到 -135.59
    # 换墨盘/换容器后必须重新确认这个值。
    TRANSIT_LIFT_MM = 60.0
    SPEED_FACTOR = 20           # 连接时下发的全局速度倍率（1~100）
                                # ★ 现在是"首次试机"的保守值，验证通过后改回 50

    CONNECT_TIMEOUT = 5.0       # TCP 连接超时（秒）
    REPLY_TIMEOUT = 8.0         # 单条指令等待响应超时（秒）
    ENABLE_DELAY = 2.0          # 使能后等待稳定（秒）
    MOVE_DELAY = 0.01           # 逐点下发时最小间隔（秒）

    # 姿态角：每次 MovL 都会带上这三个值。
    # ★ 必须填"你标定坐标时笔所在的那个姿态"，否则笔尖会整体偏移
    #   （笔尖 = 法兰 + R·t，姿态一变偏移方向就变）。
    #   量四角时只用「坐标系点动 → 用户」的 X/Y 按钮，姿态就不会变，
    #   然后把面板上的 RX/RY/RZ 抄到这里。
    #
    # ★★ 2026-09-25 换基准（换新纸那一轮重新标定后确定）：
    #    本组就是【这一轮所有实测值量测时所处的姿态】，也是机械臂当前姿态：
    #      - 新纸四角 CanvasConfig.ROBOT_CORNERS（四角 RX/RY/RZ 完全一致）
    #      - ArmConfig.PAPER_Z = -39.1238
    #      - PALETTE[0] 墨盘 dip/surface_z、pigment_measurements.json 的 black/red
    #    所以绘画、蘸墨、停车全部按这一组下发，实测值和执行姿态零偏差。
    #    ⚠ 换姿态就等于换基准：以后只要动了 RX/RY/RZ，上面这些坐标都要重量。
    #    ⚠ RZ≈0.28° 很接近过去出过问题的那组（2026-09-23 填 (0,0,0) 时 J5=91.53°
    #      贴腕部奇异点，控制器报【报警 17 = 逆解算错误】）。本轮已实测可达，
    #      但上纸前务必先跑 `python main.py --probe`（空跑四角+中心，用本姿态）
    #      确认控制器能解算、笔尖对得上纸角。
    #    ★ 历史值（回退参考）：(-3.0351, -0.8231, 30.8036)
    RPY = (-2.2793, -0.7103, 0.2786)


# =====================================================================
# 2. 画布（纸张上真正绘制的区域）
# =====================================================================
class CanvasConfig:
    # 纸张四角 (x, y)，顺序固定：左上, 右上, 右下, 左下（用户坐标系 mm）
    # ★ 只用 (x, y)！这里不填 Z —— 纸面高度由 ArmConfig.PAPER_Z 统一管。
    #
    # 2026-09-28 用户重新实测，User 2 / Tool 0。
    # 截图测角时 Z=-17.3789，未确认为触纸高度，故不据此修改 PAPER_Z。
    # 旧四角已备份至 archive/画布更新前_*；旧任务不可直接用于新画布。
    ROBOT_CORNERS = [
        (291.5345, 205.0421),  # 左上
        (-4.8478, 205.0421),   # 右上
        (-4.8478, -1.7736),    # 右下
        (292.3996, -1.7736),   # 左下
    ]

    # 输入照片里纸张只占一部分时，可填 4 个像素坐标圈定画面区域（同一顺序）；
    # None = 用整张图，此时按 FIT_MODE 处理长宽比
    IMAGE_CORNERS = None

    # 图像与纸张长宽比不一致时：
    #   "contain" = 等比缩放 + 留白（推荐，形状不变形）
    #   "stretch" = 拉伸填满（形状会变形）
    FIT_MODE = "contain"

    PAD_COLOR_BGR = (255, 255, 255)     # 留白填充色（留白区域不会被绘制）


# =====================================================================
# 3. 可达范围保护（超出范围的轨迹直接拒绝执行）
# =====================================================================
class ReachConfig:
    X_MIN = -400.0
    X_MAX = 400.0
    Y_MIN = -400.0
    Y_MAX = 400.0
    # Z 的最小/最大值。Z_MIN 是抬得最高的位置；Z_MAX 是压得最深的位置。
    # 注意：水盂蘸洗会插到水面以下十几毫米，所以 Z_MAX 必须比纸面更深，
    #       否则 wash 动作会被判为超范围。请按实际台面/水盂深度现场确认。
    #
    # ★ Z_MIN 必须低于"站点间移动高度"：
    #     transit_z = PAPER_Z - TRANSIT_LIFT_MM = -100.08 - 60 = -160.08 ... 不对，
    #   见下面 Z_MAX 的说明 —— PAPER_Z 换笔后会变，这两个值必须跟着一起改。
    Z_MIN = -140.0
    # ★★ Z_MAX（压得最深的位置）★★
    # 它必须 ≥ 所有动作里最深的 Z，其中最深的是"行笔时的最大笔压"：
    #     Z_MAX_需要 = PAPER_Z + ShuimoConfig.PRESS_MAX_MM + 余量
    #
    # 2026-09-23 换笔历史：
    #   宽笔时代  PAPER_Z = -52.88  ->  Z_MAX = -45.0 够用
    #   换尖锋笔  PAPER_Z = -40.08  <-- 比 -45 更深！
    #            旧 Z_MAX=-45 会把【每一个绘制点】都判为越界、直接拒绝执行。
    #            所以必须跟着放宽到 -37.5（= -40.08 + 1.1 + 1.5mm 余量）。
    #
    # ★★ 换笔/换纸/量完 --calibrate-z 之后，一定要回来检查这个值！★★
    # 2026-09-25 换笔后：PAPER_Z = -12.4464（比上一版抬高 26.68 mm）
    #   -> 必须放宽到 -10.0（= -12.4464 + 0.95 + 1.5 mm 余量），
    #      否则每一个绘制点都会被判越界、直接拒绝执行。
    Z_MAX = -10.0


# =====================================================================
# 4. 站点坐标（★ 全部必须现场测量 ★）
# =====================================================================
class StationConfig:
    STATIONS_ARE_PLACEHOLDER = False    # ★ 2026-09-23 墨盘已实测 -> 放行执行
                                        #   注意：调色板已精简成只有一格墨，
                                        #   不存在"没量过的色格"，放行才安全

    # ★★ 站点接近高度（避开高杯/高碗的沿）★★
    # ArmConfig.TRANSIT_LIFT_MM 是"纸面上方多少毫米"，用来避开【低的】障碍物。
    # 但如果墨盘是个【高碗】（比如碗沿比纸面高 60~80mm），那个高度不够 ——
    # 横移过去会【把碗撞翻】。
    # 这里单独给一个更高的接近高度：0 = 用 ArmConfig.TRANSIT_LIFT_MM。
    #
    # ★ 接近是【三步式】的（看 job.dip_waypoints / main.cmd_calibrate_ink）：
    #     ① 先在原地抬到这个高度（不横移）
    #     ② 再水平移到站点上方
    #     ③ 最后垂直下降
    #   绝不能"一步斜着过去" —— 直线插补会一边横移一边下降，半路就撞上碗沿。
    #
    # 2026-09-23：用户的墨碗较高，正在实测需要多少。先设 75（= 纸面上方 75mm，
    # 比墨面高约 62mm）。换容器/挪位置后必须重新确认。
    APPROACH_LIFT_MM = 75.0

    # ★ 蘸完墨在碗上方停多久，让笔毛上多余的墨滴回碗里。
    #   不停就走的话，湿墨会在空跑时被甩出去 —— 纸上出现一圈飞溅黑点。
    #   0 = 不停。墨稀的时候这个值可以大一点（0.5~1.0s）。
    DIP_DRIP_PAUSE_S = 0.8

    # 水盂：笔尖插入水中上下搅动，洗掉旧色
    WASH = {
        "x": 260.0, "y": 0.0,
        "z_surface": -50.0,      # 水面高度（笔尖刚接触水面时的 Z）
        "dip_mm": 18.0,          # 插入水中的深度
        "stir_times": 3,         # 上下搅动次数
        "stir_dx_mm": 6.0,       # 搅动时的横向摆动
        "pause_s": 0.4,
    }

    # 擦笔布/海绵：擦掉多余水分和残色
    WIPE = {
        "x0": 300.0, "y0": -20.0,
        "x1": 300.0, "y1": 20.0,
        "z_surface": -52.0,      # 布面高度
        "press_mm": 1.0,         # 擦拭下压量
        "passes": 2,             # 往返次数
        "speed": 30,
    }

    # 调色盘：色格按网格摆放。第 1 个色格中心 + 行列间距
    PALETTE_ORIGIN = (300.0, 60.0)
    PALETTE_DX = 45.0
    PALETTE_DY = 45.0
    PALETTE_COLUMNS = 3
    PALETTE_SURFACE_Z = -52.0    # 颜料表面高度（各色格假定同高）
    DIP_MM = 2.0                 # 蘸色时插入颜料的深度
    DIP_PAUSE_S = 0.8            # 蘸色后停留，让笔毛吸色

    # 试色区：换色后先画一小条，人眼确认颜色对不对
    TEST_STRIP = {
        "enabled": True,
        "x0": 300.0, "y0": 100.0,
        "x1": 330.0, "y1": 100.0,
    }


# =====================================================================
# 5. 毛笔档位（笔宽 <-> 下压量）
# =====================================================================
# press_mm 是相对纸面的下压量（正值）。用 `python main.py --test-strokes`
# 依次画 4 条线，量出实际宽度后把数值填回来。
# 警告：press 不要超过现场标定的"最粗线"下压量（约 2.0mm），否则容易戳破纸。
BRUSH_PRESETS = {
    "fine": {"width_mm": 6.0,  "press_mm": 0.5, "speed": 30},
    "mid":  {"width_mm": 13.0, "press_mm": 1.0, "speed": 40},
    "wide": {"width_mm": 22.0, "press_mm": 1.6, "speed": 50},
    "flat": {"width_mm": 32.0, "press_mm": 2.0, "speed": 55},
}


# =====================================================================
# 5B. 黑盘坐标的唯一来源（PALETTE[0] 由它派生，不要再手写第二份）
# =====================================================================
def _black_dish_from_measurements():
    """黑盘的杯心 (x, y) 与液面 Z 只从 pigment_measurements.json 读一次。

    为什么这么改：这两项原来在 config.py 和 pigment_measurements.json 里各写一份，
    必须手工同步；漏同步就会出现"起点检查要你在 A 点、蘸墨却扎到 B 点"，
    或者梅花流程与山水/试笔两条路线的蘸深基准不一致。
    现在唯一来源是 json（实测数据），config 只是派生。
    读不到（文件缺失/字段缺失/JSON 损坏）时退回内置值并打印警告，保证项目仍能导入。
    """
    fallback = {"dip": (-48.7899, 56.8080), "surface_z": -14.0747}
    path = Path(__file__).with_name("pigment_measurements.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        black = data["black"]
        return {"dip": (float(black["x"]), float(black["y"])),
                "surface_z": float(black["surface_z"]),
                "dip_mm": float(black.get("dip_mm", .5)),
                "dip_time_s": float(black.get("dwell_s", .6))}
    except Exception as exc:
        print("[config] 警告：读不到 %s（%s），PALETTE[0] 暂用内置坐标 dip=%r surface_z=%r"
              % (path.name, exc, fallback["dip"], fallback["surface_z"]))
        return fallback


_BLACK_DISH = _black_dish_from_measurements()


# =====================================================================
# 6. 调色板：颜色 -> 笔档 / 第几层画 / 填充方向
# =====================================================================
# layer：越小越先画。0 远景底色  1 中景(水面/淡墨)  2 植被土坡/中墨  3 主体(山体)  4 暗部细节
# fill ："auto" 按色块主轴 / "h" 水平 / "v" 垂直 / "d1" 45度 / "d2" -45度 / 角度数字
#
# 每个颜色还可以按需单独覆盖下面这些：
#   "dip": (x, y)      该色格中心坐标（不填则按 PALETTE_ORIGIN 网格自动排布）
#   "surface_z": z     该色格颜料表面高度（不填则用 PALETTE_SURFACE_Z）
#   "dip_mm": mm       该色的蘸色深度（不填则用 DIP_MM）
#   "dip_time_s": s    该色的停留时间（不填则用 DIP_PAUSE_S，浓稠颜料可加长）
#
# ★ 2026-09-23：先把调色板精简成【只有一格墨】。
#   原因：StationConfig.STATIONS_ARE_PLACEHOLDER 改成 False 之后会【全盘放行】，
#   而 validate() 不逐格校验坐标 —— 只要调色板里还有没量过的色格，
#   色块路线就可能一头撞进占位坐标 (300, 60) 那一带。
#   只留一格墨，就不存在"没量过的色格"，执行才是真安全的。
#   水墨框架（--mode shuimo）本来就只用一种墨，不受影响。
#   以后要做彩色上色（Pass 2）再按上面的格式把颜色逐个加回来，
#   每加一个就必须量一次它的 dip / surface_z。
PALETTE = [
    {"name": "墨黑", "rgb": (38, 38, 38), "layer": 4, "brush": "fine",
     "fill": "auto",
     # ★ 唯一来源：pigment_measurements.json（实测数据）。config 不再手写第二份，
     #   避免出现"起点检查要你在 A 点、蘸墨却扎到 B 点"或者两条路线蘸深基准不一致。
     "dip": _BLACK_DISH["dip"],               # json → black.x / black.y
     "surface_z": _BLACK_DISH["surface_z"],   # json → black.surface_z（山水/试笔/--calibrate-ink 也读它）
                                     #   离纸边 49.9 mm（要求 >=20 mm）
                                     #   注意：X 一定是【负】的 —— 填正值墨盘会落进纸内；
                                     #   截图里 [X-] / [X+] 是按钮标签，不是数值的负号，
                                     #   一律以 check_pose（GetPose(User2/Tool0)）读数为准。
                                     #   ★ 换笔/补墨都会让它整体平移（它是"笔尖刚好触液面时
                                     #     【工具原点】的 Z"），所以换笔后纸面和墨面两个都要重标。
                                     #   ★ 别去改 StationConfig.PALETTE_SURFACE_Z（那只是全局兜底默认）。
     "dip_mm": _BLACK_DISH.get("dip_mm", .5),  # 与 JSON black.dip_mm 同一来源
     "dip_time_s": _BLACK_DISH.get("dip_time_s", .6)},  # balanced 停留由 InkPolicy 控制
]

# 下面是原来的 11 色调色板，留着做参考（要做彩色上色时逐个量坐标再加回来）
# PALETTE = [
#     {"name": "天蓝", "rgb": (92, 152, 200),  "layer": 0, "brush": "wide", "fill": "h"},
#     {"name": "湖蓝", "rgb": (58, 96, 176),   "layer": 1, "brush": "wide", "fill": "h"},
#     {"name": "淡墨", "rgb": (188, 188, 188), "layer": 1, "brush": "mid",  "fill": "auto"},
#     {"name": "中墨", "rgb": (112, 112, 112), "layer": 2, "brush": "mid",  "fill": "auto"},
#     {"name": "浅绿", "rgb": (126, 170, 94),  "layer": 2, "brush": "mid",  "fill": "auto"},
#     {"name": "深绿", "rgb": (46, 108, 64),   "layer": 2, "brush": "mid",  "fill": "auto"},
#     {"name": "赭棕", "rgb": (150, 96, 58),   "layer": 2, "brush": "mid",  "fill": "auto"},
#     {"name": "土黄", "rgb": (198, 158, 96),  "layer": 2, "brush": "mid",  "fill": "auto"},
#     {"name": "朱红", "rgb": (196, 48, 40),   "layer": 3, "brush": "wide", "fill": "v"},
#     {"name": "暗红", "rgb": (132, 32, 32),   "layer": 4, "brush": "mid",  "fill": "v"},
#     {"name": "墨黑", "rgb": (38, 38, 38),    "layer": 4, "brush": "fine", "fill": "auto"},
# ]


# =====================================================================
# 7. 图像处理参数
# =====================================================================
class VisionConfig:
    WORK_SIZE = 640             # 分割用的工作分辨率（长边像素）
    CLUSTER_SAMPLE = 40000      # KMeans 采样像素上限
    EXTRA_CLUSTERS = 1          # 除调色板外额外留几个簇给"纸张/背景"

    # 留白判定（标准 Lab：L* 0~100，彩度 = sqrt(a^2+b^2)）
    PAPER_TOLERANCE = 16.0      # 与检测到的纸色的 Lab 距离小于此值 -> 留白
    PAPER_L_MIN = 90.0          # 亮度高于此值
    PAPER_CHROMA_MAX = 12.0     # 且彩度低于此值 -> 留白（云、纸底）

    # 光照校正：只对"拍照得到的画"有用（去除灯光渐变）。
    # 警告：对纯色块插画/线稿是灾难 —— 平坦色区的大尺度模糊≈自身颜色，
    #       相除后整块会塌成均值灰，导致颜色分区全乱。默认关闭。
    NORMALIZE_LIGHTING = False
    LIGHTING_SIGMA_PX = 40.0

    MIN_REGION_AREA_MM2 = 40.0  # 小于此面积的色块整块跳过（避免满地碎笔）
    MORPH_CLOSE_MM = 1.2        # 闭运算填小洞
    MORPH_OPEN_MM = 0.8         # 开运算去毛刺

    SPACING_FACTOR = 0.62       # 扫掠行距 = 笔宽 × 系数（越小越密越慢）
    SAMPLE_STEP_MM = 1.5        # 沿笔触方向的采样步长（决定轨迹点密度）
    MIN_STROKE_LEN_MM = 8.0     # 短于此长度的笔触丢弃（自动简化细节）
    MAX_STROKE_LEN_MM = 110.0   # 单笔最长长度，超过切成多笔
    OVERSHOOT_MM = 1.2          # 笔触两端外延，模拟手绘出头
    STROKE_ANGLE_JITTER_DEG = 3.0   # 每条扫掠线的角度抖动（去掉机械感）
    STROKE_BOW_MM = 1.2         # 笔触中段轻微弯曲幅度
    FILL_DIRECTION = "auto"     # 全局默认填充方向（PALETTE 里的 fill 优先）


# =====================================================================
# 8. 绘制节奏与输出
# =====================================================================
class JobConfig:
    SKIP_PIGMENT_ACTIONS = False    # True = 不洗笔不蘸色（调试轨迹用）
    WASH_BEFORE_FIRST_COLOR = False # 第一个颜色前是否也洗笔（干净笔可跳过）
    REDIP_LENGTH_MM = 400.0         # 同色连续画这么多毫米后补蘸一次
    DOWN_SPEED = 20                 # 落笔/抬笔（纯 Z 动作）的速度百分比
    TEST_AFTER_DIP = False          # 蘸色后是否在试色区画一小条
                                    # ★ 暂时关闭：试色区坐标还没量。
                                    #   量好之后填 StationConfig.TEST_STRIP 再打开

    MOTION_BATCH_SIZE = 100         # 每多少点等一次控制器缓存（原项目实测约 260 点会满）
    CONTINUOUS_PATH_RATIO = 20      # 行笔线段的平滑过渡比例 cp（0~100）
    MOTION_TIMEOUT = 120            # 等待运动完成超时（秒）
    REPORT_EVERY = 25               # 每画多少笔打印一次进度

    JOB_JSON = "paint_job.json"
    PREVIEW_PNG = "paint_preview.png"
    LABELS_PNG = "paint_labels.png"
    PREVIEW_PX_PER_MM = 3.0         # 预览图分辨率


# =====================================================================
# 9. 水墨（--mode shuimo）—— 极细狼毫专用参数
# =====================================================================
# 【和色块路线的关系】
#   本节参数只服务 `python main.py --mode shuimo` 这一条路线，完全独立。
#   BRUSH_PRESETS / VisionConfig / PALETTE 那套（色块铺色）不受影响，
#   两条路线共用 geometry / robot / job 执行层与安全检查。
#
# 【为什么单独一套笔档】
#   色块路线用的是宽笔（最细 6mm），水墨要的是**极细狼毫**：
#     - 笔锋极细极硬 -> 能出 0.3~0.5mm 的发丝线（勾线、皴线）
#     - 储墨量远小于羊毫 -> 补蘸必须比色块路线频繁得多（400mm -> 150mm）
#     - 笔锋硬、压过头会劈叉甚至伤纸 -> 机械安全上限压到 1.1mm
#   所以不能复用 BRUSH_PRESETS，换笔后必须重新标定下表。
class ShuimoConfig:
    # ---- 笔档：press_mm 是相对纸面的下压量，width_mm 是实测笔痕宽度 ----
    # 现场用 `python main.py --brush-test` 画笔压递增的线，量宽度填回。
    # ★ 必须带着墨标定，不能干笔量：笔痕宽度和含墨量强相关。
    #
    # ★★ 注意：这张表【只影响预览图和统计数字，不影响机械臂运动】★★
    #    真正决定线条粗细的是 PRESS_MIN_MM / PRESS_MAX_MM。
    #
    # ★★ 2026-09-25 换笔后实测（5 条线：0.22/0.28/0.50/0.70/0.95 mm，
    #    照片 + 切割垫刻度标定，量每条线的中段半高宽 FWHM）：
    #      笔压 0.22 -> 0.90 mm      笔压 0.70 -> 1.30 mm
    #      笔压 0.28 -> 1.00 mm      笔压 0.95 -> 1.50 mm
    #      笔压 0.50 -> 1.20 mm      （每条线沿线波动约 ±0.1 mm）
    #    ★ 结论一：压深翻 4 倍，线宽只涨约 70%（0.9 -> 1.5 mm，已经饱和）。
    #      所以比 1.5 mm 更粗的笔画【压不出来】，只能靠多笔并排铺毫。
    #    ★ 结论二：这 5 条线的墨芯最深只比纸面暗 30%（灰度 129~140 / 纸 200），
    #      偏淡。若成画后浓淡对比不够，先把 press 上限试到 1.2~1.5 mm 再定 PRESS_MAX。
    #    ★ 结论三：1.5 mm 以下的细枝完全够用（参考图里 <=1.5mm 的细枝占 52%）。
    #    ★ 2026-09-25 追加：实画要更深，PRESS_MAX 提到 1.40，所以补一个估值档
    #      "heavy"（1.40 mm -> 约 1.9 mm），否则预览里超过 0.95 的笔压会被
    #      插值表夹住、显示得比实际细。下次量 1.2/1.4 那条线时把这个估值换掉。
    # ★★ 2026-09-26 实测量标（PAPER_Z 修正到 -11.95 之后）★★
    #    用 scripts/press_ladder.py 在同一张纸上画 6 条 40mm 等压直线，
    #    拍照后按"线长 40mm"定标量出的真实笔痕宽度：
    #        笔压 0.10 -> 0.86 mm      笔压 0.60 -> 2.04 mm
    #        笔压 0.25 -> 1.20 mm      笔压 0.85 -> 2.89 mm
    #        笔压 0.40 -> 1.54 mm      笔压 1.20 -> 3.98 mm
    #    旧表（0.22->0.90 … 1.40->1.90）是在【错的 PAPER_Z】下测的，明显偏平：
    #    1.20 那一档旧表说 1.72mm，实测 3.98mm —— 差 2.3 倍。
    #    这正是"计划以为只有 0.95~1.90mm，纸上却糊成一片"的来源。
    #    ★ 本表只用于【预览渲染】和【记录里的 width_mm】；
    #      press 本身是由下面 PRESS_MIN_MM/PRESS_MAX_MM 按参考图墨宽归一化得到的，
    #      不受本表影响 —— 所以换这张表【不会改变计划、不影响续画】。
    #    重新标定：uv run python main.py --test-strokes，或 scripts/press_ladder.py。
    BRUSHES = {
        "tip":   {"width_mm": 0.86, "press_mm": 0.10, "speed": 20},
        "fine":  {"width_mm": 1.20, "press_mm": 0.25, "speed": 25},
        "mid":   {"width_mm": 1.54, "press_mm": 0.40, "speed": 32},
        "broad": {"width_mm": 2.04, "press_mm": 0.60, "speed": 40},
        "heavy": {"width_mm": 2.89, "press_mm": 0.85, "speed": 45},
        "huge":  {"width_mm": 3.98, "press_mm": 1.20, "speed": 50},
    }
    DEFAULT_BRUSH = "fine"

    # 行笔过程中允许的笔压区间。
    # ★ 2026-09-23 实测（在纸上画 4 条压力递增的线，0.2/0.5/1.0/1.6mm）：
    #   4 条全部留下明显墨痕 -> 说明 PAPER_Z 正确、整个压程都有效。
    #   但 PRESS_MIN 不能是 0：那是给尖锋狼毫设的（笔尖是一个点，0 压即点接触）。
    #   宽笔的笔毛前端是一条平边，0 压时只有一两根毛尖碰纸，线条会断续发虚。
    #   实测把下限抬到 0.3 就能让起笔/收笔的 ramp 底部也保持接触。
    #   ★ 换回尖锋狼毫后，把 PRESS_MIN_MM 改回 0.0。
    #
    # ★★ 2026-09-23 第二次下调（现场实画后发现太深）：
    #    兰花整幅画出来是一团黑块、周围还有飞溅墨点 —— 压太深 + 墨太稀，
    #    墨在纸上洇开把相邻笔触连成一片。
    #    原值 0.3 / 1.6 -> 改成 0.15 / 0.7（整体约降一半）。
    #    墨稀的时候宁可压浅一点：线条淡了可以多蘸/多画一遍，
    #    但洇开了就救不回来了。
    # ★★ 2026-09-23 第三次调整：换笔（尖锋笔，笔尖收拢成一点）
    #    宽笔（扁头）时代的值是 0.15 / 0.70，那套对尖锋不适用：
    #      扁头 0 压只有一两根毛碰纸 -> 必须留 0.15 的底
    #      尖锋 0 压就是"笔尖点上纸" -> MIN 可以给 0，能画出真正的细线
    #    换回宽笔的话把这两个值改回 0.15 / 0.70。
    #
    # ★★ 2026-09-23 第四次调整：现场选档（画 6 条压力递增的线让用户挑）
    #    6 条线笔压 0.00 / 0.22 / 0.44 / 0.66 / 0.88 / 1.10，
    #    用户反馈【中间那两条（0.44 / 0.66）最合心意】。
    #    所以把整个区间挪到以 ~0.52 为中心：0.20 ~ 0.85。
    #      下限 0.20：留一点底，避免细线断续（尖锋 0 压理论上能点纸，但换墨/换纸后容易虚）
    #      上限 0.85：尖锋压过头会劈叉，而且太粗就没有水墨的"清透"感了
    #
    #    ★ 真实绘制时用的是"归一化笔压"：原图最细的墨线 -> PRESS_MIN，
    #      最粗的 -> PRESS_MAX。
    #      ★★ 但实测【平均笔压只落在区间的 20~25% 位置】★★
    #         （水墨原图本来就是细线多、粗线少，分布是偏的）
    #         兰花实测 均值0.25/区间[0.15,0.70] -> 位置 18%
    #         石头实测 均值0.36/区间[0.20,0.85] -> 位置 25%
    #      所以区间不能正好压在你挑中的 0.44~0.66 上，必须整体上移，
    #      让【平均值】落在那一档：
    #         0.30 + 0.22 × (0.95-0.30) = 0.44   <-- 平均值正好落在你满意档的下沿
    #
    #    ★ 觉得画出来太细 -> 两个值一起往上抬 0.1（0.40 / 1.05）
    #      觉得太粗     -> 一起往下压 0.1（0.20 / 0.85）
    #      想要粗细对比更强 -> 把区间拉大（0.15 / 1.05）
    #
    # ★★ 2026-09-25 实画反馈"太浅了、笔触再深一点"（换笔 + 蘸墨深度加大后重定）：
    #    整段上移并拉大：0.30/0.95 -> 0.45/1.40。
    #    最深 Z = PAPER_Z -12.4464 + 1.40 = -11.0464，仍在 Z_MAX -10.0 范围内 ✓。
    #    ★ 注意这是"笔压"不是"墨量"：如果线条发淡但又不断，先加蘸墨深度
    #      （pigment_station.py 的 dip_mm / dwell_s），别一味加压。
    # ★★ 2026-09-26 定案：【只用正锋】。姿态锁死在标定值（不做侧锋/卧锋），
    #    "哪里粗哪里细、哪里深哪里淡"全部靠【笔尖下压深度 press】表达。
    #    实测（换笔后量笔宽那次）：0.22->0.90mm、0.28->1.00、0.50->1.20、0.70->1.33、0.95->1.50mm
    #      - 压深主要决定【线宽】；0.95mm 之后基本饱和（再压只多约 0.1mm）
    #      - 【浓淡】主要靠蘸墨量（dip_mm / dwell_s），不是继续压深
    #    所以区间取宽一点以拉开粗细对比：0.25 ~ 1.40（线宽约 0.95 ~ 1.50 mm）
    #      下限 0.25：实测 0.22 就有清晰墨痕，取 0.25 留余量，能画细枝
    #      上限 1.40：受 Z_MAX 约束（最深 Z = PAPER_Z+1.40 = -11.05，仍在 -10.0 内）
    # ★★ 2026-09-26 实测量标后重新定案（PAPER_Z 修正到 -11.95，BRUSHES 换成实测表）★★
    #    实测压->宽：0.12->0.89  0.25->1.20  0.40->1.54  0.50->1.80  0.60->2.04 mm
    #    旧区间 0.25~1.40 在实测表下会画出 p50 2.14mm / max 3.98mm 的线，
    #    而参考图枝干墨宽只有 1.0~1.5mm —— 粗了一到两倍，主干和铺毫会糊成一坨。
    #    新区间 0.12 ~ 0.50 -> 实测线宽 0.89 ~ 1.80 mm，与参考图一致，粗细对比约 2 倍。
    #    ★ 这两个值【会改变计划】（press 是按参考图墨宽在本区间内归一化插出来的），
    #      所以改动之后旧的续画记录一律失效，必须换纸重画。
    #      觉得太细 -> 两个值一起往上抬 0.05（0.17 / 0.55）
    #      觉得太粗 -> 一起往下压 0.05（0.07 / 0.45）
    #      想要粗细对比更强 -> 拉大区间（0.10 / 0.70，实测 0.86 ~ 2.30 mm）
    PRESS_MIN_MM = 0.06
    PRESS_MAX_MM = 0.80      # 2026-09-30 0.30->0.80: 原值已顶到上限, 黑线无加深余量

    SPEED_BASE = 28         # 行笔基准速度百分比（笔压越大越慢，见 shuimo.py）

    # ★★ 点状/串珠笔触（复现参考图那种斑驳风格）★★
    # 把每条长笔触按【弧长】切成小段，每段独立落笔、抬笔。
    #   每小段的两端都会积墨（就是之前用"出锋"修掉的那个收笔墨疙瘩）
    #   -> 一串重叠的墨点连起来 = 参考图里那种"串珠/点苔"效果。
    #
    #   0.0  = 关闭（连续笔触，行笔流畅 —— 之前一直在用的行为）
    #   4.0  = 约 4mm 一段（接近参考图的颗粒感）
    #   2.0  = 更碎、更斑驳，但抬笔次数翻倍、更慢
    #
    # ★ 代价：
    #     · 绘制时间大约变成 2~3 倍（每次抬笔都要减速-停-加速）
    #     · "路径平滑"的效果会被切段削弱，行笔不再连绵
    #     · 出锋会自动关掉（短笔触加出锋的话整条都是尾巴，见 job.stroke_waypoints）
    # ★ 2026-09-23 现场结论：用户实画后反馈"一点一点的看起来很怪，要连续动作"。
    #   所以关闭点状模式，恢复连续笔触。开关保留，随时可以再开。
    DAB_LENGTH_MM = 0.0

    # ---- 抬笔与补蘸 ----
    LIFT_MM = 4.0               # 水墨抬笔比色块低：减少空跑、保持笔势连贯
    # ★ 单次蘸墨能画多长（毫米）。极细狼毫储墨量小，这个值必须现场实测调整：
    #   让它连着画不断墨，看第几笔开始变淡，把那个笔程填回来。
    #
    # 排程是**前瞻式**的（scheduler.build_framework_plan）：画每条笔触之前
    # 先看这笔要画多长，不够就**先蘸饱再落笔**。所以补蘸永远发生在笔触之间，
    # 一笔之内绝不会因为补蘸而抬笔断线。
    # 程序会自动把试墨区那一小条的长度从容量里扣掉。
    # ★ 2026-09-23：蘸墨时间从 1.2 秒加长到 3.0 秒（用户要求"蘸久一点"），
    #   笔腹吸得更饱 -> 一笔能画更长。容器量也要跟着放大，
    #   否则"吸得饱"被浪费掉：还是每 150mm 就被叫回来重蘸一次。
    #   150 -> 240（+60%，和蘸墨时间 1.2->3.0 大致匹配）。
    #   ★ 判断调得对不对：盯着看每一段的后半截会不会变淡。
    #     变淡 -> 调小（比如 180）；一直很黑 -> 还能再调大（省时间）。
    REDIP_LENGTH_MM = 240.0
    TEST_AFTER_DIP = False      # 蘸墨后画一小条，肉眼确认墨量/线宽
                                # ★ 暂时关闭：试墨区坐标还没量

    # ---- 墨线提取参数（移植自 DobotE6Calligraphy/landscape_painting.py）----
    BLUR_KERNEL = 3             # 高斯模糊核（去 JPEG 噪声）
    WHITE_BG_RATIO = 0.70       # 白底占比达到此值才启用"浅墨滞后连接"
    LIGHT_INK_GRAY = 238        # 浅墨灰度上限
    LIGHT_INK_CLOSE_PX = 3      # 浅墨闭运算核
    MIN_COMPONENT_PX = 2        # 小于此像素数的连通域当噪点丢弃
    SUPPORT_DILATE_PX = 5       # 深墨种子膨胀核（滞后连接的支撑域）

    # 缺口桥接的启用条件。旧引擎只看"白底占比 >= 70%"，但**宣纸画的纸底
    # 是米黄不是纯白**，白底占比接近 0，于是满纸的短碎笔永远接不起来。
    # 所以再加一条与纸色无关的判据：墨迹只占画面很小一部分 -> 说明这是
    # 线稿而不是"满幅灰调山水"，桥接是安全的。
    INK_SPARSE_RATIO = 0.25     # 墨迹像素占比 <= 此值也启用缺口桥接

    COLLINEAR_GAP_PX = 2.5      # 交叉点续接的最大间距
    COLLINEAR_COS = 0.5         # 切线夹角余弦下限（约 60°）

    DIRECTIONAL_GAP_MM = 4.0    # 方向约束缺口续接的最大间距
    DIRECTIONAL_ANGLE_DEG = 38.0
    DIRECTIONAL_MIN_LENGTH_MM = 1.5
    BRIDGE_INK_RATIO = 0.18     # 缺口中央至少要有这么多比例的浅墨证据
    BRIDGE_STEP_PX = 3.0        # 补桥点的间距

    DENSE_RADIUS_MM = 3.0       # 骨架局部密度统计半径
    DENSE_PRESS_REDUCTION = 0.55    # 密集区减压比例（飞白/枯笔的来源）
    DENSE_SPEED_BOOST = 0.45        # 密集区加速比例

    # ---- 笔法（一笔之内的形态）----
    RAMP = 0.12                 # 起笔/收笔各占笔长的比例
    LINE_UNDULATION = 0.08      # 行笔时笔压的正弦微起伏（mm）

    # ---- 路径平滑：消除"直来直去"----
    # 【问题】骨架是像素台阶，简化后变成"一段段直线 + 突然拐弯"。
    #   实测（兰花 23 条笔触 / 380 个顶点）：
    #     转折角中位 10.09°，一半以上顶点 >10°，26% >20°，最大 81°
    #     转角序列长这样： 0 0 0 0 47 36 0 0 0 0 21 22 0 0 0 ...
    #   机械臂照着走就是硬折线 —— 这是"没有水墨笔触感、直来直去"的主因，
    #   和用什么笔无关。
    #
    # 【修法】对重采样后的点做滑动平均，把台阶抹成曲线（首尾端点保持不动）。
    #   PATH_SMOOTH_MM 是窗口长度：太小去不掉台阶，太大笔触会走形。
    #   6mm 约等于 4 个采样点，够抹平像素台阶又不吃掉笔势。
    PATH_SMOOTH_MM = 6.0        # 0 = 关闭

    # 手抖扰动：仿人的自然抖动（旧项目用的是 ±0.05mm）。
    # 对 6mm 宽笔几乎看不出来，但对 0.4mm 的狼毫线就是 20% 的宽度，能去掉"机械感"。
    HAND_JITTER_MM = 0.08       # 0 = 关闭

    # 行笔时的平滑过渡比例 cp（0~100）。太小则每个轨迹点都"停一下再走"，
    # 走出来是一段段直线；水墨要的是流畅的行笔，所以给大一些。
    # 代价：cp 越大越会"抄近路"（切掉拐角），所以必须先把路径本身平滑好。
    CONTINUOUS_PATH_RATIO = 70

    # ---- 收笔出锋（消除收笔处的墨疙瘩）----
    # 【问题】笔触结束时原本是"原地垂直抬笔"：笔尖停在收笔点上，
    #   Z 从"压着"升到"抬起"的过程中笔仍接触纸面、但 XY 不动
    #   -> 墨全堆在收笔点，形成一个明显的墨疙瘩。
    #   （起笔那头是同一个道理，但"起笔顿一下"在水墨里是正常的笔法，所以先不动它。）
    #
    # 【修法】让笔在抬起来之前，沿笔势方向再走一小段、同时把笔压收到最轻
    #   —— 这就是"出锋"。抬笔动作发生在锋的末端，墨就摊在锋上而不是堆成一点。
    #
    #   对宽笔效果最明显（宽笔一出墨就是一大坨）；
    #   对尖锋狼毫也能让收笔更利落（避免收笔处发胖）。
    #   设 TAIL_MM = 0 可关闭，恢复原来的原地抬笔。
    TAIL_MM = 4.0               # 出锋长度（mm）
    TAIL_POINTS = 4             # 出锋段的轨迹点数（越多越平滑，点也越多）
    TAIL_PRESS_RATIO = 0.0      # 出锋末端的笔压 = 收笔笔压 × 此比例（0 = 收到最轻）
    TAIL_SPEED_BOOST = 1.25     # 出锋时速度略快，像真的收锋
    WIDTH_PCT_LOW = 10.0        # 笔宽归一化的低分位
    WIDTH_PCT_HIGH = 90.0       # 笔宽归一化的高分位
    DENSITY_PCT_LOW = 30.0      # 密度归一化的低分位
    DENSITY_PCT_HIGH = 90.0     # 密度归一化的高分位

    # ---- 轨迹点密度 ----
    PATH_EPSILON_PX = 0.65      # 骨架路径简化容差（越小越贴合原线，点越多）
    TRAJ_STEP_MM = 1.5          # 沿笔触按弧长重采样的步长
    MAX_POINTS_PER_STROKE = 80  # 单笔最多轨迹点（防止控制器缓存被一笔打满）
    MIN_STROKE_MM = 2.0         # 短于此长度的笔触丢弃

    # 笔宽/密度曲线的平滑长度（毫米）。
    # distanceTransform 在单像素骨架上只有 1.0 / 1.414 两档取值，逐点直接
    # 用它算笔压会让笔痕边缘一跳一跳的（锯齿）。毛笔的接触宽度在两个相邻
    # 轨迹点之间不可能突变，所以沿弧长做一次滑动平均。
    PROFILE_SMOOTH_MM = 4.0

    # ---- 顺序：模拟人的作画次序 ----
    LAYER_LONG_RATIO = 0.35     # 长度 ≥ 最长笔的此比例 -> 层0 主体长轮廓
    LAYER_MID_RATIO = 0.12      # 长度 ≥ 此比例 -> 层1 中等结构线；其余 -> 层2 短皴线

    # ---- 输出（只写文件名；main.py 会拼到 project_paths.OUTPUT_DIR = 输出/ 下面）----
    PREVIEW_PNG = "shuimo_preview.png"
    JOB_JSON = "shuimo_job.json"


# =====================================================================
# 9B. 黑笔轮廓：沿枝干两侧描边（双钩）
# =====================================================================
class OutlineConfig:
    """把"单线骨架"升级成"沿枝干两侧描边"，用来表现主干粗细。

    背景：尖锋笔单笔最宽只有 ~1.5mm（实测），参考图里主干 p90 = 8.2mm、最粗 9.9mm，
    只走一条中线永远画不粗、也画不出"两侧的边"。

    做法（只动黑笔模块，不动红笔）：
      1. 仍从骨架取中线和逐点墨宽（单位 mm，来自参考图的 distanceTransform）；
      2. 局部宽度 >= EDGE_MIN_WIDTH_MM 的笔触，沿【法向】偏移 ±(半宽 - EDGE_INSET_MM)
         生成左右两条边线 —— 这样笔痕的外缘正好落在参考图的墨迹边缘上；
      3. 偏移沿弧长平滑并夹在 EDGE_MAX_OFFSET_MM 以内（圆头/急弯处不会甩出去）；
      4. 只有"够宽"的那一段才生成边线，两端收细的地方由中线一笔带过（不会三笔叠一起积墨）；
      5. 默认保留中线（INCLUDE_CENTERLINE=True）：边线 + 中线 = 有骨有肉的观感；
         想要纯"白描双钩"（中间留白）就把它改成 False。
    """
    EDGE_LINES = True            # 总开关：True=描边（双钩），False=回到原来的单线骨架
    INCLUDE_CENTERLINE = True    # 描边时是否仍画中线那一笔
    EDGE_MIN_WIDTH_MM = 2.2      # 参考图局部宽度 >= 此值才分左右两条边（细枝保持单线）
    # ★ 2026-09-26 新增一级：主干（很粗的那些）在双钩之外**再铺内芯线**，
    #   这样"下面枝条粗、上面枝条细"的对比才出得来 —— 笔尖单笔最宽只有 1.5mm，
    #   双钩中间是空的，看着仍然细。宽度 >= FILL_MIN_WIDTH_MM 的才铺。
    FILL_MIN_WIDTH_MM = 5.0      # 参考图局部宽度 >= 此值才铺内芯线（只有主干/下部粗枝够）
    FILL_LINE_SPACING_MM = 1.6   # 内芯线间距（约等于实测笔痕宽 + 一点余量，能连成片）
    FILL_MAX_PER_SIDE = 2        # 每侧最多铺几条内芯线（最多 中线+4 条）
    FILL_PRESS_SCALE = 0.9       # 内芯线笔压相对中线的比例
    EDGE_MIN_LEN_MM = 8.0        # 太短的笔触不分边
    EDGE_MIN_OFFSET_MM = 0.4     # 单侧偏移小于此值的段落不画边（两端收细段交给中线）
    EDGE_MAX_OFFSET_MM = 4.0     # 单侧最大偏移，防止急弯处甩出去
    EDGE_INSET_MM = 0.6          # 从"半宽"里内缩多少（≈笔痕半宽），让笔痕外缘落在原图边缘
    EDGE_PRESS_SCALE = 0.85      # 边线笔压相对中线的比例（边线略轻，避免比原图还重）
    # ---- 实验开关（默认关闭；用来对比"机器味太重"能不能缓解）----
    # 等距、左右完全对称的偏移看起来"很机械"；这两项给偏移加一点自然感。
    EDGE_JITTER_MM = 0.0         # 沿笔迹给偏移加多大随机抖动（mm）；0 = 关闭
    EDGE_ASYMMETRY = 0.0         # 左右边线偏移差多少比例（0.25 = 一边 75%、一边 125%）；0 = 关闭
    # ---- 宽墨块铺毫（2026-09-26 新增）----
    # 问题：那片最宽 9.9mm 的主干墨块，骨架化后被切成 45 条交叉碎线，看着像乱麻。
    # 做法：找出"局部宽度 >= MASS_MIN_WIDTH_MM"的墨块 -> 每块只留最长的那条骨架当"脊"
    #       -> 沿脊按笔宽铺满到边缘（铺毫）-> 其余碎线从线通道剔除 -> 再随机留飞白。
    MASS_FILL = True             # 总开关
    MASS_MIN_WIDTH_MM = 4.0      # 局部宽度 >= 此值算"墨块"
    MASS_DROP_RATIO = 0.6        # 一条笔触有这么大比例的点落在墨块里 -> 当碎线处理
    MASS_KEEP_LONGEST = 1        # 每个墨块保留最长的几条骨架当脊
    MASS_FILL_SPACING_MM = 1.5   # 铺毫行距（约等于实测笔痕宽）
    MASS_FILL_MAX_PER_SIDE = 4   # 脊两侧各最多铺几条（4 -> 最宽可覆盖 ~13mm）
    MASS_FILL_PRESS_SCALE = 0.85 # 铺毫线的笔压
    MASS_FLY_WHITE = 0.25        # 飞白：随机裁掉这么大比例的铺毫线段
    EDGE_MAX_TURN_DEG = 30.0     # 骨架转弯超过这个角度的段不描边
                                 # （整片浓墨细化后骨架会分叉成乱麻，那里画双线只会更乱）
    EDGE_MAX_WIDTH_RATIO = 5.0   # 沿笔触的 宽度p90/宽度p10 上限
                                 # 干净枝条实测 1.9~4.8；浓墨块细化出的乱麻 >=5.4 甚至无穷大
                                 # （骨架跑到墨外 -> p10≈0），那种段落描边只会画出一团乱线


# =====================================================================
# 10. 辅助函数
# =====================================================================
def brush_of(entry):
    """取某颜色使用的笔档参数（缺失回退 mid）"""
    return BRUSH_PRESETS.get(entry.get("brush", "mid"), BRUSH_PRESETS["mid"])


def dip_position(index):
    """第 index 个颜色（按 PALETTE 顺序）的色格坐标；写了 dip 则优先"""
    entry = PALETTE[index]
    if entry.get("dip") is not None:
        return (float(entry["dip"][0]), float(entry["dip"][1]))
    columns = max(1, int(StationConfig.PALETTE_COLUMNS))
    row = index // columns
    col = index % columns
    x = StationConfig.PALETTE_ORIGIN[0] + col * StationConfig.PALETTE_DX
    y = StationConfig.PALETTE_ORIGIN[1] + row * StationConfig.PALETTE_DY
    return (float(x), float(y))


def palette_surface_z(index):
    """该色格颜料表面高度（每色可单独覆盖）"""
    entry = PALETTE[index]
    if entry.get("surface_z") is not None:
        return float(entry["surface_z"])
    return float(StationConfig.PALETTE_SURFACE_Z)


def dip_depth_of(index):
    """该色的蘸色深度（每色可单独覆盖；浓稠颜料可深一点）"""
    entry = PALETTE[index]
    if entry.get("dip_mm") is not None:
        return float(entry["dip_mm"])
    return float(StationConfig.DIP_MM)


def dip_time_of(index):
    """该色的蘸色停留时间（每色可单独覆盖）"""
    entry = PALETTE[index]
    if entry.get("dip_time_s") is not None:
        return float(entry["dip_time_s"])
    return float(StationConfig.DIP_PAUSE_S)


def z_for_press(press_mm):
    """下压量（正值）-> 绝对 Z"""
    if ArmConfig.Z_INCREASES_DOWN:
        return ArmConfig.PAPER_Z + float(press_mm)
    return ArmConfig.PAPER_Z - float(press_mm)


def z_for_lift(height_mm):
    """离纸高度（正值）-> 绝对 Z"""
    return z_for_press(-float(height_mm))


def transit_z():
    """站点间移动安全高度对应的绝对 Z"""
    return z_for_lift(ArmConfig.TRANSIT_LIFT_MM)


def station_approach_z():
    """
    站点（墨盘/水盂/擦笔布）的接近高度。

    比 transit_z() 更高，用来避开【高的】容器沿（高碗、高杯）。
    设 StationConfig.APPROACH_LIFT_MM > 0 时用它，否则退回 transit_z()。

    ★ 必须配合"三步式接近"使用：先原地抬高 -> 再横移 -> 最后下降。
      直接一步斜着飞过去会撞翻容器。
    """
    lift = float(getattr(StationConfig, "APPROACH_LIFT_MM", 0.0) or 0.0)
    if lift > 0.0:
        return z_for_lift(lift)
    return transit_z()


def station_dip_z(surface_z, dip_mm):
    """站点插入深度 -> 绝对 Z"""
    if ArmConfig.Z_INCREASES_DOWN:
        return float(surface_z) + float(dip_mm)
    return float(surface_z) - float(dip_mm)


def validate(need_stations=True):
    """配置自检，返回问题列表（空 = 通过）"""
    problems = []
    if len(CanvasConfig.ROBOT_CORNERS) != 4:
        problems.append("CanvasConfig.ROBOT_CORNERS 必须是 4 个点（左上/右上/右下/左下）")
    if CanvasConfig.IMAGE_CORNERS is not None and len(CanvasConfig.IMAGE_CORNERS) != 4:
        problems.append("CanvasConfig.IMAGE_CORNERS 必须是 4 个点或 None")
    if len(PALETTE) < 1:
        problems.append("PALETTE 至少要有一个颜色")
    worst = BRUSH_PRESETS["flat"]["press_mm"]
    for i, entry in enumerate(PALETTE):
        if entry.get("brush") not in BRUSH_PRESETS:
            problems.append("PALETTE[%d] %s 的 brush 档位不存在: %r"
                            % (i, entry.get("name"), entry.get("brush")))
        else:
            press = brush_of(entry)["press_mm"]
            if press > worst + 1e-6:
                problems.append("PALETTE[%d] %s 下压量 %.2fmm 超过最粗档 %.2fmm"
                                % (i, entry.get("name"), press, worst))
    if need_stations and StationConfig.STATIONS_ARE_PLACEHOLDER:
        problems.append("StationConfig.STATIONS_ARE_PLACEHOLDER 仍为 True："
                        "站点坐标未测量，禁止执行绘制（可先 --probe 空跑）")
    if not (ReachConfig.Z_MIN < ArmConfig.PAPER_Z < ReachConfig.Z_MAX):
        problems.append("PAPER_Z=%.1f 超出 Z 可达范围 [%.1f, %.1f]"
                        % (ArmConfig.PAPER_Z, ReachConfig.Z_MIN, ReachConfig.Z_MAX))

    # ★★ 换笔后最容易忘、后果最隐蔽的一个坑 ★★
    # PAPER_Z 换笔后会整体平移（-52.88 -> -40.08 实测），但 Z_MAX 是手写的。
    # 忘了改的话，check_waypoints() 会把【每一个绘制点】都判为越界，
    # 表现是"点了 --execute 却什么都不画、直接拒绝执行"，很难看出原因。
    deepest = ArmConfig.PAPER_Z + ShuimoConfig.PRESS_MAX_MM
    if ReachConfig.Z_MAX < deepest - 1e-6:
        problems.append(
            "ReachConfig.Z_MAX=%.2f 太浅：行笔最深要到 %.2f"
            "（= PAPER_Z %.2f + ShuimoConfig.PRESS_MAX_MM %.2f）。"
            "不改的话每一个绘制点都会被判越界、直接拒绝执行。"
            % (ReachConfig.Z_MAX, deepest, ArmConfig.PAPER_Z,
               ShuimoConfig.PRESS_MAX_MM))
    # 抬笔高度也必须浅于最深点，否则"落笔"命令本身就超范围
    lift_z = z_for_lift(ArmConfig.LIFT_MM)
    if ReachConfig.Z_MAX < lift_z - 1e-6:
        problems.append("ReachConfig.Z_MAX=%.2f 比抬笔高度 %.2f 还浅，"
                        "落笔/抬笔命令都会被拒" % (ReachConfig.Z_MAX, lift_z))
    return problems


def describe():
    """打印配置概览（--check 用）"""
    print("=" * 70)
    print(" 机械臂绘画 - 配置概览")
    print("=" * 70)
    print(" 机械臂 %s:%s  USER_INDEX=%d TOOL_INDEX=%d 全局速度=%d%%"
          % (ArmConfig.IP, ArmConfig.PORT, ArmConfig.USER_INDEX,
             ArmConfig.TOOL_INDEX, ArmConfig.SPEED_FACTOR))
    print(" 纸面 Z=%.2f  Z增大=下压:%s  抬笔=%.1fmm  站点间高度=%.1fmm"
          % (ArmConfig.PAPER_Z, ArmConfig.Z_INCREASES_DOWN,
             ArmConfig.LIFT_MM, ArmConfig.TRANSIT_LIFT_MM))
    print(" 画布四角（用户坐标系 mm）:")
    for name, pt in zip(("左上", "右上", "右下", "左下"), CanvasConfig.ROBOT_CORNERS):
        print("   %s (%8.1f, %8.1f)" % (name, pt[0], pt[1]))
    print(" 调色板 %d 色:" % len(PALETTE))
    for i, entry in enumerate(PALETTE):
        b = brush_of(entry)
        pos = dip_position(i)
        print("   [%d] %-4s rgb=%-16s 层=%d 档=%-4s 宽=%.0fmm 压=%.2fmm 色格=(%.1f, %.1f)"
              % (i, entry["name"], str(entry["rgb"]), entry["layer"],
                 entry.get("brush"), b["width_mm"], b["press_mm"], pos[0], pos[1]))
    wash = StationConfig.WASH
    wipe = StationConfig.WIPE
    print(" 水盂 (%.1f, %.1f) 水面 %.1f 插深 %.1fmm" %
          (wash["x"], wash["y"], wash["z_surface"], wash["dip_mm"]))
    print(" 擦笔 (%.1f, %.1f)->(%.1f, %.1f) 布面 %.1f" %
          (wipe["x0"], wipe["y0"], wipe["x1"], wipe["y1"], wipe["z_surface"]))
    print(" 试色区 启用=%s" % StationConfig.TEST_STRIP["enabled"])
    print(" 站立坐标是否已测量: %s" % (not StationConfig.STATIONS_ARE_PLACEHOLDER))
    print("-" * 70)
    problems = validate(need_stations=False)
    if problems:
        print(" 配置问题:")
        for p in problems:
            print("   [!] %s" % p)
    else:
        print(" 配置自检通过")
    print("=" * 70)
    return problems


if __name__ == "__main__":
    describe()
