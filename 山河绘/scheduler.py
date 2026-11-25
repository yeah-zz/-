# -*- coding: utf-8 -*-
"""
scheduler.py - 绘制顺序与换色调度（纯逻辑，不含坐标/Z）
====================================================================

【排程规则】三层排序，优先级从高到低：
  1. 层号 layer：先铺远景/大面积底色，再画中景，最后画主体与暗部（后画的覆盖先画的）
  2. 颜色 pal_index：同一颜色一次画完 —— 换色一次要洗笔+擦笔+蘸色+试色，
     约 20~30 秒，这是整幅画最大的时间开销，必须集中
  3. 空间：同色内部用最近邻串成一串，减少空跑

【产出的"步骤"是纯逻辑对象】
      {"kind": "wash"}                     洗笔
      {"kind": "wipe"}                     擦笔
      {"kind": "dip",  "pal_index": i, ...} 蘸色
      {"kind": "test", "pal_index": i}      试色（画一小条）
      {"kind": "stroke", "stroke": {...}}   一条宽笔触
  由 job.py 负责把这些步骤翻译成带坐标与 Z 的机械臂轨迹。
"""

import math

from config import JobConfig, PALETTE, ShuimoConfig, StationConfig


def _distance(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _chain(strokes, start_xy):
    """
    最近邻串接：每次挑离当前位置最近的笔触端点，必要时反向。
    strokes 数量在几百级，O(n^2) 完全够用。
    """
    remaining = list(strokes)
    ordered = []
    current = start_xy
    while remaining:
        best_index = 0
        best_distance = None
        best_reverse = False
        for i, stroke in enumerate(remaining):
            head = stroke["points"][0]
            tail = stroke["points"][-1]
            if current is None:
                # 第一笔：从图像左上角开始，扫得自然一点
                d_head = math.hypot(float(head[0]), float(head[1]))
                d_tail = math.hypot(float(tail[0]), float(tail[1]))
            else:
                d_head = _distance(current, head)
                d_tail = _distance(current, tail)
            if best_distance is None or min(d_head, d_tail) < best_distance:
                best_distance = min(d_head, d_tail)
                best_index = i
                best_reverse = d_tail < d_head
        stroke = remaining.pop(best_index)
        if best_reverse:
            stroke = dict(stroke)
            stroke["points"] = list(reversed(stroke["points"]))
        ordered.append(stroke)
        current = stroke["points"][-1]
    return ordered


def group_strokes(strokes):
    """按 (layer, 颜色序号) 分组并排序，返回 [(key, [stroke, ...]), ...]"""
    buckets = {}
    for stroke in strokes:
        key = (int(stroke["layer"]), int(stroke["pal_index"]))
        buckets.setdefault(key, []).append(stroke)
    return [(key, buckets[key]) for key in sorted(buckets.keys())]


def build_plan(strokes):
    """
    输入笔触列表（图像坐标），输出排好序的步骤列表与统计信息。
    """
    groups = group_strokes(strokes)
    steps = []
    stats = {
        "groups": [],
        "strokes": 0,
        "length_mm": 0.0,
        "dips": 0,
        "washes": 0,
        "wipes": 0,
        "tests": 0,
    }
    current_xy = None

    for group_index, (key, group) in enumerate(groups):
        layer, pal_index = key
        ordered = _chain(group, current_xy)
        entry = PALETTE[pal_index]

        # 换色前的清洗：第一个颜色看配置决定是否也要洗
        if not JobConfig.SKIP_PIGMENT_ACTIONS:
            if group_index > 0 or JobConfig.WASH_BEFORE_FIRST_COLOR:
                steps.append({"kind": "wash", "pal_index": pal_index})
                stats["washes"] += 1
                steps.append({"kind": "wipe", "pal_index": pal_index})
                stats["wipes"] += 1
            steps.append({"kind": "dip", "pal_index": pal_index, "reason": "换色"})
            stats["dips"] += 1
            if JobConfig.TEST_AFTER_DIP:
                steps.append({"kind": "test", "pal_index": pal_index})
                stats["tests"] += 1

        group_length = 0.0
        group_total = 0.0
        group_count = 0          # 注意：不要叫 group_strokes，会遮蔽同名函数
        for stroke in ordered:
            steps.append({"kind": "stroke", "stroke": stroke,
                          "pal_index": pal_index, "layer": layer})
            group_length += float(stroke["length_mm"])
            group_total += float(stroke["length_mm"])
            group_count += 1
            stats["strokes"] += 1
            stats["length_mm"] += float(stroke["length_mm"])
            current_xy = stroke["points"][-1]

            # 同一颜色画太久了：补蘸一次（此时仍在同一颜色的段落中）
            if (not JobConfig.SKIP_PIGMENT_ACTIONS) and \
                    group_length >= float(JobConfig.REDIP_LENGTH_MM):
                steps.append({"kind": "dip", "pal_index": pal_index, "reason": "补蘸"})
                stats["dips"] += 1
                if JobConfig.TEST_AFTER_DIP:
                    steps.append({"kind": "test", "pal_index": pal_index})
                    stats["tests"] += 1
                group_length = 0.0

        stats["groups"].append({
            "layer": layer,
            "name": entry["name"],
            "pal_index": pal_index,
            "strokes": group_count,
            "length_mm": round(group_total, 1),
        })

    return steps, stats


def print_plan(stats):
    """打印排程摘要"""
    print("[排程] 共 %d 条笔触，总长 %.0f mm" % (stats["strokes"], stats["length_mm"]))
    print("[排程] 颜色段 %d 个：洗笔 %d / 擦笔 %d / 蘸色 %d / 试色 %d"
          % (len(stats["groups"]), stats["washes"], stats["wipes"],
             stats["dips"], stats["tests"]))
    for g in stats["groups"]:
        print("   层%d %-4s  %3d 笔  长度 %7.0f mm"
              % (g["layer"], g["name"], g["strokes"], g["length_mm"]))


# =====================================================================
# 水墨框架（Pass 1）排程
# =====================================================================
def build_framework_plan(strokes, pal_index):
    """
    水墨框架的排程：只有一种墨、一支笔，所以没有换色循环。

    三条和色块路线不同的规矩：

    1. **保持生成器给的顺序**，不做最近邻重排。
       色块路线用 _chain() 重排是为了省空跑；水墨的落笔次序直接影响观感 ——
       shuimo.py 已经按"主体长轮廓 -> 中等结构线 -> 短皴线"排好（模拟人的
       作画次序），重排会让皴线跑到轮廓前面，画面结构就散了。

    2. 补蘸是**前瞻式**的：画每条笔触**之前**先看这笔要画多长，
       如果笔上剩的墨不够，就**先蘸饱再落笔**。

       为什么不能"画完再补"（累计式）：想补蘸就必须抬笔，抬笔就断线。
       累计式会让"一条长笔触用半干的笔起笔"，画到一半莫名其妙地枯掉，
       而且枯的位置和笔触长度无关，看起来像随机断墨。
       前瞻式保证每条笔触都是**饱墨起笔**，补蘸永远发生在笔触之间。

    3. 容量按**极细狼毫**设（默认 150mm，远小于色块路线的 400mm）。
       狼毫储墨量小，这个值必须现场实测：让它连着画，看第几笔开始变淡。
    """
    steps = []
    stats = {"groups": [], "strokes": 0, "length_mm": 0.0,
             "dips": 0, "washes": 0, "wipes": 0, "tests": 0,
             "capacity_mm": 0.0, "lookahead_dips": 0,
             "overlong_count": 0, "overlong_max_mm": 0.0}
    if not strokes:
        return steps, stats

    brush = ShuimoConfig.BRUSHES.get(ShuimoConfig.DEFAULT_BRUSH) or {}
    test_press = brush.get("press_mm")
    test_speed = brush.get("speed")

    # 每次蘸墨后要在试墨区画一小条，那一小条也吃墨，必须从容量里扣掉，
    # 否则最后一条笔触会差这一点点墨而提前变枯。
    test_len = 0.0
    if ShuimoConfig.TEST_AFTER_DIP:
        strip = StationConfig.TEST_STRIP
        if strip.get("enabled", True):
            test_len = math.hypot(float(strip["x1"]) - float(strip["x0"]),
                                  float(strip["y1"]) - float(strip["y0"]))
    capacity = max(1.0, float(ShuimoConfig.REDIP_LENGTH_MM) - test_len)
    stats["capacity_mm"] = capacity

    skip = bool(JobConfig.SKIP_PIGMENT_ACTIONS)
    state = {"remaining": capacity}

    def dip(reason_text):
        """蘸墨（+试墨），并把笔上的墨量重置为满容量"""
        steps.append({"kind": "dip", "pal_index": pal_index, "reason": reason_text})
        stats["dips"] += 1
        if ShuimoConfig.TEST_AFTER_DIP:
            steps.append({"kind": "test", "pal_index": pal_index,
                          "press_mm": test_press, "speed": test_speed})
            stats["tests"] += 1
        state["remaining"] = capacity

    if not skip:
        # 开画第一笔：干净笔可以直接蘸墨。WASH_BEFORE_FIRST_COLOR 打开时才洗。
        if JobConfig.WASH_BEFORE_FIRST_COLOR:
            steps.append({"kind": "wash", "pal_index": pal_index})
            stats["washes"] += 1
            steps.append({"kind": "wipe", "pal_index": pal_index})
            stats["wipes"] += 1
        dip("开画蘸墨")

    # ★ 出锋（TAIL_MM）只在【够长的笔触】上才会真的加，见 job.stroke_waypoints：
    #     · 点状/串珠模式（DAB_LENGTH_MM>0）下出锋被整体关闭
    #     · 单独一条笔触短于 2×TAIL_MM 时也不加
    #   这里是"墨量账"，必须和实际走的路一致 —— 算多了会导致蘸墨次数翻倍。
    #   踩过的坑：点状模式 295 条 × 4mm = 1180mm 虚假笔长，
    #             蘸墨从 9 次虚报成 17 次，白跑 3 米多。
    _tail_mm = float(getattr(ShuimoConfig, "TAIL_MM", 0.0))
    if float(getattr(ShuimoConfig, "DAB_LENGTH_MM", 0.0) or 0.0) > 1e-6:
        _tail_mm = 0.0

    for stroke in strokes:
        # 实际笔程要把"收笔出锋"那一小段算进去 —— 每个笔触末尾会多走 TAIL_MM
        # （job.stroke_waypoints 里加的），不算的话补蘸会偏晚。
        _len = float(stroke["length_mm"])
        tail = float(stroke.get("tail_mm", _tail_mm)) if _tail_mm else 0.0
        length = _len + (tail if _len >= tail * 2.0 else 0.0)

        if not skip:
            # ★ 前瞻：这笔比"笔上还剩的墨"能画的更长 -> 先蘸饱再落笔
            if length > state["remaining"] + 1e-6:
                if length > capacity + 1e-6:
                    # 蘸满也不够画完 -> 这笔必然中途出枯笔，记下来提醒
                    stats["overlong_count"] += 1
                    stats["overlong_max_mm"] = max(
                        stats["overlong_max_mm"], length)
                # A fresh load cannot be made fuller: avoid a duplicate dip
                # before an over-capacity first stroke.
                if state["remaining"] < capacity - 1e-6:
                    dip("补蘸")
                    stats["lookahead_dips"] += 1
            state["remaining"] -= length

        steps.append({"kind": "stroke", "stroke": stroke, "pal_index": pal_index,
                      "layer": int(stroke.get("layer", 0))})
        stats["strokes"] += 1
        stats["length_mm"] += length

    stats["groups"].append({
        "layer": 1,
        "name": PALETTE[pal_index]["name"],
        "pal_index": pal_index,
        "strokes": stats["strokes"],
        "length_mm": round(stats["length_mm"], 1),
    })
    return steps, stats


def print_framework_plan(stats):
    """打印水墨框架排程摘要"""
    if stats.get("policy") == "balanced-experimental":
        print("[排程] %d 笔；预测蘸墨 %d 次；墨内停留合计 %.1fs（未标定模型）" %
              (stats["strokes"], stats["dips"], stats["total_dip_dwell_s"]))
        print("[排程] 容量为笔宽加权的等效长度，实际执行按时钟重新判断；每20笔高位休息20s。")
        return
    print("[排程] 框架：%d 条笔触，总笔长 %.0f mm" %
          (stats["strokes"], stats["length_mm"]))
    print("[排程] 蘸墨 %d 次（其中前瞻补蘸 %d 次）/ 试墨 %d 次"
          % (stats["dips"], stats.get("lookahead_dips", 0), stats["tests"]))
    print("[排程] 单次蘸墨有效容量 %.0f mm —— 补蘸只在笔触之间发生，"
          "一笔之内绝不会抬笔" % stats.get("capacity_mm", 0.0))

    over = int(stats.get("overlong_count", 0))
    if over:
        print("[诊断] 有 %d 条笔触长于单次蘸墨容量（最长 %.0f mm）："
              % (over, stats.get("overlong_max_mm", 0.0)))
        print("       这些笔触即使蘸满，画到后段也会自然出枯笔。三个选择：")
        print("       1) 接受 —— 干笔/飞白本来就是水墨追求的效果")
        print("       2) 墨调稀一点，同体积墨能画更长（线会更淡）")
        print("       3) 调大 ShuimoConfig.REDIP_LENGTH_MM —— 但先确认笔真能画那么长")
