# -*- coding: utf-8 -*-
"""景区 AI 画师对话：名家介绍、选画家、选树/花鸟、触发生成。"""
from master_studio import profiles

NAMES = {a["name"]: a for a in profiles.ARTISTS}
ALIASES = {"黄公望": "黄公望", "大痴": "黄公望", "倪瓒": "倪瓒", "云林": "倪瓒",
           "石涛": "石涛", "苦瓜": "石涛", "龚贤": "龚贤", "半千": "龚贤",
           "八大山人": "八大山人", "朱耷": "八大山人"}


def find_artist(text):
    for k, v in ALIASES.items():
        if k in text:
            return v
    return None


def reply(text, state=None):
    state = state or {}
    text = (text or "").strip()
    artist = find_artist(text) or state.get("artist") or "黄公望"
    a = NAMES.get(artist) or profiles.ARTISTS[0]

    if any(k in text for k in ("你是谁", "介绍你", "能做什么", "帮助", "怎么用")):
        return dict(text="我是山河，景区画房里的 AI 画师。你可以问我某位名家的故事，也可以直接说：\n\n"
                          "· “介绍一下黄公望”\n· “画四幅倪瓒的山水”\n· “我想要一棵松树”\n· “画一幅荷花”\n\n"
                          "我会带你选画家、选题材，再生成可交给机械臂的稿子。",
                    action=None, artist=artist)

    if any(k in text for k in ("介绍", "是谁", "生平", "故事", "风格")):
        return dict(text=f"{a['name']}（{a['dynasty']}）\n\n{a['intro']}\n\n代表作：{a['work']}\n\n"
                          f"他的用笔：{a['theory']}\n\n要不要我直接用他的风格生成四幅山水？",
                    action=None, artist=artist)

    if any(k in text for k in ("松", "树", "枯枝", "柳")):
        kind = "bare" if "枯" in text else "pine" if "松" in text else "broadleaf"
        return dict(text=f"好，给你一棵{'枯枝' if kind=='bare' else '松树' if kind=='pine' else '杂树'}。\n"
                          "树法现在是独立模块，不再混进山水；生成后可以单独导出。",
                    action=dict(type="tree", kind=kind), artist=artist)

    if any(k in text for k in ("花", "荷", "梅", "兰", "菊", "鸟")):
        kind = "lotus" if "荷" in text else "plum" if "梅" in text else "orchid" if "兰" in text else "chrysanthemum"
        return dict(text=f"花鸟也单独成稿。先给你一幅{'荷花' if kind=='lotus' else '梅花' if kind=='plum' else '兰花' if kind=='orchid' else '菊花'}小品。",
                    action=dict(type="flower", kind=kind), artist=artist)

    if any(k in text for k in ("画", "生成", "预览", "来", "做", "幅")):
        count = 4 if any(k in text for k in ("四", "4")) else 2
        return dict(text=f"用{a['name']}的笔意，给你 {count} 幅景区山水。\n"
                          "生成后可以选一幅继续画成机械臂稿。",
                    action=dict(type="landscape", count=count), artist=artist)

    if artist and artist in text:
        return dict(text=f"已切换到{a['name']}。你可以说“画四幅”，或问我他的皴法。",
                    action=None, artist=artist)

    return dict(text=f"我在。当前可以围绕{a['name']}的山水，或者单独画树、画花鸟。\n"
                      "你可以说：介绍黄公望 / 画四幅倪瓒的山水 / 我要一棵松树 / 画荷花。",
                action=None, artist=artist)