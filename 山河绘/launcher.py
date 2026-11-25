# -*- coding: utf-8 -*-
"""
launcher.py - 图形化之外的"傻瓜菜单"入口
====================================================================
双击项目里的 run.bat 就会进这个菜单（run.bat 只有三行，不依赖 cmd 的复杂语法）。

  uv run python launcher.py        # 也可以直接在终端里跑

它只是把常用的 main.py 参数做成编号菜单，所有功能仍然是 main.py 提供的。
"""

import os
import sys
from project_paths import PROJECT_DIR, OUTPUT_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def setup_console():
    """
    把控制台输入/输出都固定成 UTF-8：
      - 否则中文可能乱码
      - 否则从管道喂进来的输入会按 GBK 解码（首行的 BOM 会变成乱码字符）
    """
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
    except Exception:
        pass
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


MENU = [
    ("1", "读取 JSON 绘画 / 继续上次绘画", ["--json-painting"], False),
    ("2", "从图片生成绘画 JSON 和预览", ["--unified-painting"], False),
    ("3", "放弃旧任务（换新纸）", ["--abandon"], False),
]

def read_line(prompt=""):
    """读一行输入，去掉 BOM 与首尾空白"""
    raw = input(prompt)
    return raw.replace("\ufeff", "").strip()


def normalise_choice(raw):
    """
    菜单选项容错：取开头连续的数字（这样 10、11 这类两位选项也能识别）。
    同时容忍输入被加上 BOM 或夹杂控制字符（某些环境会这样）。
    """
    text = raw.replace("\ufeff", "").strip()
    digits = ""
    for char in text:
        if char.isdigit():
            digits += char
        elif digits:
            break
    return digits if digits else text.lower()


def ask_image():
    raw = read_line("\n图片/范图路径（可把文件拖进窗口；直接回车返回菜单）：")
    if not raw:
        return None
    return raw.strip('"').strip("'")


def run_action(argv, need_image):
    if argv == ["--json-painting"]:
        from run_json_painting import main as json_main
        return json_main([])
    if argv == ["--abandon"]:
        from abandon_old_task import main as abandon_main
        return abandon_main()
    if argv and argv[0] == "--unified-painting":
        from unified_painting import main as painting_main
        try:
            return painting_main()
        except Exception as exc:
            print("[停止] %s: %s" % (type(exc).__name__, exc))
            return 2
    if argv == ["--plum-experiment"]:
        from run_plum_experiment import main as experiment_main
        experiment_main()
        return
    if argv == ["--painting-experiment"]:
        from run_painting_experiment import main as experiment_main
        experiment_main()
        return
    if argv == ["--color-experiment"]:
        from run_color_experiment import main as experiment_main
        experiment_main()
        return
    image = None
    if need_image:
        image = ask_image()
        if not image:
            print("已取消。")
            return
        if not os.path.isfile(image):
            print("[!] 找不到文件：%s" % image)
            return
    args = [image if item == "{img}" else item for item in argv] if image else list(argv)
    if "painting" in args:
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args += ["--job-out", str(OUTPUT_DIR / (stamp+".json")),
                 "--preview-out", str(OUTPUT_DIR / (stamp+".png"))]
    print("")
    print("[执行] main.py %s" % " ".join(args))
    print("-" * 66)
    import main
    try:
        main.main(args)
    except KeyboardInterrupt:
        print("\n[中断] 已停止。")
    except SystemExit:
        pass
    except Exception as exc:                      # noqa: BLE001
        print("[错误] %s: %s" % (type(exc).__name__, exc))
    print("-" * 66)


def main():
    os.chdir(PROJECT_DIR)
    setup_console()
    keys = [key for key, _label, _argv, _img in MENU]
    hint = "/".join(keys + ["0"])
    while True:
        print("")
        print("=" * 66)
        print("  机械臂绘画 - 控制台        (%s)" % HERE)
        print("=" * 66)
        for key, label, _argv, _img in MENU:
            print("   %-3s %s" % (key + ".", label))
        print("   0   退出")
        print("=" * 66)
        try:
            choice = normalise_choice(read_line("请选择 [%s]：" % hint))
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0
        if choice in ("", "0", "q", "Q"):
            return 0
        picked = None
        for key, label, argv, need_image in MENU:
            if key == choice:
                picked = (label, argv, need_image)
                break
        if picked is None:
            print("[!] 请输入 %s 之一" % hint)
            continue
        label, argv, need_image = picked
        print("\n>>> %s" % label)
        if any(flag in argv for flag in ("--teach", "--probe", "--calibrate-z",
                                         "--test-strokes")):
            print("    （需要机械臂已开机、已切到 TCP/IP 控制模式、且 config.py 里")
            print("      的 USER_INDEX / TOOL_INDEX 已设成你在 DobotStudio Pro 里建好的那套）")
        run_action(argv, need_image)
        try:
            input("\n按回车返回菜单...")
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0


if __name__ == "__main__":
    sys.exit(main())
