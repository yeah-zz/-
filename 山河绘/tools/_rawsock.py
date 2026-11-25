# -*- coding: utf-8 -*-
"""
裸 socket 诊断（跑完即删）—— 看控制器在连接层面到底怎么反应
====================================================================
依次试：
  A. 连上之后什么都不发，看有没有欢迎横幅、连接稳不稳
  B. 连上后等 2 秒再发一条最简单的指令
  C. 连发两条
  D. 换一个思路：立刻重连第二次（看是不是连接数限制）
全程只读，不发运动指令。
"""

# --- path bootstrap: this script now lives in 绘画/tools/, put 绘画/ on sys.path ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import socket
import time

IP, PORT = "192.168.5.1", 29999


def raw_try(label, delay_before_send=0.0, cmds=("RobotMode()",), read_banner=True):
    print("=" * 70)
    print(" %s" % label)
    print("=" * 70)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(6.0)
        s.connect((IP, PORT))
        print("   connect        : OK")
    except Exception as exc:
        print("   connect        : 失败 %s" % exc)
        return

    if read_banner:
        s.settimeout(1.0)
        try:
            data = s.recv(4096)
            print("   欢迎横幅       : %r" % data)
        except socket.timeout:
            print("   欢迎横幅       : (无，超时)")
        except Exception as exc:
            print("   欢迎横幅       : 读取异常 %s" % exc)

    if delay_before_send:
        print("   等待 %.1f 秒..." % delay_before_send)
        time.sleep(delay_before_send)

    s.settimeout(6.0)
    for i, cmd in enumerate(cmds):
        try:
            s.sendall((cmd + "\n").encode("utf-8"))
            print("   发送 [%d] %-14s: OK" % (i + 1, cmd))
        except Exception as exc:
            print("   发送 [%d] %-14s: 失败 %s" % (i + 1, cmd, exc))
            break
        try:
            buf = b""
            end = time.time() + 4.0
            while time.time() < end:
                if b";" in buf:
                    break
                chunk = s.recv(4096)
                if not chunk:
                    buf += b"<PEER-CLOSED>"
                    break
                buf += chunk
            print("   应答 [%d]        : %r" % (i + 1, buf.decode("utf-8", "replace")))
        except socket.timeout:
            print("   应答 [%d]        : (超时，无应答)" % (i + 1))
        except Exception as exc:
            print("   应答 [%d]        : 异常 %s" % (i + 1, exc))
            break
    try:
        s.close()
    except Exception:
        pass
    print("   已关闭")
    print("")


raw_try("A. 连上就发 RobotMode()")
time.sleep(1.0)
raw_try("B. 连上后等 2 秒再发")
time.sleep(1.0)
raw_try("C. 连发两条", cmds=("RobotMode()", "GetErrorID()"))
time.sleep(1.0)
raw_try("D. 连上不发任何东西（只看连接稳不稳）", cmds=())
