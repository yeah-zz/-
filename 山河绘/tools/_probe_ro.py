# -*- coding: utf-8 -*-
"""
只读探测（跑完即删）
====================================================================
只做三件事：
  1. TCP 能不能连上 192.168.5.1:29999
  2. 读状态：RobotMode / GetErrorID
  3. 读位姿：GetPose(user=2,tool=0) 与 GetPose(user=0,tool=0) 对比
     -> 直接验证"用户坐标系 2 到底存不存在"

★ 不发任何运动指令（没有 MovL/MovJ/回零）
★ 不改任何状态（没有 EnableRobot / DisableRobot / User() / SpeedFactor）
"""

# --- path bootstrap: this script now lives in 绘画/tools/, put 绘画/ on sys.path ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import socket
import time

IP = "192.168.5.1"
PORT = 29999
TIMEOUT = 5.0

print("=" * 66)
print(" 只读探测：只查询，不运动、不改状态")
print("=" * 66)

print("\n[1] TCP 连接 %s:%d ..." % (IP, PORT))
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(TIMEOUT)
    sock.connect((IP, PORT))
except Exception as exc:
    print("    x 连接失败: %s" % exc)
    print("    -> 机械臂没开机 / 网线没插 LAN1 / 没切到 TCP/IP 控制模式")
    raise SystemExit(1)
print("    OK 已连上（此刻还没发过任何指令）")


def read_frame():
    """累加读到 ';' 为止（TCP 会把一个应答拆成多次 recv）"""
    buf = b""
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        if b";" in buf:
            break
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            break
        except Exception as exc:
            return "<recv 失败: %s>" % exc
        if not chunk:
            return "<连接被对端关闭>"
        buf += chunk
    return buf.decode("utf-8", errors="replace").split(";")[0].strip()


# 连上后可能有欢迎横幅之类的残留数据，先排空
sock.settimeout(0.4)
try:
    while True:
        if not sock.recv(4096):
            break
except Exception:
    pass
sock.settimeout(TIMEOUT)


def ask(cmd):
    try:
        sock.sendall((cmd + "\n").encode("utf-8"))
    except Exception as exc:
        print("    %-26s -> <发送失败: %s>" % (cmd, exc))
        return ""
    reply = read_frame()
    print("    %-26s -> %s" % (cmd, reply))
    return reply


print("\n[2] 只读查询")
mode_reply = ask("RobotMode()")
ask("GetErrorID()")

print("\n[3] 位姿对比（验证用户坐标系 2 是否存在）")
pose2 = ask("GetPose(user=2,tool=0)")
pose0 = ask("GetPose(user=0,tool=0)")
ask("GetAngle()")

sock.close()
print("\n[4] 已断开。全程未发运动指令、未改状态。")

print("\n" + "=" * 66)
print(" 判读")
print("=" * 66)
print("""
 RobotMode 应答形如 "0,{5},RobotMode();"
   第一个数 0 = 指令被接受；{ } 里是模式值
     5=使能且空闲(完成) 6=拖拽 7=运行中 8=单次运动 9=报警 10=暂停 11=碰撞
     1=初始化 2=抱闸松开 3=下电 4=未使能

 GetErrorID 应答形如 "0,{},GetErrorID();"
   {} 为空 = 当前没有报警

 GetPose(user=2,...) 若返回 {x,y,z,rx,ry,rz} 的 6 个数 = User 2 存在且可用
   若返回错误码（负数）或空 {} = 该索引没定义过，需要重新用三点法标定
""")
