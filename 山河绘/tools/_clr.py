# -*- coding: utf-8 -*-
"""清报警 + 读状态（跑完即删）—— 无运动指令"""

# --- path bootstrap: this script now lives in 绘画/tools/, put 绘画/ on sys.path ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import socket
import time

IP, PORT, TIMEOUT = "192.168.5.1", 29999, 6.0
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(TIMEOUT)
s.connect((IP, PORT))
s.settimeout(0.8)
try:
    banner = s.recv(256)
    if banner:
        print("[横幅] %r" % banner)
except Exception:
    pass
s.settimeout(TIMEOUT)


def ask(cmd):
    s.sendall((cmd + "\n").encode("utf-8"))
    buf = b""
    end = time.time() + TIMEOUT
    while time.time() < end:
        if b";" in buf:
            break
        try:
            c = s.recv(4096)
        except socket.timeout:
            break
        if not c:
            break
        buf += c
    r = buf.decode("utf-8", "replace").split(";")[0].strip()
    print("  %-24s -> %s" % (cmd, r))
    return r


print("[1] 报警时的状态")
ask("RobotMode()")
ask("GetErrorID()")

print("\n[2] 清报警")
ask("ClearError()")
time.sleep(1.5)

print("\n[3] 清完再读")
ask("RobotMode()")
ask("GetErrorID()")
ask("GetPose(user=2,tool=0)")
ask("GetAngle()")

print("\n[4] 各关节温度 / 负载（如果能读）")
for c in ("GetTemperature()", "GetLoad()"):
    ask(c)

s.close()
print("\n已断开。全程未发运动指令。")
