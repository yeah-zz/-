# -*- coding: utf-8 -*-
"""
robot.py - Dobot Magician E6 TCP/IP Dashboard 驱动（自包含，只用标准库）
====================================================================

【协议要点（对照 Dobot V4 接口文档）】
  端口 29999，文本指令，以 "\\n" 结束；应答格式 "ErrorID,返回值,指令名(参数);"
  - 使能/断使能 : EnableRobot() / DisableRobot()
  - 速度倍率    : SpeedFactor(1~100)，实际速度 = 指令 v × 再现设置 × 全局倍率
  - 坐标系      : User(index) / Tool(index)
  - 直线运动    : MovL(pose={x,y,z,rx,ry,rz},user=..,tool=..,v=..,cp=..)
                  v 是速度百分比(1~100)，cp 是平滑过渡比例(0~100)
  - 状态查询    : RobotMode()（5=已使能空闲，7=运行中，9=报警）、GetErrorID()

【相对旧项目驱动做的加固】
  1. 应答按 ";" 分帧并累积读取（TCP 可能把一个应答拆成多次 recv）
  2. 每条指令失败都返回原始文本，由调用方决定是否中断
  3. wait_idle 里对 socket 异常做了保护，不会因为一次抖动直接崩栈
  4. 坐标保留 2 位小数（0.01mm），避免旧项目 0.1mm 量化吃掉细节
"""

import socket
import time
import threading

from config import ArmConfig


class RobotError(Exception):
    """机械臂通信或指令错误"""
    pass


# ---------------------------------------------------------------------
# RobotMode() 返回值枚举（对照《Dobot TCP/IP 二次开发接口文档 V4.6.5》p35~36）
# ---------------------------------------------------------------------
# 旧版本这里的注释写成"6 运行 7 暂停 8 错误 9 报警 10 急停"，和文档实际不符，
# 导致 wait_idle 把正常的"单次运动(8)"误判成异常。下面按文档更正。
MODE_INIT = 1           # 初始化
MODE_BRAKE_OPEN = 2     # 有任意关节抱闸松开
MODE_POWEROFF = 3       # 下电
MODE_DISABLED = 4       # 未使能（无抱闸松开）
MODE_ENABLE = 5         # 使能且空闲  <- 运动完成就是回到这个状态
MODE_BACKDRIVE = 6      # 拖拽模式
MODE_RUNNING = 7        # 运行中（工程 / TCP 队列运动）
MODE_SINGLE_MOVE = 8    # 单次运动（点动、RunTo）
MODE_ERROR = 9          # 有未清除的报警（优先级最高）
MODE_PAUSE = 10         # 暂停
MODE_COLLISION = 11     # 碰撞检测触发

# 还在动 —— 继续等，不能判为失败
MODE_MOVING = (MODE_BACKDRIVE, MODE_RUNNING, MODE_SINGLE_MOVE, MODE_PAUSE)
# 真异常 —— 立刻失败
MODE_BAD = (MODE_INIT, MODE_BRAKE_OPEN, MODE_POWEROFF, MODE_DISABLED,
            MODE_ERROR, MODE_COLLISION)


class E6Robot:
    def __init__(self, ip=None, port=None):
        self.ip = ip or ArmConfig.IP
        self.port = int(port or ArmConfig.PORT)
        self.sock = None
        self.buffer = ""
        self.connected = False
        self._io_lock = threading.RLock()

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------
    def connect(self, enable=True):
        """
        建立连接并使能。返回 (ok, message)。
        注意：机械臂必须已在 DobotStudio Pro 中切到 TCP/IP 控制模式。
        """
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(float(ArmConfig.CONNECT_TIMEOUT))
            self.sock.connect((self.ip, self.port))
            self.sock.settimeout(float(ArmConfig.REPLY_TIMEOUT))
            self.connected = True
            self.buffer = ""
        except Exception as exc:
            self.connected = False
            return False, "连接 %s:%s 失败: %s" % (self.ip, self.port, exc)

        # ★ 控制器只允许【一个】客户端占用 29999。如果已被占用，它会先回一句
        #   "Connection refused, IP:Port has been occupied" 再断开连接 —— 而
        #   TCP 握手是成功的，所以光看 connect() 会以为连上了。
        #   这里主动读一下横幅，给出明确提示（否则只会看到"连接已被机械臂关闭"，
        #   而且横幅还会被当成第一条指令的应答，把后面的解析全带歪）。
        try:
            self.sock.settimeout(0.6)
            banner = self.sock.recv(256)
        except socket.timeout:
            banner = b""
        except Exception:
            banner = b""
        if banner:
            text = banner.decode("utf-8", errors="replace").strip()
            low = text.lower()
            if "occupied" in low or "refused" in low:
                self.close(disable=False, verbose=False)
                return False, ("29999 端口已被占用（控制器原话: %s）\n"
                               "        -> DobotStudio Pro 还连着机械臂。"
                               "请在软件里点【断开】或直接关掉它，再重试。\n"
                               "        -> 控制器同一时刻只允许一个 29999 客户端。"
                               % text)
            # 其它横幅（正常固件没有）就丢弃，不要让它污染第一条指令的应答
        self.sock.settimeout(float(ArmConfig.REPLY_TIMEOUT))

        # ★ 下面这一段也必须兜住异常。控制器可能在任意一条指令之后直接断开
        #   （没切 TCP/IP 控制模式、连接被别的程序顶掉、控制器重启…），
        #   那时 send()/_read_frame() 会抛 RobotError。不兜住的话调用方拿不到
        #   (False, 原因)，程序直接崩栈，用户只看到一个 traceback。
        try:
            response = self.send("EnableRobot()")
            if not self._ok(response):
                self.send("ClearError()")
                time.sleep(0.3)
                response = self.send("EnableRobot()")
                if not self._ok(response):
                    return False, ("EnableRobot 失败: %r（请检查报警/急停是否已解除，"
                                   "以及是否已在 DobotStudio Pro 里切到 TCP/IP 控制模式）"
                                   % response)

            time.sleep(float(ArmConfig.ENABLE_DELAY))
            self.send("SpeedFactor(%d)" % int(max(1, min(100, ArmConfig.SPEED_FACTOR))))
            self.set_user(ArmConfig.USER_INDEX)
            self.set_tool(ArmConfig.TOOL_INDEX)
        except RobotError as exc:
            try:
                self.close(disable=False, verbose=False)
            except Exception:
                pass
            return False, ("连接后初始化失败: %s"
                           "（十有八九是没切到 TCP/IP 控制模式，"
                           "或者界面/别的脚本正占着 29999 端口）" % exc)

        return True, "已连接并使能 %s:%d" % (self.ip, self.port)

    def close(self, disable=False, verbose=True):
        if self.sock is not None and self.connected:
            if disable:
                try:
                    self.send("DisableRobot()")
                    time.sleep(0.2)
                except Exception:
                    pass
            try:
                self.sock.close()
            except Exception:
                pass
        self.connected = False
        self.sock = None
        if verbose:
            print("[机器人] 连接已关闭（%s）" % ("已断使能" if disable else "保持使能"))

    # ------------------------------------------------------------------
    # 底层收发
    # ------------------------------------------------------------------
    def send(self, command, wait=True):
        # A monitor must never take another command's reply. Journal callbacks
        # are local writes; all Dashboard exchanges share this transaction lock.
        from motion_diagnostics import attached
        with self._io_lock:
            journal = attached(self)
            if journal and journal.failed and command.split('(', 1)[0] in (
                    'MovL', 'MovJ', 'EnableRobot', 'ClearError', 'ResetRobot'):
                raise RobotError('Diagnostic fault latched; no automatic motion or recovery')
            number = journal.before_command(command) if journal else None
            try:
                response = self._send_transaction(command, wait)
            except BaseException as exc:
                if journal:
                    journal.record('command_exception', command_number=number,
                                   command=command, error=str(exc))
                raise
            if journal:
                journal.reply(number, command, response)
            return response

    def _send_transaction(self, command, wait=True):
        """
        发送一条指令并取回第一个 ";" 结束的应答帧。
        失败返回空字符串（调用方通过 _ok 判断）。
        """
        if self.sock is None or not self.connected:
            raise RobotError("机械臂未连接")
        try:
            self.sock.sendall((command + "\n").encode("utf-8"))
        except Exception as exc:
            self.connected = False
            raise RobotError("发送失败(%s): %s" % (command, exc))
        if not wait:
            return ""
        return self._read_frame()

    def _read_frame(self):
        """累积读取直到拿到一个以 ';' 结尾的完整应答帧"""
        deadline = time.time() + float(ArmConfig.REPLY_TIMEOUT)
        while time.time() < deadline:
            if ";" in self.buffer:
                frame, _, self.buffer = self.buffer.partition(";")
                return frame.strip()
            try:
                chunk = self.sock.recv(4096)
            except socket.timeout:
                break
            except Exception as exc:
                self.connected = False
                raise RobotError("接收失败: %s" % exc)
            if not chunk:
                self.connected = False
                raise RobotError("连接已被机械臂关闭")
            self.buffer += chunk.decode("utf-8", errors="replace")
        if self.buffer:
            frame, self.buffer = self.buffer, ""
            return frame.strip()
        return ""

    @staticmethod
    def _ok(response):
        return bool(response) and response.startswith("0,")

    # ------------------------------------------------------------------
    # 状态与控制指令
    # ------------------------------------------------------------------
    def enable(self):
        return self.send("EnableRobot()")

    def disable(self):
        return self.send("DisableRobot()")

    def clear_error(self):
        return self.send("ClearError()")

    def reset(self):
        return self.send("ResetRobot()")

    def set_speed_factor(self, ratio):
        ratio = int(max(1, min(100, ratio)))
        return self.send("SpeedFactor(%d)" % ratio)

    def set_user(self, index):
        index = int(max(0, min(9, index)))
        response = self.send("User(%d)" % index)
        if not self._ok(response):
            print("[机器人] 警告: 切换用户坐标系 %d 失败: %r"
                  "（该索引的坐标系可能尚未在机械臂中定义）" % (index, response))
        return response

    def set_tool(self, index):
        index = int(max(0, min(9, index)))
        response = self.send("Tool(%d)" % index)
        if not self._ok(response):
            print("[机器人] 警告: 切换工具坐标系 %d 失败: %r" % (index, response))
        return response

    def get_mode(self):
        """RobotMode(): 见文件顶部的 MODE_* 枚举（5=使能且空闲，7=运行中）"""
        try:
            response = self.send("RobotMode()")
        except RobotError:
            return -1
        parts = response.split(",")
        if len(parts) >= 2:
            text = parts[1].replace("{", "").replace("}", "").strip()
            try:
                return int(float(text))
            except ValueError:
                return -1
        return -1

    def get_error_id(self):
        try:
            return self.send("GetErrorID()")
        except RobotError as exc:
            return "GetErrorID 失败: %s" % exc

    def get_pose(self):
        """
        读取当前位姿：GetPose(user=..,tool=..)，返回 (x, y, z, rx, ry, rz) 或 None。

        【标定时的重要前提】
          返回的是指定 User/Tool 坐标系下的 TCP 位姿。当前 Tool0 的毛笔偏置
          未作为精确笔尖 TCP 标定，不能把读数直接当作真实笔尖位置。
          标定和恢复前应核对毛笔安装、User/Tool 及现场间隙。
        """
        try:
            response = self.send("GetPose(user=%d,tool=%d)"
                                 % (int(ArmConfig.USER_INDEX), int(ArmConfig.TOOL_INDEX)))
        except RobotError as exc:
            print("[机器人] GetPose 失败: %s" % exc)
            return None
        if not response or not self._ok(response):
            print("[机器人] GetPose 应答异常: %r" % response)
            return None
        try:
            body = response.split("{", 1)[1].split("}", 1)[0]
            values = [float(v) for v in body.split(",")]
        except (IndexError, ValueError):
            print("[机器人] 无法解析 GetPose 应答: %r" % response)
            return None
        if len(values) < 6:
            print("[机器人] GetPose 返回字段不足: %r" % response)
            return None
        return tuple(values[:6])

    # ------------------------------------------------------------------
    # 运动
    # ------------------------------------------------------------------
    def move_linear(self, x, y, z, velocity=30, continuous_path=0):
        """
        MovL 直线运动。velocity 是速度百分比(1~100)，continuous_path 是 cp(0~100)。
        返回原始应答文本。
        """
        velocity = int(max(1, min(100, round(float(velocity)))))
        cp = int(max(0, min(100, round(float(continuous_path)))))
        rx, ry, rz = ArmConfig.RPY
        command = ("MovL(pose={%.2f,%.2f,%.2f,%.1f,%.1f,%.1f},"
                   "user=%d,tool=%d,v=%d,cp=%d)"
                   % (float(x), float(y), float(z), float(rx), float(ry), float(rz),
                      int(ArmConfig.USER_INDEX), int(ArmConfig.TOOL_INDEX),
                      velocity, cp))
        return self.send(command)

    def move_safe(self, x, y, z, velocity=30, timeout=None):
        """单步安全移动：先查状态，再运动，然后等到位。失败抛 RobotError"""
        mode = self.get_mode()
        if mode not in (5,):
            print("[机器人] 当前模式=%d 不是空闲使能态，尝试重新使能..." % mode)
            self.enable()
            time.sleep(0.5)
        response = self.move_linear(x, y, z, velocity, 0)
        if not self._ok(response):
            raise RobotError("移动失败 target=(%.1f, %.1f, %.1f) 响应=%r"
                             % (x, y, z, response))
        if not self.wait_idle(timeout):
            raise RobotError("等待到位超时 target=(%.1f, %.1f, %.1f)" % (x, y, z))
        return response

    def wait_idle(self, timeout=None):
        """
        轮询 RobotMode 直到回到"使能且空闲"(5)。超时或真异常返回 False。

        判定按文档 p35~36：
          5            -> 到位
          6/7/8/10     -> 还在动（拖拽/运行/单次运动/暂停），继续等
          1/2/3/4/9/11 -> 真异常（初始化/抱闸/下电/失能/报警/碰撞），立刻失败
        旧实现把 (8,9,10) 一律当异常，其中 8=SINGLE_MOVE 是正常瞬时态，
        所以机器人只要处于单次运动就会被误判成"异常"。
        """
        limit = float(timeout if timeout is not None else 60.0)
        deadline = time.time() + limit
        while time.time() < deadline:
            mode = self.get_mode()
            if mode == MODE_ENABLE:
                return True
            if mode in MODE_BAD:
                print("[机器人] 异常状态 mode=%d（1 初始化/2 抱闸/3 下电/4 失能/"
                      "9 报警/11 碰撞）" % mode)
                return False
            # MODE_MOVING 或读取失败(-1) -> 继续等
            time.sleep(0.15)
        print("[机器人] 等待到位超时（%.0f 秒）" % limit)
        return False


if __name__ == "__main__":
    robot = E6Robot()
    ok, message = robot.connect()
    print(message)
    if ok:
        print("RobotMode = %d" % robot.get_mode())
        print("ErrorID   = %s" % robot.get_error_id())
        robot.close(disable=False)
