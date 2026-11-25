# -*- coding: utf-8 -*-
"""
imgio.py - 支持中文路径的图片读写
====================================================================

【为什么需要这个模块】
  OpenCV 在 Windows 上处理带中文的路径时会失败（cv2.imwrite 直接返回 False，
  表现为"预览图写出失败"）。本项目所在目录名就是中文，所以必须绕开：
    - 读：np.fromfile 读字节 -> cv2.imdecode
    - 写：cv2.imencode 得字节 -> Python open(..., "wb") 写盘
  Python 自己的文件 IO 完全支持 Unicode 路径。
"""

import os

import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    """读图（支持中文路径）。失败返回 None"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path, image):
    """写图（支持中文路径）。成功返回 True"""
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    extension = os.path.splitext(path)[1]
    if not extension:
        extension = ".png"
    ok, buffer = cv2.imencode(extension, image)
    if not ok:
        return False
    try:
        with open(path, "wb") as handle:
            handle.write(buffer.tobytes())
    except OSError:
        return False
    return True


def exists(path):
    return os.path.isfile(path)
