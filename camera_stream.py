#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IP 摄像头实时流读取与抓拍工具。
"""
from typing import Generator, Optional

import cv2
import numpy as np


def build_stream_url(
    ip: str,
    port: int = 554,
    username: str = "",
    password: str = "",
    protocol: str = "rtsp",
    stream_path: str = "11",
) -> str:
    if protocol.lower() == "rtsp":
        return f"rtsp://{username}:{password}@{ip}:{port}/{stream_path}"
    if protocol.lower() == "http":
        return f"http://{username}:{password}@{ip}:{port}/{stream_path}"
    raise ValueError(f"Unsupported protocol: {protocol}. Only 'rtsp' or 'http' are allowed.")


def get_camera_frames(
    ip: str,
    port: int = 554,
    username: str = "",
    password: str = "",
    protocol: str = "rtsp",
    stream_path: str = "11",
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Generator[np.ndarray, None, None]:
    """
    封装函数：连接 IP 摄像头，迭代获取最新视频帧。
    """
    stream_url = build_stream_url(
        ip=ip,
        port=port,
        username=username,
        password=password,
        protocol=protocol,
        stream_path=stream_path,
    )

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise ConnectionError(f"Cannot open camera stream at {stream_url}")

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()
        cv2.destroyAllWindows()
