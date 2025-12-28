#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小米摄像头视频流处理工具。

说明：
- 推荐通过 ha_xiaomi_home 获取 stream_url，并将其写入 configs/xiaomi_camera.json。
- 若未配置 stream_url，可通过 provider.module/provider.function 调用自定义函数生成。
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
from typing import Any, Dict, Generator, Optional

import cv2
import numpy as np


DEFAULT_CONFIG_PATH = os.path.join("configs", "xiaomi_camera.json")


def load_xiaomi_config(path: str = DEFAULT_CONFIG_PATH) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_stream_url(config: Dict[str, Any]) -> str:
    if not config:
        raise RuntimeError("未找到小米摄像头配置，请先创建 configs/xiaomi_camera.json")

    stream_url = config.get("stream_url")
    if stream_url:
        return stream_url

    provider = config.get("provider") or {}
    if isinstance(provider, str):
        provider = {"module": provider}

    module_path = provider.get("module", "ha_xiaomi_home")
    function_name = provider.get("function", "get_camera_stream_url")

    if importlib.util.find_spec(module_path) is None:
        raise RuntimeError(f"未安装 {module_path}，且配置未提供 stream_url")

    module = importlib.import_module(module_path)
    resolver = getattr(module, function_name, None)
    if not callable(resolver):
        raise RuntimeError(
            f"{module_path} 缺少 {function_name} 方法，请在配置中提供 stream_url "
            "或指定 provider.function"
        )

    return resolver(config)


def _open_stream(stream_url: str, width: Optional[int], height: Optional[int]) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise ConnectionError(f"无法打开小米视频流: {stream_url}")
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def get_xiaomi_frames(
    config_path: str = DEFAULT_CONFIG_PATH,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Generator[np.ndarray, None, None]:
    config = load_xiaomi_config(config_path)
    stream_url = resolve_stream_url(config)
    cap = _open_stream(stream_url, width, height)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()
        cv2.destroyAllWindows()
