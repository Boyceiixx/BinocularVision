# -*-coding: utf-8 -*-
import cv2
import numpy as np


def get_filter_disparity(imgL, imgR, use_wls=True):
    """
    计算视差图并进行基础滤波
    :param imgL: 左图像（灰度图）
    :param imgR: 右图像（灰度图）
    :param use_wls: 是否使用滤波（这里使用基础滤波替代WLS）
    :return: 视差图
    """
    # SGBM参数设置
    window_size = 5
    min_disp = 0
    num_disp = 128

    # 创建SGBM对象
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    # 计算视差图
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    if use_wls:
        # 使用基础滤波替代WLS滤波
        # 双边滤波
        disparity = cv2.bilateralFilter(disparity, 9, 75, 75)

        # 中值滤波去除噪声
        disparity = cv2.medianBlur(disparity.astype(np.uint8), 5).astype(np.float32)

    return disparity


def get_simple_disparity(imgL, imgR):
    """
    简单的视差计算（不使用滤波）
    :param imgL: 左图像（灰度图）
    :param imgR: 右图像（灰度图）
    :return: 视差图
    """
    # SGBM参数设置
    window_size = 5
    min_disp = 0
    num_disp = 128

    # 创建SGBM对象
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    # 计算视差图
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    return disparity


def get_visual_disparity(disparity):
    """
    将视差图转换为可视化的彩色图像
    :param disparity: 视差图
    :return: 彩色视差图
    """
    # 归一化视差图
    disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # 应用颜色映射
    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

    return disp_color


def get_visual_depth(depth):
    """
    将深度图转换为可视化的彩色图像
    :param depth: 深度图
    :return: 彩色深度图
    """
    # 处理无效深度值
    depth_valid = np.where(depth > 0, depth, 0)

    # 归一化深度图
    if depth_valid.max() > 0:
        depth_norm = cv2.normalize(depth_valid, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    else:
        depth_norm = np.zeros_like(depth, dtype=np.uint8)

    # 应用颜色映射
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_RAINBOW)

    return depth_color


def get_depth(disparity, Q, scale=1.0):
    """
    从视差图计算深度图
    :param disparity: 视差图
    :param Q: 重投影矩阵
    :param scale: 缩放因子
    :return: 深度图
    """
    # 避免除零错误
    disparity_safe = np.where(disparity > 0, disparity, 0.1)

    # 计算深度
    baseline = 1 / Q[3, 2]
    fx = abs(Q[2, 3])
    depth = (fx * baseline) / disparity_safe

    # 处理无效视差
    depth = np.where(disparity > 0, depth, 0)
    depth = depth * scale

    return depth.astype(np.float32)