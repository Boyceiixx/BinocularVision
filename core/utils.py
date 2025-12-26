# -*-coding: utf-8 -*-
"""
@Author : pan-author
@E-mail : 390737991@qq.com
@Date : 2020-04-10 18:24:06
"""
import cv2
import numpy as np


class ImageUtils:
    """图像处理工具类"""

    @staticmethod
    def get_video_capture(video):
        """
        获取视频捕获对象
        :param video: 视频文件路径或摄像头ID
        :return: VideoCapture对象
        """
        if isinstance(video, str):
            cap = cv2.VideoCapture(video)
        else:
            cap = cv2.VideoCapture(video)

        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video}")

        return cap

    @staticmethod
    def get_video_info(cap):
        """
        获取视频信息
        :param cap: VideoCapture对象
        :return: width, height, numFrames, fps
        """
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        return width, height, numFrames, fps

    @staticmethod
    def addMouseCallback(window_name, xyz_coord, info="coords"):
        """
        添加鼠标回调函数，用于显示点击位置的3D坐标
        :param window_name: 窗口名称
        :param xyz_coord: 3D坐标数组
        :param info: 信息前缀
        """

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if 0 <= y < xyz_coord.shape[0] and 0 <= x < xyz_coord.shape[1]:
                    world_x = xyz_coord[y, x, 0]
                    world_y = xyz_coord[y, x, 1]
                    world_z = xyz_coord[y, x, 2]
                    print(
                        f"{info}: pixel=({x},{y}), world coords=(x={world_x:.1f}, y={world_y:.1f}, depth={world_z:.1f})mm")

        cv2.setMouseCallback(window_name, mouse_callback)


class FileUtils:
    """文件处理工具类"""

    @staticmethod
    def create_dir(directory):
        """
        创建目录
        :param directory: 目录路径
        """
        import os
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def get_image_files(directory, image_format="png"):
        """
        获取目录下的图像文件列表
        :param directory: 目录路径
        :param image_format: 图像格式
        :return: 图像文件列表
        """
        import glob
        import os
        pattern = os.path.join(directory, f"*.{image_format}")
        return sorted(glob.glob(pattern))


# 创建全局实例
image_utils = ImageUtils()
file_utils = FileUtils()