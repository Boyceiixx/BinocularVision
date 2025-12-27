# -*-coding: utf-8 -*-
import cv2
import numpy as np


def get_stereo_coefficients(stereo_file):
    """
    读取双目相机标定参数
    :param stereo_file: 双目标定参数文件路径
    :return: 双目相机参数字典
    """
    # 读取双目标定参数
    fs = cv2.FileStorage(stereo_file, cv2.FILE_STORAGE_READ)

    # 基本参数
    size = tuple(fs.getNode("size").mat().astype(int).flatten())
    K1 = fs.getNode("K1").mat()
    D1 = fs.getNode("D1").mat()
    K2 = fs.getNode("K2").mat()
    D2 = fs.getNode("D2").mat()
    R = fs.getNode("R").mat()
    T = fs.getNode("T").mat()
    E = fs.getNode("E").mat()
    F = fs.getNode("F").mat()

    # 校正参数
    R1 = fs.getNode("R1").mat()
    R2 = fs.getNode("R2").mat()
    P1 = fs.getNode("P1").mat()
    P2 = fs.getNode("P2").mat()
    Q = fs.getNode("Q").mat()

    # 映射矩阵
    left_map_x = fs.getNode("left_map_x").mat()
    left_map_y = fs.getNode("left_map_y").mat()
    right_map_x = fs.getNode("right_map_x").mat()
    right_map_y = fs.getNode("right_map_y").mat()

    fs.release()

    camera_config = {
        "size": size,
        "K1": K1, "D1": D1, "K2": K2, "D2": D2,
        "R": R, "T": T, "E": E, "F": F,
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
        "left_map_x": left_map_x, "left_map_y": left_map_y,
        "right_map_x": right_map_x, "right_map_y": right_map_y
    }

    return camera_config


def get_rectify_transform(K1, D1, K2, D2, R, T, image_size):
    """
    计算立体校正的变换矩阵
    :param K1: 左相机内参矩阵
    :param D1: 左相机畸变系数
    :param K2: 右相机内参矩阵
    :param D2: 右相机畸变系数
    :param R: 旋转矩阵
    :param T: 平移向量
    :param image_size: 图像尺寸 (width, height)
    :return: 校正变换参数
    """
    # 立体校正
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T, alpha=0
    )

    # 计算校正映射
    left_map_x, left_map_y = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1, image_size, cv2.CV_32FC1
    )
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(
        K2, D2, R2, P2, image_size, cv2.CV_32FC1
    )

    return {
        'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2, 'Q': Q,
        'left_map_x': left_map_x, 'left_map_y': left_map_y,
        'right_map_x': right_map_x, 'right_map_y': right_map_y
    }


def save_camera_params(save_path, K, D):
    """
    保存相机内参和畸变系数
    :param save_path: 保存路径
    :param K: 相机内参矩阵
    :param D: 畸变系数
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fs = cv2.FileStorage(save_path, cv2.FILE_STORAGE_WRITE)
    fs.write("K", K)
    fs.write("D", D)
    fs.release()
    print(f"Camera parameters saved to {save_path}")


def load_camera_params(file_path):
    """
    加载相机参数
    :param file_path: 参数文件路径
    :return: 相机内参和畸变系数
    """
    fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("K").mat()
    D = fs.getNode("D").mat()
    fs.release()
    return K, D