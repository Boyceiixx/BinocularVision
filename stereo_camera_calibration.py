# -*-coding: utf-8 -*-
"""
@Author : pan-author
@E-mail : 390737991@qq.com
@Date : 2020-04-10 18:24:06
"""
import os
import cv2
import glob
import argparse
import numpy as np


class StereoCameraCalibration(object):
    """双目相机标定"""

    def __init__(self, width, height, square_size):
        """
        :param width: 棋盘格宽方向黑白格子相交点个数
        :param height: 棋盘格长方向黑白格子相交点个数
        :param square_size: 棋盘格每个方格的边长，单位为毫米
        """
        self.width = width
        self.height = height
        self.square_size = square_size

        # 设置寻找亚像素角点的参数
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # 世界坐标系中的棋盘格点
        self.objp = np.zeros((width * height, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
        self.objp = self.objp * square_size

        # 储存棋盘格角点的世界坐标和图像坐标对
        self.objpoints = []  # 在世界坐标系中的三维点
        self.imgpoints_l = []  # 左相机在图像平面的二维点
        self.imgpoints_r = []  # 右相机在图像平面的二维点

    def load_camera_params(self, left_file, right_file):
        """
        加载左右相机的内参和畸变系数
        :param left_file: 左相机参数文件
        :param right_file: 右相机参数文件
        :return: 左右相机内参和畸变系数
        """
        # 读取左相机参数
        fs_left = cv2.FileStorage(left_file, cv2.FILE_STORAGE_READ)
        K1 = fs_left.getNode("K").mat()
        D1 = fs_left.getNode("D").mat()
        fs_left.release()

        # 读取右相机参数
        fs_right = cv2.FileStorage(right_file, cv2.FILE_STORAGE_READ)
        K2 = fs_right.getNode("K").mat()
        D2 = fs_right.getNode("D").mat()
        fs_right.release()

        return K1, D1, K2, D2

    def detect_corners(self, image):
        """
        检测棋盘格角点
        :param image: 输入图像
        :return: 是否检测到角点，角点坐标
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 寻找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (self.width, self.height), None)

        # 如果找到足够点对，将其存储起来
        if ret:
            # 在原角点的基础上寻找亚像素角点
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            return ret, corners2
        else:
            return ret, None

    def stereo_calibration(self, left_dir, right_dir, left_prefix, right_prefix,
                           image_format, K1, D1, K2, D2, show=False):
        """
        双目相机标定
        :param left_dir: 左图像目录
        :param right_dir: 右图像目录
        :param left_prefix: 左图像前缀
        :param right_prefix: 右图像前缀
        :param image_format: 图像格式
        :param K1: 左相机内参
        :param D1: 左相机畸变系数
        :param K2: 右相机内参
        :param D2: 右相机畸变系数
        :param show: 是否显示检测结果
        :return: 双目标定结果
        """
        # 获取左右图像列表
        left_images = sorted(glob.glob(os.path.join(left_dir, f"{left_prefix}_*.{image_format}")))
        right_images = sorted(glob.glob(os.path.join(right_dir, f"{right_prefix}_*.{image_format}")))

        if len(left_images) == 0 or len(right_images) == 0:
            print(f"No images found in directories")
            return None

        print(f"Found {len(left_images)} left images and {len(right_images)} right images")

        # 确保左右图像数量一致
        min_images = min(len(left_images), len(right_images))
        left_images = left_images[:min_images]
        right_images = right_images[:min_images]

        img_shape = None

        for i, (left_fname, right_fname) in enumerate(zip(left_images, right_images)):
            img_left = cv2.imread(left_fname)
            img_right = cv2.imread(right_fname)

            if img_left is None or img_right is None:
                continue

            if img_shape is None:
                img_shape = img_left.shape[:2][::-1]  # (width, height)

            # 检测左右图像的角点
            ret_left, corners_left = self.detect_corners(img_left)
            ret_right, corners_right = self.detect_corners(img_right)

            if ret_left and ret_right:
                self.objpoints.append(self.objp)
                self.imgpoints_l.append(corners_left)
                self.imgpoints_r.append(corners_right)

                if show:
                    # 显示角点检测结果
                    cv2.drawChessboardCorners(img_left, (self.width, self.height), corners_left, ret_left)
                    cv2.drawChessboardCorners(img_right, (self.width, self.height), corners_right, ret_right)

                    combined = np.hstack((img_left, img_right))
                    cv2.imshow('Stereo Calibration', combined)
                    cv2.waitKey(500)
            else:
                print(f"Cannot find corners in pair {i}: {left_fname}, {right_fname}")

        if show:
            cv2.destroyAllWindows()

        if len(self.objpoints) == 0:
            print("No valid calibration image pairs found")
            return None

        print(f"Stereo calibration with {len(self.objpoints)} image pairs")

        # 双目标定
        flags = cv2.CALIB_FIX_INTRINSIC
        ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l, self.imgpoints_r,
            K1, D1, K2, D2, img_shape,
            criteria=self.criteria, flags=flags
        )

        print(f"Stereo calibration completed")
        print(f"RMS error: {ret}")

        # 立体校正
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            K1, D1, K2, D2, img_shape, R, T, alpha=0
        )

        # 计算校正映射
        left_map_x, left_map_y = cv2.initUndistortRectifyMap(
            K1, D1, R1, P1, img_shape, cv2.CV_32FC1
        )
        right_map_x, right_map_y = cv2.initUndistortRectifyMap(
            K2, D2, R2, P2, img_shape, cv2.CV_32FC1
        )

        return {
            'K1': K1, 'D1': D1, 'K2': K2, 'D2': D2,
            'R': R, 'T': T, 'E': E, 'F': F,
            'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2, 'Q': Q,
            'left_map_x': left_map_x, 'left_map_y': left_map_y,
            'right_map_x': right_map_x, 'right_map_y': right_map_y,
            'size': img_shape
        }

    def save_stereo_params(self, save_path, params):
        """
        保存双目相机参数
        :param save_path: 保存路径
        :param params: 双目标定参数
        """
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存参数
        fs = cv2.FileStorage(save_path, cv2.FILE_STORAGE_WRITE)

        # 基本参数
        fs.write("size", params['size'])
        fs.write("K1", params['K1'])
        fs.write("D1", params['D1'])
        fs.write("K2", params['K2'])
        fs.write("D2", params['D2'])
        fs.write("R", params['R'])
        fs.write("T", params['T'])
        fs.write("E", params['E'])
        fs.write("F", params['F'])

        # 校正参数
        fs.write("R1", params['R1'])
        fs.write("R2", params['R2'])
        fs.write("P1", params['P1'])
        fs.write("P2", params['P2'])
        fs.write("Q", params['Q'])

        # 映射矩阵
        fs.write("left_map_x", params['left_map_x'])
        fs.write("left_map_y", params['left_map_y'])
        fs.write("right_map_x", params['right_map_x'])
        fs.write("right_map_y", params['right_map_y'])

        fs.release()
        print(f"Stereo camera parameters saved to {save_path}")


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', 'y', '1')


def get_parser():
    parser = argparse.ArgumentParser(description='Stereo Camera calibration')
    parser.add_argument('--left_file', type=str, default='configs/camera/left_cam.yml',
                        help='left camera calibration file')
    parser.add_argument('--right_file', type=str, default='configs/camera/right_cam.yml',
                        help='right camera calibration file')
    parser.add_argument('--left_prefix', type=str, default='left', help='left image prefix')
    parser.add_argument('--right_prefix', type=str, default='right', help='right image prefix')
    parser.add_argument('--width', type=int, default=8, help='chessboard width size')
    parser.add_argument('--height', type=int, default=11, help='chessboard height size')
    parser.add_argument('--left_dir', type=str, default='data/camera', help='left image directory')
    parser.add_argument('--right_dir', type=str, default='data/camera', help='right image directory')
    parser.add_argument('--image_format', type=str, default='png', help='image format')
    parser.add_argument('--square_size', type=float, default=20.0, help='chessboard square size (mm)')
    parser.add_argument('--save_dir', type=str, default='configs/camera',
                        help='directory to save calibration results')
    parser.add_argument('--show', type=str2bool, nargs='?', const=True, default=False,
                        help='show detection results')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    print("args={}".format(args))

    # 创建双目标定对象
    calibrator = StereoCameraCalibration(args.width, args.height, args.square_size)

    # 加载左右相机参数
    try:
        K1, D1, K2, D2 = calibrator.load_camera_params(args.left_file, args.right_file)
        print("Loaded camera parameters successfully")
    except Exception as e:
        print(f"Error loading camera parameters: {e}")
        exit(1)

    # 执行双目标定
    params = calibrator.stereo_calibration(
        args.left_dir, args.right_dir, args.left_prefix, args.right_prefix,
        args.image_format, K1, D1, K2, D2, args.show
    )

    if params is not None:
        # 保存双目标定结果
        save_path = os.path.join(args.save_dir, "stereo_cam.yml")
        calibrator.save_stereo_params(save_path, params)

        print("Stereo calibration completed successfully!")
        print(f"Rotation matrix R:\n{params['R']}")
        print(f"Translation vector T:\n{params['T']}")
    else:
        print("Stereo calibration failed!")