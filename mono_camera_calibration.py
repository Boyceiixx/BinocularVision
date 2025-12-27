# -*-coding: utf-8 -*-
import os
import cv2
import glob
import argparse
import numpy as np


class MonoCameraCalibration(object):
    """单目相机标定"""

    def __init__(self, width, height, square_size):
        """
        :param width: 棋盘格宽方向黑白格子相交点个数
        :param height: 棋盘格长方向黑白格子相交点个数
        :param square_size: 棋盘格每个方格的边长，单位为毫米
        """
        self.width = width
        self.height = height
        self.square_size = square_size

        # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
        self.objp = np.zeros((width * height, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
        self.objp = self.objp * square_size  # 20为方格边长

        # 储存棋盘格角点的世界坐标和图像坐标对
        self.objpoints = []  # 在世界坐标系中的三维点
        self.imgpoints = []  # 在图像平面的二维点

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

    def calibration(self, image_dir, image_format, prefix="", show=False):
        """
        相机标定
        :param image_dir: 图像目录
        :param image_format: 图像格式
        :param prefix: 图像前缀
        :param show: 是否显示检测结果
        :return: 相机内参矩阵，畸变系数，旋转向量，平移向量
        """
        # 获取图像列表
        if prefix:
            images = glob.glob(os.path.join(image_dir, f"{prefix}_*.{image_format}"))
        else:
            images = glob.glob(os.path.join(image_dir, f"*.{image_format}"))

        if len(images) == 0:
            print(f"No images found in {image_dir} with format {image_format}")
            return None, None, None, None

        print(f"Found {len(images)} images for calibration")

        for fname in images:
            img = cv2.imread(fname)
            if img is None:
                continue

            ret, corners = self.detect_corners(img)

            if ret:
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)

                if show:
                    # 将角点在图像上显示
                    cv2.drawChessboardCorners(img, (self.width, self.height), corners, ret)
                    cv2.imshow('findCorners', img)
                    cv2.waitKey(500)
            else:
                print(f"Cannot find corners in {fname}")

        if show:
            cv2.destroyAllWindows()

        if len(self.objpoints) == 0:
            print("No valid calibration images found")
            return None, None, None, None

        # 标定
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints,
                                                           gray.shape[::-1], None, None)

        print(f"Calibration completed with {len(self.objpoints)} images")
        print(f"RMS error: {ret}")

        return mtx, dist, rvecs, tvecs

    def save_params(self, save_path, mtx, dist):
        """
        保存相机参数
        :param save_path: 保存路径
        :param mtx: 相机内参矩阵
        :param dist: 畸变系数
        """
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存参数
        fs = cv2.FileStorage(save_path, cv2.FILE_STORAGE_WRITE)
        fs.write("K", mtx)
        fs.write("D", dist)
        fs.release()
        print(f"Camera parameters saved to {save_path}")


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', 'y', '1')


def get_parser():
    parser = argparse.ArgumentParser(description='Mono Camera calibration')
    parser.add_argument('--image_dir', type=str, default='data/camera', help='image directory')
    parser.add_argument('--image_format', type=str, default='png', help='image format')
    parser.add_argument('--square_size', type=float, default=20.0, help='chessboard square size (mm)')
    parser.add_argument('--width', type=int, default=8, help='chessboard width size')
    parser.add_argument('--height', type=int, default=11, help='chessboard height size')
    parser.add_argument('--prefix', type=str, default='left', help='image prefix')
    parser.add_argument('--save_dir', type=str, default='configs/camera', help='directory to save calibration results')
    parser.add_argument('--show', type=str2bool, nargs='?', const=True, default=True, help='show detection results')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    print("args={}".format(args))

    # 创建标定对象
    calibrator = MonoCameraCalibration(args.width, args.height, args.square_size)

    # 执行标定
    mtx, dist, rvecs, tvecs = calibrator.calibration(
        args.image_dir, args.image_format, args.prefix, args.show
    )

    if mtx is not None:
        # 保存标定结果
        save_path = os.path.join(args.save_dir, f"{args.prefix}_cam.yml")
        calibrator.save_params(save_path, mtx, dist)

        print("Camera matrix:")
        print(mtx)
        print("Distortion coefficients:")
        print(dist)
    else:
        print("Calibration failed!")