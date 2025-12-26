# -*-coding: utf-8 -*-
"""
@Author : pan-author
@E-mail : 390737991@qq.com
@Date : 2020-04-10 20:24:06
"""
import os
import cv2
import argparse
import numpy as np
from core.utils import image_utils, file_utils
from core import camera_params, stereo_matcher


class StereoDepth(object):
    """双目测距"""

    def __init__(self, stereo_file, width=640, height=480, filter=True, use_open3d=True):
        """
        :param stereo_file: 双目相机内外参数配置文件
        :param width: 相机分辨率width
        :param height: 相机分辨率height
        :param filter: 是否使用WLS滤波器对视差图进行滤波
        :param use_open3d: 是否使用open3d显示点云
        """
        self.count = 0
        self.filter = filter
        self.camera_config = camera_params.get_stereo_coefficients(stereo_file)
        self.use_open3d = use_open3d

        # 初始化3D点云
        if self.use_open3d:
            try:
                import open3d as o3d
                self.open3d_available = True
                print("Open3D is available for point cloud visualization")
            except ImportError:
                self.open3d_available = False
                print("Open3D not available, skipping point cloud visualization")

        assert (width, height) == self.camera_config["size"], Exception("Error:{}".format(self.camera_config["size"]))

    def test_pair_image_file(self, left_file, right_file):
        """
        测试一对左右图像
        :param left_file: 左路图像文件
        :param right_file: 右路图像文件
        :return:
        """
        frameL = cv2.imread(left_file)
        frameR = cv2.imread(right_file)
        if frameL is None or frameR is None:
            print(f"Cannot load images: {left_file}, {right_file}")
            return
        self.task(frameL, frameR, waitKey=0)

    def capture1(self, video):
        """
        用于采集单USB连接线的双目摄像头(左右摄像头被拼接在同一个视频中显示)
        :param video: int or str,视频路径或者摄像头ID
        """
        cap = image_utils.get_video_capture(video)
        width, height, numFrames, fps = image_utils.get_video_info(cap)
        self.count = 0

        while True:
            success, frame = cap.read()
            if not success:
                print("No more frames")
                break

            frameL = frame[:, :int(width / 2), :]
            frameR = frame[:, int(width / 2):, :]
            self.count += 1
            self.task(frameL, frameR, waitKey=5)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def capture2(self, left_video, right_video):
        """
        用于采集双USB连接线的双目摄像头
        :param left_video: int or str,左路视频路径或者摄像头ID
        :param right_video: int or str,右视频路径或者摄像头ID
        :return:
        """
        capL = image_utils.get_video_capture(left_video)
        capR = image_utils.get_video_capture(right_video)
        self.count = 0

        while True:
            successL, frameL = capL.read()
            successR, frameR = capR.read()
            if not (successL and successR):
                print("No more frames")
                break

            self.count += 1
            self.task(frameL, frameR, waitKey=30)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capL.release()
        capR.release()
        cv2.destroyAllWindows()

    def get_3dpoints(self, disparity, Q, scale=1.0):
        """
        计算像素点的3D坐标（左相机坐标系下）
        :param disparity: 视差图
        :param Q: 重投影矩阵
        :param scale: 单位变换尺度,默认scale=1.0,单位为毫米
        :return points_3d: 返回三维坐标points_3d，三个通道分布表示(X,Y,Z)
        """
        # 返回三维坐标points_3d，三个通道分布表示(X,Y,Z)
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        points_3d = points_3d * scale
        points_3d = np.asarray(points_3d, dtype=np.float32)
        return points_3d

    def get_disparity(self, imgL, imgR, use_wls=True):
        """
        :param imgL: 畸变校正和立体校正后的左视图
        :param imgR: 畸变校正和立体校正后的右视图
        :param use_wls: 是否使用WLS滤波器对视差图进行滤波
        :return dispL: 返回视差图
        """
        if use_wls:
            dispL = stereo_matcher.get_filter_disparity(imgL, imgR, use_wls=True)
        else:
            dispL = stereo_matcher.get_simple_disparity(imgL, imgR)
        return dispL

    def get_rectify_image(self, imgL, imgR):
        """
        畸变校正和立体校正
        :param imgL: 左图像
        :param imgR: 右图像
        :return: 校正后的左右图像
        """
        left_map_x, left_map_y = self.camera_config["left_map_x"], self.camera_config["left_map_y"]
        right_map_x, right_map_y = self.camera_config["right_map_x"], self.camera_config["right_map_y"]

        rectifiedL = cv2.remap(imgL, left_map_x, left_map_y, cv2.INTER_LINEAR, borderValue=cv2.BORDER_CONSTANT)
        rectifiedR = cv2.remap(imgR, right_map_x, right_map_y, cv2.INTER_LINEAR, borderValue=cv2.BORDER_CONSTANT)

        return rectifiedL, rectifiedR

    def task(self, frameL, frameR, waitKey=5):
        """
        主要处理任务
        :param frameL: 左路视频帧图像(BGR)
        :param frameR: 右路视频帧图像(BGR)
        """
        # 畸变校正和立体校正
        rectifiedL, rectifiedR = self.get_rectify_image(imgL=frameL, imgR=frameR)

        # 转换为灰度图
        grayL = cv2.cvtColor(rectifiedL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(rectifiedR, cv2.COLOR_BGR2GRAY)

        # 获取视差图
        dispL = self.get_disparity(grayL, grayR, self.filter)

        # 计算3D坐标
        points_3d = self.get_3dpoints(disparity=dispL, Q=self.camera_config["Q"])

        # 显示点云
        self.show_3dcloud_for_open3d(frameL, frameR, points_3d)

        # 显示2D图像
        self.show_2dimage(frameL, frameR, points_3d, dispL, waitKey=waitKey)

    def show_3dcloud_for_open3d(self, frameL, frameR, points_3d):
        """
        使用open3d显示点云
        :param frameL: 左图像
        :param frameR: 右图像
        :param points_3d: 3D坐标
        :return:
        """
        if self.use_open3d and self.open3d_available:
            try:
                import open3d as o3d

                # 提取深度图
                x, y, depth = cv2.split(points_3d)

                # 创建点云
                height, width = depth.shape
                color_image = cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB)

                # 创建有效点的掩码
                valid_mask = (depth > 0) & (depth < 5000)  # 限制深度范围

                if np.any(valid_mask):
                    # 获取有效点的坐标和颜色
                    valid_points = points_3d[valid_mask]
                    valid_colors = color_image[valid_mask] / 255.0

                    # 创建Open3D点云对象
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(valid_points)
                    pcd.colors = o3d.utility.Vector3dVector(valid_colors)

                    # 可视化点云（非阻塞）
                    if not hasattr(self, 'vis'):
                        self.vis = o3d.visualization.Visualizer()
                        self.vis.create_window("Point Cloud", width=800, height=600)
                        self.vis.add_geometry(pcd)
                    else:
                        self.vis.update_geometry(pcd)

                    self.vis.poll_events()
                    self.vis.update_renderer()

            except Exception as e:
                print(f"Open3D visualization error: {e}")

    def show_2dimage(self, frameL, frameR, points_3d, dispL, waitKey=0):
        """
        显示2D图像和相关信息
        :param frameL: 左图像
        :param frameR: 右图像
        :param points_3d: 3D坐标
        :param dispL: 视差图
        :return:
        """
        x, y, depth = cv2.split(points_3d)

        # 生成可视化图像
        depth_colormap = stereo_matcher.get_visual_depth(depth)
        dispL_colormap = stereo_matcher.get_visual_disparity(dispL)

        # 添加鼠标回调
        image_utils.addMouseCallback("left", points_3d, info="world coords=(x,y,depth)")
        image_utils.addMouseCallback("right", points_3d, info="world coords=(x,y,depth)")
        image_utils.addMouseCallback("disparity-color", points_3d, info="world coords=(x,y,depth)")
        image_utils.addMouseCallback("depth-color", points_3d, info="world coords=(x,y,depth)")

        # 显示图像
        cv2.imshow('left', frameL)
        cv2.imshow('right', frameR)
        cv2.imshow('disparity-color', dispL_colormap)
        cv2.imshow('depth-color', depth_colormap)

        key = cv2.waitKey(waitKey)
        self.save_images({
            "frameL": frameL,
            "frameR": frameR,
            "disparity": dispL_colormap,
            "depth": depth_colormap
        }, self.count, key)

        # 调整窗口位置（仅第一次）
        if self.count <= 1:
            cv2.moveWindow("left", 100, 0)
            cv2.moveWindow("right", 750, 0)
            cv2.moveWindow("disparity-color", 100, 400)
            cv2.moveWindow("depth-color", 750, 400)

    def save_images(self, result, count, key, save_dir="./data/temp"):
        """
        保存图像
        :param result: 结果字典
        :param count: 计数
        :param key: 按键
        :param save_dir: 保存目录
        :return:
        """
        if key == ord('q'):
            exit(0)
        elif key == ord('c') or key == ord('s'):
            file_utils.create_dir(save_dir)
            print("save image:{:0=4d}".format(count))
            cv2.imwrite(os.path.join(save_dir, "left_{:0=4d}.png".format(count)), result["frameL"])
            cv2.imwrite(os.path.join(save_dir, "right_{:0=4d}.png".format(count)), result["frameR"])
            cv2.imwrite(os.path.join(save_dir, "disparity_{:0=4d}.png".format(count)), result["disparity"])
            cv2.imwrite(os.path.join(save_dir, "depth_{:0=4d}.png".format(count)), result["depth"])


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', 'y', '1')


def get_parser():
    stereo_file = "configs/stereo_cam.yml"
    left_video = None
    right_video = None
    left_file = "data/left.png"
    right_file = "data/right.png"

    parser = argparse.ArgumentParser(description='Stereo Vision Demo')
    parser.add_argument('--stereo_file', type=str, default=stereo_file, help='stereo calibration file')
    parser.add_argument('--left_video', default=left_video, help='left video file or camera ID')
    parser.add_argument('--right_video', default=right_video, help='right video file or camera ID')
    parser.add_argument('--left_file', type=str, default=left_file, help='left image file')
    parser.add_argument('--right_file', type=str, default=right_file, help='right image file')
    parser.add_argument('--filter', type=str2bool, nargs='?', default=True, help='use disparity filter')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    print("args={}".format(args))

    stereo = StereoDepth(args.stereo_file, filter=args.filter)

    if args.left_video is not None and args.right_video is not None:
        # 双USB连接线的双目摄像头
        stereo.capture2(left_video=args.left_video, right_video=args.right_video)
    elif args.left_video is not None:
        # 单USB连接线的双目摄像头
        stereo.capture1(video=args.left_video)
    elif args.right_video is not None:
        # 单USB连接线的双目摄像头
        stereo.capture1(video=args.right_video)

    if args.left_file and args.right_file:
        # 测试一对左右图像
        stereo.test_pair_image_file(args.left_file, args.right_file)