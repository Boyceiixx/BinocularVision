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

    def __init__(self, stereo_file, width=640, height=480, filter=True, use_open3d=True,
                 enable_fusion=False):
        """
        :param stereo_file: 双目相机内外参数配置文件
        :param width: 相机分辨率width
        :param height: 相机分辨率height
        :param filter: 是否使用WLS滤波器对视差图进行滤波
        :param use_open3d: 是否使用open3d显示点云
        :param enable_fusion: 是否开启多视角点云融合(需要Open3D)
        """
        self.count = 0
        self.filter = filter
        self.camera_config = camera_params.get_stereo_coefficients(stereo_file)
        self.use_open3d = use_open3d
        self.enable_fusion = enable_fusion

        self.roi_start = None
        self.roi_end = None
        self.roi_box = None
        self.roi_dragging = False
        self.auto_roi = True
        self.latest_points_3d = None

        # 初始化3D点云
        if self.use_open3d:
            try:
                import open3d as o3d
                self.open3d_available = True
                print("Open3D is available for point cloud visualization")
            except ImportError:
                self.open3d_available = False
                print("Open3D not available, skipping point cloud visualization")
        else:
            self.open3d_available = False

        self.fused_pcd = None
        self.last_pcd = None

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

        # ROI过滤并计算包围盒
        roi_mask = self.get_roi_mask(points_3d)
        bbox_info = self.get_bounding_box(points_3d, roi_mask)

        # 显示点云
        self.show_3dcloud_for_open3d(frameL, frameR, points_3d, roi_mask, bbox_info)

        # 显示2D图像
        self.show_2dimage(frameL, frameR, points_3d, dispL, bbox_info, waitKey=waitKey)

    def show_3dcloud_for_open3d(self, frameL, frameR, points_3d, roi_mask, bbox_info):
        """
        使用open3d显示点云
        :param frameL: 左图像
        :param frameR: 右图像
        :param points_3d: 3D坐标
        :param roi_mask: ROI mask
        :param bbox_info: 包围盒信息
        :return:
        """
        if self.use_open3d and self.open3d_available:
            try:
                import open3d as o3d

                color_image = cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB)

                # 创建有效点的掩码
                depth = points_3d[:, :, 2]
                valid_mask = (depth > 0) & (depth < 5000)  # 限制深度范围
                if roi_mask is not None:
                    valid_mask = valid_mask & roi_mask

                if np.any(valid_mask):
                    # 获取有效点的坐标和颜色
                    valid_points = points_3d[valid_mask]
                    valid_colors = color_image[valid_mask] / 255.0

                    # 创建Open3D点云对象
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(valid_points)
                    pcd.colors = o3d.utility.Vector3dVector(valid_colors)

                    if self.enable_fusion:
                        pcd = self.fuse_point_cloud(pcd)

                    geometries = [pcd]
                    if bbox_info and bbox_info.get("obb") is not None:
                        geometries.append(bbox_info["obb"])
                    elif bbox_info and bbox_info.get("aabb") is not None:
                        geometries.append(bbox_info["aabb"])

                    # 可视化点云（非阻塞）
                    if not hasattr(self, 'vis'):
                        self.vis = o3d.visualization.Visualizer()
                        self.vis.create_window("Point Cloud", width=800, height=600)
                        for geo in geometries:
                            self.vis.add_geometry(geo)
                    else:
                        self.vis.clear_geometries()
                        for geo in geometries:
                            self.vis.add_geometry(geo)

                    self.vis.poll_events()
                    self.vis.update_renderer()

            except Exception as e:
                print(f"Open3D visualization error: {e}")

    def show_2dimage(self, frameL, frameR, points_3d, dispL, bbox_info, waitKey=0):
        """
        显示2D图像和相关信息
        :param frameL: 左图像
        :param frameR: 右图像
        :param points_3d: 3D坐标
        :param dispL: 视差图
        :param bbox_info: 包围盒信息
        :return:
        """
        x, y, depth = cv2.split(points_3d)

        # 生成可视化图像
        depth_colormap = stereo_matcher.get_visual_depth(depth)
        dispL_colormap = stereo_matcher.get_visual_disparity(dispL)

        # 添加鼠标回调
        self.latest_points_3d = points_3d
        self.set_roi_mouse_callback("left")
        image_utils.addMouseCallback("right", points_3d, info="world coords=(x,y,depth)")
        image_utils.addMouseCallback("disparity-color", points_3d, info="world coords=(x,y,depth)")
        image_utils.addMouseCallback("depth-color", points_3d, info="world coords=(x,y,depth)")

        # 显示图像
        left_vis = frameL.copy()
        self.draw_roi(left_vis)
        self.draw_bbox_text(left_vis, bbox_info)

        cv2.imshow('left', left_vis)
        cv2.imshow('right', frameR)
        cv2.imshow('disparity-color', dispL_colormap)
        cv2.imshow('depth-color', depth_colormap)

        key = cv2.waitKey(waitKey)
        self.save_images({
            "frameL": left_vis,
            "frameR": frameR,
            "disparity": dispL_colormap,
            "depth": depth_colormap
        }, self.count, key)

        if key == ord('r'):
            self.reset_roi()

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

    def set_roi_mouse_callback(self, window_name):
        """
        鼠标框选ROI
        """

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.roi_dragging = True
                self.roi_start = (x, y)
                self.roi_end = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.roi_dragging:
                self.roi_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self.roi_dragging = False
                self.roi_end = (x, y)
                self.roi_box = self.normalize_box(self.roi_start, self.roi_end)
                self.auto_roi = False
                print(f"ROI选框: {self.roi_box} (按 r 重置为自动ROI)")
            elif event == cv2.EVENT_RBUTTONDOWN:
                if self.latest_points_3d is None:
                    return
                if 0 <= y < self.latest_points_3d.shape[0] and 0 <= x < self.latest_points_3d.shape[1]:
                    world_x = self.latest_points_3d[y, x, 0]
                    world_y = self.latest_points_3d[y, x, 1]
                    world_z = self.latest_points_3d[y, x, 2]
                    print(
                        f"world coords=(x={world_x:.1f}, y={world_y:.1f}, depth={world_z:.1f})mm")

        cv2.setMouseCallback(window_name, mouse_callback)

    def normalize_box(self, start, end):
        if start is None or end is None:
            return None
        x1, y1 = start
        x2, y2 = end
        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)
        if left == right or top == bottom:
            return None
        return (left, top, right, bottom)

    def get_roi_mask(self, points_3d):
        depth = points_3d[:, :, 2]
        valid = (depth > 0) & (depth < 5000) & np.isfinite(depth)
        if self.roi_box is not None:
            left, top, right, bottom = self.roi_box
            roi_mask = np.zeros(depth.shape, dtype=bool)
            roi_mask[top:bottom, left:right] = True
            return valid & roi_mask
        if not self.auto_roi:
            return None
        return self.auto_roi_mask(depth, valid)

    def auto_roi_mask(self, depth, valid_mask):
        height, width = depth.shape
        center_box = (
            int(width * 0.3), int(height * 0.3),
            int(width * 0.7), int(height * 0.7)
        )
        left, top, right, bottom = center_box
        center_depth = depth[top:bottom, left:right]
        center_valid = valid_mask[top:bottom, left:right]
        if not np.any(center_valid):
            return None
        median_depth = np.median(center_depth[center_valid])
        if not np.isfinite(median_depth) or median_depth <= 0:
            return None
        depth_min = median_depth * 0.85
        depth_max = median_depth * 1.15
        return valid_mask & (depth >= depth_min) & (depth <= depth_max)

    def get_bounding_box(self, points_3d, roi_mask):
        if roi_mask is None or not np.any(roi_mask):
            return None
        roi_points = points_3d[roi_mask]
        roi_points = roi_points[np.all(np.isfinite(roi_points), axis=1)]
        if roi_points.size == 0:
            return None

        bbox_info = {}
        min_pt = roi_points.min(axis=0)
        max_pt = roi_points.max(axis=0)
        size = max_pt - min_pt
        bbox_info["aabb_size"] = size

        if self.open3d_available:
            try:
                import open3d as o3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(roi_points)
                aabb = pcd.get_axis_aligned_bounding_box()
                aabb.color = (0.0, 1.0, 0.0)
                bbox_info["aabb"] = aabb

                obb = pcd.get_oriented_bounding_box()
                obb.color = (1.0, 0.0, 0.0)
                bbox_info["obb"] = obb
                bbox_info["obb_size"] = obb.extent
            except Exception as e:
                print(f"Open3D bounding box error: {e}")
        return bbox_info

    def draw_roi(self, image):
        if self.roi_start and self.roi_end:
            box = self.normalize_box(self.roi_start, self.roi_end)
            if box:
                left, top, right, bottom = box
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 255), 2)
        elif self.roi_box:
            left, top, right, bottom = self.roi_box
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 255), 2)

    def draw_bbox_text(self, image, bbox_info):
        if not bbox_info:
            return
        if bbox_info.get("obb_size") is not None:
            size = bbox_info["obb_size"]
            label = "OBB"
        else:
            size = bbox_info.get("aabb_size")
            label = "AABB"
        if size is None:
            return
        length, width, height = size
        text = f"{label} L/W/H: {length:.1f}/{width:.1f}/{height:.1f} mm"
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

    def reset_roi(self):
        self.roi_start = None
        self.roi_end = None
        self.roi_box = None
        self.auto_roi = True
        print("ROI已重置为自动分割")

    def fuse_point_cloud(self, pcd):
        if not self.open3d_available:
            return pcd
        try:
            import open3d as o3d
            pcd = pcd.voxel_down_sample(voxel_size=5.0)
            if self.fused_pcd is None:
                self.fused_pcd = pcd
                self.last_pcd = pcd
                return self.fused_pcd

            threshold = 30.0
            result = o3d.pipelines.registration.registration_icp(
                pcd, self.last_pcd, threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            pcd.transform(result.transformation)
            self.fused_pcd += pcd
            self.last_pcd = pcd
            return self.fused_pcd
        except Exception as e:
            print(f"Point cloud fusion error: {e}")
            return pcd


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
    parser.add_argument('--multi_view', type=str2bool, nargs='?', default=False,
                        help='enable multi-view fusion (requires Open3D)')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    print("args={}".format(args))

    stereo = StereoDepth(args.stereo_file, filter=args.filter, enable_fusion=args.multi_view)

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
