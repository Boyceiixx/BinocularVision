# -*-coding: utf-8 -*-
"""
双目三维重建系统完整演示
包含双目标定、立体校正、视差计算、深度估计等功能
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from core import camera_params, stereo_matcher


class StereoVisionDemo:
    """双目视觉演示系统"""

    def __init__(self, config_file="configs/stereo_cam.yml"):
        """
        初始化双目视觉系统
        :param config_file: 双目相机配置文件路径
        """
        self.config_file = config_file
        self.camera_config = None

        # 如果配置文件存在，加载配置
        if os.path.exists(config_file):
            try:
                self.camera_config = camera_params.get_stereo_coefficients(config_file)
                print("✓ 双目相机配置加载成功")
            except Exception as e:
                print(f"⚠ 配置文件加载失败: {e}")
                self.create_default_config()
        else:
            print("⚠ 配置文件不存在，创建默认配置")
            self.create_default_config()

    def create_default_config(self):
        """创建默认的双目相机配置"""
        # 创建默认的相机参数
        image_size = (640, 480)

        # 左相机内参
        K1 = np.array([[535.91, 0, 342.28],
                       [0, 535.91, 235.05],
                       [0, 0, 1]], dtype=np.float64)

        # 右相机内参
        K2 = np.array([[535.91, 0, 342.28],
                       [0, 535.91, 235.05],
                       [0, 0, 1]], dtype=np.float64)

        # 畸变系数
        D1 = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float64)
        D2 = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float64)

        # 旋转和平移
        R = np.eye(3, dtype=np.float64)
        T = np.array([-60.0, 0, 0], dtype=np.float64)  # 基线60mm

        # 计算立体校正参数
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )

        # 计算重映射矩阵
        left_map_x, left_map_y = cv2.initUndistortRectifyMap(
            K1, D1, R1, P1, image_size, cv2.CV_32FC1
        )
        right_map_x, right_map_y = cv2.initUndistortRectifyMap(
            K2, D2, R2, P2, image_size, cv2.CV_32FC1
        )

        self.camera_config = {
            'size': image_size,
            'K1': K1, 'D1': D1,
            'K2': K2, 'D2': D2,
            'R': R, 'T': T,
            'R1': R1, 'R2': R2,
            'P1': P1, 'P2': P2,
            'Q': Q,
            'left_map_x': left_map_x, 'left_map_y': left_map_y,
            'right_map_x': right_map_x, 'right_map_y': right_map_y
        }

        print("✓ 默认双目相机配置创建成功")

    def rectify_images(self, left_img, right_img):
        """
        立体校正
        :param left_img: 左图像
        :param right_img: 右图像
        :return: 校正后的左右图像
        """
        left_map_x = self.camera_config['left_map_x']
        left_map_y = self.camera_config['left_map_y']
        right_map_x = self.camera_config['right_map_x']
        right_map_y = self.camera_config['right_map_y']

        rectified_left = cv2.remap(left_img, left_map_x, left_map_y,
                                   cv2.INTER_LINEAR, borderValue=cv2.BORDER_CONSTANT)
        rectified_right = cv2.remap(right_img, right_map_x, right_map_y,
                                    cv2.INTER_LINEAR, borderValue=cv2.BORDER_CONSTANT)

        return rectified_left, rectified_right

    def compute_disparity(self, left_img, right_img, use_filter=True):
        """
        计算视差图
        :param left_img: 左图像（灰度）
        :param right_img: 右图像（灰度）
        :param use_filter: 是否使用WLS滤波
        :return: 视差图
        """
        return stereo_matcher.get_filter_disparity(left_img, right_img, use_wls=use_filter)

    def compute_depth(self, disparity):
        """
        计算深度图
        :param disparity: 视差图
        :return: 深度图
        """
        Q = self.camera_config['Q']
        return stereo_matcher.get_depth(disparity, Q, scale=1.0)

    def compute_3d_points(self, disparity):
        """
        计算3D点云
        :param disparity: 视差图
        :return: 3D点坐标
        """
        Q = self.camera_config['Q']
        points_3d = cv2.reprojectImageTo3D(disparity, Q)
        return points_3d

    def visualize_results(self, left_img, right_img, disparity, depth, save_path="results"):
        """
        可视化结果
        """
        os.makedirs(save_path, exist_ok=True)

        # 创建可视化图像
        disparity_vis = stereo_matcher.get_visual_disparity(disparity)
        depth_vis = stereo_matcher.get_visual_depth(depth)

        # 保存结果
        cv2.imwrite(f"{save_path}/left_rectified.png", left_img)
        cv2.imwrite(f"{save_path}/right_rectified.png", right_img)
        cv2.imwrite(f"{save_path}/disparity.png", disparity_vis)
        cv2.imwrite(f"{save_path}/depth.png", depth_vis)

        # 创建对比图
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('双目三维重建结果', fontsize=16)

        axes[0, 0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('左相机图像')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('右相机图像')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(disparity_vis, cmap='jet')
        axes[1, 0].set_title('视差图')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(depth_vis, cmap='jet')
        axes[1, 1].set_title('深度图')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(f"{save_path}/comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ 结果已保存到 {save_path} 目录")

    def process_stereo_pair(self, left_path, right_path):
        """
        处理双目图像对
        :param left_path: 左图像路径
        :param right_path: 右图像路径
        """
        print(f"处理双目图像对: {left_path}, {right_path}")

        # 读取图像
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)

        if left_img is None or right_img is None:
            print("❌ 图像读取失败")
            return

        print(f"✓ 图像尺寸: {left_img.shape}")

        # 立体校正
        print("🔄 执行立体校正...")
        rectified_left, rectified_right = self.rectify_images(left_img, right_img)

        # 转换为灰度图
        gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

        # 计算视差
        print("🔄 计算视差图...")
        disparity = self.compute_disparity(gray_left, gray_right, use_filter=True)

        # 计算深度
        print("🔄 计算深度图...")
        depth = self.compute_depth(disparity)

        # 计算3D点云
        print("🔄 计算3D点云...")
        points_3d = self.compute_3d_points(disparity)

        # 可视化结果
        print("🔄 生成可视化结果...")
        self.visualize_results(rectified_left, rectified_right, disparity, depth)

        # 输出统计信息
        valid_disparity = disparity[disparity > 0]
        valid_depth = depth[depth > 0]

        print("\n📊 处理结果统计:")
        print(f"   有效视差点数: {len(valid_disparity)}")
        print(f"   视差范围: {valid_disparity.min():.1f} - {valid_disparity.max():.1f}")
        print(f"   深度范围: {valid_depth.min():.1f} - {valid_depth.max():.1f} mm")
        print(f"   平均深度: {valid_depth.mean():.1f} mm")

        return {
            'rectified_left': rectified_left,
            'rectified_right': rectified_right,
            'disparity': disparity,
            'depth': depth,
            'points_3d': points_3d
        }

    def demo_calibration_process(self):
        """演示标定过程"""
        print("\n🎯 双目相机标定演示")
        print("=" * 50)

        # 显示标定步骤
        steps = [
            "1. 准备标定板（棋盘格）",
            "2. 采集左右相机图像对",
            "3. 检测棋盘格角点",
            "4. 单目相机标定",
            "5. 双目相机标定",
            "6. 立体校正参数计算"
        ]

        for step in steps:
            print(f"   {step}")

        print("\n✓ 当前系统已完成标定，可直接进行三维重建")

    def demo_measurement(self, left_path, right_path, click_points=None):
        """演示测距功能"""
        print("\n📏 双目测距演示")
        print("=" * 50)

        # 处理图像
        result = self.process_stereo_pair(left_path, right_path)
        if result is None:
            return

        points_3d = result['points_3d']

        # 如果没有指定点击点，使用默认点
        if click_points is None:
            click_points = [(320, 240), (200, 150), (450, 300)]

        print("\n📍 测距结果:")
        for i, (x, y) in enumerate(click_points):
            if 0 <= x < points_3d.shape[1] and 0 <= y < points_3d.shape[0]:
                point_3d = points_3d[y, x]
                depth = point_3d[2]
                if depth > 0:
                    print(f"   点{i + 1} ({x}, {y}): 深度 = {depth:.1f} mm")
                else:
                    print(f"   点{i + 1} ({x}, {y}): 无效深度")
            else:
                print(f"   点{i + 1} ({x}, {y}): 坐标超出范围")


def main():
    """主函数"""
    print("🚀 双目三维重建系统演示")
    print("=" * 60)

    # 创建演示系统
    demo = StereoVisionDemo()

    # 演示标定过程
    demo.demo_calibration_process()

    # 检查演示图像
    left_path = "data/left.png"
    right_path = "data/right.png"

    if os.path.exists(left_path) and os.path.exists(right_path):
        print(f"\n🖼️ 使用演示图像: {left_path}, {right_path}")

        # 处理双目图像对
        result = demo.process_stereo_pair(left_path, right_path)

        # 演示测距功能
        demo.demo_measurement(left_path, right_path)

    else:
        print("❌ 演示图像不存在，请先运行 generate_demo_images.py")

    print("\n🎉 演示完成！")
    print("💡 提示: 运行 python web_demo.py 启动Web界面进行交互式体验")


if __name__ == "__main__":
    main()