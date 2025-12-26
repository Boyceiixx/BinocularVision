# -*-coding: utf-8 -*-
"""
双目三维重建系统Web演示界面
"""
import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
from core import camera_params, stereo_matcher
import json

app = Flask(__name__)


class StereoVisionDemo:
    def __init__(self):
        # 创建示例双目相机参数
        self.create_sample_stereo_params()
        self.last_results = None

    def create_sample_stereo_params(self):
        """创建示例双目相机参数"""
        # 模拟双目相机参数
        self.camera_config = {
            "size": (640, 480),
            "K1": np.array([[535.9, 0, 342.3], [0, 535.9, 235.0], [0, 0, 1]], dtype=np.float32),
            "D1": np.array([0.123, -0.239, -0.001, -0.001, 0.149], dtype=np.float32),
            "K2": np.array([[535.9, 0, 342.3], [0, 535.9, 235.0], [0, 0, 1]], dtype=np.float32),
            "D2": np.array([0.123, -0.239, -0.001, -0.001, 0.149], dtype=np.float32),
            "R": np.array([[0.9999, 0.0008, 0.0035], [-0.0008, 0.9999, 0.0027], [-0.0035, -0.0027, 0.9999]],
                          dtype=np.float32),
            "T": np.array([-60.0, 0.0, 0.0], dtype=np.float32),
            "Q": np.array([[1, 0, 0, -342.3], [0, 1, 0, -235.0], [0, 0, 0, 535.9], [0, 0, -0.0167, 0]],
                          dtype=np.float32)
        }

        # 计算立体校正映射
        self.setup_rectification()

    def setup_rectification(self):
        """设置立体校正映射"""
        size = self.camera_config["size"]
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            self.camera_config["K1"], self.camera_config["D1"],
            self.camera_config["K2"], self.camera_config["D2"],
            size, self.camera_config["R"], self.camera_config["T"]
        )

        # 计算映射矩阵
        left_map_x, left_map_y = cv2.initUndistortRectifyMap(
            self.camera_config["K1"], self.camera_config["D1"], R1, P1, size, cv2.CV_32FC1
        )
        right_map_x, right_map_y = cv2.initUndistortRectifyMap(
            self.camera_config["K2"], self.camera_config["D2"], R2, P2, size, cv2.CV_32FC1
        )

        self.camera_config.update({
            "left_map_x": left_map_x,
            "left_map_y": left_map_y,
            "right_map_x": right_map_x,
            "right_map_y": right_map_y,
            "Q": Q
        })

    def create_sample_stereo_images(self):
        """创建示例双目图像"""
        # 创建左图像（带有一些几何形状）
        left_img = np.zeros((480, 640, 3), dtype=np.uint8)
        left_img.fill(50)  # 深灰色背景

        # 添加一些几何形状到左图像
        cv2.rectangle(left_img, (100, 100), (200, 200), (0, 255, 0), -1)  # 绿色矩形
        cv2.circle(left_img, (400, 150), 50, (255, 0, 0), -1)  # 蓝色圆形
        cv2.rectangle(left_img, (300, 300), (500, 400), (0, 0, 255), -1)  # 红色矩形

        # 创建右图像（稍微偏移以模拟视差）
        right_img = np.zeros((480, 640, 3), dtype=np.uint8)
        right_img.fill(50)  # 深灰色背景

        # 添加偏移的几何形状到右图像
        offset = 10  # 视差偏移
        cv2.rectangle(right_img, (100 - offset, 100), (200 - offset, 200), (0, 255, 0), -1)
        cv2.circle(right_img, (400 - offset, 150), 50, (255, 0, 0), -1)
        cv2.rectangle(right_img, (300 - offset, 300), (500 - offset, 400), (0, 0, 255), -1)

        return left_img, right_img

    def process_stereo_images(self, left_img, right_img):
        """处理双目图像"""
        # 立体校正
        rectified_left = cv2.remap(left_img, self.camera_config["left_map_x"],
                                   self.camera_config["left_map_y"], cv2.INTER_LINEAR)
        rectified_right = cv2.remap(right_img, self.camera_config["right_map_x"],
                                    self.camera_config["right_map_y"], cv2.INTER_LINEAR)

        # 转换为灰度图
        gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

        # 计算视差图
        disparity = stereo_matcher.get_simple_disparity(gray_left, gray_right)

        # 计算深度图
        depth = stereo_matcher.get_depth(disparity, self.camera_config["Q"])

        # 生成可视化图像
        disparity_vis = stereo_matcher.get_visual_disparity(disparity)
        depth_vis = stereo_matcher.get_visual_depth(depth)

        return {
            'left': left_img,
            'right': right_img,
            'rectified_left': rectified_left,
            'rectified_right': rectified_right,
            'disparity': disparity_vis,
            'depth': depth_vis,
            'disparity_raw': disparity,
            'depth_raw': depth
        }

    def image_to_base64(self, image):
        """将图像转换为base64编码"""
        _, buffer = cv2.imencode('.png', image)
        img_str = base64.b64encode(buffer).decode()
        return f"data:image/png;base64,{img_str}"

    def update_last_results(self, results):
        """缓存最新处理结果"""
        self.last_results = results

    def get_or_create_results(self):
        """获取最新结果，没有则生成示例图像结果"""
        if self.last_results is None:
            left_img, right_img = self.create_sample_stereo_images()
            self.last_results = self.process_stereo_images(left_img, right_img)
        return self.last_results


# 创建全局演示实例
demo = StereoVisionDemo()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    """处理双目图像"""
    try:
        # 创建示例图像
        left_img, right_img = demo.create_sample_stereo_images()

        # 处理图像
        results = demo.process_stereo_images(left_img, right_img)
        demo.update_last_results(results)

        # 转换为base64
        response = {}
        for key, img in results.items():
            if key not in ['disparity_raw', 'depth_raw']:
                response[key] = demo.image_to_base64(img)

        return jsonify({
            'success': True,
            'images': response,
            'message': '双目三维重建处理完成！'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'处理失败: {str(e)}'
        })


@app.route('/upload_process', methods=['POST'])
def upload_process():
    """处理上传的双目图像"""
    try:
        left_file = request.files.get('left_image')
        right_file = request.files.get('right_image')

        if not left_file or not right_file:
            return jsonify({
                'success': False,
                'message': '请同时上传左图和右图'
            })

        left_img = decode_uploaded_image(left_file)
        right_img = decode_uploaded_image(right_file)

        if left_img is None or right_img is None:
            return jsonify({
                'success': False,
                'message': '上传的图像无法解析'
            })

        left_img, right_img = resize_stereo_pair(left_img, right_img, demo.camera_config["size"])

        results = demo.process_stereo_images(left_img, right_img)
        demo.update_last_results(results)

        response = {}
        for key, img in results.items():
            if key not in ['disparity_raw', 'depth_raw']:
                response[key] = demo.image_to_base64(img)

        return jsonify({
            'success': True,
            'images': response,
            'message': '上传图像处理完成！'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'处理失败: {str(e)}'
        })


@app.route('/get_depth', methods=['POST'])
def get_depth():
    """获取指定像素点的深度信息"""
    try:
        data = request.json
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))

        results = demo.get_or_create_results()

        # 获取深度值
        depth_raw = results['depth_raw']
        if 0 <= y < depth_raw.shape[0] and 0 <= x < depth_raw.shape[1]:
            depth_value = depth_raw[y, x]
            disparity_value = results['disparity_raw'][y, x]

            return jsonify({
                'success': True,
                'x': x,
                'y': y,
                'depth': float(depth_value),
                'disparity': float(disparity_value),
                'message': f'坐标({x}, {y})处的深度: {depth_value:.1f}mm'
            })
        else:
            return jsonify({
                'success': False,
                'message': '坐标超出图像范围'
            })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取深度失败: {str(e)}'
        })


def decode_uploaded_image(file_storage):
    """解析上传的图像文件"""
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    if file_bytes.size == 0:
        return None
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def resize_stereo_pair(left_img, right_img, target_size):
    """调整左右图像尺寸保持一致"""
    target_width, target_height = target_size
    if left_img.shape[:2] != (target_height, target_width):
        left_img = cv2.resize(left_img, (target_width, target_height))
    if right_img.shape[:2] != (target_height, target_width):
        right_img = cv2.resize(right_img, (target_width, target_height))
    return left_img, right_img


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5008, debug=True)
