# -*-coding: utf-8 -*-
"""
双目三维重建系统Web演示界面
"""
import os
import re
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify, Response
from core import camera_params, stereo_matcher
from camera_stream import get_camera_frames, build_stream_url
from xiaomi_stream import get_xiaomi_frames
import json

app = Flask(__name__)

CAMERA_CONFIG = {
    "ip": "172.16.0.108",
    "port": 554,
    "username": "",
    "password": "",
    "protocol": "rtsp",
    "stream_path": "11",
    "width": None,
    "height": None,
}


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


class MiddleburyDemo:
    """Middlebury 2006 双目数据集演示（2 views）"""

    def __init__(self, data_root="data"):
        preferred_root = os.path.join(data_root, "two-views")
        self.data_root = preferred_root if os.path.isdir(preferred_root) else data_root
        self.last_results = {}
        self.default_dmin = 0
        self.default_ndisp = 256
        self.default_baseline = 0.10
        self.default_focal = 1.0

    def list_scenes(self):
        """查找包含 view1.png/view5.png 的场景目录"""
        scenes = []
        if not os.path.isdir(self.data_root):
            return scenes
        for entry in sorted(os.listdir(self.data_root)):
            scene_dir = os.path.join(self.data_root, entry)
            if not os.path.isdir(scene_dir):
                continue
            left_path = os.path.join(scene_dir, "view1.png")
            right_path = os.path.join(scene_dir, "view5.png")
            if os.path.exists(left_path) and os.path.exists(right_path):
                scenes.append(entry)
        return scenes

    def parse_calibration(self, calib_path):
        """解析 Middlebury calib.txt"""
        if not os.path.exists(calib_path):
            return None
        calib = {}
        with open(calib_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = [part.strip() for part in line.split("=", 1)]
                if value.startswith("[") and value.endswith("]"):
                    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
                    data = np.array([float(n) for n in nums], dtype=np.float32)
                    if data.size == 9:
                        calib[key] = data.reshape(3, 3)
                    else:
                        calib[key] = data
                else:
                    try:
                        calib[key] = float(value)
                    except ValueError:
                        calib[key] = value
        return calib

    def get_scene_paths(self, scene_name):
        scene_dir = os.path.normpath(os.path.join(self.data_root, scene_name))
        if not os.path.isdir(scene_dir) and os.path.isdir(scene_name):
            scene_dir = scene_name
        left_path = os.path.join(scene_dir, "view1.png")
        right_path = os.path.join(scene_dir, "view5.png")
        calib_path = os.path.join(scene_dir, "calib.txt")
        return scene_dir, left_path, right_path, calib_path

    def process_scene(self, scene_name):
        scene_dir, left_path, right_path, calib_path = self.get_scene_paths(scene_name)
        left_img = cv2.imread(left_path, cv2.IMREAD_COLOR)
        right_img = cv2.imread(right_path, cv2.IMREAD_COLOR)
        if left_img is None or right_img is None:
            raise ValueError("无法读取 Middlebury 图像")

        if left_img.shape != right_img.shape:
            raise ValueError("左右图像尺寸不一致")

        calib = self.parse_calibration(calib_path) or {}
        dmin = int(calib.get("dmin", calib.get("vmin", self.default_dmin)))
        num_disp = int(calib.get("ndisp", self.default_ndisp))
        baseline = float(calib.get("baseline", self.default_baseline))
        focal = float(calib.get("f", self.default_focal))
        if num_disp <= 0:
            num_disp = self.default_ndisp

        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        if num_disp % 16 != 0:
            num_disp = (num_disp // 16 + 1) * 16
        min_disp = 0

        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=5,
            P1=8 * 3 * 5 ** 2,
            P2=32 * 3 * 5 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
        disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        if dmin != 0:
            disparity = disparity + float(dmin)

        disparity_vis = stereo_matcher.get_visual_disparity(disparity)

        results = {
            "left": left_img,
            "right": right_img,
            "disparity": disparity_vis,
            "disparity_raw": disparity,
            "width": left_img.shape[1],
            "height": left_img.shape[0],
            "dmin": dmin,
            "baseline": baseline,
            "focal": focal,
            "scene": scene_name
        }
        self.last_results[scene_name] = results
        return results

    def get_or_create_results(self, scene_name):
        if scene_name in self.last_results:
            return self.last_results[scene_name]
        return self.process_scene(scene_name)


# 创建全局演示实例
demo = StereoVisionDemo()
middlebury_demo = MiddleburyDemo()
latest_camera_frame = None
latest_xiaomi_frame = None
CAMERA_DISPARITY_SHIFT = 8


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


@app.route('/middlebury/list', methods=['GET'])
def middlebury_list():
    """获取 Middlebury 场景列表"""
    scenes = middlebury_demo.list_scenes()
    return jsonify({
        'success': True,
        'scenes': scenes
    })


@app.route('/middlebury/process', methods=['POST'])
def middlebury_process():
    """处理 Middlebury 双目图像"""
    try:
        data = request.json or {}
        scene = data.get('scene')
        if not scene:
            return jsonify({
                'success': False,
                'message': '未指定场景'
            })
        results = middlebury_demo.process_scene(scene)
        response = {}
        for key, img in results.items():
            if key not in ['disparity_raw', 'width', 'height', 'scene']:
                response[key] = demo.image_to_base64(img)
        return jsonify({
            'success': True,
            'images': response,
            'scene': scene,
            'width': results['width'],
            'height': results['height'],
            'message': 'Middlebury 视差计算完成！'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'处理失败: {str(e)}'
        })


@app.route('/middlebury/get_disparity', methods=['POST'])
def middlebury_get_disparity():
    """获取 Middlebury 指定像素的视差"""
    try:
        data = request.json or {}
        scene = data.get('scene')
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        if not scene:
            return jsonify({
                'success': False,
                'message': '未指定场景'
            })

        results = middlebury_demo.get_or_create_results(scene)
        disparity_raw = results['disparity_raw']
        if 0 <= y < disparity_raw.shape[0] and 0 <= x < disparity_raw.shape[1]:
            value = float(disparity_raw[y, x])
            return jsonify({
                'success': True,
                'x': x,
                'y': y,
                'disparity': value,
                'message': f'坐标({x}, {y})处视差: {value:.2f}'
            })
        return jsonify({
            'success': False,
            'message': '坐标超出图像范围'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取视差失败: {str(e)}'
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


def _camera_frame_generator():
    global latest_camera_frame
    frames = get_camera_frames(
        ip=CAMERA_CONFIG["ip"],
        port=CAMERA_CONFIG["port"],
        username=CAMERA_CONFIG["username"],
        password=CAMERA_CONFIG["password"],
        protocol=CAMERA_CONFIG["protocol"],
        stream_path=CAMERA_CONFIG["stream_path"],
        width=CAMERA_CONFIG["width"],
        height=CAMERA_CONFIG["height"],
    )
    for frame in frames:
        latest_camera_frame = frame
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


def _render_error_frame(message, size=(640, 480)):
    width, height = size
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, message, (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame


def _xiaomi_frame_generator():
    global latest_xiaomi_frame
    try:
        frames = get_xiaomi_frames()
        for frame in frames:
            latest_xiaomi_frame = frame
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    except Exception as exc:
        error_frame = _render_error_frame(str(exc))
        success, buffer = cv2.imencode(".jpg", error_frame)
        if success:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.route("/camera/stream")
def camera_stream():
    return Response(_camera_frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/camera/xiaomi/stream")
def xiaomi_camera_stream():
    return Response(_xiaomi_frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/camera/capture", methods=["POST"])
def camera_capture():
    global latest_camera_frame
    if latest_camera_frame is None:
        return jsonify({
            "success": False,
            "message": "尚未获取到摄像头帧，请先打开视频流"
        })
    success, buffer = cv2.imencode(".jpg", latest_camera_frame)
    if not success:
        return jsonify({
            "success": False,
            "message": "抓拍失败"
        })
    img_str = base64.b64encode(buffer).decode()
    return jsonify({
        "success": True,
        "image": f"data:image/jpeg;base64,{img_str}"
    })


@app.route("/camera/xiaomi/capture", methods=["POST"])
def xiaomi_camera_capture():
    global latest_xiaomi_frame
    if latest_xiaomi_frame is None:
        return jsonify({
            "success": False,
            "message": "尚未获取到小米摄像头帧，请先打开视频流"
        })
    success, buffer = cv2.imencode(".jpg", latest_xiaomi_frame)
    if not success:
        return jsonify({
            "success": False,
            "message": "小米摄像头抓拍失败"
        })
    img_str = base64.b64encode(buffer).decode()
    return jsonify({
        "success": True,
        "image": f"data:image/jpeg;base64,{img_str}"
    })


def _shift_frame(frame, shift_pixels):
    height, width = frame.shape[:2]
    shifted = np.zeros_like(frame)
    if shift_pixels >= 0:
        shifted[:, :-shift_pixels] = frame[:, shift_pixels:]
    else:
        shift_pixels = abs(shift_pixels)
        shifted[:, shift_pixels:] = frame[:, :-shift_pixels]
    return shifted


@app.route("/camera/capture_disparity", methods=["POST"])
def camera_capture_disparity():
    global latest_camera_frame, latest_xiaomi_frame
    if latest_camera_frame is None:
        return jsonify({
            "success": False,
            "message": "尚未获取到摄像头帧，请先打开视频流"
        })
    left = latest_camera_frame
    right = latest_xiaomi_frame
    message = "已生成测试视差图（单摄像头模拟）"
    if right is None:
        right = _shift_frame(left, CAMERA_DISPARITY_SHIFT)
    else:
        if left.shape[:2] != right.shape[:2]:
            right = cv2.resize(right, (left.shape[1], left.shape[0]))
        message = "已生成视差图（双摄像头示例，未标定）"
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    disparity = stereo_matcher.get_simple_disparity(gray_left, gray_right)
    disparity_vis = stereo_matcher.get_visual_disparity(disparity)
    success, buffer = cv2.imencode(".jpg", disparity_vis)
    if not success:
        return jsonify({
            "success": False,
            "message": "视差计算失败"
        })
    img_str = base64.b64encode(buffer).decode()
    return jsonify({
        "success": True,
        "image": f"data:image/jpeg;base64,{img_str}",
        "message": message
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5008, debug=True)
