# -*-coding: utf-8 -*-
"""
生成演示用的双目图像对
"""
import cv2
import numpy as np
import os


def create_demo_stereo_images():
    """创建演示用的双目图像对"""

    # 确保data目录存在
    os.makedirs('data', exist_ok=True)

    # 创建一个简单的场景图像
    height, width = 480, 640

    # 创建左图像
    left_img = np.zeros((height, width, 3), dtype=np.uint8)

    # 添加一些几何图形作为特征点
    # 绘制矩形
    cv2.rectangle(left_img, (100, 100), (200, 200), (0, 255, 0), -1)
    cv2.rectangle(left_img, (300, 150), (400, 250), (255, 0, 0), -1)
    cv2.rectangle(left_img, (450, 200), (550, 300), (0, 0, 255), -1)

    # 绘制圆形
    cv2.circle(left_img, (150, 350), 50, (255, 255, 0), -1)
    cv2.circle(left_img, (350, 350), 40, (255, 0, 255), -1)
    cv2.circle(left_img, (500, 350), 30, (0, 255, 255), -1)

    # 添加文字
    cv2.putText(left_img, 'LEFT CAMERA', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 创建右图像（模拟视差）
    right_img = left_img.copy()

    # 模拟视差效果 - 将物体向左移动几个像素
    disparity_shift = 10

    # 清空右图像
    right_img = np.zeros((height, width, 3), dtype=np.uint8)

    # 重新绘制右图像的物体，位置稍微偏移
    cv2.rectangle(right_img, (100 - disparity_shift, 100), (200 - disparity_shift, 200), (0, 255, 0), -1)
    cv2.rectangle(right_img, (300 - disparity_shift + 2, 150), (400 - disparity_shift + 2, 250), (255, 0, 0), -1)
    cv2.rectangle(right_img, (450 - disparity_shift + 4, 200), (550 - disparity_shift + 4, 300), (0, 0, 255), -1)

    cv2.circle(right_img, (150 - disparity_shift, 350), 50, (255, 255, 0), -1)
    cv2.circle(right_img, (350 - disparity_shift + 2, 350), 40, (255, 0, 255), -1)
    cv2.circle(right_img, (500 - disparity_shift + 4, 350), 30, (0, 255, 255), -1)

    cv2.putText(right_img, 'RIGHT CAMERA', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 保存图像
    cv2.imwrite('data/left.png', left_img)
    cv2.imwrite('data/right.png', right_img)

    print("演示图像已生成:")
    print("- data/left.png")
    print("- data/right.png")

    return left_img, right_img


def create_chessboard_images():
    """创建棋盘格标定图像"""

    # 创建棋盘格图像用于标定演示
    board_size = (8, 11)  # 内角点数量
    square_size = 30  # 每个格子的像素大小

    # 计算图像尺寸
    img_width = (board_size[0] + 1) * square_size
    img_height = (board_size[1] + 1) * square_size

    # 创建棋盘格
    chessboard = np.zeros((img_height, img_width), dtype=np.uint8)

    for i in range(board_size[1] + 1):
        for j in range(board_size[0] + 1):
            if (i + j) % 2 == 0:
                y1 = i * square_size
                y2 = (i + 1) * square_size
                x1 = j * square_size
                x2 = (j + 1) * square_size
                chessboard[y1:y2, x1:x2] = 255

    # 转换为3通道
    chessboard_color = cv2.cvtColor(chessboard, cv2.COLOR_GRAY2BGR)

    # 保存棋盘格图像
    os.makedirs('data/calibration', exist_ok=True)
    cv2.imwrite('data/calibration/chessboard.png', chessboard_color)

    print("标定用棋盘格图像已生成: data/calibration/chessboard.png")


if __name__ == "__main__":
    create_demo_stereo_images()
    create_chessboard_images()