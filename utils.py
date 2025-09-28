"""
工具函数模块

包含通用的辅助函数：
- 文件IO操作
- 图像读取和归一化
- 目录管理
- 像素-毫米换算
"""

import os
import numpy as np
import cv2


def ensure_dir(path):
    """确保目录存在，如不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)


def imread_gray(path):
    """
    读取图像并转换为灰度图
    
    Args:
        path: 图像文件路径
        
    Returns:
        tuple: (彩色图像, 灰度图像)
    """
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"无法读取图像：{path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def normalize01(x):
    """
    将数组归一化到[0,1]范围
    
    Args:
        x: 输入数组
        
    Returns:
        numpy.ndarray: 归一化后的数组
    """
    x = x.astype(np.float32)
    mn, mx = np.min(x), np.max(x)
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def circular_pad_h(img, pad_left, pad_right):
    """水平方向环形填充，以贴合圆柱展开的周向周期边界"""
    return np.hstack([img[:, -pad_left:], img, img[:, :pad_right]])


def circular_unpad_h(img_padded, pad_left, pad_right):
    """移除水平方向的环形填充"""
    h, w = img_padded.shape[:2]
    return img_padded[:, pad_left:w - pad_right]


def estimate_px_per_mm(width_px, height_px, circum_mm=94.25, depth_mm=500.0):
    """
    根据钻孔参数计算像素-毫米换算比例
    
    Args:
        width_px: 图像宽度（像素）
        height_px: 图像高度（像素）
        circum_mm: 钻孔周长（毫米）
        depth_mm: 孔深（毫米）
        
    Returns:
        tuple: (横向px/mm, 纵向px/mm, 平均px/mm)
    """
    px_per_mm_x = width_px / float(circum_mm)
    px_per_mm_y = height_px / float(depth_mm)
    px_per_mm_mean = 0.5 * (px_per_mm_x + px_per_mm_y)
    return px_per_mm_x, px_per_mm_y, px_per_mm_mean