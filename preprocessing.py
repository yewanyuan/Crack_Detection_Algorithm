"""
预处理模块

包含图像预处理功能：
- 光照矫正
- 竖条纹抑制
- 环形边界处理
"""

import numpy as np
import cv2
from utils import normalize01


def illumination_correction(gray):
    """
    同态/Retinex风格的光照矫正 + 自适应直方图均衡
    
    Args:
        gray: 灰度图像 (uint8)
        
    Returns:
        numpy.ndarray: 光照矫正后的图像
    """
    g = gray.astype(np.float32) / 255.0
    # 大尺度模糊估计背景
    bg = cv2.GaussianBlur(g, (0, 0), sigmaX=15, sigmaY=15)
    # 归一化抑制非均匀照明
    norm = normalize01(g / (bg + 1e-3))
    # 自适应均衡（CLAHE）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = clahe.apply((norm * 255).astype(np.uint8))
    return out


def suppress_vertical_stripes(gray, strength=0.25):
    """
    利用方向形态学估计竖条纹（泥浆/拼接线）并减弱之
    
    Args:
        gray: 灰度图像
        strength: 抑制强度 [0, 1]
        
    Returns:
        numpy.ndarray: 抑制竖条纹后的图像
    """
    h, w = gray.shape
    ksize = max(9, int(0.03 * h))  # 竖向核高度
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ksize))
    vertical_est = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    vertical_est = cv2.GaussianBlur(vertical_est, (0, 0), 3)
    out = cv2.addWeighted(gray, 1.0, vertical_est, -float(strength), 0)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def preprocess_image(gray_img):
    """
    完整的图像预处理流程
    
    Args:
        gray_img: 输入的灰度图像
        
    Returns:
        numpy.ndarray: 预处理后的图像
    """
    # 光照矫正
    corrected = illumination_correction(gray_img)
    
    # 竖条纹抑制
    cleaned = suppress_vertical_stripes(corrected)
    
    return cleaned