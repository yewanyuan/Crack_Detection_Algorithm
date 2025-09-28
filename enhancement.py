"""
裂隙增强模块

包含裂隙特征增强算法：
- Sato滤波器
- Canny边缘检测
- Gabor滤波器（备选）
"""

import numpy as np
import cv2
from utils import normalize01

# 尝试导入scikit-image
try:
    from skimage.filters import sato, gabor
    from skimage.feature import canny
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False


def enhance_ridges(gray_norm):
    """
    使用Sato（黑色细线增强，适合裂隙）与Canny边缘融合
    
    Args:
        gray_norm: 归一化的灰度图像 [0,1]
        
    Returns:
        numpy.ndarray: 裂隙概率图 [0,1]
    """
    if SKIMAGE_OK:
        # Sato：黑色细线增强。sigmas可按分辨率适当调整
        sato_resp = sato(gray_norm, sigmas=(1, 2, 3, 4), black_ridges=True)
        sato_resp = normalize01(sato_resp)

        # Canny边缘（对比度变化处），加入一定权重
        edges = canny(gray_norm, sigma=1.2, low_threshold=0.1, high_threshold=0.3)
        edges_f = edges.astype(np.float32)

        # 组合概率（可调权重）
        prob = normalize01(0.75 * sato_resp + 0.25 * edges_f)
        return prob
    else:
        # Gabor滤波组备选：多方向响应最大值
        return _gabor_enhancement_fallback(gray_norm)


def _gabor_enhancement_fallback(gray_norm):
    """
    Gabor滤波备选方案（当scikit-image不可用时）
    
    Args:
        gray_norm: 归一化的灰度图像 [0,1]
        
    Returns:
        numpy.ndarray: 裂隙概率图 [0,1]
    """
    if SKIMAGE_OK:
        thetas = np.linspace(0, np.pi, 8, endpoint=False)
        gab_max = np.zeros_like(gray_norm, dtype=np.float32)
        for th in thetas:
            filt_real, filt_imag = gabor(gray_norm, frequency=0.15, theta=th)
            gab_max = np.maximum(gab_max, normalize01(np.abs(filt_real)))
        
        # 简易Canny融合
        edges = cv2.Canny((gray_norm * 255).astype(np.uint8), 60, 150)
        edges = edges.astype(np.float32) / 255.0
        prob = normalize01(0.8 * gab_max + 0.2 * edges)
        return prob
    else:
        # 如果连Gabor都不可用，使用基本的边缘检测
        edges = cv2.Canny((gray_norm * 255).astype(np.uint8), 60, 150)
        return edges.astype(np.float32) / 255.0


def enhance_cracks(gray_img):
    """
    完整的裂缝增强流程
    
    Args:
        gray_img: 预处理后的灰度图像
        
    Returns:
        numpy.ndarray: 裂隙概率图 [0,1]
    """
    # 归一化
    gray_norm = normalize01(gray_img)
    
    # 裂隙增强
    prob_map = enhance_ridges(gray_norm)
    
    return prob_map