"""
分割与后处理模块

包含图像分割和后处理功能：
- Otsu自适应阈值分割
- 形态学操作
- 几何特征筛选
- 竖条纹过滤
"""

import math
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt

# 尝试导入scikit-image
try:
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects, remove_small_holes
    from skimage.measure import label, regionprops
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False


def segment_cracks(prob_map):
    """
    自适应阈值（Otsu）得到初始二值裂隙掩膜
    
    Args:
        prob_map: 裂隙概率图 [0,1]
        
    Returns:
        numpy.ndarray: 二值掩膜 (0/1)
    """
    if SKIMAGE_OK:
        thr = threshold_otsu(prob_map)
    else:
        # OpenCV 的 Otsu 要求8位图
        thr, _ = cv2.threshold((prob_map * 255).astype(np.uint8), 0, 255, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = thr / 255.0
    
    mask = (prob_map >= thr).astype(np.uint8)
    return mask


def postprocess_mask(mask, px_per_mm_mean, min_length_mm=5.0, 
                    width_mm_threshold=1.0, vertical_suppress=True):
    """
    微调版后处理：平衡裂缝完整性和误检测控制
    
    Args:
        mask: 初始分割掩膜
        px_per_mm_mean: 平均像素/毫米比例
        min_length_mm: 最小长度（毫米）
        width_mm_threshold: 宽度阈值（毫米）
        vertical_suppress: 是否抑制竖条纹
        
    Returns:
        numpy.ndarray: 处理后的掩膜
    """
    if not SKIMAGE_OK:
        return _postprocess_fallback(mask, px_per_mm_mean, min_length_mm)
    
    # 小孔/小岛处理 - 稍微放宽参数
    mask = remove_small_objects(mask.astype(bool), min_size=16).astype(np.uint8)
    mask = remove_small_holes(mask.astype(bool), area_threshold=32).astype(np.uint8)

    # 形态学操作：先开放去噪再闭合连接
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, 
                           cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    
    # 使用方向性核进行闭合，更好地连接水平/倾斜裂缝
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  # 水平连接核
    diagonal_kernel = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.uint8)     # 对角连接核
    
    # 多方向闭合但避免竖直连接
    temp1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, horizontal_kernel, iterations=1)
    temp2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, diagonal_kernel, iterations=1) 
    temp3 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.flip(diagonal_kernel, 1), iterations=1)
    mask = np.maximum.reduce([temp1, temp2, temp3])

    # 基于距离变换估计厚度
    dt = distance_transform_edt(mask > 0)
    thickness_px = dt * 2.0
    px_per_mm = px_per_mm_mean
    thickness_mm = thickness_px / px_per_mm
    mask = (thickness_mm >= width_mm_threshold).astype(np.uint8)

    # 连通域分析：放宽几何筛选条件
    lbl = label(mask, connectivity=2)
    props = regionprops(lbl)

    h, w = mask.shape
    min_len_px = int(max(6, round(min_length_mm * px_per_mm * 0.8)))  # 稍微降低长度要求
    cleaned = np.zeros_like(mask, dtype=np.uint8)

    for rp in props:
        coords = rp.coords
        area = rp.area
        
        # 面积过滤 - 放宽一些
        if area < max(min_len_px * 0.8, 5):  # 使用更小的阈值
            continue

        # 长宽比筛选：放宽条件
        if rp.minor_axis_length > 0:
            aspect_ratio = rp.major_axis_length / rp.minor_axis_length
            if aspect_ratio < 2.0:  # 从2.5降到2.0
                continue

        # 主轴方向分析
        theta = rp.orientation
        is_verticalish = abs(math.cos(theta)) < 0.2  # 恢复到原来的0.2（约78度）
        
        # 竖条纹过滤 - 保持严格但添加例外
        if vertical_suppress and is_verticalish:
            # 多重条件判断竖条纹，但为短裂缝添加例外
            is_long_vertical = rp.major_axis_length > 0.5 * h  # 恢复到0.5
            is_thin = rp.minor_axis_length < 0.03 * w
            is_straight = rp.extent > 0.7  # 提高直线度要求
            
            # 如果是短的竖直特征，可能是真实裂缝，不过滤
            if rp.major_axis_length < 0.3 * h:
                pass  # 短的竖直特征保留
            elif (is_long_vertical and is_thin and is_straight):
                continue  # 只过滤明显的长直竖条纹
        
        # 形状规整性检查 - 稍微放宽
        if rp.solidity > 0.99:  # 从0.98提高到0.99，只过滤极度规整的
            continue
            
        # 紧凑性检查 - 稍微放宽条件
        if area > 0:
            perimeter = rp.perimeter if rp.perimeter > 0 else 1
            compactness = (perimeter ** 2) / (4 * np.pi * area)
            if compactness < 1.2:  # 从1.5降到1.2，允许稍微紧凑的形状
                continue

        cleaned[coords[:, 0], coords[:, 1]] = 1

    # 最终清理 - 使用更宽松的size要求
    cleaned = remove_small_objects(cleaned.astype(bool), 
                                  min_size=max(4, min_len_px//3)).astype(np.uint8)
    
    # 最后的形态学清理
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, 
                              cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    return cleaned


def _postprocess_fallback(mask, px_per_mm_mean, min_length_mm):
    """
    当scikit-image不可用时的简化后处理
    """
    # 基本的形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask


def segment_and_postprocess(prob_map, px_per_mm_mean, **kwargs):
    """
    完整的分割和后处理流程
    
    Args:
        prob_map: 裂隙概率图
        px_per_mm_mean: 平均像素/毫米比例
        **kwargs: 后处理参数
        
    Returns:
        numpy.ndarray: 最终的裂缝掩膜
    """
    # 分割
    mask = segment_cracks(prob_map)
    
    # 后处理
    mask_processed = postprocess_mask(mask, px_per_mm_mean, **kwargs)
    
    return mask_processed