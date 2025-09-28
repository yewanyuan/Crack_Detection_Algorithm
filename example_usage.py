"""
使用示例：如何在其他项目中使用裂缝检测模块
"""

import numpy as np
from utils import imread_gray, estimate_px_per_mm
from preprocessing import preprocess_image
from enhancement import enhance_cracks
from segmentation import segment_and_postprocess
from visualization import make_visual_overlay


def detect_cracks_in_image(image_path, circum_mm=94.25, depth_mm=500.0):
    """
    对单张图像进行裂缝检测的完整流程示例
    
    Args:
        image_path: 图像路径
        circum_mm: 钻孔周长
        depth_mm: 孔深
        
    Returns:
        tuple: (原图, 裂缝掩膜, 可视化结果)
    """
    # 1. 读取图像
    bgr_img, gray_img = imread_gray(image_path)
    h, w = gray_img.shape
    
    # 2. 计算像素-毫米比例
    _, _, px_per_mm = estimate_px_per_mm(w, h, circum_mm, depth_mm)
    
    # 3. 预处理
    preprocessed = preprocess_image(gray_img)
    
    # 4. 特征增强
    prob_map = enhance_cracks(preprocessed)
    
    # 5. 分割和后处理
    crack_mask = segment_and_postprocess(prob_map, px_per_mm)
    
    # 6. 创建可视化
    overlay = make_visual_overlay(bgr_img, crack_mask)
    
    return bgr_img, crack_mask, overlay


if __name__ == "__main__":
    # 使用示例
    image_path = "data/1.jpg"
    
    try:
        original, mask, visualization = detect_cracks_in_image(image_path)
        
        print(f"成功处理图像: {image_path}")
        print(f"图像尺寸: {original.shape}")
        print(f"检测到的裂缝像素数: {np.sum(mask)}")
        
        # 保存结果（可选）
        import cv2
        cv2.imwrite("example_mask.png", (1 - mask) * 255)
        cv2.imwrite("example_overlay.jpg", visualization)
        print("结果已保存: example_mask.png, example_overlay.jpg")
        
    except Exception as e:
        print(f"处理失败: {e}")