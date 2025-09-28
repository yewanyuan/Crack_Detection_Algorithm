"""
可视化模块

包含图像可视化和结果保存功能：
- 叠加可视化
- 结果保存
"""

import cv2
import numpy as np


def make_visual_overlay(bgr_img, mask, color=(0, 0, 255), alpha=0.6):
    """
    在原图上叠加半透明裂隙掩膜
    
    Args:
        bgr_img: BGR彩色图像
        mask: 裂缝掩膜 (0/1)
        color: 叠加颜色 (B,G,R)
        alpha: 透明度
        
    Returns:
        numpy.ndarray: 叠加后的图像
    """
    overlay = bgr_img.copy()
    color_img = np.zeros_like(bgr_img)
    color_img[:, :] = color
    mask3 = np.dstack([mask]*3).astype(bool)
    overlay[mask3] = cv2.addWeighted(bgr_img, 1 - alpha, color_img, alpha, 0)[mask3]
    return overlay


def save_results(output_dir, base_name, mask, overlay):
    """
    保存处理结果
    
    Args:
        output_dir: 输出目录
        base_name: 基础文件名（不含扩展名）
        mask: 裂缝掩膜
        overlay: 叠加可视化图像
    """
    import os
    
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.jpg")
    
    # 保存二值掩膜（裂隙=黑0，非裂隙=白255）
    cv2.imwrite(mask_path, (1 - mask) * 255)
    
    # 保存可视化叠加图
    cv2.imwrite(overlay_path, overlay)
    
    return mask_path, overlay_path