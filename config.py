"""
配置文件

包含算法的默认参数和配置选项
"""

# 钻孔几何参数
BOREHOLE_CONFIG = {
    'circumference_mm': 94.25,  # 钻孔周长(mm)
    'depth_mm': 500.0,          # 孔深(mm)
}

# 预处理参数
PREPROCESSING_CONFIG = {
    'vertical_stripe_strength': 0.25,  # 竖条纹抑制强度
    'clahe_clip_limit': 2.0,           # CLAHE限制
    'clahe_tile_grid_size': (8, 8),    # CLAHE网格大小
}

# 裂隙增强参数
ENHANCEMENT_CONFIG = {
    'sato_sigmas': (1, 2, 3, 4),       # Sato滤波器尺度
    'sato_weight': 0.75,               # Sato权重
    'canny_weight': 0.25,              # Canny权重
    'canny_sigma': 1.2,                # Canny sigma
    'canny_low_threshold': 0.1,        # Canny低阈值
    'canny_high_threshold': 0.3,       # Canny高阈值
}

# 分割参数
SEGMENTATION_CONFIG = {
    'use_otsu': True,                  # 使用Otsu阈值
}

# 后处理参数
POSTPROCESSING_CONFIG = {
    'min_length_mm': 5.0,              # 最小长度(mm)
    'width_mm_threshold': 1.0,         # 宽度阈值(mm)
    'vertical_suppress': True,         # 竖条纹抑制
    'aspect_ratio_threshold': 2.0,     # 长宽比阈值
    'solidity_threshold': 0.99,        # 规整性阈值
    'compactness_threshold': 1.2,      # 紧凑性阈值
    'small_object_size': 16,           # 小对象大小
    'small_hole_area': 32,             # 小孔面积
}

# 可视化参数
VISUALIZATION_CONFIG = {
    'overlay_color': (0, 0, 255),      # BGR颜色
    'overlay_alpha': 0.6,              # 透明度
}