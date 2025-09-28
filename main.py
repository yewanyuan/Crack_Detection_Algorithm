"""
钻孔成像裂隙像素级识别 - 主程序

功能：
- 批量读取 --input_dir 下的 JPG/PNG 图片
- 自动完成预处理(照明/竖条纹抑制)、裂隙增强、分割、后处理
- 以原图同尺寸输出二值掩膜(裂隙=黑0，非裂隙=白255)与可视化叠加

依赖：
- Python 3.8+
- numpy, opencv-python, scikit-image, scipy

参数：
--input_dir   输入目录（默认 data）
--output_dir  输出目录（默认 results）
--circum_mm   钻孔周长mm（默认 94.25）
--depth_mm    单孔图像对应孔深mm（默认 500）

作者：Claude Code Assistant
版本：2.0 (模块化版本)
"""

import os
import glob
import argparse

# 导入自定义模块
from utils import ensure_dir, imread_gray, estimate_px_per_mm
from preprocessing import preprocess_image
from enhancement import enhance_cracks
from segmentation import segment_and_postprocess
from visualization import make_visual_overlay, save_results


def process_single_image(img_path, args):
    """
    处理单张图像
    
    Args:
        img_path: 图像文件路径
        args: 命令行参数
        
    Returns:
        tuple: (是否成功, 错误信息)
    """
    try:
        # 读取图片
        bgr, gray = imread_gray(img_path)
        h, w = gray.shape[:2]
        
        # 计算像素-毫米换算比例
        px_per_mm_x, px_per_mm_y, px_per_mm_mean = estimate_px_per_mm(
            w, h, args.circum_mm, args.depth_mm
        )
        
        print(f"尺寸: {w}x{h}, 像素/毫米: X={px_per_mm_x:.2f}, Y={px_per_mm_y:.2f}")

        # 预处理
        preprocessed = preprocess_image(gray)
        
        # 裂隙增强
        prob_map = enhance_cracks(preprocessed)
        
        # 分割和后处理
        mask_processed = segment_and_postprocess(
            prob_map, 
            px_per_mm_mean,
            min_length_mm=5.0,
            width_mm_threshold=1.0,
            vertical_suppress=True
        )
        
        # 创建可视化结果
        overlay = make_visual_overlay(bgr, mask_processed)
        
        # 保存结果
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path, overlay_path = save_results(args.output_dir, base_name, 
                                             mask_processed, overlay)
        
        print(f"结果已保存: {mask_path}, {overlay_path}")
        return True, None
        
    except Exception as e:
        error_msg = f"处理 {img_path} 时出错: {str(e)}"
        print(error_msg)
        return False, error_msg


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='钻孔成像裂隙识别系统',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_dir', type=str, default='data', 
                       help='输入图片目录')
    parser.add_argument('--output_dir', type=str, default='results', 
                       help='输出结果目录')
    parser.add_argument('--circum_mm', type=float, default=94.25, 
                       help='钻孔周长(mm)')
    parser.add_argument('--depth_mm', type=float, default=500.0, 
                       help='单孔图像对应孔深(mm)')
    
    args = parser.parse_args()

    # 确保输出目录存在
    ensure_dir(args.output_dir)

    # 获取输入目录下所有图片文件
    extensions = ['*.jpg', '*.png', '*.JPG', '*.PNG']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
    
    if not image_paths:
        print(f"错误：在 {args.input_dir} 中未找到任何图片文件")
        return

    print(f"找到 {len(image_paths)} 张图片，开始处理...")
    print(f"输出目录: {args.output_dir}")
    print(f"钻孔参数: 周长={args.circum_mm}mm, 孔深={args.depth_mm}mm")
    print("-" * 60)

    # 处理每张图片
    success_count = 0
    for i, img_path in enumerate(image_paths, 1):
        print(f"\\n处理图片 {i}/{len(image_paths)}: {os.path.basename(img_path)}")
        
        success, error = process_single_image(img_path, args)
        if success:
            success_count += 1

    # 输出统计信息
    print("-" * 60)
    print(f"处理完成! 成功: {success_count}/{len(image_paths)}")
    
    if success_count < len(image_paths):
        print(f"失败: {len(image_paths) - success_count} 张图片处理失败")


if __name__ == '__main__':
    main()