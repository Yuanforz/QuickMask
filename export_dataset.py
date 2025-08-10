#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据导出脚本
专门针对标注数据导出，将所有含有标注掩码且标注非空的图像和掩码导出到指定格式
"""

import os
import cv2
import numpy as np
import shutil
from pathlib import Path
import argparse


def is_mask_empty(mask_path):
    """检查掩码文件是否为空（只有背景像素）"""
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return True
        # 检查是否有非零像素（标注区域）
        return not np.any(mask > 0)
    except Exception as e:
        print(f"读取掩码失败 {mask_path}: {e}")
        return True


def validate_and_convert_mask(mask_path, output_path):
    """
    验证并转换掩码格式
    确保掩码符合语义分割标准：
    - 0 = 背景（黑色）
    - 1 = 第1类缺陷
    - 2 = 第2类缺陷  
    - 3 = 第3类缺陷
    """
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"无法读取掩码文件: {mask_path}")
            return False
        
        # 获取掩码中的唯一值
        unique_values = np.unique(mask)
        print(f"掩码 {os.path.basename(mask_path)} 包含像素值: {unique_values}")
        
        # 检查是否有超出范围的值
        max_class_id = 3  # 支持0-3类别
        if np.any(unique_values > max_class_id):
            print(f"警告: 掩码包含超出范围的类别ID (>3): {unique_values[unique_values > max_class_id]}")
            # 将超出范围的值截断到最大类别
            mask = np.clip(mask, 0, max_class_id)
        
        # 保存转换后的掩码
        success = cv2.imwrite(output_path, mask)
        if not success:
            print(f"保存掩码失败: {output_path}")
            return False
        
        return True
    
    except Exception as e:
        print(f"处理掩码时出错 {mask_path}: {e}")
        return False


def export_dataset(source_image_dir="images", source_mask_dir="masks", output_dir="data"):
    """
    导出标注数据集
    
    Args:
        source_image_dir: 源图像文件夹
        source_mask_dir: 源掩码文件夹  
        output_dir: 输出文件夹
    """
    
    # 创建输出目录结构
    output_images_dir = os.path.join(output_dir, "images")
    output_masks_dir = os.path.join(output_dir, "masks")
    
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)
    
    # 统计信息
    total_masks = 0
    exported_pairs = 0
    empty_masks = 0
    missing_images = 0
    
    print(f"开始导出数据集...")
    print(f"源图像目录: {source_image_dir}")
    print(f"源掩码目录: {source_mask_dir}")
    print(f"输出目录: {output_dir}")
    print("-" * 50)
    
    # 遍历所有掩码文件
    for mask_file in os.listdir(source_mask_dir):
        if not mask_file.lower().endswith('.png'):
            continue
            
        total_masks += 1
        mask_path = os.path.join(source_mask_dir, mask_file)
        
        # 检查掩码是否为空
        if is_mask_empty(mask_path):
            empty_masks += 1
            print(f"跳过空掩码: {mask_file}")
            continue
        
        # 查找对应的图像文件
        image_basename = os.path.splitext(mask_file)[0]
        image_file = None
        
        # 支持多种图像格式
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            candidate = image_basename + ext
            if os.path.exists(os.path.join(source_image_dir, candidate)):
                image_file = candidate
                break
        
        if image_file is None:
            missing_images += 1
            print(f"警告: 找不到对应的图像文件: {image_basename}")
            continue
        
        # 复制图像文件
        source_image_path = os.path.join(source_image_dir, image_file)
        output_image_path = os.path.join(output_images_dir, image_file)
        
        try:
            shutil.copy2(source_image_path, output_image_path)
        except Exception as e:
            print(f"复制图像失败 {image_file}: {e}")
            continue
        
        # 验证并转换掩码文件
        output_mask_path = os.path.join(output_masks_dir, mask_file)
        if validate_and_convert_mask(mask_path, output_mask_path):
            exported_pairs += 1
            print(f"✓ 导出: {image_file} -> {mask_file}")
        else:
            # 如果掩码处理失败，删除已复制的图像
            if os.path.exists(output_image_path):
                os.remove(output_image_path)
            print(f"✗ 导出失败: {image_file}")
    
    # 打印统计结果
    print("-" * 50)
    print(f"导出完成!")
    print(f"总掩码文件: {total_masks}")
    print(f"空掩码文件: {empty_masks}")
    print(f"缺失图像文件: {missing_images}")
    print(f"成功导出的图像-掩码对: {exported_pairs}")
    print(f"导出目录: {os.path.abspath(output_dir)}")
    
    if exported_pairs > 0:
        print(f"\n数据集结构:")
        print(f"{output_dir}/")
        print(f"├── images/           # 原始图像 ({exported_pairs} 个文件)")
        print(f"└── masks/            # 分割掩码 ({exported_pairs} 个文件)")
        print(f"    └── 像素值: 0=背景, 1=缺陷1, 2=缺陷2, 3=缺陷3")
    
    return exported_pairs


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='导出标注数据集')
    parser.add_argument('--images', default='D:\\Download\\Tencent Files\\1370883911\\FileRecv\\wcx\\images', help='源图像文件夹路径')
    parser.add_argument('--masks', default='D:\\Download\\Tencent Files\\1370883911\\FileRecv\\wcx\\masks', help='源掩码文件夹路径')
    parser.add_argument('--output', default='D:\\Download\\Tencent Files\\1370883911\\FileRecv\\wcx\\data_wcx', help='输出文件夹路径')
    parser.add_argument('--check-only', action='store_true', help='仅检查数据，不执行导出')
    
    args = parser.parse_args()
    
    # 检查源目录是否存在
    if not os.path.exists(args.images):
        print(f"错误: 图像目录不存在: {args.images}")
        return
    
    if not os.path.exists(args.masks):
        print(f"错误: 掩码目录不存在: {args.masks}")
        return
    
    if args.check_only:
        # 仅检查模式
        print("=== 数据检查模式 ===")
        mask_files = [f for f in os.listdir(args.masks) if f.lower().endswith('.png')]
        valid_pairs = 0
        
        for mask_file in mask_files:
            mask_path = os.path.join(args.masks, mask_file)
            if not is_mask_empty(mask_path):
                image_basename = os.path.splitext(mask_file)[0]
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    if os.path.exists(os.path.join(args.images, image_basename + ext)):
                        valid_pairs += 1
                        break
        
        print(f"总掩码文件: {len(mask_files)}")
        print(f"有效的图像-掩码对: {valid_pairs}")
    else:
        # 执行导出
        exported_count = export_dataset(args.images, args.masks, args.output)
        if exported_count == 0:
            print("没有数据被导出。请检查源目录和标注质量。")


if __name__ == '__main__':
    main()
