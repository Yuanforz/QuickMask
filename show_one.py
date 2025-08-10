import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import glob

# 添加中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SegmentationVisualizer:
    def __init__(self, image_dir, mask_dir):
        """
        初始化分割可视化器
        :param image_dir: 图片目录
        :param mask_dir: 标注掩码目录
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.class_colors = {
            0: [0, 0, 0],      # 背景 - 黑色
            1: [255, 0, 0],    # 类别1 - 红色
            2: [0, 255, 0],    # 类别2 - 绿色
            3: [0, 0, 255]     # 类别3 - 蓝色
        }
        
    def find_typical_samples(self, num_classes=3):
        """
        为每个类别寻找典型样本
        :param num_classes: 类别数量
        :return: 每个类别的典型样本路径
        """
        image_files = glob.glob(os.path.join(self.image_dir, "*.jpg")) + \
                     glob.glob(os.path.join(self.image_dir, "*.png")) + \
                     glob.glob(os.path.join(self.image_dir, "*.jpeg"))
        
        typical_samples = {}
        
        for class_id in range(1, num_classes + 1):
            best_sample = None
            max_pixel_count = 0
            
            for img_path in image_files:
                img_name = os.path.basename(img_path)
                mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
                mask_path = os.path.join(self.mask_dir, mask_name)
                
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        pixel_count = np.sum(mask == class_id)
                        if pixel_count > max_pixel_count:
                            max_pixel_count = pixel_count
                            best_sample = (img_path, mask_path, pixel_count)
                            # if class_id == 3:
                            #     break
            
            if best_sample:
                typical_samples[class_id] = best_sample
                
        return typical_samples
    
    def create_overlay(self, image, mask, alpha=0.5):
        """
        创建图片和标注的混合显示
        :param image: 原始图片
        :param mask: 分割掩码
        :param alpha: 透明度
        :return: 混合后的图片
        """
        overlay = image.copy()
        colored_mask = np.zeros_like(image)
        
        # 为每个类别着色
        for class_id, color in self.class_colors.items():
            if class_id == 0:  # 跳过背景
                continue
            mask_region = (mask == class_id)
            colored_mask[mask_region] = color
        
        # 混合原图和彩色掩码
        result = cv2.addWeighted(overlay, 1-alpha, colored_mask, alpha, 0)
        return result, colored_mask
    
    def visualize_samples(self, typical_samples):
        """
        可视化典型样本
        :param typical_samples: 典型样本字典
        """
        num_samples = len(typical_samples)
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (class_id, (img_path, mask_path, pixel_count)) in enumerate(typical_samples.items()):
            # 加载图片和掩码
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # 创建混合图像
            overlay, colored_mask = self.create_overlay(image, mask)
            
            # 显示原图
            axes[idx, 0].imshow(image)
            axes[idx, 0].set_title(f'类别 {class_id} - 原图', fontsize=12)
            axes[idx, 0].axis('off')
            
            # 显示掩码
            axes[idx, 1].imshow(mask, cmap='gray')
            axes[idx, 1].set_title(f'类别 {class_id} - 标注掩码', fontsize=12)
            axes[idx, 1].axis('off')
            
            # 显示彩色掩码
            axes[idx, 2].imshow(colored_mask)
            axes[idx, 2].set_title(f'类别 {class_id} - 彩色标注', fontsize=12)
            axes[idx, 2].axis('off')
            
            # 显示混合结果
            axes[idx, 3].imshow(overlay)
            axes[idx, 3].set_title(f'类别 {class_id} - 混合显示\n像素数: {pixel_count}', fontsize=12)
            axes[idx, 3].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def show_statistics(self, typical_samples):
        """
        显示统计信息
        :param typical_samples: 典型样本字典
        """
        print("=" * 50)
        print("3类分割数据统计信息")
        print("=" * 50)
        
        for class_id, (img_path, mask_path, pixel_count) in typical_samples.items():
            img_name = os.path.basename(img_path)
            print(f"类别 {class_id}:")
            print(f"  典型样本: {img_name}")
            print(f"  像素数量: {pixel_count}")
            print(f"  颜色标识: RGB{self.class_colors[class_id]}")
            print("-" * 30)

def main():
    # 设置图片和标注目录路径
    image_dir = "images"  # 请根据实际路径修改
    mask_dir = "masks"    # 请根据实际路径修改
    
    # 如果目录不存在，创建示例目录结构提示
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print("请确保以下目录存在并包含数据:")
        print(f"图片目录: {os.path.abspath(image_dir)}")
        print(f"标注目录: {os.path.abspath(mask_dir)}")
        print("\n目录结构应该如下:")
        print("dataset/")
        print("├── images/")
        print("│   ├── image1.jpg")
        print("│   ├── image2.jpg")
        print("│   └── ...")
        print("└── masks/")
        print("    ├── image1.png")
        print("    ├── image2.png")
        print("    └── ...")
        return
    
    # 创建可视化器
    visualizer = SegmentationVisualizer(image_dir, mask_dir)
    
    # 寻找典型样本
    print("正在寻找每个类别的典型样本...")
    typical_samples = visualizer.find_typical_samples(num_classes=3)
    
    if not typical_samples:
        print("未找到有效的样本数据，请检查数据路径和格式")
        return
    
    # 显示统计信息
    visualizer.show_statistics(typical_samples)
    
    # 可视化结果
    print("正在生成可视化结果...")
    visualizer.visualize_samples(typical_samples)

if __name__ == "__main__":
    main()
