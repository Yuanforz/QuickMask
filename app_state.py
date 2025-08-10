# 文件: app_state.py
import os
import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

class AppState(QObject):
    state_changed = pyqtSignal()
    mask_updated = pyqtSignal()
    progress_updated = pyqtSignal(int, int)
    file_list_updated = pyqtSignal(list)
    files_deleted = pyqtSignal()
    # 新增信号：当模式改变时发出
    mode_changed = pyqtSignal(str)

    def __init__(self, image_dir="images", mask_dir="masks"):
        super().__init__()
        self.image_dir = image_dir; self.mask_dir = mask_dir
        os.makedirs(self.image_dir, exist_ok=True); os.makedirs(self.mask_dir, exist_ok=True)
        self.image_paths = []; self.mask_existence = []
        self.current_index = -1; self.current_image = None; self.current_mask = None
        self.brush_size = 20; self.current_class_id = 1
        self.undo_stack = []; self.max_undo_steps = 10
        
        # 应用模式状态 ('draw', 'rect_seg', 'sam_assist')
        self.mode = 'draw'
        
        # 矩形分割相关状态
        self.last_rect_seg_region = None  # 存储最后一次矩形分割的区域信息 (x1, y1, x2, y2)
        self.last_rect_seg_mask = None    # 存储最后一次矩形分割生成的掩码
        self.can_invert_rect_seg = False  # 标记是否可以对矩形分割结果进行取反
        
        # SAM相关状态（延迟初始化）
        self.sam_model = None
        self.sam_predictor = None
        self.sam_device = None
        self.current_embedding = False
        self.sam_positive_points = []  # 存储正点击点 [(x, y), ...]
        self.sam_negative_points = []  # 存储负点击点 [(x, y), ...]
        self.sam_point_history = []    # 存储点击历史 [(x, y, is_positive), ...] 按时间顺序
        self.sam_temp_mask = None      # 临时生成的SAM掩码

    def set_mode(self, mode: str):
        """设置应用模式，并发出信号通知UI"""
        if self.mode != mode:
            # 从SAM辅助模式退出时清理状态（但保留embedding以便重新进入）
            if self.mode == 'sam_assist':
                self.clear_sam_state(reset_embedding=False)
            
            self.mode = mode
            print(f"模式切换到: {self.mode}")
            self.mode_changed.emit(self.mode)
            
    # ... 其他所有方法保持不变 ...
    def load_files(self):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        self.image_paths.clear(); self.mask_existence.clear()
        files = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(valid_extensions)])
        for f in files:
            self.image_paths.append(os.path.join(self.image_dir, f))
            mask_filename = os.path.splitext(f)[0] + ".png"
            mask_path = os.path.join(self.mask_dir, mask_filename)
            self.mask_existence.append(os.path.exists(mask_path))
        self.file_list_updated.emit(self.mask_existence)
        if self.image_paths: self.navigate_to(0, save_before_nav=False)
        else: print("错误: 在 'images' 文件夹中未找到任何图片。"); self.files_deleted.emit()
    def navigate_to(self, index, save_before_nav=True):
        if not (0 <= index < len(self.image_paths)):
            if not self.image_paths:
                self.current_index = -1; self.current_image = None; self.current_mask = None
                self.files_deleted.emit()
            return
            
        # 从SAM模式切换图片时自动退出SAM模式，并重置embedding
        if self.mode == 'sam_assist':
            self.clear_sam_state(reset_embedding=True)
            self.mode = 'draw'
            print(f"模式切换到: {self.mode}")
            self.mode_changed.emit(self.mode)
        else:
            # 即使不在SAM模式，切换图片也要重置embedding状态
            self.current_embedding = False
            
        if save_before_nav: self.save_current_mask()
        self.current_index = index
        image_path = self.image_paths[self.current_index]
        self.current_image = cv2.imread(image_path)
        if self.current_image is None:
            print(f"错误: 无法加载图片 {image_path}")
            self.image_paths.pop(self.current_index); self.mask_existence.pop(self.current_index)
            self.navigate_to(self.current_index, save_before_nav=False)
            return
        mask_filename = os.path.splitext(os.path.basename(image_path))[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_filename)
        if os.path.exists(mask_path):
            self.current_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # 确保掩码尺寸与图像匹配
            if self.current_mask.shape[:2] != self.current_image.shape[:2]:
                self.current_mask = cv2.resize(self.current_mask, self.current_image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            self.mask_existence[self.current_index] = True
        else:
            h, w = self.current_image.shape[:2]
            self.current_mask = np.zeros((h, w), dtype=np.uint8)
        self.undo_stack.clear()
        # 清理矩形分割状态
        self.last_rect_seg_region = None
        self.last_rect_seg_mask = None
        self.can_invert_rect_seg = False
        self.file_list_updated.emit(self.mask_existence)
        self.state_changed.emit()
        self.progress_updated.emit(self.current_index, len(self.image_paths))
    def save_current_mask(self):
        if self.current_mask is None or self.current_index == -1: return
        is_mask_empty = not np.any(self.current_mask)
        image_path = self.image_paths[self.current_index]
        mask_filename = os.path.splitext(os.path.basename(image_path))[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_filename)
        if not is_mask_empty or os.path.exists(mask_path):
            cv2.imwrite(mask_path, self.current_mask)
            if not self.mask_existence[self.current_index]:
                self.mask_existence[self.current_index] = True
                self.file_list_updated.emit(self.mask_existence)
    def delete_current_mask(self):
        if self.current_mask is None or self.current_index == -1: return
        image_path = self.image_paths[self.current_index]
        mask_filename = os.path.splitext(os.path.basename(image_path))[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_filename)
        if os.path.exists(mask_path): os.remove(mask_path); print(f"已删除掩码: {mask_path}")
        h, w = self.current_image.shape[:2]
        self.current_mask = np.zeros((h, w), dtype=np.uint8)
        self.mask_existence[self.current_index] = False
        self.mask_updated.emit(); self.file_list_updated.emit(self.mask_existence)
    def delete_current_image_and_mask(self):
        if self.current_index == -1: return
        image_path = self.image_paths[self.current_index]
        mask_filename = os.path.splitext(os.path.basename(image_path))[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_filename)
        if os.path.exists(image_path): os.remove(image_path); print(f"已删除图片: {image_path}")
        if os.path.exists(mask_path): os.remove(mask_path); print(f"已删除掩码: {mask_path}")
        self.image_paths.pop(self.current_index); self.mask_existence.pop(self.current_index)
        next_index = min(self.current_index, len(self.image_paths) - 1)
        self.navigate_to(next_index, save_before_nav=False)
    def next_image(self):
        if self.current_index < len(self.image_paths) - 1: self.navigate_to(self.current_index + 1)
    def prev_image(self):
        if self.current_index > 0: self.navigate_to(self.current_index - 1)
    def set_brush_size(self, size):
        self.brush_size = max(1, min(size, 200)); self.mask_updated.emit()
    def set_class_id(self, class_id):
        self.current_class_id = class_id; print(f"切换到类别: {self.current_class_id}")
    def add_to_undo_stack(self, is_invert_operation=False):
        if len(self.undo_stack) >= self.max_undo_steps: self.undo_stack.pop(0)
        # 检查current_mask是否为None
        if self.current_mask is not None:
            self.undo_stack.append(self.current_mask.copy())
        else:
            # 如果当前掩码为None，添加一个空掩码
            if self.current_image is not None:
                empty_mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
                self.undo_stack.append(empty_mask)
        
        # 非取反操作会清除矩形分割取反标志
        if not is_invert_operation:
            self.can_invert_rect_seg = False
    def undo(self):
        if self.mode == 'sam_assist':
            # SAM模式下的撤销：按时间顺序撤销最后添加的点
            if self.sam_point_history:
                # 获取最后添加的点
                last_point = self.sam_point_history.pop()
                x, y, is_positive = last_point
                
                # 从对应的列表中移除
                if is_positive:
                    if (x, y) in self.sam_positive_points:
                        self.sam_positive_points.remove((x, y))
                        print(f"撤销正点击: ({x}, {y})")
                else:
                    if (x, y) in self.sam_negative_points:
                        self.sam_negative_points.remove((x, y))
                        print(f"撤销负点击: ({x}, {y})")
                
                # 重新生成预测
                self._generate_sam_prediction()
                self.mask_updated.emit()
            else:
                # 没有点击点时，撤销到普通模式的上一步
                if self.undo_stack:
                    self.current_mask = self.undo_stack.pop()
                    self.set_mode('draw')
                    self.mask_updated.emit()
                    print("撤销SAM标注，返回绘制模式")
        else:
            # 普通撤销
            if self.undo_stack: 
                self.current_mask = self.undo_stack.pop()
                self.mask_updated.emit()
                print("操作已撤销")
        
        # 任何撤销操作都清除矩形分割取反标志
        self.can_invert_rect_seg = False
    
    # SAM2.1辅助标注相关方法
    def init_sam_model(self):
        """初始化SAM2.1模型（延迟加载，带异常处理）"""
        if self.sam_model is not None:
            return True
            
        try:
            # 检查PyTorch
            try:
                import torch
            except ImportError:
                print("错误: PyTorch未安装，请先安装: pip install torch torchvision")
                return False
            
            # 检查SAM2
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError:
                print("错误: SAM2未安装，请运行: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
                return False
            
            # 自动检测设备
            if torch.cuda.is_available():
                self.sam_device = 'cuda'
                print("使用CUDA加速SAM2.1模型")
            else:
                self.sam_device = 'cpu'
                print("使用CPU运行SAM2.1模型")
            
            # 查找模型文件
            model_candidates = [
                ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml", "Tiny (149MB)"),
                ("checkpoints/sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml", "Tiny (149MB)"),
                ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml", "Small"),
                ("checkpoints/sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml", "Small"),
            ]
            
            found_checkpoint = None
            found_config = None
            found_name = None
            
            for checkpoint, config, name in model_candidates:
                if os.path.exists(checkpoint):
                    # SAM2的配置文件在包内部，我们只需要配置名称
                    found_checkpoint = checkpoint
                    found_config = config
                    found_name = name
                    break
            
            if found_checkpoint is None:
                print("错误: 未找到SAM2.1模型文件")
                print("请下载模型到项目根目录或checkpoints/文件夹:")
                print("https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt")
                return False
            
            print(f"加载SAM2.1模型: {found_name} - {found_checkpoint}")
            
            # 构建模型（使用异常处理）
            sam2_model = build_sam2(found_config, found_checkpoint, device=self.sam_device)
            self.sam_predictor = SAM2ImagePredictor(sam2_model)
            self.sam_model = sam2_model
            
            print("SAM2.1模型初始化成功")
            return True
            
        except Exception as e:
            print(f"SAM2.1模型初始化失败: {e}")
            print("请检查模型文件是否正确下载")
            return False
    
    def generate_image_embedding(self):
        """为当前图像生成embedding（带异常处理）"""
        if self.current_image is None or self.sam_predictor is None:
            return False
            
        try:
            # SAM2需要RGB格式
            rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            
            # 设置图像
            self.sam_predictor.set_image(rgb_image)
            self.current_embedding = True
            print("SAM2.1图像embedding生成完成")
            return True
            
        except Exception as e:
            print(f"生成embedding失败: {e}")
            self.current_embedding = False
            return False
    
    def add_sam_point(self, x, y, is_positive=True):
        """添加SAM点击点"""
        if is_positive:
            self.sam_positive_points.append((x, y))
            print(f"添加正点击: ({x}, {y})")
        else:
            self.sam_negative_points.append((x, y))
            print(f"添加负点击: ({x}, {y})")
        
        # 添加到历史记录（按时间顺序）
        self.sam_point_history.append((x, y, is_positive))
        
        # 生成新的预测
        self._generate_sam_prediction()
    
    def _generate_sam_prediction(self):
        """根据当前点击点生成SAM2.1预测（带异常处理）"""
        if self.sam_predictor is None:
            print("SAM预测器未初始化")
            return
            
        if not self.current_embedding:
            print("图像embedding未生成")
            return
        
        if not self.sam_positive_points:
            print("没有正点击，清空预测")
            self.sam_temp_mask = None
            return
            
        try:
            # 准备输入点和标签
            points = []
            labels = []
            
            print(f"正点击数量: {len(self.sam_positive_points)}")
            print(f"负点击数量: {len(self.sam_negative_points)}")
            
            for x, y in self.sam_positive_points:
                points.append([x, y])
                labels.append(1)  # 正点
                
            for x, y in self.sam_negative_points:
                points.append([x, y])
                labels.append(0)  # 负点
            
            points = np.array(points, dtype=np.float32)
            labels = np.array(labels, dtype=np.int32)
            
            print(f"输入点坐标: {points}")
            print(f"输入点标签: {labels}")
            
            # 生成预测
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=False
            )
            
            print(f"预测结果 - masks形状: {masks.shape}, scores: {scores}")
            
            if len(masks) > 0 and masks.shape[0] > 0:
                # 取第一个掩码，转换为与当前掩码相同的格式
                sam_mask = masks[0].astype(np.uint8)
                print(f"原始掩码值范围: {sam_mask.min()} - {sam_mask.max()}")
                print(f"掩码形状: {sam_mask.shape}")
                print(f"掩码中True像素数量: {np.sum(sam_mask)}")
                
                # 将二值掩码转换为当前类别的掩码
                self.sam_temp_mask = sam_mask * self.current_class_id
                print(f"SAM2.1预测生成完成，置信度: {scores[0]:.3f}")
                print(f"转换后掩码值范围: {self.sam_temp_mask.min()} - {self.sam_temp_mask.max()}")
            else:
                print("预测结果为空")
                self.sam_temp_mask = None
                
        except Exception as e:
            print(f"SAM2.1预测失败: {e}")
            import traceback
            traceback.print_exc()
            self.sam_temp_mask = None
    
    def apply_sam_prediction(self):
        """应用SAM预测结果到当前掩码"""
        if self.sam_temp_mask is None:
            return False
            
        # 保存到撤销栈
        self.add_to_undo_stack()
        
        # 初始化current_mask如果为None
        if self.current_mask is None:
            if self.current_image is not None:
                self.current_mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
            else:
                return False
        
        # 将SAM结果与现有掩码合并（不覆盖已有标注）
        self.current_mask = np.maximum(self.current_mask, self.sam_temp_mask)
        
        # 清理SAM状态
        self.clear_sam_state()
        
        # 清除矩形分割取反标志
        self.can_invert_rect_seg = False
        
        # 切换回绘制模式
        self.set_mode('draw')
        
        self.mask_updated.emit()
        print("SAM标注已应用")
        return True
    
    def clear_sam_state(self, reset_embedding=False):
        """清理SAM辅助标注状态"""
        self.sam_positive_points.clear()
        self.sam_negative_points.clear()
        self.sam_point_history.clear()  # 清理历史记录
        self.sam_temp_mask = None
        if reset_embedding:
            self.current_embedding = False  # 只在切换图片时重置embedding
        print("SAM状态已清理")
    
    def clear_rect_seg_invert_flag(self):
        """清除矩形分割取反标志（用于非取反的其他操作）"""
        self.can_invert_rect_seg = False
    
    def invert_last_rect_segmentation(self):
        """取反最后一次矩形分割的结果"""
        if not self.can_invert_rect_seg:
            print("当前无法进行矩形分割取反操作")
            return False
            
        if self.last_rect_seg_region is None or self.last_rect_seg_mask is None:
            print("没有可取反的矩形分割结果")
            return False
            
        if self.current_mask is None:
            print("当前掩码为空，无法取反")
            return False
            
        try:
            # 保存到撤销栈（标记为取反操作）
            self.add_to_undo_stack(is_invert_operation=True)
            
            x1, y1, x2, y2 = self.last_rect_seg_region
            
            # 获取原始区域掩码
            original_roi_mask = self.current_mask[y1:y2, x1:x2].copy()
            
            # 创建取反掩码：在分割区域内，原来是0的变成当前类别，原来是当前类别的变成0
            inverted_mask = self.last_rect_seg_mask.copy()
            
            # 取反逻辑：255变成0，0变成255
            inverted_binary = 255 - inverted_mask
            
            # 转换为类别掩码
            inverted_class_mask = (inverted_binary // 255) * self.current_class_id
            
            # 应用取反结果：先清除原来的分割结果，再应用取反结果
            # 在分割区域内，保留非当前类别的标注，然后应用取反结果
            roi_without_current_class = original_roi_mask.copy()
            roi_without_current_class[self.last_rect_seg_mask == 255] = 0  # 清除原分割结果
            
            # 应用取反结果
            new_roi_mask = np.maximum(roi_without_current_class, inverted_class_mask)
            
            # 写回主掩码
            self.current_mask[y1:y2, x1:x2] = new_roi_mask
            
            # 更新最后分割掩码为取反后的结果
            self.last_rect_seg_mask = inverted_binary
            
            # 取反后禁用进一步的取反操作
            self.can_invert_rect_seg = False
            
            self.mask_updated.emit()
            print("矩形分割结果已取反")
            return True
            
        except Exception as e:
            print(f"取反矩形分割失败: {e}")
            return False