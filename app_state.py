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
        # ... (原有属性保持不变) ...
        super().__init__()
        self.image_dir = image_dir; self.mask_dir = mask_dir
        os.makedirs(self.image_dir, exist_ok=True); os.makedirs(self.mask_dir, exist_ok=True)
        self.image_paths = []; self.mask_existence = []
        self.current_index = -1; self.current_image = None; self.current_mask = None
        self.brush_size = 20; self.current_class_id = 1
        self.undo_stack = []; self.max_undo_steps = 10
        
        # 新增：应用模式状态 ('draw' 或 'rect_seg')
        self.mode = 'draw'

    def set_mode(self, mode: str):
        """设置应用模式，并发出信号通知UI"""
        if self.mode != mode:
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
            self.mask_existence[self.current_index] = True
        else:
            h, w = self.current_image.shape[:2]
            self.current_mask = np.zeros((h, w), dtype=np.uint8)
        self.undo_stack.clear()
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
    def add_to_undo_stack(self):
        if len(self.undo_stack) >= self.max_undo_steps: self.undo_stack.pop(0)
        self.undo_stack.append(self.current_mask.copy())
    def undo(self):
        if self.undo_stack: self.current_mask = self.undo_stack.pop(); self.mask_updated.emit(); print("操作已撤销")