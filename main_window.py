# 文件: main_window.py
import sys
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QSlider, QLabel, QLineEdit, QStatusBar, QMessageBox, QApplication)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor
from app_state import AppState
from annotation_canvas import AnnotationCanvas

# MaskIndicatorBar 类保持不变
class MaskIndicatorBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mask_data = []
        self.setFixedHeight(10)
    def set_mask_data(self, data: list):
        self.mask_data = data; self.update()
    def paintEvent(self, event):
        painter = QPainter(self)
        if not self.mask_data: return
        total_items = len(self.mask_data)
        item_width = self.width() / total_items
        for i, exists in enumerate(self.mask_data):
            color = QColor("#4CAF50") if exists else QColor("#F44336")
            painter.setBrush(color); painter.setPen(Qt.NoPen)
            painter.drawRect(int(i * item_width), 0, int(item_width + 1), self.height())


class MainWindow(QMainWindow):
    # ... __init__, _init_ui, _connect_signals, _update_ui_on_state_change 等方法保持不变 ...
    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state
        self.setWindowTitle("QuickMask - 快速图像分割标注工具")
        self.setGeometry(100, 100, 1200, 800)
        self._init_ui()
        self._connect_signals()
        
        # 启动时立即初始化SAM模型
        self._init_sam_model_async()

    def _init_ui(self):
        self.canvas = AnnotationCanvas(self.app_state)
        control_layout = QHBoxLayout()
        self.progress_label = QLabel("0/0")
        self.progress_slider = QSlider(Qt.Horizontal)
        self.jump_input = QLineEdit()
        self.jump_input.setPlaceholderText("跳转到(索引)...")
        self.jump_input.setFixedWidth(100)
        control_layout.addWidget(self.progress_label)
        control_layout.addWidget(self.progress_slider)
        control_layout.addWidget(self.jump_input)
        self.indicator_bar = MaskIndicatorBar()
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.canvas)
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.indicator_bar)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("")
        self.status_bar.addWidget(self.status_label)

    def _connect_signals(self):
        self.app_state.state_changed.connect(self._update_ui_on_state_change)
        self.app_state.mask_updated.connect(self.canvas.update)
        self.app_state.progress_updated.connect(self._update_progress)
        self.app_state.file_list_updated.connect(self.indicator_bar.set_mask_data)
        self.app_state.files_deleted.connect(self._on_files_deleted)
        self.progress_slider.valueChanged.connect(self._on_slider_change)
        self.jump_input.returnPressed.connect(self._on_jump_to_image)

    def _update_ui_on_state_change(self):
        if self.app_state.current_image is not None:
            total_images = len(self.app_state.image_paths)
            if total_images > 0:
              self.progress_slider.setRange(0, total_images - 1)
            self.setWindowTitle(f"QuickMask - {self.app_state.image_paths[self.app_state.current_index]}")
        self.app_state.set_mode('draw')
        self.canvas.fit_to_view()
        self._update_status_bar()

    def _init_sam_model_async(self):
        """异步初始化SAM模型"""
        def init_sam():
            self.status_label.setText("正在初始化SAM2.1模型...")
            success = self.app_state.init_sam_model()
            if success:
                self.status_label.setText("SAM2.1模型加载完成，可以使用E键进入辅助标注模式")
            else:
                self.status_label.setText("SAM2.1模型加载失败，将只使用基本功能")
        
        # 使用QTimer延迟执行，避免阻塞UI
        QTimer.singleShot(100, init_sam)

    def _on_files_deleted(self):
        total = len(self.app_state.image_paths)
        if total == 0:
            self.setWindowTitle("QuickMask - 无图片")
            self.progress_label.setText("0/0")
            self.progress_slider.setRange(0, 0)
            self.indicator_bar.set_mask_data([])
            self.canvas.update()
        else:
             self._update_progress(self.app_state.current_index, total)
             
    # <--- CHANGED --->
    # 唯一的改动在这里
    def _on_jump_to_image(self):
        try:
            # 如果输入框为空，则不执行任何操作
            if not self.jump_input.text():
                return
            index = int(self.jump_input.text()) - 1
            self.app_state.navigate_to(index)
        except (ValueError, IndexError):
            # 如果输入了无效数字或超出范围的索引，则忽略
            pass
        finally:
            # 无论跳转成功与否，都执行以下操作：
            self.jump_input.clear()  # 清空输入框
            self.canvas.setFocus()   # <<< THE FIX: 将键盘焦点设置回主画布
    
    # ... keyPressEvent, wheelEvent, 和其他方法保持不变 ...
    def keyPressEvent(self, event):
        if self.jump_input.hasFocus():
            super().keyPressEvent(event); return
        key = event.key(); modifiers = event.modifiers()
        
        # SAM辅助标注相关快捷键
        if key == Qt.Key_E:
            if self.app_state.mode == 'sam_assist':
                # 在SAM模式下按E表示拒绝并退出
                self.app_state.set_mode('draw')
                self.status_bar.showMessage("已退出SAM辅助标注模式", 2000)
            else:
                # 进入SAM模式
                self._enter_sam_mode()
        elif key == Qt.Key_Space and self.app_state.mode == 'sam_assist':
            # 在SAM模式下按空格接受并应用结果
            if self.app_state.apply_sam_prediction():
                self.status_bar.showMessage("SAM标注已应用", 2000)
            else:
                self.status_bar.showMessage("无SAM预测结果可应用", 2000)
            return
            
        # 在SAM模式下切换图片会自动退出SAM模式
        if self.app_state.mode == 'sam_assist' and key in (Qt.Key_A, Qt.Key_D):
            self.app_state.set_mode('draw')
            self.status_bar.showMessage("切换图片，已退出SAM模式", 2000)
        
        if key == Qt.Key_R:
            if self.app_state.mode == 'draw':
                self.app_state.set_mode('rect_seg')
            else:
                self.app_state.set_mode('draw')
        elif key == Qt.Key_X:
            reply = QMessageBox.question(self, '确认删除', '您确定要删除当前图片的【标注掩码】吗？\n此操作不可撤销。', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes: self.app_state.delete_current_mask()
        elif key == Qt.Key_Delete:
            reply = QMessageBox.question(self, '确认删除', '您确定要永久删除【当前图片及其标注】吗？\n此操作不可撤销！', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes: self.app_state.delete_current_image_and_mask()
        elif key == Qt.Key_D: self.app_state.next_image()
        elif key == Qt.Key_A: self.app_state.prev_image()
        elif key == Qt.Key_W: self.canvas.zoom(1.2); self._update_status_bar()
        elif key == Qt.Key_S and not modifiers: self.canvas.zoom(1 / 1.2); self._update_status_bar()
        elif key == Qt.Key_1: self.app_state.set_class_id(1); self._update_status_bar()
        elif key == Qt.Key_2: self.app_state.set_class_id(2); self._update_status_bar()
        elif key == Qt.Key_3: self.app_state.set_class_id(3); self._update_status_bar()
        elif key == Qt.Key_S and modifiers == Qt.ControlModifier:
            self.app_state.save_current_mask(); self.status_bar.showMessage("掩码已保存!", 2000)
        elif key == Qt.Key_Z and modifiers == Qt.ControlModifier: self.app_state.undo()
        else:
            super().keyPressEvent(event)
            
    def wheelEvent(self, event):
        if self.canvas.underMouse():
            delta = event.angleDelta().y()
            step = 2 if not (event.modifiers() & Qt.ShiftModifier) else 1
            if delta > 0: self.app_state.set_brush_size(self.app_state.brush_size + step)
            elif delta < 0: self.app_state.set_brush_size(self.app_state.brush_size - step)
            self._update_status_bar()
            self.canvas.update()

    def _update_progress(self, current, total):
        self.progress_label.setText(f"{current + 1}/{total}")
        self.progress_slider.blockSignals(True)
        self.progress_slider.setValue(current)
        self.progress_slider.blockSignals(False)

    def _on_slider_change(self, value):
        if not hasattr(self, 'slider_timer'):
            self.slider_timer = QTimer(); self.slider_timer.setSingleShot(True)
            self.slider_timer.timeout.connect(lambda: self.app_state.navigate_to(self.progress_slider.value()))
        self.slider_timer.start(200)

    def _update_status_bar(self):
        mode_names = {
            'draw': '绘制',
            'rect_seg': '矩形选择', 
            'sam_assist': 'SAM2.1辅助标注'
        }
        current_mode_text = mode_names.get(self.app_state.mode, '未知模式')
        
        class_name = f"类别 {self.app_state.current_class_id}"
        brush_size = f"笔刷 {self.app_state.brush_size}"
        zoom_text = f"缩放 {self.canvas.zoom_level:.2f}x"
        
        # 在SAM模式下显示额外信息
        if self.app_state.mode == 'sam_assist':
            pos_count = len(self.app_state.sam_positive_points)
            neg_count = len(self.app_state.sam_negative_points)
            sam_info = f"正点击:{pos_count} 负点击:{neg_count}"
            self.status_label.setText(f"模式: {current_mode_text} | {class_name} | {sam_info} | {zoom_text}")
        else:
            self.status_label.setText(f"模式: {current_mode_text} | {class_name} | {brush_size} | {zoom_text}")

    def _enter_sam_mode(self):
        """进入SAM辅助标注模式"""
        # 检查SAM模型是否已初始化
        if self.app_state.sam_predictor is None:
            if not self.app_state.init_sam_model():
                QMessageBox.warning(self, 'SAM模型错误', 
                                   'SAM模型初始化失败！\n请检查依赖安装和模型文件。')
                return
        
        # 检查是否有当前图像
        if self.app_state.current_image is None:
            QMessageBox.warning(self, '图像错误', '请先加载图像！')
            return
        
        # 立即为当前图像生成embedding
        self.status_label.setText("正在生成图像embedding...")
        QApplication.processEvents()  # 更新UI
        
        if not self.app_state.generate_image_embedding():
            QMessageBox.warning(self, 'SAM处理错误', 
                               '无法为当前图像生成embedding！')
            return
        
        # 清除之前的SAM状态
        self.app_state.clear_sam_state()
        
        # 切换到SAM模式
        self.app_state.set_mode('sam_assist')
        self.status_label.setText("SAM2.1辅助标注模式 - 左键正点击，右键负点击，空格接受，E键退出")

    def closeEvent(self, event):
        print("正在关闭，保存当前掩码..."); self.app_state.save_current_mask(); event.accept()