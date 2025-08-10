# 文件: annotation_canvas.py
import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPixmap, QImage, QPen, QColor, QCursor
from PyQt5.QtCore import Qt, QPoint, QPointF, QRectF

class AnnotationCanvas(QWidget):
    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.setMouseTracking(True)
        # ... 原有属性 ...
        self.left_mouse_down = False; self.right_mouse_down = False; self.middle_mouse_down = False
        self.last_pan_pos = QPointF(); self.zoom_level = 1.0; self.pan_offset = QPointF(0.0, 0.0)
        self.last_draw_pos = None; self.class_colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}

        # 新增：矩形选择模式相关属性
        self.rect_start_pos = None # 矩形选择的起始点 (视图坐标)
        self.rect_end_pos = None   # 矩形选择的结束点 (视图坐标)

        # 新增：SAM辅助标注模式相关属性
        self.sam_point_radius = 6  # SAM点击点的显示半径

        # 连接模式切换信号
        self.app_state.mode_changed.connect(self.on_mode_change)

    def on_mode_change(self, mode):
        """模式切换时的清理工作"""
        # 从矩形模式切换走时，清除未完成的矩形
        self.rect_start_pos = None
        self.rect_end_pos = None
        
        # 根据模式更新鼠标指针
        if mode == 'rect_seg':
            self.setCursor(Qt.CrossCursor)
        elif mode == 'sam_assist':
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor) # ArrowCursor 会被笔刷预览覆盖
        self.update()

    def paintEvent(self, event):
        # ... (原有 paintEvent 逻辑) ...
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.app_state.current_image is None:
            painter.fillRect(self.rect(), Qt.black); return
        painter.save()
        painter.translate(self.pan_offset); painter.scale(self.zoom_level, self.zoom_level)
        blended_image = self._get_blended_image()
        h, w, ch = blended_image.shape
        q_img = QImage(blended_image.data, w, h, w * ch, QImage.Format_RGB888).rgbSwapped()
        painter.drawPixmap(0, 0, QPixmap.fromImage(q_img))
        painter.restore()

        # 绘制 UI 元素（不受缩放影响）
        if self.app_state.mode == 'draw':
            self._draw_brush_preview(painter)
        elif self.app_state.mode == 'rect_seg':
            self._draw_selection_rect(painter)
        elif self.app_state.mode == 'sam_assist':
            self._draw_sam_points(painter)

    def _draw_selection_rect(self, painter):
        """在视图上绘制正在选择的矩形"""
        if self.rect_start_pos and self.rect_end_pos:
            painter.setPen(QPen(QColor(255, 255, 0, 200), 2, Qt.DashLine))
            painter.setBrush(QColor(255, 255, 0, 50))
            rect = QRectF(self.rect_start_pos, self.rect_end_pos).normalized()
            painter.drawRect(rect)

    def _draw_sam_points(self, painter):
        """绘制SAM辅助标注的点击点"""
        painter.save()
        
        # 绘制正点击点（绿色圆圈）
        for x, y in self.app_state.sam_positive_points:
            view_pos = self._image_to_view_coords(x, y)
            if view_pos:
                painter.setPen(QPen(QColor(0, 255, 0), 3))
                painter.setBrush(QColor(0, 255, 0, 100))
                painter.drawEllipse(view_pos, self.sam_point_radius * 2, self.sam_point_radius * 2)
                # 绘制十字标记
                painter.setPen(QPen(QColor(255, 255, 255), 2))
                painter.drawLine(int(view_pos.x() - self.sam_point_radius), int(view_pos.y()), 
                               int(view_pos.x() + self.sam_point_radius), int(view_pos.y()))
                painter.drawLine(int(view_pos.x()), int(view_pos.y() - self.sam_point_radius),
                               int(view_pos.x()), int(view_pos.y() + self.sam_point_radius))
        
        # 绘制负点击点（红色圆圈）
        for x, y in self.app_state.sam_negative_points:
            view_pos = self._image_to_view_coords(x, y)
            if view_pos:
                painter.setPen(QPen(QColor(255, 0, 0), 3))
                painter.setBrush(QColor(255, 0, 0, 100))
                painter.drawEllipse(view_pos, self.sam_point_radius * 2, self.sam_point_radius * 2)
                # 绘制X标记
                painter.setPen(QPen(QColor(255, 255, 255), 2))
                painter.drawLine(int(view_pos.x() - self.sam_point_radius//2), int(view_pos.y() - self.sam_point_radius//2),
                               int(view_pos.x() + self.sam_point_radius//2), int(view_pos.y() + self.sam_point_radius//2))
                painter.drawLine(int(view_pos.x() - self.sam_point_radius//2), int(view_pos.y() + self.sam_point_radius//2),
                               int(view_pos.x() + self.sam_point_radius//2), int(view_pos.y() - self.sam_point_radius//2))
        
        painter.restore()

    def mousePressEvent(self, event):
        # SAM辅助标注模式的逻辑
        if self.app_state.mode == 'sam_assist':
            # 处理中键拖拽（平移功能）
            if event.button() == Qt.MiddleButton:
                self.middle_mouse_down = True
                self.last_pan_pos = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
                return
            
            # SAM点击逻辑
            img_coords = self._view_to_image_coords(event.pos())
            if img_coords:
                x, y = img_coords
                if event.button() == Qt.LeftButton:
                    # 左键添加正点击点
                    self.app_state.add_sam_point(x, y, is_positive=True)
                elif event.button() == Qt.RightButton:
                    # 右键添加负点击点
                    self.app_state.add_sam_point(x, y, is_positive=False)
                self.update()
            return

        # 矩形分割模式的逻辑
        if self.app_state.mode == 'rect_seg':
            if event.button() == Qt.LeftButton:
                self.rect_start_pos = event.pos()
                self.rect_end_pos = event.pos()
                self.left_mouse_down = True
            elif event.button() == Qt.RightButton:
                # 右键取消矩形模式，返回绘制模式
                self.app_state.set_mode('draw')
            return

        # 绘制模式的逻辑 (和之前类似)
        if event.button() == Qt.LeftButton:
            self.left_mouse_down = True; self.app_state.add_to_undo_stack()
            self.last_draw_pos = self._draw_at_pos(event.pos())
        elif event.button() == Qt.RightButton:
            self.right_mouse_down = True; self.app_state.add_to_undo_stack()
            self.last_draw_pos = self._draw_at_pos(event.pos())
        elif event.button() == Qt.MiddleButton:
            self.middle_mouse_down = True; self.last_pan_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        # SAM辅助标注模式：支持中键拖拽平移，但不处理绘制拖拽
        if self.app_state.mode == 'sam_assist':
            if self.middle_mouse_down:
                delta = QPointF(event.pos() - self.last_pan_pos)
                self.pan_offset += delta
                self.last_pan_pos = event.pos()
                self.update()
            else:
                self.update()  # 更新鼠标位置显示
            return
            
        # 矩形分割模式的逻辑
        if self.app_state.mode == 'rect_seg':
            if self.left_mouse_down:
                self.rect_end_pos = event.pos()
                self.update()
            return
        
        # 绘制模式的逻辑
        if self.left_mouse_down or self.right_mouse_down:
            current_pos = self._draw_at_pos(event.pos(), connect_from=self.last_draw_pos)
            self.last_draw_pos = current_pos
        elif self.middle_mouse_down:
            delta = QPointF(event.pos() - self.last_pan_pos)
            self.pan_offset += delta; self.last_pan_pos = event.pos()
            self.update()
        else:
            self.update()

    def mouseReleaseEvent(self, event):
        # SAM辅助标注模式：只处理中键释放
        if self.app_state.mode == 'sam_assist':
            if event.button() == Qt.MiddleButton:
                self.middle_mouse_down = False
                self.setCursor(Qt.ArrowCursor)
            return
            
        # 矩形分割模式的逻辑
        if self.app_state.mode == 'rect_seg':
            if event.button() == Qt.LeftButton and self.left_mouse_down:
                self.left_mouse_down = False
                self._perform_auto_segmentation()
                # 完成后自动返回绘制模式
                self.app_state.set_mode('draw')
            return
            
        # 绘制模式的逻辑
        if event.button() == Qt.LeftButton: self.left_mouse_down = False
        if event.button() == Qt.RightButton: self.right_mouse_down = False
        if event.button() == Qt.MiddleButton:
            self.middle_mouse_down = False; self.setCursor(Qt.ArrowCursor)
        self.last_draw_pos = None

    def _perform_auto_segmentation(self):
        """执行自动分割的核心逻辑"""
        if not self.rect_start_pos or not self.rect_end_pos: return

        # 1. 将视图坐标的矩形转换为图像坐标
        p1_img = self._view_to_image_coords(self.rect_start_pos)
        p2_img = self._view_to_image_coords(self.rect_end_pos)
        
        if p1_img is None or p2_img is None: return

        x1, y1 = min(p1_img[0], p2_img[0]), min(p1_img[1], p2_img[1])
        x2, y2 = max(p1_img[0], p2_img[0]), max(p1_img[1], p2_img[1])

        # 确保矩形在图像范围内
        img_h, img_w = self.app_state.current_image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x1 >= x2 or y1 >= y2: return # 矩形无效

        # 2. 为撤销做准备
        self.app_state.add_to_undo_stack()

        # 3. 提取ROI并进行处理
        roi_image = self.app_state.current_image[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        
        # 使用Otsu's方法自动确定最佳阈值
        thresh_val, binary_mask = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 4. 将处理结果应用到主掩码
        # 将二值化结果（0或255）转换为（0或当前类别ID）
        class_mask = (binary_mask // 255) * self.app_state.current_class_id
        
        # 只在新增区域进行标注，不覆盖已有标注
        # current_roi_mask 是当前掩码图上对应区域的副本
        current_roi_mask = self.app_state.current_mask[y1:y2, x1:x2]
        # new_mask 是在 current_roi_mask 的基础上，加上新的标注
        new_mask = np.maximum(current_roi_mask, class_mask)
        
        # 将更新后的区域写回主掩码
        self.app_state.current_mask[y1:y2, x1:x2] = new_mask

        # 5. 清理并更新UI
        self.rect_start_pos = None
        self.rect_end_pos = None
        self.app_state.mask_updated.emit()
        print(f"自动分割完成，Otsu阈值: {thresh_val}")


    # ... 其他所有方法保持不变 ...
    def fit_to_view(self): ...
    def resizeEvent(self, event): ...
    def zoom(self, factor): ...
    def _view_to_image_coords(self, view_pos): ...
    def _draw_at_pos(self, view_pos, connect_from=None): ...
    def _draw_brush_preview(self, painter): ...
    def _get_blended_image(self): ...
    # 为了简洁，这里省略了未改动的方法体，请在您的代码中保留它们
    def fit_to_view(self):
        if self.app_state.current_image is None: self.update(); return
        img_h, img_w = self.app_state.current_image.shape[:2]
        scale_w = self.width() / img_w
        scale_h = self.height() / img_h
        self.zoom_level = min(scale_w, scale_h)
        scaled_w = img_w * self.zoom_level
        scaled_h = img_h * self.zoom_level
        self.pan_offset = QPointF((self.width() - scaled_w) / 2, (self.height() - scaled_h) / 2)
        self.update()
    def resizeEvent(self, event):
        self.fit_to_view()
    def zoom(self, factor):
        old_zoom = self.zoom_level
        self.zoom_level *= factor
        self.zoom_level = max(0.1, min(self.zoom_level, 20))
        mouse_pos = self.mapFromGlobal(QCursor.pos())
        pan_factor = self.zoom_level / old_zoom
        self.pan_offset = mouse_pos - (mouse_pos - self.pan_offset) * pan_factor
        self.update()
    def _view_to_image_coords(self, view_pos):
        if self.app_state.current_image is None: return None
        img_x = (view_pos.x() - self.pan_offset.x()) / self.zoom_level
        img_y = (view_pos.y() - self.pan_offset.y()) / self.zoom_level
        img_h, img_w = self.app_state.current_image.shape[:2]
        if 0 <= img_x < img_w and 0 <= img_y < img_h:
            return (int(img_x), int(img_y))
        return None
    
    def _image_to_view_coords(self, img_x, img_y):
        """将图像坐标转换为视图坐标"""
        if self.app_state.current_image is None: 
            return None
        view_x = img_x * self.zoom_level + self.pan_offset.x()
        view_y = img_y * self.zoom_level + self.pan_offset.y()
        return QPointF(view_x, view_y)
    def _draw_at_pos(self, view_pos, connect_from=None):
        img_coords = self._view_to_image_coords(view_pos)
        if img_coords is None: return None
        draw_value = 0
        if self.left_mouse_down: draw_value = self.app_state.current_class_id
        elif self.right_mouse_down: draw_value = 0
        brush_thickness = self.app_state.brush_size * 2
        if connect_from and connect_from != img_coords:
            cv2.line(self.app_state.current_mask, pt1=connect_from, pt2=img_coords, color=draw_value, thickness=brush_thickness, lineType=cv2.LINE_8)
        cv2.circle(self.app_state.current_mask, center=img_coords, radius=self.app_state.brush_size, color=draw_value, thickness=-1)
        self.app_state.mask_updated.emit()
        return img_coords
    def _draw_brush_preview(self, painter):
        painter.setPen(QPen(Qt.white, 1, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        scaled_brush_radius = self.app_state.brush_size * self.zoom_level
        current_pos = self.mapFromGlobal(QCursor.pos())
        painter.drawEllipse(current_pos, scaled_brush_radius, scaled_brush_radius)
    def _get_blended_image(self):
        image = self.app_state.current_image
        mask = self.app_state.current_mask
        color_mask = np.zeros_like(image, dtype=np.uint8)
        
        # 渲染现有掩码
        for class_id, color in self.class_colors.items():
            color_mask[mask == class_id] = color
        
        # 如果在SAM模式下且有临时掩码，以半透明方式叠加显示
        if self.app_state.mode == 'sam_assist' and self.app_state.sam_temp_mask is not None:
            # 创建SAM预览掩码（使用当前类别颜色，但更透明）
            sam_color = self.class_colors.get(self.app_state.current_class_id, (255, 255, 255))
            sam_preview = np.zeros_like(image, dtype=np.uint8)
            sam_preview[self.app_state.sam_temp_mask > 0] = sam_color
            
            # 先混合现有掩码，再叠加SAM预览
            blended = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)
            final_result = cv2.addWeighted(blended, 0.8, sam_preview, 0.2, 0)
            return final_result
        else:
            return cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)