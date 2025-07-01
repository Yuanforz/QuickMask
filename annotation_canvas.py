# 文件: annotation_canvas.py
import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPixmap, QImage, QPen, QColor, QCursor
from PyQt5.QtCore import Qt, QPoint, QPointF

class AnnotationCanvas(QWidget):
    # ... __init__ 和其他方法保持不变 ...
    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.setMouseTracking(True)
        self.left_mouse_down = False
        self.right_mouse_down = False
        self.middle_mouse_down = False
        self.last_pan_pos = QPointF()
        self.zoom_level = 1.0
        self.pan_offset = QPointF(0.0, 0.0)
        self.last_draw_pos = None
        self.class_colors = {
            1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255),
        }

    # ... mousePressEvent, mouseMoveEvent, mouseReleaseEvent 保持不变 ...
    def mousePressEvent(self, event):
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
        if event.button() == Qt.LeftButton: self.left_mouse_down = False
        if event.button() == Qt.RightButton: self.right_mouse_down = False
        if event.button() == Qt.MiddleButton:
            self.middle_mouse_down = False; self.setCursor(Qt.ArrowCursor)
        self.last_draw_pos = None
        
    # <--- CHANGED --->
    # _draw_at_pos 方法中的 cv2.line() 的 lineType 被修改
    def _draw_at_pos(self, view_pos, connect_from=None):
        img_coords = self._view_to_image_coords(view_pos)
        if img_coords is None:
            return None

        draw_value = 0
        if self.left_mouse_down:
            draw_value = self.app_state.current_class_id
        elif self.right_mouse_down:
            draw_value = 0

        brush_thickness = self.app_state.brush_size * 2
        
        if connect_from and connect_from != img_coords:
            # 关键修复：移除抗锯齿。使用 cv2.LINE_8（8邻域连接）代替 cv2.LINE_AA
            cv2.line(
                self.app_state.current_mask,
                pt1=connect_from,
                pt2=img_coords,
                color=draw_value,
                thickness=brush_thickness,
                lineType=cv2.LINE_8 
            )
        
        cv2.circle(
            self.app_state.current_mask,
            center=img_coords,
            radius=self.app_state.brush_size,
            color=draw_value,
            thickness=-1
        )

        self.app_state.mask_updated.emit()
        return img_coords

    # ... 其他所有方法保持不变 ...
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
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        # 注意：这里的 Antialiasing 是对 Qt 绘制而言的，比如笔刷预览的圆圈，这是好的，需要保留。
        # 我们修复的是 OpenCV 在后台对 numpy 数组操作时的行为。
        painter.setRenderHint(QPainter.Antialiasing)
        if self.app_state.current_image is None:
            painter.fillRect(self.rect(), Qt.black); return
        painter.save()
        painter.translate(self.pan_offset)
        painter.scale(self.zoom_level, self.zoom_level)
        blended_image = self._get_blended_image()
        h, w, ch = blended_image.shape
        q_img = QImage(blended_image.data, w, h, w * ch, QImage.Format_RGB888).rgbSwapped()
        painter.drawPixmap(0, 0, QPixmap.fromImage(q_img))
        painter.restore()
        self._draw_brush_preview(painter)
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
        for class_id, color in self.class_colors.items():
            color_mask[mask == class_id] = color
        return cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)