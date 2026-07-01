"""
增强型图片查看器
基于 QGraphicsView 实现缩放、拖拽平移、适应窗口功能。
"""

import logging
import math

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QVBoxLayout, QHBoxLayout, QWidget, QLabel, QSlider, QPushButton
)
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QWheelEvent, QMouseEvent,
    QColor, QTransform
)

from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class ZoomableImageViewer(QGraphicsView):
    """支持鼠标滚轮缩放 + 拖拽平移的图片查看器"""

    zoomChanged = Signal(float)  # 当前缩放比例

    def __init__(self, title: str = ""):
        super().__init__()
        self._title = title
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._original_pixmap: QPixmap | None = None
        self._current_zoom = 1.0
        self._fit_enabled = True

        # 拖拽状态
        self._is_dragging = False
        self._drag_start_pos = None

        # 外观设置
        self.setRenderHints(
            QPainter.RenderHint.SmoothPixmapTransform |
            QPainter.RenderHint.Antialiasing
        )
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setMinimumSize(350, 350)
        self.setStyleSheet("""
            QGraphicsView {
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #f0f0f0;
            }
        """)

        # placeholder 文字
        placeholder = self._scene.addText(self._title)
        placeholder.setDefaultTextColor(QColor("#999"))
        font = placeholder.font()
        font.setPointSize(16)
        placeholder.setFont(font)
        placeholder.setPos(self.width() / 2 - 50, self.height() / 2 - 10)

    def setImage(self, pil_image: Image.Image, show_checkerboard: bool = False):
        """设置显示图片"""
        display_image = pil_image.copy()

        if show_checkerboard and pil_image.mode == 'RGBA':
            display_image = self._create_checkerboard_bg(pil_image)

        if display_image.mode == 'RGBA':
            data = display_image.tobytes('raw', 'RGBA')
            qimage = QImage(data, display_image.width, display_image.height,
                            QImage.Format.Format_RGBA8888)
        else:
            data = display_image.tobytes('raw', 'RGB')
            qimage = QImage(data, display_image.width, display_image.height,
                            QImage.Format.Format_RGB888)

        self._original_pixmap = QPixmap.fromImage(qimage)

        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(self._original_pixmap)
        self._scene.setSceneRect(QRectF(self._original_pixmap.rect()))

        if self._fit_enabled:
            self.fitInView()

    def _create_checkerboard_bg(self, pil_image: Image.Image, tile_size: int = 20) -> Image.Image:
        """创建棋盘格透明背景指示图"""
        width, height = pil_image.size
        board = Image.new('RGBA', (width, height), (255, 255, 255, 255))
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                if ((x // tile_size) + (y // tile_size)) % 2 == 0:
                    box = (x, y, min(x + tile_size, width), min(y + tile_size, height))
                    tile = Image.new('RGBA', (box[2] - box[0], box[3] - box[1]),
                                     (200, 200, 200, 255))
                    board.paste(tile, box)
        board.paste(pil_image, mask=pil_image.split()[3])
        return board

    def clear(self):
        """清空图片"""
        self._scene.clear()
        self._original_pixmap = None
        self._pixmap_item = None
        self._current_zoom = 1.0
        self._fit_enabled = True

        # 重新显示 placeholder
        placeholder = self._scene.addText(self._title)
        placeholder.setDefaultTextColor(QColor("#999"))
        font = placeholder.font()
        font.setPointSize(16)
        placeholder.setFont(font)
        placeholder.setPos(self.width() / 2 - 50, self.height() / 2 - 10)

    # ── 缩放控制 ──

    def setZoom(self, factor: float):
        """设置缩放倍率 (1.0 = 100%)"""
        if not self._original_pixmap:
            return

        factor = max(0.1, min(factor, 20.0))
        self._fit_enabled = False

        transform = QTransform().scale(factor, factor)
        self.setTransform(transform)
        self._current_zoom = factor
        self.zoomChanged.emit(factor)

    def fitInView(self):
        """适应窗口显示"""
        if not self._pixmap_item:
            return

        self._fit_enabled = True
        super().fitInView(
            self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio
        )
        # 计算实际缩放比例
        self._current_zoom = self.transform().m11()
        self.zoomChanged.emit(self._current_zoom)

    def zoomIn(self):
        """放大"""
        self.setZoom(self._current_zoom * 1.25)

    def zoomOut(self):
        """缩小"""
        self.setZoom(self._current_zoom / 1.25)

    def zoomToActual(self):
        """100% 原图大小"""
        self.setZoom(1.0)

    def is_empty(self) -> bool:
        """检查是否没有图片"""
        return self._original_pixmap is None

    # ── 事件处理 ──

    def wheelEvent(self, event: QWheelEvent):
        """滚轮缩放"""
        if self.is_empty():
            super().wheelEvent(event)
            return

        delta = event.angleDelta().y()
        if delta > 0:
            self.zoomIn()
        else:
            self.zoomOut()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下：进入拖拽模式"""
        if (event.button() == Qt.MouseButton.LeftButton
                and not self.is_empty()):
            self._is_dragging = True
            self._drag_start_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动：拖拽平移"""
        if self._is_dragging and self._drag_start_pos:
            delta = event.pos() - self._drag_start_pos
            self._drag_start_pos = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放：退出拖拽模式"""
        if event.button() == Qt.MouseButton.LeftButton and self._is_dragging:
            self._is_dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        """窗口大小变化时自适应"""
        super().resizeEvent(event)
        if self._fit_enabled and not self.is_empty():
            self.fitInView()


class ViewerControls(QWidget):
    """图片查看器底部的缩放控制栏"""

    zoomInClicked = Signal()
    zoomOutClicked = Signal()
    fitClicked = Signal()
    actualClicked = Signal()
    zoomChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(6)

        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setFixedSize(28, 28)
        self.btn_zoom_out.setToolTip("缩小")
        self.btn_zoom_out.setStyleSheet(self._button_style())

        self.zoom_label = QLabel("100%")
        self.zoom_label.setFixedWidth(45)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setStyleSheet("font-size: 11px; color: #666;")

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedSize(28, 28)
        self.btn_zoom_in.setToolTip("放大")
        self.btn_zoom_in.setStyleSheet(self._button_style())

        self.btn_fit = QPushButton("适应")
        self.btn_fit.setFixedSize(40, 28)
        self.btn_fit.setToolTip("适应窗口")
        self.btn_fit.setStyleSheet(self._button_style())

        self.btn_actual = QPushButton("1:1")
        self.btn_actual.setFixedSize(36, 28)
        self.btn_actual.setToolTip("原始大小")
        self.btn_actual.setStyleSheet(self._button_style())

        layout.addStretch()
        layout.addWidget(self.btn_zoom_out)
        layout.addWidget(self.zoom_label)
        layout.addWidget(self.btn_zoom_in)
        layout.addWidget(self.btn_fit)
        layout.addWidget(self.btn_actual)
        layout.addStretch()

        # 信号转发
        self.btn_zoom_out.clicked.connect(self.zoomOutClicked.emit)
        self.btn_zoom_in.clicked.connect(self.zoomInClicked.emit)
        self.btn_fit.clicked.connect(self.fitClicked.emit)
        self.btn_actual.clicked.connect(self.actualClicked.emit)

    def updateZoomLabel(self, factor: float):
        self.zoom_label.setText(f"{int(round(factor * 100))}%")

    @staticmethod
    def _button_style() -> str:
        return """
            QPushButton {
                font-size: 13px;
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 4px;
                background: #fafafa;
                color: #333;
                padding: 0px;
            }
            QPushButton:hover {
                background: #e8e8e8;
                border-color: #4a90d9;
            }
            QPushButton:pressed {
                background: #d0d0d0;
            }
        """
