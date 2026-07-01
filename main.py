"""
图片抠图工具 — Main Program
Supports: load an image → AI background removal → transparent/green-screen → zoom/pan → save
All processing is manual, user-triggered. No automatic or hidden processing.
"""

import sys
import os
import logging

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QFrame, QSplitter,
    QColorDialog, QCheckBox, QSlider, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal, QPoint
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QDragEnterEvent, QDropEvent

from PIL import Image, UnidentifiedImageError
from io import BytesIO

from rembg import remove, new_session

from config import AppConfig
from image_utils import (
    preprocess_pipeline,
    postprocess_pipeline,
    auto_enhance,
    is_supported_image,
    validate_image_file,
    MAX_IMAGE_SIZE,
    SUPPORTED_IMAGE_EXTS,
)

try:
    from enhanced_viewer import ZoomableImageViewer, ViewerControls
    HAS_ENHANCED_VIEWER = True
except ImportError:
    HAS_ENHANCED_VIEWER = False

# ─── Logging ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("matting_tool.log", encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ]
)
logger = logging.getLogger("matting")


# ─── Constants ───────────────────────────────────────────

_BUTTON_PRIMARY_STYLE = """
    QPushButton {
        font-size: 15px;
        padding: 8px 20px;
        border-radius: 8px;
        background-color: #4a90d9;
        color: white;
        border: none;
        font-weight: bold;
    }
    QPushButton:hover { background-color: #3a7bc8; }
    QPushButton:disabled { background-color: #d0d0d0; color: #999; }
"""

_BUTTON_RESET_STYLE = """
    QPushButton {
        font-size: 15px;
        padding: 8px 20px;
        border-radius: 8px;
        background-color: #e74c3c;
        color: white;
        border: none;
        font-weight: bold;
    }
    QPushButton:hover { background-color: #c0392b; }
    QPushButton:disabled { background-color: #d0d0d0; color: #999; }
"""

_BUTTON_LOAD_STYLE = """
    QPushButton {
        font-size: 18px;
        padding: 16px 40px;
        border-radius: 12px;
        background-color: #27ae60;
        color: white;
        border: none;
        font-weight: bold;
    }
    QPushButton:hover { background-color: #219a52; }
    QPushButton:disabled { background-color: #d0d0d0; color: #999; }
"""

_DROP_AREA_DEFAULT_STYLE = """
    QFrame {
        border: 3px dashed #aaa;
        border-radius: 10px;
        background-color: #fafafa;
        min-height: 350px;
    }
    QFrame:hover {
        border-color: #4a90d9;
        background-color: #f0f8ff;
    }
"""

_DROP_AREA_DRAG_OVER_STYLE = """
    QFrame {
        border: 3px dashed #4a90d9;
        border-radius: 10px;
        background-color: #e6f3ff;
        min-height: 350px;
    }
"""

_TITLE_STYLE = """
    QLabel {
        font-size: 26px;
        font-weight: bold;
        color: #333;
        padding: 5px;
    }
"""

_STATUS_LOADING_STYLE = "color: #e67e22; font-size: 14px; padding: 5px; font-weight: bold;"
_STATUS_SUCCESS_STYLE = "color: #27ae60; font-size: 14px; padding: 5px; font-weight: bold;"
_STATUS_ERROR_STYLE = "color: #e74c3c; font-size: 14px; padding: 5px;"
_STATUS_INFO_STYLE = "color: #666; font-size: 14px; padding: 5px;"
_SUB_TITLE_STYLE = "font-size: 14px; font-weight: bold; color: #555; padding: 5px;"
_PROGRESS_STYLE = """
    QProgressBar {
        border: 1px solid #ddd;
        border-radius: 6px;
        background: #f0f0f0;
        text-align: center;
        font-size: 12px;
        height: 22px;
    }
    QProgressBar::chunk {
        background: #4a90d9;
        border-radius: 5px;
    }
"""


# ─── Worker Threads ──────────────────────────────────────

class ModelLoaderThread(QThread):
    """Loads the u2net AI model in a background thread."""
    finished = Signal(object)  # emits session object
    error = Signal(str)

    def run(self):
        try:
            logger.info("[model] loading u2net AI model (this may take a moment)...")
            session = new_session('u2net')
            logger.info("[model] u2net AI model loaded successfully")
            self.finished.emit(session)
        except Exception as e:
            logger.error(f"[model] failed to load model: {e}")
            self.error.emit(str(e))


class ImageProcessThread(QThread):
    """Performs AI background removal in a background thread.
    
    This is the ONLY place where rembg.remove() is called. There is no
    polling, no time.sleep(), no simulated processing — the thread
    genuinely waits for the AI model to process the image and emits
    the result via signal when done.
    """
    finished = Signal(object)  # emits processed PIL Image
    error = Signal(str)
    progress = Signal(int)     # emits progress percentage (approximate stages)

    def __init__(self, session, image_data, mode='transparent',
                 refine_strength=0.5, green_color=(0, 177, 64),
                 auto_enhance_enabled=True):
        super().__init__()
        self.session = session
        self.image_data = image_data
        self.mode = mode
        self.refine_strength = refine_strength
        self.green_color = green_color
        self.auto_enhance_enabled = auto_enhance_enabled

    def run(self):
        try:
            # Stage 1: decode image (5%)
            self.progress.emit(5)
            img = Image.open(BytesIO(self.image_data))
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            # Stage 2: preprocessing (15%)
            self.progress.emit(15)
            if self.auto_enhance_enabled:
                img = auto_enhance(img)
                logger.debug("[process] preprocessing (auto-enhance) done")

            # Stage 3: AI background removal (20% → 70%)
            # This is the real rembg.remove() call — blocks until the
            # U2Net model finishes processing the image.
            self.progress.emit(20)
            logger.info("[process] executing rembg.remove() AI inference...")
            output = remove(img, session=self.session)
            logger.info("[process] rembg.remove() AI inference completed")
            self.progress.emit(70)

            # Stage 4: postprocessing (70% → 95%)
            self.progress.emit(75)
            result = postprocess_pipeline(
                output,
                mode=self.mode,
                refine_strength=self.refine_strength,
                green_color=self.green_color
            )
            self.progress.emit(95)

            # Stage 5: done
            logger.info("[process] image processing pipeline complete")
            self.progress.emit(100)
            self.finished.emit(result)

        except Exception as e:
            logger.exception("[process] image processing failed")
            self.error.emit(str(e))


# ─── DropArea ────────────────────────────────────────────

class DropArea(QFrame):
    imageDropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet(_DROP_AREA_DEFAULT_STYLE)

        layout = QVBoxLayout(self)
        self.label = QLabel("拖入图片到此处\n或点击选择图片\n\n支持格式: PNG / JPG / BMP / WebP / GIF / TIFF")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                color: #888;
                border: none;
                background: transparent;
                line-height: 1.5;
            }
        """)
        layout.addWidget(self.label)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(_DROP_AREA_DRAG_OVER_STYLE)

    def dragLeaveEvent(self, event):
        self.resetStyle()

    def dropEvent(self, event: QDropEvent):
        self.resetStyle()
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            fp = files[0]
            if os.path.splitext(fp)[1].lower() in SUPPORTED_IMAGE_EXTS:
                self.imageDropped.emit(fp)
                event.acceptProposedAction()
            else:
                QMessageBox.warning(
                    self, "不支持的格式",
                    f"不支持的文件格式: {os.path.splitext(fp)[1]}"
                )

    def mousePressEvent(self, event):
        fp, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp *.gif *.tiff *.tif *.ico);;所有文件 (*.*)"
        )
        if fp:
            self.imageDropped.emit(fp)

    def resetStyle(self):
        self.setStyleSheet(_DROP_AREA_DEFAULT_STYLE)


# ─── Main Window ─────────────────────────────────────────

class MainWindow(QMainWindow):
    # ── Processing stage labels (visible to user) ──
    _STAGE_LABELS = {
        0:   "等待中...",
        5:   "[1/5] 解码图片...",
        15:  "[2/5] 预处理...",
        20:  "[3/5] AI 推理 (rembg) 进行中...",
        70:  "[4/5] 边缘后处理...",
        95:  "[5/5] 收尾中...",
        100: "完成",
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("图片抠图工具")
        self.setMinimumSize(1100, 720)

        # Core state
        self.session = None
        self.original_image = None
        self.original_path = None
        self.processed_pil_image = None
        self.current_mode = None

        # Threads
        self.model_thread = None
        self.process_thread = None

        # Config
        self.config = AppConfig()

        # Build UI
        self._build_ui()

        # Restore window state
        self._restore_window_state()

        # ⚠ NO automatic model loading — user must click "load model" first

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # ── Title ──
        title = QLabel("图片抠图工具")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(_TITLE_STYLE)
        main_layout.addWidget(title)

        # ── Model loading section (shown before model is loaded) ──
        self.model_load_widget = QWidget()
        ml_layout = QVBoxLayout(self.model_load_widget)
        ml_layout.setSpacing(16)
        ml_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        ml_status = QLabel("AI 模型未加载，请点击下方按钮加载")
        ml_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ml_status.setStyleSheet("font-size: 15px; color: #e67e22; font-weight: bold;")
        ml_layout.addWidget(ml_status)
        self._ml_status = ml_status

        self.btn_load_model = QPushButton("加载 AI 模型 (u2net) [约 200MB]")
        self.btn_load_model.setStyleSheet(_BUTTON_LOAD_STYLE)
        self.btn_load_model.setFixedWidth(380)
        self.btn_load_model.setMinimumHeight(60)
        self.btn_load_model.clicked.connect(self.onUserRequestModelLoad)
        ml_layout.addWidget(self.btn_load_model, 0, Qt.AlignmentFlag.AlignCenter)

        ml_note = QLabel("首次下载需要联网，下载后会缓存到本地")
        ml_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ml_note.setStyleSheet("font-size: 12px; color: #999;")
        ml_layout.addWidget(ml_note)

        self.model_progress = QProgressBar()
        self.model_progress.setRange(0, 0)  # indeterminate
        self.model_progress.setFixedWidth(380)
        self.model_progress.hide()
        ml_layout.addWidget(self.model_progress, 0, Qt.AlignmentFlag.AlignCenter)

        main_layout.addWidget(self.model_load_widget)

        # ── Drop area (shown after model is loaded) ──
        self.drop_area = DropArea()
        self.drop_area.imageDropped.connect(self.loadImage)
        self.drop_area.hide()
        main_layout.addWidget(self.drop_area)

        # ── Image comparison area ──
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)

        # Original panel
        orig_container = QWidget()
        orig_layout = QVBoxLayout(orig_container)
        orig_layout.setContentsMargins(0, 0, 0, 0)
        orig_layout.setSpacing(4)
        orig_title = QLabel("原图")
        orig_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        orig_title.setStyleSheet(_SUB_TITLE_STYLE)
        orig_layout.addWidget(orig_title)

        if HAS_ENHANCED_VIEWER:
            self.original_viewer = ZoomableImageViewer("原图")
            self.original_viewer_ctrl = ViewerControls()
            self.original_viewer.zoomChanged.connect(self.original_viewer_ctrl.updateZoomLabel)
            self.original_viewer_ctrl.zoomInClicked.connect(self.original_viewer.zoomIn)
            self.original_viewer_ctrl.zoomOutClicked.connect(self.original_viewer.zoomOut)
            self.original_viewer_ctrl.fitClicked.connect(self.original_viewer.fitInView)
            self.original_viewer_ctrl.actualClicked.connect(self.original_viewer.zoomToActual)
            orig_layout.addWidget(self.original_viewer)
            orig_layout.addWidget(self.original_viewer_ctrl)
        else:
            self.original_viewer = QLabel("原图")
            self.original_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.original_viewer.setMinimumSize(350, 350)
            orig_layout.addWidget(self.original_viewer)

        # Processed panel
        proc_container = QWidget()
        proc_layout = QVBoxLayout(proc_container)
        proc_layout.setContentsMargins(0, 0, 0, 0)
        proc_layout.setSpacing(4)
        proc_title = QLabel("处理后")
        proc_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        proc_title.setStyleSheet(_SUB_TITLE_STYLE)
        proc_layout.addWidget(proc_title)

        if HAS_ENHANCED_VIEWER:
            self.processed_viewer = ZoomableImageViewer("处理后")
            self.processed_viewer_ctrl = ViewerControls()
            self.processed_viewer.zoomChanged.connect(self.processed_viewer_ctrl.updateZoomLabel)
            self.processed_viewer_ctrl.zoomInClicked.connect(self.processed_viewer.zoomIn)
            self.processed_viewer_ctrl.zoomOutClicked.connect(self.processed_viewer.zoomOut)
            self.processed_viewer_ctrl.fitClicked.connect(self.processed_viewer.fitInView)
            self.processed_viewer_ctrl.actualClicked.connect(self.processed_viewer.zoomToActual)
            proc_layout.addWidget(self.processed_viewer)
            proc_layout.addWidget(self.processed_viewer_ctrl)
        else:
            self.processed_viewer = QLabel("处理后")
            self.processed_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.processed_viewer.setMinimumSize(350, 350)
            proc_layout.addWidget(self.processed_viewer)

        splitter.addWidget(orig_container)
        splitter.addWidget(proc_container)
        splitter.setSizes([480, 480])

        self._splitter = splitter
        self.image_container = splitter
        self.image_container.hide()
        main_layout.addWidget(self.image_container)

        # ── Status bar ──
        self.status_label = QLabel("请先加载 AI 模型")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(_STATUS_INFO_STYLE)
        main_layout.addWidget(self.status_label)

        # ── Processing progress bar ──
        self.process_progress = QProgressBar()
        self.process_progress.setRange(0, 100)
        self.process_progress.setValue(0)
        self.process_progress.setStyleSheet(_PROGRESS_STYLE)
        self.process_progress.hide()
        main_layout.addWidget(self.process_progress)

        # ── Options bar ──
        options_layout = QHBoxLayout()
        options_layout.setSpacing(16)

        options_layout.addWidget(QLabel("边缘精修:"))
        self.slider_refine = QSlider(Qt.Orientation.Horizontal)
        self.slider_refine.setRange(0, 100)
        self.slider_refine.setValue(int(self.config.get_refine_strength() * 100))
        self.slider_refine.setFixedWidth(120)
        self.slider_refine.setToolTip("控制 AI 抠图后边缘处理的力度")
        options_layout.addWidget(self.slider_refine)
        self.label_refine = QLabel(f"{self.slider_refine.value()}%")
        self.label_refine.setFixedWidth(35)
        self.label_refine.setStyleSheet("font-size: 12px; color: #666;")
        options_layout.addWidget(self.label_refine)
        self.slider_refine.valueChanged.connect(
            lambda v: self.label_refine.setText(f"{v}%")
        )

        self.cb_auto = QCheckBox("预处理增强")
        self.cb_auto.setChecked(self.config.get_auto_enhance())
        self.cb_auto.setToolTip("抠图前进行对比度/锐化增强，有助于提高 AI 识别精度")
        options_layout.addWidget(self.cb_auto)

        self.btn_color = QPushButton("绿幕颜色")
        self.btn_color.setToolTip("点击自定义绿幕背景的 RGB 颜色")
        self.btn_color.setFixedHeight(30)
        self.btn_color.setStyleSheet("""
            QPushButton {
                font-size: 12px; padding: 4px 12px;
                border: 1px solid #ccc; border-radius: 4px;
                background: #fafafa; color: #333;
            }
            QPushButton:hover { background: #e8e8e8; border-color: #4a90d9; }
        """)
        self._green_color = self.config.get_green_color()
        self.btn_color.clicked.connect(self._pick_green_color)
        options_layout.addWidget(self.btn_color)

        self._color_preview = QFrame()
        self._color_preview.setFixedSize(24, 24)
        self._color_preview.setStyleSheet(
            f"background: rgb{self._green_color}; border: 1px solid #999; border-radius: 3px;"
        )
        options_layout.addWidget(self._color_preview)

        options_layout.addStretch()
        main_layout.addLayout(options_layout)

        # ── Action buttons ──
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        self.btn_cutout_transparent = QPushButton("透明背景抠图")
        self.btn_cutout_green = QPushButton("添加绿幕背景")
        self.btn_save = QPushButton("保存图片")
        self.btn_reset = QPushButton("重新选择")

        # [manual] user clicks → triggers processing
        self.btn_cutout_transparent.clicked.connect(lambda: self.processImage('transparent'))
        self.btn_cutout_green.clicked.connect(lambda: self.processImage('green'))
        self.btn_save.clicked.connect(self.saveImage)
        self.btn_reset.clicked.connect(self.resetApp)

        for btn in [self.btn_cutout_transparent, self.btn_cutout_green,
                    self.btn_save, self.btn_reset]:
            btn.setMinimumHeight(46)
            btn.setStyleSheet(_BUTTON_RESET_STYLE if btn == self.btn_reset else _BUTTON_PRIMARY_STYLE)
            btn.setEnabled(False)
            button_layout.addWidget(btn)

        main_layout.addLayout(button_layout)

    # ── Color picker ──
    def _pick_green_color(self):
        initial = QColor(*self._green_color)
        color = QColorDialog.getColor(initial, self, "选择绿幕颜色")
        if color.isValid():
            self._green_color = (color.red(), color.green(), color.blue())
            self._color_preview.setStyleSheet(
                f"background: rgb{self._green_color}; border: 1px solid #999; border-radius: 3px;"
            )
            self.config.set_green_color(*self._green_color)

    # ── Window state ──
    def _restore_window_state(self):
        geo = self.config.get_window_geometry()
        if geo:
            self.restoreGeometry(geo)
        state = self.config.get_window_state()
        if state:
            self.restoreState(state)

    def _save_window_state(self):
        self.config.set_window_geometry(self.saveGeometry())
        self.config.set_window_state(self.saveState())
        if hasattr(self, '_splitter'):
            st = self._splitter.saveState()
            if st:
                self.config.set_splitter_state(st)
        self.config.sync()

    def closeEvent(self, event):
        self._save_window_state()
        # Clean up threads
        for t in [self.model_thread, self.process_thread]:
            if t and t.isRunning():
                t.quit()
                t.wait(3000)
        logger.info("[app] exiting")
        event.accept()

    # ── [MANUAL] Model loading ──
    def onUserRequestModelLoad(self):
        """Triggered ONLY by user clicking the 'Load Model' button."""
        self.btn_load_model.setEnabled(False)
        self.btn_load_model.setText("加载中...")
        self._ml_status.setText("正在加载 AI 模型，请稍候...")
        self._ml_status.setStyleSheet(_STATUS_LOADING_STYLE)
        self.model_progress.show()
        QApplication.processEvents()

        self.model_thread = ModelLoaderThread()
        self.model_thread.finished.connect(self._onModelLoadSuccess)
        self.model_thread.error.connect(self._onModelLoadError)
        self.model_thread.finished.connect(self.model_thread.deleteLater)
        self.model_thread.error.connect(self.model_thread.deleteLater)
        logger.info("[app] user initiated model loading")
        self.model_thread.start()

    def _onModelLoadSuccess(self, session):
        self.session = session
        logger.info("[app] model loaded, transitioning UI")
        # Transition from model-load screen to drop area
        self.model_load_widget.hide()
        self.drop_area.show()
        self.status_label.setText("模型加载完成！拖入图片或点击选择图片开始使用")
        self.status_label.setStyleSheet(_STATUS_SUCCESS_STYLE)

    def _onModelLoadError(self, error_msg):
        self.btn_load_model.setEnabled(True)
        self.btn_load_model.setText("加载 AI 模型 (u2net)")
        self.model_progress.hide()
        self._ml_status.setText(f"模型加载失败: {error_msg}")
        self._ml_status.setStyleSheet(_STATUS_ERROR_STYLE)
        QMessageBox.critical(
            self, "模型错误",
            f"AI 模型加载失败:\n{error_msg}\n\n"
            "请检查网络连接后重试。"
        )
        logger.error(f"[app] model load failed: {error_msg}")

    # ── [MANUAL] Image loading ──
    def loadImage(self, file_path):
        if not self.session:
            QMessageBox.warning(self, "提示", "AI 模型尚未加载")
            return

        valid, err_msg = validate_image_file(file_path)
        if not valid:
            QMessageBox.warning(self, "文件无效", err_msg)
            return

        try:
            img, err = preprocess_pipeline(file_path)
            if img is None:
                QMessageBox.critical(self, "加载失败", err)
                return

            self.original_path = file_path
            self.original_image = img

            self.drop_area.hide()
            self.image_container.show()

            if HAS_ENHANCED_VIEWER:
                self.original_viewer.setImage(img)
                self.processed_viewer.clear()
                self.original_viewer.fitInView()
            else:
                self._set_label_pixmap(self.original_viewer, img)
                self.processed_viewer.clear()
                self.processed_viewer.setText("处理后")

            sz_mb = os.path.getsize(file_path) / (1024 * 1024)
            self.status_label.setText(
                f"已加载: {os.path.basename(file_path)} ({img.width}x{img.height}, {sz_mb:.1f}MB) — 请选择抠图方式"
            )
            self.status_label.setStyleSheet(_STATUS_INFO_STYLE)

            self.btn_cutout_transparent.setEnabled(True)
            self.btn_cutout_green.setEnabled(True)
            self.btn_save.setEnabled(False)
            self.btn_reset.setEnabled(True)
            self.processed_pil_image = None
            self.current_mode = None

            self.config.set_last_directory(os.path.dirname(file_path))
            logger.info(f"[app] image loaded: {os.path.basename(file_path)} ({img.width}x{img.height})")

        except Exception as e:
            logger.exception("[app] image load exception")
            QMessageBox.critical(self, "错误", f"图片加载失败: {str(e)}")

    # ── [MANUAL] Image processing ──
    def processImage(self, mode):
        """Called ONLY by user clicking 'remove background' or 'green screen' button."""
        if not self.original_image or not self.session:
            return

        # Disable all action buttons during processing
        self._set_processing_buttons_enabled(False)
        self.status_label.setText("正在处理图片，大图可能需要较长时间...")
        self.status_label.setStyleSheet(_STATUS_LOADING_STYLE)
        self.process_progress.setValue(0)
        self.process_progress.show()
        QApplication.processEvents()

        self.current_mode = mode
        self.config.set_last_mode(mode)

        refine_strength = self.slider_refine.value() / 100.0
        auto_enhance_enabled = self.cb_auto.isChecked()
        green_color = self._green_color

        buf = BytesIO()
        self.original_image.save(buf, format='PNG')
        image_data = buf.getvalue()

        # Kill any stale thread
        if self.process_thread and self.process_thread.isRunning():
            self.process_thread.quit()
            self.process_thread.wait(2000)

        self.process_thread = ImageProcessThread(
            self.session, image_data, mode,
            refine_strength=refine_strength,
            green_color=green_color,
            auto_enhance_enabled=auto_enhance_enabled
        )
        self.process_thread.finished.connect(self._onProcessDone)
        self.process_thread.error.connect(self._onProcessError)
        self.process_thread.progress.connect(self._onProcessProgress)
        self.process_thread.finished.connect(self.process_thread.deleteLater)
        self.process_thread.error.connect(self.process_thread.deleteLater)
        self.process_thread.start()

        mode_name = "透明背景" if mode == "transparent" else "绿幕背景"
        logger.info(f"[app] user initiated {mode_name} processing (refine={refine_strength}, enhance={auto_enhance_enabled})")

    def _onProcessProgress(self, pct: int):
        """Real-time progress updates from the worker thread."""
        self.process_progress.setValue(pct)
        label = self._STAGE_LABELS.get(pct, f"processing ({pct}%)")
        self.status_label.setText(f"处理中: {label}")
        self.status_label.setStyleSheet(_STATUS_LOADING_STYLE)

    def _onProcessDone(self, result):
        self.processed_pil_image = result
        self.process_progress.hide()

        show_checkerboard = (self.current_mode == 'transparent')

        if HAS_ENHANCED_VIEWER:
            self.processed_viewer.setImage(result, show_checkerboard=show_checkerboard)
            self.processed_viewer.fitInView()
        else:
            self._set_label_pixmap(self.processed_viewer, result, show_checkerboard)

        mode_text = "透明背景" if self.current_mode == "transparent" else "绿幕背景"
        self.status_label.setText(f"处理完成 — {mode_text} 结果已就绪，可保存或调整设置")
        self.status_label.setStyleSheet(_STATUS_SUCCESS_STYLE)

        self.btn_cutout_transparent.setEnabled(True)
        self.btn_cutout_green.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.btn_reset.setEnabled(True)

    def _onProcessError(self, error_msg):
        self.process_progress.hide()
        QMessageBox.critical(self, "处理错误", f"图片处理失败:\n{error_msg}")
        self.status_label.setText("处理失败 — 请调整参数后重试")
        self.status_label.setStyleSheet(_STATUS_ERROR_STYLE)
        self._set_processing_buttons_enabled(True)

    def _set_processing_buttons_enabled(self, enabled: bool):
        for btn in [self.btn_cutout_transparent, self.btn_cutout_green,
                     self.btn_save, self.btn_reset]:
            btn.setEnabled(enabled)

    # ── [MANUAL] Save ──
    def saveImage(self):
        """Called ONLY by user clicking 'save' button (or Ctrl+S)."""
        if not self.processed_pil_image:
            return

        base_name = "untitled"
        if self.original_path:
            base_name = os.path.splitext(os.path.basename(self.original_path))[0]

        last_dir = self.config.get_last_directory()
        if self.current_mode == 'transparent':
            default_name = os.path.join(last_dir, f"{base_name}_transparent.png")
            filter_str = "PNG 图片 (*.png)"
        else:
            default_name = os.path.join(last_dir, f"{base_name}_greenscreen.png")
            filter_str = "PNG 图片 (*.png);;JPEG 图片 (*.jpg *.jpeg)"

        file_path, sel_filter = QFileDialog.getSaveFileName(
            self, "保存图片", default_name, filter_str
        )
        if not file_path:
            return

        try:
            if self.current_mode == 'transparent':
                if not file_path.lower().endswith('.png'):
                    file_path += '.png'
                self.processed_pil_image.save(file_path, 'PNG')
            else:
                is_jpeg = (file_path.lower().endswith('.jpg') or
                           file_path.lower().endswith('.jpeg') or
                           'JPEG' in sel_filter)
                if is_jpeg:
                    if not (file_path.lower().endswith('.jpg') or
                            file_path.lower().endswith('.jpeg')):
                        file_path += '.jpg'
                    self.processed_pil_image.convert('RGB').save(
                        file_path, 'JPEG', quality=95
                    )
                else:
                    if not file_path.lower().endswith('.png'):
                        file_path += '.png'
                    self.processed_pil_image.save(file_path, 'PNG')

            self.config.set_last_directory(os.path.dirname(file_path))
            self.status_label.setText(f"已保存: {os.path.basename(file_path)}")
            self.status_label.setStyleSheet(_STATUS_SUCCESS_STYLE)
            QMessageBox.information(self, "成功", f"图片已保存到:\n{file_path}")
            logger.info(f"[app] saved: {file_path}")

        except Exception as e:
            logger.exception("[app] save failed")
            QMessageBox.critical(self, "保存错误", f"保存失败: {str(e)}")

    # ── [MANUAL] Reset ──
    def resetApp(self):
        """Called ONLY by user clicking 'reset' button (or Esc)."""
        self.original_image = None
        self.original_path = None
        self.processed_pil_image = None
        self.current_mode = None

        if HAS_ENHANCED_VIEWER:
            self.original_viewer.clear()
            self.processed_viewer.clear()
        else:
            self.original_viewer.clear()
            self.original_viewer.setText("原图")
            self.processed_viewer.clear()
            self.processed_viewer.setText("处理后")

        self.image_container.hide()
        self.drop_area.show()
        self.drop_area.resetStyle()
        self.process_progress.hide()

        if self.session:
            self.status_label.setText("模型加载完成！拖入图片或点击选择图片开始使用")
            self.status_label.setStyleSheet(_STATUS_SUCCESS_STYLE)
        else:
            self.status_label.setText("请先加载 AI 模型")
            self.status_label.setStyleSheet(_STATUS_INFO_STYLE)

        for btn in [self.btn_cutout_transparent, self.btn_cutout_green,
                    self.btn_save, self.btn_reset]:
            btn.setEnabled(False)
        logger.info("[app] reset complete")

    # ── Legacy QLabel viewer helper ──
    def _set_label_pixmap(self, label, pil_image, show_checkerboard=False):
        img = pil_image.copy()
        if show_checkerboard and pil_image.mode == 'RGBA':
            board = Image.new('RGBA', (pil_image.width, pil_image.height), (255, 255, 255, 255))
            for y in range(0, pil_image.height, 20):
                for x in range(0, pil_image.width, 20):
                    if ((x // 20) + (y // 20)) % 2 == 0:
                        b = (x, y, min(x + 20, pil_image.width), min(y + 20, pil_image.height))
                        tile = Image.new('RGBA', (b[2] - b[0], b[3] - b[1]),
                                         (200, 200, 200, 255))
                        board.paste(tile, b)
            board.paste(pil_image, mask=pil_image.split()[3])
            img = board

        if img.mode == 'RGBA':
            data = img.tobytes('raw', 'RGBA')
            qimg = QImage(data, img.width, img.height, QImage.Format.Format_RGBA8888)
        else:
            data = img.tobytes('raw', 'RGB')
            qimg = QImage(data, img.width, img.height, QImage.Format.Format_RGB888)

        pm = QPixmap.fromImage(qimg)
        scaled = pm.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(scaled)

    # ── Keyboard shortcuts ──
    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key.Key_Escape and self.original_image:
            self.resetApp()
        elif k == Qt.Key.Key_S and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.saveImage()
        super().keyPressEvent(event)


# ─── Entry Point ─────────────────────────────────────────

def main():
    logger.info("=" * 50)
    logger.info("图片抠图工具 starting")
    logger.info("=" * 50)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
