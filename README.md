# Image Matting Tool / 图片抠图工具

A desktop application for AI-powered background removal and green-screen compositing, built with PySide6 and rembg.

基于 PySide6 和 rembg 的桌面端 AI 抠图工具，支持透明背景和绿幕合成两种模式。

---

## Features / 功能特性

- **AI Background Removal** — Remove image backgrounds automatically using the u2net model
- **Green Screen Compositing** — Replace background with customizable green-screen color
- **Real-time Progress** — 5-stage progress bar with detailed status feedback
- **Zoom & Pan** — Scroll-wheel zoom, drag-to-pan image viewer (powered by QGraphicsView)
- **Edge Refinement** — Morphological alpha matte refinement via OpenCV
- **EXIF Auto-Correction** — Automatic orientation correction for photos
- **Preprocessing Enhancement** — Optional contrast/sharpness boost before AI processing
- **Config Persistence** — Window state, preferences, and color settings remembered across sessions
- **Batch-Friendly** — Drag & drop support, keyboard shortcuts (Esc reset, Ctrl+S save)
- **Full Manual Control** — All processing is user-triggered, no automatic/hidden operations

---

## Screenshot / 界面截图

```
┌──────────────────────────────────────────────────────────────┐
│                     Image Matting Tool                        │
├──────────────────────────┬───────────────────────────────────┤
│       Original           │           Processed                │
│    ┌──────────────┐      │       ┌──────────────────┐        │
│    │              │      │       │   (transparent)   │        │
│    │   (image)    │      │       │  ┌────────────┐   │        │
│    │              │      │       │  │  subject   │   │        │
│    └──────────────┘      │       │  │  remains   │   │        │
│                          │       │  │  bg removed│   │        │
│                          │       │  └────────────┘   │        │
│                          │       └──────────────────┘        │
├──────────────────────────┴───────────────────────────────────┤
│  [edge refine: █████░░░░░ 50%] [✓ pre-enhance] [green color] │
├──────────────────────────────────────────────────────────────┤
│  [remove background] [add green screen]  [save]  [reset]     │
└──────────────────────────────────────────────────────────────┘
```

---

## Tech Stack / 技术栈

| Category | Library | Version | Purpose |
|----------|---------|---------|---------|
| **GUI Framework** | [PySide6](https://pypi.org/project/PySide6/) | ≥6.5.0 | Qt6 bindings for Python desktop GUI |
| **AI Model** | [rembg](https://github.com/danielgatis/rembg) | ≥2.0.50 | Background removal using u2net ONNX model |
| **Image Processing** | [Pillow](https://python-pillow.org/) | ≥10.0.0 | Image I/O, format conversion, enhancement |
| **Numerical** | [NumPy](https://numpy.org/) | ≥1.24.0 | Array operations for image data |
| **AI Runtime** | [ONNX Runtime](https://onnxruntime.ai/) | ≥1.16.0 | ONNX model inference engine |
| **Alpha Refinement** | [OpenCV](https://opencv.org/) (headless) | ≥4.8.0 | Morphological operations for edge cleanup |
| **EXIF Handling** | [piexif](https://github.com/hMatoba/piexif) | ≥1.1.0 | EXIF orientation tag reading and correction |
| **Image Processing** | [scikit-image](https://scikit-image.org/) | (via rembg) | Additional image algorithms |
| **Dependency Mgr** | pip | — | Package management |

---

## Project Modules / 项目模块

```
d:\toumingbeijingkoutu\
├── main.py              — Application entry point, UI construction, event handling
├── image_utils.py       — Image preprocessing pipeline (EXIF, enhance, resize, alpha refine)
├── config.py            — Settings persistence via QSettings
├── enhanced_viewer.py   — Zoomable image viewer based on QGraphicsView
├── requirements.txt     — Python dependency list with version constraints
├── run.bat              — Windows launcher with venv detection and auto-install
├── .gitignore           — Git ignore rules
└── README.md            — This file
```

---

## Quick Start / 快速开始

### Prerequisites / 前置要求

- Python 3.10+
- Windows / macOS / Linux (with display server)

### Installation / 安装

```bash
# Clone the repository
git clone https://github.com/zxz88601151/image-matting-tool.git
cd image-matting-tool

# Install dependencies (use mirror for faster download in China)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# Or on Windows, simply double-click run.bat
```

### Usage / 使用

```bash
python main.py
```

1. **Load Model** — Click "Load AI Model (u2net)" to download and initialize (requires internet for first run, ~200MB model)
2. **Select Image** — Drag & drop or click to browse for an image
3. **Process** — Click "remove background (transparent)" or "add green screen"
4. **Adjust** — Tweak edge refinement slider, toggle pre-enhance, pick green color
5. **Save** — Save as PNG (transparent) or PNG/JPEG (green screen)

### Shortcuts / 快捷键

| Key | Action |
|-----|--------|
| `Esc` | Reset to image selection |
| `Ctrl+S` | Save processed result |

---

## Processing Pipeline / 处理管线

All processing is **manually triggered** by the user. The pipeline runs in a dedicated QThread and reports real-time progress via signals.

```
[User clicks button]
        ↓
[1/5] Decode image → RGBA conversion          (5%)
[2/5] Preprocessing → auto-enhance if enabled  (15%)
[3/5] AI inference → rembg.remove()            (20% → 70%)  ← real u2net model
[4/5] Post-processing → alpha refine + format   (75%)
[5/5] Finalizing                                 (95%)
[Done] Result displayed                          (100%)
```

---

## Dependencies / 依赖清单

This project builds upon the following open-source libraries. We express our gratitude to their authors.

| Dependency | License | Repository |
|-----------|---------|------------|
| PySide6 | LGPL-3.0 | https://github.com/qt/qt6 |
| rembg | MIT | https://github.com/danielgatis/rembg |
| Pillow | Historical | https://github.com/python-pillow/Pillow |
| NumPy | BSD-3 | https://github.com/numpy/numpy |
| ONNX Runtime | MIT | https://github.com/microsoft/onnxruntime |
| OpenCV | Apache-2.0 | https://github.com/opencv/opencv |
| piexif | MIT | https://github.com/hMatoba/piexif |
| scikit-image | BSD-3 | https://github.com/scikit-image/scikit-image |

### AI Model

The background removal is powered by the **u2net** model, automatically downloaded by rembg from HuggingFace on first run:
- Model: [u2net.onnx](https://github.com/danielgatis/rembg#models)
- Source: U-2-Net: Going Deeper with Nested U-Structure for Salient Object Detection (Qin et al., 2020)

---

## Security / 安全说明

- **Fully Offline Processing** — All image data is processed locally; no data is sent to external servers
- **No Network Requests** — Except for the initial model download from HuggingFace
- **No Credentials** — No API keys, tokens, or passwords are stored or transmitted
- **Input Validation** — Files are validated for format, size (max 50MB), and integrity before loading

---

## License / 许可

This project is open-source and available under the MIT License.

---

## Credits / 致谢

- [rembg](https://github.com/danielgatis/rembg) by Daniel Gatis — the core background removal engine
- [Qt for Python](https://www.qt.io/qt-for-python) — the GUI framework
- [OpenCV](https://opencv.org/) — image processing algorithms
- All open-source contributors whose libraries made this project possible
