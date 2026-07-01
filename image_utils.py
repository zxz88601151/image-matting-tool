"""
图像预处理与后处理工具模块
提供: EXIF方向矫正、自适应增强、Alpha边缘精修、尺寸保护、文件格式校验
"""

import os
import logging
from io import BytesIO

from PIL import Image, ImageFilter, ImageEnhance, UnidentifiedImageError

# ─── 导出常量 ────────────────────────────────────────────

MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB 图片大小上限
SUPPORTED_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif', '.tiff', '.tif', '.ico'}

logger = logging.getLogger(__name__)


# ─── 格式验证工具 ─────────────────────────────────────────

def is_supported_image(file_path: str) -> bool:
    """检查文件扩展名是否属于支持的图片格式"""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in SUPPORTED_IMAGE_EXTS


def validate_image_file(file_path: str) -> tuple[bool, str]:
    """验证图片文件是否可用，返回 (是否有效, 错误信息)"""
    if not os.path.exists(file_path):
        return False, "文件不存在"
    if not is_supported_image(file_path):
        return False, f"不支持的文件格式: {os.path.splitext(file_path)[1]}"
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        return False, "文件为空"
    if file_size > MAX_IMAGE_SIZE:
        return False, f"文件过大（{file_size / 1024 / 1024:.1f}MB），上限为 {MAX_IMAGE_SIZE / 1024 / 1024:.0f}MB"
    return True, ""


# ─── EXIF 方向矫正 ───────────────────────────────────────

try:
    import piexif

    _HAS_PIEXIF = True
except ImportError:
    _HAS_PIEXIF = False
    logger.info("piexif 未安装，EXIF 方向矫正功能不可用")


def correct_orientation(pil_image: Image.Image) -> Image.Image:
    """根据 EXIF 元数据自动旋转/翻转图片

    手机/相机拍摄的照片通常带有方向标记（Orientation tag），
    但很多图像查看/处理库不会自动应用该标记。本函数读取 EXIF
    Orientation 字段并执行对应的像素级变换。
    """
    exif_data = pil_image.info.get('exif', b'')
    if not exif_data:
        return pil_image

    try:
        import piexif
        exif_dict = piexif.load(exif_data)
        orientation = exif_dict.get('0th', {}).get(piexif.ImageIFD.Orientation, 1)
    except Exception:
        # 如果 piexif 解析失败，回退到 PIL 的基础支持
        try:
            from PIL import ImageOps
            return ImageOps.exif_transpose(pil_image) or pil_image
        except Exception:
            return pil_image

    if orientation == 1:
        return pil_image

    ops = {
        2: (Image.Transpose.FLIP_LEFT_RIGHT,),
        3: (Image.Transpose.ROTATE_180,),
        4: (Image.Transpose.FLIP_TOP_BOTTOM,),
        5: (Image.Transpose.TRANSPOSE,),
        6: (Image.Transpose.ROTATE_270,),
        7: (Image.Transpose.TRANSVERSE,),
        8: (Image.Transpose.ROTATE_90,),
    }

    if orientation in ops:
        return pil_image.transpose(ops[orientation][0])

    return pil_image


# ─── 自适应图像增强 ───────────────────────────────────────

def auto_enhance(pil_image: Image.Image, strength: float = 0.3) -> Image.Image:
    """轻度自适应增强: 对比度 + 锐化

    在抠图前进行轻度增强，有助于提升 AI 模型对边缘的识别准确率。
    strength 控制增强幅度 (0.0~1.0)，默认 0.3 为轻度增强。
    """
    img = pil_image.convert('RGBA')

    # 分离 alpha 通道
    r, g, b, a = img.split()
    rgb = Image.merge('RGB', (r, g, b))

    # 轻度对比度增强
    enhancer = ImageEnhance.Contrast(rgb)
    rgb = enhancer.enhance(1.0 + strength * 0.3)

    # 轻度锐化
    rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1, percent=int(50 * strength), threshold=2))

    # 合并回 alpha
    rgb_r, rgb_g, rgb_b = rgb.split()
    return Image.merge('RGBA', (rgb_r, rgb_g, rgb_b, a))


# ─── Alpha 边缘精修 ───────────────────────────────────────

def refine_alpha(pil_image: Image.Image, strength: float = 0.5) -> Image.Image:
    """使用形态学操作精修 Alpha 通道边缘

    用于后处理，清理 AI 抠图产生的边缘噪声、孤立像素和锯齿。
    strength: 0.0~1.0，控制精修力度
    """
    img = pil_image.convert('RGBA')
    alpha = img.split()[3]

    try:
        import cv2
        import numpy as np

        alpha_np = np.array(alpha, dtype=np.uint8)

        kernel_size = max(1, int(3 * strength))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # 轻度开运算: 去除前景内部的孤立噪点
        alpha_np = cv2.morphologyEx(alpha_np, cv2.MORPH_OPEN, kernel)

        # 轻度闭运算: 填补前景内部的小空洞
        alpha_np = cv2.morphologyEx(alpha_np, cv2.MORPH_CLOSE, kernel)

        # 边缘羽化: 高斯模糊平滑二值边缘
        if strength > 0.3:
            blur_radius = max(1, int(strength * 2))
            alpha_np = cv2.GaussianBlur(alpha_np, (blur_radius | 1, blur_radius | 1), 0)

        refined_alpha = Image.fromarray(alpha_np, mode='L')
        r, g, b, _ = img.split()
        return Image.merge('RGBA', (r, g, b, refined_alpha))

    except ImportError:
        # opencv 不可用时的回退方案: 使用 PIL 的滤镜
        logger.warning("OpenCV 不可用，使用 PIL 回退方案进行 alpha 精修")
        if strength > 0.5:
            alpha = alpha.filter(ImageFilter.SMOOTH_MORE)
        return Image.merge('RGBA', img.split()[:3] + (alpha,))


# ─── 图像尺寸保护 ───────────────────────────────────────

MAX_PIXELS = 50_000_000  # 5000万像素上限（约 7071x7071）


def guard_resize(pil_image: Image.Image, max_pixels: int = MAX_PIXELS) -> Image.Image:
    """如果图片像素数超过上限，等比缩小至上限以内

    rembg 处理超大图片会消耗大量内存和显存，且耗时很长。
    在送入模型前缩小，处理成本可控。
    """
    width, height = pil_image.size
    current_pixels = width * height

    if current_pixels <= max_pixels:
        return pil_image

    ratio = (max_pixels / current_pixels) ** 0.5
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    logger.info(f"图片过大 ({width}x{height}), 等比缩放到 {new_width}x{new_height}")
    return pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)


# ─── 统一预处理管线 ───────────────────────────────────────

def preprocess_pipeline(file_path: str) -> tuple[Image.Image | None, str]:
    """完整的图片预处理管线

    接收文件路径，返回 (PIL Image, 错误信息)。
    处理流程: 文件验证 -> 格式解析 -> EXIF矫正 -> 尺寸保护
    """
    if not os.path.exists(file_path):
        return None, "文件不存在"

    try:
        img = Image.open(file_path)

        # 验证图片完整性
        img.verify()

        # verify() 后必须重新打开
        img = Image.open(file_path)

        # EXIF 方向矫正
        img = correct_orientation(img)

        # 确保 RGBA 模式
        if img.mode not in ('RGBA', 'RGB', 'LA', 'L'):
            img = img.convert('RGBA')
        elif img.mode != 'RGBA':
            img = img.convert('RGBA')

        # 尺寸保护（缩放到合理范围）
        img = guard_resize(img)

        return img, ""

    except UnidentifiedImageError:
        return None, "无法识别的图片格式"
    except Exception as e:
        logger.exception("预处理失败")
        return None, f"预处理失败: {str(e)}"


def postprocess_pipeline(
    pil_image: Image.Image,
    mode: str = 'transparent',
    refine_strength: float = 0.5,
    green_color: tuple = (0, 177, 64)
) -> Image.Image:
    """完整的图片后处理管线

    接收 AI 处理后的 RGBA 图像，执行边缘精修并转换到目标格式。
    """
    img = pil_image.convert('RGBA')

    # Alpha 边缘精修
    img = refine_alpha(img, strength=refine_strength)

    if mode == 'green':
        background = Image.new('RGBA', img.size, green_color + (255,))
        background.paste(img, mask=img.split()[3])
        return background.convert('RGB')

    return img
