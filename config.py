"""
配置持久化管理模块
基于 QSettings 保存/加载用户偏好设置。
"""

import logging
from typing import Optional

from PySide6.QtCore import QSettings, QByteArray

logger = logging.getLogger(__name__)

# ─── Key 常量 ────────────────────────────────────────────

_COMPANY = "KuaiLvTools"
_APP = "ImageMattingTool"

KEY_LAST_DIR = "save/last_directory"
KEY_WINDOW_GEOMETRY = "window/geometry"
KEY_WINDOW_STATE = "window/state"
KEY_SPLITTER_STATE = "window/splitter_state"
KEY_LAST_MODE = "preferences/last_mode"
KEY_GREEN_R = "preferences/green_r"
KEY_GREEN_G = "preferences/green_g"
KEY_GREEN_B = "preferences/green_b"
KEY_REFINE_STRENGTH = "preferences/refine_strength"
KEY_AUTO_ENHANCE = "preferences/auto_enhance"


class AppConfig:
    """应用配置管理器，封装 QSettings 操作"""

    def __init__(self):
        self._settings = QSettings(_COMPANY, _APP)

    # ── 文件路径 ──

    def get_last_directory(self) -> str:
        return self._settings.value(KEY_LAST_DIR, "", type=str)

    def set_last_directory(self, path: str):
        self._settings.setValue(KEY_LAST_DIR, path)

    # ── 窗口状态 ──

    def get_window_geometry(self) -> Optional[QByteArray]:
        val = self._settings.value(KEY_WINDOW_GEOMETRY)
        return val if isinstance(val, QByteArray) else None

    def set_window_geometry(self, geometry: QByteArray):
        self._settings.setValue(KEY_WINDOW_GEOMETRY, geometry)

    def get_window_state(self) -> Optional[QByteArray]:
        val = self._settings.value(KEY_WINDOW_STATE)
        return val if isinstance(val, QByteArray) else None

    def set_window_state(self, state: QByteArray):
        self._settings.setValue(KEY_WINDOW_STATE, state)

    def get_splitter_state(self) -> Optional[QByteArray]:
        val = self._settings.value(KEY_SPLITTER_STATE)
        return val if isinstance(val, QByteArray) else None

    def set_splitter_state(self, state: QByteArray):
        self._settings.setValue(KEY_SPLITTER_STATE, state)

    # ── 偏好设置 ──

    def get_last_mode(self) -> str:
        return self._settings.value(KEY_LAST_MODE, "transparent", type=str)

    def set_last_mode(self, mode: str):
        self._settings.setValue(KEY_LAST_MODE, mode)

    def get_green_color(self) -> tuple[int, int, int]:
        r = self._settings.value(KEY_GREEN_R, 0, type=int)
        g = self._settings.value(KEY_GREEN_G, 177, type=int)
        b = self._settings.value(KEY_GREEN_B, 64, type=int)
        return (r, g, b)

    def set_green_color(self, r: int, g: int, b: int):
        self._settings.setValue(KEY_GREEN_R, r)
        self._settings.setValue(KEY_GREEN_G, g)
        self._settings.setValue(KEY_GREEN_B, b)

    def get_refine_strength(self) -> float:
        return self._settings.value(KEY_REFINE_STRENGTH, 0.5, type=float)

    def set_refine_strength(self, strength: float):
        strength = max(0.0, min(1.0, strength))
        self._settings.setValue(KEY_REFINE_STRENGTH, strength)

    def get_auto_enhance(self) -> bool:
        return self._settings.value(KEY_AUTO_ENHANCE, True, type=bool)

    def set_auto_enhance(self, enabled: bool):
        self._settings.setValue(KEY_AUTO_ENHANCE, enabled)

    def sync(self):
        """立即写入磁盘"""
        self._settings.sync()
