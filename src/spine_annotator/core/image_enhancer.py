"""图像增强算法层（纯函数，不涉及 UI）。

为 X 光片标注辅助提供以下增强能力（不修改原图，仅作显示用）：
- 亮度 / 对比度
- 伽马校正
- CLAHE（自适应直方图均衡）
- 反相

所有算法接受 OpenCV BGR 或灰度 numpy 数组，返回处理后的同形状数组。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict

import cv2
import numpy as np


@dataclass
class EnhanceParams:
    """图像增强参数集合。

    各参数取值范围（默认值即"无变化"）：
    - brightness: [-100, 100]，0 = 不变
    - contrast: [-100, 100]，0 = 不变（>0 增对比度，<0 降）
    - gamma: [0.1, 3.0]，1.0 = 不变（<1 提亮暗部，>1 压暗）
    - clahe: [0, 10]，0 = 不启用，>0 为 clipLimit（建议 2~4）
    - invert: 是否反相（黑白互换）
    """

    brightness: int = 0
    contrast: int = 0
    gamma: float = 1.0
    clahe: float = 0.0
    invert: bool = False

    def is_identity(self) -> bool:
        """无任何增强 → 跳过处理直接返回原图，提升性能。"""
        return (
            self.brightness == 0
            and self.contrast == 0
            and abs(self.gamma - 1.0) < 1e-6
            and self.clahe <= 0.0
            and not self.invert
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnhanceParams":
        if not isinstance(data, dict):
            return cls()
        return cls(
            brightness=int(data.get("brightness", 0)),
            contrast=int(data.get("contrast", 0)),
            gamma=float(data.get("gamma", 1.0)),
            clahe=float(data.get("clahe", 0.0)),
            invert=bool(data.get("invert", False)),
        )


def _apply_brightness_contrast(img: np.ndarray, brightness: int, contrast: int) -> np.ndarray:
    """alpha * img + beta，alpha 控制对比度，beta 控制亮度。

    contrast ∈ [-100, 100] → alpha ∈ [0.0, 2.0]
    brightness ∈ [-100, 100] → beta ∈ [-100, 100]
    """
    if brightness == 0 and contrast == 0:
        return img
    alpha = 1.0 + contrast / 100.0  # 100 → 2.0; -100 → 0.0
    beta = float(brightness)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def _apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma 校正：output = (input / 255) ^ (1/gamma) * 255。"""
    if abs(gamma - 1.0) < 1e-6:
        return img
    inv_gamma = 1.0 / max(gamma, 1e-3)
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
    ).astype(np.uint8)
    return cv2.LUT(img, table)


def _apply_clahe(img: np.ndarray, clip_limit: float) -> np.ndarray:
    """CLAHE 自适应直方图均衡，对 X 光片细节增强非常显著。

    彩色图像在 LAB 空间的 L 通道做 CLAHE，避免色彩偏移。
    """
    if clip_limit <= 0.0:
        return img
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    if img.ndim == 2:
        return clahe.apply(img)
    # BGR → LAB → CLAHE on L → BGR
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def apply_enhancement(img: np.ndarray, params: EnhanceParams) -> np.ndarray:
    """应用完整增强管线。

    顺序: CLAHE → 伽马 → 亮度/对比度 → 反相
    （CLAHE 放最前，让后续操作基于"已展开动态范围"的图）
    """
    if params is None or params.is_identity():
        return img

    out = img
    out = _apply_clahe(out, params.clahe)
    out = _apply_gamma(out, params.gamma)
    out = _apply_brightness_contrast(out, params.brightness, params.contrast)
    if params.invert:
        out = cv2.bitwise_not(out)
    return out
