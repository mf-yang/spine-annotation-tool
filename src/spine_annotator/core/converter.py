"""YOLO format reader/writer for AABB and OBB annotations."""

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import ImageAnnotation, OBBAnnotation, Point


class YOLOConverter:
    """Convert between YOLO formats and internal annotation model."""

    def __init__(self, class_names: Optional[Dict[int, str]] = None):
        self.class_names = class_names or {
            0: "Vertebra",
            1: "scoliosis spine",
            2: "normal spine",
        }

    def validate_dataset(self, dataset_root: str) -> Tuple[bool, str]:
        """Validate if directory is a valid YOLO dataset.
        
        Returns: (is_valid, message)
        """
        root = Path(dataset_root)
        if not root.exists():
            return False, f"目录不存在: {dataset_root}"
        if not root.is_dir():
            return False, f"不是目录: {dataset_root}"

        splits = ["train", "valid", "test"]
        is_root = any((root / s / "images").exists() for s in splits)

        if is_root:
            found = []
            for s in splits:
                img_dir = root / s / "images"
                if img_dir.exists():
                    count = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
                    found.append(f"{s}: {count} 张")
            if not found:
                return False, "在 train/valid/test 目录下未找到图片（支持 .jpg/.png）"
            return True, f"检测到数据集结构: {', '.join(found)}"
        else:
            # Check if it's a single split directory
            img_dir = root / "images"
            if img_dir.exists():
                count = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
                if count > 0:
                    return True, f"检测到单分片目录，包含 {count} 张图片"
            # Check if images are directly in this dir
            jpg_count = len(list(root.glob("*.jpg")))
            png_count = len(list(root.glob("*.png")))
            if jpg_count + png_count > 0:
                return False, (
                    f"目录中有 {jpg_count + png_count} 张图片，但不符合 YOLO 数据集格式。\n\n"
                    "期望格式：\n"
                    "  根目录/\n"
                    "    train/images/ + train/labels/\n"
                    "    valid/images/ + valid/labels/\n"
                    "    test/images/ + test/labels/\n\n"
                    "或者单分片：\n"
                    "  目录/\n"
                    "    images/\n"
                    "    labels/"
                )
            return False, (
                "不是有效的 YOLO 数据集目录。\n\n"
                "期望格式：\n"
                "  根目录/\n"
                "    train/images/ + train/labels/\n"
                "    valid/images/ + valid/labels/\n"
                "    test/images/ + test/labels/\n\n"
                "或者单分片：\n"
                "  目录/images/ + 目录/labels/"
            )

    def scan_dataset(self, dataset_root: str) -> List[dict]:
        """Scan dataset and return list of image info (without loading pixels).
        
        Auto-detects: if selected dir contains train/valid/test, use as root.
        If selected dir IS a split (contains images/), scan directly.
        """
        root = Path(dataset_root)
        result = []

        # Detect if this is the dataset root (has train/, valid/, test/ subdirs)
        splits = ["train", "valid", "test"]
        is_root = any((root / s / "images").exists() for s in splits)

        if is_root:
            scan_dirs = [(root / s / "images", root / s / "labels", s) for s in splits]
        else:
            # Assume selected dir is already a split directory
            scan_dirs = [(root / "images", root / "labels", root.name)]

        for images_dir, labels_dir, split_name in scan_dirs:
            if not images_dir.exists():
                continue

            for img_path in sorted(images_dir.glob("*.jpg")):
                img_abs = str(img_path.resolve())
                label_path = labels_dir / (img_path.stem + ".txt")

                # Get image dimensions without loading full pixel data
                from PIL import Image
                try:
                    with Image.open(img_abs) as img:
                        w_img, h_img = img.size
                except Exception:
                    continue

                result.append({
                    "image_path": img_abs,
                    "label_path": str(label_path) if label_path.exists() else None,
                    "width": w_img,
                    "height": h_img,
                    "has_labels": label_path.exists(),
                    "split": split_name,
                })

        return result

    # cache 中存放元信息（如 last_image_path）的特殊 key，
    # 以双下划线包围避免与真实图片路径冲突
    META_KEY = "__meta__"

    def load_single(self, image_path: str, label_path: Optional[str],
                    img_w: int, img_h: int,
                    cache_entry: Optional[dict] = None) -> ImageAnnotation:
        """Load annotations for a single image on demand.

        加载优先级：
        1. 若 cache_entry 中含有完整 `points`（OBB 四角点像素坐标），
           使用 cache 的几何状态重建（保留之前的旋转/移动/可见性编辑）
        2. 否则从原始 YOLO label 文件读 AABB（水平矩形）

        Args:
            cache_entry: 可选，用于从缓存恢复每个标注的 OBB 几何与可见性
        """
        annotation = ImageAnnotation(
            image_path=image_path,
            image_width=img_w,
            image_height=img_h,
        )

        if label_path and Path(label_path).exists():
            annotation.annotations = self._load_labels(
                Path(label_path), img_w, img_h
            )

            # 从 cache 恢复编辑后的 OBB 几何 + keypoint_visibility
            # （按索引一一对应；如果索引超出范围则保留原 AABB 几何）
            if cache_entry and "annotation_states" in cache_entry:
                states = cache_entry["annotation_states"]
                for i, ann in enumerate(annotation.annotations):
                    if i >= len(states):
                        continue
                    state = states[i]
                    # 恢复几何（如果 cache 中有完整 4 点坐标）
                    pts = state.get("points")
                    if (
                        isinstance(pts, list) and len(pts) == 4
                        and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in pts)
                    ):
                        ann.points = [Point(float(p[0]), float(p[1])) for p in pts]
                        ann._update_geometry()
                    # 恢复可见性
                    ann.keypoint_visibility = int(
                        state.get("keypoint_visibility", 2)
                    )

        return annotation

    def save_progress_cache(self, cache_path: str, progress: dict):
        """Save progress cache to JSON file."""
        import json
        with open(cache_path, "w") as f:
            json.dump(progress, f, indent=2)

    def load_progress_cache(self, cache_path: str) -> dict:
        """Load progress cache from JSON file."""
        import json
        if not Path(cache_path).exists():
            return {}
        with open(cache_path, "r") as f:
            return json.load(f)

    # --- cache 元信息 helpers ---

    def get_last_image_path(self, cache: dict) -> Optional[str]:
        """读取上次编辑的图片路径（用于启动时智能跳转）。"""
        meta = cache.get(self.META_KEY, {})
        return meta.get("last_image_path")

    def set_last_image_path(self, cache: dict, image_path: str) -> None:
        """记录当前正在编辑的图片路径到 cache（不写盘，调用方负责落盘）。"""
        meta = cache.setdefault(self.META_KEY, {})
        meta["last_image_path"] = image_path

    def build_annotation_states(self, annotation: ImageAnnotation) -> list:
        """从 ImageAnnotation 提取每个标注的可序列化状态，用于写入 cache。

        保存的状态：
          - points: 4 个角点的像素坐标 [[x, y], ...]，保留旋转/移动/角点拖拽编辑结果
          - keypoint_visibility: YOLOv8-pose v 字段
        """
        states = []
        for ann in annotation.annotations:
            states.append({
                "points": [[round(p.x, 3), round(p.y, 3)] for p in ann.points],
                "keypoint_visibility": int(ann.keypoint_visibility),
            })
        return states

    def _load_labels(self, label_path: Path,
                     img_w: int, img_h: int) -> List[OBBAnnotation]:
        """Load YOLOv5 format labels and convert to OBB annotations."""
        annotations = []

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                cx_norm = float(parts[1])
                cy_norm = float(parts[2])
                w_norm = float(parts[3])
                h_norm = float(parts[4])

                # Convert to pixel coordinates
                cx = cx_norm * img_w
                cy = cy_norm * img_h
                w = w_norm * img_w
                h = h_norm * img_h

                class_name = self.class_names.get(class_id, f"class_{class_id}")
                ann = OBBAnnotation.from_aabb(class_id, class_name, cx, cy, w, h)
                annotations.append(ann)

        return annotations

    def save_obb_yolov8(self, annotation: ImageAnnotation,
                        output_dir: str, overwrite: bool = False):
        """Save annotations in YOLOv8-OBB format.
        
        Format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        img_path = Path(annotation.image_path)
        label_name = img_path.stem + ".txt"
        label_path = output / label_name

        if label_path.exists() and not overwrite:
            return False

        w_img = annotation.image_width
        h_img = annotation.image_height

        with open(label_path, "w") as f:
            for ann in annotation.annotations:
                # 4 corner points, normalized
                coords = []
                for p in ann.points:
                    coords.append(f"{p.x / w_img:.6f}")
                    coords.append(f"{p.y / h_img:.6f}")

                line = f"{ann.class_id} {' '.join(coords)}\n"
                f.write(line)

        return True

    def save_obb_xywhr(self, annotation: ImageAnnotation,
                       output_dir: str, overwrite: bool = False):
        """Save annotations in YOLOv8-OBB xywhr format.
        
        Format: class_id cx cy w h angle (normalized, angle in radians [-pi/4, pi/4))
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        img_path = Path(annotation.image_path)
        label_name = img_path.stem + ".txt"
        label_path = output / label_name

        if label_path.exists() and not overwrite:
            return False

        w_img = annotation.image_width
        h_img = annotation.image_height

        with open(label_path, "w") as f:
            for ann in annotation.annotations:
                cx, cy, w, h, angle = ann.to_xywhr()

                # Normalize angle to [-pi/4, pi/4)
                angle = self._normalize_angle(angle)

                line = (
                    f"{ann.class_id} "
                    f"{cx / w_img:.6f} {cy / h_img:.6f} "
                    f"{w / w_img:.6f} {h / h_img:.6f} "
                    f"{angle:.6f}\n"
                )
                f.write(line)

        return True

    def save_pose_yolov8(self, annotation: ImageAnnotation,
                        output_dir: str, overwrite: bool = False):
        """Save annotations in YOLOv8-pose format.

        Format per line (all normalized to [0, 1]):
            class_id  cx cy w h  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4

        - bbox(cx, cy, w, h): 包围 OBB 四个角点的 AABB
        - keypoints: 椎骨矩形的 4 个角点，顺时针排列
            x1,y1 = 左上, x2,y2 = 右上, x3,y3 = 右下, x4,y4 = 左下
        - v: 可见性 (0=不可见, 1=遮挡, 2=可见)
          取自 OBBAnnotation.keypoint_visibility（对该标注 4 个点统一生效），默认 2
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        img_path = Path(annotation.image_path)
        label_name = img_path.stem + ".txt"
        label_path = output / label_name

        if label_path.exists() and not overwrite:
            return False

        w_img = annotation.image_width
        h_img = annotation.image_height

        with open(label_path, "w") as f:
            for ann in annotation.annotations:
                xs = [p.x for p in ann.points]
                ys = [p.y for p in ann.points]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                bbox_w = x_max - x_min
                bbox_h = y_max - y_min
                bbox_cx = x_min + bbox_w / 2
                bbox_cy = y_min + bbox_h / 2

                v = int(ann.keypoint_visibility)

                parts = [
                    str(ann.class_id),
                    f"{bbox_cx / w_img:.6f}",
                    f"{bbox_cy / h_img:.6f}",
                    f"{bbox_w / w_img:.6f}",
                    f"{bbox_h / h_img:.6f}",
                ]

                # Keypoints: 顺时针 左上, 右上, 右下, 左下
                # OBBAnnotation.points 即按此顺序存储
                for p in ann.points:
                    parts.append(f"{p.x / w_img:.6f}")
                    parts.append(f"{p.y / h_img:.6f}")
                    parts.append(str(v))

                f.write(" ".join(parts) + "\n")

        return True

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi/4, pi/4) range for YOLOv8-OBB."""
        # Reduce to [-pi/2, pi/2)
        angle = angle % math.pi
        if angle >= math.pi / 2:
            angle -= math.pi

        # If angle is outside [-pi/4, pi/4), swap width/height
        if angle >= math.pi / 4:
            angle -= math.pi / 2
        elif angle < -math.pi / 4:
            angle += math.pi / 2

        return angle

    @staticmethod
    def load_yaml_config(yaml_path: str) -> dict:
        """Load dataset YAML config."""
        import yaml
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)
