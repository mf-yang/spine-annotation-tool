"""Main application window for the spine annotation tool."""

import math
import os
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import Qt, QSettings, QTimer
from PyQt5.QtGui import QBrush, QColor, QKeySequence, QPalette
from PyQt5.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QRadioButton, QShortcut,
    QVBoxLayout, QWidget,
)

from ..core.converter import YOLOConverter
from ..core.models import ImageAnnotation
from .image_canvas import AnnotationCanvas


class MainWindow(QMainWindow):
    """Main application window."""

    # QSettings 的 key 常量
    SETTINGS_LAST_DATASET = "last_dataset_dir"
    SETTINGS_LAST_OUTPUT = "last_output_dir"
    SETTINGS_LAST_FORMAT = "last_export_format"  # 字符串：yolov8_obb / yolov8_xywhr / yolov8_pose

    def __init__(self):
        super().__init__()
        self.setWindowTitle("脊柱椎骨标注工具 - Spine Annotator")
        self.resize(1400, 900)

        # 持久化设置（macOS 写到 ~/Library/Preferences、Linux 写 INI、Windows 写注册表）
        self._settings = QSettings("spine-annotator", "spine-annotator")

        # Data
        self._converter = YOLOConverter()
        self._image_infos: List[dict] = []  # scanned image metadata
        self._current_index: int = -1
        self._output_dir: Optional[str] = None
        self._export_format: str = "yolov8_obb"
        self._dataset_root: Optional[str] = None
        self._progress_cache_path: Optional[str] = None

        # Current loaded annotation (only one at a time)
        self._current_annotation: Optional[ImageAnnotation] = None
        self._cache: dict = {}  # progress cache
        
        # Layer visibility
        self._show_vertebrae = True
        self._show_spine = True

        self._init_ui()
        self._init_shortcuts()
        self._init_statusbar()

        # 启动后异步恢复上次会话（让窗口先显示出来，避免大数据集扫描时白屏）
        QTimer.singleShot(0, self._restore_last_session)

    def _init_ui(self):
        """Initialize the UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # --- Left: image list panel ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(QLabel("图片列表:"))

        self._image_list_widget = QListWidget()
        self._image_list_widget.currentRowChanged.connect(self._on_image_selected)
        left_layout.addWidget(self._image_list_widget)

        # Progress
        self._progress_bar = QProgressBar()
        self._progress_bar.setFormat("已标注: %v / %m")
        left_layout.addWidget(self._progress_bar)

        left_panel.setFixedWidth(220)

        # --- Center: canvas ---
        self._canvas = AnnotationCanvas()
        self._canvas.selection_changed.connect(self._on_annotation_selected)
        self._canvas.annotation_modified.connect(self._on_annotation_modified)

        # --- Right: controls panel ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)

        # Dataset controls
        right_layout.addWidget(self._create_section_label("数据集"))
        self._btn_open_dataset = QPushButton("打开 YOLO 数据集")
        self._btn_open_dataset.clicked.connect(self._open_dataset)
        right_layout.addWidget(self._btn_open_dataset)

        self._dataset_path_label = QLabel("未加载")
        self._dataset_path_label.setWordWrap(True)
        self._dataset_path_label.setStyleSheet(self._muted_text_style(11))
        right_layout.addWidget(self._dataset_path_label)

        # Export controls
        right_layout.addWidget(self._create_section_label("导出"))
        self._btn_set_output = QPushButton("设置输出目录")
        self._btn_set_output.clicked.connect(self._set_output_dir)
        right_layout.addWidget(self._btn_set_output)

        self._output_path_label = QLabel("未设置")
        self._output_path_label.setWordWrap(True)
        self._output_path_label.setStyleSheet(self._muted_text_style(11))
        right_layout.addWidget(self._output_path_label)

        self._format_combo = QComboBox()
        self._format_combo.addItems([
            "YOLOv8-OBB (四角点)",
            "YOLOv8-OBB (xywhr)",
            "YOLOv8-pose (bbox + 4 关键点)",
        ])
        self._format_combo.currentIndexChanged.connect(self._on_format_changed)
        right_layout.addWidget(self._format_combo)

        self._btn_save = QPushButton("保存当前 (Ctrl+S)")
        self._btn_save.clicked.connect(self._save_current)
        self._btn_save.setEnabled(False)
        right_layout.addWidget(self._btn_save)

        self._btn_save_all = QPushButton("全部导出")
        self._btn_save_all.clicked.connect(self._save_all)
        self._btn_save_all.setEnabled(False)
        right_layout.addWidget(self._btn_save_all)

        right_layout.addSpacing(16)

        # Layer control
        right_layout.addWidget(self._create_section_label("图层控制"))
        
        self._chk_vertebrae = QCheckBox("显示椎骨框 (绿色)")
        self._chk_vertebrae.setChecked(True)
        self._chk_vertebrae.stateChanged.connect(self._on_layer_changed)
        right_layout.addWidget(self._chk_vertebrae)
        
        self._chk_spine = QCheckBox("显示脊柱框 (红色/蓝色)")
        self._chk_spine.setChecked(True)
        self._chk_spine.stateChanged.connect(self._on_layer_changed)
        right_layout.addWidget(self._chk_spine)

        right_layout.addSpacing(12)

        # Annotation info
        right_layout.addWidget(self._create_section_label("当前标注"))
        self._ann_info_label = QLabel("无选中")
        self._ann_info_label.setWordWrap(True)
        right_layout.addWidget(self._ann_info_label)

        # Keypoint visibility (YOLOv8-pose v 字段，对该标注 4 个角点统一生效)
        right_layout.addWidget(self._create_section_label("关键点可见性"))
        vis_row = QHBoxLayout()
        self._vis_group = QButtonGroup(self)
        self._rb_vis_2 = QRadioButton("可见 (2)")
        self._rb_vis_1 = QRadioButton("遮挡 (1)")
        self._rb_vis_0 = QRadioButton("不可见 (0)")
        self._rb_vis_2.setToolTip("肉眼清晰可见，是默认值")
        self._rb_vis_1.setToolTip("被金属植入物 / 伪影遮挡，坐标可信")
        self._rb_vis_0.setToolTip("肉眼无法看清（脊柱骨过于透明），坐标基于相邻椎骨推断")
        self._vis_group.addButton(self._rb_vis_2, 2)
        self._vis_group.addButton(self._rb_vis_1, 1)
        self._vis_group.addButton(self._rb_vis_0, 0)
        self._rb_vis_2.setChecked(True)
        vis_row.addWidget(self._rb_vis_2)
        vis_row.addWidget(self._rb_vis_1)
        vis_row.addWidget(self._rb_vis_0)
        right_layout.addLayout(vis_row)
        self._vis_group.buttonClicked.connect(self._on_visibility_changed)

        # Rotation controls
        right_layout.addWidget(self._create_section_label("旋转微调"))
        rotate_row = QHBoxLayout()
        self._btn_rotate_ccw = QPushButton("← 逆时针")
        self._btn_rotate_ccw.clicked.connect(lambda: self._canvas.rotate_selected(-5))
        self._btn_rotate_cw = QPushButton("顺时针 →")
        self._btn_rotate_cw.clicked.connect(lambda: self._canvas.rotate_selected(5))
        rotate_row.addWidget(self._btn_rotate_ccw)
        rotate_row.addWidget(self._btn_rotate_cw)
        right_layout.addLayout(rotate_row)

        # Fine rotation
        fine_row = QHBoxLayout()
        self._btn_rotate_ccw_fine = QPushButton("-1°")
        self._btn_rotate_ccw_fine.clicked.connect(lambda: self._canvas.rotate_selected(-1))
        self._btn_rotate_cw_fine = QPushButton("+1°")
        self._btn_rotate_cw_fine.clicked.connect(lambda: self._canvas.rotate_selected(1))
        fine_row.addWidget(self._btn_rotate_ccw_fine)
        fine_row.addWidget(self._btn_rotate_cw_fine)
        right_layout.addLayout(fine_row)

        # Custom angle
        angle_row = QHBoxLayout()
        self._angle_spin = QDoubleSpinBox()
        self._angle_spin.setRange(-180, 180)
        self._angle_spin.setDecimals(1)
        self._angle_spin.setSingleStep(1.0)
        self._angle_spin.setSuffix("°")
        self._angle_spin.valueChanged.connect(self._on_angle_spin_changed)
        angle_row.addWidget(self._angle_spin)

        self._btn_set_angle = QPushButton("应用")
        self._btn_set_angle.clicked.connect(self._apply_angle)
        angle_row.addWidget(self._btn_set_angle)
        right_layout.addLayout(angle_row)

        # Navigation
        right_layout.addSpacing(16)
        right_layout.addWidget(self._create_section_label("导航"))
        nav_row = QHBoxLayout()
        self._btn_prev = QPushButton("← 上一张")
        self._btn_prev.clicked.connect(self._prev_image)
        self._btn_next = QPushButton("下一张 →")
        self._btn_next.clicked.connect(self._next_image)
        nav_row.addWidget(self._btn_prev)
        nav_row.addWidget(self._btn_next)
        right_layout.addLayout(nav_row)

        # Shortcut reference
        right_layout.addSpacing(12)
        right_layout.addWidget(self._create_section_label("快捷键"))
        shortcut_help = QLabel(
            "旋转: R/E ±5° | T/Y ±1°\n"
            "精旋: Shift+R/E ±0.5°\n"
            "移动: W/A/S/D 5px\n"
            "精移: Shift+W/A/S/D 1px\n"
            "导航: ←/→ 图片 | ↑/↓ 标注\n"
            "跳转: Ctrl+N 下一未标注\n"
            "      Ctrl+B 上一未标注\n"
            "其他: F 适配 | Esc 取消\n"
            "      Ctrl+S 保存"
        )
        shortcut_help.setStyleSheet(self._muted_text_style(11) + " line-height: 1.4;")
        shortcut_help.setWordWrap(True)
        right_layout.addWidget(shortcut_help)

        # Color legend
        right_layout.addSpacing(8)
        color_legend = QLabel(
            "■ 绿色: 椎骨 | ■ 红/蓝: 脊柱"
        )
        color_legend.setStyleSheet(self._muted_text_style(11))
        color_legend.setWordWrap(True)
        right_layout.addWidget(color_legend)

        right_layout.addStretch()
        right_panel.setFixedWidth(240)

        # --- Assemble ---
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self._canvas, stretch=1)
        main_layout.addWidget(right_panel)

    def _create_section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        # 不设 color，让 Qt 自动用主题默认文字色（深 / 浅模式都能看清）
        # 仅用加粗 + 字号体现层次
        label.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px 0;")
        return label

    def _muted_text_style(self, font_size: int = 11) -> str:
        """返回适合当前主题（深 / 浅）的"次要信息"文本样式。

        深色模式：浅灰文字（#aaaaaa），在深色背景上清晰可读
        浅色模式：深灰文字（#666666），传统次要信息观感
        """
        is_dark = self.palette().color(QPalette.Window).lightness() < 128
        color = "#aaaaaa" if is_dark else "#666666"
        return f"color: {color}; font-size: {font_size}px;"

    def _init_shortcuts(self):
        """Initialize keyboard shortcuts."""
        shortcuts = {
            # Navigation
            QKeySequence("Left"): self._prev_image,
            QKeySequence("Right"): self._next_image,
            QKeySequence("Up"): self._select_prev_annotation,
            QKeySequence("Down"): self._select_next_annotation,
            # Rotation (coarse ±5°)
            QKeySequence("R"): lambda: self._canvas.rotate_selected(-5),
            QKeySequence("E"): lambda: self._canvas.rotate_selected(5),
            # Rotation (fine ±1°)
            QKeySequence("T"): lambda: self._canvas.rotate_selected(-1),
            QKeySequence("Y"): lambda: self._canvas.rotate_selected(1),
            # Rotation (super fine ±0.5°)
            QKeySequence("Shift+R"): lambda: self._canvas.rotate_selected(-0.5),
            QKeySequence("Shift+E"): lambda: self._canvas.rotate_selected(0.5),
            # Move (coarse 5px)
            QKeySequence("W"): lambda: self._canvas.move_selected(0, -5),
            QKeySequence("S"): lambda: self._canvas.move_selected(0, 5),
            QKeySequence("A"): lambda: self._canvas.move_selected(-5, 0),
            QKeySequence("D"): lambda: self._canvas.move_selected(5, 0),
            # Move (fine 1px)
            QKeySequence("Shift+W"): lambda: self._canvas.move_selected(0, -1),
            QKeySequence("Shift+S"): lambda: self._canvas.move_selected(0, 1),
            QKeySequence("Shift+A"): lambda: self._canvas.move_selected(-1, 0),
            QKeySequence("Shift+D"): lambda: self._canvas.move_selected(1, 0),
            # Save / undo / fit
            QKeySequence("Ctrl+S"): self._save_current,
            QKeySequence("Ctrl+Z"): self._undo,
            QKeySequence("Escape"): lambda: self._canvas.select_annotation(-1),
            QKeySequence("F"): self._fit_view,
            # 跳到下一张 / 上一张未标注图片（断点续标核心快捷键）
            QKeySequence("Ctrl+N"): self._jump_to_next_unannotated,
            QKeySequence("Ctrl+B"): self._jump_to_prev_unannotated,
        }
        for key, callback in shortcuts.items():
            shortcut = QShortcut(key, self)
            shortcut.activated.connect(callback)

    def _init_statusbar(self):
        # 永久显示在右侧的"进度统计"标签（已标注 N/Total · X%）
        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("color: #2c7be5; font-weight: bold;")
        self.statusBar().addPermanentWidget(self._progress_label)
        # 永久显示在最右侧的"当前图片"信息
        self._status_label = QLabel("")
        self.statusBar().addPermanentWidget(self._status_label)

    # --- Dataset Operations ---

    def _open_dataset(self):
        """用户主动选择 YOLOv5/v8 数据集目录。"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择 YOLO 数据集目录",
            self._settings.value(self.SETTINGS_LAST_DATASET, str(Path.home())),
        )
        if not dir_path:
            return
        self._load_dataset(dir_path)

    def _load_dataset(self, dir_path: str, *, silent_invalid: bool = False) -> bool:
        """加载指定路径的 YOLO 数据集（共用：用户主动打开 & 启动自动恢复）。

        Args:
            silent_invalid: True 时，校验失败不弹窗（用于启动恢复，避免打扰）
        Returns:
            是否加载成功
        """
        is_valid, message = self._converter.validate_dataset(dir_path)
        if not is_valid:
            if not silent_invalid:
                QMessageBox.warning(self, "数据集格式错误", message)
            return False

        self._dataset_root = dir_path
        self._progress_cache_path = os.path.join(dir_path, ".annotate_progress.json")
        self._dataset_path_label.setText(f"{dir_path}\n{message}")
        self.statusBar().showMessage("正在扫描数据集...")

        # Scan images (fast, no pixel loading)
        self._image_infos = self._converter.scan_dataset(dir_path)

        # Load progress cache
        self._cache = self._converter.load_progress_cache(self._progress_cache_path)

        # Populate list (颜色按缓存状态决定)
        self._image_list_widget.clear()
        for idx, info in enumerate(self._image_infos):
            name = Path(info["image_path"]).stem
            split = info.get("split", "")
            display = f"[{split}] {name}" if split else name
            item = QListWidgetItem(display)
            self._image_list_widget.addItem(item)
            self._apply_item_style(idx)

        self._progress_bar.setMaximum(len(self._image_infos))
        self._update_progress()

        self.statusBar().showMessage(f"已扫描 {len(self._image_infos)} 张图片")
        self._btn_save_all.setEnabled(True)

        # 持久化数据集目录
        self._settings.setValue(self.SETTINGS_LAST_DATASET, dir_path)

        # 智能跳转：1) 优先恢复 last_image_path  2) 否则跳到第一张未标注
        target_index = self._resolve_initial_index()
        if target_index >= 0:
            self._go_to_image(target_index)
        return True

    def _restore_last_session(self):
        """启动后自动恢复上次的数据集 + 输出目录 + 导出格式。"""
        # 1) 导出格式（无 IO，最先恢复）
        last_fmt = self._settings.value(self.SETTINGS_LAST_FORMAT, "")
        if last_fmt in ("yolov8_obb", "yolov8_xywhr", "yolov8_pose"):
            self._export_format = last_fmt
            fmt_index = {"yolov8_obb": 0, "yolov8_xywhr": 1, "yolov8_pose": 2}[last_fmt]
            self._format_combo.blockSignals(True)
            self._format_combo.setCurrentIndex(fmt_index)
            self._format_combo.blockSignals(False)

        # 2) 输出目录（先于数据集恢复，确保保存功能可用）
        last_output = self._settings.value(self.SETTINGS_LAST_OUTPUT, "")
        if last_output and Path(last_output).is_dir():
            self._output_dir = last_output
            self._output_path_label.setText(last_output)
            self._btn_save.setEnabled(True)

        # 3) 数据集目录（异步加载）
        last_dataset = self._settings.value(self.SETTINGS_LAST_DATASET, "")
        if last_dataset and Path(last_dataset).is_dir():
            ok = self._load_dataset(last_dataset, silent_invalid=True)
            if ok:
                self.statusBar().showMessage(
                    f"已自动恢复上次会话：{Path(last_dataset).name}", 4000
                )

    def _resolve_initial_index(self) -> int:
        """决定打开数据集后应该跳转到哪张图片。

        策略：
        1. 如果 cache 中记录了 last_image_path 且该图片仍存在于扫描结果中：
           - 若该图片**未标注**，直接跳过去（用户继续上次的工作）
           - 若已标注，转入策略 2
        2. 跳到第一张未标注图片
        3. 全部已标注：跳回 last_image_path 或第一张
        """
        if not self._image_infos:
            return -1

        last_path = self._converter.get_last_image_path(self._cache)
        last_index = -1
        if last_path:
            for i, info in enumerate(self._image_infos):
                if info["image_path"] == last_path:
                    last_index = i
                    break

        # 优先：上次位置且未标注
        if last_index >= 0 and not self._is_saved(self._image_infos[last_index]["image_path"]):
            return last_index

        # 其次：第一张未标注
        for i, info in enumerate(self._image_infos):
            if not self._is_saved(info["image_path"]):
                return i

        # 兜底：全部已标注 → 上次位置或第一张
        return last_index if last_index >= 0 else 0

    def _is_saved(self, image_path: str) -> bool:
        return bool(self._cache.get(image_path, {}).get("saved"))

    def _set_output_dir(self):
        """Set the output directory for exported annotations."""
        start_dir = self._settings.value(self.SETTINGS_LAST_OUTPUT, str(Path.home()))
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择输出目录", start_dir,
        )
        if dir_path:
            self._output_dir = dir_path
            self._output_path_label.setText(dir_path)
            self._btn_save.setEnabled(True)
            self._settings.setValue(self.SETTINGS_LAST_OUTPUT, dir_path)

    def _on_format_changed(self, index: int):
        if index == 0:
            self._export_format = "yolov8_obb"
        elif index == 1:
            self._export_format = "yolov8_xywhr"
        else:
            self._export_format = "yolov8_pose"
        self._settings.setValue(self.SETTINGS_LAST_FORMAT, self._export_format)

    # --- Navigation ---

    def _go_to_image(self, index: int):
        """Navigate to a specific image (lazy load on demand)."""
        if not (0 <= index < len(self._image_infos)):
            return

        # 切换前持久化当前张：
        #   - 已设置 output_dir：正常保存（写 labels 文件 + cache）
        #   - 未设置 output_dir：仍把 OBB 几何状态记入 cache，避免丢失编辑
        if self._current_index >= 0 and self._current_annotation and self._current_annotation.modified:
            if self._output_dir:
                self._save_current(silent=True)
            else:
                self._checkpoint_geometry_to_cache()

        prev_index = self._current_index
        self._current_index = index
        info = self._image_infos[index]

        # Lazy load annotations for this image
        cache_entry = self._cache.get(info["image_path"])
        self._current_annotation = self._converter.load_single(
            info["image_path"], info["label_path"], info["width"], info["height"],
            cache_entry=cache_entry,
        )

        # Apply layer visibility
        self._apply_layer_visibility()

        self._image_list_widget.setCurrentRow(index)
        self._canvas.load_image(info["image_path"], self._current_annotation.annotations)

        # 记录最后访问位置 + 落盘（便于断点续标）
        if self._progress_cache_path:
            self._converter.set_last_image_path(self._cache, info["image_path"])
            self._converter.save_progress_cache(self._progress_cache_path, self._cache)

        # 切换前后两行的视觉样式（更新加粗状态）
        if prev_index >= 0 and prev_index != index:
            self._apply_item_style(prev_index)
        self._apply_item_style(index)

        self._update_status()

    def _prev_image(self):
        self._go_to_image(self._current_index - 1)

    def _next_image(self):
        self._go_to_image(self._current_index + 1)

    def _on_image_selected(self, row: int):
        if row >= 0 and row != self._current_index:
            self._go_to_image(row)

    def _select_prev_annotation(self):
        if not self._image_infos:
            return
        if self._canvas._obb_items:
            idx = self._canvas._current_selection - 1
            if idx < 0:
                idx = len(self._canvas._obb_items) - 1
            self._canvas.select_annotation(idx)

    def _select_next_annotation(self):
        if not self._image_infos:
            return
        if self._canvas._obb_items:
            idx = self._canvas._current_selection + 1
            if idx >= len(self._canvas._obb_items):
                idx = 0
            self._canvas.select_annotation(idx)

    def _fit_view(self):
        """Fit the image to the canvas view."""
        if self._canvas._scene.sceneRect():
            self._canvas.fitInView(
                self._canvas._scene.sceneRect(), Qt.KeepAspectRatio
            )

    # --- Annotation Operations ---

    def _on_annotation_selected(self, index: int):
        """Update info panel when selection changes."""
        if index < 0 or not self._current_annotation:
            self._ann_info_label.setText("无选中")
            self._sync_visibility_radios(default_v=2, enabled=False)
            return

        # Map scene index to original annotation index
        if 0 <= index < len(self._canvas._index_map):
            orig_idx = self._canvas._index_map[index]
        else:
            orig_idx = index

        if 0 <= orig_idx < len(self._current_annotation.annotations):
            ann = self._current_annotation.annotations[orig_idx]
            angle_deg = math.degrees(ann.angle)
            v_label = {2: "可见", 1: "遮挡", 0: "不可见"}.get(
                int(ann.keypoint_visibility), "可见"
            )
            self._ann_info_label.setText(
                f"类别: {ann.class_name} (ID={ann.class_id})\n"
                f"角度: {angle_deg:.1f}°\n"
                f"尺寸: {ann.width:.0f} x {ann.height:.0f}\n"
                f"中心: ({ann.center.x:.0f}, {ann.center.y:.0f})\n"
                f"可见性: v={int(ann.keypoint_visibility)} ({v_label})"
            )
            # Update angle spinbox without triggering signal
            self._angle_spin.blockSignals(True)
            self._angle_spin.setValue(angle_deg)
            self._angle_spin.blockSignals(False)
            # Sync visibility radio buttons
            self._sync_visibility_radios(int(ann.keypoint_visibility), enabled=True)

    def _sync_visibility_radios(self, default_v: int, enabled: bool):
        """同步右侧"关键点可见性" radio button 至给定 v 值（不触发回调）。"""
        self._vis_group.blockSignals(True)
        btn = self._vis_group.button(default_v)
        if btn is not None:
            btn.setChecked(True)
        for v in (0, 1, 2):
            b = self._vis_group.button(v)
            if b is not None:
                b.setEnabled(enabled)
        self._vis_group.blockSignals(False)

    def _on_visibility_changed(self, button):
        """用户切换关键点可见性 radio button 时回调。"""
        ann = self._canvas.get_selected_annotation()
        if ann is None:
            return
        new_v = self._vis_group.id(button)
        if int(ann.keypoint_visibility) == new_v:
            return
        ann.keypoint_visibility = new_v
        if self._current_annotation:
            self._current_annotation.modified = True
        # 重绘画布（边框样式可能变化）+ 刷新信息面板
        if 0 <= self._canvas._current_selection < len(self._canvas._obb_items):
            self._canvas._obb_items[self._canvas._current_selection].update()
        self._canvas.viewport().update()
        self._on_annotation_selected(self._canvas._current_selection)
        self._update_status()
        # 列表项状态颜色更新（modified 状态变化）
        if self._current_index >= 0:
            self._apply_item_style(self._current_index)

    def _on_annotation_modified(self):
        """Mark current image as modified."""
        if self._current_annotation:
            self._current_annotation.modified = True
            self._on_annotation_selected(self._canvas._current_selection)
            self._update_status()
            if self._current_index >= 0:
                self._apply_item_style(self._current_index)

    def _on_layer_changed(self):
        """Handle layer visibility checkbox changes."""
        self._show_vertebrae = self._chk_vertebrae.isChecked()
        self._show_spine = self._chk_spine.isChecked()
        self._apply_layer_visibility()
        self._canvas.viewport().update()

    def _apply_layer_visibility(self):
        """Apply layer visibility to all annotations."""
        if not self._current_annotation:
            return
        for ann in self._current_annotation.annotations:
            if ann.class_id == 0:  # Vertebra
                ann.visible = self._show_vertebrae
            else:  # Spine boxes
                ann.visible = self._show_spine
        # Update canvas items
        for item in self._canvas._obb_items:
            item.setVisible(item.annotation.visible)

    def _on_angle_spin_changed(self, value: float):
        """Called when angle spinbox value changes (but not applied yet)."""
        pass

    def _apply_angle(self):
        """Apply the angle from spinbox to selected annotation."""
        ann = self._canvas.get_selected_annotation()
        if ann is None:
            return

        target_angle = math.radians(self._angle_spin.value())
        delta = target_angle - ann.angle
        self._canvas.rotate_selected(math.degrees(delta))

    def _undo(self):
        """Simple undo: reload original annotations for current image."""
        # For now, just reload from disk
        # TODO: implement proper undo stack
        pass

    # --- Save Operations ---

    def _save_current(self, silent: bool = False):
        """Save current image annotations."""
        if not self._image_infos or not self._output_dir:
            if not silent:
                QMessageBox.warning(self, "提示", "请先设置输出目录")
            return

        if not self._current_annotation:
            return

        info = self._image_infos[self._current_index]

        # Compute split-aware output directory
        split = info.get("split", "")
        if split:
            out_dir = os.path.join(self._output_dir, split, "labels")
        else:
            out_dir = self._output_dir

        if self._export_format == "yolov8_obb":
            self._converter.save_obb_yolov8(self._current_annotation, out_dir, overwrite=True)
        elif self._export_format == "yolov8_xywhr":
            self._converter.save_obb_xywhr(self._current_annotation, out_dir, overwrite=True)
        else:  # yolov8_pose
            self._converter.save_pose_yolov8(self._current_annotation, out_dir, overwrite=True)

        # Update cache (含每个标注的 keypoint_visibility 状态)
        img_path = info["image_path"]
        self._cache[img_path] = {
            "modified": False,
            "saved": True,
            "annotation_states": self._converter.build_annotation_states(
                self._current_annotation
            ),
        }
        self._current_annotation.modified = False

        # Save cache to disk
        self._converter.save_progress_cache(self._progress_cache_path, self._cache)

        # Update UI
        self._apply_item_style(self._current_index)
        self._update_progress()
        self._update_status()

        if not silent:
            self.statusBar().showMessage(f"已保存: {Path(img_path).name}")

    def _checkpoint_geometry_to_cache(self):
        """把当前张 OBB 几何写入 cache（不算正式保存，仅防止编辑丢失）。

        触发场景：用户尚未设置输出目录但已经在编辑，切图时调用本方法
        把几何状态持久化到 .annotate_progress.json，下次回到该图能恢复编辑。
        """
        if not (self._current_annotation and self._progress_cache_path):
            return
        info = self._image_infos[self._current_index]
        img_path = info["image_path"]
        existing = self._cache.get(img_path, {})
        existing.update({
            # 注意：不改 saved 字段（仍保持 false / 之前的值），仅记录几何
            "annotation_states": self._converter.build_annotation_states(
                self._current_annotation
            ),
        })
        # 如果之前没有 saved 字段，确保至少有个 false
        existing.setdefault("saved", False)
        existing["modified"] = True
        self._cache[img_path] = existing
        self._converter.save_progress_cache(self._progress_cache_path, self._cache)

    def _save_all(self):
        """Export all images that have been processed."""
        if not self._output_dir:
            QMessageBox.warning(self, "提示", "请先设置输出目录")
            return

        count = 0
        for i, info in enumerate(self._image_infos):
            # Only save if in cache or currently loaded
            img_path = info["image_path"]
            if img_path in self._cache or i == self._current_index:
                # Load if not current
                if i != self._current_index:
                    ann = self._converter.load_single(
                        info["image_path"], info["label_path"],
                        info["width"], info["height"]
                    )
                else:
                    ann = self._current_annotation

                # Compute split-aware output directory
                split = info.get("split", "")
                if split:
                    out_dir = os.path.join(self._output_dir, split, "labels")
                else:
                    out_dir = self._output_dir

                if self._export_format == "yolov8_obb":
                    self._converter.save_obb_yolov8(ann, out_dir, overwrite=True)
                elif self._export_format == "yolov8_xywhr":
                    self._converter.save_obb_xywhr(ann, out_dir, overwrite=True)
                else:  # yolov8_pose
                    self._converter.save_pose_yolov8(ann, out_dir, overwrite=True)

                self._cache[img_path] = {
                    "modified": False,
                    "saved": True,
                    "annotation_states": self._converter.build_annotation_states(ann),
                }
                count += 1

        # Save cache
        self._converter.save_progress_cache(self._progress_cache_path, self._cache)
        self._update_progress()
        # 刷新所有列表项样式
        for i in range(len(self._image_infos)):
            self._apply_item_style(i)

        self.statusBar().showMessage(f"已导出 {count} 个标注文件")

    def _update_progress(self):
        """Update progress bar + 永久进度标签。"""
        saved = self._count_saved()
        total = len(self._image_infos)
        self._progress_bar.setMaximum(max(total, 1))
        self._progress_bar.setValue(saved)
        if total > 0:
            pct = saved / total * 100
            self._progress_label.setText(
                f"已标注 {saved} / {total}  ·  {pct:.1f}%"
            )
        else:
            self._progress_label.setText("")

    def _count_saved(self) -> int:
        return sum(
            1 for k, v in self._cache.items()
            if k != YOLOConverter.META_KEY and isinstance(v, dict) and v.get("saved")
        )

    def _update_status(self):
        """Update status bar info."""
        if not self._image_infos or not self._current_annotation:
            self._status_label.setText("")
            return
        info = self._image_infos[self._current_index]
        modified_str = " [未保存]" if self._current_annotation.modified else ""
        saved_str = " ✓" if self._is_saved(info["image_path"]) else ""
        self._status_label.setText(
            f"{self._current_index + 1}/{len(self._image_infos)} | "
            f"{Path(info['image_path']).name}{saved_str} | "
            f"{len(self._current_annotation.annotations)} 个标注{modified_str}"
        )

    # --- 列表项三态视觉样式 ---

    def _apply_item_style(self, index: int):
        """更新单个列表项的颜色 + 加粗状态。

        三态颜色（自动适配深 / 浅色主题）：
          - 已修改未保存：橙色（高饱和度，两种主题下都醒目）
          - 已保存：使用系统 Disabled 文本色（浅色主题→灰，深色主题→暗灰）
          - 未标注：使用系统 Active 文本色（浅色主题→黑，深色主题→白）
        当前选中项额外加粗。
        """
        if not (0 <= index < self._image_list_widget.count()):
            return
        item = self._image_list_widget.item(index)
        if item is None:
            return

        info = self._image_infos[index]
        img_path = info["image_path"]
        is_current = (index == self._current_index)
        is_modified = (
            is_current
            and self._current_annotation is not None
            and self._current_annotation.modified
        )
        is_saved = self._is_saved(img_path)

        palette = self._image_list_widget.palette()
        if is_modified:
            # 橙色：浅色主题下深一点，深色主题下也清晰可读
            item.setForeground(QBrush(QColor("#e8590c")))
        elif is_saved:
            # 系统 Disabled 文本色（自动深浅模式适配）
            item.setForeground(palette.brush(QPalette.Disabled, QPalette.Text))
        else:
            # 系统默认文本色（深色主题下会是浅色，浅色主题下会是深色）
            item.setForeground(palette.brush(QPalette.Active, QPalette.Text))

        font = item.font()
        font.setBold(is_current)
        item.setFont(font)

    # --- 断点续标：跳转未标注图片 ---

    def _jump_to_next_unannotated(self):
        """从当前位置往后找下一张未标注图片。"""
        if not self._image_infos:
            return
        start = max(self._current_index + 1, 0)
        n = len(self._image_infos)
        # 先从 start 找到末尾，再从开头找到 start（环绕）
        for offset in range(n):
            i = (start + offset) % n
            if not self._is_saved(self._image_infos[i]["image_path"]):
                self._go_to_image(i)
                self.statusBar().showMessage(
                    f"跳转到第 {i + 1} 张未标注图片", 2000
                )
                return
        self.statusBar().showMessage("已全部标注完成 🎉", 3000)

    def _jump_to_prev_unannotated(self):
        """从当前位置往前找上一张未标注图片。"""
        if not self._image_infos:
            return
        start = self._current_index - 1
        n = len(self._image_infos)
        for offset in range(n):
            i = (start - offset) % n
            if not self._is_saved(self._image_infos[i]["image_path"]):
                self._go_to_image(i)
                self.statusBar().showMessage(
                    f"跳转到第 {i + 1} 张未标注图片", 2000
                )
                return
        self.statusBar().showMessage("已全部标注完成 🎉", 3000)

    # --- 关闭前未保存确认 ---

    def closeEvent(self, event):
        """关闭前如果当前图片未保存，弹窗询问。"""
        if self._current_annotation is not None and self._current_annotation.modified:
            name = Path(self._current_annotation.image_path).name
            choice = QMessageBox.question(
                self,
                "未保存的修改",
                f"图片 {name} 有未保存的修改，是否保存？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if choice == QMessageBox.Cancel:
                event.ignore()
                return
            if choice == QMessageBox.Save:
                if not self._output_dir:
                    QMessageBox.warning(
                        self, "无法保存",
                        "尚未设置输出目录，请先设置后再关闭。"
                    )
                    event.ignore()
                    return
                self._save_current(silent=True)
        # 最后再 flush 一次 cache（确保 last_image_path 落盘）
        if self._progress_cache_path:
            self._converter.save_progress_cache(self._progress_cache_path, self._cache)
        super().closeEvent(event)
