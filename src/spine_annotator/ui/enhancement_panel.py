"""画布顶部的图像增强工具条与调参弹窗。

UI 设计：
- 工具条：一排按钮 + 状态指示，常驻于画布上方
  [对比度/亮度] [伽马] [CLAHE] [反相 ☐] [重置]
- 弹窗：点击按钮后弹出对应参数滑块，实时预览
- 反相为开关型按钮，直接切换不弹窗
- 重置一键归位

参数变更通过 params_changed 信号广播，由 main_window 监听并同步到画布与缓存。
"""
from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.image_enhancer import EnhanceParams


class _SliderRow(QWidget):
    """一行：标签 + 滑块 + 数值框，实时同步。"""

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        step: float,
        value: float,
        decimals: int = 0,
        parent: QWidget = None,
    ):
        super().__init__(parent)
        self._decimals = decimals
        self._step = step
        self._scale = 10 ** decimals  # 滑块只接受 int，所以乘以 scale 再转

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel(label)
        self._label.setMinimumWidth(80)
        layout.addWidget(self._label)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(int(minimum * self._scale))
        self._slider.setMaximum(int(maximum * self._scale))
        self._slider.setSingleStep(max(1, int(step * self._scale)))
        self._slider.setValue(int(value * self._scale))
        layout.addWidget(self._slider, 1)

        if decimals == 0:
            self._spin = QSpinBox()
            self._spin.setRange(int(minimum), int(maximum))
            self._spin.setValue(int(value))
        else:
            self._spin = QDoubleSpinBox()
            self._spin.setDecimals(decimals)
            self._spin.setRange(minimum, maximum)
            self._spin.setSingleStep(step)
            self._spin.setValue(value)
        self._spin.setFixedWidth(70)
        layout.addWidget(self._spin)

        self._slider.valueChanged.connect(self._on_slider)
        self._spin.valueChanged.connect(self._on_spin)

    def _on_slider(self, v: int):
        real = v / self._scale
        self._spin.blockSignals(True)
        # QSpinBox 只接受 int，QDoubleSpinBox 接受 float
        if self._decimals == 0:
            self._spin.setValue(int(real))
        else:
            self._spin.setValue(float(real))
        self._spin.blockSignals(False)
        self.valueChanged.emit(float(real))

    def _on_spin(self, v):
        self._slider.blockSignals(True)
        self._slider.setValue(int(float(v) * self._scale))
        self._slider.blockSignals(False)
        self.valueChanged.emit(float(v))

    def value(self) -> float:
        return float(self._spin.value())

    def setValue(self, v: float):
        if self._decimals == 0:
            self._spin.setValue(int(v))
        else:
            self._spin.setValue(float(v))


class EnhancementDialog(QDialog):
    """图像增强参数调整弹窗（实时预览）。"""

    params_changed = pyqtSignal(EnhanceParams)

    def __init__(self, params: EnhanceParams, parent: QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("图像增强")
        self.setModal(False)  # 非模态，方便边调边看
        self.setMinimumWidth(420)

        self._params = EnhanceParams(**params.to_dict())

        layout = QVBoxLayout(self)

        form = QFormLayout()
        form.setContentsMargins(8, 8, 8, 8)
        form.setLabelAlignment(Qt.AlignRight)

        # 亮度
        self._row_brightness = _SliderRow("亮度", -100, 100, 1, self._params.brightness)
        self._row_brightness.valueChanged.connect(
            lambda v: self._update("brightness", int(v))
        )
        form.addRow(self._row_brightness)

        # 对比度
        self._row_contrast = _SliderRow("对比度", -100, 100, 1, self._params.contrast)
        self._row_contrast.valueChanged.connect(
            lambda v: self._update("contrast", int(v))
        )
        form.addRow(self._row_contrast)

        # 伽马
        self._row_gamma = _SliderRow("伽马", 0.1, 3.0, 0.05, self._params.gamma, decimals=2)
        self._row_gamma.valueChanged.connect(
            lambda v: self._update("gamma", float(v))
        )
        form.addRow(self._row_gamma)

        # CLAHE
        self._row_clahe = _SliderRow("CLAHE", 0.0, 10.0, 0.5, self._params.clahe, decimals=1)
        self._row_clahe.valueChanged.connect(
            lambda v: self._update("clahe", float(v))
        )
        form.addRow(self._row_clahe)

        # 反相
        self._chk_invert = QCheckBox("反相（黑白互换）")
        self._chk_invert.setChecked(self._params.invert)
        self._chk_invert.toggled.connect(lambda v: self._update("invert", bool(v)))
        form.addRow(self._chk_invert)

        layout.addLayout(form)

        # 按钮区
        btn_layout = QHBoxLayout()
        self._btn_reset = QPushButton("重置默认")
        self._btn_reset.clicked.connect(self._reset)
        btn_layout.addWidget(self._btn_reset)
        btn_layout.addStretch(1)

        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.close)
        btn_layout.addWidget(bb)

        layout.addLayout(btn_layout)

        # 提示
        hint = QLabel(
            "提示：\n"
            "- CLAHE 对 X 光片细节增强最明显（建议 2.0~4.0）\n"
            "- 伽马 < 1 提亮暗部，> 1 压暗\n"
            "- 调整不影响原图与标注坐标，仅辅助显示"
        )
        hint.setStyleSheet("color: #666; font-size: 11px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

    def _update(self, field: str, value):
        setattr(self._params, field, value)
        self.params_changed.emit(EnhanceParams(**self._params.to_dict()))

    def _reset(self):
        self._params = EnhanceParams()
        self._row_brightness.setValue(self._params.brightness)
        self._row_contrast.setValue(self._params.contrast)
        self._row_gamma.setValue(self._params.gamma)
        self._row_clahe.setValue(self._params.clahe)
        self._chk_invert.blockSignals(True)
        self._chk_invert.setChecked(False)
        self._chk_invert.blockSignals(False)
        self.params_changed.emit(EnhanceParams(**self._params.to_dict()))


class EnhancementToolbar(QFrame):
    """画布顶部的图像增强工具条。

    布局：
        [⚙ 调整图像] [✨ 反相]            [↺ 重置]   增强中: ✓
    """

    open_dialog_requested = pyqtSignal()
    invert_toggled = pyqtSignal(bool)
    reset_requested = pyqtSignal()

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet(
            "EnhancementToolbar { background: #f5f5f5; border-bottom: 1px solid #ddd; }"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        self._btn_adjust = QToolButton()
        self._btn_adjust.setText("⚙ 调整图像")
        self._btn_adjust.setToolTip("打开调参弹窗：亮度 / 对比度 / 伽马 / CLAHE")
        self._btn_adjust.clicked.connect(self.open_dialog_requested.emit)
        layout.addWidget(self._btn_adjust)

        self._btn_invert = QToolButton()
        self._btn_invert.setText("✨ 反相")
        self._btn_invert.setCheckable(True)
        self._btn_invert.setToolTip("黑白互换（部分骨密度低的片子反相后更清晰）")
        self._btn_invert.toggled.connect(self.invert_toggled.emit)
        layout.addWidget(self._btn_invert)

        self._btn_reset = QToolButton()
        self._btn_reset.setText("↺ 重置")
        self._btn_reset.setToolTip("还原所有图像增强参数")
        self._btn_reset.clicked.connect(self.reset_requested.emit)
        layout.addWidget(self._btn_reset)

        layout.addStretch(1)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #2563eb; font-size: 12px;")
        layout.addWidget(self._status_label)

    def sync_from_params(self, params: EnhanceParams):
        """根据参数同步按钮状态与状态指示。"""
        self._btn_invert.blockSignals(True)
        self._btn_invert.setChecked(params.invert)
        self._btn_invert.blockSignals(False)

        if params.is_identity():
            self._status_label.setText("")
        else:
            tags = []
            if params.brightness != 0:
                tags.append(f"亮度{params.brightness:+d}")
            if params.contrast != 0:
                tags.append(f"对比{params.contrast:+d}")
            if abs(params.gamma - 1.0) > 1e-6:
                tags.append(f"γ={params.gamma:.2f}")
            if params.clahe > 0:
                tags.append(f"CLAHE={params.clahe:.1f}")
            if params.invert:
                tags.append("反相")
            self._status_label.setText("增强中: " + " / ".join(tags))
