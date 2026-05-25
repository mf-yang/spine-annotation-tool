"""清空标注数据确认对话框（不可恢复）。

设计要点：
1. 启动时扫描并展示将要删除的内容统计
2. 默认勾选「缓存」+「当前 split 标注」
3. 必须输入 CONFIRM 字符串才会激活红色「确定清空」按钮
4. 直接物理删除，不备份
"""

from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
    QLineEdit, QTextBrowser, QVBoxLayout,
)


CONFIRM_TOKEN = "CONFIRM"


class ClearDataDialog(QDialog):
    """清空标注数据确认对话框。

    使用方式：
        dlg = ClearDataDialog(parent, cache_path, output_dir, split, label_files)
        if dlg.exec_() == QDialog.Accepted:
            opts = dlg.get_options()
            # opts: {'clear_cache': bool, 'clear_labels': bool}
    """

    def __init__(
        self,
        parent,
        cache_path: Optional[str],
        output_dir: Optional[str],
        split: str,
        label_files: List[Path],
    ):
        super().__init__(parent)
        self.setWindowTitle("清空标注数据")
        self.setModal(True)
        self.resize(560, 460)

        self._cache_path = cache_path
        self._output_dir = output_dir
        self._split = split
        self._label_files = label_files

        layout = QVBoxLayout(self)

        # 顶部警告
        warn = QLabel(
            "⚠️ 此操作将<b>物理删除</b>下列文件，<span style='color:#d9534f'><b>不可恢复</b></span>。"
        )
        warn.setTextFormat(Qt.RichText)
        warn.setWordWrap(True)
        layout.addWidget(warn)

        # 信息区
        info = QTextBrowser()
        info.setOpenExternalLinks(False)
        info.setHtml(self._build_info_html())
        info.setMaximumHeight(180)
        layout.addWidget(info)

        # 选项
        cache_exists = bool(cache_path) and Path(cache_path).exists() if cache_path else False
        self._chk_cache = QCheckBox(
            f"清空标注进度缓存  (.annotate_progress.json)  "
            f"{'[存在]' if cache_exists else '[不存在]'}"
        )
        self._chk_cache.setChecked(cache_exists)
        self._chk_cache.setEnabled(cache_exists)
        layout.addWidget(self._chk_cache)

        labels_count = len(label_files)
        split_label = split if split else "（根目录）"
        self._chk_labels = QCheckBox(
            f"清空训练标注文件  ({split_label})  共 {labels_count} 个 *.txt"
        )
        self._chk_labels.setChecked(labels_count > 0)
        self._chk_labels.setEnabled(labels_count > 0)
        layout.addWidget(self._chk_labels)

        layout.addSpacing(8)

        # 二次确认输入
        confirm_row = QHBoxLayout()
        confirm_row.addWidget(QLabel(f"请输入 <b>{CONFIRM_TOKEN}</b> 以激活清空按钮："))
        self._input_confirm = QLineEdit()
        self._input_confirm.setPlaceholderText(CONFIRM_TOKEN)
        self._input_confirm.textChanged.connect(self._update_confirm_button)
        confirm_row.addWidget(self._input_confirm, 1)
        layout.addLayout(confirm_row)

        # 按钮
        self._btns = QDialogButtonBox()
        self._btn_ok = self._btns.addButton("确定清空", QDialogButtonBox.AcceptRole)
        self._btn_cancel = self._btns.addButton("取消", QDialogButtonBox.RejectRole)
        self._btn_ok.setStyleSheet(
            "QPushButton { background-color: #d9534f; color: white; padding: 6px 14px; }"
            "QPushButton:disabled { background-color: #f0a8a5; color: #f3f3f3; }"
        )
        self._btn_ok.setEnabled(False)
        self._btns.accepted.connect(self.accept)
        self._btns.rejected.connect(self.reject)
        layout.addWidget(self._btns)

        # 初始触发一次
        self._chk_cache.toggled.connect(self._update_confirm_button)
        self._chk_labels.toggled.connect(self._update_confirm_button)

    def _build_info_html(self) -> str:
        cache_path = self._cache_path or "(未设置)"
        cache_exists = bool(self._cache_path) and Path(self._cache_path).exists() if self._cache_path else False

        out_dir = self._output_dir or "(未设置)"
        split_label = self._split if self._split else "（根目录）"

        labels_count = len(self._label_files)
        if labels_count == 0:
            files_html = "<span style='color:#888'>无文件</span>"
        else:
            preview = self._label_files[:5]
            tail = "..." if labels_count > 5 else ""
            files_html = "<br>".join(f"&nbsp;&nbsp;{p.name}" for p in preview) + (
                f"<br>&nbsp;&nbsp;{tail}" if tail else ""
            )

        return (
            "<b>进度缓存</b><br>"
            f"&nbsp;&nbsp;路径：{cache_path}<br>"
            f"&nbsp;&nbsp;状态：{'存在' if cache_exists else '不存在'}<br><br>"
            "<b>训练标注</b><br>"
            f"&nbsp;&nbsp;输出目录：{out_dir}<br>"
            f"&nbsp;&nbsp;Split：{split_label}<br>"
            f"&nbsp;&nbsp;待删除文件数：{labels_count}<br>"
            f"{files_html}"
        )

    def _update_confirm_button(self):
        token_ok = self._input_confirm.text().strip() == CONFIRM_TOKEN
        any_checked = self._chk_cache.isChecked() or self._chk_labels.isChecked()
        self._btn_ok.setEnabled(token_ok and any_checked)

    def get_options(self) -> dict:
        return {
            "clear_cache": self._chk_cache.isChecked() and self._chk_cache.isEnabled(),
            "clear_labels": self._chk_labels.isChecked() and self._chk_labels.isEnabled(),
        }
