"""文件列表工具（供 word_seg 等脚本使用）。"""
from glob import glob
from pathlib import Path
from typing import List, Union


def get_files_list(folder: Union[str, Path], postfix: str = "*.txt") -> List[str]:
    """返回目录下匹配 glob 后缀的文件路径列表（排序后，绝对路径字符串）。"""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        return []
    pattern = str(root / postfix)
    return sorted(glob(pattern))
