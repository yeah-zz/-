"""Stable paths independent of VS Code's current working directory.

绘画/ 现在是一个独立项目：本目录里就有 pyproject.toml 和 .venv，
所以项目根目录就是本文件所在目录。如果这个文件夹被放回旧的两层结构
（只有外层才有 pyproject.toml），会自动回退到上一层作为项目根。
"""
from pathlib import Path

PAINTING_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PAINTING_DIR if (PAINTING_DIR / 'pyproject.toml').is_file() else PAINTING_DIR.parent
REFERENCE_IMAGE = PAINTING_DIR / '参考图' / 'watercolor_landscape_reference.png'
PLUM_REFERENCE_IMAGE = PAINTING_DIR / '参考图' / 'plum_blossom_robot_ready.png'
OUTPUT_DIR = PAINTING_DIR / '输出'
# Keep one shared history/lock directory for old and new robot executors.
ROBOT_LOG_DIR = PROJECT_DIR / 'robot_runs'
