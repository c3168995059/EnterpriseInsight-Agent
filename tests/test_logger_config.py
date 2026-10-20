
# tests/test_logger_config.py

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from logger_config import setup_logging


def test_logging_info_level():
    """测试 INFO 日志级别配置。"""

    # 重置根日志的处理器配置，便于在 pytest 中测试
    original_handlers = logging.root.handlers[:]
    original_level = logging.root.level

    try:
        logging.root.handlers.clear()

        setup_logging("INFO")

        assert logging.root.level == logging.INFO

    finally:
        logging.root.handlers.clear()
        logging.root.handlers.extend(original_handlers)
        logging.root.setLevel(original_level)


def test_logging_debug_level():
    """测试 DEBUG 日志级别配置。"""

    original_handlers = logging.root.handlers[:]
    original_level = logging.root.level

    try:
        logging.root.handlers.clear()

        setup_logging("DEBUG")

        assert logging.root.level == logging.DEBUG

    finally:
        logging.root.handlers.clear()
        logging.root.handlers.extend(original_handlers)
        logging.root.setLevel(original_level)
