
# src/logger_config.py

import logging


def setup_logging(level: str = "INFO"):
    """统一配置项目日志。"""

    # 将字符串转换为日志级别
    log_level = getattr(
        logging,
        level.upper(),
        logging.INFO
    )

    # 配置日志格式及输出级别
    logging.basicConfig(
        level=log_level,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S"
    )
