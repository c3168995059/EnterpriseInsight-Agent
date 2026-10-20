
# tests/test_tool_executor.py

import sys
from pathlib import Path

# 将 src 目录加入 Python 模块搜索路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tool_executor import execute_tool


def test_unknown_tool():
    """测试未知工具是否被正确处理。"""

    result = execute_tool(
        "unknown_function",
        {}
    )

    assert result["status"] == "failed"
    assert result["error_type"] == "unknown_tool"
    assert "unknown_function" in result["error"]



def test_missing_arguments():
    """测试设备参数缺失时的错误分类。"""

    device_data = {
        "type": "L",
        "air_temperature": 300
    }

    result = execute_tool(
        "predict_device_failure",
        device_data
    )

    assert result["status"] == "failed"
    assert result["error_type"] == "invalid_arguments"

    assert "torque" in result["error"]
    assert "tool_wear" in result["error"]


def test_invalid_argument_type():
    """测试工具参数不是字典时的错误分类。"""

    result = execute_tool(
        "predict_device_failure",
        [1, 2, 3]
    )

    assert result["status"] == "failed"
    assert result["error_type"] == "invalid_type"

    assert "字典类型" in result["error"]


def test_http_422():
    """测试 HTTP 422 是否正确分类。"""

    from unittest.mock import patch
    from urllib.error import HTTPError

    device_data = {
        "type": "L",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 50.0,
        "tool_wear": 150
    }

    # 模拟 FastAPI 返回 HTTP 422
    error_422 = HTTPError(
        url="http://127.0.0.1:8000/predict",
        code=422,
        msg="Unprocessable Entity",
        hdrs=None,
        fp=None
    )

    with patch(
        "tool_executor.urlopen",
        side_effect=error_422
    ):
        result = execute_tool(
            "predict_device_failure",
            device_data
        )

    assert result["status"] == "failed"
    assert result["error_type"] == "invalid_arguments"

def test_http_500():
    """测试 HTTP 500 是否正确分类。"""

    from unittest.mock import patch
    from urllib.error import HTTPError

    device_data = {
        "type": "L",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 50.0,
        "tool_wear": 150
    }

    # 模拟 FastAPI 返回 HTTP 500
    error_500 = HTTPError(
        url="http://127.0.0.1:8000/predict",
        code=500,
        msg="Internal Server Error",
        hdrs=None,
        fp=None
    )

    with patch(
        "tool_executor.urlopen",
        side_effect=error_500
    ):
        result = execute_tool(
            "predict_device_failure",
            device_data
        )

    assert result["status"] == "failed"
    assert result["error_type"] == "http_error"
    assert "500" in result["error"]


def test_timeout():
    """测试直接抛出的 TimeoutError。"""

    from unittest.mock import patch

    device_data = {
        "type": "L",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 50.0,
        "tool_wear": 150
    }

    # 模拟 FastAPI 请求超时
    with patch(
        "tool_executor.urlopen",
        side_effect=TimeoutError("Request timed out")
    ):
        result = execute_tool(
            "predict_device_failure",
            device_data
        )

    assert result["status"] == "failed"
    assert result["error_type"] == "timeout"
    assert "请求超时" in result["error"]

def test_wrapped_timeout():
    """测试 URLError 包装的 TimeoutError。"""

    from unittest.mock import patch
    from urllib.error import URLError

    device_data = {
        "type": "L",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 50.0,
        "tool_wear": 150
    }

    # 模拟底层超时被包装为 URLError
    wrapped_error = URLError(
        TimeoutError("timed out")
    )

    with patch(
        "tool_executor.urlopen",
        side_effect=wrapped_error
    ):
        result = execute_tool(
            "predict_device_failure",
            device_data
        )

    assert result["status"] == "failed"
    assert result["error_type"] == "timeout"
    assert "请求超时" in result["error"]


def test_connection_failure():
    """测试普通网络连接失败。"""

    from unittest.mock import patch
    from urllib.error import URLError

    device_data = {
        "type": "L",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 50.0,
        "tool_wear": 150
    }

    # 模拟普通连接失败
    connection_error = URLError(
        "Connection refused"
    )

    with patch(
        "tool_executor.urlopen",
        side_effect=connection_error
    ):
        result = execute_tool(
            "predict_device_failure",
            device_data
        )

    assert result["status"] == "failed"
    assert result["error_type"] == "service_unavailable"
    assert "连接失败" in result["error"]


def test_execute_tool_success():
    """测试工具正常执行预测。"""

    from unittest.mock import patch

    device_data = {
        "type": "L",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 50.0,
        "tool_wear": 150
    }

    # 模拟预测服务返回的真实历史测试结果
    expected_prediction = {
        "failure_probability": 0.12271667996754301,
        "prediction": 0
    }

    # 模拟底层预测函数正常返回
    with patch(
        "tool_executor.predict_device_failure",
        return_value=expected_prediction
    ):
        result = execute_tool(
            "predict_device_failure",
            device_data
        )

    assert result["status"] == "success"
    assert result["data"]["prediction"] == 0

    assert abs(
        result["data"]["failure_probability"]
        - 0.12271667996754301
    ) < 1e-6
