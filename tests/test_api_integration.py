# tests/test_api_integration.py

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tool_executor import execute_tool


@pytest.mark.integration
def test_real_prediction_api():
    """测试真实 FastAPI 预测接口。"""

    device_data = {
        "type": "L",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500,
        "torque": 50.0,
        "tool_wear": 150
    }

    result = execute_tool(
        "predict_device_failure",
        device_data
    )

    assert result["status"] == "success"

    prediction = result["data"]

    assert prediction["prediction"] == 0

    assert abs(
        prediction["failure_probability"]
        - 0.12271667996754301
    ) < 1e-6
