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

@pytest.mark.integration
def test_batch_prediction_api():
    """验证真实 FastAPI 批量预测接口。"""

    import json
    from urllib.request import Request, urlopen

    devices = [
        {
            "type": "L",
            "air_temperature": 300.0,
            "process_temperature": 310.0,
            "rotational_speed": 1500.0,
            "torque": 40.0,
            "tool_wear": 100.0,
        },
        {
            "type": "M",
            "air_temperature": 302.0,
            "process_temperature": 312.0,
            "rotational_speed": 1400.0,
            "torque": 55.0,
            "tool_wear": 180.0,
        },
    ]

    body = json.dumps(devices).encode("utf-8")

    request = Request(
        "http://127.0.0.1:8000/predict/batch",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=10) as response:
        assert response.status == 200
        result = json.load(response)

    assert result["total_devices"] == 2
    assert len(result["results"]) == 2

    for index, prediction in enumerate(result["results"]):
        assert prediction["device_index"] == index
        assert prediction["prediction"] in (0, 1)
        assert 0.0 <= prediction["failure_probability"] <= 1.0

@pytest.mark.integration
def test_invalid_device_type():
    """验证非法设备类型会被 API 拒绝。"""

    import json
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError

    device = {
        "type": "X",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500.0,
        "torque": 40.0,
        "tool_wear": 100.0,
    }

    request = Request(
        "http://127.0.0.1:8000/predict",
        data=json.dumps(device).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with pytest.raises(HTTPError) as exc_info:
        urlopen(request, timeout=10)

    assert exc_info.value.code == 422

@pytest.mark.integration
def test_batch_empty_list():
    """验证批量预测接口拒绝空设备列表。"""

    from urllib.request import Request, urlopen
    from urllib.error import HTTPError

    request = Request(
        "http://127.0.0.1:8000/predict/batch",
        data=b"[]",
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with pytest.raises(HTTPError) as exc_info:
        urlopen(request, timeout=10)

    assert exc_info.value.code == 422

@pytest.mark.integration
def test_batch_exceeds_limit():
    """验证批量预测接口拒绝超过100台设备。"""

    import json
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError

    device = {
        "type": "L",
        "air_temperature": 300.0,
        "process_temperature": 310.0,
        "rotational_speed": 1500.0,
        "torque": 40.0,
        "tool_wear": 100.0,
    }

    devices = [device.copy() for _ in range(101)]

    request = Request(
        "http://127.0.0.1:8000/predict/batch",
        data=json.dumps(devices).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with pytest.raises(HTTPError) as exc_info:
        urlopen(request, timeout=10)

    assert exc_info.value.code == 422