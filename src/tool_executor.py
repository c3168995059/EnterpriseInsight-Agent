# src/tool_executor.py
import socket
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

def predict_device_failure(arguments: dict) -> dict:
    """
    调用本地 FastAPI 接口进行设备故障预测。

    参数：
        arguments: 包含六项设备参数的字典

    返回：
        FastAPI 返回的预测结果字典
    """
    # 检查输入是否为字典
    if not isinstance(arguments, dict):
        raise TypeError(
            "工具参数格式错误：必须提供字典类型的数据"
        )
     # 检查工具调用参数是否完整
    required_fields = [
        "type",
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear"
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in arguments or arguments[field] is None
    ]

    if missing_fields:
        raise ValueError(
            f"设备参数不完整，缺少：{', '.join(missing_fields)}"
        )

    request_data = json.dumps(arguments).encode("utf-8")

    request = Request(
        url="http://127.0.0.1:8000/predict",
        data=request_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urlopen(request, timeout=10) as response:
        result = json.loads(
            response.read().decode("utf-8")
        )

    return result

def execute_tool(tool_name: str, arguments: dict) -> dict:
    """根据工具名称执行工具，并返回统一格式的结果。"""

    try:
        if tool_name == "predict_device_failure":
            result = predict_device_failure(arguments)
        else:
            return {
                "status": "failed",
                "error_type": "unknown_tool",
                "error": f"不支持的工具：{tool_name}"
            }

        return {
            "status": "success",
            "data": result
        }

    except TypeError as e:
        return {
            "status": "failed",
            "error_type": "invalid_type",
            "error": str(e)
        }

    except ValueError as e:
        return {
            "status": "failed",
            "error_type": "invalid_arguments",
            "error": str(e)
        }

    except HTTPError as e:
        if e.code == 422:
            return {
                "status": "failed",
                "error_type": "invalid_arguments",
                "error": "预测服务拒绝了输入参数，请检查字段类型和取值。"
            }

        return {
            "status": "failed",
            "error_type": "http_error",
            "error": f"预测服务返回 HTTP {e.code} 错误。"
        }

    except TimeoutError:
        return {
            "status": "failed",
            "error_type": "timeout",
            "error": "设备预测服务请求超时，请稍后重试。"
        }

    except URLError as e:

        # 检查底层原因是否为超时
        if isinstance(e.reason, TimeoutError):
            return {
                "status": "failed",
                "error_type": "timeout",
                "error": "设备预测服务请求超时，请稍后重试。"
            }

        # 其他 URL 连接异常
        return {
            "status": "failed",
            "error_type": "service_unavailable",
            "error": (
                "设备预测服务连接失败，"
                "请检查 FastAPI 服务是否正常运行。"
            )
        }
