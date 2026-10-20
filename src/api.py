from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel, Field
from typing import Annotated
from typing import Literal

from predict import predict_failure, predict_batch

# 创建 FastAPI 应用
app = FastAPI(
    title="EnterpriseInsight-Agent",
    description="企业数据智能分析与设备故障预测系统",
    version="0.1.0"
)


@app.get("/health")
def health_check():
    """检查 API 服务是否正常运行。"""

    return {
        "status": "ok",
        "message": "EnterpriseInsight-Agent API is running"
    }


# 定义请求数据结构
class DeviceInput(BaseModel):
    type: Literal["L", "M", "H"]
    air_temperature: float = Field(..., ge=0)
    process_temperature: float = Field(..., ge=0)
    rotational_speed: float = Field(..., ge=0)
    torque: float = Field(..., ge=0)
    tool_wear: float = Field(..., ge=0)

def device_to_model_input(device: DeviceInput):
    """将 API 输入转换为模型要求的字段格式。"""

    return {
        "Type": device.type,
        "Air temperature [K]": device.air_temperature,
        "Process temperature [K]": device.process_temperature,
        "Rotational speed [rpm]": device.rotational_speed,
        "Torque [Nm]": device.torque,
        "Tool wear [min]": device.tool_wear
    }


@app.post("/predict")
def predict_device(device: DeviceInput):
    """接收设备数据并返回故障预测。"""

    # 将 API 字段转换为模型使用的列名
    device_data = device_to_model_input(device)

    try:
        result = predict_failure(device_data)
        return result

    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )


# ===== 第五十二步：批量预测接口 =====

@app.post("/predict/batch")
def predict_devices(
    devices: Annotated[
        list[DeviceInput],
        Field(min_length=1, max_length=100)
    ]
):
    """批量预测设备故障风险。"""

    # 将 API 输入转换为模型所需的字段
    device_list = [
    device_to_model_input(device)
    for device in devices
]

    try:
        results = predict_batch(device_list)

        return {
            "total_devices": len(results),
            "results": results
        }

    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )
