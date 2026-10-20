import joblib
import pandas as pd
from pathlib import Path
from functools import lru_cache

# 1. 获取模型文件路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression.joblib"


@lru_cache(maxsize=1)
def load_model():
    """首次调用时加载模型，后续复用缓存。"""
    print("正在从磁盘加载模型...")
    return joblib.load(MODEL_PATH)


# 模型要求的6个输入特征
REQUIRED_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

NUMERICAL_FEATURES = REQUIRED_FEATURES[1:]


def validate_device_data(device_data):
    """校验单条设备数据。"""

    # 1. 检查输入类型
    if not isinstance(device_data, dict):
        raise TypeError("设备数据必须是字典")

    # 2. 检查缺失字段
    missing = [
        feature for feature in REQUIRED_FEATURES
        if feature not in device_data
    ]

    if missing:
        raise ValueError(f"缺少必要字段：{missing}")

    # 3. 检查产品类型
    if device_data["Type"] not in ["L", "M", "H"]:
        raise ValueError("Type 必须是 L、M 或 H")

    # 4. 检查数值特征
    import math
    import numbers

    for feature in NUMERICAL_FEATURES:
        value = device_data[feature]

        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(value)
        ):
            raise ValueError(
                f"{feature} 必须是有限的数值"
            )

    return True


def predict_failure(device_data):
    """根据设备特征预测故障风险。"""

   # 先检查输入
    validate_device_data(device_data)


    # 使用缓存的模型
    model = load_model()

    input_df = pd.DataFrame([device_data])

    failure_probability = model.predict_proba(input_df)[0, 1]
    prediction = int(failure_probability >= 0.5)

    return {
        "failure_probability": float(failure_probability),
        "prediction": prediction
    }

def predict_batch(devices):
    """批量预测设备故障风险。"""

    # 1. 检查批量输入
    if not isinstance(devices, list):
        raise TypeError("批量设备数据必须是列表")

    if len(devices) == 0:
        raise ValueError("设备列表不能为空")

    # 2. 逐条校验设备数据
    for device in devices:
        validate_device_data(device)

    # 3. 加载缓存中的模型
    model = load_model()

    # 4. 将设备列表转换为 DataFrame
    input_df = pd.DataFrame(devices)

    # 5. 批量预测故障概率
    probabilities = model.predict_proba(input_df)[:, 1]

    # 6. 使用默认阈值生成预测标签
    predictions = (probabilities >= 0.5).astype(int)

    # 7. 组织返回结果
    results = []

    for i, (prob, pred) in enumerate(
        zip(probabilities, predictions)
    ):
        results.append({
            "device_index": i,
            "failure_probability": float(prob),
            "prediction": int(pred)
        })

    return results


if __name__ == "__main__":

    # 模拟一条设备运行记录
    device = {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 50.0,
        "Tool wear [min]": 150
    }


    print("\n===== 第一次预测 =====")
    result1 = predict_failure(device)
    print(result1)

    print("\n===== 第二次预测 =====")
    result2 = predict_failure(device)
    print(result2)

    print("\n===== 错误输入测试 =====")

    invalid_device = {
        "Type": "L",
        "Torque [Nm]": 50.0
    }

    try:
        result = predict_failure(invalid_device)
        print(result)

    except (ValueError, TypeError) as e:
        print("输入校验失败：", e)

    print("\n===== 批量预测测试 =====")

    devices = [
        {
            "Type": "L",
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 50.0,
            "Tool wear [min]": 150
        },
        {
            "Type": "M",
            "Air temperature [K]": 298.0,
            "Process temperature [K]": 308.0,
            "Rotational speed [rpm]": 1600,
            "Torque [Nm]": 35.0,
            "Tool wear [min]": 60
        },
        {
            "Type": "H",
            "Air temperature [K]": 303.0,
            "Process temperature [K]": 313.0,
            "Rotational speed [rpm]": 1200,
            "Torque [Nm]": 65.0,
            "Tool wear [min]": 220
        }
    ]

    batch_results = predict_batch(devices)

    for result in batch_results:
        print(result)