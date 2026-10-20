# src/agent_tools.py

DEVICE_PREDICTION_TOOL = {
    "type": "function",
    "function": {
        "name": "predict_device_failure",
        "description": (
            "根据工业设备的运行参数，调用故障预测模型，"
            "返回设备故障风险分数及预测类别。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["L", "M", "H"],
                    "description": "设备产品类型"
                },
                "air_temperature": {
                    "type": "number",
                    "description": "空气温度，单位 K"
                },
                "process_temperature": {
                    "type": "number",
                    "description": "过程温度，单位 K"
                },
                "rotational_speed": {
                    "type": "number",
                    "description": "转速，单位 rpm"
                },
                "torque": {
                    "type": "number",
                    "description": "扭矩，单位 Nm"
                },
                "tool_wear": {
                    "type": "number",
                    "description": "刀具磨损时间，单位 min"
                }
            },
            "required": [
                "type",
                "air_temperature",
                "process_temperature",
                "rotational_speed",
                "torque",
                "tool_wear"
            ],
            "additionalProperties": False
        }
    }
}
