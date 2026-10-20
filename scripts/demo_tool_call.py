# scripts/demo_tool_call.py

import os
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from agent_tools import DEVICE_PREDICTION_TOOL
from tool_executor import predict_device_failure


def main():
    # 1. 加载环境变量
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")
    model_name = os.getenv("DEEPSEEK_MODEL")

    if not all([api_key, base_url, model_name]):
        raise ValueError("DeepSeek 配置不完整")

    # 2. 创建 DeepSeek 客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=30.0,
        max_retries=0
    )

    # 3. 模拟用户提问
    user_message = (
        "帮我分析一台 L 类设备，"
        "空气温度 300K，过程温度 310K，"
        "转速 1500rpm，扭矩 50Nm，"
        "刀具磨损 150min，它有没有故障风险？"
    )

    # 4. 请求大模型选择工具
    print("正在请求 DeepSeek 进行工具调用...")

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是工业设备智能分析助手。"
                    "当用户请求分析设备故障风险时，"
                    "应调用设备故障预测工具，不要自行编造预测结果。"
                )
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        tools=[DEVICE_PREDICTION_TOOL],
        tool_choice="auto",
        max_tokens=300
    )

    # 5. 查看大模型是否请求调用工具
    message = response.choices[0].message

    print("\n===== Tool Calling 检查 =====")
    print("结束原因：", response.choices[0].finish_reason)

    if message.tool_calls:
        for tool_call in message.tool_calls:
            print("\n工具名称：", tool_call.function.name)

            arguments = json.loads(tool_call.function.arguments)

            print("工具参数：")
            print(
                json.dumps(
                    arguments,
                    ensure_ascii=False,
                    indent=2
                )
            )
            # 6. 执行工具：调用本地 FastAPI 预测接口
            # 6. 根据工具名称执行对应功能
            if tool_call.function.name == "predict_device_failure":
                prediction_result = predict_device_failure(arguments)
            else:
                raise ValueError(
                    f"未知工具：{tool_call.function.name}"
                )
            # 7. 将工具调用及其结果加入对话历史
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是一名工业设备智能分析助手。"
                        "你的任务是根据工具返回的预测结果，"
                        "生成准确、简洁、可追溯的分析报告。"

                        "必须遵守以下规则："
                        "1. 预测分数必须来自工具结果，不得自行编造。"
                        "2. prediction=0 表示模型判为正常，"
                        "prediction=1 表示模型判为故障。"
                        "3. failure_probability 是模型输出的风险分数，"
                        "尚未验证概率校准，不能直接当作真实故障发生概率。"
                        "4. 不得自行推断设备安全阈值、磨损上限、"
                        "正常温差范围或故障原因。"
                        "5. 如果缺少设备规格、历史数据或维护标准，"
                        "应明确说明信息不足。"
                        "6. 模型预测仅供辅助判断，"
                        "不能代替实际设备检测与专业诊断。"

                        "回答应包含：预测结果、结果解释、注意事项。"
                    )
                },
                {
                    "role": "user",
                    "content": user_message
                },
                message.model_dump(exclude_none=True),
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        prediction_result,
                        ensure_ascii=False
                    )
                }
            ]

            # 8. 再次调用 DeepSeek，生成最终回答
            print("\n正在生成设备分析报告...")

            final_response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=400
            )

            print("\n===== Agent 最终回答 =====")
            print(final_response.choices[0].message.content)


    else:
        print("本次没有生成工具调用")
        print("模型回答：", message.content)


if __name__ == "__main__":
    main()
