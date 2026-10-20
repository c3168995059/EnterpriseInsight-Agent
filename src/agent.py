# src/agent.py
import logging
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from agent_tools import DEVICE_PREDICTION_TOOL
from tool_executor import execute_tool

logger = logging.getLogger(__name__)

# 1. 加载环境变量
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

def extract_complete_answer(response) -> str:
    """提取模型回答，并检查是否因长度限制而截断。"""

    choice = response.choices[0]

    if choice.finish_reason == "length":
        raise RuntimeError(
            "DeepSeek 回答达到 Token 上限，内容可能不完整"
        )

    return choice.message.content or ""


def run_agent(
    user_message: str,
    history: list[dict] | None = None
) -> str:
    """接收用户问题，调用预测工具并返回 Agent 回答。"""

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")
    model_name = os.getenv("DEEPSEEK_MODEL")

    if not all([api_key, base_url, model_name]):
        raise ValueError("DeepSeek API 配置不完整")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=30.0,
        max_retries=0
    )

    messages = [
        {
            "role": "system",
            "content": (
                "你是一名工业设备智能分析助手。"
                "当用户询问设备故障风险时，应调用预测工具。"
                "不得自行编造故障概率、设备阈值或故障原因。"
                "工具返回的 failure_probability 是模型风险分数，"
                "尚未验证概率校准，不代表真实故障发生概率。"
                "prediction=0 表示模型判为正常，"
                "prediction=1 表示模型判为故障。"
                "不要在缺乏依据时划分高低风险等级。"
                "如果缺少设备参数，应向用户询问，"
                "不得自行猜测。"
                "预测结果仅供辅助判断，不能替代专业诊断。"
                "如果工具返回 status=failed，"
                "必须明确说明预测未成功。"
                "如果 error_type=invalid_arguments，"
                "应根据用户原始输入核实参数。"
                "如果用户已经提供相关参数，"
                "不得直接断言用户遗漏了这些参数。"
                "不得编造预测结果。"
            )
        },
    ]
    # 加入之前的对话历史
    if history is not None:
        for item in history:
            if (
                isinstance(item, dict)
                and item.get("role") in ("user", "assistant")
                and isinstance(item.get("content"), str)
            ):
                messages.append({
                    "role": item["role"],
                    "content": item["content"]
                })

    # 最后加入当前用户问题
    messages.append({
        "role": "user",
        "content": user_message
    })

    # 2. 第一次调用：由 DeepSeek 决定是否使用工具
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=[DEVICE_PREDICTION_TOOL],
        tool_choice="auto",
        max_tokens=1500
    )

    assistant_message = response.choices[0].message
    logger.debug(
        "First DeepSeek response finish_reason=%s",
        response.choices[0].finish_reason
    )

        # 3. 如果不需要工具，直接返回模型回答
    if not assistant_message.tool_calls:
        return extract_complete_answer(response)

    # 4. 保存模型发出的工具调用请求
    messages.append(
        assistant_message.model_dump(exclude_none=True)
    )

    # 5. 执行模型请求的工具
    for tool_call in assistant_message.tool_calls:

        tool_name = tool_call.function.name

        # 解析 DeepSeek 返回的 JSON 参数
        try:
            arguments = json.loads(
                tool_call.function.arguments
            )

        except json.JSONDecodeError as e:
            result = {
                "status": "failed",
                "error_type": "invalid_json",
                "error": str(e)
            }

        else:
            # 使用统一工具执行入口
            result = execute_tool(
                tool_name,
                arguments
            )

        # 将执行结果添加到对话历史
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(
                result,
                ensure_ascii=False
            )
        })


    # 6. 第二次调用：根据工具结果生成最终回答
    final_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=1500
    )

    logger.debug(
    "Final DeepSeek response finish_reason=%s",
    final_response.choices[0].finish_reason
)

    return extract_complete_answer(final_response)
