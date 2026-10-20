import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

def main():
    # 1. 加载项目根目录下的 .env
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(PROJECT_ROOT / ".env")

    # 2. 读取 DeepSeek 配置
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")
    model_name = os.getenv("DEEPSEEK_MODEL")

    # 3. 检查必要配置
    if not all([api_key, base_url, model_name]):
        raise ValueError("DeepSeek 配置不完整，请检查 .env 文件")

    # 4. 创建 API 客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=30.0,
        max_retries=0
    )

    # 5. 发送第一条大模型请求
    print("正在调用 DeepSeek API...")

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "你是一名工业设备智能分析助手。"
            },
            {
                "role": "user",
                "content": "请用一句话介绍什么是设备预测性维护。"
            }
        ],
        max_tokens=100
    )

    # 6. 获取大模型回答
    answer = response.choices[0].message.content

    print("\n===== DeepSeek 回答 =====")
    print(answer)

    print("\n===== 响应结构检查 =====")

    print("模型名称：", response.model)

    print("结束原因：", response.choices[0].finish_reason)

    print("输入 Token：", response.usage.prompt_tokens)

    print("输出 Token：", response.usage.completion_tokens)

    print("总 Token：", response.usage.total_tokens)

    print(
        "工具调用信息：",
        response.choices[0].message.tool_calls
    )


if __name__ == "__main__":
    main()
