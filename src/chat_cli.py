
# src/chat_cli.py
import argparse
from conversation import ConversationManager, chat
from logger_config import setup_logging

def main():
    """运行命令行多轮对话程序。"""

    # 1. 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="EnterpriseInsight-Agent 命令行助手"
    )

    # 2. 添加日志级别参数
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="设置日志输出级别，默认 INFO"
    )

    # 3. 解析参数并初始化日志
    args = parser.parse_args()
    setup_logging(args.log_level)

    # 4. 创建会话管理器
    conversation = ConversationManager()


    print("===== EnterpriseInsight-Agent =====")
    print("工业设备智能分析助手")
    print("输入 exit 退出，输入 clear 清空历史。")

    while True:
        user_message = input("\n用户：").strip()

        # 1. 退出程序
        if user_message.lower() == "exit":
            print("聊天已结束。")
            break

        # 2. 清空历史
        if user_message.lower() == "clear":
            conversation.clear_history()
            print("历史记录已清空。")
            continue

        # 3. 忽略空输入
        if not user_message:
            print("请输入问题。")
            continue

        # 4. 调用 Agent
        try:
            answer = chat(conversation, user_message)
            print("\nAgent：", answer)

            print(
                f"\n[当前历史消息："
                f"{len(conversation.get_history())} 条]"
            )

        except Exception as e:
            print(f"\nAgent 调用失败：{type(e).__name__}")
            print("本轮对话未保存，请检查服务或网络。")


if __name__ == "__main__":
    main()
