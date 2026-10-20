# scripts/demo_auto_conversation.py

from conversation import ConversationManager, chat


def main():
    # 1. 创建会话管理器
    conversation = ConversationManager()


    # 2. 第一轮对话
    question1 = "什么是设备预测性维护？"

    print("===== 第一轮对话 =====")
    print("用户：", question1)

    answer1 = chat(conversation, question1)

    print("Agent：", answer1)
    print("当前历史消息数量：", len(conversation.get_history()))


    # 3. 第二轮对话
    question2 = "那它和定期维护有什么区别？"

    print("\n===== 第二轮对话 =====")
    print("用户：", question2)

    answer2 = chat(conversation, question2)

    print("Agent：", answer2)
    print("当前历史消息数量：", len(conversation.get_history()))


    # 4. 查看保存的历史记录
    print("\n===== 历史记录 =====")

    for index, message in enumerate(
        conversation.get_history(), start=1
    ):
        print(
            f"{index}. {message['role']}: "
            f"{message['content'][:60]}..."
        )


if __name__ == "__main__":
    main()
