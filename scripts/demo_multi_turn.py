# scripts/demo_multi_turn.py

from agent import run_agent


def main():
    # 第一轮：用户提出问题
    question1 = "什么是设备预测性维护？"

    print("===== 第一轮对话 =====")
    print("用户：", question1)

    answer1 = run_agent(question1)

    print("Agent：", answer1)


    # 保存第一轮对话历史
    history = [
        {
            "role": "user",
            "content": question1
        },
        {
            "role": "assistant",
            "content": answer1
        }
    ]


    # 第二轮：用户使用代词提问
    question2 = "那它和定期维护有什么区别？"

    print("\n===== 第二轮对话 =====")
    print("用户：", question2)

    answer2 = run_agent(
        question2,
        history=history
    )

    print("Agent：", answer2)


if __name__ == "__main__":
    main()
