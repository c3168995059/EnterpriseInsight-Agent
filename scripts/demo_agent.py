from agent import run_agent


def main():
    user_message = (
        "帮我分析一台 L 类设备，"
        "空气温度 300K，过程温度 310K，"
        "转速 1500rpm，扭矩 50Nm，"
        "刀具磨损 150min，它有没有故障风险？"
    )


    print("正在运行 EnterpriseInsight-Agent...")

    answer = run_agent(user_message)

    print("\n===== Agent 回答 =====")
    print(answer)


if __name__ == "__main__":
    main()
