
# src/conversation.py


class ConversationManager:
    """管理单个会话的历史消息。"""

    def __init__(self):
        self.history = []

    def add_user_message(self, content: str):
        """保存用户消息。"""
        self.history.append({
            "role": "user",
            "content": content
        })

    def add_assistant_message(self, content: str):
        """保存 Agent 的回答。"""
        self.history.append({
            "role": "assistant",
            "content": content
        })

    def get_history(self) -> list[dict]:
        """返回当前会话历史的副本。"""
        return [message.copy() for message in self.history]

    def clear_history(self):
        """清空当前会话历史。"""
        self.history.clear()



def chat(
    conversation: ConversationManager,
    user_message: str
) -> str:
    """调用 Agent，并自动维护对话历史。"""

    # 1. 获取当前会话的历史消息
    history = conversation.get_history()

    # 2. 调用 Agent，传入历史和当前问题
    from agent import run_agent

    answer = run_agent(
        user_message,
        history=history
    )

    # 3. Agent 成功返回后，保存本轮问答
    conversation.add_user_message(user_message)
    conversation.add_assistant_message(answer)

    # 4. 返回 Agent 回答
    return answer
