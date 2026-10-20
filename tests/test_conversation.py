
# tests/test_conversation.py
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conversation import ConversationManager, chat


def test_add_messages():
    """测试保存和读取用户及 Agent 消息。"""

    # 1. 创建会话管理对象
    conversation = ConversationManager()

    # 2. 添加两条消息
    conversation.add_user_message(
        "什么是预测性维护？"
    )

    conversation.add_assistant_message(
        "预测性维护通过设备数据预测潜在故障。"
    )

    # 3. 获取历史记录
    history = conversation.get_history()

    # 4. 验证消息数量
    assert len(history) == 2

    # 5. 验证用户消息
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "什么是预测性维护？"

    # 6. 验证 Agent 消息
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == (
        "预测性维护通过设备数据预测潜在故障。"
    )



def test_clear_history():
    """测试清空历史记录后还能继续使用会话。"""

    # 1. 创建会话管理器
    conversation = ConversationManager()

    # 2. 添加历史消息
    conversation.add_user_message("你好")
    conversation.add_assistant_message("你好！")

    assert len(conversation.get_history()) == 2

    # 3. 清空历史
    conversation.clear_history()

    # 4. 验证历史已被清空
    assert conversation.get_history() == []

    # 5. 清空后重新添加消息
    conversation.add_user_message("新的问题")

    # 6. 验证会话可以继续使用
    history = conversation.get_history()

    assert len(history) == 1
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "新的问题"



def test_history_copy():
    """测试外部修改历史副本不会影响内部记录。"""

    # 1. 创建会话管理器
    conversation = ConversationManager()

    conversation.add_user_message("你好")
    conversation.add_assistant_message("你好！")

    # 2. 获取历史记录副本
    history = conversation.get_history()

    # 3. 修改返回的字典
    history[0]["content"] = "被修改的消息"

    # 4. 向返回的列表追加消息
    history.append({
        "role": "user",
        "content": "额外添加的消息"
    })

    # 5. 重新读取内部历史
    original_history = conversation.get_history()

    # 内部仍然只有两条消息
    assert len(original_history) == 2

    # 第一条消息没有被修改
    assert original_history[0]["content"] == "你好"

    # 第二条消息保持不变
    assert original_history[1]["content"] == "你好！"



def test_chat_auto_history():
    """测试 chat() 自动传递和保存对话历史。"""

    # 1. 创建会话管理器
    conversation = ConversationManager()

    # 2. 准备已有历史
    conversation.add_user_message("什么是预测性维护？")
    conversation.add_assistant_message(
        "预测性维护通过设备数据识别潜在故障风险。"
    )

    # 3. 模拟 Agent 回答，不调用真实 DeepSeek
    with patch("agent.run_agent") as mock_agent:

        mock_agent.return_value = (
            "定期维护按照固定周期执行，"
            "预测性维护则参考设备实际状态。"
        )

        # 4. 调用 chat()
        answer = chat(
            conversation,
            "那它和定期维护有什么区别？"
        )

        # 5. 验证 Agent 收到了正确的历史
        mock_agent.assert_called_once_with(
            "那它和定期维护有什么区别？",
            history=[
                {
                    "role": "user",
                    "content": "什么是预测性维护？"
                },
                {
                    "role": "assistant",
                    "content": (
                        "预测性维护通过设备数据识别潜在故障风险。"
                    )
                }
            ]
        )

    # 6. 获取更新后的历史
    history = conversation.get_history()

    # 7. 验证新增了一轮问答
    assert len(history) == 4

    assert history[2] == {
        "role": "user",
        "content": "那它和定期维护有什么区别？"
    }

    assert history[3] == {
        "role": "assistant",
        "content": (
            "定期维护按照固定周期执行，"
            "预测性维护则参考设备实际状态。"
        )
    }

    # 8. 验证返回值
    assert answer == history[3]["content"]



def test_chat_agent_failure():
    """测试 Agent 调用异常时，历史记录保持不变。"""

    # 1. 创建会话管理器
    conversation = ConversationManager()

    conversation.add_user_message("什么是预测性维护？")
    conversation.add_assistant_message(
        "预测性维护通过设备数据识别故障风险。"
    )

    # 2. 保存调用前的历史
    original_history = conversation.get_history()

    # 3. 模拟 Agent 调用失败
    with patch("agent.run_agent") as mock_agent:

        mock_agent.side_effect = RuntimeError(
            "模拟 DeepSeek API 调用失败"
        )

        # 4. 验证 chat() 会向上抛出异常
        with pytest.raises(RuntimeError):
            chat(
                conversation,
                "那它和定期维护有什么区别？"
            )

        # 5. 验证 Agent 只调用了一次
        mock_agent.assert_called_once()

    # 6. 验证聊天历史没有被修改
    history_after_failure = conversation.get_history()

    assert history_after_failure == original_history
    assert len(history_after_failure) == 2



def test_chat_truncated_response():
    """测试 DeepSeek 回答截断时，不保存不完整的对话。"""

    # 1. 创建会话并保存原有历史
    conversation = ConversationManager()

    conversation.add_user_message("什么是预测性维护？")
    conversation.add_assistant_message(
        "预测性维护通过设备数据识别潜在故障风险。"
    )

    original_history = conversation.get_history()

    # 2. 模拟 DeepSeek 返回被截断的回答
    mock_response = MagicMock()
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].message.content = (
        "预测性维护主要包括以下几个步骤：第一"
    )
    mock_response.choices[0].finish_reason = "length"

    # 3. 仅模拟 OpenAI 客户端，实际执行 run_agent()
    with patch("agent.OpenAI") as mock_openai:
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.return_value = (
            mock_response
        )

        # 4. 验证异常能从 run_agent() 传递到 chat()
        with pytest.raises(RuntimeError, match="Token 上限"):
            chat(
                conversation,
                "请详细介绍预测性维护的技术流程"
            )

        mock_client.chat.completions.create.assert_called_once()

    # 5. 验证历史记录没有被修改
    assert conversation.get_history() == original_history
