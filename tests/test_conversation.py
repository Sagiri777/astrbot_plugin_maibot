"""Verify Neo-MoFox-inspired framing in real AstrBot requests and decisions."""

import copy
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.message import Message, dump_messages_with_checkpoints
from astrbot.core.message.components import At, Image, Plain, Reply
from astrbot.core.message.message_event_result import MessageEventResult

from .test_main import MaiHumanizerPlugin, _event, package_name
from .test_output_quality import _context

conversation = importlib.import_module(f"{package_name}.conversation")


def _message(text, message_id, *, components=(), private=False):
    event = _event(private=private)
    event.get_message_str.return_value = text
    event.message_obj.message_id = message_id
    event.get_messages.return_value = list(components)
    return event


@pytest.mark.parametrize(
    ("parts", "addresses_bot"),
    [
        ([At(qq="other-user", name="机器人")], False),
        ([At(qq="all")], False),
        ([At(qq="bot-1", name="别的昵称")], True),
        ([Reply(id="old", sender_id="bot-1", message_str="已发送")], True),
    ],
)
def test_addressing_uses_exact_platform_ids(parts, addresses_bot):
    event = _message("怎么看", "new", components=parts)
    result = conversation.event_addressing(event)
    assert result["addresses_bot"] is addresses_bot
    assert result["message_id"] == "new"
    assert result["speaker_id"] == "user-1"


def test_multimodal_quote_falls_back_to_plain_text_only():
    quote = Reply(
        id="old",
        sender_id="other",
        message_str="",
        chain=[Plain("原文"), Image.fromURL("https://example.com/private-image.png")],
    )
    result = conversation.event_addressing(_message("嗯", "new", components=[quote]))
    assert result["quoted_text"] == "原文"
    assert "private-image" not in json.dumps(result)


@pytest.mark.asyncio
async def test_reply_to_older_message_focuses_retrieval_before_context_limit():
    context = _context()
    context.llm_generate = AsyncMock()
    plugin = MaiHumanizerPlugin(context, {"context_injection_messages": 2})
    await plugin._record_message(_message("发布遇到 zzz 故障", "release"))
    await plugin._record_message(
        _message("先查构建", "followup", components=[Reply(id="release")])
    )
    for index in range(4):
        await plugin._record_message(_message("大家在玩 xxx 游戏", f"game-{index}"))
    event = _message(
        "已经修好了，我们明天发布", "latest", components=[Reply(id="release")]
    )
    await plugin._record_message(event)
    plugin._store["jargon"]["test:session"] = [
        {"term": "zzz", "meaning": "发布故障代号"},
        {"term": "xxx", "meaning": "游戏代号"},
    ]
    request = ProviderRequest(
        prompt="已经修好了，我们明天发布", system_prompt="你是小夏"
    )
    await plugin.humanize_request(event, request)
    hint = request.extra_user_content_parts[0].text
    assert "发布故障代号" in hint and "先查构建" in hint
    assert "游戏代号" not in hint and "大家在玩" not in hint
    assert "已经修好了" not in hint  # The current message is already req.prompt.
    assert request.prompt == "已经修好了，我们明天发布"
    context.llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_quoted_history_uses_quote_without_unrelated_background():
    plugin = MaiHumanizerPlugin(_context(), {})
    await plugin._record_message(_message("旁人闲聊", "ambient"))
    event = _message(
        "现在换个计划",
        "latest",
        components=[
            Reply(
                id="outside-buffer",
                sender_id="other",
                message_str="之前计划周五出门",
            )
        ],
    )
    request = ProviderRequest(prompt="现在换个计划")
    await plugin.humanize_request(event, request)
    hint = request.extra_user_content_parts[0].text
    assert "之前计划周五出门" in hint
    assert "旁人闲聊" not in hint
    assert "outside-buffer" in hint


@pytest.mark.asyncio
async def test_private_continuity_is_sent_temporarily_without_observation():
    context = _context()
    context.llm_generate = AsyncMock()
    plugin = MaiHumanizerPlugin(context, {})
    request = ProviderRequest(
        prompt="今天不想说这个了",
        system_prompt="你是小夏",
        contexts=[
            {"role": "user", "content": "我今天不太开心"},
            {"role": "assistant", "content": "愿意讲讲发生了什么吗？"},
            {
                "role": "assistant",
                "content": "内部工具规划",
                "tool_calls": [{"id": "call-1"}],
            },
            {"role": "tool", "content": "工具秘密"},
            {
                "role": "assistant",
                "content": [
                    {"type": "think", "think": "内部思考"},
                    {"type": "text", "text": "临时推演", "_no_save": True},
                ],
            },
        ],
    )
    original_contexts = copy.deepcopy(request.contexts)
    await plugin.humanize_request(
        _message("今天不想说这个了", "p1", private=True), request
    )
    sent = await request.assemble_context()
    sent_text = json.dumps(sent, ensure_ascii=False)
    assert "愿意讲讲发生了什么吗？" in sent_text
    assert "recently_sent_bot_texts_newest_first" in sent_text
    assert (
        "内部" not in sent_text
        and "工具秘密" not in sent_text
        and "临时推演" not in sent_text
    )
    saved = json.dumps(
        dump_messages_with_checkpoints([Message.model_validate(sent)]),
        ensure_ascii=False,
    )
    assert "今天不想说这个了" in saved
    assert "愿意讲讲" not in saved and "mai_humanizer_runtime" not in saved
    assert request.contexts == original_contexts
    assert plugin._buffers == {} and plugin._learning_buffers == {}
    context.llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_interaction_switch_disables_guidance_and_thread_filtering():
    plugin = MaiHumanizerPlugin(_context(), {"interaction_context_enabled": False})
    await plugin._record_message(_message("周围的话题", "ambient"))
    event = _message(
        "继续", "latest", components=[Reply(id="missing", message_str="引用原文")]
    )
    request = ProviderRequest(
        prompt="继续", contexts=[{"role": "assistant", "content": "先前的问题？"}]
    )
    await plugin.humanize_request(event, request)
    hint = request.extra_user_content_parts[0].text
    assert "周围的话题" in hint
    assert "引用原文" not in hint and "recently_sent" not in hint
    assert conversation.INTERACTION_GUIDANCE not in request.system_prompt


@pytest.mark.asyncio
async def test_zero_context_uses_explicit_quote_but_does_not_recall_buffer():
    plugin = MaiHumanizerPlugin(_context(), {"context_injection_messages": 0})
    await plugin._record_message(_message("缓冲中的 xxx", "quoted"))
    plugin._store["jargon"]["test:session"] = [{"term": "xxx", "meaning": "不应召回"}]
    event = _message(
        "继续", "latest", components=[Reply(id="quoted", message_str="平台引用原文")]
    )
    request = ProviderRequest(prompt="继续")
    await plugin.humanize_request(event, request)
    hint = request.extra_user_content_parts[0].text
    assert "平台引用原文" in hint
    assert "缓冲中的" not in hint and "不应召回" not in hint


def test_thread_keeps_sent_reply_without_inventing_outbound_platform_id():
    records = [
        {"message_id": "q1", "text": "问题"},
        {"role": "assistant", "reply_to_id": "q1", "text": "真实回复"},
        {"message_id": "unrelated", "text": "旁人的话"},
    ]
    focused = conversation.focus_reply_thread(records, {"reply_to_id": "q1"})
    assert [item["text"] for item in focused] == ["问题", "真实回复"]


def test_recent_own_speech_is_bounded_and_deduplicated():
    texts = ["旧的问题？", "a" * 500, "b" * 500, "a" * 500, "刚才的问题？"]
    hint = conversation.build_interaction_hint(
        {},
        recent_messages=[],
        contexts=[{"role": "assistant", "content": text} for text in texts],
    )
    payload = json.loads(hint[hint.index("{", hint.index("\n")) :])
    assert payload["recently_sent_bot_texts_newest_first"] == [
        "刚才的问题？",
        "a" * 400,
        "b" * 400,
    ]


@pytest.mark.asyncio
async def test_planner_latest_event_and_decision_continuity_are_scoped_and_bounded():
    context = _context()
    context.llm_generate = AsyncMock(
        return_value=SimpleNamespace(
            completion_text='{"action":"ignore","confidence":0.9,"reason":"让他们先聊"}',
        )
    )
    plugin = MaiHumanizerPlugin(
        context,
        {
            "autonomous_reply_enabled": True,
            "planner_decision_interval": 1,
            "autonomous_reply_cooldown_seconds": 0,
        },
    )
    for index in range(6):
        event = _message(f"聊天{index}", f"id-{index}")
        assert [item async for item in plugin.observe_and_plan(event)] == []
    decisions = plugin._planner_decisions["test:session"]
    assert len(decisions) == 4
    assert [item["source_id"] for item in decisions] == ["m3", "m4", "m5", "m6"]
    payload = json.loads(context.llm_generate.await_args.kwargs["prompt"])
    assert [item["text"] for item in payload["latest_events"]] == ["聊天5"]
    assert all(item["text"] != "聊天5" for item in payload["conversation_context"])
    assert len(payload["recent_decisions"]) == 4
    other_event = _message("新群的请求", "new")
    other_event.unified_msg_origin = "other:group"
    assert [item async for item in plugin.observe_and_plan(other_event)] == []
    other_payload = json.loads(context.llm_generate.await_args.kwargs["prompt"])
    assert other_payload["recent_decisions"] == []
    assert [item["text"] for item in other_payload["latest_events"]] == ["新群的请求"]
    plugin._persist_store = AsyncMock()
    assert len([item async for item in plugin.forget_knowledge(event, "scope")]) == 1
    assert "test:session" not in plugin._planner_decisions
    assert "other:group" in plugin._planner_decisions


@pytest.mark.asyncio
async def test_planner_quote_focus_and_delivery_only_after_send():
    context = _context()
    context.llm_generate = AsyncMock(
        return_value=SimpleNamespace(
            completion_text=(
                '{"action":"reply","confidence":0.95,"reason":"可以回答","reply_guide":"回应发布计划"}'
            )
        )
    )
    context.conversation_manager.get_curr_conversation_id = AsyncMock(return_value="c1")
    context.conversation_manager.get_conversation = AsyncMock(return_value=object())
    plugin = MaiHumanizerPlugin(
        context,
        {
            "autonomous_reply_enabled": True,
            "planner_decision_interval": 1,
            "planner_context_messages": 2,
            "autonomous_reply_cooldown_seconds": 0,
        },
    )
    await plugin._record_message(_message("原来的发布计划", "old"))
    await plugin._record_message(_message("无关的游戏话题", "ambient"))
    event = _message(
        "改为明天发布", "new", components=[Reply(id="old", sender_id="user-1")]
    )
    event.request_llm.return_value = ProviderRequest(prompt="改为明天发布")
    assert len([item async for item in plugin.observe_and_plan(event)]) == 1
    payload = json.loads(context.llm_generate.await_args.kwargs["prompt"])
    assert [item["text"] for item in payload["conversation_context"]] == [
        "原来的发布计划"
    ]
    assert payload["current_message_addressing"]["reply_to_id"] == "old"
    decision = plugin._planner_decisions["test:session"][0]
    assert decision["delivered"] is False
    await plugin.capture_autonomous_reply(
        event, SimpleNamespace(completion_text="那明天再发布。", role="assistant")
    )
    assert decision["delivered"] is False
    assert not any(
        item["role"] == "assistant" for item in plugin._buffers["test:session"]
    )
    event.get_result.return_value = MessageEventResult().message("那明天再发布。")
    await plugin.track_autonomous_reply(event)
    assert decision["delivered"] is True
    sent = plugin._buffers["test:session"][-1]
    assert sent["role"] == "assistant" and sent["speaker_id"] == "bot-1"
    assert sent.get("message_id", "") != "new"  # Incoming IDs are not outgoing IDs.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result", [None, MessageEventResult().file_image("/tmp/only-image.png")]
)
async def test_generated_text_is_not_recorded_when_final_result_has_no_speech(result):
    plugin = MaiHumanizerPlugin(_context(), {})
    event = _message("有人吗", "m1")
    event.set_extra("_mai_autonomous_reply", True)
    event.set_extra("_mai_source_id", "m1")
    plugin._planner_decisions["test:session"] = [
        {"source_id": "m1", "delivered": False}
    ]
    await plugin.capture_autonomous_reply(
        event,
        SimpleNamespace(completion_text="没有实际发送的生成文本", role="assistant"),
    )
    event.get_result.return_value = result
    await plugin.track_autonomous_reply(event)
    assert plugin._buffers == {}
    assert plugin._planner_decisions["test:session"][0]["delivered"] is False


@pytest.mark.asyncio
async def test_quote_text_is_preserved_when_adapter_has_no_message_id():
    plugin = MaiHumanizerPlugin(_context(), {})
    event = _message(
        "先不聊这个",
        "m1",
        components=[Reply(id="", sender_id="bot-1", message_str="刚才的问题？")],
    )
    request = ProviderRequest(prompt="先不聊这个")
    await plugin.humanize_request(event, request)
    assert "刚才的问题？" in request.extra_user_content_parts[0].text
