"""Regression tests for natural group-member behavior."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from astrbot.api.provider import ProviderRequest

from .test_main import MaiHumanizerPlugin, _event, package_name
from .test_output_quality import _context

naturalness = importlib.import_module(f"{package_name}.naturalness")
planner = importlib.import_module(f"{package_name}.planner")


def test_register_hint_follows_short_chat_and_flags_repeated_opening():
    hint = naturalness.build_naturalness_hint(
        current_message="这也太离谱了",
        is_private_chat=False,
        recent_messages=[
            {"role": "user", "text": "笑死"},
            {"role": "user", "text": "怎么了"},
            {"role": "assistant", "text": "确实有点，我看看"},
            {"role": "assistant", "text": "确实有点，没想到"},
            {"role": "user", "text": "就刚刚"},
        ],
    )
    assert "随口玩梗" in hint
    assert "短句" in hint
    assert "确实有点" in hint


def test_no_context_or_formal_task_does_not_invent_a_casual_style():
    assert (
        naturalness.build_naturalness_hint(
            current_message="你好", recent_messages=[], is_private_chat=False
        )
        == ""
    )
    assert (
        naturalness.build_naturalness_hint(
            current_message="请按 JSON 格式输出详细步骤",
            recent_messages=[{"role": "user", "text": "哈哈"}],
            is_private_chat=False,
        )
        == ""
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("（笑了笑）\n这个我知道。", "这个我知道。"),
        ("*歪头看你* 怎么了？", "怎么了？"),
        ("这个我知道。\n希望对你有帮助。", "这个我知道。"),
        ("（笑）这个确实离谱。", "（笑）这个确实离谱。"),
    ],
)
def test_visible_reply_cleanup_is_conservative(raw, expected):
    assert naturalness.clean_visible_reply(raw, current_message="你知道吗") == expected


def test_cleanup_preserves_code_and_explicit_roleplay():
    code = "```python\nprint('（点头）')\n```"
    assert naturalness.clean_visible_reply(code, current_message="写代码") == code
    roleplay = "（笑了笑）我来了。"
    assert (
        naturalness.clean_visible_reply(roleplay, current_message="请角色扮演")
        == roleplay
    )


@pytest.mark.asyncio
async def test_llm_response_is_cleaned_before_capture_and_send_tracking():
    plugin = MaiHumanizerPlugin(_context(), {})
    event = _event(private=False)
    event.get_message_str.return_value = "你怎么看"
    await plugin.humanize_request(event, ProviderRequest(prompt="你怎么看"))
    response = SimpleNamespace(
        completion_text="（点头）\n我也这么觉得。\n希望对你有帮助。",
        role="assistant",
    )
    await plugin.capture_autonomous_reply(event, response)
    assert response.completion_text == "我也这么觉得。"
    assert event.get_extra("_mai_generated_reply_text") == "我也这么觉得。"


@pytest.mark.asyncio
async def test_social_register_uses_existing_buffer_without_extra_model_call():
    context = _context()
    context.llm_generate = AsyncMock()
    plugin = MaiHumanizerPlugin(context, {})
    plugin._buffers["test:session"] = [
        {"role": "user", "text": "笑死"},
        {"role": "user", "text": "怎么了"},
        {"role": "user", "text": "就刚刚"},
    ]
    request = ProviderRequest(prompt="这也太离谱了")
    await plugin.humanize_request(_event(private=False), request)
    assert "本轮群聊自然度" in request.extra_user_content_parts[0].text
    context.llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_consecutive_autonomous_replies_force_a_real_pause():
    context = _context()
    context.llm_generate = AsyncMock()
    plugin = MaiHumanizerPlugin(
        context,
        {
            "autonomous_reply_enabled": True,
            "planner_decision_interval": 1,
            "autonomous_reply_cooldown_seconds": 0,
            "max_autonomous_reply_streak": 2,
        },
    )
    plugin._planner_decisions["test:session"] = [
        {"action": "reply", "delivered": True},
        {"action": "reply", "delivered": True},
    ]
    event = _event(private=False)
    event.get_message_str.return_value = "然后又开始聊别的了"
    assert [item async for item in plugin.observe_and_plan(event)] == []
    context.llm_generate.assert_not_awaited()
    assert plugin._planner_decisions["test:session"][-1]["action"] == "ignore"
    assert (
        naturalness.delivered_reply_streak(plugin._planner_decisions["test:session"])
        == 0
    )


def test_undelivered_plan_does_not_count_as_real_participation():
    assert (
        naturalness.delivered_reply_streak(
            [
                {"action": "reply", "delivered": True},
                {"action": "reply", "delivered": False},
            ]
        )
        == 0
    )


def test_low_information_trigger_can_focus_a_recent_substantive_message():
    messages = [
        {
            "source_id": "m1",
            "role": "user",
            "text": "有人知道部署后一直显示旧版本该怎么排查吗",
        },
        {"source_id": "m2", "role": "assistant", "text": "先看看缓存"},
        {"source_id": "m3", "role": "user", "text": "哈哈"},
    ]
    focus = naturalness.select_conversation_focus(
        messages, current_message_id="m3", current_addressing={}
    )
    assert focus["source_id"] == "m1"


@pytest.mark.parametrize(
    ("text", "addressing", "reply_to_id"),
    [
        ("这个方案为什么一直显示旧版本", {}, ""),
        ("哈哈", {"addresses_bot": True}, ""),
        ("哈哈", {}, "message-1"),
    ],
)
def test_focus_never_overrides_new_requests_mentions_or_quotes(
    text, addressing, reply_to_id
):
    messages = [
        {"source_id": "m1", "role": "user", "text": "昨天那个游戏终于通关了"},
        {
            "source_id": "m2",
            "role": "user",
            "text": text,
            "reply_to_id": reply_to_id,
        },
    ]
    focus = naturalness.select_conversation_focus(
        messages,
        current_message_id="m2",
        current_addressing=addressing,
    )
    assert focus["source_id"] == "m2"


def test_focus_does_not_replace_one_low_information_message_with_another():
    messages = [
        {"source_id": "m1", "role": "user", "text": "确实"},
        {"source_id": "m2", "role": "user", "text": "哈哈"},
    ]
    focus = naturalness.select_conversation_focus(
        messages, current_message_id="m2", current_addressing={}
    )
    assert focus["source_id"] == "m2"


def test_closing_hint_prevents_chasing_or_starting_a_new_topic():
    hint = naturalness.build_naturalness_hint(
        current_message="我先去忙了，回头聊",
        recent_messages=[{"role": "user", "text": "今天事情有点多"}],
        is_private_chat=False,
    )
    assert naturalness.is_conversation_closing("我先去忙了，回头聊")
    assert "不要追问" in hint
    assert "另起话题" in hint
    assert not naturalness.is_conversation_closing("最近忙着写项目")
    assert not naturalness.is_conversation_closing("我睡了三个小时")
    assert not naturalness.is_conversation_closing("先这样做，再检查输出")
    assert "自然收尾" in naturalness.build_naturalness_hint(
        current_message="晚安",
        recent_messages=[],
        is_private_chat=True,
    )


def test_repeated_content_phrases_and_question_habit_are_visible_to_replyer():
    hint = naturalness.build_naturalness_hint(
        current_message="今晚吃什么",
        recent_messages=[
            {"role": "assistant", "text": "我觉得这个还挺不错的，你觉得呢？"},
            {"role": "assistant", "text": "我觉得那个还挺不错的，你要试试吗？"},
            {"role": "user", "text": "先看看"},
        ],
        is_private_chat=False,
    )
    assert "反复说过" in hint
    assert "最近两条都以问句收尾" in hint


@pytest.mark.asyncio
async def test_planner_receives_focus_and_closing_state_without_extra_calls():
    context = _context()
    context.llm_generate = AsyncMock(
        return_value=SimpleNamespace(
            completion_text='{"action":"ignore","confidence":0.9,"reason":"话题已结束"}'
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
    plugin._buffers["test:session"] = [
        {
            "source_id": "old-1",
            "role": "user",
            "speaker_name": "Bob",
            "text": "有人知道部署后一直显示旧版本该怎么排查吗",
            "message_id": "",
            "speaker_id": "user-2",
            "reply_to_id": "",
        }
    ]
    event = _event(private=False)
    event.get_message_str.return_value = "哈哈"
    assert [item async for item in plugin.observe_and_plan(event)] == []
    payload = json.loads(context.llm_generate.await_args.kwargs["prompt"])
    assert payload["current_message_id"] != payload["focus_message"]["source_id"]
    assert payload["focus_message"]["source_id"] == "old-1"
    assert payload["conversation_closing"] is False
    assert context.llm_generate.await_count == 1


def test_planner_closing_rule_applies_to_participation_and_reply_guide():
    system, user = planner.build_planner_prompts(
        bot_name="机器人",
        behavior_style="自然参与",
        recent_messages=[
            {"source_id": "m1", "role": "user", "text": "我先去忙了，回头聊"}
        ],
        current_message_id="m1",
        focus_message={"source_id": "m1", "text": "我先去忙了，回头聊"},
        conversation_closing=True,
    )
    assert "通常安静" in system
    assert "不追问" in system
    assert json.loads(user)["conversation_closing"] is True
