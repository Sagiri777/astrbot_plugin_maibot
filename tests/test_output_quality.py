"""Regression checks for the paths that can improve or degrade actual replies."""

import asyncio
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.message import (
    Message,
    dump_messages_with_checkpoints,
)
from astrbot.core.message.message_event_result import MessageEventResult

from .test_main import MaiHumanizerPlugin, _event, package_name

expression = importlib.import_module(f"{package_name}.expression")
evaluation = importlib.import_module(f"{package_name}.evaluation")
learning = importlib.import_module(f"{package_name}.learning")
planner = importlib.import_module(f"{package_name}.planner")
behavior = importlib.import_module(f"{package_name}.behavior")


def _context():
    context = MagicMock()
    context.get_config.return_value = {
        "provider_ltm_settings": {"group_icl_enable": False}
    }
    context.get_current_chat_provider_id = AsyncMock(return_value="chat-model")
    return context


def _expressions():
    return [
        {
            "situation": "分享进展",
            "style": "简短肯定具体进展",
            "keywords": ["完成"],
            "count": 1,
        },
        {
            "situation": "玩笑",
            "style": "随口开个玩笑",
            "keywords": ["笑死"],
            "count": 100,
        },
    ]


@pytest.mark.asyncio
async def test_unrelated_high_frequency_habit_does_not_pollute_reply():
    plugin = MaiHumanizerPlugin(_context(), {})
    plugin._store["expressions"]["test:session"] = _expressions()
    request = ProviderRequest(
        prompt="请按 JSON 格式返回城市列表", system_prompt="你是严谨的助手。"
    )
    await plugin.humanize_request(_event(private=True), request)
    assert request.extra_user_content_parts == []
    assert request.prompt == "请按 JSON 格式返回城市列表"
    assert request.system_prompt.startswith("你是严谨的助手。")


@pytest.mark.asyncio
async def test_zero_context_means_no_recent_context_or_jargon_recall():
    plugin = MaiHumanizerPlugin(_context(), {"context_injection_messages": 0})
    plugin._buffers["test:session"] = [{"speaker_name": "Bob", "text": "xswl"}]
    plugin._store["jargon"]["test:session"] = [{"term": "xswl", "meaning": "笑死我了"}]
    request = ProviderRequest(prompt="你好")
    await plugin.humanize_request(_event(private=False), request)
    assert request.extra_user_content_parts == []


@pytest.mark.asyncio
async def test_ascii_jargon_does_not_match_inside_another_word():
    plugin = MaiHumanizerPlugin(_context(), {})
    plugin._store["jargon"]["test:session"] = [
        {"term": "AI", "meaning": "本群游戏的特殊缩写"}
    ]
    request = ProviderRequest(prompt="rain today")
    await plugin.humanize_request(_event(private=True), request)
    assert request.extra_user_content_parts == []


@pytest.mark.asyncio
async def test_semantic_selection_can_recall_without_keyword_and_runs_once():
    context = _context()
    context.llm_generate = AsyncMock(
        return_value=SimpleNamespace(completion_text='{"selected_ids":[0,0,true,999]}')
    )
    plugin = MaiHumanizerPlugin(
        context,
        {
            "expression_selection_enabled": True,
            "expression_selector_provider_id": "selector",
        },
    )
    plugin._store["expressions"]["test:session"] = _expressions()
    request = ProviderRequest(prompt="终于交稿啦")
    event = _event(private=True)
    await plugin.humanize_request(event, request)
    await plugin.humanize_request(event, request)
    assert len(request.extra_user_content_parts) == 1
    assert "简短肯定具体进展" in request.extra_user_content_parts[0].text
    assert "随口开个玩笑" not in request.extra_user_content_parts[0].text
    context.llm_generate.assert_awaited_once()
    assert context.llm_generate.await_args.kwargs["chat_provider_id"] == "selector"
    assert (
        json.loads(context.llm_generate.await_args.kwargs["prompt"])["current_message"]
        == "终于交稿啦"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    ['{"selected_ids":[]}', '{"selected_ids":null}', "not json", TimeoutError()],
)
async def test_empty_or_failed_selector_uses_persona_without_forced_habit(response):
    context = _context()
    context.llm_generate = AsyncMock()
    if isinstance(response, Exception):
        context.llm_generate.side_effect = response
    else:
        context.llm_generate.return_value = SimpleNamespace(completion_text=response)
    plugin = MaiHumanizerPlugin(context, {"expression_selection_enabled": True})
    plugin._store["expressions"]["test:session"] = _expressions()
    request = ProviderRequest(prompt="完成")
    await plugin.humanize_request(_event(private=True), request)
    assert request.extra_user_content_parts == []


@pytest.mark.asyncio
async def test_real_astrbot_serialization_sends_hints_but_does_not_persist_them():
    plugin = MaiHumanizerPlugin(_context(), {})
    plugin._store["expressions"]["test:session"] = _expressions()
    request = ProviderRequest(prompt="我完成了", system_prompt="base")
    await plugin.humanize_request(_event(private=True), request)
    serialized = await request.assemble_context()
    assert "简短肯定具体进展" in json.dumps(serialized, ensure_ascii=False)
    runtime_message = Message.model_validate(serialized)
    persisted = dump_messages_with_checkpoints([runtime_message])
    persisted_text = json.dumps(persisted, ensure_ascii=False)
    assert "我完成了" in persisted_text
    assert "简短肯定具体进展" not in persisted_text
    assert "mai_humanizer_runtime" not in persisted_text


@pytest.mark.asyncio
async def test_normal_sent_reply_is_visible_to_planner_but_not_learned():
    plugin = MaiHumanizerPlugin(_context(), {})
    event = _event(private=False)
    await plugin.humanize_request(event, ProviderRequest(prompt="进展如何"))
    await plugin.capture_autonomous_reply(
        event, SimpleNamespace(completion_text="已经完成了。", role="assistant")
    )
    assert "test:session" not in plugin._buffers
    event.get_result.return_value = MessageEventResult().message("已经完成了。")
    await plugin.track_autonomous_reply(event)
    await plugin.track_autonomous_reply(event)
    records = plugin._buffers["test:session"]
    assert len(records) == 1
    assert records[0]["role"] == "assistant"
    assert records[0]["text"] == "已经完成了。"
    assert plugin._learning_buffers == {}
    assert plugin._pending_behavior_feedback == {}


@pytest.mark.asyncio
async def test_private_reply_is_not_observed_when_private_learning_is_disabled():
    plugin = MaiHumanizerPlugin(_context(), {})
    event = _event(private=True)
    await plugin.humanize_request(event, ProviderRequest(prompt="秘密"))
    await plugin.capture_autonomous_reply(
        event, SimpleNamespace(completion_text="收到。")
    )
    event.get_result.return_value = MessageEventResult().message("收到。")
    await plugin.track_autonomous_reply(event)
    assert plugin._buffers == {}


@pytest.mark.asyncio
async def test_concurrent_planner_runs_once_and_discards_outdated_reply():
    context = _context()
    entered, release = asyncio.Event(), asyncio.Event()

    async def plan(**kwargs):
        entered.set()
        await release.wait()
        return SimpleNamespace(
            completion_text='{"action":"reply","confidence":0.95,"reply_guide":"回答旧问题"}'
        )

    context.llm_generate = AsyncMock(side_effect=plan)
    plugin = MaiHumanizerPlugin(
        context,
        {
            "autonomous_reply_enabled": True,
            "planner_decision_interval": 1,
            "autonomous_reply_cooldown_seconds": 0,
        },
    )
    first, second = _event(private=False), _event(private=False)
    first.get_message_str.return_value = "谁能帮我看一下"
    second.get_message_str.return_value = "已经解决了，不用了"

    async def collect(event):
        return [item async for item in plugin.observe_and_plan(event)]

    task = asyncio.create_task(collect(first))
    await asyncio.wait_for(entered.wait(), 2)
    assert await collect(second) == []
    release.set()
    assert await asyncio.wait_for(task, 2) == []
    context.llm_generate.assert_awaited_once()
    first.request_llm.assert_not_called()
    assert plugin._planning_scopes == set()
    assert plugin._planner_decisions == {}


@pytest.mark.asyncio
async def test_cancelling_planner_releases_scope():
    context = _context()
    entered = asyncio.Event()

    async def plan(**kwargs):
        entered.set()
        await asyncio.Event().wait()

    context.llm_generate = AsyncMock(side_effect=plan)
    plugin = MaiHumanizerPlugin(
        context,
        {
            "autonomous_reply_enabled": True,
            "planner_decision_interval": 1,
            "autonomous_reply_cooldown_seconds": 0,
        },
    )
    event = _event(private=False)
    event.get_message_str.return_value = "有人吗"

    async def collect():
        return [item async for item in plugin.observe_and_plan(event)]

    task = asyncio.create_task(collect())
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert plugin._planning_scopes == set()


@pytest.mark.parametrize(
    "confidence", ["NaN", "Infinity", "-Infinity", '"nan"', "null", "[]"]
)
def test_nonfinite_or_malformed_planner_confidence_cannot_trigger_reply(confidence):
    result = planner.parse_planner_decision(
        '{"action":"reply","confidence":' + confidence + "}"
    )
    assert result["confidence"] == 0.0
    assert result["action"] == "ignore"


def test_learning_rejects_wrong_speaker_and_jargon_absent_from_cited_evidence():
    result = learning.parse_learning_result(
        json.dumps(
            {
                "expressions": [{"situation": "x", "style": "y", "source_ids": None}],
                "profiles": [
                    {
                        "speaker_id": "alice",
                        "facts": [
                            {"fact": "喜欢下棋", "source_ids": ["m2"]},
                            {"fact": "喜欢徒步", "source_ids": ["m1"]},
                        ],
                    }
                ],
                "jargon": [
                    {
                        "term": "xswl",
                        "meaning": "笑死我了",
                        "confidence": 0.9,
                        "source_ids": ["m1"],
                    },
                    {
                        "term": "awsl",
                        "meaning": "啊我死了",
                        "confidence": 0.9,
                        "source_ids": ["m2"],
                    },
                ],
            }
        ),
        valid_source_ids={"m1", "m2"},
        valid_speaker_ids={"alice", "bob"},
        learn_expressions=True,
        learn_profiles=True,
        learn_jargon=True,
        source_messages={
            "m1": {"speaker_id": "alice", "text": "我喜欢徒步"},
            "m2": {"speaker_id": "bob", "text": "awsl"},
        },
    )
    assert result["expressions"] == []
    assert result["profiles"][0]["facts"] == [
        {"fact": "喜欢徒步", "source_ids": ["m1"]}
    ]
    assert [item["term"] for item in result["jargon"]] == ["awsl"]


@pytest.mark.asyncio
async def test_profile_injection_is_scoped_and_legacy_facts_are_not_guessed():
    plugin = MaiHumanizerPlugin(
        _context(), {"profile_learning_enabled": True, "inject_profiles_in_group": True}
    )
    plugin._store["profiles"]["test-platform:user-1"] = [
        {"fact": "当前群的公开偏好", "scope": "test:session"},
        {"fact": "来自私聊的偏好", "scope": "private:session"},
        {"fact": "旧记录没有来源"},
    ]
    request = ProviderRequest(prompt="我喜欢什么")
    await plugin.humanize_request(_event(private=False), request)
    hint = request.extra_user_content_parts[0].text
    assert "当前群的公开偏好" in hint
    assert "来自私聊的偏好" not in hint
    assert "旧记录没有来源" not in hint


def test_behavior_does_not_recall_unrelated_high_scoring_experience():
    assert (
        behavior.select_behavior_experiences(
            [
                {
                    "situation": "聊游戏",
                    "action": "开玩笑",
                    "lesson": "适度接梗",
                    "keywords": ["游戏"],
                    "score": 3,
                }
            ],
            context_text="今天很难过",
        )
        == []
    )
    assert (
        behavior.parse_behavior_feedback_result(
            '{"status":"success","source_ids":null}', valid_feedback_source_ids={"m1"}
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("swapped", [False, True])
async def test_live_comparison_uses_real_augmented_request_and_blind_order(
    monkeypatch, swapped
):
    monkeypatch.setattr(evaluation.secrets, "choice", lambda _: swapped)
    context = _context()
    context.llm_generate = AsyncMock(
        side_effect=[
            SimpleNamespace(completion_text="原始回复"),
            SimpleNamespace(completion_text="增强回复"),
            SimpleNamespace(
                completion_text=json.dumps(
                    {
                        "A": {
                            "task": 5,
                            "grounding": 5,
                            "naturalness": 4 if swapped else 2,
                        },
                        "B": {
                            "task": 5,
                            "grounding": 5,
                            "naturalness": 2 if swapped else 4,
                        },
                        "reason": "针对当前意图回应更自然",
                    }
                )
            ),
        ]
    )
    plugin = MaiHumanizerPlugin(context, {})
    event = _event(private=False)
    plugin._buffers["test:session"] = [
        {"role": "user", "speaker_name": "Alice", "text": "刚刚交稿了"}
    ]
    await plugin.humanize_request(
        event, ProviderRequest(prompt="你好", system_prompt="你是小夏。")
    )
    plugin._store["expressions"]["test:session"] = _expressions()
    event.plain_result.side_effect = lambda text: text
    reports = [item async for item in plugin.evaluate_output(event, "我完成了")]
    assert len(reports) == 1
    assert "自然表达：原版 2/5 → 增强 4/5" in reports[0]
    baseline, enhanced, judge = [
        call.kwargs for call in context.llm_generate.await_args_list
    ]
    assert baseline["chat_provider_id"] == enhanced["chat_provider_id"] == "chat-model"
    assert baseline["system_prompt"] == "你是小夏。"
    assert "<mai_humanizer>" in enhanced["system_prompt"]
    assert baseline["contexts"][:-1] == enhanced["contexts"][:-1]
    assert "简短肯定具体进展" not in json.dumps(
        baseline["contexts"], ensure_ascii=False
    )
    assert "简短肯定具体进展" in json.dumps(enhanced["contexts"], ensure_ascii=False)
    answers = json.loads(judge["prompt"])["answers"]
    assert answers["A"] == ("增强回复" if swapped else "原始回复")
    assert "原版" not in answers["A"] and "原版" not in answers["B"]
    assert plugin._evaluation_scopes == set()
    assert len(plugin._buffers["test:session"]) == 1
    assert not event.get_extra("_mai_eval_context_supplied")


@pytest.mark.asyncio
async def test_judge_failure_keeps_both_generated_outputs():
    context = _context()
    context.llm_generate = AsyncMock(
        side_effect=[
            SimpleNamespace(completion_text="输出一"),
            SimpleNamespace(completion_text="输出二"),
            RuntimeError("judge unavailable"),
        ]
    )
    report = await evaluation.compare_replies(
        context,
        "chat-model",
        ProviderRequest(prompt="你好"),
        ProviderRequest(prompt="你好"),
    )
    assert "输出一" in report and "输出二" in report
    assert "评审未完成" in report


@pytest.mark.asyncio
async def test_evaluation_requires_actual_persona_before_making_calls():
    context = _context()
    context.llm_generate = AsyncMock()
    plugin = MaiHumanizerPlugin(context, {})
    event = _event(private=True)
    event.plain_result.side_effect = lambda text: text
    reports = [item async for item in plugin.evaluate_output(event, "你好")]
    assert "先在当前聊天正常" in reports[0]
    context.llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_persistence_cannot_overwrite_newer_state_with_slow_old_snapshot():
    plugin = MaiHumanizerPlugin(_context(), {})
    entered, release = asyncio.Event(), asyncio.Event()
    written = []

    async def write(key, value):
        if not written:
            entered.set()
            await release.wait()
        written.append(value)

    plugin.put_kv_data = AsyncMock(side_effect=write)
    first = asyncio.create_task(plugin._persist_store())
    await asyncio.wait_for(entered.wait(), 2)
    plugin._store["expressions"]["test:session"] = _expressions()
    second = asyncio.create_task(plugin._persist_store())
    release.set()
    await asyncio.gather(first, second)
    assert written[0]["expressions"] == {}
    assert written[-1]["expressions"]["test:session"] == _expressions()


@pytest.mark.asyncio
async def test_latest_message_is_not_repeated_as_chat_history():
    plugin = MaiHumanizerPlugin(_context(), {})
    first = _event(private=False)
    first.get_message_str.return_value = "旧消息"
    second = _event(private=False)
    second.get_message_str.return_value = "当前问题"
    await plugin._record_message(first)
    await plugin._record_message(second)
    request = ProviderRequest(prompt="当前问题")
    await plugin.humanize_request(second, request)
    hint = request.extra_user_content_parts[0].text
    assert "旧消息" in hint
    assert "当前问题" not in hint


@pytest.mark.asyncio
async def test_bot_speech_does_not_retrigger_jargon_reference():
    plugin = MaiHumanizerPlugin(_context(), {"inject_recent_group_context": False})
    plugin._buffers["test:session"] = [{"role": "assistant", "text": "xswl"}]
    plugin._store["jargon"]["test:session"] = [{"term": "xswl", "meaning": "笑死我了"}]
    request = ProviderRequest(prompt="你好")
    await plugin.humanize_request(_event(private=False), request)
    hint = request.extra_user_content_parts[0].text
    assert "recently_sent_bot_texts_newest_first" in hint
    assert "圈内用语参考" not in hint
    assert "笑死我了" not in hint


@pytest.mark.asyncio
async def test_empty_selection_is_not_called_twice_for_same_request():
    context = _context()
    context.llm_generate = AsyncMock(
        return_value=SimpleNamespace(completion_text='{"selected_ids":[]}')
    )
    plugin = MaiHumanizerPlugin(context, {"expression_selection_enabled": True})
    plugin._store["expressions"]["test:session"] = _expressions()
    event = _event(private=True)
    request = ProviderRequest(prompt="完成")
    await plugin.humanize_request(event, request)
    await plugin.humanize_request(event, request)
    context.llm_generate.assert_awaited_once()
    assert request.extra_user_content_parts == []


def test_evaluation_command_accepts_full_question_with_spaces():
    from astrbot.core.star.filter.command import CommandFilter

    command = CommandFilter(
        "maieval",
        handler_md=SimpleNamespace(handler=MaiHumanizerPlugin.evaluate_output),
    )
    event = _event(private=True)
    event.is_at_or_wake_command = True
    event.get_message_str.return_value = "maieval 请只返回 JSON including English words"
    assert command.filter(event, {})
    assert (
        event.get_extra("parsed_params")["question"]
        == "请只返回 JSON including English words"
    )
