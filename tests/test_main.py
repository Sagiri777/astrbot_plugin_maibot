import asyncio
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from astrbot.api.provider import ProviderRequest
from astrbot.core.message.message_event_result import MessageEventResult

if __package__ and "." in __package__:
    package_name = __package__.rsplit(".", 1)[0]
else:
    package_name = "_maihumanizer_test_package"
    package = ModuleType(package_name)
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules[package_name] = package

behavior_module = importlib.import_module(f"{package_name}.behavior")
learning_module = importlib.import_module(f"{package_name}.learning")
main_module = importlib.import_module(f"{package_name}.main")
planner_module = importlib.import_module(f"{package_name}.planner")

parse_behavior_feedback_result = behavior_module.parse_behavior_feedback_result
select_behavior_experiences = behavior_module.select_behavior_experiences
build_learning_prompts = learning_module.build_learning_prompts
parse_learning_result = learning_module.parse_learning_result
select_episodic_memories = learning_module.select_episodic_memories
_PROMPT_MARKER = main_module._PROMPT_MARKER
_RUNTIME_MARKER = main_module._RUNTIME_MARKER
MaiHumanizerPlugin = main_module.MaiHumanizerPlugin
parse_planner_decision = planner_module.parse_planner_decision


def _event(*, private: bool) -> MagicMock:
    event = MagicMock()
    event.is_private_chat.return_value = private
    event.unified_msg_origin = "test:session"
    event.get_platform_id.return_value = "test-platform"
    event.get_sender_id.return_value = "user-1"
    event.get_sender_name.return_value = "Alice"
    event.get_self_id.return_value = "bot-1"
    event.get_group_id.return_value = "" if private else "group-1"
    extras = {}
    event.get_extra.side_effect = lambda key, default=None: extras.get(key, default)
    event.set_extra.side_effect = lambda key, value: extras.__setitem__(key, value)
    event.is_at_or_wake_command = False
    return event


@pytest.mark.asyncio
async def test_injects_prompt_once_and_preserves_existing_persona() -> None:
    plugin = MaiHumanizerPlugin(MagicMock(), {})
    request = ProviderRequest(
        prompt="今天好累",
        session_id="session-1",
        system_prompt="你是小夏。",
    )

    await plugin.humanize_request(_event(private=True), request)
    await plugin.humanize_request(_event(private=True), request)

    assert request.system_prompt.startswith("你是小夏。")
    assert request.system_prompt.count(_PROMPT_MARKER) == 1
    assert "先建立连接" in request.system_prompt
    assert "私聊" in request.system_prompt


@pytest.mark.asyncio
async def test_respects_chat_scope_switches() -> None:
    plugin = MaiHumanizerPlugin(
        MagicMock(),
        {
            "apply_to_group": False,
            "apply_to_private": True,
        },
    )
    request = ProviderRequest(prompt="在吗", system_prompt="base")

    await plugin.humanize_request(_event(private=False), request)

    assert request.system_prompt == "base"
    assert request.extra_user_content_parts == []


@pytest.mark.asyncio
async def test_adds_expression_habits_to_stable_prompt() -> None:
    plugin = MaiHumanizerPlugin(
        MagicMock(),
        {
            "expression_habits": [
                {
                    "situation": "对方分享小成果",
                    "style": "先简短肯定，再顺着细节接一句",
                },
                {"situation": "", "style": "ignored"},
            ]
        },
    )
    request = ProviderRequest(prompt="我终于写完了")

    await plugin.humanize_request(_event(private=False), request)

    assert "对方分享小成果" in request.system_prompt
    assert "先简短肯定，再顺着细节接一句" in request.system_prompt
    assert "ignored" not in request.system_prompt


@pytest.mark.asyncio
async def test_alternate_style_is_temporary_and_not_duplicated() -> None:
    plugin = MaiHumanizerPlugin(
        MagicMock(),
        {
            "style_variation_probability": 1.0,
            "alternate_styles": ["这次只说一两句。"],
        },
    )
    request = ProviderRequest(prompt="你怎么看", session_id="session-2")
    event = _event(private=True)

    await plugin.humanize_request(event, request)
    await plugin.humanize_request(event, request)

    assert len(request.extra_user_content_parts) == 1
    runtime_hint = request.extra_user_content_parts[0]
    assert _RUNTIME_MARKER in runtime_hint.text
    assert "这次只说一两句。" in runtime_hint.text
    assert runtime_hint._no_save is True


def test_planner_parser_fails_closed_and_normalizes_confidence() -> None:
    invalid = parse_planner_decision("not json")
    valid = parse_planner_decision(
        'prefix {"action":"reply","confidence":2,'
        '"reason":"被直接询问","reply_guide":"接住问题","tone":"轻松"} suffix'
    )

    assert invalid["action"] == "ignore"
    assert valid == {
        "action": "reply",
        "confidence": 1.0,
        "reason": "被直接询问",
        "reply_guide": "接住问题",
        "tone": "轻松",
    }


def test_learning_parser_requires_real_evidence_ids() -> None:
    result = parse_learning_result(
        """
        {
          "expressions": [
            {
              "situation": "看到离谱结果",
              "style": "用短句表达惊讶",
              "keywords": ["离谱"],
              "source_ids": ["m1"]
            },
            {
              "situation": "无证据",
              "style": "不应保存",
              "keywords": [],
              "source_ids": ["missing"]
            }
          ],
          "jargon": [
            {
              "term": "xswl",
              "meaning": "笑死我了",
              "confidence": 0.9,
              "source_ids": ["m2"]
            }
          ],
                  "profiles": [
            {
              "speaker_id": "test-platform:user-1",
              "facts": [{"fact": "喜欢短回复", "source_ids": ["m1"]}]
            }
          ]
        }
        """,
        valid_source_ids={"m1", "m2"},
        valid_speaker_ids={"test-platform:user-1"},
        learn_expressions=True,
        learn_jargon=True,
        learn_profiles=True,
    )

    assert len(result["expressions"]) == 1
    assert result["jargon"][0]["term"] == "xswl"
    assert result["profiles"][0]["facts"][0]["fact"] == "喜欢短回复"


@pytest.mark.asyncio
async def test_injects_relevant_learned_expression_and_jargon() -> None:
    context = MagicMock()
    context.get_config.return_value = {
        "provider_ltm_settings": {"group_icl_enable": True}
    }
    plugin = MaiHumanizerPlugin(context, {"episodic_memory_enabled": True})
    plugin._store["expressions"]["test:session"] = [
        {
            "situation": "看到离谱结果",
            "style": "用很短的句子表达惊讶",
            "keywords": ["离谱"],
            "count": 3,
        }
    ]
    plugin._store["jargon"]["test:session"] = [
        {
            "term": "xswl",
            "meaning": "笑死我了",
            "confidence": 0.9,
            "count": 2,
        }
    ]
    plugin._store["memories"]["test:session"] = [
        {
            "id": "e1",
            "title": "之前的异常结果",
            "summary": "大家曾遇到过一次离谱的测试结果，后来发现是缓存没有刷新。",
            "recall_cues": ["离谱", "缓存"],
            "importance": 0.7,
        }
    ]
    request = ProviderRequest(prompt="这个结果太离谱了，xswl")

    await plugin.humanize_request(_event(private=False), request)

    runtime_hint = request.extra_user_content_parts[0].text
    assert "看到离谱结果" in runtime_hint
    assert "xswl：笑死我了" in runtime_hint
    assert "之前的异常结果" in runtime_hint


@pytest.mark.asyncio
async def test_autonomous_planner_hands_intent_to_replyer() -> None:
    context = MagicMock()
    context.get_config.return_value = {
        "provider_ltm_settings": {
            "active_reply": {"enable": False},
            "group_icl_enable": False,
        }
    }
    context.llm_generate = AsyncMock(
        return_value=SimpleNamespace(
            completion_text=(
                '{"action":"reply","confidence":0.9,"reason":"话题在等回应",'
                '"reply_guide":"顺着对方的问题给出简短看法","tone":"自然"}'
            )
        )
    )
    context.conversation_manager.get_curr_conversation_id = AsyncMock(
        return_value="conversation-1"
    )
    context.conversation_manager.get_conversation = AsyncMock(return_value=object())
    plugin = MaiHumanizerPlugin(
        context,
        {
            "autonomous_reply_enabled": True,
            "behavior_learning_enabled": True,
            "episodic_memory_enabled": True,
            "planner_provider_id": "planner-provider",
            "planner_decision_interval": 1,
            "autonomous_reply_cooldown_seconds": 0,
            "autonomous_reply_confidence_threshold": 0.7,
        },
    )
    plugin._store["behaviors"]["test:session"] = [
        {
            "id": "b1",
            "situation": "群友邀请评价方案",
            "action": "先给简短观点",
            "outcome": "其他人继续接话",
            "lesson": "开放讨论中给别人留接话空间",
            "keywords": ["方案"],
            "score": 0.8,
            "enabled": True,
        }
    ]
    plugin._store["memories"]["test:session"] = [
        {
            "id": "e1",
            "title": "方案边界讨论",
            "summary": "大家上次决定先明确方案边界，再继续拆任务。",
            "recall_cues": ["方案"],
            "importance": 0.8,
        }
    ]
    event = _event(private=False)
    event.get_message_str.return_value = "你们觉得这个方案怎么样"
    event.get_extra.return_value = {}
    event.is_at_or_wake_command = False
    generated_request = ProviderRequest(prompt="你们觉得这个方案怎么样")
    event.request_llm.return_value = generated_request

    yielded = [result async for result in plugin.observe_and_plan(event)]

    assert yielded == [generated_request]
    event.set_extra.assert_any_call(
        "_mai_planner_reply_guide",
        "顺着对方的问题给出简短看法",
    )
    event.set_extra.assert_any_call("_mai_planner_tone", "自然")
    event.stop_event.assert_called_once()
    planner_system_prompt = context.llm_generate.await_args.kwargs["system_prompt"]
    assert "开放讨论中给别人留接话空间" in planner_system_prompt
    assert "大家上次决定先明确方案边界" in planner_system_prompt
    assert plugin._store["behaviors"]["test:session"][0]["activation_count"] == 1


@pytest.mark.asyncio
async def test_learning_batch_merges_and_persists_scoped_records() -> None:
    context = MagicMock()
    context.llm_generate = AsyncMock(
        return_value=SimpleNamespace(
            completion_text="""
            {
              "expressions": [
                {
                  "situation": "看到离谱结果",
                  "style": "用短句表达惊讶",
                  "keywords": ["离谱"],
                  "source_ids": ["m1"]
                }
              ],
              "jargon": [
                {
                  "term": "xswl",
                  "meaning": "笑死我了",
                  "confidence": 0.9,
                  "source_ids": ["m2"]
                }
              ],
              "profiles": [
                {
                  "speaker_id": "test-platform:user-1",
                  "facts": [
                    {"fact": "喜欢短回复", "source_ids": ["m1"]}
                      ]
                    }
                  ],
                  "memories": [
                    {
                      "title": "讨论离谱的测试结果",
                      "summary": "大家确认测试结果异常，准备继续检查原因。",
                      "recall_cues": ["测试结果", "异常原因"],
                      "importance": 0.7,
                      "source_ids": ["m1", "m2"]
                    }
                  ]
                }
            """
        )
    )
    plugin = MaiHumanizerPlugin(
        context,
        {
            "learner_provider_id": "learner-provider",
            "expression_learning_enabled": True,
            "jargon_learning_enabled": True,
            "profile_learning_enabled": True,
            "episodic_memory_enabled": True,
        },
    )
    plugin._persist_store = AsyncMock()
    plugin._learning_scopes.add("test:session")
    messages = [
        {
            "source_id": "m1",
            "profile_key": "test-platform:user-1",
            "speaker_name": "Alice",
            "text": "这个结果太离谱了",
            "timestamp": "2026-07-29T10:00:00+00:00",
        },
        {
            "source_id": "m2",
            "profile_key": "test-platform:user-1",
            "speaker_name": "Alice",
            "text": "xswl",
            "timestamp": "2026-07-29T10:00:01+00:00",
        },
    ]

    counts = await plugin._run_learning_batch("test:session", messages)

    assert counts == {
        "expressions": 1,
        "jargon": 1,
        "profiles": 1,
        "memories": 1,
    }
    assert plugin._store["expressions"]["test:session"][0]["style"] == "用短句表达惊讶"
    assert plugin._store["jargon"]["test:session"][0]["meaning"] == "笑死我了"
    assert plugin._store["profiles"]["test-platform:user-1"][0]["fact"] == "喜欢短回复"
    assert plugin._store["memories"]["test:session"][0]["title"] == "讨论离谱的测试结果"
    assert "test:session" not in plugin._learning_scopes
    plugin._persist_store.assert_awaited_once()


def test_record_count_tolerates_malformed_legacy_values() -> None:
    assert MaiHumanizerPlugin._record_count({"count": "4"}) == 4
    assert MaiHumanizerPlugin._record_count({"count": "broken"}) == 1
    assert MaiHumanizerPlugin._record_count({"count": -2}) == 1


@pytest.mark.asyncio
async def test_forget_scope_removes_learned_and_buffered_data() -> None:
    plugin = MaiHumanizerPlugin(MagicMock(), {})
    plugin._store["expressions"]["test:session"] = [{"situation": "x"}]
    plugin._store["jargon"]["test:session"] = [{"term": "y"}]
    plugin._store["memories"]["test:session"] = [{"title": "z"}]
    plugin._buffers["test:session"] = [{"text": "recent"}]
    plugin._learning_buffers["test:session"] = [{"text": "pending"}]
    plugin._persist_store = AsyncMock()
    event = _event(private=False)
    event.plain_result.side_effect = lambda text: text

    yielded = [
        result async for result in plugin.forget_knowledge(event, target="scope")
    ]

    assert "test:session" not in plugin._store["expressions"]
    assert "test:session" not in plugin._store["jargon"]
    assert "test:session" not in plugin._buffers
    assert "test:session" not in plugin._learning_buffers
    assert "test:session" not in plugin._store["memories"]
    assert yielded == ["已清除当前聊天的表达、黑话、行为经验与回想，共 3 条。"]
    plugin._persist_store.assert_awaited_once()


def test_episodic_memory_parser_and_recall_require_evidence_and_relevance() -> None:
    result = parse_learning_result(
        """
        {
          "memories": [
            {
              "title": "确定新版发布计划",
              "summary": "大家决定周五先发布测试版，并在发布前完成回归检查。",
              "recall_cues": ["新版发布", "周五测试版", "回归检查"],
              "importance": 0.8,
              "source_ids": ["m1", "m2"]
            },
            {
              "title": "虚构回想",
              "summary": "没有真实来源。",
              "recall_cues": ["虚构"],
              "importance": 1,
              "source_ids": ["missing"]
            },
            {
              "title": "保存电话号码",
              "summary": "对方的电话号码是 13800138000。",
              "recall_cues": ["电话"],
              "importance": 1,
              "source_ids": ["m1"]
            }
          ]
        }
        """,
        valid_source_ids={"m1", "m2"},
        valid_speaker_ids={"test-platform:user-1"},
        learn_expressions=False,
        learn_jargon=False,
        learn_profiles=False,
        learn_memories=True,
    )

    assert len(result["memories"]) == 1
    selected = select_episodic_memories(
        result["memories"],
        context_text="新版发布前的回归检查做完了吗",
    )
    unrelated = select_episodic_memories(
        result["memories"],
        context_text="今天午饭吃什么",
    )
    vague_recall = select_episodic_memories(
        result["memories"],
        context_text="上次那个事情后来怎么样了",
    )

    assert selected[0]["title"] == "确定新版发布计划"
    assert unrelated == []
    assert vague_recall[0]["title"] == "确定新版发布计划"


def test_learning_prompt_contains_valid_json_contract() -> None:
    system_prompt, _user_prompt = build_learning_prompts(
        messages=[],
        learn_expressions=True,
        learn_jargon=True,
        learn_profiles=True,
        learn_memories=True,
    )
    json_contract = system_prompt.split(
        "只输出下面结构的 JSON 对象，不输出 Markdown：\n",
        1,
    )[1]

    assert set(json.loads(json_contract)) == {
        "expressions",
        "jargon",
        "profiles",
        "memories",
    }


def test_behavior_feedback_parser_requires_followup_evidence() -> None:
    accepted = parse_behavior_feedback_result(
        """
        {
          "status": "success",
          "score_delta": 2,
          "situation": "群友询问方案看法",
          "action": "简短表态并给出一个具体理由",
          "outcome": "有人继续讨论这个理由",
          "lesson": "开放讨论中可以先给短观点，给别人留接话空间",
          "keywords": ["方案", "看法"],
          "source_ids": ["m3"]
        }
        """,
        valid_feedback_source_ids={"m3"},
    )
    rejected = parse_behavior_feedback_result(
        """
        {
          "status": "success",
          "score_delta": 0.8,
          "situation": "无后续证据",
          "action": "随意插话",
          "outcome": "声称有效",
          "lesson": "不应保存",
          "keywords": [],
          "source_ids": ["before-reply"]
        }
        """,
        valid_feedback_source_ids={"m3"},
    )

    assert accepted is not None
    assert accepted["score_delta"] == 1.0
    assert accepted["source_ids"] == ["m3"]
    assert rejected is None


def test_behavior_selector_prefers_keyword_match_and_skips_disabled() -> None:
    selected = select_behavior_experiences(
        [
            {
                "id": "b1",
                "situation": "讨论游戏",
                "action": "接梗",
                "lesson": "有共同语境时适合",
                "keywords": ["游戏"],
                "score": 0.2,
                "enabled": True,
            },
            {
                "id": "b2",
                "situation": "讨论方案",
                "action": "短观点",
                "lesson": "开放讨论时适合",
                "keywords": ["方案"],
                "score": 0.1,
                "enabled": True,
            },
            {
                "id": "b3",
                "situation": "失败经验",
                "action": "抢话",
                "lesson": "不要使用",
                "keywords": ["方案"],
                "score": 3,
                "enabled": False,
            },
        ],
        context_text="你们觉得这个方案怎么样",
    )

    assert [item["id"] for item in selected] == ["b2"]


@pytest.mark.asyncio
async def test_behavior_feedback_persists_generalized_experience() -> None:
    context = MagicMock()
    context.llm_generate = AsyncMock(
        return_value=SimpleNamespace(
            completion_text="""
            {
              "status": "success",
              "score_delta": 0.7,
              "situation": "群友邀请大家评价方案",
              "action": "简短表达观点并给出一个理由",
              "outcome": "其他人继续讨论了这个理由",
              "lesson": "开放讨论中先给短观点，不要一次说完所有角度",
              "keywords": ["方案", "评价"],
              "source_ids": ["m3", "m4"]
            }
            """
        )
    )
    plugin = MaiHumanizerPlugin(
        context,
        {
            "behavior_feedback_provider_id": "feedback-provider",
            "behavior_learning_enabled": True,
        },
    )
    plugin._persist_store = AsyncMock()
    plugin._feedback_scopes.add("test:session")
    pending = {
        "planner_reason": "话题在等待回应",
        "reply_guide": "给简短看法",
        "tone": "自然",
        "bot_reply": "我觉得方向对，不过先把边界定清楚会更稳。",
        "context_messages": [],
        "feedback_messages": [
            {"source_id": "m3", "speaker_name": "Bob", "text": "边界是得先定"},
            {"source_id": "m4", "speaker_name": "Alice", "text": "那先列一下"},
        ],
    }

    learned = await plugin._run_behavior_feedback("test:session", pending)

    assert learned is True
    record = plugin._store["behaviors"]["test:session"][0]
    assert record["id"] == "b1"
    assert record["score"] == 0.7
    assert record["success_count"] == 1
    assert record["lesson"] == "开放讨论中先给短观点，不要一次说完所有角度"
    assert "test:session" not in plugin._feedback_scopes
    plugin._persist_store.assert_awaited_once()


@pytest.mark.asyncio
async def test_after_send_starts_behavior_feedback_window() -> None:
    plugin = MaiHumanizerPlugin(
        MagicMock(),
        {"behavior_learning_enabled": True},
    )
    event = _event(private=False)
    extras = {
        "_mai_autonomous_reply": True,
        "_mai_planner_reason": "等待回应",
        "_mai_planner_reply_guide": "简短接话",
        "_mai_planner_tone": "轻松",
        "_mai_planner_context": [{"source_id": "m1", "text": "有人吗"}],
    }
    event.get_extra.side_effect = lambda key, default=None: extras.get(key, default)
    event.get_result.return_value = MessageEventResult().message("我在，怎么啦")

    await plugin.track_autonomous_reply(event)

    pending = plugin._pending_behavior_feedback["test:session"]
    assert pending["bot_reply"] == "我在，怎么啦"
    assert pending["feedback_messages"] == []


@pytest.mark.asyncio
async def test_followup_message_schedules_behavior_feedback() -> None:
    plugin = MaiHumanizerPlugin(
        MagicMock(),
        {
            "behavior_learning_enabled": True,
            "behavior_feedback_messages": 1,
        },
    )
    plugin._run_behavior_feedback = AsyncMock(return_value=True)
    plugin._pending_behavior_feedback["test:session"] = {
        "bot_reply": "先定一下边界吧",
        "feedback_messages": [],
    }
    event = _event(private=False)
    event.get_message_str.return_value = "对，先列出范围"
    event.get_extra.return_value = {}

    await plugin._record_message(event)
    await asyncio.sleep(0)

    plugin._run_behavior_feedback.assert_awaited_once()
    assert "test:session" not in plugin._pending_behavior_feedback


@pytest.mark.asyncio
async def test_focus_uses_one_slot_and_explicit_wake_can_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 100.0
    monkeypatch.setattr(
        f"{MaiHumanizerPlugin.__module__}.time.monotonic",
        lambda: now,
    )
    plugin = MaiHumanizerPlugin(MagicMock(), {"focus_hold_seconds": 120})

    assert await plugin._claim_focus("scope-a") is True
    assert await plugin._claim_focus("scope-b") is False
    assert await plugin._claim_focus("scope-b", force=True) is True
    assert plugin._focused_scope == "scope-b"

    now = 221.0
    assert await plugin._claim_focus("scope-c") is True
    assert plugin._focused_scope == "scope-c"
