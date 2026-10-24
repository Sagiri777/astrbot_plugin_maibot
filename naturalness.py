"""Local, deterministic helpers for casual conversation and visible replies."""

from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Any

_STRUCTURED_REQUEST_RE = re.compile(
    r"```|\bjson\b|\bxml\b|\bhtml\b|\bsql\b|"
    r"代码|编程|调试|报错|堆栈|教程|步骤|详细|逐条|表格|列表|"
    r"总结|翻译|润色|论文|报告|分析|对比|攻略|格式|输出为|按.*格式",
    re.IGNORECASE,
)
_ROLEPLAY_REQUEST_RE = re.compile(
    r"角色扮演|扮演|人设|语c|跑团|动作描写|神态描写|剧情|"
    r"\brole\s*play\b|\brp\b",
    re.IGNORECASE,
)
_VENTING_RE = re.compile(r"好累|难受|好烦|烦死|崩溃|不开心|郁闷|气死|委屈|焦虑")
_PLAYFUL_RE = re.compile(r"哈哈|笑死|乐死|离谱|绷不住|\b草\b|xswl", re.IGNORECASE)
_CLOSING_RE = re.compile(
    r"(?:晚安|睡了|睡觉去了|去睡了|先睡|下线了|先下了|去忙了|先忙了|忙去了|"
    r"先不聊|回头(?:再)?聊|下次(?:再)?聊|改天(?:再)?聊|先这样|"
    r"就到这(?:里)?吧|先撤了|溜了溜了)(?=$|[吧了啊呀~～，,。！!])"
)
_LOW_INFORMATION_RE = re.compile(
    r"(?:哈+|呵+|嘿+|嗯+|哦+|噢+|呃+|啊+|行|好|好的|好嘞|可以|是的|对|"
    r"确实|笑死|乐|草|绝了|离谱|牛|牛逼|666+|1+|\+1|[？?！!。.…~～]+)",
    re.IGNORECASE,
)
_DIRECT_REQUEST_RE = re.compile(
    r"[？?]|为什么|怎么办|如何|怎么|请教|帮我|你能|你会|你知道|讲一下|说说|看看"
)
_STAGE_ACTION_RE = re.compile(
    r"笑了笑|微笑|轻笑|点头|摇头|眨眼|歪头|叹气|皱眉|耸肩|"
    r"拍手|挥手|靠近|后退|走到|走来|站起|坐下|蹲下|"
    r"摸了摸|拍了拍|揉了揉|戳了戳|捏了捏|抱起|"
    r"轻声说|小声说|小声嘀咕|心里想|心想|内心|旁白",
    re.IGNORECASE,
)
_BRACKETED_LINE_RE = re.compile(
    r"^\s*(?:\*([^*\n]+)\*|\(([^()\n]+)\)|（([^（）\n]+)）|【([^【】\n]+)】)\s*$"
)
_STAGE_DIRECTION_RE = re.compile(
    r"\*[^*\n]{1,40}\*|（[^（）\n]{1,40}）|\([^()\n]{1,40}\)"
)
_BOILERPLATE_SUFFIX_RE = re.compile(
    r"(?:\n|^)\s*(?:"
    r"希望(?:这|以上|这些)?(?:能)?对你有帮助[。！!]*|"
    r"如果(?:你)?还有其他问题[，,]?(?:欢迎)?(?:随时)?(?:告诉|问|找)我[。！!]*|"
    r"如果(?:你)?还有(?:其他)?问题[，,]?随时(?:告诉|问|找)我[。！!]*|"
    r"需要我进一步(?:解释|说明|展开)吗[？?]*"
    r")\s*$",
    re.IGNORECASE,
)


def is_structured_request(text: str) -> bool:
    """Return whether casual-speech shortening could damage the requested task."""
    return bool(_STRUCTURED_REQUEST_RE.search(str(text or "")))


def is_roleplay_request(text: str) -> bool:
    """Explicit roleplay keeps its requested action and narration conventions."""
    return bool(_ROLEPLAY_REQUEST_RE.search(str(text or "")))


def is_conversation_closing(text: str) -> bool:
    """Recognize an explicit goodbye without treating ordinary busy talk as one."""
    return bool(_CLOSING_RE.search(str(text or "")))


def _message_information_score(text: str) -> float:
    """Estimate whether a short group message carries something worth answering."""
    text = str(text or "").strip()
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return 0.0
    unique_chars = len(set(compact))
    word_chars = len(re.findall(r"[\w\u4e00-\u9fff]", compact))
    sentence_markers = len(re.findall(r"[。！？.!?;；:\n,，]", text))
    return round(
        min(len(compact), 48) / 48 * 0.42
        + min(unique_chars, 24) / 24 * 0.22
        + min(word_chars, 40) / 40 * 0.18
        + min(sentence_markers, 4) * 0.08,
        4,
    )


def select_conversation_focus(
    recent_messages: list[dict[str, Any]],
    *,
    current_message_id: str,
    current_addressing: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Choose a nearby substantive message only after a low-information trigger.

    Explicit mentions, quotes, questions and normal-length messages always keep the
    latest message as focus. This prevents an old topic from overriding a new request.
    """
    human_messages = [
        item
        for item in recent_messages
        if item.get("role", "user") == "user"
        and str(item.get("text", "") or "").strip()
    ]
    current = next(
        (
            item
            for item in reversed(human_messages)
            if item.get("source_id") == current_message_id
        ),
        None,
    )
    if current is None:
        return None
    addressing = current_addressing or {}
    current_text = str(current.get("text", "") or "").strip()
    if (
        addressing.get("addresses_bot")
        or current.get("reply_to_id")
        or is_structured_request(current_text)
        or is_roleplay_request(current_text)
        or _DIRECT_REQUEST_RE.search(current_text)
        or not _LOW_INFORMATION_RE.fullmatch(current_text)
    ):
        return current

    candidates = human_messages[-4:-1]
    if not candidates:
        return current
    scored: list[tuple[float, dict[str, Any]]] = []
    for offset, item in enumerate(reversed(candidates)):
        recency_bonus = max(0.0, 0.12 - offset * 0.04)
        scored.append(
            (
                _message_information_score(str(item.get("text", "") or ""))
                + recency_bonus,
                item,
            )
        )
    score, selected = max(scored, key=lambda pair: pair[0])
    current_score = _message_information_score(current_text)
    return selected if score >= 0.45 and score >= current_score + 0.35 else current


def _is_stage_direction(content: str) -> bool:
    content = content.strip()
    return bool(content and len(content) <= 40 and _STAGE_ACTION_RE.search(content))


def _remove_stage_direction(match: re.Match[str]) -> str:
    token = match.group(0)
    content = token.strip("*（）() ")
    return "" if _is_stage_direction(content) else token


def clean_visible_reply(text: str, *, current_message: str = "") -> str:
    """Conservatively remove roleplay directions and stock casual closers.

    Code, structured tasks and explicit roleplay are deliberately left untouched.
    The original text is returned if cleanup would accidentally erase everything.
    """
    original = str(text or "")
    if (
        not original.strip()
        or "```" in original
        or is_structured_request(current_message)
        or is_roleplay_request(current_message)
    ):
        return original

    cleaned_lines: list[str] = []
    for line in original.splitlines():
        match = _BRACKETED_LINE_RE.match(line)
        if match:
            content = next((part for part in match.groups() if part is not None), "")
            if _is_stage_direction(content):
                continue
        cleaned_lines.append(line)
    cleaned = _STAGE_DIRECTION_RE.sub(_remove_stage_direction, "\n".join(cleaned_lines))
    cleaned = _BOILERPLATE_SUFFIX_RE.sub("", cleaned).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned if cleaned.strip() else original


def _repeated_edge(texts: list[str], *, opening: bool) -> str:
    normalized = [re.sub(r"\s+", "", text).strip("，。！？!?…") for text in texts]
    normalized = [text for text in normalized if len(text) >= 4][-3:]
    if len(normalized) < 2:
        return ""
    sizes = (4, 6, 8) if opening else (6, 8, 10)
    for size in sizes:
        edges = [text[:size] if opening else text[-size:] for text in normalized]
        fragment, count = Counter(edges).most_common(1)[0]
        if count >= 2 and fragment:
            return fragment
    return ""


def _repeated_fragments(texts: list[str], *, excluded: set[str]) -> list[str]:
    """Find recurring content phrases across separate bot messages, not just edges."""
    normalized = [
        re.sub(r"[^\w\u4e00-\u9fff]+", "", text).casefold()
        for text in texts
        if str(text or "").strip()
    ][-3:]
    if len(normalized) < 2:
        return []
    for size in (8, 6, 4):
        occurrences: Counter[str] = Counter()
        for text in normalized:
            occurrences.update(
                {text[index : index + size] for index in range(len(text) - size + 1)}
            )
        candidates = [
            fragment
            for fragment, count in occurrences.most_common()
            if count >= 2
            and not any(
                fragment in edge or edge in fragment for edge in excluded if edge
            )
        ]
        selected: list[str] = []
        for fragment in candidates:
            if any(
                fragment in existing or existing in fragment for existing in selected
            ):
                continue
            selected.append(fragment)
            if len(selected) == 2:
                return selected
        if selected:
            return selected
    return []


def build_naturalness_hint(
    *,
    current_message: str,
    recent_messages: list[dict[str, Any]],
    is_private_chat: bool,
) -> str:
    """Build a bounded per-turn hint from visible register and actual bot speech."""
    current = str(current_message or "").strip()
    if not current or is_structured_request(current) or is_roleplay_request(current):
        return ""

    human_texts = [
        str(item.get("text", "") or "").strip()
        for item in recent_messages
        if item.get("role", "user") == "user"
        and str(item.get("text", "") or "").strip()
    ]
    bot_texts = [
        str(item.get("text", "") or "").strip()
        for item in recent_messages
        if item.get("role") == "assistant" and str(item.get("text", "") or "").strip()
    ]
    if not human_texts and not bot_texts and not is_conversation_closing(current):
        return ""
    hints: list[str] = []

    if is_conversation_closing(current):
        hints.append("对方正在结束这轮聊天；自然收尾即可，不要追问、留话头或另起话题。")
    elif _VENTING_RE.search(current):
        hints.append("这轮先接住感受，别急着讲道理、列建议或追问细节。")
    elif _PLAYFUL_RE.search(current) and len(current) <= 100:
        hints.append("这轮是随口玩梗，接得住就短一点，别解释笑点或顺势上课。")
    elif len(current) <= 24 and not re.search(
        r"为什么|怎么办|如何|怎么弄|请教", current
    ):
        hints.append(
            "这轮更像随口接话，优先直接给一个自然反应，不必补全背景或主动收尾。"
        )

    if len(human_texts) >= 3:
        lengths = [len(text) for text in human_texts[-10:]]
        if statistics.median(lengths) <= 24:
            hints.append(
                "近期真人多用短句；没有复杂任务时跟随这个节奏，别突然写成多段小作文。"
            )

    repeated_opening = _repeated_edge(bot_texts, opening=True)
    repeated_closing = _repeated_edge(bot_texts, opening=False)
    if repeated_opening:
        hints.append(
            f"最近自己多次用“{repeated_opening}”开头，本轮换个自然切入，不要复读这个句式。"
        )
    if repeated_closing and repeated_closing != repeated_opening:
        hints.append(
            f"最近自己多次用“{repeated_closing}”收尾，本轮不要沿用这个固定收尾。"
        )
    repeated_fragments = _repeated_fragments(
        bot_texts,
        excluded={repeated_opening, repeated_closing},
    )
    if repeated_fragments:
        hints.append(
            "最近自己反复说过"
            + "、".join(f"“{fragment}”" for fragment in repeated_fragments)
            + "；本轮不要原样复读这些口癖或句式，优先换一个真实的新反应。"
        )
    recent_bot_questions = [text for text in bot_texts[-2:] if text]
    if len(recent_bot_questions) == 2 and all(
        text.rstrip().endswith(("？", "?")) for text in recent_bot_questions
    ):
        hints.append("最近两条都以问句收尾；本轮不要习惯性再抛问题，把话说完就可以停。")

    if not hints:
        return ""
    scope = "私聊" if is_private_chat else "群聊"
    return (
        f"【本轮{scope}自然度】\n"
        "以下是根据当前消息和近期真实发言生成的局部提示，不影响事实与任务要求：\n- "
        + "\n- ".join(hints[:6])
    )


def delivered_reply_streak(decision_history: list[dict[str, Any]]) -> int:
    """Count consecutive delivered autonomous replies since the last real pause."""
    streak = 0
    for decision in reversed(decision_history):
        if decision.get("action") != "reply" or not decision.get("delivered"):
            break
        streak += 1
    return streak
