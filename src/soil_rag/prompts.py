"""Prompt templates used by the answer-generation stage."""

from __future__ import annotations

from .schemas import QueryPlan


SYSTEM_PROMPT = """你是水土保持领域问答助手。只使用证据区中的信息回答，不补造政策名称、条款编号、日期、地区要求或审批结论。\n当证据不足、来源冲突或时效无法确认时，明确写出“暂无可靠依据”，并列出需要补充的条件。"""

USER_TEMPLATE = """用户问题：{query}\n\n检索请求：{rewritten_query}\n意图：{intent}\n槽位：{slots}\n\n证据区：\n{context}\n\n输出要求：\n1. 先给出结论；\n2. 说明适用条件或办理阶段；\n3. 每个关键事实后使用 [E#] 标注证据；\n4. 无法从证据确认的内容不得猜测；\n5. 证据不足时输出“暂无可靠依据”，并说明需要补充的信息。"""


def build_prompt(plan: QueryPlan, context: str) -> str:
    return USER_TEMPLATE.format(
        query=plan.original_query,
        rewritten_query=plan.rewritten_query,
        intent=plan.intent.value,
        slots=dict(plan.slots),
        context=context,
    )
