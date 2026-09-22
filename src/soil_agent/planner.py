"""Query decomposition and bounded ReAct action planning."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from soil_rag.query import QueryPlanner

from .state import SubTask


PLANNING_PROMPT = """你是水土保持问答 Agent 的任务规划器。将用户复合问题拆成最少的、可独立检索的子任务。\n每个子任务只能选择一个工具：rag_search、graph_lookup、memory_lookup。\n只返回 JSON 数组，每项包含 id、kind、query、tool、depends_on。\n用户问题：{query}\n意图：{intent}\n槽位：{slots}"""


@dataclass
class QueryDecomposer:
    """Rule-first decomposer with an optional LangChain LLM JSON planner."""

    llm: Any = None

    def decompose(self, query: str, intent: str, slots: dict[str, str]) -> list[SubTask]:
        if self.llm is not None:
            try:
                prompt = PLANNING_PROMPT.format(query=query, intent=intent, slots=slots)
                response = self.llm.invoke(prompt)
                raw = response.content if hasattr(response, "content") else str(response)
                parsed = json.loads(raw[raw.find("[") : raw.rfind("]") + 1])
                tasks = [SubTask(**item) for item in parsed]
                if tasks:
                    return self._deduplicate(tasks)
            except (ValueError, TypeError, json.JSONDecodeError, AttributeError):
                pass
        return self._rule_decompose(query, intent, slots)

    def _rule_decompose(self, query: str, intent: str, slots: dict[str, str]) -> list[SubTask]:
        tasks: list[SubTask] = []
        normalized = query.strip()
        has_policy = any(word in normalized for word in ("政策", "依据", "条款", "规定", "哪条"))
        has_process = any(word in normalized for word in ("审批", "流程", "办理", "是否需要", "阶段"))
        has_material = any(word in normalized for word in ("材料", "资料", "清单", "提交"))
        has_graph = any(word in normalized for word in ("关联", "依据哪条", "哪个阶段", "适用地区", "关系"))

        if has_policy or intent == "policy_query":
            tasks.append(SubTask("policy-1", "policy", f"检索与以下问题相关的政策、条款和适用条件：{normalized}", "rag_search"))
        if has_process or intent == "process_consultation":
            tasks.append(SubTask("process-1", "process", f"检索以下问题对应的审批流程、前置条件和办理节点：{normalized}", "rag_search"))
        if has_material or intent == "material_list":
            tasks.append(SubTask("material-1", "material", f"检索以下问题对应的基础资料和申报材料：{normalized}", "rag_search"))
        if has_graph or len(tasks) >= 2:
            tasks.append(SubTask("graph-1", "relation", f"查询以下问题中的实体、条款和流程关系：{normalized}", "graph_lookup"))
        if not tasks:
            tasks.append(SubTask("general-1", "general", normalized, "rag_search"))
        if slots:
            qualifier = "；".join(f"{key}={value}" for key, value in slots.items() if value and value != "false")
            if qualifier:
                tasks = [SubTask(task.id, task.kind, f"{task.query}（{qualifier}）", task.tool, task.status, task.attempts, task.depends_on, task.result_ids, task.error) for task in tasks]
        return self._deduplicate(tasks)

    @staticmethod
    def _deduplicate(tasks: list[SubTask]) -> list[SubTask]:
        seen: set[tuple[str, str]] = set()
        result: list[SubTask] = []
        for task in tasks:
            key = (task.tool, re.sub(r"\s+", "", task.query).lower())
            if key in seen:
                continue
            seen.add(key)
            task.id = task.id or f"task-{len(result) + 1}"
            result.append(task)
        return result


class ReActPlanner:
    """Chooses the next pending task under a strict call budget."""

    def __init__(self, max_tool_calls: int = 8, max_retries: int = 1):
        self.max_tool_calls = max_tool_calls
        self.max_retries = max_retries

    def next_task(self, tasks: list[SubTask], tool_call_count: int) -> SubTask | None:
        if tool_call_count >= self.max_tool_calls:
            return None
        for task in tasks:
            if task.status == "pending":
                task.status = "running"
                task.attempts += 1
                return task
        return None

    def has_pending(self, tasks: list[SubTask]) -> bool:
        return any(task.status in {"pending", "running"} for task in tasks)
