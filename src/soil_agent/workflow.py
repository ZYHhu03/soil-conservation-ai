"""LangGraph workflow and local fallback runner for the ReAct Agent."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from soil_rag.context import ContextAssembler
from soil_rag.generation import ExtractiveAnswerGenerator
from soil_rag.schemas import Chunk, Evidence, QueryPlan, SearchResult
from soil_rag.query import QueryPlanner

from .memory import EpisodicMemory, MemoryStore, SemanticMemory, WorkingMemory
from .planner import QueryDecomposer, ReActPlanner
from .state import AgentState, Claim, SubTask, ToolCallRecord, initial_state, utc_now
from .tools import ToolRegistry, ToolResult, idempotency_key
from .verification import ClaimExtractor, ClaimLevelVerifier


class AgentRuntime:
    """State machine dependencies plus node implementations."""

    def __init__(
        self,
        rag_pipeline,
        graph=None,
        chunks_by_id: dict[str, Chunk] | None = None,
        tool_registry: ToolRegistry | None = None,
        memory_store: MemoryStore | None = None,
        planner: QueryPlanner | None = None,
        decomposer: QueryDecomposer | None = None,
        max_tool_calls: int = 8,
        max_retries: int = 1,
    ):
        self.rag_pipeline = rag_pipeline
        self.graph = graph
        self.chunks_by_id = chunks_by_id or {}
        self.memory_store = memory_store or MemoryStore()
        self.query_planner = planner or QueryPlanner()
        self.decomposer = decomposer or QueryDecomposer()
        self.react_planner = ReActPlanner(max_tool_calls=max_tool_calls, max_retries=max_retries)
        self.max_tool_calls = max_tool_calls
        self.max_retries = max_retries
        self.tool_registry = tool_registry or self._default_tools()
        self.working_memories: dict[str, WorkingMemory] = {}
        self._idempotency_cache: dict[str, ToolResult] = {}
        self.answer_generator = ExtractiveAnswerGenerator()
        self.claim_extractor = ClaimExtractor()
        self.claim_verifier = ClaimLevelVerifier(graph=graph)
        self.context_assembler = ContextAssembler(max_chars=6000)

    def _default_tools(self) -> ToolRegistry:
        from .tools import KnowledgeGraphTool, MemoryLookupTool, RagSearchTool

        tools = [RagSearchTool(self.rag_pipeline), MemoryLookupTool(self.memory_store)]
        if self.graph is not None:
            tools.append(KnowledgeGraphTool(self.graph))
        return ToolRegistry(tools)

    def node_intent(self, state: AgentState) -> AgentState:
        plan = self.query_planner.plan(state["user_query"])
        state["intent"] = plan.intent.value
        state["slots"] = dict(plan.slots)
        return state

    def node_slots(self, state: AgentState) -> AgentState:
        working = self.working_memories.setdefault(state["session_id"], WorkingMemory(state["session_id"]))
        merged = working.context()
        for key, value in state.get("slots", {}).items():
            if value and value != "false":
                merged[key] = value
        state["slots"] = {key: value for key, value in merged.items() if isinstance(value, str) and value}
        return state

    def node_memory(self, state: AgentState) -> AgentState:
        state["memory_context"] = self.memory_store.context_for(state["session_id"], state["user_query"])
        return state

    def node_plan(self, state: AgentState) -> AgentState:
        tasks = [SubTask.from_dict(item) for item in state.get("subtasks", [])]
        if not tasks:
            tasks = self.decomposer.decompose(state["user_query"], state.get("intent", "other"), state.get("slots", {}))
        elif state.get("replan_requested") and not any(task.status == "pending" for task in tasks):
            retry_id = f"retry-{state.get('retry_count', 0)}"
            tasks.append(SubTask(retry_id, "verification_retry", f"重新检索并核对以下问题的证据：{state['user_query']}", "rag_search"))
            state["replan_requested"] = False
        task = self.react_planner.next_task(tasks, state.get("tool_call_count", 0))
        state["subtasks"] = [item.to_dict() for item in tasks]
        state["current_task_id"] = task.id if task else ""
        return state

    def node_act(self, state: AgentState) -> AgentState:
        task_id = state.get("current_task_id", "")
        tasks = [SubTask.from_dict(item) for item in state.get("subtasks", [])]
        task = next((item for item in tasks if item.id == task_id), None)
        if task is None:
            return state
        if state.get("tool_call_count", 0) >= state.get("max_tool_calls", self.max_tool_calls):
            task.status = "failed"
            task.error = "tool call budget exhausted"
            state["subtasks"] = [item.to_dict() for item in tasks]
            return state

        arguments = {"query": task.query, "session_id": state["session_id"], "limit": 8}
        cache_key = idempotency_key(task.tool, arguments)
        cached = self._idempotency_cache.get(cache_key)
        attempts = 0
        result: ToolResult | None = cached
        if cached is not None:
            trace = ToolCallRecord(f"cached-{cache_key[:10]}", task.id, task.tool, arguments, "deduplicated", 0, list(cached.result_ids)).to_dict()
            state.setdefault("tool_trace", []).append(trace)
        else:
            while attempts <= self.max_retries:
                attempts += 1
                state["tool_call_count"] = state.get("tool_call_count", 0) + 1
                started_at = utc_now()
                try:
                    result = self.tool_registry.invoke(task.tool, arguments)
                except Exception as exc:  # tool boundaries convert errors into fallback state
                    result = ToolResult(task.tool, False, error=str(exc))
                trace = ToolCallRecord(f"call-{state['tool_call_count']}", task.id, task.tool, arguments, "ok" if result.ok else "failed", attempts, list(result.result_ids), result.error, started_at=started_at, finished_at=utc_now()).to_dict()
                state.setdefault("tool_trace", []).append(trace)
                if result.ok:
                    self._idempotency_cache[cache_key] = result
                    break
        if result is None:
            result = ToolResult(task.tool, False, error="tool returned no result")
        if result.ok:
            task.status = "completed"
            task.result_ids.extend(result.result_ids)
            state.setdefault("evidence", []).extend(self._normalize_evidence(result))
        else:
            task.status = "failed"
            task.error = result.error
            # A failed graph/memory lookup falls back to the primary RAG tool.
            if task.tool != "rag_search" and state.get("tool_call_count", 0) < state.get("max_tool_calls", self.max_tool_calls):
                fallback_id = f"{task.id}-fallback"
                if not any(item.id == fallback_id for item in tasks):
                    tasks.append(SubTask(fallback_id, task.kind, task.query, "rag_search"))
        state["subtasks"] = [item.to_dict() for item in tasks]
        state["current_task_id"] = ""
        return state

    def _normalize_evidence(self, result: ToolResult) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for raw in result.evidence:
            item = dict(raw)
            chunk_id = str(item.get("id", ""))
            chunk = self.chunks_by_id.get(chunk_id)
            if chunk is not None:
                item.setdefault("text", chunk.text)
                item.setdefault("title", chunk.title)
                item.setdefault("document_id", chunk.document_id)
                item.setdefault("clause_no", chunk.clause_no)
                item.setdefault("metadata", dict(chunk.metadata))
            item.setdefault("tool", result.tool)
            normalized.append(item)
        return normalized

    def node_aggregate(self, state: AgentState) -> AgentState:
        unique: dict[str, dict[str, Any]] = {}
        for item in state.get("evidence", []):
            if item.get("id"):
                current = unique.get(item["id"])
                if current is None or float(item.get("score", 0.0)) > float(current.get("score", 0.0)):
                    unique[item["id"]] = item
        evidence = sorted(unique.values(), key=lambda item: float(item.get("score", 0.0)), reverse=True)
        for index, item in enumerate(evidence, 1):
            item["label"] = f"E{index}"
        state["evidence"] = evidence
        return state

    def _evidence_objects(self, state: AgentState) -> tuple[Evidence, ...]:
        objects: list[Evidence] = []
        for item in state.get("evidence", []):
            chunk = Chunk(
                id=str(item.get("id", "")),
                document_id=str(item.get("document_id", "")),
                text=str(item.get("text", "")),
                title_path=tuple(str(item.get("title", "")).split(" / ")) if item.get("title") else (),
                clause_no=str(item.get("clause_no", "")),
                metadata=item.get("metadata", {}),
            )
            result = SearchResult(chunk, float(item.get("score", 0.0)), str(item.get("tool", "agent")), int(item.get("rank", 0)), tuple(item.get("graph_path", ())))
            objects.append(Evidence(str(item.get("label", f"E{len(objects) + 1}")), result))
        return tuple(objects)

    def node_generate(self, state: AgentState) -> AgentState:
        plan = self.query_planner.plan(state["user_query"])
        evidence = self._evidence_objects(state)
        context, _ = self.context_assembler.assemble([item.result for item in evidence])
        generated = self.answer_generator.generate(plan, context, evidence)
        state["draft_answer"] = generated.text
        state["answer"] = generated.text
        state["abstained"] = generated.refused
        state["citations"] = [{"label": label, "chunk_id": next((item.result.chunk.id for item in evidence if item.label == label), "")} for label in self._citation_labels(generated.text)]
        return state

    @staticmethod
    def _citation_labels(text: str) -> list[str]:
        import re

        return list(dict.fromkeys(re.findall(r"\[(E\d+)\]", text)))

    def node_claims(self, state: AgentState) -> AgentState:
        claims = self.claim_extractor.extract(state.get("draft_answer", ""))
        state["claims"] = [claim.to_dict() for claim in claims]
        return state

    def node_verify(self, state: AgentState) -> AgentState:
        claims = [Claim(**item) for item in state.get("claims", [])]
        summary = self.claim_verifier.verify(claims, state.get("evidence", []))
        state["claim_checks"] = [check.to_dict() for check in summary.checks]
        state["verification"] = summary.to_dict()
        if summary.valid:
            state["status"] = "verified"
        elif summary.needs_retry and state.get("retry_count", 0) < state.get("max_retries", self.max_retries) and state.get("tool_call_count", 0) < state.get("max_tool_calls", self.max_tool_calls):
            state["retry_count"] = state.get("retry_count", 0) + 1
            state["replan_requested"] = True
        else:
            state["abstained"] = True
            state["answer"] = "暂无可靠依据。"
            state["status"] = "abstained"
        return state

    def node_finalize(self, state: AgentState) -> AgentState:
        session_id = state["session_id"]
        working = self.working_memories.setdefault(session_id, WorkingMemory(session_id))
        working.update(state.get("slots", {}), [str(item.get("id", "")) for item in state.get("evidence", [])[:5]])
        answer = state.get("answer", "暂无可靠依据。")
        source_ids = [str(item.get("id", "")) for item in state.get("evidence", [])]
        self.memory_store.add_episode(EpisodicMemory(f"episode-{len(self.memory_store.episodes) + 1}", session_id, state["user_query"], answer, source_ids=source_ids))
        verification = state.get("verification", {})
        if verification.get("valid"):
            self.memory_store.add_semantic(SemanticMemory(
                id=f"semantic-{len(self.memory_store.semantics) + 1}",
                subject=state["user_query"],
                content=answer,
                source_ids=source_ids,
                confidence=float(verification.get("coverage", 0.0)),
                conditions=dict(state.get("slots", {})),
            ))
            state["status"] = "completed"
        elif state.get("status") != "abstained":
            state["status"] = "abstained"
        return state

    def route_after_plan(self, state: AgentState) -> str:
        return "act" if state.get("current_task_id") else "aggregate"

    def route_after_act(self, state: AgentState) -> str:
        tasks = [SubTask.from_dict(item) for item in state.get("subtasks", [])]
        return "plan" if self.react_planner.has_pending(tasks) and state.get("tool_call_count", 0) < state.get("max_tool_calls", self.max_tool_calls) else "aggregate"

    def route_after_verify(self, state: AgentState) -> str:
        return "plan" if state.get("replan_requested") else "finalize"

    def run_local(self, query: str, session_id: str = "default") -> AgentState:
        state = initial_state(query, session_id, self.max_tool_calls, self.max_retries)
        self.node_intent(state)
        self.node_slots(state)
        self.node_memory(state)
        while True:
            self.node_plan(state)
            if self.route_after_plan(state) == "act":
                self.node_act(state)
                continue
            self.node_aggregate(state)
            self.node_generate(state)
            self.node_claims(state)
            self.node_verify(state)
            if self.route_after_verify(state) == "plan":
                continue
            self.node_finalize(state)
            return state

    def compile_langgraph(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:
            raise RuntimeError("compile_langgraph requires langgraph") from exc

        graph = StateGraph(AgentState)
        graph.add_node("intent", self.node_intent)
        graph.add_node("slots", self.node_slots)
        graph.add_node("memory", self.node_memory)
        graph.add_node("plan", self.node_plan)
        graph.add_node("act", self.node_act)
        graph.add_node("aggregate", self.node_aggregate)
        graph.add_node("generate", self.node_generate)
        graph.add_node("claims", self.node_claims)
        graph.add_node("verify", self.node_verify)
        graph.add_node("finalize", self.node_finalize)
        graph.set_entry_point("intent")
        graph.add_edge("intent", "slots")
        graph.add_edge("slots", "memory")
        graph.add_edge("memory", "plan")
        graph.add_conditional_edges("plan", self.route_after_plan, {"act": "act", "aggregate": "aggregate"})
        graph.add_conditional_edges("act", self.route_after_act, {"plan": "plan", "aggregate": "aggregate"})
        graph.add_edge("aggregate", "generate")
        graph.add_edge("generate", "claims")
        graph.add_edge("claims", "verify")
        graph.add_conditional_edges("verify", self.route_after_verify, {"plan": "plan", "finalize": "finalize"})
        graph.add_edge("finalize", END)
        return graph.compile()

    def invoke(self, query: str, session_id: str = "default", use_langgraph: bool = False) -> AgentState:
        if use_langgraph:
            workflow = self.compile_langgraph()
            return workflow.invoke(initial_state(query, session_id, self.max_tool_calls, self.max_retries))
        return self.run_local(query, session_id)
