# ReAct Agent architecture

## State graph

```text
AgentState
  ├─ user_query / session_id
  ├─ intent / slots / memory_context
  ├─ subtasks / current_task_id / tool_trace
  ├─ evidence / claims / claim_checks
  ├─ draft_answer / citations / verification
  └─ retry_count / tool_call_count / status

intent → slots → memory → plan ⇄ act → aggregate → generate → claims → verify
                                      ↑                         │
                                      └──── retry / replan ─────┘
                                                              ↓
                                                           finalize
```

`AgentRuntime.compile_langgraph()` uses the same node methods to construct a LangGraph `StateGraph`. `run_local()` executes the identical transitions without importing LangGraph, which keeps unit tests deterministic.

## ReAct planning

`QueryDecomposer` first attempts a JSON-only planning prompt through an optional LangChain LLM. Invalid or unavailable LLM output falls back to deterministic decomposition based on intent and slots. Each `SubTask` has an id, kind, query, tool, dependency list, status and attempt count.

`ReActPlanner` chooses one pending task at a time. `AgentRuntime` enforces a maximum tool-call budget, retries failed calls, writes every attempt to `tool_trace`, deduplicates calls with an idempotency key and creates a `rag_search` fallback when graph or memory tools fail.

## MCP tool boundary

`MCPToolServer` implements the MCP JSON-RPC method shape:

- `tools/list` returns tool names and input schemas;
- `tools/call` invokes a registered typed tool and returns structured content.

`MCPToolClient` can use an in-process transport, a stdio bridge or an external MCP transport without changing the Agent state machine. `scripts/mcp_server.py` provides a line-delimited JSON bridge for local integration.

## Claim-level verification

The draft answer is split into atomic claims. Each claim is matched to evidence using token overlap, exact normalized text and graph entity-to-chunk relations. Numeric conflicts and current-version claims without source version metadata are marked as conflicts. The verifier returns per-claim support, confidence, evidence ids, coverage and a retry/abstention decision.

When verification fails and the retry budget remains, the workflow creates a verification-retry subtask and loops through retrieval again. When the budget is exhausted, the final answer is `暂无可靠依据。` and the state status is `abstained`.

## Memory rails

- Working memory is session-scoped and mutable; it stores project type, region, approval stage, confirmed entities and missing slots.
- Episodic memory stores each query/answer event, source ids, timestamp and feedback.
- Semantic memory stores verified answers with source ids, confidence, applicability conditions and validity timestamps. Semantic records are only written during `finalize` after a valid claim verification.

`FileMemoryStore` persists episodic and semantic records as JSONL, while the same interface can be backed by a database or vector store in deployment.
