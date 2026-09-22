from soil_rag.embeddings import HashEmbedding
from soil_rag.graph import KnowledgeGraph
from soil_rag.pipeline import RAGPipeline
from soil_rag.retrieval import BM25Retriever, GraphRetriever, HybridRetriever, VectorRetriever
from soil_rag.schemas import Chunk
from soil_rag.stores import InMemoryVectorStore
from soil_agent.mcp import MCPToolClient, MCPToolServer
from soil_agent.memory import MemoryStore
from soil_agent.tools import ToolRegistry
from soil_agent.verification import ClaimExtractor, ClaimLevelVerifier
from soil_agent.workflow import AgentRuntime


def build_runtime():
    chunks = [
        Chunk("c1", "policy", "第一条 项目是否需要编制水土保持方案，应结合项目类型、建设规模和项目所在地判断。", ("审批流程",), "第一条"),
        Chunk("c2", "policy", "第二条 制作方案前应准备项目位置、建设内容、用地范围和施工组织资料。", ("材料清单",), "第二条"),
        Chunk("c3", "policy", "第三条 方案完成内部校核后，按照项目所在地流程提交主管部门审查。", ("审批流程",), "第三条"),
    ]
    graph = KnowledgeGraph()
    graph.add_chunks(chunks)
    store = InMemoryVectorStore(HashEmbedding(128))
    store.add(chunks)
    retriever = HybridRetriever(BM25Retriever(chunks), VectorRetriever(store), GraphRetriever(graph, {item.id: item for item in chunks}))
    return AgentRuntime(RAGPipeline(retriever), graph=graph, chunks_by_id={item.id: item for item in chunks}, memory_store=MemoryStore()), graph


def test_react_agent_decomposes_and_records_tool_trace():
    runtime, _ = build_runtime()
    state = runtime.invoke("项目是否需要审批、需要哪些材料、依据哪条政策？", session_id="s1")
    assert state["intent"]
    assert len(state["subtasks"]) >= 2
    assert state["tool_trace"]
    assert state["status"] in {"completed", "abstained"}
    assert state["tool_call_count"] <= state["max_tool_calls"]


def test_claim_verification_uses_evidence_and_rejects_numeric_conflict():
    _, graph = build_runtime()
    claims = ClaimExtractor().extract("制作方案前应准备项目位置和用地范围。[E1]")
    evidence = [{"id": "c2", "text": "制作方案前应准备项目位置、建设内容、用地范围和施工组织资料。"}]
    summary = ClaimLevelVerifier(graph).verify(claims, evidence)
    assert summary.valid
    conflict_claims = ClaimExtractor().extract("第一条规定规模为 500 公顷。[E1]")
    conflict = ClaimLevelVerifier(graph).verify(conflict_claims, [{"id": "c1", "text": "第一条规定规模为 50 公顷。"}])
    assert conflict.abstain


def test_mcp_tools_list_and_call():
    runtime, _ = build_runtime()
    server = MCPToolServer(runtime.tool_registry)
    client = MCPToolClient(server.handle)
    names = {item["name"] for item in client.list_tools()}
    assert "rag_search" in names
    result = client.call_tool("rag_search", {"query": "需要哪些材料？"})
    assert result.ok
    assert result.result_ids


def test_semantic_memory_is_written_only_after_verified_answer():
    runtime, _ = build_runtime()
    state = runtime.invoke("制作方案前需要哪些材料？", session_id="s2")
    if state["status"] == "completed":
        assert runtime.memory_store.semantics
    assert runtime.memory_store.episodes
