from soil_rag.chunking import StructuredChunker
from soil_rag.embeddings import HashEmbedding
from soil_rag.graph import KnowledgeGraph
from soil_rag.pipeline import RAGPipeline
from soil_rag.query import QueryPlanner
from soil_rag.retrieval import BM25Retriever, GraphRetriever, HybridRetriever, VectorRetriever
from soil_rag.schemas import Chunk, SourceDocument
from soil_rag.stores import InMemoryVectorStore


def _pipeline(chunks):
    graph = KnowledgeGraph()
    graph.add_chunks(chunks)
    store = InMemoryVectorStore(HashEmbedding(128))
    store.add(chunks)
    retriever = HybridRetriever(
        BM25Retriever(chunks),
        VectorRetriever(store),
        GraphRetriever(graph, {chunk.id: chunk for chunk in chunks}),
    )
    return RAGPipeline(retriever)


def test_structured_chunker_preserves_heading_and_clause():
    document = SourceDocument(
        id="policy-1",
        title="政策",
        text="# 第一章 总则\n\n## 适用范围\n\n第一条 本办法适用于生产建设项目。\n\n第二条 项目应落实水土保持措施。",
    )
    chunks = StructuredChunker(max_chars=200, min_chars=1).chunk(document)
    assert len(chunks) == 2
    assert chunks[0].clause_no == "第一条"
    assert chunks[0].title_path == ("第一章 总则", "适用范围")
    assert "生产建设项目" in chunks[0].text


def test_query_planner_rewrites_slots_and_classifies():
    plan = QueryPlanner().plan("广东的光伏项目要不要批，施工阶段需要啥材料？")
    assert plan.intent.value in {"process_consultation", "material_list"}
    assert plan.slots["region"] == "广东"
    assert plan.slots["stage"] == "施工"
    assert plan.expansions


def test_pipeline_returns_verified_citations():
    chunks = [
        Chunk("c1", "d1", "制作方案前应收集项目位置、建设内容和用地范围。", ("流程",), "第一条"),
        Chunk("c2", "d1", "方案完成内部校核后提交主管部门审查。", ("流程",), "第二条"),
    ]
    response = _pipeline(chunks).invoke("制作方案前需要准备哪些资料？")
    assert response.answer
    assert response.verification.valid
    assert response.citations
    assert all(citation.valid for citation in response.citations)


def test_graph_round_trip_keeps_entity_links():
    chunks = [Chunk("c1", "d1", "光伏项目施工阶段应设置临时排水措施。")]
    graph = KnowledgeGraph()
    graph.add_chunks(chunks)
    restored = KnowledgeGraph.from_dict(graph.to_dict())
    assert restored.related_chunk_ids("光伏项目施工阶段", limit=5)
