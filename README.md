# Soil Conservation RAG

面向水土保持领域的领域模型适配、RAG 和状态化 ReAct Agent 实验项目。

项目包含三个可独立运行的部分：使用 LoRA 将 Qwen-7B-Chat 适配到水土保持专业表达，基于结构化文档、向量检索和知识图谱的证据增强问答链路，以及基于 LangGraph 的状态化 ReAct Agent。

## 项目边界

- 训练方法：LoRA，`r=16`，`alpha=32`，默认 dropout `0.05`。
- 基座模型：`Qwen/Qwen-7B-Chat`（通过配置可替换为兼容的 Qwen 模型）。
- 训练数据：JSONL 指令问答，字段包含场景分类、问题、答案和来源信息。
- 评测：固定测试集，对比基座模型和 LoRA adapter 的生成结果，统计关键词命中、拒答约束和人工复核输入。
- 数据集覆盖专业术语、审批流程、材料清单、技术措施、场景问答和边界回答等任务类型。

## RAG 链路

RAG 部分将政策法规、业务规范和监测说明解析为结构化文档，保留标题层级、条款编号和来源元数据，并通过以下流程完成问答：

```text
文档解析 → 标题/条款切分 → Embedding + BM25 → 知识图谱增强
        → RRF 混合召回 → Rerank → 证据上下文组装
        → 答案生成 → Claim/Citation 校验 → 低证据重检或拒答
```

- `StructuredChunker`：按 Markdown 标题、条款编号和句子边界切分，保留 `title_path`、`clause_no` 和文档元数据。
- `to_langchain_documents` / `build_langchain_chain`：将文档和端到端查询链接入 LangChain Runnable 体系。
- `HashEmbedding` / `SentenceTransformerEmbedding`：本地确定性向量基线和 Sentence-Transformers 适配器。
- `BM25Retriever`、`VectorRetriever`、`GraphRetriever`：稀疏、稠密和实体关联三路召回。
- `HybridRetriever`：使用加权 Reciprocal Rank Fusion 合并检索结果。
- `LexicalReranker` / `CrossEncoderReranker`：本地词项重排和 Cross-Encoder 重排适配器。
- `KnowledgeGraph`：抽取地区、项目类型、审批阶段、技术措施和条款实体，构建 chunk-entity 关联。
- `CitationVerifier`：解析 `[E#]` 引用，检查引用是否属于证据集合并计算 evidence coverage。
- `BiLSTMAttentionExtractor`：答案抽取模型头；`KeywordAnswerExtractor` 用于本地无模型评测。
- `QueryPlanner`：覆盖 12 类意图、槽位抽取、Query Rewriting 和 Query Expansion。
- `VoiceRAGService`：串联 ASR、RAG 和 TTS，`HTTPASRClient` / `HTTPTTSClient` 对接通用语音服务。

意图类别为：政策查询、流程咨询、材料清单、项目范围、技术措施、监测说明、验收咨询、费用标准、办理时限、地区要求、术语解释和其他。

## ReAct Agent

Agent 使用显式 `AgentState` 管理意图、槽位、子任务、工具轨迹、证据、断言、校验结果和记忆上下文，LangGraph 节点编排如下：

```text
意图识别 → 槽位抽取 → 记忆加载 → 任务规划
    → 工具调用循环 → 证据聚合 → 答案生成
    → Claim 抽取 → Claim-level Verification → 完成 / 重检 / Abstention
```

- `QueryDecomposer`：基于任务类型和 Prompt JSON 输出将复合问题拆为政策、流程、材料、关系等子任务。
- `ReActPlanner`：逐个选择待执行任务，限制工具调用次数，记录尝试次数和失败原因。
- `ToolRegistry`：统一封装 `rag_search`、`graph_lookup`、`memory_lookup` 等 Typed Tools。
- `MCPToolServer` / `MCPToolClient`：通过 `tools/list` 和 `tools/call` 暴露与调用工具。
- `ClaimLevelVerifier`：将答案拆为事实断言，与召回 chunk 和图谱实体关系对齐，输出覆盖率、冲突和重检决策。
- `WorkingMemory`：保存当前 session 的项目类型、地区、阶段、已确认实体和缺失槽位。
- `EpisodicMemory`：记录历史提问、回答、来源和纠错反馈。
- `SemanticMemory`：仅写入已验证答案，保存来源、时间、置信度和适用条件。

```bash
# 运行本地 ReAct Agent
python scripts/run_agent.py \
  --chunks data/rag/index/chunks.jsonl \
  --graph data/rag/index/graph.json \
  --query "项目是否需要审批、需要哪些材料、依据哪条政策？" \
  --session-id demo-session

# 使用 LangGraph 编译后的工作流
python scripts/run_agent.py \
  --chunks data/rag/index/chunks.jsonl \
  --graph data/rag/index/graph.json \
  --query "项目是否需要审批、需要哪些材料、依据哪条政策？" \
  --use-langgraph
```

## 目录

```text
.github/workflows/test.yml    GitHub Actions 测试工作流
CONTRIBUTING.md               开发和数据格式说明
LICENSE                       MIT License
configs/lora.yaml             训练与评测配置
configs/rag.yaml              RAG 切分、Embedding、检索和生成配置
configs/agent.yaml            Agent 调用预算和记忆配置
data/raw/demo.jsonl           水土保持领域问答数据
data/rag/documents/           RAG 示例文档
data/rag/eval.jsonl           检索评测集
docs/rag_architecture.md      RAG 数据流与组件说明
docs/agent_architecture.md    Agent 状态图与校验说明
scripts/prepare_dataset.py    校验并切分数据
scripts/train_lora.py         LoRA 训练入口
scripts/evaluate.py           基座/adapter 对比评测入口
scripts/infer.py              单条问题推理入口
scripts/ingest_documents.py   文档解析、切分、索引和图谱构建
scripts/run_rag.py            单条 RAG 查询入口
scripts/evaluate_retrieval.py BM25/向量/混合检索对比
scripts/voice_query.py        ASR → RAG → TTS 语音查询入口
scripts/mcp_server.py         MCP tools/list/tools/call JSON-RPC 服务
scripts/run_agent.py          LangGraph/ReAct Agent 查询入口
src/soil_lora/data.py         数据 schema、模板和数据集构造
src/soil_lora/metrics.py      轻量离线指标
src/soil_rag/                 RAG 核心组件
src/soil_agent/               Agent 状态、规划、工具、校验和记忆
tests/                        不依赖模型下载的单元测试
```

## 快速开始

Python 3.10+，训练环境为带 CUDA 的 Linux 环境。7B 模型训练使用 GPU 显存；本仓库不在 CI 中自动下载模型。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[train,rag,agent,test]'

# 1. 校验并按 id 稳定切分数据
python scripts/prepare_dataset.py \
  --input data/raw/demo.jsonl \
  --output-dir data/processed

# 2. 训练 LoRA adapter
python scripts/train_lora.py \
  --config configs/lora.yaml \
  --train-file data/processed/train.jsonl \
  --valid-file data/processed/valid.jsonl

# 3. 对 adapter 做离线生成评测
python scripts/evaluate.py \
  --config configs/lora.yaml \
  --test-file data/processed/test.jsonl \
  --adapter outputs/soil-qwen-lora
```

### 运行 RAG

```bash
# 1. 解析文档、结构化切分、构建本地向量索引和知识图谱
python scripts/ingest_documents.py \
  --input data/rag/documents \
  --output-dir data/rag/index \
  --config configs/rag.yaml

# 2. 执行一条带引用校验的 RAG 查询
python scripts/run_rag.py \
  --chunks data/rag/index/chunks.jsonl \
  --graph data/rag/index/graph.json \
  --query "制作水土保持方案前需要准备哪些资料？"

# 3. 对比 BM25、向量和知识图谱增强混合检索
python scripts/evaluate_retrieval.py \
  --chunks data/rag/index/chunks.jsonl \
  --graph data/rag/index/graph.json \
  --eval-file data/rag/eval.jsonl
```

默认配置使用确定性的 `HashEmbedding` 和内存向量库，便于本地复现；生产配置可切换到 `SentenceTransformerEmbedding` 和 `MilvusVectorStore`。

## 数据格式

每行一个 JSON 对象。`instruction` 和 `output` 是必填字段；`input` 可为空，`source` 用于保留可追溯信息，不会自动被模型当作证据。

```json
{
  "id": "term-0001",
  "category": "术语",
  "instruction": "解释水土保持方案中的表土剥离。",
  "input": "面向项目管理人员，用简洁语言回答。",
  "output": "表土剥离是对施工扰动范围内具有保存价值的表层土壤进行剥离、临时堆存和后续利用的措施。具体厚度和堆存要求应以项目设计及所在地现行要求为准。",
  "source": "水土保持领域资料",
  "source_version": "2026-01"
}
```

## 训练设计

训练目标是让模型学会领域表达和回答格式，而不是让模型记住一份会变化的法规库。因此，政策时效、适用地区和具体条款在完整系统中仍应交给 RAG 检索和引用校验。微调阶段重点覆盖：

1. 专业术语解释和同义表达归一化；
2. 审批流程的角色、阶段、前置条件和材料清单；
3. 信息不足时主动说明缺失条件，不虚构法规条款；
4. 统一输出“结论—适用条件—待确认信息”的结构。

## 评测解释

当前脚本提供可自动化的轻量指标：答案关键词命中率、拒答约束命中率和 JSONL 结果导出。评测集按术语、流程、材料、条件缺失和时效性分层，记录准确率、拒答正确率、格式合规率以及显存/时延。

## RAG 评测

`evaluate_retrieval.py` 对 BM25、向量检索和知识图谱增强混合检索使用同一批标注问题，输出 `Recall@10`、`MRR`、答案准确率和 `P95` 检索延迟。评测样本通过 `relevant_document_ids` 或 `relevant_chunk_ids` 指定相关证据，答案准确率由答案抽取结果与必答术语集合计算。
