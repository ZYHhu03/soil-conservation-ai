# RAG architecture

## Data flow

```text
Source documents
  └─ parser.py: txt / Markdown / HTML / JSON / PDF
      └─ chunking.py: heading stack + clause number + sentence window
          ├─ embeddings.py → InMemoryVectorStore / MilvusVectorStore
          └─ graph.py → entity nodes + chunk relations

Query
  └─ query.py: intent + slots + rewrite + expansion
      └─ retrieval.py: BM25 + vector + graph → weighted RRF
          └─ rerank → context.py → prompts.py
              └─ generator → citation.py
                  └─ verified answer / expanded retry / abstention
```

## Structured chunk fields

Each chunk has a stable id, document id, text, title path, clause number and source metadata. The title path and clause number are included in BM25/vector text and are also available to the graph and reranker. A long clause is split at sentence boundaries with overlap; short residual fragments are merged with the preceding chunk.

## Knowledge graph

The graph extracts policy clauses, regions, project types, approval stages and technical measures. Entity nodes connect to chunks, and chunks sharing domain terms receive a lightweight relation. A query that mentions a project type, region or stage activates the corresponding entity neighborhood and contributes a graph channel to RRF.

## Retrieval and reranking

BM25 handles exact clause numbers and technical terms. The vector channel handles paraphrases. The graph channel contributes entity-linked evidence and related chunks. `HybridRetriever` assigns channel weights and fuses ranks using:

```text
score(chunk) = Σ channel_weight / (rrf_k + rank_channel(chunk))
```

`LexicalReranker` is the local baseline. `CrossEncoderReranker` uses a sentence-transformers cross encoder when the model is available.

## Generation and verification

The context assembler labels each evidence block `[E1]`, `[E2]`, ... and carries the chunk id, title and clause into the prompt. The generator must cite those labels. `CitationVerifier` rejects unknown labels, measures evidence coverage and triggers a second retrieval pass when the answer cites too little evidence. The default extractive generator is deterministic; `LangChainAnswerGenerator` accepts any LangChain chat model with `invoke`.

## Answer extraction

`BiLSTMAttentionExtractor` encodes context tokens with a bidirectional LSTM, computes token attention, fuses the pooled attention context back into each position and predicts start/end logits. `KeywordAnswerExtractor` provides a no-model sentence baseline for local retrieval evaluation.

## Voice interaction

`VoiceRAGService` performs ASR → query planning/RAG → TTS. `HTTPASRClient` and `HTTPTTSClient` use a JSON/base64 HTTP contract and can be replaced by provider-specific clients without changing the RAG pipeline.
