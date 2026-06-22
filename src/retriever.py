"""混合检索引擎 — BGE-M3 Dense+Sparse + RRF 融合 + BGE-Reranker 精排。

使用方式：
    from src.retriever import Retriever
    retriever = Retriever.from_persist("knowledge_base/")
    results = retriever.search("年终奖单独计税怎么算")
"""

import json
import logging
from pathlib import Path

import numpy as np
from langchain_community.vectorstores import Chroma

from src.schemas import INTENT_KEYWORDS, RetrievalResult, TaxSubCategory

_logger = logging.getLogger("retriever")


class Retriever:
    """混合检索器，封装 BGE-M3 + Chroma + Reranker。"""

    def __init__(self, vectorstore: Chroma, sparse_index: dict):
        self.vectorstore = vectorstore
        self.sparse_index = sparse_index

        # 延迟加载模型，首次 search() 调用时加载
        self._bge_model = None
        self._reranker = None

    @classmethod
    def from_persist(cls, persist_dir: str = "knowledge_base/", model_name: str = "BAAI/bge-m3"):
        """从持久化目录恢复 Retriever 实例。"""
        persist_path = Path(persist_dir)
        if not persist_path.exists():
            raise FileNotFoundError(f"向量库目录不存在: {persist_path.absolute()}，请先运行 build_kb.py")

        # 加载 Chroma
        chroma = Chroma(persist_directory=str(persist_path), embedding_function=None)

        # 加载 Sparse 索引
        sparse_path = persist_path / "sparse_index.json"
        if sparse_path.exists():
            sparse_index = json.loads(sparse_path.read_text(encoding="utf-8"))
        else:
            _logger.warning("Sparse 索引文件不存在: %s，仅使用 Dense 检索", sparse_path)
            sparse_index = {}

        retriever = cls(chroma, sparse_index)
        retriever._model_name = model_name
        return retriever

    def _ensure_models(self):
        """延迟加载 BGE-M3 和 Reranker（首次检索时加载，~3-5 GB 内存）。"""
        if self._bge_model is None:
            from FlagEmbedding import BGEM3FlagModel, FlagReranker
            model_name = getattr(self, "_model_name", "BAAI/bge-m3")
            _logger.info("加载 BGE-M3 模型: %s ...", model_name)
            self._bge_model = BGEM3FlagModel(model_name, use_fp16=True)
            _logger.info("BGE-M3 加载完成")

            reranker_name = "BAAI/bge-reranker-v2-m3"
            _logger.info("加载 BGE-Reranker: %s ...", reranker_name)
            self._reranker = FlagReranker(reranker_name, use_fp16=True)
            _logger.info("BGE-Reranker 加载完成")

    # ── 意图分类 ─────────────────────────────────────

    @staticmethod
    def classify_intent(query: str) -> str | None:
        """关键词匹配 → tax_subcategory，无匹配返回 None（全量检索）。"""
        # 长词优先排序
        all_keywords = [(kw, cat) for cat, kws in INTENT_KEYWORDS.items() for kw in kws]
        all_keywords.sort(key=lambda x: -len(x[0]))  # 长 → 短

        for keyword, category in all_keywords:
            if keyword in query:
                return category
        return None

    # ── 编码 ─────────────────────────────────────────

    def _encode_query(self, query: str) -> tuple[np.ndarray, dict]:
        """BGE-M3 编码查询 → (dense_vec, sparse_dict)。"""
        self._ensure_models()
        output = self._bge_model.encode(
            [query], return_dense=True, return_sparse=True, batch_size=1
        )
        dense = np.array(output["dense_vecs"][0], dtype=np.float32)
        sparse = output.get("lexical_weights", [{}])[0]
        # sparse 的 key 是 token_id (int)，转为 str 与索引一致
        sparse = {str(k): float(v) for k, v in sparse.items()}
        return dense, sparse

    # ── 检索 ─────────────────────────────────────────

    def _dense_search(self, dense_vec: np.ndarray, k: int = 10, filter_cat: str | None = None) -> list[RetrievalResult]:
        """Dense 向量 → Chroma 语义检索。"""
        chroma_filter = None
        if filter_cat:
            chroma_filter = {"tax_subcategory": filter_cat}

        results = self.vectorstore.similarity_search_by_vector_with_relevance_scores(
            dense_vec.tolist(), k=k, filter=chroma_filter
        )
        # results: [(Document, score), ...]
        return [
            RetrievalResult(
                chunk_id=doc.metadata.get("chunk_id", ""),
                content=doc.page_content,
                tax_subcategory=doc.metadata.get("tax_subcategory", ""),
                document_source=doc.metadata.get("document_source", ""),
                effective_date=doc.metadata.get("effective_date", ""),
                is_expired=doc.metadata.get("is_expired", False),
                score=score,
            )
            for doc, score in results
        ]

    def _sparse_search(self, sparse_vec: dict, k: int = 10) -> list[tuple[str, float]]:
        """Sparse 向量 → 与预计算索引做内积匹配。"""
        scores: list[tuple[str, float]] = []
        for chunk_id, chunk_sparse in self.sparse_index.items():
            # 内积：Σ query_weight[token] × chunk_weight[token]
            score = 0.0
            for token, q_weight in sparse_vec.items():
                if token in chunk_sparse:
                    score += q_weight * chunk_sparse[token]
            if score > 0:
                scores.append((chunk_id, score))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    def _sparse_scores_to_results(self, sparse_scores: list[tuple[str, float]], k: int = 10) -> list[RetrievalResult]:
        """将 Sparse 匹配结果转为 RetrievalResult（需要回填 Chroma 中的文档内容）。"""
        results = []
        seen = set()
        for chunk_id, score in sparse_scores:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            # 从 Chroma 按 chunk_id 查询文档
            docs = self.vectorstore.get(ids=[chunk_id])
            if docs and docs["documents"]:
                meta = (docs["metadatas"] or [{}])[0]
                results.append(RetrievalResult(
                    chunk_id=chunk_id,
                    content=docs["documents"][0],
                    tax_subcategory=meta.get("tax_subcategory", ""),
                    document_source=meta.get("document_source", ""),
                    effective_date=meta.get("effective_date", ""),
                    is_expired=meta.get("is_expired", False),
                    score=score,
                ))
            if len(results) >= k:
                break
        return results

    # ── 融合 ─────────────────────────────────────────

    @staticmethod
    def rrf_fusion(
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        k: int = 60,
    ) -> list[RetrievalResult]:
        """RRF 融合：RRF(d) = Σ 1/(k + rank_i(d))。"""
        scores: dict[str, tuple[RetrievalResult, float]] = {}

        for rank, r in enumerate(dense_results):
            scores[r.chunk_id] = (r, 1.0 / (k + rank + 1))

        for rank, r in enumerate(sparse_results):
            _, existing_score = scores.get(r.chunk_id, (r, 0))
            scores[r.chunk_id] = (r, existing_score + 1.0 / (k + rank + 1))

        merged = sorted(scores.values(), key=lambda x: -x[1])
        for result, score in merged:
            result.score = score
        return [r for r, _ in merged]

    # ── Reranker ─────────────────────────────────────

    def _rerank(self, query: str, candidates: list[RetrievalResult], top_k: int = 5) -> list[RetrievalResult]:
        """BGE-Reranker Cross-Encoding 精排。"""
        if not candidates:
            return []

        self._ensure_models()
        pairs = [[query, c.content] for c in candidates]
        scores = self._reranker.compute_score(pairs, normalize=True)

        for result, score in zip(candidates, scores):
            result.score = float(score)

        candidates.sort(key=lambda x: -x.score)
        return candidates[:top_k]

    # ── 入口 ─────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """主入口：意图识别 → BGE-M3 编码 → Dense+Sparse → RRF → Reranker → Top-k。

        Args:
            query: 用户查询
            top_k: 返回结果数量

        Returns:
            按相关性排序的检索结果列表
        """
        # 1. 意图分类
        intent = self.classify_intent(query)
        _logger.info("意图: %s", intent or "无（全量检索）")

        # 2. BGE-M3 编码
        dense_vec, sparse_vec = self._encode_query(query)

        # 3. Dense 检索
        dense_results = self._dense_search(dense_vec, k=10, filter_cat=intent)

        # 4. Sparse 检索
        if self.sparse_index:
            sparse_scores = self._sparse_search(sparse_vec, k=10)
            sparse_results = self._sparse_scores_to_results(sparse_scores, k=10)
        else:
            sparse_results = []

        # 5. RRF 融合
        candidates = self.rrf_fusion(dense_results, sparse_results)
        _logger.info("RRF 融合后候选: %d 条", len(candidates))

        # 6. Reranker 精排
        final = self._rerank(query, candidates[:10], top_k=top_k)
        return final


# ── 便利函数 ─────────────────────────────────────────

# 模块级单例
_retriever: Retriever | None = None


def get_retriever(persist_dir: str = "knowledge_base/") -> Retriever:
    """获取 Retriever 单例。"""
    global _retriever
    if _retriever is None:
        _retriever = Retriever.from_persist(persist_dir)
    return _retriever
