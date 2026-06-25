"""混合检索引擎 — BGE-M3 Dense+Sparse + RRF 融合 + BGE-Reranker 精排。

使用方式：
    from src.retriever import Retriever
    retriever = Retriever.from_database()
    results = retriever.search("年终奖单独计税怎么算")
"""

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.schemas import RetrievalResult, classify_text

_logger = logging.getLogger("retriever")


def _run_async(coro):
    """在同步上下文中安全地运行异步协程。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # 已在异步上下文中，用线程池隔离
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


class Retriever:
    """混合检索器，封装 BGE-M3 + pgvector + Reranker。"""

    def __init__(self, database_url: str):
        self._database_url = database_url
        self._conn = None  # 延迟连接
        self._sparse_index: dict[str, dict[str, float]] = {}

        # 延迟加载模型
        self._bge_model = None
        self._reranker = None
        self._model_name = "BAAI/bge-m3"

    # ── 数据库连接 ─────────────────────────────────────

    async def _get_conn(self):
        """获取或创建数据库连接。"""
        if self._conn is None:
            import asyncpg
            from pgvector.asyncpg import register_vector

            self._conn = await asyncpg.connect(self._database_url)
            await register_vector(self._conn)
        return self._conn

    async def _ensure_sparse_index(self):
        """加载稀疏索引到内存。"""
        if self._sparse_index:
            return
        conn = await self._get_conn()
        rows = await conn.fetch("SELECT id, sparse_embedding FROM chunks")
        self._sparse_index = {
            row["id"]: {
                str(k): float(v)
                for k, v in (json.loads(row["sparse_embedding"]) or {}).items()
            }
            for row in rows
        }
        _logger.info("稀疏索引加载完成，共 %d 条", len(self._sparse_index))

    # ── 工厂方法 ───────────────────────────────────────

    @classmethod
    def from_database(cls, database_url: str | None = None) -> "Retriever":
        """从 pgvector 数据库创建 Retriever 实例。

        Args:
            database_url: Supabase 连接串（默认从 DATABASE_URL 环境变量读取）
        """
        if database_url is None:
            from dotenv import load_dotenv
            load_dotenv()
            database_url = os.environ.get("DATABASE_URL", "")
        if not database_url:
            raise RuntimeError("未配置 DATABASE_URL，请在 .env 中设置 Supabase 连接串")

        retriever = cls(database_url)
        # 预加载稀疏索引
        _run_async(retriever._ensure_sparse_index())
        return retriever

    # ── 模型加载 ───────────────────────────────────────

    def _ensure_models(self):
        """延迟加载 BGE-M3 和 Reranker（首次检索时加载，~3-5 GB 内存）。"""
        if self._bge_model is None:
            from FlagEmbedding import BGEM3FlagModel, FlagReranker

            _logger.info("加载 BGE-M3 模型: %s ...", self._model_name)
            self._bge_model = BGEM3FlagModel(self._model_name, use_fp16=True)
            _logger.info("BGE-M3 加载完成")

            reranker_name = "BAAI/bge-reranker-v2-m3"
            _logger.info("加载 BGE-Reranker: %s ...", reranker_name)
            self._reranker = FlagReranker(reranker_name, use_fp16=True)
            _logger.info("BGE-Reranker 加载完成")

    # ── 意图分类 ───────────────────────────────────────

    @staticmethod
    def classify_intent(query: str) -> str | None:
        """关键词匹配 → tax_subcategory，无匹配返回 None（全量检索）。

        统一使用 schemas.classify_text()，建库端与检索端共用同一份关键词表。
        """
        return classify_text(query)

    # ── 编码 ───────────────────────────────────────────

    def _encode_query(self, query: str) -> tuple[np.ndarray, dict]:
        """BGE-M3 编码查询 → (dense_vec, sparse_dict)。"""
        self._ensure_models()
        output = self._bge_model.encode(
            [query], return_dense=True, return_sparse=True, batch_size=1
        )
        dense = np.array(output["dense_vecs"][0], dtype=np.float32)
        sparse = output.get("lexical_weights", [{}])[0]
        sparse = {str(k): float(v) for k, v in sparse.items()}
        return dense, sparse

    # ── Dense 检索 ─────────────────────────────────────

    async def _dense_search_async(self, dense_vec: np.ndarray, k: int = 10,
                                   filter_cat: str | None = None) -> list[RetrievalResult]:
        """pgvector 余弦相似度检索。"""
        conn = await self._get_conn()
        vec = dense_vec.tolist()

        if filter_cat:
            rows = await conn.fetch(
                """SELECT id, content, tax_subcategory, document_source,
                          effective_date, is_expired,
                          1 - (embedding <=> $1) AS score
                   FROM chunks
                   WHERE tax_subcategory = $2
                   ORDER BY embedding <=> $1
                   LIMIT $3""",
                vec, filter_cat, k,
            )
        else:
            rows = await conn.fetch(
                """SELECT id, content, tax_subcategory, document_source,
                          effective_date, is_expired,
                          1 - (embedding <=> $1) AS score
                   FROM chunks
                   ORDER BY embedding <=> $1
                   LIMIT $2""",
                vec, k,
            )

        return [
            RetrievalResult(
                chunk_id=row["id"],
                content=row["content"],
                tax_subcategory=row["tax_subcategory"] or "",
                document_source=row["document_source"] or "",
                effective_date=str(row["effective_date"]) if row["effective_date"] else "",
                is_expired=row["is_expired"] or False,
                score=max(0.0, float(row["score"])),  # cosine distance → similarity
            )
            for row in rows
        ]

    def _dense_search(self, dense_vec: np.ndarray, k: int = 10,
                      filter_cat: str | None = None) -> list[RetrievalResult]:
        return _run_async(self._dense_search_async(dense_vec, k, filter_cat))

    # ── Sparse 检索 ────────────────────────────────────

    def _sparse_search(self, sparse_vec: dict, k: int = 10) -> list[tuple[str, float]]:
        """Sparse 向量 → 与内存中稀疏索引做内积匹配。"""
        scores: list[tuple[str, float]] = []
        for chunk_id, chunk_sparse in self._sparse_index.items():
            score = 0.0
            for token, q_weight in sparse_vec.items():
                if token in chunk_sparse:
                    score += q_weight * chunk_sparse[token]
            if score > 0:
                scores.append((chunk_id, score))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    async def _sparse_scores_to_results_async(self, sparse_scores: list[tuple[str, float]],
                                               k: int = 10) -> list[RetrievalResult]:
        """将 Sparse 匹配结果回填文档内容。"""
        if not sparse_scores:
            return []

        ids = [chunk_id for chunk_id, _ in sparse_scores[:k * 2]]
        conn = await self._get_conn()
        rows = await conn.fetch(
            """SELECT id, content, tax_subcategory, document_source,
                      effective_date, is_expired
               FROM chunks WHERE id = ANY($1)""",
            ids,
        )
        row_map = {row["id"]: row for row in rows}

        results = []
        seen = set()
        for chunk_id, score in sparse_scores:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            row = row_map.get(chunk_id)
            if row:
                results.append(RetrievalResult(
                    chunk_id=chunk_id,
                    content=row["content"],
                    tax_subcategory=row["tax_subcategory"] or "",
                    document_source=row["document_source"] or "",
                    effective_date=str(row["effective_date"]) if row["effective_date"] else "",
                    is_expired=row["is_expired"] or False,
                    score=score,
                ))
            if len(results) >= k:
                break
        return results

    def _sparse_scores_to_results(self, sparse_scores: list[tuple[str, float]],
                                   k: int = 10) -> list[RetrievalResult]:
        return _run_async(self._sparse_scores_to_results_async(sparse_scores, k))

    # ── 融合 ───────────────────────────────────────────

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

    # ── Reranker ───────────────────────────────────────

    def _rerank(self, query: str, candidates: list[RetrievalResult],
                top_k: int = 5) -> list[RetrievalResult]:
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

    # ── 入口 ───────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """主入口：意图识别 → BGE-M3 编码 → Dense+Sparse → RRF → Reranker → Top-k。"""
        # 1. 意图分类
        intent = self.classify_intent(query)
        _logger.info("意图: %s", intent or "无（全量检索）")

        # 2. BGE-M3 编码
        dense_vec, sparse_vec = self._encode_query(query)

        # 3. Dense 检索
        dense_results = self._dense_search(dense_vec, k=10, filter_cat=intent)

        # 4. Sparse 检索
        if self._sparse_index:
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


# ── 便利函数 ───────────────────────────────────────────

_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """获取 Retriever 单例。"""
    global _retriever
    if _retriever is None:
        _retriever = Retriever.from_database()
    return _retriever
