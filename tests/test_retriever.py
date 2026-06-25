"""retriever 模块单元测试 — 验证意图分类、RRF 融合、检索接口。

运行: pytest tests/test_retriever.py -v
"""

import pytest

from src.retriever import Retriever
from src.schemas import RetrievalResult, TaxSubCategory


class TestClassifyIntent:
    """意图分类 — 关键词匹配。"""

    @pytest.mark.parametrize("query,expected", [
        ("子女教育怎么扣除", TaxSubCategory.CHILD_EDUCATION.value),
        ("我在上海租房能扣多少", TaxSubCategory.HOUSING_RENT.value),
        ("房贷利息扣除标准", TaxSubCategory.HOUSING_LOAN.value),
        ("年终奖单独计税怎么算", TaxSubCategory.ANNUAL_BONUS.value),
        ("赡养父母能扣多少钱", TaxSubCategory.ELDERLY_SUPPORT.value),
        ("大病医疗报销后还能扣吗", TaxSubCategory.MAJOR_MEDICAL.value),
        ("继续教育考证能扣吗", TaxSubCategory.CONTINUING_EDUCATION.value),
        ("个税税率是多少", TaxSubCategory.TAX_RATE.value),
        ("综合所得年度汇算", TaxSubCategory.ANNUAL_SETTLEMENT.value),
        ("工资薪金怎么算税", TaxSubCategory.COMPREHENSIVE_INCOME.value),
        ("今天天气真好", None),  # 无匹配
    ])
    def test_classify(self, query, expected):
        assert Retriever.classify_intent(query) == expected

    def test_long_keyword_priority(self):
        """长词优先：'住房贷款利息' 不应被 '住房' 误匹配为 housing_rent。"""
        assert Retriever.classify_intent("住房贷款利息怎么算") == TaxSubCategory.HOUSING_LOAN.value


class TestRRFFusion:
    """RRF 融合算法。"""

    def _make_result(self, chunk_id: str, score: float = 0.0) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk_id, content=f"content_{chunk_id}",
            tax_subcategory="housing_rent", document_source="test",
            effective_date="2019-01-01", is_expired=False, score=score,
        )

    def test_empty_lists(self):
        result = Retriever.rrf_fusion([], [])
        assert result == []

    def test_only_dense(self):
        dense = [self._make_result("a"), self._make_result("b")]
        result = Retriever.rrf_fusion(dense, [])
        assert [r.chunk_id for r in result] == ["a", "b"]

    def test_fusion_order(self):
        """Dense 和 Sparse 都命中同一条时，排名应上升。"""
        dense = [self._make_result("a"), self._make_result("b"), self._make_result("c")]
        sparse = [self._make_result("b"), self._make_result("d")]
        result = Retriever.rrf_fusion(dense, sparse)
        # "b" 在两路都出现，应排第一
        assert result[0].chunk_id == "b"
        # 其次应是只出现在 Dense 中的 "a"
        assert result[1].chunk_id == "a"

    def test_dedup(self):
        """同一 chunk_id 不重复出现。"""
        dense = [self._make_result("x", score=0.9)]
        sparse = [self._make_result("x", score=0.8)]
        result = Retriever.rrf_fusion(dense, sparse)
        assert len(result) == 1
        assert result[0].chunk_id == "x"
