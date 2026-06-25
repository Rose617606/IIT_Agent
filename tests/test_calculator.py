"""calculator 模块单元测试 — 验证综合所得年度汇算和年终奖对比。"""
from decimal import Decimal

import pytest

from src.calculator import calc_annual_tax, compare_bonus_methods


def _D(v: str) -> Decimal:
    return Decimal(v)


class TestCalcAnnualTax:
    """综合所得年度汇算 — 核心计算。"""

    def test_basic_case_no_deductions(self):
        """年收入 12 万，无任何扣除。"""
        tax = calc_annual_tax(
            annual_income=_D("120000"),
            social_insurance=_D("0"),
            special_deductions=_D("0"),
            other_deductions=_D("0"),
        )
        # 应纳税所得额 = 120000 - 60000 = 60000
        # 36000 * 3% = 1080, 24000 * 10% = 2400, total = 3480
        assert tax == _D("3480")

    def test_basic_case_with_deductions(self):
        """年收入 20 万，三险一金 2.4 万，专项附加扣除 1.2 万。"""
        tax = calc_annual_tax(
            annual_income=_D("200000"),
            social_insurance=_D("24000"),
            special_deductions=_D("12000"),
            other_deductions=_D("0"),
        )
        # 应纳税所得额 = 200000 - 60000 - 24000 - 12000 = 104000
        # 36000 * 3% = 1080, (104000-36000)*10% = 6800, total = 7880
        assert tax == _D("7880")

    def test_zero_tax(self):
        """年收入不到 6 万，无需缴税。"""
        tax = calc_annual_tax(
            annual_income=_D("50000"),
            social_insurance=_D("0"),
            special_deductions=_D("0"),
            other_deductions=_D("0"),
        )
        assert tax == _D("0")

    def test_high_income(self):
        """年收入 100 万，触发最高档税率。"""
        tax = calc_annual_tax(
            annual_income=_D("1000000"),
            social_insurance=_D("0"),
            special_deductions=_D("0"),
            other_deductions=_D("0"),
        )
        # 应纳税所得额 = 1000000 - 60000 = 940000
        # 36000*3%=1080, 108000*10%=10800, 156000*20%=31200,
        # 120000*25%=30000, 240000*30%=72000, 280000*35%=98000, total≈243080
        assert tax > _D("240000")
        assert tax < _D("250000")


class TestCompareBonusMethods:
    """年终奖两种计税方式对比。"""

    def test_bonus_separate_lower(self):
        """低年终奖场景 — 单独计税可能更优。"""
        result = compare_bonus_methods(
            annual_income=_D("200000"),
            annual_bonus=_D("30000"),
            social_insurance=_D("24000"),
            special_deductions=_D("12000"),
            other_deductions=_D("0"),
        )
        # 两种方案都应返回
        assert "separate" in result
        assert "merged" in result
        # 都有应纳税额
        assert result["separate"]["total_tax"] >= _D("0")
        assert result["merged"]["total_tax"] >= _D("0")
        # 推荐省税方案
        assert "recommendation" in result
        assert result["recommendation"] in ("separate", "merged", "same")

    def test_bonus_high_income_merged_better(self):
        """高收入+高年终奖 — 并入综合所得可能更优。"""
        result = compare_bonus_methods(
            annual_income=_D("500000"),
            annual_bonus=_D("200000"),
            social_insurance=_D("30000"),
            special_deductions=_D("12000"),
            other_deductions=_D("0"),
        )
        assert "separate" in result
        assert "merged" in result

    def test_bonus_tax_details(self):
        """验证返回结构包含详细信息。"""
        result = compare_bonus_methods(
            annual_income=_D("300000"),
            annual_bonus=_D("80000"),
            social_insurance=_D("24000"),
            special_deductions=_D("12000"),
            other_deductions=_D("0"),
        )
        separate = result["separate"]
        assert "bonus_tax" in separate
        assert "income_tax" in separate
        assert "total_tax" in separate
        merged = result["merged"]
        assert "total_tax" in merged
        # 差额
        assert "diff" in result
