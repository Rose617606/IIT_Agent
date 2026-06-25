"""structured_tools 模块测试 — 验证工具注册、Schema 校验、调用行为。"""
from decimal import Decimal

import pytest

from src.structured_tools import TOOLS, compare_bonus_tool, calc_annual_tax_tool


class TestToolSchemas:
    """工具 Schema 定义正确性。"""

    def test_calc_annual_tax_schema(self):
        """综合所得计算工具 — 入参 Schema 包含必需字段。"""
        schema = calc_annual_tax_tool.args_schema
        fields = schema.model_fields
        assert "annual_income" in fields
        assert "social_insurance" in fields
        assert "special_deductions" in fields
        assert "other_deductions" in fields

    def test_compare_bonus_schema(self):
        """年终奖对比工具 — 入参 Schema 包含必需字段。"""
        schema = compare_bonus_tool.args_schema
        fields = schema.model_fields
        assert "annual_income" in fields
        assert "annual_bonus" in fields

    def test_default_values(self):
        """非必需参数应有默认值。"""
        schema = calc_annual_tax_tool.args_schema
        assert schema.model_fields["social_insurance"].default == 0
        assert schema.model_fields["special_deductions"].default == 0


class TestToolExecution:
    """工具调用行为。"""

    def test_calc_annual_tax_tool(self):
        """直接调用工具，返回计算结果文本。"""
        result = calc_annual_tax_tool.invoke({
            "annual_income": 120000,
            "social_insurance": 0,
            "special_deductions": 0,
            "other_deductions": 0,
        })
        assert "3480" in result or "3,480" in result

    def test_compare_bonus_tool(self):
        """直接调用工具，返回对比结果文本。"""
        result = compare_bonus_tool.invoke({
            "annual_income": 200000,
            "annual_bonus": 30000,
            "social_insurance": 24000,
            "special_deductions": 12000,
            "other_deductions": 0,
        })
        # 应包含方案说明
        assert "单独计税" in result
        assert "并入综合所得" in result
        assert "推荐" in result

    def test_tool_description(self):
        """工具描述包含三要素：触发条件、必需参数、返回内容。"""
        desc = compare_bonus_tool.description
        assert "年终奖" in desc or "全年一次性奖金" in desc
        desc_calc = calc_annual_tax_tool.description
        assert "综合所得" in desc_calc


class TestToolsRegistry:
    """工具注册列表。"""

    def test_tools_list(self):
        """TOOLS 包含两个工具。"""
        assert len(TOOLS) == 2
        names = {t.name for t in TOOLS}
        assert "calc_annual_tax" in names
        assert "compare_bonus_methods" in names
