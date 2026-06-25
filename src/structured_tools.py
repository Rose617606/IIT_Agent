"""结构化工具 — 将计算函数包装为 LangChain StructuredTool，供 Agent 调用。

每个工具的 docstring 遵循三要素规范：
- 触发条件：Agent 何时调用此工具
- 必需参数：调用时必须提供的参数
- 返回内容：工具返回的结果格式
"""

from decimal import Decimal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.calculator import calc_annual_tax, compare_bonus_methods


# ── 入参 Schema ─────────────────────────────────────────

class CalcAnnualTaxInput(BaseModel):
    """综合所得年度汇算 — 入参。"""
    annual_income: float = Field(
        description="年度综合所得收入（元），包含工资薪金、劳务报酬、稿酬、特许权使用费",
    )
    social_insurance: float = Field(
        default=0,
        description="三险一金年缴总额（元），包含基本养老保险、基本医疗保险、失业保险、住房公积金",
    )
    special_deductions: float = Field(
        default=0,
        description="专项附加扣除年总额（元），包含子女教育、继续教育、大病医疗、住房贷款利息/租金、赡养老人、婴幼儿照护",
    )
    other_deductions: float = Field(
        default=0,
        description="其他扣除年总额（元），包含企业年金、职业年金、商业健康保险、个人养老金等",
    )


class CompareBonusInput(BaseModel):
    """年终奖计税方式对比 — 入参。"""
    annual_income: float = Field(
        description="年度综合所得收入（元），不含年终奖",
    )
    annual_bonus: float = Field(
        description="全年一次性奖金金额（元）",
    )
    social_insurance: float = Field(
        default=0,
        description="三险一金年缴总额（元）",
    )
    special_deductions: float = Field(
        default=0,
        description="专项附加扣除年总额（元）",
    )
    other_deductions: float = Field(
        default=0,
        description="其他扣除年总额（元）",
    )


# ── 工具函数 ────────────────────────────────────────────

def _to_decimal(value: float) -> Decimal:
    """float → Decimal，保留两位小数。"""
    return Decimal(str(value)).quantize(Decimal("0.01"))


def _format_number(value: Decimal) -> str:
    """格式化金额，加千分位。"""
    return f"{value:,.2f}"


def _calc_annual_tax_impl(
    annual_income: float,
    social_insurance: float = 0,
    special_deductions: float = 0,
    other_deductions: float = 0,
) -> str:
    """综合所得年度汇算工具实现。

    触发条件：用户询问年度个税计算、汇算清缴、应纳税额。
    必需参数：annual_income（年度综合所得收入）
    返回：应纳税额及计算过程。
    """
    tax = calc_annual_tax(
        _to_decimal(annual_income),
        _to_decimal(social_insurance),
        _to_decimal(special_deductions),
        _to_decimal(other_deductions),
    )
    taxable = (
        _to_decimal(annual_income)
        - Decimal("60000")
        - _to_decimal(social_insurance)
        - _to_decimal(special_deductions)
        - _to_decimal(other_deductions)
    )
    lines = [
        "【综合所得年度汇算】",
        f"年度收入：{_format_number(_to_decimal(annual_income))} 元",
        f"基本减除费用：60,000.00 元",
    ]
    if social_insurance:
        lines.append(f"三险一金：{_format_number(_to_decimal(social_insurance))} 元")
    if special_deductions:
        lines.append(f"专项附加扣除：{_format_number(_to_decimal(special_deductions))} 元")
    if other_deductions:
        lines.append(f"其他扣除：{_format_number(_to_decimal(other_deductions))} 元")
    lines.append(f"应纳税所得额：{_format_number(max(taxable, Decimal('0')))} 元")
    lines.append(f"应纳税额：{_format_number(tax)} 元")
    return "\n".join(lines)


def _compare_bonus_impl(
    annual_income: float,
    annual_bonus: float,
    social_insurance: float = 0,
    special_deductions: float = 0,
    other_deductions: float = 0,
) -> str:
    """年终奖计税方式对比工具实现。

    触发条件：用户询问年终奖怎么计税、单独计税还是并入综合所得、哪种方式更省税。
    必需参数：annual_income（不含年终奖的年度收入）、annual_bonus（年终奖金额）
    返回：两种方案的应纳税额对比及推荐。
    """
    result = compare_bonus_methods(
        _to_decimal(annual_income),
        _to_decimal(annual_bonus),
        _to_decimal(social_insurance),
        _to_decimal(special_deductions),
        _to_decimal(other_deductions),
    )
    sep = result["separate"]
    mer = result["merged"]
    diff = result["diff"]
    rec = result["recommendation"]

    rec_text = {"separate": "单独计税更省税", "merged": "并入综合所得更省税", "same": "两种方式税额相同"}[rec]

    lines = [
        "【年终奖计税方式对比】",
        f"年度收入（不含年终奖）：{_format_number(_to_decimal(annual_income))} 元",
        f"年终奖金额：{_format_number(_to_decimal(annual_bonus))} 元",
        "",
        "方案A：单独计税",
        f"  年终奖应纳税：{_format_number(sep['bonus_tax'])} 元",
        f"  工资部分应纳税：{_format_number(sep['income_tax'])} 元",
        f"  合计：{_format_number(sep['total_tax'])} 元",
        "",
        "方案B：并入综合所得",
        f"  合计应纳税：{_format_number(mer['total_tax'])} 元",
        "",
        f"差额：{_format_number(abs(diff))} 元",
        f"推荐：{rec_text}",
    ]
    return "\n".join(lines)


# ── StructuredTool 注册 ─────────────────────────────────

calc_annual_tax_tool = StructuredTool.from_function(
    func=_calc_annual_tax_impl,
    name="calc_annual_tax",
    description=(
        "计算综合所得年度汇算应纳税额。"
        "触发条件：用户询问年度个税计算、汇算清缴、应纳税额。"
        "必需参数：annual_income（年度综合所得收入，单位元）。"
        "返回：应纳税额及计算明细。"
    ),
    args_schema=CalcAnnualTaxInput,
)

compare_bonus_tool = StructuredTool.from_function(
    func=_compare_bonus_impl,
    name="compare_bonus_methods",
    description=(
        "对比年终奖两种计税方式（单独计税 vs 并入综合所得），推荐更省税的方案。"
        "触发条件：用户询问年终奖怎么计税、哪种方式更省税。"
        "必需参数：annual_income（不含年终奖的年度收入）、annual_bonus（年终奖金额）。"
        "返回：两种方案的应纳税额对比及推荐。"
    ),
    args_schema=CompareBonusInput,
)

# 工具列表，供 Agent 注册
TOOLS = [calc_annual_tax_tool, compare_bonus_tool]
