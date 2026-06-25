"""算税引擎 — 综合所得年度汇算 + 年终奖对比，纯函数，Decimal 精度。

所有金额字段使用 Decimal，禁止 float。
"""

from decimal import Decimal

# ── 综合所得年度税率表（7 级超额累进） ─────────────────

_ANNUAL_BRACKETS: list[tuple[Decimal, Decimal, Decimal]] = [
    # (应纳税所得额上限, 税率, 速算扣除数)
    (Decimal("36000"),   Decimal("0.03"), Decimal("0")),
    (Decimal("144000"),  Decimal("0.10"), Decimal("2520")),
    (Decimal("300000"),  Decimal("0.20"), Decimal("16920")),
    (Decimal("420000"),  Decimal("0.25"), Decimal("31920")),
    (Decimal("660000"),  Decimal("0.30"), Decimal("52920")),
    (Decimal("960000"),  Decimal("0.35"), Decimal("85920")),
    (Decimal("inf"),     Decimal("0.45"), Decimal("181920")),
]

# ── 年终奖月税率表（全年一次性奖金单独计税） ──────────

_MONTHLY_BRACKETS: list[tuple[Decimal, Decimal, Decimal]] = [
    (Decimal("3000"),   Decimal("0.03"), Decimal("0")),
    (Decimal("12000"),  Decimal("0.10"), Decimal("210")),
    (Decimal("25000"),  Decimal("0.20"), Decimal("1410")),
    (Decimal("35000"),  Decimal("0.25"), Decimal("2660")),
    (Decimal("55000"),  Decimal("0.30"), Decimal("4410")),
    (Decimal("80000"),  Decimal("0.35"), Decimal("7160")),
    (Decimal("inf"),    Decimal("0.45"), Decimal("15160")),
]

# 基本减除费用
_BASIC_DEDUCTION = Decimal("60000")


def _calc_tax_from_brackets(
    taxable: Decimal,
    brackets: list[tuple[Decimal, Decimal, Decimal]],
) -> Decimal:
    """在给定的税率表中查档计算应纳税额。"""
    if taxable <= 0:
        return Decimal("0")
    for ceiling, rate, quick_deduction in brackets:
        if taxable <= ceiling or ceiling == Decimal("inf"):
            return (taxable * rate - quick_deduction).quantize(Decimal("0.01"))
    return Decimal("0")


def calc_annual_tax(
    annual_income: Decimal,
    social_insurance: Decimal = Decimal("0"),
    special_deductions: Decimal = Decimal("0"),
    other_deductions: Decimal = Decimal("0"),
) -> Decimal:
    """综合所得年度汇算。

    Args:
        annual_income: 年度综合所得收入（工资薪金+劳务报酬+稿酬+特许权使用费）
        social_insurance: 三险一金（基本养老保险+基本医疗保险+失业保险+住房公积金）
        special_deductions: 专项附加扣除（子女教育+继续教育+大病医疗+住房贷款利息/租金+赡养老人+婴幼儿照护）
        other_deductions: 其他扣除（企业年金+职业年金+商业健康保险+个人养老金等）
    Returns:
        年度应纳税额
    """
    taxable = annual_income - _BASIC_DEDUCTION - social_insurance - special_deductions - other_deductions
    return _calc_tax_from_brackets(taxable, _ANNUAL_BRACKETS)


def _calc_bonus_tax_separate(annual_bonus: Decimal) -> Decimal:
    """年终奖单独计税：奖金/12 → 查月税率表 → 应纳税额 = 奖金 × 税率 - 速算扣除数。"""
    if annual_bonus <= 0:
        return Decimal("0")
    monthly = (annual_bonus / Decimal("12")).quantize(Decimal("0.01"))
    return _calc_tax_from_brackets(monthly, _MONTHLY_BRACKETS)


def compare_bonus_methods(
    annual_income: Decimal,
    annual_bonus: Decimal,
    social_insurance: Decimal = Decimal("0"),
    special_deductions: Decimal = Decimal("0"),
    other_deductions: Decimal = Decimal("0"),
) -> dict:
    """对比年终奖两种计税方式。

    Returns:
        {
            "separate": {"bonus_tax": ..., "income_tax": ..., "total_tax": ...},
            "merged":   {"total_tax": ...},
            "diff": 两种方案的税额差额（separate - merged，正数表示并入更省）,
            "recommendation": "separate" | "merged" | "same",
        }
    """
    # 方案A：单独计税 — 奖金单独用月表，工资用年表
    bonus_tax = _calc_bonus_tax_separate(annual_bonus)
    income_tax = calc_annual_tax(annual_income, social_insurance, special_deductions, other_deductions)
    separate_total = bonus_tax + income_tax

    # 方案B：并入综合所得 — 奖金+工资合并后用年表
    merged_income = annual_income + annual_bonus
    merged_total = calc_annual_tax(merged_income, social_insurance, special_deductions, other_deductions)

    diff = separate_total - merged_total
    if diff > 0:
        recommendation = "merged"
    elif diff < 0:
        recommendation = "separate"
    else:
        recommendation = "same"

    return {
        "separate": {
            "bonus_tax": bonus_tax,
            "income_tax": income_tax,
            "total_tax": separate_total,
        },
        "merged": {
            "total_tax": merged_total,
        },
        "diff": diff,
        "recommendation": recommendation,
    }
