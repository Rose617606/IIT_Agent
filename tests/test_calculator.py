"""calculator 模块单元测试 — 穷尽税率表7档 + 扣除项 + 年终奖对比 + 边界。
所有数值手动按公式：应纳税额 = 应纳税所得额 × 税率 - 速算扣除数，逐笔验证。
"""
from decimal import Decimal

import pytest

from src.calculator import calc_annual_tax, compare_bonus_methods


def _D(v: str) -> Decimal:
    return Decimal(v)


# ══════════════════════════════════════════════════════════════════════
# 综合所得年度税率表 — 7 档全覆盖
# 公式: taxable = income - 60000 - social - special - other
#       tax = taxable × rate - quick_deduction
# ══════════════════════════════════════════════════════════════════════

class TestAnnualBrackets:
    """综合所得7档超额累进税率，每档边界值 + 档内典型值。"""

    # ── 第1档: ≤36,000 → 3%, 速算扣除0 ──

    def test_bracket1_lower_boundary(self):
        """应纳税所得额=0 → 不交税。"""
        assert calc_annual_tax(_D("60000")) == _D("0")

    def test_bracket1_inside(self):
        """应纳税所得额=20000 → 3%。"""
        # 20000 × 3% = 600
        assert calc_annual_tax(_D("80000")) == _D("600.00")

    def test_bracket1_upper_boundary(self):
        """应纳税所得额=36000（第1档上限）→ 3%。"""
        # 36000 × 3% = 1080
        assert calc_annual_tax(_D("96000")) == _D("1080.00")

    # ── 第2档: 36,000~144,000 → 10%, 速算扣除2520 ──

    def test_bracket2_just_entering(self):
        """应纳税所得额=36001 → 跨入10%档。"""
        # 36001 × 10% - 2520 = 1080.10
        assert calc_annual_tax(_D("96001")) == _D("1080.10")

    def test_bracket2_inside(self):
        """应纳税所得额=60000 → 10%档内。"""
        # 60000 × 10% - 2520 = 3480
        assert calc_annual_tax(_D("120000")) == _D("3480.00")

    def test_bracket2_upper_boundary(self):
        """应纳税所得额=144000（第2档上限）→ 10%。"""
        # 144000 × 10% - 2520 = 11880
        assert calc_annual_tax(_D("204000")) == _D("11880.00")

    # ── 第3档: 144,000~300,000 → 20%, 速算扣除16920 ──

    def test_bracket3_just_entering(self):
        """应纳税所得额=144001 → 跨入20%档。"""
        # 144001 × 20% - 16920 = 11880.20
        assert calc_annual_tax(_D("204001")) == _D("11880.20")

    def test_bracket3_inside(self):
        """应纳税所得额=250000 → 20%档内。"""
        # 250000 × 20% - 16920 = 33080
        assert calc_annual_tax(_D("310000")) == _D("33080.00")

    def test_bracket3_upper_boundary(self):
        """应纳税所得额=300000（第3档上限）→ 20%。"""
        # 300000 × 20% - 16920 = 43080
        assert calc_annual_tax(_D("360000")) == _D("43080.00")

    # ── 第4档: 300,000~420,000 → 25%, 速算扣除31920 ──

    def test_bracket4_just_entering(self):
        """应纳税所得额=300001 → 跨入25%档。"""
        # 300001 × 25% - 31920 = 43080.25
        assert calc_annual_tax(_D("360001")) == _D("43080.25")

    def test_bracket4_inside(self):
        """应纳税所得额=350000 → 25%档内。"""
        # 350000 × 25% - 31920 = 55580
        assert calc_annual_tax(_D("410000")) == _D("55580.00")

    def test_bracket4_upper_boundary(self):
        """应纳税所得额=420000（第4档上限）→ 25%。"""
        # 420000 × 25% - 31920 = 73080
        assert calc_annual_tax(_D("480000")) == _D("73080.00")

    # ── 第5档: 420,000~660,000 → 30%, 速算扣除52920 ──

    def test_bracket5_just_entering(self):
        """应纳税所得额=420001 → 跨入30%档。"""
        # 420001 × 30% - 52920 = 73080.30
        assert calc_annual_tax(_D("480001")) == _D("73080.30")

    def test_bracket5_inside(self):
        """应纳税所得额=500000 → 30%档内。"""
        # 500000 × 30% - 52920 = 97080
        assert calc_annual_tax(_D("560000")) == _D("97080.00")

    def test_bracket5_upper_boundary(self):
        """应纳税所得额=660000（第5档上限）→ 30%。"""
        # 660000 × 30% - 52920 = 145080
        assert calc_annual_tax(_D("720000")) == _D("145080.00")

    # ── 第6档: 660,000~960,000 → 35%, 速算扣除85920 ──

    def test_bracket6_just_entering(self):
        """应纳税所得额=660001 → 跨入35%档。"""
        # 660001 × 35% - 85920 = 145080.35
        assert calc_annual_tax(_D("720001")) == _D("145080.35")

    def test_bracket6_inside(self):
        """应纳税所得额=800000 → 35%档内。"""
        # 800000 × 35% - 85920 = 194080
        assert calc_annual_tax(_D("860000")) == _D("194080.00")

    def test_bracket6_upper_boundary(self):
        """应纳税所得额=960000（第6档上限）→ 35%。"""
        # 960000 × 35% - 85920 = 250080
        assert calc_annual_tax(_D("1020000")) == _D("250080.00")

    # ── 第7档: >960,000 → 45%, 速算扣除181920 ──

    def test_bracket7_just_entering(self):
        """应纳税所得额=960001 → 跨入45%档。"""
        # 960001 × 45% - 181920 = 250080.45
        assert calc_annual_tax(_D("1020001")) == _D("250080.45")

    def test_bracket7_inside(self):
        """应纳税所得额=1500000 → 45%档内。"""
        # 1500000 × 45% - 181920 = 493080
        assert calc_annual_tax(_D("1560000")) == _D("493080.00")

    def test_bracket7_max(self):
        """极高收入 500万 应纳税所得额，确保不溢出。"""
        # 5000000 × 45% - 181920 = 2068080
        tax = calc_annual_tax(_D("5060000"))
        assert tax == _D("2068080.00")


# ══════════════════════════════════════════════════════════════════════
# 年终奖月税率表 — 7 档全覆盖
# 公式: monthly = bonus / 12, 查档确定税率和速算扣除数
#       tax = bonus × rate - quick_deduction  （税率应用于全年奖金！）
# ══════════════════════════════════════════════════════════════════════

class TestBonusMonthlyBrackets:
    """年终奖单独计税7档，直接测 _calc_bonus_tax_separate 效果。"""

    def _bonus_tax(self, bonus: str) -> Decimal:
        """通过 compare_bonus_methods 间接验证单独计税。"""
        return compare_bonus_methods(_D("0"), _D(bonus))["separate"]["bonus_tax"]

    # ── 第1档: 月均 ≤3000 → 3%, 速算扣除0 ──

    def test_bonus_bracket1_zero(self):
        """年终奖=0 → 不交税。"""
        assert self._bonus_tax("0") == _D("0")

    def test_bonus_bracket1_inside(self):
        """年终奖=12000 → 月均1000 → 3%档。"""
        # 12000 × 3% = 360
        assert self._bonus_tax("12000") == _D("360.00")

    def test_bonus_bracket1_upper(self):
        """年终奖=36000 → 月均3000（第1档上限）。"""
        # 36000 × 3% = 1080
        assert self._bonus_tax("36000") == _D("1080.00")

    # ── 第2档: 月均 3000~12000 → 10%, 速算扣除210 ──

    def test_bonus_bracket2_just_entering(self):
        """年终奖=36001 → 月均3000.08 → 跨入10%档。"""
        # 36001 × 10% - 210 = 3390.10
        assert self._bonus_tax("36001") == _D("3390.10")

    def test_bonus_bracket2_inside(self):
        """年终奖=80000 → 月均6666.67 → 10%档内。"""
        # 80000 × 10% - 210 = 7790
        assert self._bonus_tax("80000") == _D("7790.00")

    def test_bonus_bracket2_upper(self):
        """年终奖=144000 → 月均12000（第2档上限）。"""
        # 144000 × 10% - 210 = 14190
        assert self._bonus_tax("144000") == _D("14190.00")

    # ── 第3档: 月均 12000~25000 → 20%, 速算扣除1410 ──

    def test_bonus_bracket3_just_entering(self):
        """年终奖=144001 → 跨入20%档。"""
        # 144001 × 20% - 1410 = 27390.20
        assert self._bonus_tax("144001") == _D("27390.20")

    def test_bonus_bracket3_inside(self):
        """年终奖=200000 → 月均16666.67 → 20%档内。"""
        # 200000 × 20% - 1410 = 38590
        assert self._bonus_tax("200000") == _D("38590.00")

    def test_bonus_bracket3_upper(self):
        """年终奖=300000 → 月均25000（第3档上限）。"""
        # 300000 × 20% - 1410 = 58590
        assert self._bonus_tax("300000") == _D("58590.00")

    # ── 第4档: 月均 25000~35000 → 25%, 速算扣除2660 ──

    def test_bonus_bracket4_just_entering(self):
        """年终奖=300001 → 跨入25%档。"""
        # 300001 × 25% - 2660 = 72340.25
        assert self._bonus_tax("300001") == _D("72340.25")

    def test_bonus_bracket4_inside(self):
        """年终奖=360000 → 月均30000 → 25%档内。"""
        # 360000 × 25% - 2660 = 87340
        assert self._bonus_tax("360000") == _D("87340.00")

    def test_bonus_bracket4_upper(self):
        """年终奖=420000 → 月均35000（第4档上限）。"""
        # 420000 × 25% - 2660 = 102340
        assert self._bonus_tax("420000") == _D("102340.00")

    # ── 第5档: 月均 35000~55000 → 30%, 速算扣除4410 ──

    def test_bonus_bracket5_just_entering(self):
        """年终奖=420001 → 跨入30%档。"""
        # 420001 × 30% - 4410 = 121590.30
        assert self._bonus_tax("420001") == _D("121590.30")

    def test_bonus_bracket5_inside(self):
        """年终奖=600000 → 月均50000 → 30%档内。"""
        # 600000 × 30% - 4410 = 175590
        assert self._bonus_tax("600000") == _D("175590.00")

    def test_bonus_bracket5_upper(self):
        """年终奖=660000 → 月均55000（第5档上限）。"""
        # 660000 × 30% - 4410 = 193590
        assert self._bonus_tax("660000") == _D("193590.00")

    # ── 第6档: 月均 55000~80000 → 35%, 速算扣除7160 ──

    def test_bonus_bracket6_just_entering(self):
        """年终奖=660001 → 跨入35%档。"""
        # 660001 × 35% - 7160 = 223840.35
        assert self._bonus_tax("660001") == _D("223840.35")

    def test_bonus_bracket6_inside(self):
        """年终奖=800000 → 月均66666.67 → 35%档内。"""
        # 800000 × 35% - 7160 = 272840
        assert self._bonus_tax("800000") == _D("272840.00")

    def test_bonus_bracket6_upper(self):
        """年终奖=960000 → 月均80000（第6档上限）。"""
        # 960000 × 35% - 7160 = 328840
        assert self._bonus_tax("960000") == _D("328840.00")

    # ── 第7档: 月均 >80000 → 45%, 速算扣除15160 ──

    def test_bonus_bracket7_just_entering(self):
        """年终奖=960001 → 跨入45%档。"""
        # 960001 × 45% - 15160 = 416840.45
        assert self._bonus_tax("960001") == _D("416840.45")

    def test_bonus_bracket7_inside(self):
        """年终奖=1200000 → 月均100000 → 45%档内。"""
        # 1200000 × 45% - 15160 = 524840
        assert self._bonus_tax("1200000") == _D("524840.00")


# ══════════════════════════════════════════════════════════════════════
# 扣除项组合 — 穷尽各类扣除对税额的影响
# ══════════════════════════════════════════════════════════════════════

class TestDeductions:
    """各项扣除如何影响应纳税所得额和税额。"""

    def test_basic_deduction_only(self):
        """仅基本减除费用 6 万，年收入刚好 6 万不交税。"""
        assert calc_annual_tax(_D("60000")) == _D("0")

    def test_social_insurance(self):
        """三险一金可全额扣除。"""
        # 年收入 12万，三险一金 2.4万
        # taxable = 120000 - 60000 - 24000 = 36000
        # 36000 × 3% = 1080
        tax = calc_annual_tax(_D("120000"), social_insurance=_D("24000"))
        assert tax == _D("1080.00")

    def test_special_deductions_child_education(self):
        """子女教育 2000/月 = 24000/年。"""
        # 年收入 12万，专项扣除 24000
        # taxable = 120000 - 60000 - 24000 = 36000 → 1080
        tax = calc_annual_tax(_D("120000"), special_deductions=_D("24000"))
        assert tax == _D("1080.00")

    def test_special_deductions_housing_rent(self):
        """住房租金 1500/月（直辖市/省会）= 18000/年。"""
        tax = calc_annual_tax(_D("120000"), special_deductions=_D("18000"))
        # taxable = 120000 - 60000 - 18000 = 42000
        # 36000×3%=1080, 6000×10%=600, total=1680
        # 42000 × 10% - 2520 = 1680
        assert tax == _D("1680.00")

    def test_special_deductions_elderly_support(self):
        """赡养老人 独生子女 3000/月 = 36000/年。"""
        tax = calc_annual_tax(_D("120000"), special_deductions=_D("36000"))
        # taxable = 120000 - 60000 - 36000 = 24000 → 24000×3%=720
        assert tax == _D("720.00")

    def test_special_deductions_infant_care(self):
        """婴幼儿照护 2000/月 = 24000/年。"""
        tax = calc_annual_tax(_D("120000"), special_deductions=_D("24000"))
        # taxable = 120000 - 60000 - 24000 = 36000 → 1080
        assert tax == _D("1080.00")

    def test_special_deductions_continuing_education(self):
        """继续教育 学历 400/月 = 4800/年。"""
        tax = calc_annual_tax(_D("120000"), special_deductions=_D("4800"))
        # taxable = 120000 - 60000 - 4800 = 55200
        # 55200 × 10% - 2520 = 3000
        assert tax == _D("3000.00")

    def test_special_deductions_housing_loan(self):
        """住房贷款利息 1000/月 = 12000/年。"""
        tax = calc_annual_tax(_D("120000"), special_deductions=_D("12000"))
        # taxable = 120000 - 60000 - 12000 = 48000
        # 48000 × 10% - 2520 = 2280
        assert tax == _D("2280.00")

    def test_other_deductions_personal_pension(self):
        """个人养老金 12000/年 计入其他扣除。"""
        tax = calc_annual_tax(_D("120000"), other_deductions=_D("12000"))
        # taxable = 120000 - 60000 - 12000 = 48000
        # 48000 × 10% - 2520 = 2280
        assert tax == _D("2280.00")

    def test_all_deductions_combined(self):
        """所有扣除一起上：三险一金+子女教育+住房租金+赡养老人+个人养老金。"""
        social = _D("24000")   # 三险一金
        special = sum([
            _D("24000"),  # 子女教育
            _D("18000"),  # 住房租金
            _D("36000"),  # 赡养老人
        ])  # = 78000
        other = _D("12000")  # 个人养老金
        # taxable = 200000 - 60000 - 24000 - 78000 - 12000 = 26000
        # 26000 × 3% = 780
        tax = calc_annual_tax(_D("200000"), social, special, other)
        assert tax == _D("780.00")

    def test_deductions_push_across_bracket(self):
        """扣除刚好把应纳税所得额从20%档拉回10%档。"""
        # 年收入 40万，无扣除: taxable=340000, 25%档
        # 加扣除 20万: taxable=140000, 10%档
        without = calc_annual_tax(_D("400000"))
        # 340000 × 25% - 31920 = 53080
        assert without == _D("53080.00")

        with_ded = calc_annual_tax(_D("400000"),
                                    social_insurance=_D("40000"),
                                    special_deductions=_D("120000"),
                                    other_deductions=_D("40000"))
        # taxable = 400000 - 60000 - 40000 - 120000 - 40000 = 140000
        # 140000 × 10% - 2520 = 11480
        assert with_ded == _D("11480.00")

    def test_deductions_reduce_to_zero(self):
        """扣除超过收入，应纳税所得额为负 → 税额=0。"""
        tax = calc_annual_tax(_D("60000"),
                               social_insurance=_D("10000"),
                               special_deductions=_D("50000"),
                               other_deductions=_D("10000"))
        # taxable = 60000 - 60000 - 10000 - 50000 - 10000 = -70000 → 0
        assert tax == _D("0")


# ══════════════════════════════════════════════════════════════════════
# 年终奖对比 — 各种收入+奖金组合穷尽
# ══════════════════════════════════════════════════════════════════════

class TestBonusComparison:
    """年终奖单独 vs 并入，覆盖典型和高低极端组合。"""

    def test_low_income_low_bonus(self):
        """年收入8万+年终奖1万 → 单独计税更省。"""
        # 单独: 工资 taxable=20000→3%=600, 奖金 10000×3%=300, 合计=900
        # 并入: taxable=30000→3%=900, 相等
        r = compare_bonus_methods(_D("80000"), _D("10000"))
        assert r["recommendation"] == "same"

    def test_mid_income_mid_bonus_separate_better(self):
        """年收入20万+年终奖3万 → 单独计税更省（经典场景）。"""
        # 单独: 工资 taxable=140000→10%档, 140000×10%-2520=11480
        #       奖金 30000×3%=900, 合计=12380
        # 并入: taxable=170000→20%档, 170000×20%-16920=17080
        r = compare_bonus_methods(_D("200000"), _D("30000"))
        assert r["separate"]["bonus_tax"] == _D("900.00")
        assert r["separate"]["income_tax"] == _D("11480.00")
        assert r["separate"]["total_tax"] == _D("12380.00")
        assert r["merged"]["total_tax"] == _D("17080.00")
        assert r["diff"] == _D("-4700.00")
        assert r["recommendation"] == "separate"

    def test_high_income_low_bonus_separate_better(self):
        """年收入50万+年终奖2万 → 单独计税更省。"""
        # 单独: 工资 taxable=440000→30%档, 440000×30%-52920=79080
        #       奖金 20000×3%=600, 合计=79680
        # 并入: taxable=460000→30%档, 460000×30%-52920=85080
        r = compare_bonus_methods(_D("500000"), _D("20000"))
        assert r["recommendation"] == "separate"
        assert r["separate"]["total_tax"] == _D("79680.00")
        assert r["merged"]["total_tax"] == _D("85080.00")
        assert r["diff"] == _D("-5400.00")

    def test_high_income_high_bonus_merged_better(self):
        """年收入50万+年终奖50万 → 并入可能更优（奖金拉高税率有限）。"""
        # 单独: 工资 taxable=440000→30%档, 440000×30%-52920=79080
        #       奖金 500000/12=41666.67→30%档, 500000×30%-4410=145590
        #       合计=224670
        # 并入: taxable=440000+500000=940000→35%档
        #       940000×35%-85920=243080
        r = compare_bonus_methods(_D("500000"), _D("500000"))
        assert r["recommendation"] == "separate"
        # 验证具体数值
        assert r["separate"]["bonus_tax"] == _D("145590.00")
        assert r["separate"]["income_tax"] == _D("79080.00")
        assert r["separate"]["total_tax"] == _D("224670.00")
        assert r["merged"]["total_tax"] == _D("243080.00")
        assert r["diff"] == _D("-18410.00")

    def test_same_result(self):
        """年终奖很低（≤36000）且收入刚好跨档 → 可能相等。"""
        # 年收入 95999 → taxable=35999, 税率3%→1080(约)
        # 年终奖 1 → 税率3%→0
        r = compare_bonus_methods(_D("95999"), _D("1"))
        # 几乎相同
        assert abs(r["diff"]) <= _D("1")

    def test_with_all_deductions(self):
        """带有大量扣除项时，年终奖对比仍然正确。"""
        r = compare_bonus_methods(
            annual_income=_D("300000"),
            annual_bonus=_D("80000"),
            social_insurance=_D("24000"),
            special_deductions=_D("60000"),  # 子女教育+住房租金+赡养老人
            other_deductions=_D("12000"),     # 个人养老金
        )
        # 单独: 工资 taxable=300000-60000-24000-60000-12000=144000
        #       144000×10%-2520=11880
        #       奖金 80000×10%-210=7790
        #       合计=19670
        # 并入: taxable=380000-60000-24000-60000-12000=224000
        #       224000×20%-16920=27880
        assert r["separate"]["income_tax"] == _D("11880.00")
        assert r["separate"]["bonus_tax"] == _D("7790.00")
        assert r["separate"]["total_tax"] == _D("19670.00")
        assert r["merged"]["total_tax"] == _D("27880.00")
        assert r["recommendation"] == "separate"
        assert r["diff"] == _D("-8210.00")

    def test_bonus_boundary_edge(self):
        """年终奖刚好在税率临界点（36000 → 3%→10%跳变），验证不出现突变。"""
        # 年终奖 36000（3%上限）
        r1 = compare_bonus_methods(_D("200000"), _D("36000"))
        # 年终奖 36001（刚进10%）
        r2 = compare_bonus_methods(_D("200000"), _D("36001"))
        # 税负不应跳跃过大（36001比36000多交的税应接近但不超过~3390）
        diff_1 = r1["separate"]["total_tax"]
        diff_2 = r2["separate"]["total_tax"]
        # 36001多1元收入，税多约3390（因为跳档）
        # 36000 bonus tax = 1080, 36001 bonus tax = 3390.10
        assert diff_2 - diff_1 == _D("2310.10")


# ══════════════════════════════════════════════════════════════════════
# 边界与极端情况
# ══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """零收入、负应纳税所得额、极大值等。"""

    def test_zero_income(self):
        """零收入 → 税额为零。"""
        assert calc_annual_tax(_D("0")) == _D("0")

    def test_exactly_at_basic_deduction(self):
        """年收入刚好 = 6万，不交税。"""
        assert calc_annual_tax(_D("60000")) == _D("0")

    def test_just_above_basic_deduction(self):
        """年收入 60001，应纳税所得额=1 → 3% → 0.03。"""
        assert calc_annual_tax(_D("60001")) == _D("0.03")

    def test_negative_taxable(self):
        """扣除 > 收入 → taxable 为负 → 税额=0。"""
        tax = calc_annual_tax(_D("50000"),
                               social_insurance=_D("30000"),
                               special_deductions=_D("50000"))
        assert tax == _D("0")

    def test_zero_bonus(self):
        """零年终奖对比。"""
        r = compare_bonus_methods(_D("200000"), _D("0"))
        assert r["separate"]["bonus_tax"] == _D("0")
        # 单独计税的工资部分 = 并入（因为奖金=0）
        assert r["separate"]["total_tax"] == r["merged"]["total_tax"]
        assert r["recommendation"] == "same"

    def test_only_bonus_no_income(self):
        """只有年终奖，无工资收入。"""
        r = compare_bonus_methods(_D("0"), _D("120000"))
        # 单独: 工资 taxable=-60000→0, 奖金 120000×10%-210=11790, 合计=11790
        # 并入: taxable=120000-60000=60000, 60000×10%-2520=3480
        assert r["separate"]["bonus_tax"] == _D("11790.00")
        assert r["separate"]["income_tax"] == _D("0")
        assert r["merged"]["total_tax"] == _D("3480.00")
        assert r["recommendation"] == "merged"

    def test_all_deductions_zero_tax_high_income(self):
        """极高扣除让高收入也零税。"""
        # 年收入 30万，扣除 30万
        tax = calc_annual_tax(_D("300000"),
                               social_insurance=_D("50000"),
                               special_deductions=_D("150000"),
                               other_deductions=_D("100000"))
        # taxable = 300000 - 60000 - 50000 - 150000 - 100000 = -60000 → 0
        assert tax == _D("0")

    def test_decimal_precision(self):
        """Decimal 精度：不会出现浮点误差。"""
        tax = calc_annual_tax(_D("123456"), _D("7890"), _D("5678"), _D("1234"))
        taxable = _D("123456") - _D("60000") - _D("7890") - _D("5678") - _D("1234")
        # taxable = 48654 => 48654 × 10% - 2520 = 2345.40
        assert tax == _D("2345.40")
