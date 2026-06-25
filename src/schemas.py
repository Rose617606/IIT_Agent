"""共享数据模型 — Pydantic schemas 供所有模块引用。"""

from datetime import date
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class TaxSubCategory(str, Enum):
    """税种子类枚举，用于元数据预过滤。"""
    CHILD_EDUCATION = "child_education"           # 子女教育
    CONTINUING_EDUCATION = "continuing_education" # 继续教育
    MAJOR_MEDICAL = "major_medical"              # 大病医疗
    HOUSING_LOAN = "housing_loan"                # 住房贷款利息
    HOUSING_RENT = "housing_rent"                # 住房租金
    ELDERLY_SUPPORT = "elderly_support"          # 赡养老人
    INFANT_CARE = "infant_care"                  # 婴幼儿照护
    ANNUAL_BONUS = "annual_bonus"                # 年终奖计税
    COMPREHENSIVE_INCOME = "comprehensive_income" # 综合所得
    TAX_RATE = "tax_rate"                        # 税率表
    ANNUAL_SETTLEMENT = "annual_settlement"      # 汇算清缴
    BASIC_DEDUCTION = "basic_deduction"          # 基本减除费用


class DocumentMeta(BaseModel):
    """文档级元数据，从 YAML frontmatter 解析。"""
    title: str
    source: str                       # 文号，如 "国发〔2018〕41号"
    effective_date: date
    status: str = "active"            # active | amended | expired


class ChunkMeta(BaseModel):
    """切片级元数据，存入 pgvector metadata。"""
    chunk_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    tax_subcategory: str
    document_source: str = ""             # build_kb 统一回填
    effective_date: date = date(1900, 1, 1)  # build_kb 统一回填
    is_expired: bool = False
    section_title: str
    chunk_index: int = 0


class Chunk(BaseModel):
    """一个切片 = 文本 + 元数据。"""
    content: str
    meta: ChunkMeta


class RetrievalResult(BaseModel):
    """检索返回的单条结果。"""
    chunk_id: str
    content: str
    tax_subcategory: str
    document_source: str
    effective_date: str = ""                        # Chroma metadata 存字符串
    is_expired: bool = False
    score: float = 0.0


# 意图分类触发词表（长词优先，多命中取最长匹配）
INTENT_KEYWORDS: dict[str, list[str]] = {
    TaxSubCategory.CHILD_EDUCATION.value:       ["子女教育", "孩子上学", "小孩读书", "学费扣除", "学前教育"],
    TaxSubCategory.CONTINUING_EDUCATION.value:  ["继续教育", "学历提升", "考证", "职业资格", "在职教育", "成人教育", "学历教育"],
    TaxSubCategory.MAJOR_MEDICAL.value:         ["大病医疗", "医保报销", "医药费", "住院", "自付医疗"],
    TaxSubCategory.HOUSING_LOAN.value:          ["房贷利息", "首套住房", "贷款买房", "住房贷款利息", "住房贷款", "公积金贷款"],
    TaxSubCategory.HOUSING_RENT.value:          ["住房租金", "租房支出", "租金扣除", "租房", "房租"],
    TaxSubCategory.ELDERLY_SUPPORT.value:       ["赡养老人", "赡养父母", "养老扣除", "独生子女老人"],
    TaxSubCategory.INFANT_CARE.value:           ["婴幼儿照护", "幼儿照护", "婴儿照护", "3岁以下婴幼儿", "3岁以下", "婴幼儿", "育儿"],
    TaxSubCategory.ANNUAL_BONUS.value:          ["全年一次性奖金", "年终奖", "奖金计税", "单独计税"],
    TaxSubCategory.COMPREHENSIVE_INCOME.value:  ["综合所得", "工资薪金", "劳务报酬", "稿酬", "特许权使用费", "应纳税所得额"],
    TaxSubCategory.ANNUAL_SETTLEMENT.value:     ["综合所得汇算清缴", "综合所得年度汇算", "汇算清缴", "退税流程", "补税", "年度申报", "年度汇算"],
    TaxSubCategory.TAX_RATE.value:              ["税率表", "超额累进", "速算扣除数", "个税税率"],
    TaxSubCategory.BASIC_DEDUCTION.value:       ["基本减除费用", "基本减除", "起征点", "免征额", "6万元", "5000元"],
}


def classify_text(text: str) -> str | None:
    """在文本中扫描关键词表，长词优先返回对应的 tax_subcategory。

    建库端（build_kb）和检索端（retriever）共用此函数。
    返回 None 表示未命中任何关键词，调用方自行兜底。
    """
    best_category = None
    best_len = 0
    for category, kw_list in INTENT_KEYWORDS.items():
        for kw in kw_list:
            if kw in text and len(kw) > best_len:
                best_category = category
                best_len = len(kw)
    return best_category
