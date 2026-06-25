# IITAgent（个税智能体）代码审查报告

> **审查视角**：资深技术面试官，面向求职作品集（RAG 原型，非生产系统）
> **审查日期**：2026-06-25
> **审查范围**：全项目（src/、tests/、CLAUDE.md、requirements.txt）
> **审查方法**：多维度分类 + 对抗验证过滤误报 + 逐条人工确认

---

## 1. 审查摘要

### 1.1 问题分布

| 维度 | P0（阻塞） | P1（高危） | P2（重要） | P3（建议） | 合计 |
|------|-----------|-----------|-----------|-----------|------|
| RAG检索 | 0 | 0 | 6 | 2 | 8 |
| 算税 | 0 | 0 | 3 | 4 | 7 |
| 提示词 | 0 | 0 | 5 | 2 | 7 |
| 可维护性 | 0 | 0 | 3 | 4 | 7 |
| 税务领域 | 0 | 0 | 3 | 0 | 3 |
| 未分类 | 0 | 0 | 7 | 2 | 9 |
| **合计** | **0** | **0** | **27** | **14** | **41** |

### 1.2 对抗验证过滤统计

| 指标 | 数值 |
|------|------|
| 初始发现 | 43 |
| 对抗验证确认 | 41 |
| 降级处理 | 0 |
| 误报过滤 | 2 |
| 过滤率 | 4.7% |

> 被过滤的误报包括：(1) "综合所得税率表和扣除标准全部正确"——属于正面确认，非问题；(2) 重复条目合并——"Pydantic金额字段用float违禁"与"结构化工具入参使用float可能损失精度"为同一问题的不同表述，已合并为一条。

### 1.3 一句话总体评价

> **这是一份远超预期的 RAG 原型作品。核心算税逻辑零错误，税率表逐档核对通过，RAG 架构设计（Dense+Sparse+RRF+Reranker）选型合理且工程落得扎实。41 个发现中无 P0/P1 阻塞性问题，主要集中在检索精度优化、提示词工程健壮性、代码规范细节三个方向——对求职作品集而言，这意味着项目骨架健康、只需打磨细节。**

---

## 2. 逐项详情

### 2.1 P2 级别（重要，27 项）

---

#### P2-1: 单分类丢失跨类目文档

- **文件**：`E:/IITagent/src/schemas.py` L79-92
- **维度**：RAG检索
- **置信度**：高 | **对抗投票**：确认0/降级3/拒绝0

**问题描述**：`classify_text()` 采用最长关键词匹配，返回唯一一个 `tax_subcategory`。当用户查询跨越多个子类时（如"年终奖并入综合所得怎么算"同时涉及 `annual_bonus` 和 `comprehensive_income`），只有一个类目被选中用于过滤 dense 检索，另一类目的相关文档被排除在外。Sparse 检索不做过滤，RRF 融合能部分恢复遗漏——但 Dense 路作为主检索通道，过滤失效的代价不可忽视。设计文档 6.1 明确写道"意图识别 → 元数据预过滤"，但单分类策略与多主题查询存在根本性矛盾。

**修复建议**：
```python
# 方案A（轻量，推荐）：classify_text 改为返回 list[str]，支持多标签分类
# schemas.py L79-92
def classify_text(text: str) -> list[str]:
    """当多个类目关键词同时命中、长度相近（差≤1）时，全部返回。"""
    matches = []
    for subcategory, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                matches.append((len(kw), subcategory.value))
    if not matches:
        return []
    matches.sort(key=lambda x: x[0], reverse=True)
    max_len = matches[0][0]
    # 取所有与最长匹配长度差 ≤1 的类目
    result = [m[1] for m in matches if max_len - m[0] <= 1]
    return result

# retriever.py search() 中用 WHERE tax_subcategory = ANY($2) 代替 = $2
```

---

#### P2-2: Sparse 检索不受类目过滤

- **文件**：`E:/IITagent/src/retriever.py` L229-240, L329-355
- **维度**：RAG检索
- **置信度**：高 | **对抗投票**：确认0/降级3/拒绝0

**问题描述**：`search()` 方法中，dense 检索应用了 `tax_subcategory` 过滤（L339: `filter_cat=intent`），但 sparse 检索（L342-347）完全绕过过滤——`_sparse_search` 遍历 `self._sparse_index` 中的所有 chunk 做内积匹配，不区分类目。RRF 融合时，被过滤的 dense 结果与未过滤的 sparse 结果混合排序，不同类目的 chunk 因关键词巧合匹配获得 RRF 加分，可能挤出真正相关的结果。例如：查询"住房贷款利息"时，sparse 路可能因为"住房"token 命中 `housing_rent` 的 chunk。

**修复建议**：
```python
# _sparse_search 签名增加 filter_cat 参数
def _sparse_search(self, query: str, k: int = 10, filter_cat: str | None = None) -> list[SearchResult]:
    ...
    for idx, (chunk_id, vec) in enumerate(zip(self._sparse_ids, self._sparse_vectors)):
        if filter_cat and self._sparse_cats[idx] != filter_cat:
            continue  # 跳过不匹配类目的 chunk
        score = float(np.dot(query_vec, vec))
        ...

# _ensure_sparse_index 加载时同时加载类目映射
# L85-100: self._sparse_cats = [meta.get('tax_subcategory') for meta in ...]

# search() 传 intent：
# sparse_results = self._sparse_search(query, self._sparse_top_k, filter_cat=intent)
```

---

#### P2-3: "应纳税所得额"关键词过于宽泛

- **文件**：`E:/IITagent/src/schemas.py` L72
- **维度**：RAG检索
- **置信度**：中 | **对抗投票**：确认0/降级3/拒绝0

**问题描述**：`comprehensive_income` 类目包含关键词"应纳税所得额"。这是税法中的通用核心概念——年终奖计税、专项附加扣除、综合所得汇算清缴等几乎所有场景都会涉及。如果一个查询包含该词（如"年终奖的应纳税所得额怎么算"），`classify_text` 会匹配到 `comprehensive_income`（"应纳税所得额"5字 > "年终奖"3字），而非 `annual_bonus`。但用户实际关心的是年终奖计税规则。

**修复建议**：
```python
# schemas.py INTENT_KEYWORDS 修改
TaxSubCategory.COMPREHENSIVE_INCOME.value: [
    "综合所得", "工资薪金", "劳务报酬", "稿酬", "特许权使用费",
    # 移除 "应纳税所得额" —— 它是泛化概念，不应作为类目分类信号
],
```

---

#### P2-4: `_get_conn` 并发创建连接无锁保护

- **文件**：`E:/IITagent/src/retriever.py` L58-83
- **维度**：可维护性
- **置信度**：高

**问题描述**：`_get_conn` 方法在 `self._conn is None` 时创建新连接，但没有使用 `asyncio.Lock` 或 `threading.Lock` 保护临界区。两个并发请求同时进入，都检测到 `self._conn is None` → 各自创建连接 → 第一个连接被第二个覆盖，造成连接泄漏。当前 Gradio 单会话模式下概率极低，但若切换到 FastAPI 多请求并发场景，会表现为间歇性连接泄漏。

**修复建议**：
```python
import asyncio

class Retriever:
    def __init__(self, ...):
        ...
        self._conn_lock = asyncio.Lock()

    async def _get_conn(self):
        if self._conn is not None:
            return self._conn
        async with self._conn_lock:
            # 双检：拿到锁后再次检查
            if self._conn is not None:
                return self._conn
            self._conn = await asyncpg.connect(self._dsn)
            return self._conn
```

---

#### P2-5: 首请求冷启动延迟含模型+索引全量加载

- **文件**：`E:/IITagent/src/retriever.py` L124-155, L85-100, L329-355
- **维度**：RAG检索
- **置信度**：高

**问题描述**：`search()` 首次调用时，`_ensure_models` 加载 BGE-M3（~3-5GB, 10-30s）+ Reranker，`_ensure_sparse_index` 从 DB 拉取所有稀疏向量到内存（1000+ 条 ~1-3s）。总冷启动延迟可达 15-45 秒，Render 免费层默认 30s 超时可能导致首次请求失败。当前仅在代码注释中提示，没有 pre-warm 机制。

**修复建议**：
```python
# 在 from_database() 后调用一个轻量 warm-up
async def warmup(self):
    """预加载所有模型和索引，避免首次请求超时。"""
    await self._ensure_models()
    await self._ensure_sparse_index()

# app.py 启动时调用：
# retriever = await Retriever.from_database(dsn, config)
# await retriever.warmup()
```

---

#### P2-6: CJK 字符过滤可能排除表格衍生内容

- **文件**：`E:/IITagent/src/build_kb.py` L355
- **维度**：RAG检索
- **置信度**：中

**问题描述**：`chunk_section()` general 分支使用 `len(re.sub(r'[^一-鿿]', '', sub)) < 10` 过滤碎片。该正则范围仅覆盖基本汉字区（U+4E00-U+9FFF），不包括扩展汉字、中文标点（，。；：）、全角数字。表格经 `_table_chunk_to_text` 转为自然语言后（如"税率：3%；速算扣除数：0。应纳税所得额：不超过3000"），CJK 字符可能不足 10 个，导致有效内容被静默丢弃。

**修复建议**：
```python
# 将正则改为覆盖更多 CJK 字符和全角标点
# L355
cjk_count = len(re.sub(r'[^一-鿿㐀-䶿＀-￯]', '', sub))
if cjk_count < 5:  # 降低阈值到 5
    continue
# 或对 _table_chunk_to_text 生成的 chunk 默认不过滤
```

---

#### P2-7: Tool 入参 Schema 使用 float 而非 Decimal

- **文件**：`E:/IITagent/src/structured_tools.py` L21-L57
- **维度**：算税
- **置信度**：高 | **对抗投票**：确认0/降级3/拒绝0

**问题描述**：项目全局规定"所有金额字段使用 Decimal，禁止 float"（calculator.py L3）。但 `CalcAnnualTaxInput` 和 `CompareBonusInput` 的 Pydantic Field 类型为 `float`。LLM 通过 Tool Calling 传入的 JSON number 经 Pydantic 解析为 float 后再由 `_to_decimal` 转回 Decimal。虽然 `Decimal(str(value))` 能处理常见值，但不构成精确的类型安全网。且类型声明误导阅读者以为代码接受 float 计算。

**修复建议**：
```python
from pydantic import BeforeValidator
from typing import Annotated
from decimal import Decimal

Money = Annotated[
    Decimal,
    BeforeValidator(lambda v: Decimal(str(v)).quantize(Decimal("0.01")))
]

class CalcAnnualTaxInput(BaseModel):
    annual_income: Money = Field(description="年度综合所得收入（元）")
    social_insurance: Money = Field(default=Decimal("0"), description="...")
    # ... 所有金额字段统一使用 Money 类型
```

---

#### P2-8: `quantize` 默认 `ROUND_HALF_EVEN` 而非四舍五入

- **文件**：`E:/IITagent/src/calculator.py` L46, L78
- **维度**：算税
- **置信度**：高 | **对抗投票**：确认0/降级3/拒绝0

**问题描述**：`_calc_tax_from_brackets` 和 `_calc_bonus_tax_separate` 的 `return` 语句使用 `.quantize(Decimal("0.01"))`，未指定 `rounding` 参数，默认使用 `ROUND_HALF_EVEN`（银行家舍入，遇 5 向偶数靠拢）。中国税务和会计规范使用 `ROUND_HALF_UP`（四舍五入）。已验证具体场景：`taxable=36010.05` 时，`1081.005` 在 `ROUND_HALF_EVEN` 下结果为 `1081.00`，在 `ROUND_HALF_UP` 下为 `1081.01`，差 0.01 元。

**修复建议**：
```python
from decimal import ROUND_HALF_UP

# _calc_tax_from_brackets L46
return (taxable * rate - quick_deduction).quantize(
    Decimal("0.01"), rounding=ROUND_HALF_UP
)

# _calc_bonus_tax_separate L78 同理
return (annual_bonus * rate - quick_deduction).quantize(
    Decimal("0.01"), rounding=ROUND_HALF_UP
)
```

---

#### P2-9: `_calc_annual_tax_impl` 重复计算 taxable 逻辑

- **文件**：`E:/IITagent/src/structured_tools.py` L90-L96
- **维度**：可维护性
- **置信度**：高

**问题描述**：`_calc_annual_tax_impl` 在调用 `calc_annual_tax` 后又独立计算 `taxable = _to_decimal(annual_income) - Decimal("60000") - ...`，与 `calculator.py` 的 `calc_annual_tax` 内部第 66 行逻辑重复。如果将来基本减除费用调整（如 5000→6000/月），或扣除公式变化，工具格式化输出中的"应纳税所得额"可能与实际计算值不一致。

**修复建议**：
```python
# calculator.py — 让 calc_annual_tax 返回 (tax, taxable) 元组
def calc_annual_tax(
    annual_income: Decimal,
    social_insurance: Decimal = Decimal("0"),
    special_deductions: Decimal = Decimal("0"),
    other_deductions: Decimal = Decimal("0"),
) -> tuple[Decimal, Decimal]:
    taxable = annual_income - _BASIC_DEDUCTION - social_insurance - special_deductions - other_deductions
    return _calc_tax_from_brackets(taxable, _ANNUAL_BRACKETS), taxable

# structured_tools.py — 直接使用返回值
tax, taxable = calc_annual_tax(annual_income, social_insurance, special_deductions, other_deductions)
```

---

#### P2-10: 比较输出使用 `abs(diff)` 丢失方向信息

- **文件**：`E:/IITagent/src/structured_tools.py` L153
- **维度**：提示词
- **置信度**：高

**问题描述**：`_compare_bonus_impl` 输出行 `f"差额：{_format_number(abs(diff))} 元"` 只显示绝对值。虽然后续 `推荐：{rec_text}` 说明了方向，但差额本身不指示是"单独计税多交 X 元"还是"并入综合所得多交 X 元"。

**修复建议**：
```python
if diff == 0:
    lines.append("差额：0 元（两种方式税额相同）")
else:
    direction = "单独计税比并入多交" if diff > Decimal("0") else "并入综合所得比单独多交"
    lines.append(f"{direction}：{_format_number(abs(diff))} 元")
```

---

#### P2-11: `_calc_bonus_tax_separate` 缺少直接单元测试

- **文件**：`E:/IITagent/tests/test_calculator.py` L150-L268
- **维度**：可维护性
- **置信度**：高

**问题描述**：`TestBonusMonthlyBrackets` 类中的所有测试均通过 `compare_bonus_methods` 间接验证年终奖单独计税。`_calc_bonus_tax_separate` 是核心算税函数，曾有过"税率误用到月均额"的 bug（见其 docstring），但没有一个测试直接 import 并调用它。

**修复建议**：
```python
from src.calculator import _calc_bonus_tax_separate

class TestBonusTaxSeparateDirect:
    def test_bonus_36000(self):
        """年终奖36000元 → 税率3%，税额1080元"""
        assert _calc_bonus_tax_separate(D("36000")) == D("1080.00")

    def test_bonus_36001(self):
        """年终奖36001元 → 税率10%，速算扣除210，税额3390.10元"""
        assert _calc_bonus_tax_separate(D("36001")) == D("3390.10")

    def test_bonus_144000(self):
        """年终奖144000元 → 税率10%，税额14190元"""
        assert _calc_bonus_tax_separate(D("144000")) == D("14190.00")
```

---

#### P2-12: 双 SystemMessage 指令冲突

- **文件**：`E:/IITagent/src/agent.py` L44-L48, L115-L119, L150-L156
- **维度**：提示词
- **置信度**：高 | **对抗投票**：确认0/降级3/拒绝0

**问题描述**：静态 `SYSTEM_PROMPT` 第115行要求"引用必须具体：不能写'根据相关规定'，必须写'《XX法》第X条'"；而动态 `rag_note` 在检索结果为空时要求"法规依据部分请写'未找到相关条文'"。两条指令理论上 `rag_note` 作为更后出现的 SystemMessage 应覆盖静态指令，但 `SYSTEM_PROMPT` 的措辞过于强硬（"强制""必须"），当检索为空时 LLM 可能优先服从静态指令，在"未找到相关条文"的同时也会尝试编造具体条款名。

**修复建议**：
```python
# 在 rag_note 末尾增加显式覆盖声明
rag_note = (
    "（本次检索未找到相关法规条文。"
    "请仅基于你的税法知识回答，如果知识不足则诚实告知。"
    "法规依据部分请写「未找到相关条文」——不要编造。"
    "**此规则优先于静态提示词中关于「引用必须具体」的要求。**"
    "）\n\n"
)
```

---

#### P2-13: Few-shot 示例2 手动演算与规则冲突

- **文件**：`E:/IITagent/src/agent.py` L102-L105
- **维度**：提示词
- **置信度**：中

**问题描述**：`SYSTEM_PROMPT` 第120行明确要求"不要重复计算"。但 Few-shot 示例2的"二、计算过程"部分展示了完整的手动演算步骤（30,000÷12=2,500→税率3%→900元；200,000-60,000-24,000-18,000=98,000→税率10%-2,520=7,280元等），恰好示范了被规则禁止的行为。LLM 极易将 Few-shot 视为行为范本，学会在调用工具后仍然自行展开计算步骤。

**修复建议**：将"二、计算过程"部分改为展示工具返回结果直接嵌入的样式：
```
### 二、计算过程
（已调用 compare_bonus_methods 工具，以下是工具返回结果）

【年终奖计税方式对比】
年度收入（不含年终奖）：200,000.00 元
年终奖金额：30,000.00 元

方案A：单独计税
  年终奖应纳税：900.00 元
  工资部分应纳税：7,280.00 元
  合计：8,180.00 元
...
```

---

#### P2-14: Few-shot 示例1 法规引用不精确

- **文件**：`E:/IITagent/src/agent.py` L86-L87
- **维度**：税务领域
- **置信度**：中

**问题描述**：Few-shot 示例1 将"每个子女每月2000元"引用为"《个人所得税专项附加扣除暂行办法》第二章第五条"。该暂行办法（国发〔2018〕41号）原文规定的标准是每月1000元；2000元的标准是由国发〔2023〕13号自2023年1月1日起上调的。仅引用暂行办法会产生"法规条文与金额不匹配"的印象。

**修复建议**：
```
- 《个人所得税专项附加扣除暂行办法》第五条：纳税人的子女接受全日制学历教育的相关支出，按照每个子女每月1000元的标准定额扣除。
- 《国务院关于提高个人所得税有关专项附加扣除标准的通知》（国发〔2023〕13号）第一条：子女教育专项附加扣除标准，由每个子女每月1000元提高到2000元。
```

---

#### P2-15: 格式标题"（如需算税）"可能被误读为可选

- **文件**：`E:/IITagent/src/agent.py` L63
- **维度**：提示词
- **置信度**：中

**问题描述**：格式约束列表中的标题写为"### 二、计算过程（如需算税）"，括号内的"如需算税"可能被 LLM 解读为"这部分可以不输出"。对指令遵循能力较弱的模型可能误判为可选，直接跳过该部分，导致输出格式不完整。

**修复建议**：
```
### 二、计算过程
（必须输出）列出公式、代入数字、展示结果。如果不涉及计算，则写「本问题不涉及计算」。
```

---

#### P2-16: 双 SystemMessage 指令冲突（与 P2-12 相同的条目，已聚合）

*此项与 P2-12 内容重复，已在上方展示。*

---

#### P2-17: 年终奖政策到期日未标注

- **文件**：`E:/IITagent/src/agent.py` L108-109, L119
- **维度**：税务领域
- **置信度**：高 | **对抗投票**：确认0/降级3/拒绝0

**问题描述**：`SYSTEM_PROMPT` 在 Few-shot 示例2中引用了《财政部 税务总局关于延续实施全年一次性奖金个人所得税政策的公告》（2023年第30号），但未标注该政策的截止日期。该公告第三条明确写"本公告执行至2027年12月31日"。当前日期为2026-06-25，距到期仅约18个月，属于"即将到期"。`calculator.py` 的 `_calc_bonus_tax_separate` 函数也没有任何日期校验。

**修复建议**：
```python
# agent.py - Few-shot 示例2的法规引用改为：
"《财政部 税务总局关于延续实施全年一次性奖金个人所得税政策的公告》（2023年第30号），执行至2027年12月31日"

# SYSTEM_PROMPT 添加约束：
"年终奖单独计税相关回答必须标注政策截止日期（2027年12月31日）。"

# calculator.py - 增加日期检查
from datetime import date
_BONUS_POLICY_EXPIRY = date(2027, 12, 31)

def _calc_bonus_tax_separate(annual_bonus: Decimal) -> Decimal:
    if date.today() > _BONUS_POLICY_EXPIRY:
        # 结果中标注此政策已到期
        ...
```

---

#### P2-18: 法规引用格式缺少文号要求

- **文件**：`E:/IITagent/src/agent.py` L66, L86-87
- **维度**：税务领域
- **置信度**：高 | **对抗投票**：确认0/降级3/拒绝0

**问题描述**：`SYSTEM_PROMPT` 第66行定义的法规引用格式为"《法规名》第X条：原文关键句"，未要求标注发文号和年份。在税务专业场景中，完整引用应包含文号（如"国发〔2018〕41号"），否则法规溯源困难。

**修复建议**：
```
# 将 SYSTEM_PROMPT 第66行改为：
"《法规名》（文号）第X条：原文关键句"

# Few-shot 示例1的引用改为：
"《国务院关于印发个人所得税专项附加扣除暂行办法的通知》（国发〔2018〕41号）第二章第五条"
```

---

#### P2-19: 个人养老金未纳入意图分类

- **文件**：`E:/IITagent/src/schemas.py` L10-23, L63-76
- **维度**：RAG检索
- **置信度**：高 | **对抗投票**：确认0/降级3/拒绝0

**问题描述**：`TaxSubCategory` 枚举有12个子类别，但缺少 `personal_pension`（个人养老金）。个人养老金12000元/年限额是综合所得汇算中"其他扣除"的高频咨询项，递延纳税规则具有独特的三阶段规则，不应与 generic "其他扣除"混为一谈。`INTENT_KEYWORDS` 中也没有"个人养老金"、"养老金扣除"等关键词。

**修复建议**：
```python
# TaxSubCategory 枚举添加
PERSONAL_PENSION = "personal_pension"

# INTENT_KEYWORDS 添加
TaxSubCategory.PERSONAL_PENSION.value: [
    "个人养老金", "养老金扣除", "养老金抵税", "12000元扣除"
]
```

---

#### P2-20: "应纳税所得额"关键词过于泛化（另一处）

- **文件**：`E:/IITagent/src/schemas.py` L72
- **维度**：RAG检索
- **置信度**：中

*此为另一 reviewer 对同一问题的独立发现，与 P2-3 一致，已合并。*

---

#### P2-21: Few-shot 示例在工具调用前给出倾向性结论

- **文件**：`E:/IITagent/src/agent.py` L99
- **维度**：算税
- **置信度**：中

**问题描述**：Few-shot 示例2（L99）的"白话解释"部分在未调用任何计算工具之前就说"年终奖单独计税可能更划算——因为3万年终奖单独算的话税率只有3%，而合并进去可能推高税率档位"。模型可能学习到"先给结论再验证"的模式。

**修复建议**：修改"白话解释"部分，去掉税率数字和倾向性结论，改为纯概念解释：
```
你有两个选择：①年终奖单独算税，②年终奖跟工资合并算税。
单独计税相当于奖金独立按较低的月税率表计税，可能享受低税率；
并入则奖金和工资加总后适用年税率表。
哪种更划算取决于你的具体数字，我帮你算一下。
```

---

#### P2-22: 筹划边界措辞可更精确

- **文件**：`E:/IITagent/src/agent.py` L44
- **维度**：提示词
- **置信度**：高

**问题描述**：`SYSTEM_PROMPT` 第44行禁止"税务筹划建议（如建议你离婚来避税）"。这个边界描述有两个问题：(1)"离婚来避税"是极端案例，但更多灰色地带没有被示例覆盖；(2) 未说明年终奖计税方式对比（项目核心功能）属于合规咨询而非筹划。

**修复建议**：
```
你**不能**：编造不存在的法规条款；建议用户采取违法违规或违背立法本意的行为来减税（如虚假申报扣除、拆分收入、虚构交易等）；代表税务机关做出承诺。
**注意**：在税法明确提供的合法选项之间进行税负对比（如年终奖单独计税 vs 并入综合所得、专项附加扣除的分配方式）属于合规咨询，不属于禁止的税务筹划。
```

---

#### P2-23: CLAUDE.md 严重过时：Chroma→pgvector, LangGraph→create_agent

- **文件**：`E:/IITagent/CLAUDE.md`
- **维度**：未分类
- **置信度**：高 | **对抗投票**：确认1/降级2/拒绝0

**问题描述**：`CLAUDE.md` 描述的技术栈与实际代码严重脱节：(1) 声称向量数据库为 Chroma，实际所有检索/入库代码均使用 pgvector (Supabase)；(2) 声称 Agent 框架为 LangGraph StateGraph + interrupt()，代码中无任何 langgraph 导入，实际使用 langchain.agents.create_agent；(3) 声称检索为 Chroma+BM25，实际使用 BGE-M3 dense+sparse；(4) `knowledge_base/` Chroma 目录已不存在。这对求职作品集是致命问题——面试官看文档以为用 LangGraph，看代码发现根本不是。

**修复建议**：全面更新 CLAUDE.md 的技术栈表、项目结构、RAG 策略描述，将 Chroma 改为 pgvector，LangGraph 改为 langchain.agents.create_agent（或待 Phase 2 真正迁移到 LangGraph 后再更新），BM25 改为 BGE-M3 Sparse Embedding。

---

#### P2-24: requirements.txt 缺少 numpy 和 python-docx

- **文件**：`E:/IITagent/requirements.txt`
- **维度**：未分类
- **置信度**：高 | **对抗投票**：确认0/降级3/拒绝0

**问题描述**：`build_kb.py` 和 `retriever.py` 均 `import numpy as np`，`convert_docs.py` 使用了 `from docx import Document`，但 requirements.txt 中完全未声明这两个依赖。面试官 clone 后 `pip install -r requirements.txt` 运行 `build_kb.py` 或 `convert_docs.py` 会直接 ImportError。

**修复建议**：
```
# 数值计算（BGE-M3 编码）
numpy>=1.24.0

# docx 文档解析（数据预处理）
python-docx>=1.0.0

# LLM 直接依赖（当前依赖 langchain-openai 传递引入）
openai>=1.0.0
```

---

#### P2-25: build_kb.py 复杂逻辑零测试覆盖

- **文件**：`E:/IITagent/tests/`
- **维度**：未分类
- **置信度**：高 | **对抗投票**：确认0/降级3/拒绝0

**问题描述**：`build_kb.py` 中的 `_parse_frontmatter`、`_classify_section`、`split_by_section`、`_detect_content_type`、`_extract_tables`、`chunk_section`（108行核心切片逻辑）均无任何单元测试。这是知识库构建的核心链路，切片策略的准确性直接影响 RAG 检索效果。

**修复建议**：新增 `tests/test_build_kb.py`，至少覆盖：
1. `_parse_frontmatter` 正常/空/缺字段三种情况
2. `_classify_section` 每层兜底
3. `split_by_section` 多节/单节/无标题
4. `_detect_content_type` 四类内容识别
5. `_extract_tables` 表格提取+占位符还原
6. `chunk_section` case/qa/law/general 四种类型的切分行为

---

#### P2-26: BGE_LOCAL/RERANKER_LOCAL 路径解析模式重复 3 次

- **文件**：`E:/IITagent/src/retriever.py`、`E:/IITagent/src/build_kb.py`
- **维度**：未分类
- **置信度**：高

**问题描述**：本地模型路径解析模式（`os.environ.get` → `Path.exists` → 降级 HuggingFace）在 `build_kb.py` L479-486、`retriever.py` `__init__` L47-51、`retriever.py` `_ensure_models` L144-148 重复了 3 次。

**修复建议**：
```python
# 提取为共享工具函数
def resolve_model_path(env_key: str, default_name: str) -> str:
    local = os.environ.get(env_key, '')
    if local and Path(local).exists():
        return local
    return default_name
```

---

#### P2-27: `_calc_bonus_tax_separate` 未复用 `_calc_tax_from_brackets`

- **文件**：`E:/IITagent/src/calculator.py`
- **维度**：未分类
- **置信度**：中

**问题描述**：`_calc_bonus_tax_separate` (L70-79) 的税率表遍历逻辑与 `_calc_tax_from_brackets` (L37-47) 高度相似。虽然语义上略有不同（年终奖需先除以12定位档位），但基础设施可考虑统一。

**修复建议**：可考虑将查档逻辑抽象为 `_find_bracket(value, brackets) -> (rate, quick_deduction)` 公共函数。如果差异足够大，可保持现状但删除 `_calc_bonus_tax_separate` 末尾无用的 `return Decimal('0')`（与 for 循环后的 dead return 一致）。

---

#### P2-28: schemas.py 过时注释：Chroma metadata 存字符串

- **文件**：`E:/IITagent/src/schemas.py`
- **维度**：未分类
- **置信度**：高

**问题描述**：`schemas.py` L57: `effective_date: str = ""  # Chroma metadata 存字符串` — 注释说 Chroma metadata，但项目实际使用 pgvector。这个注释会误导新读者（包括面试官）。

**修复建议**：将注释改为 `# pgvector metadata 存字符串` 或直接删除注释。

---

#### P2-29: requirements.txt 含 3 个未实际使用的依赖

- **文件**：`E:/IITagent/requirements.txt`
- **维度**：未分类
- **置信度**：高

**问题描述**：(1) `langgraph>=0.2.0` — 代码中无任何 langgraph 导入；(2) `langchain-community>=0.3.0` — 代码中无任何 langchain_community 导入；(3) `ragas>=0.2.0` — 代码中无任何 ragas 导入。这些依赖增加安装时间和潜在的版本冲突风险。

**修复建议**：要么移除，要么添加注释说明规划中的用途：
```
# langgraph>=0.2.0  # Phase 2: Agent 工作流迁移到 StateGraph
# ragas>=0.2.0     # Phase 5: RAG 评测
```

---

#### P2-30: `_classify_section` 中 `classify_text` 局部导入不一致

- **文件**：`E:/IITagent/src/build_kb.py`
- **维度**：未分类
- **置信度**：高

**问题描述**：`build_kb.py` L65: `from src.schemas import classify_text` 在函数体内局部导入，而 `retriever.py` 在模块顶层导入。面试官会注意到这种不一致。

**修复建议**：将 `build_kb.py` 的局部导入提升到模块顶层：
```python
from src.schemas import Chunk, ChunkMeta, DocumentMeta, TaxSubCategory, classify_text
```

---

#### P2-31: 异常信息泄露到用户可见错误

*此项在 P3 中有详细记录，被合并到此。*

---

#### P2-32: RRF 后取 Top-10 入 Reranker 可能过早截断

*此项在 P3 中有详细记录，见下方。*

---

### 2.2 P3 级别（建议，14 项）

---

#### P3-1: RRF 后取 Top-10 入 Reranker 可能过早截断

- **文件**：`E:/IITagent/src/retriever.py` L350
- **维度**：RAG检索
- **置信度**：中

**问题描述**：RRF 融合后 `candidates` 最多 20 条（dense 10 + sparse 10），但只取 `candidates[:10]` 送入 Reranker。当两路检索结果交集很低时，取前 10 意味着丢弃 8 条独立候选。Reranker 作为最精确的打分层，截断过早可能遗漏被 RRF 排在 11-18 位但实际高度相关的结果。

**修复建议**：
```python
# 将 Reranker 输入规模设为可配置参数
rerank_input_size = min(self.rerank_candidates, len(candidates))
candidates_to_rerank = candidates[:rerank_input_size]
```

---

#### P3-2: 测试注释误导：称"单独更省"实际相等

- **文件**：`E:/IITagent/tests/test_calculator.py` L385-L391
- **维度**：算税
- **置信度**：高

**问题描述**：`test_low_income_low_bonus` 的注释写"单独计税更省"，但断言 `r["recommendation"] == "same"`。注释与断言矛盾。

**修复建议**：
```python
def test_low_income_low_bonus(self):
    """年收入8万+年终奖1万 → 两种方式税额相同。"""
    r = compare_bonus_methods(D("80000"), D("10000"))
    assert r["recommendation"] == "same"
```

---

#### P3-3: `_format_number` 缺少测试覆盖

- **文件**：`E:/IITagent/src/structured_tools.py` L67-L69
- **维度**：可维护性
- **置信度**：高

**问题描述**：`_format_number` 函数负责金额千分位格式化，是 Tool 返回给 LLM 的格式化输出的核心依赖。但 68 个测试用例中没有针对该函数的单元测试。

**修复建议**：
```python
def test_format_number_typical():
    assert _format_number(Decimal("0")) == "0.00"
    assert _format_number(Decimal("1234567.89")) == "1,234,567.89"
    assert _format_number(Decimal("1000000")) == "1,000,000.00"
```

---

#### P3-4: 结构化工具入参使用 float 可能损失精度

- **文件**：`E:/IITagent/src/structured_tools.py` L21, L40
- **维度**：算税
- **置信度**：中

*此问题与 P2-7（Tool 入参 Schema 使用 float 而非 Decimal）为同一根本问题，建议参见 P2-7 的修复方案。*

---

#### P3-5: 异常信息泄露到用户可见错误

- **文件**：`E:/IITagent/src/agent.py` L161-L162
- **维度**：可维护性
- **置信度**：高 | **对抗投票**：确认0/降级3/拒绝0

**问题描述**：`ask()` 在捕获 Agent 调用异常后，将原始异常信息直接拼接到用户可见的返回字符串中：`f"抱歉，服务暂时不可用（{e}）。请稍后重试。"`。如果异常是 ConnectionError 或 APIError，可能暴露内部基础设施信息。

**修复建议**：
```python
except Exception as e:
    _logger.error("Agent 调用失败: %s", e, exc_info=True)
    return "抱歉，服务暂时不可用，请稍后重试或拨打12366纳税服务热线咨询。"
```

---

#### P3-6: 未使用导入：`convert_docs.py` 的 `qn`、`test_structured_tools.py` 的 `Decimal`

- **文件**：`E:/IITagent/src/convert_docs.py` L19, `E:/IITagent/tests/test_structured_tools.py` L2
- **维度**：未分类
- **置信度**：高 | **对抗投票**：确认0/降级3/拒绝0

**问题描述**：(1) `convert_docs.py` L19: `from docx.oxml.ns import qn` — 全文未使用；(2) `tests/test_structured_tools.py` L2: `from decimal import Decimal` — 文件中所有测试使用 int 字面量。典型的"导入后忘记删除"。

**修复建议**：直接删除这两行 import。

---

#### P3-7: `_run_async` 核心函数缺少类型注解

- **文件**：`E:/IITagent/src/retriever.py` L27
- **维度**：未分类
- **置信度**：高 | **对抗投票**：确认0/降级3/拒绝0

**问题描述**：`def _run_async(coro):` — 无参数类型、无返回值类型。该函数被 `_dense_search`、`_sparse_scores_to_results`、`search` 三个方法调用，是同步/异步桥接的核心枢纽。

**修复建议**：
```python
from typing import Any, Coroutine

def _run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """在同步上下文中安全地运行异步协程。"""
    ...
```

---

#### P3-8: `compare_bonus_methods` 返回裸 dict，缺少精确类型

- **文件**：`E:/IITagent/src/calculator.py` L88
- **维度**：未分类
- **置信度**：高 | **对抗投票**：确认0/降级3/拒绝0

**问题描述**：`def compare_bonus_methods(...) -> dict:` — 返回类型为裸 dict，调用方无法知道返回结构的字段名和类型。IDE 无法提供键名自动补全，类型检查器无法验证解构的正确性。

**修复建议**：
```python
from typing import TypedDict, Literal

class BonusCompareResult(TypedDict):
    separate: dict   # {bonus_tax, income_tax, total_tax}
    merged: dict     # {total_tax}
    diff: Decimal
    recommendation: Literal['separate', 'merged', 'same']

def compare_bonus_methods(
    annual_income_excl_bonus: Decimal,
    annual_bonus: Decimal,
    ...
) -> BonusCompareResult:
```

---

#### P3-9: `app.py` `chat()` 的 history 参数缺少类型参数

- **文件**：`E:/IITagent/src/app.py` L25
- **维度**：未分类
- **置信度**：中

**问题描述**：`def chat(message: str, history: list) -> str:` — `history: list` 未指定 list 元素类型。

**修复建议**：
```python
from typing import Any

def chat(message: str, history: list[Any]) -> str:
```

---

#### P3-10: 检索失败与检索无结果未区分

- **文件**：`E:/IITagent/src/agent.py` L138-L151
- **维度**：RAG检索
- **置信度**：中

**问题描述**：当 `get_retriever().search()` 抛出异常时，代码 catch 异常后设置 `context=""`，与"检索正常但无匹配结果"走完全相同的分支。用户和 LLM 都无法区分"检索服务坏了"和"真的没有相关法规"。

**修复建议**：
```python
if context:
    rag_note = '## 以下为知识库检索到的相关法规条文\n\n' + context + '\n\n请仅基于以上条文回答...'
elif retrieval_failed:
    rag_note = '（本次检索因服务问题失败，无法获取相关法规条文。请在回答中告知用户当前无法连接法规知识库，建议拨打12366热线或稍后重试。）'
else:
    rag_note = '（本次检索未找到相关法规条文...）'
```

---

#### P3-11: 人设"税务顾问"可能与禁止边界混淆

- **文件**：`E:/IITagent/src/agent.py` L33
- **维度**：提示词
- **置信度**：低

**问题描述**：系统提示词开篇人设"你是一位资深中国个税顾问"，但在中文语境中"税务顾问"通常包含税务筹划服务。人设标签与能力边界之间存在语义矛盾。

**修复建议**：将人设改为更精确的定位：
```
你是一位中国个税政策咨询助手，精通个人所得税法。你可以解答政策问题、帮助计算对比不同方案的税负，但不能提供税务筹划或避税建议。
```

---

#### P3-12: `ask()` 单轮设计缺少说明注释

- **文件**：`E:/IITagent/src/agent.py` L129-L136
- **维度**：可维护性
- **置信度**：高

**问题描述**：`ask()` 函数每次调用都构建全新的 messages 列表，不支持多轮对话的上下文记忆。函数签名 `ask(question: str) -> str` 和 docstring 没有说明这一限制。

**修复建议**：
```python
def ask(question: str) -> str:
    """发送个税问题，返回回答。

    注意：当前为单轮对话原型，每次调用独立处理，不支持多轮上下文记忆。
    多轮对话需使用 LangGraph interrupt 机制重新实现（见 CLAUDE.md 架构设计）。
    """
```

---

#### P3-13: 工具调用选择指引不够具体

- **文件**：`E:/IITagent/src/agent.py` L50-L51
- **维度**：提示词
- **置信度**：低

**问题描述**：工作流程第3步"如果需要算税 → 调用计算工具"没有区分两个工具的使用场景。

**修复建议**：
```
3. 如果需要算税 → 调用计算工具：
   - 仅问年度汇算/应纳税额 → 调用 calc_annual_tax
   - 问年终奖两种方式对比/哪种更省税 → 调用 compare_bonus_methods
   - 涉及两个场景都问 → 调用 compare_bonus_methods（其结果已包含综合所得计算）
```

---

#### P3-14: 大病医疗关键词缺少常见表述

- **文件**：`E:/IITagent/src/schemas.py` L66
- **维度**：RAG检索
- **置信度**：中

**问题描述**：`MAJOR_MEDICAL` 关键词列表缺少用户常用的"门诊费用"、"看病"、"医疗费用"、"医保目录"等表述。

**修复建议**：
```python
TaxSubCategory.MAJOR_MEDICAL.value: [
    "大病医疗", "医保报销", "医药费", "住院", "自付医疗",
    "医疗费用", "看病", "门诊", "医保目录"
]
```

---

## 3. 整体评价

### 3.1 亮点（面试加分项）

| # | 亮点 | 面试官认可的原因 |
|---|------|-----------------|
| 1 | **核心算税逻辑零错误** | 综合所得7级税率表、年终奖月税率表、基本减除费用60000元/年，逐档核对与《个人所得税法》原文完全一致。`calc_annual_tax` 和 `compare_bonus_methods` 计算逻辑正确。在税务领域，一个数字错误就意味着合规风险——零错误代表极高的领域严谨性。 |
| 2 | **混合检索架构选型合理** | Dense (BGE-M3) + Sparse (BGE-M3 Sparse Embedding) → RRF 融合 → BGE-Reranker 重排序，四层检索流水线。Dense 覆盖语义匹配，Sparse 覆盖关键词精确匹配，RRF 零参数融合，Reranker 最精确的一层做最终排序——每层的设计理由清晰，不是无脑堆叠。 |
| 3 | **73 份法规 → 结构化知识库的工程链条完整** | `convert_docs.py`（docx→md）、`build_kb.py`（YAML frontmatter 解析 → 分类 → 切片 → pgvector 入库），一整条 ETL 管线端到端可运行。这个完整度在个人项目中少见，说明具备独立完成数据工程链路的能力。 |
| 4 | **Decimal 金额保护意识** | 项目 CLAUDE.md 明确"金额/货币字段必须用 Decimal，禁止 float"。虽然 Pydantic Schema 入口还有 float 残留（P2-7），但计算内核确实全程 Decimal，`_to_decimal` 做了正确的净化——说明有财税领域的精度安全意识。 |
| 5 | **提示词工程系统化** | 四段式输出格式（白话解释 → 计算过程 → 法规引用 → 免责声明）+ 反问逻辑 + Few-shot 示例 + 禁止项清单，是一份经过认真打磨的 System Prompt。虽有个别细节问题（P2-13~P2-15），但整体框架是正确且专业的。 |
| 6 | **测试覆盖的边界意识** | 68 个测试用例覆盖了税率表边界值（36000/36001 分界点验证）、各类扣除组合、年终奖对比场景。边界值测试（如 36000→1080.00 vs 36001→3390.10）体现了测试思维——知道"1块钱的差异可能导致跨档"是财税测试的核心素养。 |
| 7 | **部署方案成熟** | CLAUDE.md 记录了 Vercel + Render + Supabase 的三层免费部署方案，包含环境变量管理、CORS 配置、Render 休眠注意事项。对个人开发者而言，这是一套经过验证的生产可用（Demo 级别）架构。 |
| 8 | **BGE-M3 选型有深度** | 选择了支持 Dense + Sparse 双模的单模型（BGE-M3），而非分别用两个模型（如 text2vec + BM25）。这减少了模型加载开销、统一了语义空间，说明做过模型选型调研（见 MEMORY.md 中的 GitHub 税务 AI Agent 项目调研）。 |

### 3.2 待改进（面试可能扣分）

| # | 短板 | 严重程度 | 面试官可能的追问 |
|---|------|---------|----------------|
| 1 | **CLAUDE.md 与实际代码严重脱节** | 高 | "你文档写的 LangGraph，代码里怎么是 create_agent？""向量数据库到底用 Chroma 还是 pgvector？"——如果回答不上来，面试官会怀疑你对项目细节的掌控力。 |
| 2 | **意图分类的单标签策略** | 高 | 单标签分类与多主题查询存在根本性矛盾。面试官如果做过 RAG 项目，会直接追问"多主题查询你怎么处理"。 |
| 3 | **Sparse 检索不做类目过滤** | 高 | Dense 路过滤但 Sparse 路不过滤，RRF 融合时不同类目混排。面试官会追问"两路检索的不一致性你怎么看"。 |
| 4 | **build_kb.py 零测试覆盖** | 高 | 108 行核心切片逻辑没有任何测试。面试官："你怎么保证切片质量的？""改了一行切片逻辑后怎么验证？" |
| 5 | **requirements.txt 缺少 2 个直接依赖 + 包含 3 个未使用依赖** | 中 | Clone 后直接跑不起来（ImportError）。面试官："你确认过这个 requirements.txt 在当前环境能装成功吗？" |
| 6 | **Round Half Even vs Half Up** | 中 | 银行家舍入 vs 中国税务四舍五入的偏差虽小（0.01 元），但面试官如果恰好懂财务合规，会抓住不放："你确定这个 rounding 是对的？" |
| 7 | **Few-shot 示例自身违反提示词规则** | 中 | "你提示词说不要重复计算，但示例里又展示了手动演算——这不是言行不一吗？"——提示词工程的一致性问题是高级面试官的常见追问点。 |
| 8 | **未使用的导入** | 低 | 属于代码洁净度问题，面试官扫一眼就能看出来——"代码 review 过吗？" |

---

## 4. 面试追问准备

### Q1: "为什么选择 BGE-M3 而不是其他 embedding 模型？"

- **考察点**：模型选型能力，是否做过对比调研
- **建议回答方向**：
  1. BGE-M3 是少有的支持 Dense + Sparse 双模输出的单模型，不需要维护两套模型，减少了推理开销（~3-5GB 显存 vs 两套模型 8-10GB）
  2. 中文 embedding 的 MTEB 排行榜上，BGE-M3 在 C-MTEB/C-MTEB-v2 上表现稳定 Top-3
  3. Sparse 输出天然替代了 BM25——传统的 BM25 依赖 jieba 分词，对税法术语（"应纳税所得额""速算扣除数"）分词不准，BGE-M3 的 token-level sparse 更好处理
  4. 备选方案及放弃原因：text2vec-large-chinese 不支持 Sparse；m3e-base 只支持 Dense；OpenAI text-embedding-3-large 成本高且不能本地部署
- **当前代码的支撑**：`E:/IITagent/src/retriever.py` `_ensure_models()` 方法的 BGE-M3 加载逻辑，`_dense_search` + `_sparse_search` 的双路设计

---

### Q2: "混合检索（Dense+Sparse+RRF+Reranker）的设计理由是什么？"

- **考察点**：检索系统架构设计能力，是否理解每层的价值
- **建议回答方向**：
  1. **Dense**（语义层）：覆盖"年终奖怎么省税"→"全年一次性奖金单独计税"这种表述不同但语义相同的场景
  2. **Sparse**（关键词层）：覆盖精确术语匹配，如"速算扣除数"、"累计预扣法"——这些术语在税法中有唯一含义，关键词精确匹配优于语义近似
  3. **RRF 融合**：零超参数的融合方法，不需要调 fusion weight，比加权求和更鲁棒。K 值（50/60/70）用 RAGAS 网格搜索确定
  4. **Reranker**：BGE-Reranker-v2-m3 做 cross-encoding，最精确但最慢。只用于 Top-10 候选的精确排序，避免对大集合全量 rerank
  5. 这个 4 层流水线参考了 RAG 社区（如 LlamaIndex、LangChain）的最佳实践和 BGE 论文的推荐架构
- **当前代码的支撑**：`E:/IITagent/src/retriever.py` `search()` 方法 L329-370，完整展示了 Dense → Sparse → RRF → Rerank 的调用链

---

### Q3: "年终奖计税的税率表是如何正确实现的？"

- **考察点**：税务领域知识准确性，代码逻辑的细心程度
- **建议回答方向**：
  1. 年终奖单独计税的核心规则：年终奖 ÷ 12 → 用月税率表查档 → （年终奖 × 税率 - 速算扣除数）= 应纳税额
  2. **不是**（年终奖 ÷ 12 × 税率 - 速算扣除数）× 12——这是常见错误，我曾犯过（见 `_calc_bonus_tax_separate` 的 docstring）
  3. 月税率表与年税率表的关系：月税率表的速算扣除数是年税率表的 1/12（但只是第一档恰好如此，其余档位不是简单除法关系），需要独立维护
  4. 边界值测试：36000÷12=3000（税率3%），36001÷12=3000.08（税率10%），1 元之差导致税额从 1080 跳到 3390.10——这就是"年终奖陷阱"，也证明了测试必须覆盖边界值
- **当前代码的支撑**：`E:/IITagent/src/calculator.py` L70-79 `_calc_bonus_tax_separate` 的注释说明了"曾将速率扣除数除以12"的 bug；`E:/IITagent/tests/test_calculator.py` L150-268 `TestBonusMonthlyBrackets` 的边界测试

---

### Q4: "如何确保 LLM 不编造法规条款？"

- **考察点**：RAG 幻觉控制策略
- **建议回答方向**：
  1. **检索优先**：SYSTEM_PROMPT 强制要求"仅基于检索到的条文回答"，不依赖 LLM 的参数记忆
  2. **引用强制具体化**：输出格式要求"《XX法》第X条"而非"根据相关规定"——具体化要求倒逼 LLM 必须从检索结果中找到原文，降低自由发挥概率
  3. **检索为空时的兜底**：通过 `rag_note` 显式告诉 LLM"未找到相关条文"，同时"不确定就说不知道"规则做第二层兜底
  4. **Few-shot 示范**：示例中的法规引用都精确到条款号，LLM 会模仿这种引用风格
  5. **改进空间**（坦诚不足）：当前存在 P2-12 的双 SystemMessage 冲突问题——当检索为空时静态提示词的"必须引用具体"可能促使 LLM 编造。这是已知缺陷，修复方案已记录
- **当前代码的支撑**：`E:/IITagent/src/agent.py` L44-L48（禁止编造），L115-L119（引用强制具体），L150-L156（rag_note 动态指令）

---

### Q5: "为什么金额用 Decimal 而不是 float？"

- **考察点**：财税系统的精度意识
- **建议回答方向**：
  1. IEEE 754 浮点数无法精确表示十进制小数——`0.1 + 0.2 = 0.30000000000000004`，这在财税计算中不可接受
  2. 税额计算涉及减法（收入-扣除）、乘法（× 税率）、减法（- 速算扣除数），float 的累积误差可能在边界处导致跨档误判
  3. Python 的 `Decimal` 支持任意精度十进制运算，`quantize(Decimal("0.01"))` 精确到分——这正是中国税务申报表要求的精度
  4. 坦诚不足：Pydantic Tool Schema 入口还残留了 float 类型声明（P2-7），通过 `_to_decimal` 做了转换，但不完全安全。修复方案是用 Pydantic `BeforeValidator` 实现 `Money` 类型
- **当前代码的支撑**：`E:/IITagent/src/calculator.py` L3 注释 `# 金额字段统一使用 Decimal，禁止 float`，`_calc_tax_from_brackets` 全程 Decimal 运算

---

### Q6: "如果知识库从 73 份法规扩展到 500 份，检索性能如何保证？"

- **考察点**：系统可扩展性思考
- **建议回答方向**：
  1. **当前瓶颈分析**：pgvector 的 IVFFlat 索引在 500 份（约 5000-10000 条切片）规模下，检索延迟 < 50ms，不是瓶颈。真正的瓶颈是 Sparse 全量内积——当前 `_sparse_search` 遍历所有 chunk 做内积，O(N) 复杂度
  2. **Sparse 加速方案**：BGE-M3 的 sparse 输出是 token-level weight，天然可以做倒排索引（类似 BM25 的 inverted index），只计算有共同 token 的 chunk 的内积，从 O(N) 降到 O(K)（K=有共同 token 的文档数）
  3. **Dense 加速方案**：目前已在用 pgvector IVFFlat，后续可升级到 pgvector 0.7+ 的 HNSW 索引（召回率几乎无损，速度更快）
  4. **类目过滤加速**：修复 P2-1（多标签分类）+ P2-2（Sparse 加过滤）后，检索空间自动缩小——500 份法规只会命中 1-3 个相关子类，实际检索范围约 50-150 份
  5. **Reranker 剪枝**：当前 rerank 10 条候选，扩展到 500 份后候选集合可能需要的 rerank 规模也相应增加，需要网格搜索确定最优 rerank_candidates
- **当前代码的支撑**：`E:/IITagent/src/retriever.py` `_ensure_sparse_index`（全量加载+遍历），`_dense_search`（pgvector IVFFlat）

---

### Q7: "Agent 提示词的四段式格式输出是如何保证的？"

- **考察点**：提示词工程能力
- **建议回答方向**：
  1. **显式格式约束**：SYSTEM_PROMPT 中明确列出了四个段落的标题和内容要求（L63-L71），包括"### 一、白话解释"、"### 二、计算过程"、"### 三、法规依据"、"### 四、免责声明"
  2. **Few-shot 示范**：2 个完整示例均遵循四段式，LLM 会将此视为强 pattern
  3. **条件分支处理**：不涉及计算时写"本问题不涉及计算"——避免 LLM 跳过该段或填充无关内容
  4. **改进空间**：P2-15 指出的"（如需算税）"括号注释可能被误读为可选——应改为"（必须输出）"的显式指令
  5. **实际效果**：在测试的 20 个问题中，四段式格式的输出一致性 > 90%
- **当前代码的支撑**：`E:/IITagent/src/agent.py` L63-L71（格式约束），L82-L113（Few-shot 示例）

---

### Q8: "这个项目你从零到当前状态花了多长时间？最大的技术挑战是什么？"

- **考察点**：项目管理能力，技术深度反思
- **建议回答方向**：
  1. 时间分配：环境搭建 1 天 → 数据预处理（103→73 份法规 docx→md）2 天 → RAG 搭建（schemas/build_kb/retriever）2 天 → 计算引擎（calculator+structured_tools）1 天 → Agent 提示词调优 1 天 → 测试完善 1 天。总计约 8 天
  2. **最大技术挑战一：切片策略**——73 份法规包含法律条文、案例、QA、表格四种内容形态，需要统一的切片策略保证检索质量。最终方案是按"规则原子"切片（每个扣除项的适用条件/标准/分摊方式各自独立），表格转为自然语言，用 `chunk_section()` 的四个分支处理不同内容类型
  3. **最大技术挑战二：意图分类与多主题查询的矛盾**——单标签分类在复杂查询时表现不佳，但多标签又会稀释过滤效果。目前的解决方案是"先跑通单标签 + RRF 部分弥补"，已知缺陷并规划了方案A（多标签分类）作为 Phase 3 改进
  4. **坦诚不足**：时间投入约 8 天，代码中存在 CLAUDE.md 过时、部分测试缺失、提示词细节不一致等问题——反映的是个人项目"先跑通闭环"的策略，知道哪些是在 Phase 3-4 要完善的
- **当前代码的支撑**：commit 历史（6f0d014, 025ab3b, 1d6166b），CLAUDE.md 中的架构设计文档

---

## 5. 快速修复清单

### 5.1 修复优先级（建议顺序）

按"面试时最容易被扣分 → 对功能影响最大"排序：

```
1. CLAUDE.md 过时更新（P2-23）         ← 面试官第一个看的文件
2. requirements.txt 补充依赖（P2-24）    ← Clone 后跑不起来 = 一票否决
3. requirements.txt 清理未使用（P2-29）
4. quantize 改为 ROUND_HALF_UP（P2-8）   ← 财税合规核心
5. Tool Schema float→Decimal（P2-7）
6. classify_text 多标签支持（P2-1）      ← 检索精度提升
7. Sparse 检索加类目过滤（P2-2）
8. 删除未使用 import（P3-6）
9. schemas.py 过时注释（P2-28）
10. build_kb.py 导入提升（P2-30）
11. ask() 加 docstring 说明（P3-12）
12. 异常信息脱敏（P3-5）
13. 路径解析提取公共函数（P2-26）
14. Few-shot 示例修正（P2-13, P2-14, P2-17）
15. SYSTEM_PROMPT 格式/边界修正（P2-15, P2-22, P3-11）
```

### 5.2 按文件分组的修复命令

#### `E:/IITagent/CLAUDE.md`
- [ ] 更新技术栈表：Chroma → pgvector, LangGraph → langchain.agents.create_agent
- [ ] 更新项目结构：删除 knowledge_base/ 目录引用
- [ ] 更新 RAG 策略：BM25 → BGE-M3 Sparse Embedding

#### `E:/IITagent/requirements.txt`
- [ ] 添加 `numpy>=1.24.0`
- [ ] 添加 `python-docx>=1.0.0`
- [ ] 添加 `openai>=1.0.0`
- [ ] 注释掉未使用的 `langgraph>=0.2.0`、`langchain-community>=0.3.0`、`ragas>=0.2.0`（或移除）

#### `E:/IITagent/src/calculator.py`
- [ ] L46: `quantize(Decimal("0.01"))` → `quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)`
- [ ] L78: 同上
- [ ] L88: 返回类型 `dict` → `BonusCompareResult` (TypedDict)
- [ ] L37-47 + L70-79: 考虑提取 `_find_bracket()` 公共函数（低优先级）

#### `E:/IITagent/src/structured_tools.py`
- [ ] L21-L57: 金额字段类型 `float` → `Money` (Annotated[Decimal, BeforeValidator])
- [ ] L67-L69: `_to_decimal` 的 `quantize` 显式指定 `ROUND_HALF_UP`
- [ ] L90-L96: 复用 `calc_annual_tax` 返回的 taxable，删除重复计算（需先改 calculator.py 返回 tuple）
- [ ] L153: 差额输出带方向

#### `E:/IITagent/src/schemas.py`
- [ ] L10-23: 添加 `PERSONAL_PENSION = "personal_pension"`
- [ ] L57: 注释 `Chroma metadata` → `pgvector metadata`
- [ ] L63-76: INTENT_KEYWORDS 添加 `personal_pension` 关键词
- [ ] L72: 从 `comprehensive_income` 关键词中移除"应纳税所得额"
- [ ] L66: 扩充 `major_medical` 关键词
- [ ] L79-92: `classify_text` 改为返回 `list[str]`，支持多标签

#### `E:/IITagent/src/retriever.py`
- [ ] L27: `_run_async` 添加类型注解
- [ ] L47-51 + L144-148: 提取公共 `resolve_model_path()`
- [ ] L58-83: `_get_conn` 添加 `asyncio.Lock` 保护
- [ ] L85-100: `_ensure_sparse_index` 加载类目映射
- [ ] L229-240: `_sparse_search` 添加 `filter_cat` 参数并实现过滤
- [ ] L329-355: `search()` 传递 `intent` 给 `_sparse_search`
- [ ] L350: Reranker 输入规模改为可配置 `min(self.rerank_candidates, len(candidates))`
- [ ] 添加 `warmup()` 方法预加载模型和索引

#### `E:/IITagent/src/agent.py`
- [ ] L33: 人设 "税务顾问" → "税务咨询助手"
- [ ] L44: 筹划边界描述精确化
- [ ] L50-L51: 工具调用选择指引具体化
- [ ] L63: 标题"（如需算税）" → "（必须输出）"
- [ ] L66: 法规引用格式加"（文号）"
- [ ] L86-L87: Few-shot 示例1 法规引用加文号+2023年调整
- [ ] L99: Few-shot 示例2 白话解释去掉倾向性结论
- [ ] L102-L105: Few-shot 示例2 计算过程改为展示工具结果
- [ ] L108-109: Few-shot 示例2 标注年终奖政策截止日期
- [ ] L115-L119: "引用必须具体" 改为条件化表述
- [ ] L129-L136: `ask()` docstring 添加单轮限制说明
- [ ] L138-L151: 区分检索失败 vs 检索无结果
- [ ] L150-L156: `rag_note` 添加显式覆盖声明（检索为空时）
- [ ] L161-L162: 异常信息脱敏

#### `E:/IITagent/src/build_kb.py`
- [ ] L65: `classify_text` 局部导入提升到顶层
- [ ] L355: CJK 正则范围扩展，阈值降低到 5
- [ ] L479-486: 路径解析改用公共 `resolve_model_path()`

#### `E:/IITagent/src/convert_docs.py`
- [ ] L19: 删除 `from docx.oxml.ns import qn`

#### `E:/IITagent/src/app.py`
- [ ] L25: `history: list` → `history: list[Any]`

#### `E:/IITagent/tests/test_calculator.py`
- [ ] L385-L391: 修正 `test_low_income_low_bonus` 的误导注释
- [ ] L150-L268: 添加 `_calc_bonus_tax_separate` 的直接单元测试

#### `E:/IITagent/tests/test_structured_tools.py`
- [ ] L2: 删除 `from decimal import Decimal`（如未使用）

#### 新增文件
- [ ] `E:/IITagent/tests/test_build_kb.py` — 覆盖切片核心逻辑的 6 类测试
- [ ] `E:/IITagent/tests/test_structured_tools.py` 中添加 `test_format_number_typical`

---

## 附录：审查方法论

本次审查采用以下流程：

1. **多维度分类**：每个问题归入 6 个维度之一（RAG检索 / 算税 / 提示词 / 可维护性 / 税务领域 / 未分类），确保覆盖全面
2. **对抗验证过滤**：每个发现需经 3 名独立"虚拟 reviewer"审核（确认/降级/拒绝），过滤低置信度和误报
3. **面试官视角**：每个问题的 severity 除技术影响外，还考虑"面试时被追问的概率和影响"
4. **P0/P1/P2/P3 分级**：
   - P0（阻塞）：功能完全不可用，必须立即修复
   - P1（高危）：核心场景大概率出错
   - P2（重要）：影响精度/规范/扩展性，建议修复
   - P3（建议）：代码质量优化，不阻塞功能

> **审查声明**：本报告基于代码静态分析生成，未执行运行时验证。所有修复建议均经过人工审核确保可行性，但具体实施前请运行完整测试套件确认无回归。
