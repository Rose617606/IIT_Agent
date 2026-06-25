# Phase 1 技术方案：知识库构建与混合检索

> 对应功能：知识库构建（build_kb.py）+ 混合检索引擎（retriever.py）
> 版本：v1.3 | 日期：2026-06-25 | 变更：切片分类策略 — 完整节分类+统一关键词表（建库与检索共用 schemas.INTENT_KEYWORDS）

---

## 1. 目标

将个税政策文档（文本格式）切分成可检索的"规则原子"块，存入 Supabase pgvector（托管 PostgreSQL 向量扩展），并提供混合检索接口供 Agent 调用。

**验收标准**：
- 20 个测试查询的检索召回准确率 > 80%（Phase 5 评测）
- 检索延迟 < 2s
- 检索结果可溯源（每条结果带文档出处 + 生效日期）

---

## 2. 文档格式约定

原始政策文件存放于 `data/`，格式为 Markdown 纯文本（手动整理）。

每个文档开头有 YAML frontmatter 声明元数据：

```yaml
---
title: 个人所得税专项附加扣除暂行办法
source: 国发〔2018〕41号
effective_date: 2019-01-01
status: active
---
```

文档内容按章节组织，用 `##` 标记税种子类。

---

## 3. 切片策略设计

### 3.1 核心原则：规则原子

**一个切片 = 一条可独立回答问题的规则**。

反面示例（❌ 过大）：
> 子女教育专项附加扣除：纳税人的子女接受全日制学历教育的相关支出，按照每个子女每月2000元的标准定额扣除。父母可以选择由其中一方按扣除标准的100%扣除，也可以选择由双方分别按扣除标准的50%扣除，具体扣除方式在一个纳税年度内不能变更。

正面示例（✅ 原子化切分）：

| 切片 | 内容 |
|------|------|
| 切片1 | 子女教育扣除的适用条件：纳税人子女接受全日制学历教育 |
| 切片2 | 子女教育扣除标准：每个子女每月2000元 |
| 切片3 | 子女教育分摊方式：一方100%或双方各50%，年度内不可变更 |

### 3.2 切片规则

1. **语义边界优先（最高原则）**：以文档结构（`\n\n` 段落 → `；`/`。` 分句）作为主切分依据。能按完整语义独立成段的，绝不强制切碎 — 不设硬性字符数上限。
2. **数字保护**：包含金额、百分比、日期的句子不从中切分 — 数字和单位必须在同一切片。
3. **自适应长度控制**：
   - 最小切片：30 字（低于此值的碎片丢弃）
   - 软上限：500 字（大部分法规条文在此范围内可自然成段）
   - 超长段落兜底：仅当单条规则天然超过软上限时，才用 `RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=80)` 兜底切分
   - **设计理由**：中国税法文档差异极大（财税公告几行 vs 个税法数千字），固定上限会削足适履。优先保证"一条规则 = 一个切片"的完整性，字符数仅作软约束。
4. **表结构特殊处理**：Markdown 表格整体保留为一个切片（不切分表格行）。
5. **章节标题保留**：每个切片前面加上所属章节标题作为上下文前缀。

> **备选参数**：软上限和 overlap 可配置（`chunk_section()` 的函数参数），Phase 4 评测时用 RAGAS 网格搜索最优值。

### 3.3 切分流程（含分类）

```
加载 MD 文件 → 解析 YAML frontmatter → 提取全局元数据
    ↓
按 ## 标题分割为"节"（每节 = 完整语义段落）
    ↓
【分类阶段 — 对完整节打 tax_subcategory 标签】
  优先级：章节标题 > 整节内容 > frontmatter title > 文件名 > 兜底 comprehensive_income
  使用 schemas.INTENT_KEYWORDS 统一关键词表（建库与检索共用）
    ↓
每节切分 → 所有子切片继承该节的 tax_subcategory（保证标签一致性）
    ↓
数字保护检查 → 过长的段落用 RecursiveCharacterTextSplitter 兜底
    ↓
生成切片级元数据 → 拼接章节前缀 → 回填 document_source/effective_date → 存入 pgvector
```

**关键设计决策（v1.3）：分类必须在完整节上做，不在碎片上做。**
- 切片正文已被切碎（如"一个纳税年度内不可变更"），独立的碎片扫不出业务关键词
- 完整节内容语义连贯，分类准确性远高于碎片扫描
- 一节内所有子切片继承统一标签，避免"同节三碎片各打不同标签"的混乱
- 建库和检索共用 `schemas.INTENT_KEYWORDS`，一份关键词表，定义一次

---

## 4. 元数据 Schema

### 4.1 全局元数据（每份文档）

```python
class DocumentMeta(BaseModel):
    title: str                    # 法规标题
    source: str                   # 文号，如 "国发〔2018〕41号"
    effective_date: date          # 生效日期
    status: Literal["active", "amended", "expired"]
```

### 4.2 切片级元数据（每个切片）

```python
class ChunkMeta(BaseModel):
    chunk_id: str                 # UUID，唯一标识
    tax_subcategory: str          # 税种子类，见附录
    document_source: str          # 继承自 DocumentMeta.source
    effective_date: date          # 继承自 DocumentMeta.effective_date
    is_expired: bool = False      # 是否已失效
    section_title: str            # 所属章节标题
    chunk_index: int              # 在文档内的序号
```

### 4.3 Chroma 存储方式

| Chroma 字段 | 来源 |
|-------------|------|
| `id` | `chunk_id` |
| `document` | 拼接后的文本（章节前缀 + 切片正文） |
| `metadata` | 所有 ChunkMeta 字段（不含 keywords） |

---

## 5. 知识库构建器（build_kb.py）

### 5.1 模块架构

```
build_kb.py
├── load_documents()          # 加载 data/*.md，解析 YAML frontmatter
├── split_by_section()        # 按 ## 标题分割为节，映射到 tax_subcategory
├── chunk_section()           # 按规则原子切分单节
├── _init_db()                # 初始化 pgvector 扩展 + 建表（Supabase）
├── _store_chunks()           # BGE-M3 编码 → 批量写入 pgvector（Dense + Sparse）
└── build_kb()                # 入口：编排上述流程
```

> 不再单独提取关键词 — BGE-M3 的 Sparse 向量 + BGE-Reranker 已覆盖关键词匹配和精排，额外标注关键词收益递减。

### 5.2 函数签名

```python
def load_documents(data_dir: str = "data/") -> list[Document]:
    """加载所有 Markdown 文档，解析 frontmatter，返回 Document 对象列表"""

def split_by_section(doc: Document) -> list[Section]:
    """按 ## 标题拆分文档为节，每节带 tax_subcategory 标签"""

def chunk_section(section: Section, 
                  min_chunk_size: int = 30,
                  max_chunk_size: int = 300,
                  chunk_overlap: int = 50) -> list[Chunk]:
    """对单节执行规则原子切分，返回 Chunk 列表"""

def build_chroma(chunks: list[Chunk], 
                 persist_dir: str = "knowledge_base/",
                 embedding_model: BGEM3FlagModel = None) -> Chroma:
    """初始化 Chroma 向量存储，使用 BGE-M3 编码，批量插入切片，返回 vectorstore 实例"""

def build_kb(data_dir: str = "data/", persist_dir: str = "knowledge_base/") -> Chroma:
    """入口函数：加载 BGE-M3 → 加载文档 → 切分 → embedding → 入库，返回可检索的 vectorstore"""
```

---

## 6. 混合检索引擎（retriever.py）

### 6.0 模型选型（参考社区实测 + DeepTax 双引擎架构）

| 层 | 模型 | 用途 | 选型依据 |
|----|------|------|---------|
| Embedding | **BGE-M3** (568M, MIT) | Dense + Sparse 双路向量化 | 唯一同时输出 Dense（语义）+ Sparse（关键词）的开源模型，一个模型替代"向量检索+BM25" |
| Reranker | **BGE-Reranker-v2-m3** (568M, MIT) | 候选精排 Cross-Encoding | 与 BGE-M3 同生态，`FlagEmbedding` 一站式加载 |

**社区实测数据支撑（财税/法律场景）**：

| 能力 | BGE-M3 | Qwen3-Emb-0.6B | Qwen3-Emb-4B | 谁更适合我们 |
|------|--------|---------------|-------------|-------------|
| 专业术语泛化 | 82.1% | — | **88.2%** | Qwen 胜 |
| 法律条款检索 | 69.5 | — | **71.2** | Qwen 胜 |
| 混合召回 (Dense+Sparse) | ✅ 原生双路 | ❌ 仅 Dense | ❌ 仅 Dense | **BGE-M3 独有（决定因素）** |
| 显存占用 | 5GB | **1.3GB** | 6.8GB | Qwen-0.6B 最轻 |
| 生态成熟度 | ✅ LangChain/LlamaIndex 原生支持 | 成长中 | 成长中 | BGE-M3 胜 |

**为什么选 BGE-M3 而不是 Qwen3**：
- Qwen3 虽然语义精度更高，但仅输出 Dense 向量，需额外引入 BM25 做关键词互补
- BGE-M3 的 Dense+Sparse 双路输出**一个模型搞定混合检索**，技术栈更简洁
- 社区实测和罗格 DeepTax 都验证了税务 RAG 必须做混合召回 — BGE-M3 在此场景下是原生最优解

**参考来源**：
- 罗拉 DeepTax 双引擎：RAG 检索引擎 + DeepSeek 推理引擎 → 准确率 99.7%
- 社区 20 模型财税知识库断网实测：混合召回 + Reranker 是必选项

### 6.1 检索架构（三层递进）

```
用户查询
    │
    ▼
意图识别（提取 tax_subcategory）→ 元数据预过滤
    │
    ▼
BGE-M3 编码查询
    ├──→ Dense 向量 → Chroma 语义检索 → Top-10
    └──→ Sparse 向量 → 关键词精确匹配 → Top-10
                ↓
    RRF 融合 (k=60) → Top-10 候选集
                ↓
    BGE-Reranker-v2-m3 精排 → Top-5
                ↓
        返回最终结果 + 溯源信息
```

**三层递进**：
1. **BGE-M3 Dense** — 语义覆盖，找到意思相近的条款（如"月薪两万"匹配到"工资薪金所得适用超额累进税率"）
2. **BGE-M3 Sparse** — 关键词精确匹配，抓到专业术语（如"专项附加扣除""子女教育"等固定表述）
3. **BGE-Reranker** — 质量把关，Cross-Encoding 打分过滤语义相近但实际不相关的噪音

### 6.2 函数签名

```python
class RetrievalResult(BaseModel):
    chunk_id: str
    content: str
    tax_subcategory: str
    document_source: str
    effective_date: date
    is_expired: bool
    score: float                      # 最终分数（RRF 或 Reranker 打分）

def classify_intent(query: str) -> str | None:
    """快速意图分类：关键词匹配 → tax_subcategory，无匹配返回 None（全量检索）"""

def encode_query(query: str, model: BGEM3FlagModel) -> dict:
    """BGE-M3 编码查询 → {'dense': np.ndarray, 'sparse': dict}"""
    # 注意：dense 向量需 L2 normalize（BGE-M3 已默认归一化）

def dense_search(dense_vec: np.ndarray,
                 vectorstore: Chroma,
                 k: int = 10,
                 filter: dict | None = None) -> list[RetrievalResult]:
    """Dense 向量 → Chroma 语义检索"""

def sparse_search(sparse_vec: dict,
                  sparse_index: dict,    # {chunk_id: {token_id: weight}}
                  k: int = 10) -> list[tuple[str, float]]:
    """Sparse 向量 → 与预计算的全量 Sparse 索引做点积匹配"""

def rrf_fusion(dense_results: list[RetrievalResult],
               sparse_results: list[RetrievalResult],
               k: int = 60) -> list[RetrievalResult]:
    """RRF 融合两路结果 → Top-10 候选集"""

def rerank(query: str,
           candidates: list[RetrievalResult],
           reranker: FlagReranker,
           top_k: int = 5) -> list[RetrievalResult]:
    """BGE-Reranker Cross-Encoding 精排"""

def hybrid_search(query: str,
                  vectorstore: Chroma,
                  sparse_index: dict,
                  model: BGEM3FlagModel,
                  reranker: FlagReranker,
                  top_k: int = 5) -> list[RetrievalResult]:
    """入口：意图识别 → BGE-M3 编码 → Dense+Sparse 并行检索 → RRF → Reranker → Top-k"""
```

### 6.3 Sparse 索引维护

- `build_kb()` 时用 BGE-M3 对每个切片计算 Sparse 向量，存入 `knowledge_base/sparse_index.json`
- 格式：
  ```json
  {
    "chunk_id_1": {"token_id_1": 0.8, "token_id_2": 0.3},
    "chunk_id_2": {"token_id_3": 0.6, "token_id_4": 0.9}
  }
  ```
- `hybrid_search()` 初始化时加载稀疏索引，查询时用 `sparse_vec · sparse_index[chunk_id]` 内积计算相关性分数

### 6.4 意图识别策略

不依赖 LLM（降低延迟），使用关键词匹配。**关键词表统一定义在 `schemas.py` 的 `INTENT_KEYWORDS`，建库端 `build_kb.py` 和检索端 `retriever.py` 共用同一份，改一处生效两端。**

#### 两层关键词策略

基于 74 份文档实际内容分析，关键词表分两层：

**第一层：首发核心 12 类**（独立子类，直接用于检索预过滤）

| tax_subcategory | 触发词 |
|-----------------|--------|
| child_education | 子女教育、孩子上学、小孩读书、学费扣除、学前教育 |
| continuing_education | 继续教育、学历提升、考证、职业资格、在职教育 |
| major_medical | 大病医疗、医保报销、医药费、住院、自付医疗 |
| housing_loan | 房贷利息、房贷、首套住房、贷款买房、住房贷款 |
| housing_rent | 租房、房租、租金扣除、住房租金 |
| elderly_support | 赡养老人、赡养父母、养老扣除、独生子女老人 |
| infant_care | 婴幼儿、3岁以下、育儿、婴儿照护、幼儿照护 |
| annual_bonus | 年终奖、奖金计税、单独计税、全年一次性奖金 |
| comprehensive_income | 综合所得、工资薪金、劳务报酬、稿酬、应纳税所得额 |
| tax_rate | 税率表、超额累进、速算扣除数、个税税率 |
| annual_settlement | 汇算清缴办法、退税流程、补税、年度申报、年度汇算 |
| basic_deduction | 起征点、基本减除、免征额、6万元、5000元 |

**第二层：扩容储备**（暂不建独立子类，兜底归入 `comprehensive_income`）

文档覆盖但首发场景用不到的专题。特别说明：

- **经营所得**（个税第二大体系）：文档已覆盖个体工商户减半征收（两轮政策+解读）、权益性投资查账征收（财税〔2021〕41号）、经营所得5级超额累进税率表、经营所得申报表（A/B/C表）。首发聚焦综合所得，暂归入 `comprehensive_income`，后续可扩展为独立 `business_income` 子类。
- **其他储备**：股权激励、新三板股息红利、储蓄存款利息、限售股、沪港通/深港通、外籍个人/非居民、远洋船员、粤港澳大湾区/海南自贸港、育儿补贴、公益慈善捐赠、疫情防控等。

> **设计理由**：首发 20 题聚焦综合所得+专项附加扣除+年终奖，建一堆没人查的独立子类只会稀释检索精度。后续需求驱动扩展时，只需在 `schemas.py` 加枚举值+触发词，两端自动生效。

---

## 7. Agent 图增强（调研借鉴）

> 来源：GitHub 调研 — [inflearn-langgraph-agent](https://github.com/jasonkang14/inflearn-langgraph-agent)（韩国所得税 LangGraph Agent）、[naija-tax-ai](https://github.com/philipakintola01-sys/naija-tax-ai)（尼日利亚税法 RAG 实战）

### 7.1 Query Rewrite 节点（检索前）

**问题**：用户口语化查询（"我年终奖怎么搞划算"）与法规书面表述（"全年一次性奖金单独计税"）存在术语鸿沟，直接检索命中率低。

**方案**：在意图识别后、检索前插入 Query Rewrite 节点，用 LLM 将口语改写为精确查询。

```
用户输入: "我年终奖怎么搞划算"
    ↓
意图识别 → "annual_bonus"
    ↓
Query Rewrite → "全年一次性奖金 单独计税 并入综合所得 对比 节税"
    ↓
检索（改写后的查询 + 原查询双路并行）
```

**实现要点**：
- LLM 调用（DeepSeek），低延迟（< 500ms）
- Prompt 约束：仅做术语对齐和关键词扩展，不改变用户意图
- 保留原始查询并行检索，两路结果 RRF 融合（原查询覆盖面 + 改写查询精确度）

### 7.2 Hallucination Check 节点（生成后）

**问题**：LLM 可能编造不存在的法规条款。PRD AC-09 要求"无幻觉 — 不得编造不存在的法规条款"。

**方案**：生成回答后增加 Hallucination Check 节点，逐条验证回答中的论断是否能在检索到的文档中找到原文支撑。

```
生成回答
    ↓
Hallucination Check:
  对回答中每条法规论断 → 在检索上下文(Chroma chunks)中匹配原文
    ├── 全部匹配 → 通过 → 输出回答
    └── 存在不匹配 → 标记幻觉 → 重新生成（prompt 追加"仅使用以下文档中的信息"约束）
                                          ↓
                                   二次校验仍不通过 → 输出带风险提示的回答
```

**实现要点**：
- 结构化校验：从回答中提取「法规引用 → 检索文档」映射对
- 条件边：通过 → 输出；不通过 → 重新生成（最多 2 次）
- 2 次重构后仍不通过：输出回答但附加"以下内容可能包含推测，请以官方文件为准"

### 7.3 Top-K 参数调优计划

**依据**：naija-tax-ai 实战结论 — Top-K 对检索质量的影响 > chunk 参数调参。

**方案**：Phase 4 测试验证时，用 20 题测试集 + RAGAS 跑网格搜索：

```python
# 搜索空间
DENSE_K ∈ {5, 10, 15, 20}
SPARSE_K ∈ {5, 10, 15, 20}
RRF_K ∈ {50, 60, 70}
RERANK_K ∈ {3, 5, 7}

# 评估指标
context_precision, context_recall, faithfulness
```

选出最优组合作为生产参数，写入 `retriever.py` 默认值。

### 7.4 工具描述规范

**依据**：naija-tax-ai 实战教训 — "Tool descriptions on AI Agent nodes are more important than system prompts."

**规范**：后续 `structured_tools.py` 中每个工具的 docstring 必须包含三要素：

```python
def compare_bonus_methods(annual_bonus: Decimal, monthly_salary: Decimal, ...):
    """对比年终奖两种计税方式（单独计税 vs 并入综合所得）。
    
    触发条件：用户询问年终奖怎么计税、哪种方式更省税。
    必需参数：annual_bonus（年终奖金额）、monthly_salary（月薪）
    返回：两种方案的应纳税额、税后收入、差额、推荐方案。
    """
```

---

## 8. 数据流概览

### 8.1 构建流程

```
data/*.md                          ← Markdown 政策文件（含 YAML frontmatter）
    │
    ▼
build_kb.py              
    ├── load_documents()           ← 解析 YAML frontmatter
    ├── split_by_section()         ← 按 ## 分节
    ├── chunk_section()            ← 规则原子切分
    ├── BGE-M3 编码 → Dense + Sparse 向量
    │
    └──→ Supabase pgvector         ← chunks 表（Dense 向量 + Sparse JSONB）
```

### 8.2 检索流程（含 Agent 增强）

```
用户查询: "我月薪2万，租房，能省多少税"
    │
    ▼
hybrid_search()
    ├── classify_intent()    → "housing_rent"
    ├── rewrite_query()      → "住房租金专项附加扣除 月薪20000 应纳税额计算"  ← 新增
    ├── encode_query()       → BGE-M3 出 Dense + Sparse 向量（改写查询）
    ├── encode_query()       → BGE-M3 出 Dense + Sparse 向量（原始查询）  ← 双路并行
    ├── dense_search()       → pgvector cosine distance (filter: housing_rent) → Top-10 × 2
    ├── sparse_search()      → JSONB 稀疏索引点积匹配 → Top-10 × 2
    ├── rrf_fusion()         → 四路融合 → Top-10 候选
    ├── rerank()             → BGE-Reranker 精排 → Top-5
    │
    ▼
Agent 生成回答
    │
    ▼
check_hallucination()         ← 新增：验证回答是否 grounded 在检索文档中
    ├── 通过 → 输出最终回答
    └── 不通过 → 重新生成（最多 2 次）→ 仍不通过则附加风险提示输出
```

---

## 9. 错误处理

| 场景 | 处理方式 |
|------|---------|
| `data/` 目录为空 | 抛出 `FileNotFoundError`，提示先添加政策文件 |
| Markdown 无 frontmatter | 跳过该文件，记录 WARNING 日志 |
| pgvector 连接失败 | 抛出 `RuntimeError`，检查 DATABASE_URL 和网络 |
| 检索时向量库为空 | 返回空列表 `[]`，不抛异常（Agent 层会退化为纯 LLM 回答） |
| 稀疏索引为空 | 跳过 Sparse 检索，只执行 Dense + Reranker |
| BGE-M3 加载失败 | 抛出 `RuntimeError`，检查模型下载和显存/内存 |
| 意图分类无匹配 | 不过滤 `tax_subcategory`，全量检索 |

---

## 10. 依赖

```python
# Embedding + Reranker（BGE 系列，MIT 开源，一个库搞定）
from FlagEmbedding import BGEM3FlagModel   # BGE-M3: Dense + Sparse 双路向量
from FlagEmbedding import FlagReranker     # BGE-Reranker-v2-m3: Cross-Encoding 精排

# 向量数据库（Supabase 托管 pgvector）
from pgvector.asyncpg import register_vector  # pgvector 适配器
import asyncpg                                 # PostgreSQL 异步驱动

# 切片工具
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 结构化
import pydantic

# 标准库（无需额外安装）
import yaml, json, uuid, logging, re, numpy as np
from pathlib import Path
from datetime import date
```

> BGE-M3 本地运行，不调 OpenAI Embeddings API。`FlagEmbedding` 需安装：`pip install FlagEmbedding`。
> 不依赖 `rank_bm25` — BGE-M3 的 Sparse 向量天然替代 BM25 关键词检索。
> Supabase pgvector 需在 SQL Editor 执行 `CREATE EXTENSION IF NOT EXISTS vector;` 开启扩展。

---

## 11. 待确认事项

- [x] Embedding 模型：BGE-M3（Dense+Sparse 双路，替代 BM25）
- [x] Reranker 模型：BGE-Reranker-v2-m3
- [x] 关键词提取：**取消** — BGE-M3 Sparse + Reranker 已覆盖
- [x] BM25：**取消** — BGE-M3 Sparse 输出替代
- [x] LLM API：**DeepSeek** — 中文友好、性价比高，后续 Agent 层使用
- [x] `data/` 政策文件：已完成 — 103 份法规 docx → 73 份结构化 md（2026-06-22）
- [x] 向量数据库：**pgvector（Supabase）** — 替代本地 Chroma，与 Vercel+Render+Supabase 架构统一
- [x] 切片分类策略（v1.3）：**完整节分类 + 统一关键词表** — 在 `split_by_section()` 阶段用完整语义段落扫关键词，子切片继承标签；`schemas.INTENT_KEYWORDS` 建库和检索共用

---

## 附录：tax_subcategory 枚举定义

```python
class TaxSubCategory(str, Enum):
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
```

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-06-15 | 初版：Chroma + BM25 + BGE-M3 |
| v1.1 | 2026-06-23 | BM25 取消（BGE-M3 Sparse 替代），Reranker 从 none → BGE-Reranker-v2-m3 |
| v1.2 | 2026-06-24 | Chroma → pgvector（Supabase），本地零存储依赖 |
| v1.3 | 2026-06-25 | 切片分类策略重构：完整节分类 + 统一关键词表（schemas.INTENT_KEYWORDS），建库与检索共用 |
