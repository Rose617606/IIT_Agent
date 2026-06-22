# Phase 1 技术方案：知识库构建与混合检索

> 对应功能：知识库构建（build_kb.py）+ 混合检索引擎（retriever.py）
> 版本：v1.0 | 日期：2026-06-15

---

## 1. 目标

将个税政策文档（文本格式）切分成可检索的"规则原子"块，存入 Chroma 向量数据库，并提供混合检索接口供 Agent 调用。

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

1. **按段落 + 语义边界切分**：以 `\n\n` 为主要分隔，辅以 `；`、`。` 作为次要分隔
2. **数字保护**：包含金额、百分比、日期的句子不从中切分 — 数字和单位必须在同一切片
3. **最小/最大切片长度**：最小 30 字，最大 300 字。超出 300 字的段落用 `RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)` 作为兜底
4. **表结构特殊处理**：Markdown 表格整体保留为一个切片（不切分表格行）
5. **章节标题保留**：每个切片前面加上所属章节标题作为上下文前缀

### 3.3 切分流程

```
加载 MD 文件 → 解析 YAML frontmatter → 提取全局元数据
    ↓
按 ## 标题分割为"节"（每节 = 一个 tax_subcategory）
    ↓
每节内按段落 + 语义边界递归切分
    ↓
数字保护检查 → 过长的段落用 RecursiveCharacterTextSplitter 兜底
    ↓
生成切片级元数据 → 拼接章节前缀 → 存入 Chroma
```

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
├── build_chroma()            # BGE-M3 编码 → 初始化 Chroma，批量插入切片
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

不依赖 LLM（降低延迟），使用关键词匹配：

| tax_subcategory | 触发词 |
|-----------------|--------|
| child_education | 子女教育、孩子上学、小孩读书、学费扣除 |
| continuing_education | 继续教育、学历提升、考证、职业资格 |
| major_medical | 大病医疗、医保、医药费、住院 |
| housing_loan | 房贷利息、首套住房、贷款买房 |
| housing_rent | 租房、房租、租金扣除 |
| elderly_support | 赡养老人、父母、养老 |
| infant_care | 婴幼儿、3岁以下、育儿 |
| annual_bonus | 年终奖、奖金计税、单独计税 |
| comprehensive_income | 综合所得、汇算清缴、年度汇算 |
| tax_rate | 税率、税率表、速算扣除数 |

匹配原则：长词优先，多命中时取第一个匹配的类别。

---

## 7. 数据流概览

### 7.1 构建流程

```
data/*.md                          ← 手动整理的 Markdown 政策文件
    │
    ▼
build_kb.py              
    ├── load_documents()           ← 解析 YAML frontmatter
    ├── split_by_section()         ← 按 ## 分节
    ├── chunk_section()            ← 规则原子切分
    ├── BGE-M3 编码 → Dense + Sparse 向量
    │
    ├──→ knowledge_base/           ← Chroma 持久化（存 Dense 向量）
    └──→ knowledge_base/sparse_index.json  ← Sparse 索引
```

### 7.2 检索流程

```
用户查询: "我月薪2万，租房，能省多少税"
    │
    ▼
hybrid_search()
    ├── classify_intent() → "housing_rent"
    ├── encode_query()    → BGE-M3 出 Dense + Sparse 向量
    ├── dense_search()    → Chroma (filter: housing_rent) → Top-10
    ├── sparse_search()   → Sparse 索引点积匹配 → Top-10
    ├── rrf_fusion()      → 融合 → Top-10 候选
    ├── rerank()          → BGE-Reranker 精排 → Top-5
    │
    ▼
返回: [RetrievalResult(content="住房租金扣除标准...", score=0.92, source="国发〔2018〕41号"), ...]
```

---

## 8. 错误处理

| 场景 | 处理方式 |
|------|---------|
| `data/` 目录为空 | 抛出 `FileNotFoundError`，提示先添加政策文件 |
| Markdown 无 frontmatter | 跳过该文件，记录 WARNING 日志 |
| Chroma 初始化失败 | 抛出 `RuntimeError`，检查磁盘空间和路径权限 |
| 检索时向量库为空 | 返回空列表 `[]`，不抛异常（Agent 层会退化为纯 LLM 回答） |
| Sparse 索引文件不存在 | 记录 WARNING，只执行 Dense 检索，跳过融合 |
| BGE-M3 加载失败 | 抛出 `RuntimeError`，检查模型下载和显存/内存 |
| 意图分类无匹配 | 不过滤 `tax_subcategory`，全量检索 |

---

## 9. 依赖

```python
# Embedding + Reranker（BGE 系列，MIT 开源，一个库搞定）
from FlagEmbedding import BGEM3FlagModel   # BGE-M3: Dense + Sparse 双路向量
from FlagEmbedding import FlagReranker     # BGE-Reranker-v2-m3: Cross-Encoding 精排

# 向量数据库
from langchain_community.vectorstores import Chroma
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

---

## 10. 待确认事项

- [x] Embedding 模型：BGE-M3（Dense+Sparse 双路，替代 BM25）
- [x] Reranker 模型：BGE-Reranker-v2-m3
- [x] 关键词提取：**取消** — BGE-M3 Sparse + Reranker 已覆盖
- [x] BM25：**取消** — BGE-M3 Sparse 输出替代
- [x] LLM API：**DeepSeek** — 中文友好、性价比高，后续 Agent 层使用
- [ ] `data/` 政策文件：最少 3 份（个人所得税法 + 专项附加扣除暂行办法 + 2023年标准更新 + 汇算清缴公告）

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
