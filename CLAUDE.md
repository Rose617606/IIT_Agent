# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概况

**IITAgent（个税智能体）** — 精通中国个税法规的 AI 智能体，提供税前筹划和合规解读。

- **首发领域**：综合所得汇算清缴与专项附加扣除
- **首发高频场景**：年终奖单独计税 vs 并入综合所得的对比
- **目标**：准确回答 20 个典型个税问题（计算 + 法规引用 + 白话解释），准确率 > 80%，无幻觉
- **定位**：决策顾问，不是算税 App — 核心价值是"事前规划"和"可信解释"

## 技术栈

| 模块 | 选型 |
|------|------|
| 语言 | Python 3.11+ |
| Agent 框架 | **LangGraph**（状态图、条件分支、interrupt 反问等待） |
| LLM/RAG 基础设施 | LangChain（Chroma 集成、Tool 基类、消息模型） |
| 向量数据库 | Chroma |
| 关键词检索 | rank_bm25（混合检索的 BM25 部分） |
| 计算器 | Python 纯函数 |
| 前端（原型） | Gradio |
| 记忆持久化（迭代） | SQLite via LangGraph Checkpointer |
| 结构化输出 | Pydantic + LangChain StructuredTool |

### 关键决策：LangGraph vs LangChain

LangGraph 作为 Agent 框架，LangChain 作为基础设施层，两者分层使用：

- **Agent 流程控制**用 LangGraph — `StateGraph` + 条件边 + `interrupt()` 实现反问等待的确定性暂停
- **RAG 检索**用 LangChain — `Chroma` vectorstore、`OpenAIEmbeddings`、BM25 包装
- **LLM 调用**用 LangChain — `BaseChatModel`、`BaseTool`、消息模型

选择 LangGraph 的决定性原因是 **`interrupt()` 机制**：当用户信息不全时，LangGraph 可以在图中硬暂停等待输入后从断点恢复；LangChain AgentExecutor 的线性 Tool-calling 循环无法可靠实现此行为，只能靠提示词祈祷模型配合。

## 项目结构

```
IITagent/
├── .env                          # API Key 等敏感配置
├── requirements.txt
├── data/                         # 政策 PDF/文本原文
├── knowledge_base/               # Chroma 持久化目录
├── src/
│   ├── build_kb.py               # 知识库构建（切片 → 元数据 → 入库）
│   ├── retriever.py              # 混合检索（Chroma + BM25 → RRF 融合）
│   ├── calculator.py             # 算税纯函数
│   ├── structured_tools.py       # Pydantic schema + StructuredTool 包装
│   ├── agent.py                  # Agent 主体（LLM + 工具 + 提示词 + 记忆）
│   └── app.py                    # Gradio 对话界面
├── tests/
│   ├── test_calculator.py
│   ├── test_retriever.py
│   └── test_questions.json       # 标准测试集（20 个问题）
└── README.md
```

## 常用命令

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 安装依赖
pip install -r requirements.txt

# 构建知识库
python src/build_kb.py

# 运行测试
pytest tests/ -v

# 运行单个测试文件
pytest tests/test_calculator.py -v

# 运行 Gradio 前端
python src/app.py

# RAGAS 评测
python -c "from tests.test_questions import run_eval; run_eval()"
```

## 核心架构

### Agent 工作流（LangGraph StateGraph）

```
用户输入 ──→ [意图识别节点] ──→ [检索节点: RAG 查税法]
                                       ↓
                              [参数检查节点: 所需参数是否齐全?]
                              ↙                        ↘
                    不全 → [反问节点]              齐全 → [计算节点]
                      ↓  interrupt() 暂停               ↓
                      等待用户补充后从                    [生成节点:
                      断点恢复，回到参数检查              白话 + 计算 + 法条 + 免责声明]
                                                        ↓
                                                     输出回答
```

节点定义：
- **意图识别** — 提取用户问题中的税种子类，映射到 `tax_subcategory` 用于检索过滤
- **检索** — 调用混合检索工具，返回相关税法切片
- **参数检查** — 条件边：检查计算所需参数是否齐全（收入、扣除项、年终奖金额等）
- **反问** — `interrupt()` 硬暂停，列出缺失信息让用户补充；用户回复后从此节点恢复
- **计算** — 调用 `compare_bonus_methods()` 等计算工具
- **生成** — LLM 综合检索结果 + 计算结果，输出结构化的顾问回答

### RAG 检索策略

1. **切片策略**：按"规则原子"切分 — 每个扣除项的适用条件、扣除标准、分摊方式、起止时间各自独立成块；数字不被切碎
2. **元数据**：每切片带 `tax_subcategory`、`document_source`、`effective_date`、`is_expired`、`keywords`
3. **混合检索**：Chroma（语义 Top-5）+ BM25（关键词 Top-5）→ RRF 融合排序
4. **元数据预过滤**：先识别用户意图 → 映射到 `tax_subcategory` → 检索时加过滤条件
5. **生成约束**：强制引用出处、检查时效性、末尾追加免责声明
6. **评测**：RAGAS 评估 `context_precision`、`context_recall`、`faithfulness`、`answer_relevancy`

### 计算引擎核心函数

- `calc_annual_tax()` — 综合所得年度汇算
- `compare_bonus_methods()` — 年终奖单独计税 vs 并入综合所得对比

金额字段统一使用 `Decimal`，禁止 `float`。

### Agent 系统提示词核心要素

- **人设**：资深税务顾问
- **反问逻辑**：信息不全时暂停计算，引导用户补充缺失参数
- **输出风格**：先白话解释 → 再计算 → 最后引用法规条款
- **合规要求**：所有建议可溯源，末尾追加免责声明

## 关键设计原则

- **先跑通再优化**：用最简代码实现端到端闭环，再逐步加固
- **评测驱动开发**：每次改动用 20 题测试集验证，防止退化
- **TDD**：先写测试定义行为，再写实现
- **所有建议必须可溯源**：强制引用法规条款，杜绝模型自由发挥

## 开发流程

遵循全局 6 阶段 SOP（`~/.claude/rules/workflow.md`）：
PRD → 技术方案 → 任务拆解 → 编码（小步 TDD）→ 测试验证 → 评审 → 发布

每阶段完成后停止，等待确认再继续。

当前项目处于 **Phase 0：环境准备**。

## 提交规范

- 中文 Conventional Commits：`feat:` / `fix:` / `refactor:` / `docs:` / `test:` / `chore:`
- 一个 commit 只做一件事
- 提交前必须运行测试验证
- PR 粒度控制在 400 行以内
