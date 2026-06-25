
export const meta = {
  name: 'code-review-adversarial',
  description: '多维度代码审查 + 对抗式验证，输出面试导向的审查报告（IITagent 定制版）',
  phases: [
    { title: '多维审查', detail: '5 个 Agent 并行，各盯一个维度' },
    { title: '对抗验证', detail: '每个 P0/P1 发现经 3 个反方质疑' },
    { title: '汇总报告', detail: '生成 docs/code-review.md + 面试 Q&A' },
  ],
}

// ── Schema ──────────────────────────────────────────

const FINDINGS_SCHEMA = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string', description: '简短标题（15字以内）' },
          file: { type: 'string', description: '文件路径，如 src/calculator.py' },
          line: { type: 'string', description: '行号或范围，如 "L50-57"' },
          severity: { type: 'string', enum: ['P0', 'P1', 'P2', 'P3'], description: 'P0=致命bug, P1=影响专业印象, P2=改进建议, P3=锦上添花' },
          category: { type: 'string', description: 'RAG检索|算税|提示词|可维护性|税务领域' },
          description: { type: 'string', description: '问题详细描述，包含代码片段引用' },
          fix: { type: 'string', description: '修复建议，包含具体代码改动' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'], description: '确认为真问题的信心' },
        },
        required: ['title', 'file', 'severity', 'description', 'fix', 'confidence'],
      },
    },
  },
  required: ['findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['confirmed', 'downgraded', 'rejected'], description: 'confirmed=确认真问题, downgraded=严重度虚高, rejected=不是真问题' },
    reason: { type: 'string', description: '判决理由，必须引用具体代码或项目背景' },
    new_severity: { type: 'string', enum: ['P1', 'P2', 'P3'], description: '若 downgraded，给出新级别' },
  },
  required: ['verdict', 'reason'],
}

// ── 维度定义 ───────────────────────────────────────

const RAG_PROMPT = `你是 RAG 检索系统专家，聚焦**知识库构建与混合检索的正确性**。审查项目：E:/IITagent — 个税智能体。

## 必须读取的文件
- src/retriever.py（混合检索引擎：Dense+Sparse+RRF+Reranker）
- src/schemas.py（分类体系、INTENT_KEYWORDS、classify_text）
- src/build_kb.py（知识库构建：切片→分类→编码→入库）

## 检查清单
1. **意图分类准确性**：INTENT_KEYWORDS 的关键词覆盖是否全面？12 个子类别的关键词是否有重叠导致歧义？classify_text() 的最长匹配策略是否在所有场景正确？
2. **切片策略**：build_kb 的 split_by_section 是否保证了"规则原子"不被切碎？表格是否正确保留？_classify_section 的 5 层 fallback 顺序是否合理？
3. **混合检索融合**：RRF 的 k=60 参数是否合理？Dense Top-10 + Sparse Top-10 融合后再取 Top-10 给 Reranker 是否足够？
4. **Reranker 降级**：_rerank 在 reranker 为 None 时是否正确降级（当前已修复）？降级后是否影响检索质量？
5. **数据库连接**：_get_conn 的 event loop 检测逻辑是否正确？_conn_loop_id 基于 id(loop) 是否可靠？并发场景是否安全？
6. **稀疏索引加载**：_ensure_sparse_index 的懒加载是否在首次 search() 前完成？from_database() 不预加载后，首次请求是否会因加载延迟超时？
7. **BGE-M3 编码**：_encode_query 的 dense+sparse 编码是否正确？dense 向量维度 1024 是否与 pgvector 索引匹配？
8. **元数据过滤**：search() 的 tax_subcategory 过滤是否正确下推到 SQL？filter_cat=None 时是否正确走全量检索？

## 严重度标准
- P0：检索结果错误（命中无关文档、遗漏关键条款）或 RAG 流程崩溃
- P1：检索质量下降（排序不准、过滤失效）但不致错
- P2：性能或鲁棒性可改进`

const CALC_PROMPT = `你是财务计算引擎专家，聚焦**算税逻辑的正确性和精度**。审查项目：E:/IITagent — 个税智能体。

## 必须读取的文件
- src/calculator.py（算税纯函数：综合所得 + 年终奖对比）
- src/structured_tools.py（LangChain StructuredTool 包装）
- tests/test_calculator.py（68 个测试用例）

## 检查清单
1. **税率表正确性**：_ANNUAL_BRACKETS 的 7 级累进税率、速算扣除数是否与现行个人所得税法一致？_MONTHLY_BRACKETS 的年终奖月税率表是否正确？
2. **基本减除费用**：_BASIC_DEDUCTION = 60000 是否正确（5000/月×12）？专项附加扣除标准是否有遗漏？
3. **年终奖计税**：_calc_bonus_tax_separate 是否正确实现"月均额查档 + 全年奖金计税"？（这是刚修过的 bug——之前把税率应用到了月均额上）
4. **Decimal 精度**：所有金额字段是否使用 Decimal？_D("0.01") 的 quantize 是否会导致舍入误差？compare_bonus_methods 的 diff 计算是否正确？
5. **边界条件**：taxable ≤ 0 返回 0 是否正确？annual_bonus ≤ 0 的处理？扣除项总和 > 收入时税额为 0？
6. **工具返回值格式**：_calc_annual_tax_impl 和 _compare_bonus_impl 的格式化输出是否包含 LLM 需要的所有信息？金额千分位格式（_format_number）是否正确？
7. **测试覆盖**：68 个测试是否覆盖了所有 7 档税率边界？是否有遗漏的边界场景？测试断言是否包含具体数值而非仅结构检查？
8. **StructuredTool 注册**：TOOLS 列表是否正确包含两个工具？工具的 description 是否准确描述触发条件？入参 Schema 的 Field description 是否足够指导 LLM 调用？

## 严重度标准
- P0：计算结果错误——导致用户少交/多交税的误判
- P1：边界条件处理不当、Decimal 精度问题、工具描述误导 LLM
- P2：格式化、文档、测试覆盖改进`

const PROMPT_PROMPT = `你是 LLM 提示词工程专家，聚焦 **Agent 系统提示词的质量和安全性**。审查项目：E:/IITagent — 个税智能体。

## 必须读取的文件
- src/agent.py（Agent 主体：SYSTEM_PROMPT + create_agent + ask）
- src/structured_tools.py（工具的 description 和 docstring）
- src/app.py（Gradio 前端）

## 检查清单
1. **人设与边界**：SYSTEM_PROMPT 定义的"资深中国个税顾问"人设是否清晰？能力边界（能做什么、不能做什么）是否明确禁止了投资建议、税务筹划等越界行为？
2. **格式约束有效性**：四段式强制格式（白话解释→计算过程→法规依据→免责声明）是否足够强硬？Few-shot 示例是否与格式要求一致？
3. **反幻觉机制**：上下文注入的"仅基于以上条文回答"指令是否有效？当检索结果为空时，提示词是否给出明确行为指引（说不知道而非编造）？
4. **双 SystemMessage 冲突**：静态 SYSTEM_PROMPT 要求输出"法规依据"部分，动态 RAG 上下文要求"不要编造"。检索结果为空时两者的矛盾是否已解决？（这是刚修过的问题）
5. **工具调用指引**：提示词是否明确告诉 LLM 何时调用 calc_annual_tax vs compare_bonus_methods？是否强调"调用工具后直接用返回数字，不要重复计算"？
6. **Few-shot 质量**：示例 1（纯知识问答）和示例 2（算税场景）的答案是否稅法正确、格式符合要求？示例金额是否有实际参考价值？
7. **免责声明**：免责声明措辞是否足够（仅供参考/以税务机关为准/12366热线）？是否每次回答都强制输出？
8. **错误处理**：ask() 的异常处理是否对用户友好？检索失败和 Agent 调用失败的错误信息是否有区分？
9. **多轮对话缺失**：当前 ask() 不支持多轮对话（每次构建全新消息列表）。这在原型阶段是否可接受？是否需要加注释说明？

## 严重度标准
- P0：提示词缺陷导致幻觉（编造法规）或越界建议（税务筹划）
- P1：格式约束不够强导致输出不规范、Few-shot 示例有误导
- P2：措辞优化、人设微调`

const MAINTAIN_PROMPT = `你是代码可维护性审查专家，聚焦**代码整洁度与工程规范**。审查项目：E:/IITagent — 个税智能体（求职作品集）。

## 必须读取的文件
- src/ 下所有 .py 文件
- tests/ 下所有测试文件
- requirements.txt
- CLAUDE.md

## 检查清单
1. **未使用导入与死代码**：搜索所有 import 语句，检查是否有导入后未使用的模块。retriever.py 的 get_retriever() 是否被调用？agent.py 是否有重复的 _get_retriever？（刚修过）
2. **类型注解覆盖**：所有公开函数是否有参数和返回值类型注解？_run_async、classify_text、_calc_tax_from_brackets 等是否正确标注？
3. **重复代码**：_calc_bonus_tax_separate 的 bracket 遍历是否与 _calc_tax_from_brackets 重复？BGE_LOCAL/RERANKER_LOCAL 的路径解析模式是否重复了 3 次？
4. **函数长度与复杂度**：search() 是否过长？_ensure_models 的 double-checked locking 是否正确实现？
5. **命名一致性**：私有函数下划线前缀是否统一（_run_async vs _ensure_models vs _calc_tax_from_brackets）？模块间命名是否一致？
6. **模块耦合**：agent.py 直接导入 retriever、structured_tools、calculator——耦合度是否合理？schemas.py 被 build_kb 和 retriever 共享——是否清晰？
7. **依赖清单**：requirements.txt 是否包含所有依赖（FlagEmbedding、asyncpg、pgvector、langchain、gradio 等）？版本是否固定？
8. **Path 导入**：retriever.py 的 Path 是否已提升到模块级导入？（刚修过——之前 __init__ 局部导入导致 _ensure_models 的 NameError bug）
9. **测试组织**：68 个测试的分类是否清晰（TestAnnualBrackets / TestBonusMonthlyBrackets / TestDeductions / TestBonusComparison / TestEdgeCases）？是否有遗漏的测试维度？

## 严重度标准
- P0：几乎不存在（可维护性问题不致命）
- P1：面试官一眼能看出的不规范（未使用导入、类型缺失、重复代码）
- P2：可改进但不会扣分

记住这是求职作品集——代码整洁度直接反映专业水平。`

const TAX_DOMAIN_PROMPT = `你是中国个税领域专家，聚焦**税务法规的正确引用和计算合规性**。审查项目：E:/IITagent — 个税智能体。

## 必须读取的文件
- src/calculator.py（税率表、扣除标准）
- src/agent.py（SYSTEM_PROMPT 中的法规引用、Few-shot 示例）
- src/schemas.py（INTENT_KEYWORDS 覆盖的子类别）
- src/build_kb.py（分类体系）
- data/ 目录下的法规原文（抽查几篇）

## 检查清单
1. **综合所得税率表**：7 级超额累进税率和速算扣除数是否与《个人所得税法》附表一完全一致？3%→45% 的七档是否准确？
2. **年终奖政策时效性**：《财政部 税务总局关于延续实施全年一次性奖金个人所得税政策的公告》（2023年第30号）是否仍有效？年终奖单独计税政策是否已延期？政策截止日期是哪年？
3. **基本减除费用**：60000 元/年（5000元/月）是 2018 年修法后的标准，当前是否仍适用？是否有调整？
4. **专项附加扣除标准**：SYSTEM_PROMPT Few-shot 示例中的扣除金额是否准确？
   - 子女教育：2000元/月/每个子女？（是，国发〔2023〕13号提高至2000元）
   - 赡养老人：3000元/月（独生子女）？（是，同上文件提高至3000元）
   - 住房租金：1500/1100/800 三档是否与城市等级对应正确？
   - 婴幼儿照护：2000元/月/每孩？（是，国发〔2023〕13号新增）
   - 继续教育：学历400元/月、职业资格3600元/年？
   - 大病医疗：自付超15000元部分限额80000元？
5. **个人养老金扣除**：12000元/年限额是否正确？《财政部 税务总局公告2022年第34号》的递延纳税规则是否在提示词中正确描述？领取时 3% 税率是否准确？
6. **INTENT_KEYWORDS 的税务准确性**：12 个子类别的关键词是否覆盖了综合所得的主要场景？是否有遗漏的税种子类（如经营所得、财产租赁所得等）需要标记为"未覆盖"？
7. **法规引用格式**：SYSTEM_PROMPT 要求的法规引用格式"《法规名》第X条：原文关键句"是否合理？是否要求标注发文号和年份？
8. **筹划边界**：SYSTEM_PROMPT 禁止"税务筹划建议（如建议你离婚来避税）"——这个边界是否足够清晰？"哪种计税方式划算"算筹划还是算合规咨询？

## 严重度标准
- P0：税率、扣除标准等硬数字错误——直接导致用户做出错误税务决策
- P1：法规引用格式不规范、政策时效性未标注、子类别覆盖不全
- P2：措辞优化、边界澄清

你是面试官最可能深挖的维度——个税岗面试一定会问税率表和扣除标准。`

const DIMENSIONS = [
  { key: 'rag', label: 'RAG检索', prompt: RAG_PROMPT },
  { key: 'calc', label: '算税引擎', prompt: CALC_PROMPT },
  { key: 'prompt', label: '提示词', prompt: PROMPT_PROMPT },
  { key: 'maintain', label: '可维护性', prompt: MAINTAIN_PROMPT },
  { key: 'tax', label: '税务领域', prompt: TAX_DOMAIN_PROMPT },
]

const VERIFY_ANGLES = [
  { key: 'intent', label: '设计意图', angle: '你从**设计意图**角度反驳：这个问题是否是作者有意的 tradeoff？在项目约束（原型阶段、不上生产）下是否有理由这样做？如果作者在面试中解释这个选择，面试官会接受吗？' },
  { key: 'probability', label: '触发概率', angle: '你从**实际触发概率**角度反驳：这个问题在真实使用中是否几乎不会触发？触发条件是否极端？在原型 demo 场景下是否完全不可能出现？注意：如果触发概率 > 5%，就不能以此为由反驳。' },
  { key: 'portfolio', label: '作品集合理性', angle: '你从**作品集合理性**角度反驳：在 RAG 原型（非生产系统、不上生产）的上下文中，这个问题是否可接受？面试官是否会认为"这只是一个 demo，不需要过度设计"？但如果面试官因此质疑专业能力，就不能反驳。' },
]

// ── 阶段一：多维审查 ─────────────────────────────

phase('多维审查')

const reviewResults = await pipeline(
  DIMENSIONS,
  // Stage 1: 审查
  (dim) => agent(dim.prompt, {
    schema: FINDINGS_SCHEMA,
    label: `review:${dim.key}`,
    phase: '多维审查',
  }),
  // Stage 2: 对抗验证（仅对 P0/P1 发现）
  async (reviewResult, dim) => {
    if (!reviewResult) {
      log(`${dim.label}: 审查被跳过`)
      return { dimension: dim.label, findings: [] }
    }

    const allFindings = reviewResult.findings || []
    const p0p1 = allFindings.filter(f => f.severity === 'P0' || f.severity === 'P1')
    const p2p3 = allFindings.filter(f => f.severity === 'P2' || f.severity === 'P3')

    log(`${dim.label}: 发现 ${allFindings.length} 个问题 (P0/P1: ${p0p1.length}, P2/P3: ${p2p3.length})`)

    if (p0p1.length === 0) {
      return { dimension: dim.label, findings: allFindings }
    }

    // 对每个 P0/P1 发现进行对抗验证
    const verified = await parallel(
      p0p1.map((f, i) => async () => {
        const votes = await parallel(
          VERIFY_ANGLES.map(angle => () =>
            agent(
              `## 待反驳的审查发现
**标题**: ${f.title}
**文件**: ${f.file}:${f.line}
**严重度**: ${f.severity}
**问题描述**: ${f.description}
**修复建议**: ${f.fix}

## 你的反驳角度
${angle.angle}

请给出判决：confirmed（确认真问题）/ downgraded（严重度虚高，应降级）/ rejected（不是真问题）。
如果是 downgraded，请给出新的严重度级别（P1/P2/P3）。`,
              {
                schema: VERDICT_SCHEMA,
                label: `verify:${dim.key}:${i}`,
                phase: '对抗验证',
              }
            )
          )
        )

        const validVotes = votes.filter(Boolean)
        const rejectedCount = validVotes.filter(v => v.verdict === 'rejected').length
        const downgradedCount = validVotes.filter(v => v.verdict === 'downgraded').length
        const confirmedCount = validVotes.filter(v => v.verdict === 'confirmed').length

        // ≥2 票拒绝 → 丢弃
        if (rejectedCount >= 2) {
          log(`  ✗ 丢弃: ${f.title} (${rejectedCount}/${validVotes.length} 票拒绝)`)
          return null
        }

        // ≥2 票降级 → 降级
        if (downgradedCount >= 2) {
          const newSev = validVotes.find(v => v.verdict === 'downgraded')?.new_severity || 'P2'
          log(`  ↓ 降级: ${f.title} ${f.severity}→${newSev} (${downgradedCount}/${validVotes.length} 票降级)`)
          return { ...f, severity: newSev, verified: true, voteSummary: `确认${confirmedCount}/降级${downgradedCount}/拒绝${rejectedCount}` }
        }

        // 否则确认
        log(`  ✓ 确认: ${f.title} (${confirmedCount}/${validVotes.length} 票确认)`)
        return { ...f, verified: true, voteSummary: `确认${confirmedCount}/降级${downgradedCount}/拒绝${rejectedCount}` }
      })
    )

    const confirmedFindings = verified.filter(Boolean)
    return {
      dimension: dim.label,
      findings: [...confirmedFindings, ...p2p3],
      filteredCount: p0p1.length - confirmedFindings.length,
    }
  }
)

// ── 阶段二：汇总报告 ─────────────────────────────

phase('汇总报告')

const allFindings = reviewResults
  .filter(Boolean)
  .flatMap(r => r.findings || [])
  .map(f => ({ ...f, category: f.category || '未分类' }))

const p0Count = allFindings.filter(f => f.severity === 'P0').length
const p1Count = allFindings.filter(f => f.severity === 'P1').length
const p2Count = allFindings.filter(f => f.severity === 'P2').length
const p3Count = allFindings.filter(f => f.severity === 'P3').length
const totalFiltered = reviewResults.filter(Boolean).reduce((sum, r) => sum + (r.filteredCount || 0), 0)

log(`对抗验证完成: 过滤 ${totalFiltered} 个误报`)
log(`确认问题: P0=${p0Count} P1=${p1Count} P2=${p2Count} P3=${p3Count}`)

const findingsJson = JSON.stringify(allFindings, null, 2)

const reportResult = await agent(
  `## 任务：生成代码审查报告

你是资深技术面试官，需要基于以下审查发现，生成一份面向求职面试的代码审查报告。

## 项目背景
IITAgent（个税智能体）—— 精通中国个税法规的 AI 智能体，提供税前筹划和合规解读。
- 技术栈：Python 3.11+, LangChain, pgvector, BGE-M3, DeepSeek, Gradio
- 定位：RAG 原型项目，不上生产，核心价值是"事前规划"和"可信解释"
- 首发场景：综合所得汇算清缴 + 年终奖计税对比

## 审查发现数据
\`\`\`json
${findingsJson}
\`\`\`

## 报告要求

写入文件 E:/IITagent/docs/code-review.md，使用中文。

报告结构：

### 1. 审查摘要
- 表格：各级别问题数量（按维度分组）
- 一句话总体评价（站在面试官视角）
- 对抗验证过滤统计（过滤了多少误报）

### 2. 逐项详情
按 P0 → P1 → P2 → P3 排序，每个问题包含：
- 标题 + 级别徽章
- 文件位置（可点击）
- 问题描述
- 修复建议（具体代码 diff）
- 对抗投票结果（如有）

### 3. 整体评价
**亮点（面试加分项）**：
- 列出 5-8 个技术亮点，说明为什么面试官会认可

**待改进（面试可能扣分）**：
- 列出主要短板，按严重程度排序

### 4. 面试追问准备
基于发现的问题和项目特点，准备 5-8 个面试官可能追问的问题及建议回答：
- Q: 问题
- 考察点: 面试官想了解什么
- 建议回答方向: 你怎么回答能加分
- 当前代码的支撑: 代码中哪里体现了你的能力

示例追问方向：
- "为什么选择 BGE-M3 而不是其他 embedding 模型？"
- "混合检索（Dense+Sparse+RRF+Reranker）的设计理由是什么？"
- "年终奖计税的税率表是如何正确实现的？"
- "如何确保 LLM 不编造法规条款？"
- "为什么金额用 Decimal 而不是 float？"
- "如果知识库从 73 份法规扩展到 500 份，检索性能如何保证？"
- "Agent 提示词的四段式格式输出是如何保证的？"

### 5. 快速修复清单
P0/P1 问题的修复步骤，按文件分组，方便快速修改。

## 重要提示
- 这是求职作品集（RAG 原型），评价标准应合理（不是生产系统）
- 报告应该是"建设性的"，不只是挑刺
- 面试追问 Q&A 是最有价值的部分——帮助用户在面试中表现出色
- 每个问题都要给出具体可操作的修复建议`,
  {
    label: '生成审查报告',
    phase: '汇总报告',
  }
)

log('报告已生成: docs/code-review.md')
return { totalFindings: allFindings.length, p0Count, p1Count, p2Count, p3Count, totalFiltered }
