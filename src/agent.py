"""IITAgent — 检索增强 + 工具调用，一条链走到底。

使用方式：
    from src.agent import ask
    print(ask("年终奖8万，年收入30万，哪个划算？"))
"""

import logging
import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from src.retriever import Retriever
from src.structured_tools import TOOLS

load_dotenv()
_logger = logging.getLogger("agent")

# ── LLM ─────────────────────────────────────────────────

_llm = ChatOpenAI(
    model=os.environ.get("LLM_MODEL", "deepseek-chat"),
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    temperature=0.3,
)

# ── 提示词 ──────────────────────────────────────────────

SYSTEM_PROMPT = """你是一位资深中国个税顾问，精通个人所得税法规。

工作方式：
1. 根据提供的法规条文给出准确回答
2. 涉及税额计算时，调用计算工具，不要心算
3. 信息不完整时，引导用户补充缺失参数

回答格式：
- 先白话解释，再计算结果（如需），最后引用法规条款
- 所有建议可溯源，不编造不存在的法规
- 末尾加免责声明："以上内容仅供参考，具体以税务机关最新规定为准。"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "\n\n## 相关法规\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ── 检索器（懒加载） ────────────────────────────────────

_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever.from_database()
    return _retriever


# ── Agent ───────────────────────────────────────────────

_agent = create_tool_calling_agent(_llm, TOOLS, PROMPT)
_executor = AgentExecutor(
    agent=_agent, tools=TOOLS, verbose=False,
    handle_parsing_errors=True, max_iterations=3,
)


# ── 公开接口 ────────────────────────────────────────────

def ask(question: str) -> str:
    """发送个税问题，返回回答。"""
    try:
        ctx = _get_retriever().search(question, top_k=5)
        context = "\n\n".join(
            f"[{i}] ({r.document_source or '?'}) {r.content}"
            for i, r in enumerate(ctx, 1)
        ) if ctx else "（未找到相关法规）"
    except Exception as e:
        _logger.warning("检索失败: %s", e)
        context = "（检索暂不可用）"

    result = _executor.invoke({"input": question, "context": context})
    return result["output"]
