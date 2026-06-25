"""IITAgent 对话界面 — Gradio 前端。

启动: python -m src.app
"""

import warnings

import gradio as gr

# Python 3.14 上 pydantic v1 兼容层不可用（C 扩展不兼容），langchain-core 内部
# 尝试 import 会打 warning，但本项目的 Schema 全用 Pydantic v2，功能不受影响。
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14",
)
# Gradio/Starlette 用了旧常量名，等上游修。
warnings.filterwarnings(
    "ignore",
    message=".*422.*deprecated.*",
)

from src.agent import ask


def chat(message: str, history: list) -> str:
    """处理用户消息，返回回答。"""
    if not message.strip():
        return "请输入个税相关问题。"
    try:
        return ask(message)
    except Exception as e:
        return f"出错了: {e}"


demo = gr.ChatInterface(
    fn=chat,
    title="个税智能体 (IITAgent)",
    description="精通中国个税法规的 AI 顾问。可以回答专项附加扣除、年终奖计税、汇算清缴等问题。",
    examples=[
        "年终奖8万，年收入30万，哪种计税方式划算？",
        "子女教育专项附加扣除标准是多少？",
        "住房租金扣除分几档？",
        "赡养老人每月能扣多少？",
    ],
)

if __name__ == "__main__":
    demo.launch()
