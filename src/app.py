"""IITAgent 对话界面 — Gradio 前端。

启动: python src/app.py
"""

import gradio as gr

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
    theme="soft",
)

if __name__ == "__main__":
    demo.launch()
