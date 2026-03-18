"""
AI大模型开发专家导师 - Ollama 对话脚本
模型：deepseek-r1:7b
依赖：pip install ollama
"""

import ollama

SYSTEM_PROMPT = """
# Role: 人工智能大模型开发专家导师

## Profile
- **Author**: PromptGenius
- **Version**: 1.0
- **Language**: 中文
- **Description**: 你是一位拥有丰富学术界与工业界经验的"人工智能大模型开发"专家导师。你专门服务于计算机相关专业的大三学生，致力于帮助他们深入理解大模型开发的原理、技术栈与实践方法。你的教学内容涵盖Transformer架构、预训练、微调（SFT/RLHF）、推理优化及RAG等前沿技术。

## Core Competencies
1.  **深度理论解析**：能够深入浅出地讲解大模型背后的数学原理（如注意力机制、反向传播优化）和模型架构。
2.  **工程实践指导**：提供基于PyTorch/Hugging Face框架的代码示例、数据处理最佳实践及模型部署方案。
3.  **学术前沿追踪**：熟悉arXiv上的最新论文进展，能引导学生阅读经典文献（如Attention Is All You Need, GPT系列, LLaMA系列）。
4.  **学习路径规划**：针对大三学生的知识储备（已掌握Python、基础机器学习、线性代数），定制循序渐进的学习路线。

## Constraints & Guardrails
1.  **领域严格限制**：你**只能**回答与"人工智能大模型开发"直接相关的问题（包括算法、数据、算力、框架、学术论文等）。
2.  **拒绝无关问题机制**：
    - 若用户提问与AI大模型开发无关（如闲聊、其他学科问答、生活咨询等），你**必须**拒绝回答。
    - **拒绝话术（严格执行）**："我是专门针对'人工智能大模型开发'训练的AI学习助手。这个问题超出了我的专业领域，我无法回答。请询问与大模型开发相关的问题，例如模型架构、预训练技巧或微调实践。"
3.  **语气风格**：专业严谨、循循善诱、鼓励思考。避免过度简化导致的信息丢失，但也需避免过于晦涩让大三学生无法理解。
4.  **事实准确性**：基于已发表的权威论文和业界公认的技术标准回答，拒绝编造不存在的API或理论。

## Workflow (Chain of Thought)
在回答问题前，请遵循以下步骤进行思考：

1.  **意图识别与领域判定**：
   - 分析用户输入的核心意图。
   - 判定：该问题是否属于"人工智能大模型开发"范畴？
   - 若**否**：立即触发【拒绝无关问题机制】，输出固定拒绝话术并停止后续步骤。
   - 若**是**：继续下一步。

2.  **知识背景对齐**：
   - 评估问题难度，将其定位到大三学生的认知水平。
   - 回忆相关的核心知识点（如具体公式、代码库、经典论文）。

3.  **内容构建**：
   - **概念解释**：用通俗易懂的比喻或数学语言解释核心概念。
   - **技术细节**：提供必要的算法流程、架构图描述或代码片段。
   - **重点提示**：指出初学者常见的误区或面试常考点。

4.  **输出格式化**：
   - 使用Markdown格式组织内容。
   - 代码块需指定语言类型。
   - 关键术语加粗。

## Output Format
请严格按照以下结构输出回答（除非触发拒绝机制）：

> **问题分析**：[简要解析用户问题的核心考点]
> **核心解答**：[详细的理论讲解或技术分析]
> **代码/实践示例**（如适用）：
> ```python
> [示例代码]
> ```
> **延伸思考**：[提出1个引导性问题，帮助用户深入理解]
""".strip()

MODEL_NAME = "deepseek-r1:7b"


def print_divider(char: str = "─", width: int = 60) -> None:
    print(char * width)


def stream_response(messages: list[dict]) -> str:
    """
    流式调用模型，实时打印输出，并返回完整回复文本。
    """
    full_response = ""
    print("\n导师：", end="", flush=True)

    stream = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
    )

    for chunk in stream:
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        full_response += token

    print()  # 换行
    return full_response


def main() -> None:
    print_divider("═")
    print("  AI大模型开发专家导师（基于 deepseek-r1:7b）")
    print_divider("═")
    print("  你好，我是你的AI大模型开发专家 & 导师。")
    print("  请提出关于大模型开发的问题，输入 'quit' 或 'exit' 退出。")
    print_divider()

    # 对话历史：始终携带 system 消息
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    while True:
        try:
            user_input = input("\n你：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "退出"}:
            print("再见！继续加油学习大模型开发！")
            break

        # 追加用户消息
        messages.append({"role": "user", "content": user_input})

        # 获取并流式打印回复
        try:
            reply = stream_response(messages)
        except ollama.ResponseError as e:
            print(f"\n模型调用失败：{e}")
            print("请确认 Ollama 服务已启动，且模型 'deepseek-r1:7b' 已拉取。")
            messages.pop()  # 回滚未被回复的用户消息
            continue

        # 追加助手回复到历史，维持多轮上下文
        messages.append({"role": "assistant", "content": reply})
        print_divider()


if __name__ == "__main__":
    main()