# Agent 架构：ReAct vs Plan-and-Execute

## 一、ReAct (Reasoning + Acting)

### 核心思想

ReAct 由 Shunyu Yao 等人在 2022 年提出，核心思想是**将推理（Reasoning）与行动（Acting）交织在一起**。Agent 在每轮交互中先思考再行动，通过不断循环的"思考-行动-观察"来逐步逼近最终答案。

### 工作流程

```
Thought  →  Action  →  Observation  →  Thought  →  Action  →  ...  →  Final Answer
```

具体步骤：

1. **Thought（思考）**：模型分析当前状态，决定下一步该做什么
2. **Action（行动）**：调用一个工具（如搜索、读文件、执行命令）
3. **Observation（观察）**：接收工具返回的结果
4. 重复以上步骤，直到有足够信息给出最终答案

### 代码示例（来自本项目 `agent.py`）

```python
# ReActAgent 的核心循环
while True:
    content = self.call_model(messages)           # 请求模型生成 Thought + Action

    thought = extract(content, "<thought>")        # 提取思考
    action = extract(content, "<action>")          # 提取行动

    if "<final_answer>" in content:                # 如果输出最终答案，结束
        return extract(content, "<final_answer>")

    observation = self.tools[tool_name](*args)     # 执行工具，获取观察结果
    messages.append({"role": "user",               # 将观察结果加入对话
                     "content": f"<observation>{observation}</observation>"})
```

### 优点

- **简单直接**：无需预先规划，边想边做，适合探索式任务
- **灵活应变**：可根据中间结果动态调整下一步行动
- **可解释性强**：Thought 过程暴露了模型的推理链路

### 缺点

- **缺乏长期规划**：每次只看一步，容易陷入局部最优
- **容易跑偏**：在长序列中可能偏离原始目标
- **效率较低**：每一步都需要调用 LLM，Token 消耗大

---

## 二、Plan-and-Execute

### 核心思想

先将任务分解为多步计划，再逐一执行计划中的步骤。在完成某些步骤后，可以**重新审视并调整**原有计划。这种"先规划后执行"的方式将"规划"和"执行"两个职责解耦。

### 工作流程

```
User Input → [Planner] → Plan[Step1, Step2, ...] → [Executor] → Result → [Re-planner] → 调整计划或结束
```

核心组件：

| 组件 | 职责 | 可用模型 |
|------|------|----------|
| **Planner** | 将用户目标分解为可执行的步骤列表 | 强模型（如 GPT-4, deepseek-reasoner） |
| **Executor** | 按计划依次执行每一步 | 弱模型（如 GPT-3.5-turbo） |
| **Re-planner** | 根据执行结果判断：继续执行下一 步、修改计划、或给出最终答案 | 强模型 |

### 代码示例（来自本项目 `plan-and-execute.ipynb`）

```python
# Planner：生成计划
planner = planner_prompt | ChatOpenAI(model="gpt-3.5-turbo").with_structured_output(Plan)

# Executor：执行具体步骤
async def execute_step(state: PlanExecute):
    task = state["plan"][0]
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {"past_steps": [(task, agent_response["messages"][-1].content)]}

# Re-planner：重规划或给出最终答案
async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}   # 给出最终答案
    else:
        return {"plan": output.action.steps}           # 更新计划继续执行
```

### 状态图

```
START → Planner → Agent → Replan → Agent → Replan → ... → END
                    ↑                    │
                    └──── 调整计划 ──────┘
```

### 优点

- **长期规划能力强**：一开始就明确整体路线图
- **模型分工灵活**：Planner 用强模型，Executor 用弱模型，节省成本
- **可纠偏**：Re-planner 能在执行偏离时修正计划

### 缺点

- **额外延迟**：每次执行后都需要 Re-plan，增加一轮 LLM 调用
- **初始计划可能不准**：如果对任务理解有偏差，后续可能越偏越远
- **实现复杂度高**：需要维护状态图、规划状态等

---

## 三、对比总结

| 维度 | ReAct | Plan-and-Execute |
|------|-------|-----------------|
| **规划方式** | 边想边做，无显式计划 | 先规划，后执行 |
| **模型调用次数** | 每步 1 次 | 每步 1 次 + 额外 Re-plan |
| **适用场景** | 简单、探索式任务 | 复杂、多步骤任务 |
| **长期规划能力** | 弱 | 强 |
| **灵活性** | 高，可随时转向 | 中，依赖 Planner 质量 |
| **可解释性** | Thought 暴露推理过程 | Plan 展示整体路线图 |
| **模型分工** | 单一模型 | 可 Planner/Executor 用不同模型 |

### 如何选择？

- **简单任务**（如"查一下今天的天气"）→ **ReAct**，直接高效
- **复杂任务**（如"分析这份财报并生成 PPT"）→ **Plan-and-Execute**，先规划再执行

两种架构并非互斥——实践中可以将 ReAct 作为 Executor 的内部策略，形成 `Plan → [ReAct Loop] → Replan → ...` 的混合模式，这也是本项目 `plan-and-execute.ipynb` 中采用的方式。
