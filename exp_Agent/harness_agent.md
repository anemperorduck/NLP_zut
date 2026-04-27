# Harness Agent：Agent 的运行框架

## 什么是 Harness Agent？

Harness（直译"马具/线束"）在软件工程中通常指**测试框架**或**运行容器**。在 Agent 的语境下，**Agent Harness** 是指承载和管理 Agent 运行所需的**基础设施层**——它提供了 Agent 执行的环境，管理输入输出、工具注册、安全策略、日志追踪等横切关注点。

简而言之：**Agent 是业务逻辑，Harness 是运行环境。**

```
┌─────────────────────────────────────────────┐
│              Agent Harness                   │
│  ┌───────────────────────────────────────┐  │
│  │          Agent (业务逻辑)              │  │
│  │  ┌───────┐  ┌────────┐  ┌────────┐  │  │
│  │  │Thought│→ │ Action │→ │Observation │  │ │
│  │  └───────┘  └────────┘  └────────┘  │  │
│  └───────────────────────────────────────┘  │
│                                              │
│  工具注册 | 状态管理 | 安全策略 | 日志追踪    │
│  输入解析 | 输出格式化 | 错误处理 | Token计数 │
└─────────────────────────────────────────────┘
```

---

## Harness 的核心职责

### 1. 工具管理（Tool Registry）

注册、发现和调用工具是 Harness 的基础能力。

```python
# 本项目 agent.py 中的工具注册
class ReActAgent:
    def __init__(self, tools: List[Callable], ...):
        # 以函数名为 key 注册工具
        self.tools = {func.__name__: func for func in tools}

    def get_tool_list(self) -> str:
        # 自动生成工具描述（含函数签名和文档字符串）
        for func in self.tools.values():
            name = func.__name__
            signature = str(inspect.signature(func))
            doc = inspect.getdoc(func)
            tool_descriptions.append(f"- {name}{signature}: {doc}")
```

**Harness 应处理：**

| 职责     | 说明                                          |
| -------- | --------------------------------------------- |
| 工具注册 | 将 Python 函数注册为 Agent 可调用的工具       |
| 签名自省 | 自动提取函数参数信息，供 LLM 理解如何调用     |
| 参数解析 | 将 LLM 输出的文本转换为实际函数参数           |
| 结果封装 | 将工具返回值格式化为 LLM 可理解的 Observation |
| 错误处理 | 工具执行异常时捕获并返回友好错误信息          |

### 2. 对话 / 状态管理（Message & State Management）

Harness 需要维护 Agent 的完整执行上下文，包括历史对话和中间状态。

```python
# 本项目中 Harness 维护的消息列表
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_input},
    {"role": "assistant", "content": thought + action},   # 模型输出
    {"role": "user", "content": observation},              # 工具结果
    ...
]
```

而在 Plan-and-Execute 中，Harness 还需要维护更复杂的结构化状态：

```python
class PlanExecute(TypedDict):
    input: str                              # 用户原始输入
    plan: List[str]                         # 当前计划（步骤列表）
    past_steps: List[Tuple[str, str]]       # 已完成的步骤及结果
    response: str                           # 最终答案
```

### 3. Prompt 管理（Prompt Management）

Harness 负责渲染系统提示模板，将环境信息注入到 prompt 中。

```python
# 本项目中的 prompt 渲染
def render_system_prompt(self, template: str) -> str:
    return Template(template).substitute(
        operating_system=self.get_operating_system_name(),
        tool_list=tool_list,
        file_list=file_list,
    )
```

**典型注入的上下文信息：**

- 操作系统类型（Windows / macOS / Linux）
- 当前工作目录及文件列表
- 可用工具列表（名称、参数、用途）
- 用户目标
- 历史执行记录

### 4. 控制循环（Control Loop）

这是 Harness 最核心的部分——决定 Agent 何时思考、何时行动、何时终止。

**ReAct 的控制循环：**

```python
while True:
    response = llm.invoke(messages)
    if has_final_answer(response):
        return extract_answer(response)
    action = extract_action(response)
    observation = execute_tool(action)
    messages.append(observation)
```

**Plan-and-Execute 的控制循环（使用 LangGraph 的状态图）：**

```
START → Planner → Executor → Replanner → (Executor 或 END)
```

LangGraph / LangChain 等框架本身就是一种 **Harness 的实现**，它们提供了：

| 功能       | LangGraph 中的实现                   |
| ---------- | ------------------------------------ |
| 状态图定义 | `StateGraph`, `add_node`, `add_edge` |
| 节点执行   | 自定义 async/await 函数              |
| 条件跳转   | `add_conditional_edges`              |
| 状态持久化 | `StateGraph` 的状态传递机制          |

### 5. 安全与权限控制（Safety & Permission）

Harness 是执行安全策略的理想位置。

```python
# 本项目中简单的权限控制
should_continue = input(f"请求调用Tool-{tool_name}\n是否继续？（Y/N）") \
    if tool_name in ["run_terminal_command", "write_to_file"] else "y"
```

**典型的 Harness 安全策略：**

- **高危操作确认**：执行终端命令、删除文件等操作需用户确认
- **沙箱执行**：在隔离环境中运行工具
- **速率限制**：限制工具调用频率
- **输出过滤**：屏蔽敏感信息（API Key、密码等）

### 6. 日志与可观测性（Logging & Observability）

Harness 应记录 Agent 的完整执行轨迹，用于调试、评估和审计。

```python
# 典型日志记录内容
{
    "step": 1,
    "thought": "我需要查一下埃菲尔铁塔的高度",
    "action": "get_height('埃菲尔铁塔')",
    "observation": "埃菲尔铁塔高度约为330米",
    "token_usage": {"prompt": 150, "completion": 45, "total": 195},
    "latency_ms": 2340
}
```

---

## Harness 的不同实现层次

### Level 1: 自建简单 Harness

如本项目的 `agent.py` —— 手动管理消息列表和循环：

```
优点：完全可控，无额外依赖
缺点：需要自己处理所有边缘情况
```

### Level 2: 框架级 Harness（LangGraph / LangChain）

如 `plan-and-execute.ipynb` 中的实现：

```python
workflow = StateGraph(PlanExecute)
workflow.add_node("planner", plan_step)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", replan_step)
workflow.add_conditional_edges("replan", should_end, ["agent", END])
app = workflow.compile()  # 编译为可运行的 Harness
```

```
优点：状态管理、图执行、条件分支等开箱即用
缺点：框架抽象带来了学习成本
```

### Level 3: 生产级 Harness（LangGraph Platform / AutoGen / CrewAI）

```
- 分布式执行
- 持久化存储
- 人机交互接口
- 监控告警
- 多 Agent 协作
```

---

## 总结

**Agent Harness 是让 Agent 从"能跑"到"稳定可靠运行"的关键基础设施。** 它解决了以下问题：

1. 工具如何被 Agent 发现和调用
2. 执行上下文如何维护和传递
3. Agent 何时思考、行动、终止
4. 如何保证执行的安全性
5. 如何记录和追踪执行过程

本项目中的 `agent.py` 实现了一个最小化的 Harness，而 `plan-and-execute.ipynb` 通过 LangGraph 展示了更结构化的 Harness（状态图驱动）。理解 Harness 的设计有助于构建更健壮、可扩展的 Agent 系统。
