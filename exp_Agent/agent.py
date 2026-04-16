from typing import List, Callable, Tuple
from openai import OpenAI
from dotenv import load_dotenv
import os
import inspect
import re
import ast
from string import Template
from prompt_template import react_system_prompt_template


class ReActAgent:
    def __init__(self, tools: List[Callable], model: str, project_directory: str):
        self.tools = { func.__name__: func for func in tools}
        self.model = model
        self.project_directory = project_directory
        self.client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=ReActAgent.get_api_key(),
        )

    def run(self, user_input: str):
        messages = [
            {"role": "system", "content": self.render_system_prompt(react_system_prompt_template)},
            {"role": "user", "content": user_input}
        ]

        while True:
            # 请求模型
            content = self.call_model(messages)

            # 检测 Thought
            thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
            if thought_match:
                thought = thought_match.group(1)
            
            # 检测是否输出 Final Answer，如果是，直接返回
            if "<final_answer>" in content:
                final_answer = re.search(r"<final_answer>(.*?)</final_answer>", content, re.DOTALL)
                return final_answer.group(1)
            
            # 检测 Action
            action_match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
            if not action_match:
                raise RuntimeError("模型未输出 <action>")
            action = action_match.group(1)
            tool_name, args = self.parse_action(action)

            print(f"\n\n Action: {tool_name}({', '.join(args)})")
            
            # 只有终端命令才需要询问用户，其他工具继续执行
            should_continue = input(f"请求调用Tool-{tool_name}\n是否继续？（Y/N）") if tool_name in ["run_terminal_command", "write_to_file"] else "y"
            if should_continue.lower() != 'y':
                print("\n\n操作已取消。")
                return "操作已被用户取消"
            
            try:
                observation = self.tools[tool_name](*args)
            except Exception as e:
                observation = f"工具执行错误：{str(e)}"
            
            print(f"\n\n Observation：{observation}")
            obs_msg = f"<observation>{observation}</observation>"
            messages.append({"role": "user", "content": obs_msg})
    

    def render_system_prompt(self, system_prompt_template: str) -> str:
        """渲染系统提示模板，替换变量"""
        tool_list = self.get_tool_list()
        file_list = ", ".join(
            os.path.abspath(os.path.join(self.project_directory, f))
            for f in os.listdir(self.project_directory)
        )

        return Template(system_prompt_template).substitute(
            operating_system = self.get_operating_system_name(),
            tool_list = tool_list,
            file_list = file_list
        )


    def get_tool_list(self) -> str:
        """生成工具列表字符串，包含函数签名和简要说明"""
        tool_descriptions = []
        for func in self.tools.values():
            name = func.__name__
            signature = str(inspect.signature(func))
            doc = inspect.getdoc(func)
            tool_descriptions.append(f"- {name}{signature}: {doc}")
        
        return "\n".join(tool_descriptions)


    def get_operating_system_name(self):
        os_map = {
            "Darwin": "macOS",
            "Windows": "Windows",
            "Linux": "Linux"
        }


    def call_model(self, message):
        print("\n\n请求模型中 ··· \n\n")
        response = self.client.chat.completions.create(
            model = self.model,
            messages=message,
        )

        # 上下文记录
        content = response.choices[0].message.content
        message.append(
            {"role": "assistant", "content": content}
        )

        return content

    def parse_action(self, code_str: str) -> Tuple[str, List[str]]:
        match = re.match(r'(\w+)\((.*)\)', code_str, re.DOTALL)
        if not match:
            raise ValueError("Invalid function call syntax")

        func_name = match.group(1)
        args_str = match.group(2).strip()

        # 解析参数，特别处理包含多行内容的字符串
        args = []
        current_arg = ""
        in_string = False   # 当前是不是在引号（字符串）里面
        string_char = None      # 包围当前字符串的是单引号 ' 还是双引号 "
        paren_depth = 0     # 当前在几层括号里面？
        i = 0

        while i < len(args_str):
            char = args_str[i]

            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    string_char = char
                    current_arg += char
                elif char == "(":
                    paren_depth += 1
                    current_arg += char
                elif char ==")":
                    paren_depth -= 1
                    current_arg += char
                elif char == ',' and paren_depth == 0:
                    # 遇到顶层或逗号，结束当前参数
                    args.append(self._parse_single_arg(current_arg.strip()))
                    current_arg = ""
                else:
                    current_arg += char
            else:
                current_arg += char
                if char == string_char and (i==0 or args_str[i-1] != '\\'):
                    in_string = False
                    string_char = None

            i += 1

        # 处理最后一个参数
        if current_arg.strip():
            args.append(self._parse_single_arg(current_arg.strip()))

        return func_name, args

    def _parse_single_arg(self, arg_str: str):
        """解析单个参数"""
        arg_str = arg_str.strip()

        # 如果是字符串字面量
        if (arg_str.startswith('"') and arg_str.endswith('"')) or \
           (arg_str.startswith("'") and arg_str.endswith("'")):
            # 移除外层引号并处理转义字符
            inner_str = arg_str[1:-1]
            # 处理常见的转义字符
            inner_str = inner_str.replace('\\"', '"').replace("\\'", "'")
            inner_str = inner_str.replace('\\n', '\n').replace('\\t', '\t')
            inner_str = inner_str.replace('\\r', '\r').replace('\\\\', '\\')

            return inner_str
        
        # 尝试使用ast.literal_eval 解析其他类型
        try:
            return ast.literal_eval(arg_str)
        except (SyntaxError, ValueError):
            # 如果解析失败，则返回原始字符串
            return arg_str

    @staticmethod
    def get_api_key() -> str:
        """Load the API key from an environment variable."""
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("未找到 OPENROUTER_API_KEY 环境变量，请在 .env 文件中设置。")
        return api_key
    
