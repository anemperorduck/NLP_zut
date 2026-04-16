import click
import os
from agent import ReActAgent
from tools import *


@click.command()
@click.argument('project_dir',
                type=click.Path(exists=True, file_okay=False, dir_okay=True),
                default='.')
def main(project_dir):
    project_dir = os.path.abspath(project_dir)

    tools = [read_file, write_to_file, run_terminal_command]
    agent = ReActAgent(tools=tools, model="deepseek-chat", project_directory=project_dir)

    task = input("请输入任务：")
    final_answer = agent.run(task)

    print(f"\n\n Final Answer：{final_answer}")


if __name__ == "__main__":
    main()