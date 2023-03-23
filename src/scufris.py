import argparse
import os
from dataclasses import dataclass

import gradio as gr
from langchain.agents import AgentExecutor, initialize_agent, load_tools
from langchain.llms import OpenAI
from rich.prompt import Prompt

from tools import CUSTOM_TOOLS

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

assert OPENAI_API_KEY is not None, "You have to provide an api key for openai"


def create_agent() -> AgentExecutor:
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

    tools = load_tools(["python_repl", "terminal"])
    tools.extend(CUSTOM_TOOLS)

    return initialize_agent(
        tools, llm, agent="zero-shot-react-description", verbose=True
    )


def deploy_cli():
    agent = create_agent()

    while True:
        try:
            prompt = Prompt.ask("User")
            agent.run(prompt)
        except KeyboardInterrupt:
            print()


def deploy_gradio():
    agent = create_agent()

    def run_agent(text, state):
        response = agent.run(text)
        state = state + [(text, response)]
        return state, state

    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        chatbot = gr.Chatbot(elem_id="chatbot", label="Scufris")
        state = gr.State([])
        with gr.Row():
            with gr.Column(scale=0.7):
                txt = gr.Textbox(
                    show_label=False, placeholder="Enter text and press enter"
                ).style(container=False)
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button("Clear")

        txt.submit(run_agent, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)

        demo.launch(server_name="0.0.0.0", server_port=8888)


MODE_CLI = "cli"
MODE_GRADIO = "gradio"


@dataclass
class Args:
    mode: str


def parse_args() -> Args:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[MODE_CLI, MODE_GRADIO],
        default=MODE_CLI,
        help="the mode to use when deploying the scufris bot",
    )

    args = parser.parse_args()

    return Args(mode=args.mode)


if __name__ == "__main__":
    args = parse_args()

    if args.mode == MODE_CLI:
        deploy_cli()
    elif args.mode == MODE_GRADIO:
        deploy_gradio()
