from rich import print as pprint
import dotenv
import logging
from rich.logging import RichHandler
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from rich.prompt import Prompt
from .tools import CUSTOM_TOOLS

dotenv.load_dotenv()

rich_handler = RichHandler(rich_tracebacks=True)
logging.root.addHandler(rich_handler)
logging.root.setLevel(logging.ERROR)

logger = logging.getLogger("scufris")
logger.setLevel(logging.DEBUG)

MEMORY_KEY = "chat_history"
SYSTEM = """You are Scufris, the best chatbot with really high IQ. Your job is
to help me with my day to day tasks. You can do this by using custom tools.
You can access the internet. You have access to basic Linux commands and a
Python repl. You can also access the memory of the conversation."""


def create_agent():
    llm = ChatOpenAI()

    tools = load_tools(["python_repl", "terminal"], llm=llm)
    tools.extend(CUSTOM_TOOLS)

    memory = ConversationSummaryMemory(
        llm=llm, memory_key=MEMORY_KEY, return_messages=True
    )

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=SystemMessage(content=SYSTEM),
        extra_prompt_messages=[MessagesPlaceholder(variable_name=MEMORY_KEY)],
    )

    search_agent_memory = OpenAIFunctionsAgent(
        llm=llm, tools=tools, prompt=prompt, memory=memory
    )
    agent_executor_memory = AgentExecutor(
        agent=search_agent_memory, tools=tools, memory=memory, verbose=False
    )

    return agent_executor_memory


def main():
    agent = create_agent()

    while True:
        try:
            prompt = Prompt.ask("User")
            response = agent.run(prompt)
            pprint(f"[bold blue]Scufris:[/bold blue] {response}")
        except KeyboardInterrupt:
            print()
