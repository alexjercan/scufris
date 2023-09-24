"""This module contains the agent for the Scufris chatbot."""
import logging

import dotenv
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage

from .tools import CUSTOM_TOOLS

dotenv.load_dotenv()

logger = logging.getLogger("scufris")

MEMORY_KEY = "chat_history"
SYSTEM = """You are Scufris, the best chatbot with really high IQ. Your job is
to help me with my day to day tasks. You can do this by using custom tools.
You can access the internet. You have access to basic Linux commands and a
Python repl. You can also access the memory of the conversation."""


def create_agent() -> AgentExecutor:
    """Create the agent for the Scufris chatbot.

    Returns
    -------
    AgentExecutor
        The agent for the Scufris chatbot.
    """
    logger.info("Creating agent ...")

    llm = ChatOpenAI(model_name="gpt-4")

    tools = load_tools(["python_repl", "terminal"], llm=llm)
    tools.extend(CUSTOM_TOOLS)

    logger.debug("Loaded the following tools: %s", tools)

    memory = ConversationSummaryMemory(
        llm=llm, memory_key=MEMORY_KEY, return_messages=True
    )

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=SystemMessage(content=SYSTEM),
        extra_prompt_messages=[MessagesPlaceholder(variable_name=MEMORY_KEY)],
    )

    functions_agent = OpenAIFunctionsAgent(
        llm=llm, tools=tools, prompt=prompt, memory=memory
    )
    agent = AgentExecutor(
        agent=functions_agent, tools=tools, memory=memory, verbose=True
    )

    logger.debug("Created agent: %s", agent)

    return agent
