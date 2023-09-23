from langchain.tools import BaseTool


class CustomExitTool(BaseTool):
    name = "Exit"
    description = "useful for when you need to quit"

    def _run(self, query: str) -> str:
        return exit(0)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Exit does not support async")
