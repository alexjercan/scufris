from langchain.tools import BaseTool


class CustomExitTool(BaseTool):
    name = "Exit"
    description = "useful for when you need to quit"

    def _run(self, query: str) -> str:
        """Use the tool."""
        return exit(0)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Exit does not support async")
