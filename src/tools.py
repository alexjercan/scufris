import requests
from langchain.tools import BaseTool


class CustomExitTool(BaseTool):
    name = "Exit"
    description = "useful for when you need to quit"

    def _run(self, query: str) -> str:
        return exit(0)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Exit does not support async")


class WeatherTool(BaseTool):
    name = "Weather"
    description = (
        "useful for when you want to see the weather in a specific location;"
        "it should be called using the location as a string"
    )

    def _run(self, query: str) -> str:
        response = requests.get(f"https://wttr.in/{query}")

        return response.text

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Weather does not support async")


CUSTOM_TOOLS = [CustomExitTool(), WeatherTool()]
