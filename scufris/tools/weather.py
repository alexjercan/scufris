import requests
from langchain.tools import BaseTool


class WeatherTool(BaseTool):
    name = "Weather"
    description = (
        "useful for when you want to see the weather in a specific location;"
        "it should be called using the location as a string"
    )

    def _run(self, query: str) -> str:
        try:
            response = requests.get(f"https://wttr.in/{query}?format=4")
        except requests.exceptions.Timeout:
            return "Sorry, the weather service is not responding right now."
        except Exception:
            return "Sorry, something went wrong with the weather service."

        return response.text

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Weather does not support async")
