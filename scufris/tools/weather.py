import requests
from typing import Optional
from langchain.tools import BaseTool


class WeatherTool(BaseTool):
    name = "Weather"
    description = (
        "This tool allows you to get the weather using the wttr.in service;"
        "you can get the weather in your current location by passing no "
        "arguments or in a specific location by passing the location as an "
        "argument;"
    )

    def _run(self, query: Optional[str] = None) -> str:
        if query is None:
            query = ""

        try:
            response = requests.get(f"https://wttr.in/{query}?format=4")
        except requests.exceptions.Timeout:
            return "Sorry, the weather service is not responding right now."
        except Exception:
            return "Sorry, something went wrong with the weather service."

        return response.text

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Weather does not support async")
