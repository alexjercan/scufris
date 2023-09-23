from .stop import CustomExitTool
from .weather import WeatherTool

CUSTOM_TOOLS = [
    CustomExitTool(),
    WeatherTool(),
]

__all__ = ["CUSTOM_TOOLS"]
