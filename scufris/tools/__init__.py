""" Custom tools for the Chatbot. """
from .weather import WeatherTool

CUSTOM_TOOLS = [
    WeatherTool(),
]

__all__ = ["CUSTOM_TOOLS"]
