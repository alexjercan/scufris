import requests
import torch
from langchain.tools import BaseTool
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        response = requests.get(f"https://wttr.in/{query}?format=4")

        return response.text

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Weather does not support async")


class CodeGenTool(BaseTool):
    name = "CodeGen"
    description = (
        "useful for when you want to generate code using CodeGen;"
        "CodeGen is not that good at generating code from natual language;"
        "it is decent at generating completion to existing code;"
        "it should be called using the string as an arugment"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
    model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono").to(
        device
    )

    def _run(self, query: str) -> str:
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
        sample = self.model.generate(**inputs, max_length=128)

        return self.tokenizer.decode(
            sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"]
        )

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("CodeGen does not support async")


CUSTOM_TOOLS = [CustomExitTool(), WeatherTool(), CodeGenTool()]
