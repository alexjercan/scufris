import requests
import torch
from langchain.tools import BaseTool
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor,
)


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
        # TODO: Make sure this can never crash
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

    device: str = None
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
        self.model = AutoModelForCausalLM.from_pretrained(
            "Salesforce/codegen-350M-mono"
        ).to(self.device)

    def _run(self, query: str) -> str:
        # TODO: Make sure this can never crash
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
        sample = self.model.generate(**inputs, max_length=128)

        return self.tokenizer.decode(
            sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"]
        )

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("CodeGen does not support async")


class ImageCaptioningTool(BaseTool):
    name = "ImageCaptioning"
    description = (
        "useful for when you want to generate the caption for an image;"
        "the tool expects as input the type of the image, which can be: path, url or camera"
        "then if the type is path it expects another string with the path;"
        "if it is url it also expects a string with the url of the image;"
        "if the input is camera then it expects a number that indicates the device,"
        "for example the webcam should be 0;"
        "the inputs should be separated by semicolon"
    )

    device: str = None
    torch_dtype: str = None
    processor: BlipProcessor = None
    model: BlipForConditionalGeneration = None

    def __init__(self):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if "cuda" in self.device else torch.float32

        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=self.torch_dtype
        ).to(self.device)

    def _run(self, query: str) -> str:
        # TODO: Make sure this can never crash
        image_type, path = query.split(";", maxsplit=1)

        image = None
        if image_type == "path":
            image = Image.open(path)
        if image_type == "url":
            return path
        if image_type == "camera":
            return path
        if image is None:
            return "Invalid type for the image. Try another tool."

        inputs = self.processor(image, return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        out = self.model.generate(**inputs)
        captions = self.processor.decode(out[0], skip_special_tokens=True)

        return captions

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("ImageCaptioning does not support async")


CUSTOM_TOOLS = [
    CustomExitTool(),
    WeatherTool(),
    # CodeGenTool(),
    ImageCaptioningTool(),
]
