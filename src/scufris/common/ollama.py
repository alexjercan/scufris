
from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict


class OllamaConfig(BaseModel):
    """Configuration for Ollama."""

    model_config = SettingsConfigDict(env_prefix="SCUFRIS_OLLAMA_")

    model: str = Field(
        default="qwen3",
        description="Ollama model to use for summarization.",
    )
    think: bool = Field(
        default=True,
        description="Whether to enable 'think' mode in Ollama.",
    )

