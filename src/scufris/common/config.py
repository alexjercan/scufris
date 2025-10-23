from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Main configuration for the application."""

    model_config = SettingsConfigDict(env_prefix="SCUFRIS_")

    projects_dir: Path = Field(
        default=Path(Path.cwd(), "projects"),
        description="Directory to store projects.",
    )
