from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_port: int = 8501
    app_log_level: str = "INFO"
    app_env: str = "dev"

    default_model: Literal["resnet50", "vit_b_16"] = "resnet50"
    model_cache_dir: Path = Path("/app/.cache/torch")
    device: Literal["cpu", "cuda"] = "cpu"

    ig_steps: int = 50
    smoothgrad_samples: int = 50
    smoothgrad_sigma: float = 0.15
    occlusion_patch: int = 32
    occlusion_stride: int = 16
    lime_samples: int = Field(default=300, ge=100)
    lime_segments: int = 50

    app_hostname: str = "xai.local.pl"


settings = Settings()
