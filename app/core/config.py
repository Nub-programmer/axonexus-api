from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "AxonNexus API"
    app_version: str = "0.1.0"
    debug: bool = False
    api_key: str = "axn_test_123"
    nvidia_api_key: str = ""
    
    class Config:
        env_file = ".env"
        extra = "ignore"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
