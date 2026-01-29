from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "AxonNexus API"
    app_version: str = "0.1.0"
    debug: bool = False
    api_key: str = "axn_test_123"
    openai_key_1: str = ""
    openai_key_2: str = ""
    openai_key_3: str = ""
    
    class Config:
        env_file = ".env"
        extra = "ignore"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
