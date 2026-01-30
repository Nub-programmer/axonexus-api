from pydantic_settings import BaseSettings
from functools import lru_cache
from dotenv import load_dotenv
import os

# Load .env file at the very beginning
load_dotenv()

class Settings(BaseSettings):
    app_name: str = "AxonNexus API"
    app_version: str = "0.1.0"
    debug: bool = False
    api_key: str = os.getenv("API_KEY", "axn_test_123")
    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
    
    class Config:
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
