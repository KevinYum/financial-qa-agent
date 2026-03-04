"""Centralized configuration loaded from environment variables and .env file."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with defaults suitable for development."""

    # LLM Provider (works with OpenAI and OpenRouter)
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    # OpenRouter extras (only used when llm_base_url points to OpenRouter)
    openrouter_http_referer: str = ""
    openrouter_x_title: str = "Financial QA Agent"

    # Brave Search API
    brave_api_key: str = ""

    # ChromaDB
    chroma_persist_dir: str = "data/chroma"
    chroma_collection_name: str = "financial_knowledge"

    # Knowledge base tuning
    kb_max_results: int = 3
    kb_max_distance: float = 0.5

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
