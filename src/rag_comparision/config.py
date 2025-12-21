"""Application configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    app_name: str = 'RAG Comparison Project'
    debug: bool = False
    host: str = '0.0.0.0'
    port: int = 8000

    class Config:
        """Pydantic config."""

        env_file = '.env'
        env_file_encoding = 'utf-8'


settings = Settings()
