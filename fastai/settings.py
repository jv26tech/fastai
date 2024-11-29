from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    QDRANT_API_KEY: str
    QDRANT_URL: str
    GOOGLE_API_KEY: str
