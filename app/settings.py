from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # overridden by environment variables (in Makefile and Helm values files)
    environment: str = "dev"
    project_name: str = "xbot"
    api_version: str = "0.0.0"
    vector_store_url: str = "http://localhost:6333"


settings = Settings()
