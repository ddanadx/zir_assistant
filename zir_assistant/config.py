"""Налаштування проекту — читаються з .env + environment."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    @property
    def data_dir(self) -> Path:
        d = self.project_root / "data"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def scraped_path(self) -> Path:
        return self.data_dir / "zir.jsonl"

    @property
    def chroma_dir(self) -> Path:
        d = self.data_dir / "chroma"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # Gemini
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    gemini_chat_model: str = Field(default="gemini-2.5-flash", validation_alias="GEMINI_CHAT_MODEL")
    gemini_embed_model: str = Field(default="gemini-embedding-001", validation_alias="GEMINI_EMBED_MODEL")

    # Scraper
    scrape_base_url: str = "https://zir.tax.gov.ua/main/bz/view/"
    scrape_max_id: int = Field(default=35000, validation_alias="SCRAPE_MAX_ID")
    scrape_concurrency: int = Field(default=4, validation_alias="SCRAPE_CONCURRENCY")
    scrape_delay_ms: int = Field(default=500, validation_alias="SCRAPE_DELAY_MS")
    scrape_user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        validation_alias="SCRAPE_USER_AGENT",
    )
    scrape_timeout_s: float = 30.0

    # RAG
    rag_top_k: int = Field(default=6, validation_alias="RAG_TOP_K")
    rag_embed_batch: int = Field(default=50, validation_alias="RAG_EMBED_BATCH")
    rag_collection_name: str = "zir_qa"

    # Server
    host: str = Field(default="127.0.0.1", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")


settings = Settings()
