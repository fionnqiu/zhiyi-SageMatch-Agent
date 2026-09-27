"""Runtime settings loaded from the repo-root .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """First-slice settings: Postgres, admin gate, and one LLM endpoint."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    postgres_user: str = "postgres"
    postgres_password: str = ""
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "sagematch"

    admin_password: str = "sagematch-admin"
    anonymous_user_id: str = "local-user"

    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = "glm-5.3-flash"
    # JSON map: model -> {"input_per_million": amount, "output_per_million": amount}.
    # Unknown models have no defensible cost estimate.
    llm_pricing_json: str = "{}"
    # Optional request limits. Provider usage is required when a limit is set;
    # cost limits also require explicit per-model rates in llm_pricing_json.
    graph_token_budget: int | None = Field(default=None, ge=0)
    graph_cost_budget: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    # Only RAG model switches, identifiers, endpoints, and credentials live in .env.
    sagematch_rag_embedding_enabled: bool = True
    sagematch_rag_embedding_model: str = ""
    sagematch_rag_embedding_base_url: str = ""
    sagematch_rag_embedding_api_key: str = ""
    sagematch_rag_rerank_enabled: bool = False
    sagematch_rag_rerank_model: str = ""
    sagematch_rag_rerank_base_url: str = ""
    sagematch_rag_rerank_api_key: str = ""
    sagematch_rag_query_rewrite_enabled: bool = False
    sagematch_rag_query_rewrite_model: str = ""
    sagematch_rag_query_rewrite_base_url: str = ""
    sagematch_rag_query_rewrite_api_key: str = ""

    @property
    def database_url(self) -> str:
        # Build DSN from fields so passwords containing '#' stay intact.
        from urllib.parse import quote_plus

        user = quote_plus(self.postgres_user)
        password = quote_plus(self.postgres_password)
        return (
            f"postgresql+psycopg://{user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
