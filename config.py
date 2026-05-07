"""
config.py - Centralised configuration for the Strategic Market Intelligence Pipeline.

All secrets and tunables are loaded from environment variables (or a .env file),
keeping the codebase fully 12-factor compliant.

Usage:
    from config import settings
    print(settings.duckdb_path)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root - resolves regardless of working-directory
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Settings model (auto-populated from environment / .env)
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """
    Application-wide settings.  All values can be overridden via environment
    variables or a ``.env`` file placed at the project root.

    Priority (highest → lowest):
        1. Shell environment variables
        2. ``.env`` file
        3. Default values declared here
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Anthropic
    # ------------------------------------------------------------------
    anthropic_api_key: str = Field(
        default="",
        alias="ANTHROPIC_API_KEY",
        description="Anthropic API key - required for LLM extraction engine.",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-5",
        alias="ANTHROPIC_MODEL",
        description="Model used for signal extraction (claude-sonnet-4-5 is cost-optimal; claude-opus-4 for max quality).",
    )
    anthropic_max_tokens: int = Field(
        default=1024,
        alias="ANTHROPIC_MAX_TOKENS",
        description="Maximum tokens in LLM completion response.",
    )

    # ------------------------------------------------------------------
    # LexisNexis (stub - enterprise connector)
    # ------------------------------------------------------------------
    lexisnexis_client_id: str = Field(
        default="",
        alias="LEXISNEXIS_CLIENT_ID",
        description="LexisNexis OAuth2 client_id.",
    )
    lexisnexis_client_secret: str = Field(
        default="",
        alias="LEXISNEXIS_CLIENT_SECRET",
        description="LexisNexis OAuth2 client_secret.",
    )
    lexisnexis_base_url: str = Field(
        default="https://services-api.lexisnexis.com",
        alias="LEXISNEXIS_BASE_URL",
        description="LexisNexis REST API base URL.",
    )
    lexisnexis_scopes: str = Field(
        default="news",
        alias="LEXISNEXIS_SCOPES",
        description="Space-separated OAuth2 scopes requested from LexisNexis.",
    )

    # ------------------------------------------------------------------
    # Factiva (stub - enterprise connector placeholder)
    # ------------------------------------------------------------------
    factiva_user_key: str = Field(
        default="",
        alias="FACTIVA_USER_KEY",
        description="Dow Jones Factiva user key.",
    )

    # ------------------------------------------------------------------
    # Storage - DuckDB Medallion Lake
    # ------------------------------------------------------------------
    duckdb_path: Path = Field(
        default=PROJECT_ROOT / "data" / "market_intelligence.duckdb",
        alias="DUCKDB_PATH",
        description="File-system path for the persistent DuckDB database.",
    )

    # ------------------------------------------------------------------
    # RSS Feed Configuration
    # ------------------------------------------------------------------
    rss_feed_urls: List[str] = Field(
        default=[
            # Reinsurance / insurance industry
            "https://www.insurancejournal.com/feed/",
            "https://www.reinsurancene.ws/feed/",
            # Catastrophe & emerging risk
            "https://www.artemis.bm/feed/",
            # General financial / macro
            "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
            "https://www.ft.com/rss/home",
        ],
        alias="RSS_FEED_URLS",
        description="List of RSS/Atom feed URLs consumed by the RSSConnector.",
    )
    rss_max_entries_per_feed: int = Field(
        default=20,
        alias="RSS_MAX_ENTRIES_PER_FEED",
        description="Maximum articles to ingest per feed per pipeline run.",
    )

    # ------------------------------------------------------------------
    # Pipeline behaviour
    # ------------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Python logging level (DEBUG | INFO | WARNING | ERROR).",
    )
    pipeline_dry_run: bool = Field(
        default=False,
        alias="PIPELINE_DRY_RUN",
        description="If True, ingestion runs but LLM processing is skipped.",
    )
    max_bronze_batch_size: int = Field(
        default=50,
        alias="MAX_BRONZE_BATCH_SIZE",
        description="Maximum number of raw records processed per LLM batch.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("duckdb_path", mode="before")
    @classmethod
    def ensure_parent_exists(cls, v: object) -> Path:
        path = Path(str(v))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return upper


# ---------------------------------------------------------------------------
# Singleton instance - import this throughout the codebase
# ---------------------------------------------------------------------------
settings = Settings()

# Configure root logger as soon as settings are resolved
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

logger.debug("Settings loaded: duckdb_path=%s  model=%s", settings.duckdb_path, settings.anthropic_model)
