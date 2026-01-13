"""
Application Configuration

Pydantic-based settings management using environment variables.
Supports nested configuration, validation, and caching.

Usage:
    from backend.config import get_settings

    settings = get_settings()
    print(settings.openai.api_key)
    print(settings.database.url)
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, PostgresDsn, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseSettings):
    """OpenAI API configuration."""

    api_key: str = Field(
        ...,
        description="OpenAI API key for LLM access",
        min_length=20,
    )
    model: str = Field(
        default="gpt-4o",
        description="Default OpenAI model for SQL generation",
    )
    model_mini: str = Field(
        default="gpt-4o-mini",
        description="Lightweight model for classification tasks",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM responses (0.0 = deterministic)",
    )
    max_tokens: int = Field(
        default=2000,
        gt=0,
        le=16000,
        description="Maximum tokens per LLM response",
    )
    timeout: int = Field(
        default=30,
        gt=0,
        description="Request timeout in seconds",
    )

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        env_file=".env",
        extra="ignore",
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate OpenAI API key format."""
        if not v.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v


class DatabaseSettings(BaseSettings):
    """Database configuration."""

    url: PostgresDsn = Field(
        ...,
        description="PostgreSQL connection URL",
    )
    pool_size: int = Field(
        default=5,
        gt=0,
        le=20,
        description="Database connection pool size",
    )
    max_overflow: int = Field(
        default=10,
        ge=0,
        description="Maximum overflow connections",
    )
    pool_timeout: int = Field(
        default=30,
        gt=0,
        description="Connection pool timeout in seconds",
    )
    echo: bool = Field(
        default=False,
        description="Echo SQL queries to logs (useful for debugging)",
    )

    model_config = SettingsConfigDict(
        env_prefix="DATABASE_",
        env_file=".env",
        extra="ignore",
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: PostgresDsn) -> PostgresDsn:
        """Validate database URL scheme."""
        # PostgresDsn already validates the scheme, but we can add custom logic if needed
        return v


class ChromaSettings(BaseSettings):
    """Chroma vector store configuration."""

    persist_dir: Path = Field(
        default=Path("./chroma_data"),
        description="Directory for Chroma vector store persistence",
    )
    collection_name: str = Field(
        default="datachat_knowledge",
        description="Name of the Chroma collection",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model",
    )
    chunk_size: int = Field(
        default=512,
        gt=0,
        le=8192,
        description="Text chunk size for embeddings",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Overlap between text chunks",
    )
    top_k: int = Field(
        default=5,
        gt=0,
        le=20,
        description="Number of top results to retrieve",
    )

    model_config = SettingsConfigDict(
        env_prefix="CHROMA_",
        env_file=".env",
        extra="ignore",
    )

    @field_validator("persist_dir")
    @classmethod
    def validate_persist_dir(cls, v: Path) -> Path:
        """Ensure persist directory exists."""
        v.mkdir(parents=True, exist_ok=True)
        return v.resolve()

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "ChromaSettings":
        """Ensure chunk overlap is less than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        return self


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Application log level",
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format",
    )
    date_format: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Log timestamp format",
    )
    file: Optional[Path] = Field(
        default=None,
        description="Optional log file path (None = stdout only)",
    )

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=".env",
        extra="ignore",
    )

    def configure(self) -> None:
        """Configure Python logging with these settings."""
        handlers: list[logging.Handler] = [logging.StreamHandler()]

        if self.file:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(self.file))

        logging.basicConfig(
            level=getattr(logging, self.level),
            format=self.format,
            datefmt=self.date_format,
            handlers=handlers,
            force=True,  # Override any existing configuration
        )


class Settings(BaseSettings):
    """
    Main application settings.

    Loads configuration from environment variables and .env file.
    Settings are nested by domain (openai, database, chroma, logging).

    Environment Variables:
        ENVIRONMENT: Deployment environment (development, staging, production)
        APP_NAME: Application name for logging and metrics
        DEBUG: Enable debug mode
        API_HOST: API server host
        API_PORT: API server port
        OPENAI_*: OpenAI configuration (see OpenAISettings)
        DATABASE_*: Database configuration (see DatabaseSettings)
        CHROMA_*: Vector store configuration (see ChromaSettings)
        LOG_*: Logging configuration (see LoggingSettings)

    Example:
        >>> settings = get_settings()
        >>> settings.openai.api_key
        'sk-...'
        >>> settings.is_production
        False
    """

    # Application settings
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment",
    )
    app_name: str = Field(
        default="DataChat",
        description="Application name",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host",
    )
    api_port: int = Field(
        default=8000,
        gt=0,
        le=65535,
        description="API server port",
    )

    # Nested settings
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    @property
    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.environment == "staging"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @model_validator(mode="after")
    def configure_logging(self) -> "Settings":
        """Configure logging when settings are loaded."""
        self.logging.configure()
        return self

    def model_post_init(self, __context) -> None:
        """Log configuration on initialization."""
        logger = logging.getLogger(__name__)
        logger.info(
            f"Settings loaded for {self.app_name} ({self.environment})",
            extra={
                "environment": self.environment,
                "debug": self.debug,
                "openai_model": self.openai.model,
                "database_pool_size": self.database.pool_size,
                "chroma_collection": self.chroma.collection_name,
            }
        )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.

    Uses functools.lru_cache to ensure settings are loaded only once.
    This is the recommended way to access settings throughout the application.

    Returns:
        Settings: Singleton settings instance

    Example:
        >>> from backend.config import get_settings
        >>> settings = get_settings()
        >>> print(settings.openai.api_key)
    """
    return Settings()


def clear_settings_cache() -> None:
    """
    Clear the settings cache.

    Useful for testing when you need to reload settings with different
    environment variables.

    Example:
        >>> import os
        >>> os.environ["ENVIRONMENT"] = "production"
        >>> clear_settings_cache()
        >>> settings = get_settings()  # Reloads with new env vars
    """
    get_settings.cache_clear()
