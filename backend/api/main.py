"""
FastAPI Application

Main FastAPI application for DataChat with:
- Lifespan management for resource initialization/cleanup
- CORS middleware for frontend integration
- Global exception handlers for agent errors
- Health and chat endpoints

Usage:
    uvicorn backend.api.main:app --reload --port 8000
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.agents.base import AgentError
from backend.api.routes import chat, health
from backend.config import get_settings
from backend.connectors.base import ConnectionError as ConnectorConnectionError
from backend.connectors.base import QueryError
from backend.connectors.postgres import PostgresConnector
from backend.knowledge.graph import KnowledgeGraph
from backend.knowledge.retriever import Retriever
from backend.knowledge.vectors import VectorStore
from backend.pipeline.orchestrator import DataChatPipeline

logger = logging.getLogger(__name__)

# Global state for pipeline and components
app_state = {
    "pipeline": None,
    "vector_store": None,
    "knowledge_graph": None,
    "connector": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Lifespan context manager for startup and shutdown.

    Initializes:
    - Vector store (Chroma)
    - Knowledge graph (NetworkX)
    - Database connector (PostgreSQL)
    - Pipeline orchestrator
    """
    config = get_settings()
    logger.info("Starting DataChat API server...")

    try:
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = VectorStore()
        await vector_store.initialize()
        app_state["vector_store"] = vector_store

        # Initialize knowledge graph
        logger.info("Initializing knowledge graph...")
        knowledge_graph = KnowledgeGraph()
        app_state["knowledge_graph"] = knowledge_graph

        # Initialize retriever
        logger.info("Initializing retriever...")
        retriever = Retriever(
            vector_store=vector_store,
            knowledge_graph=knowledge_graph,
        )

        # Initialize database connector
        logger.info("Initializing database connector...")
        from urllib.parse import urlparse

        db_url_str = str(config.database.url)
        parsed = urlparse(db_url_str)
        connector = PostgresConnector(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            database=parsed.path.lstrip("/") if parsed.path else "datachat",
            user=parsed.username or "postgres",
            password=parsed.password or "",
        )
        await connector.connect()
        app_state["connector"] = connector

        # Initialize pipeline
        logger.info("Initializing pipeline orchestrator...")
        pipeline = DataChatPipeline(
            retriever=retriever,
            connector=connector,
            max_retries=3,
        )
        app_state["pipeline"] = pipeline

        logger.info("DataChat API server started successfully")

        yield  # Application runs here

    finally:
        # Cleanup on shutdown
        logger.info("Shutting down DataChat API server...")

        if app_state["connector"]:
            try:
                await app_state["connector"].close()
                logger.info("Database connector closed")
            except Exception as e:
                logger.error(f"Error closing connector: {e}")

        logger.info("DataChat API server shut down complete")


# Create FastAPI app
app = FastAPI(
    title="DataChat API",
    description="AI-powered natural language interface for data warehouses",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
config = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(AgentError)
async def agent_error_handler(request: Request, exc: AgentError) -> JSONResponse:
    """Handle agent errors with context."""
    logger.error(
        f"Agent error: {exc}",
        extra={
            "agent": exc.agent if hasattr(exc, "agent") else "unknown",
            "recoverable": exc.recoverable if hasattr(exc, "recoverable") else False,
        },
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "agent_error",
            "message": str(exc),
            "agent": exc.agent if hasattr(exc, "agent") else "unknown",
            "recoverable": exc.recoverable if hasattr(exc, "recoverable") else False,
        },
    )


@app.exception_handler(ConnectorConnectionError)
async def connection_error_handler(request: Request, exc: ConnectorConnectionError) -> JSONResponse:
    """Handle database connection errors."""
    logger.error(f"Database connection error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "connection_error",
            "message": "Database connection failed. Please try again later.",
        },
    )


@app.exception_handler(QueryError)
async def query_error_handler(request: Request, exc: QueryError) -> JSONResponse:
    """Handle query execution errors."""
    logger.error(f"Query execution error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "query_error",
            "message": "Failed to execute query. Please check your request.",
        },
    )


# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])


# Root endpoint
@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": "DataChat API",
        "version": "0.1.0",
        "description": "Natural language interface for data warehouses",
        "docs": "/docs",
    }


def get_pipeline() -> DataChatPipeline:
    """Get the initialized pipeline instance."""
    if app_state["pipeline"] is None:
        raise RuntimeError("Pipeline not initialized")
    return app_state["pipeline"]


def get_connector() -> PostgresConnector:
    """Get the initialized database connector."""
    if app_state["connector"] is None:
        raise RuntimeError("Connector not initialized")
    return app_state["connector"]
