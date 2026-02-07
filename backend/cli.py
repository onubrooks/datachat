"""
DataChat CLI

Command-line interface for interacting with DataChat.

Usage:
    datachat chat                          # Interactive REPL mode
    datachat ask "What's the revenue?"     # Single query mode
    datachat connect "connection_string"   # Set database connection
    datachat dp list                       # List DataPoints
    datachat dp add schema file.json       # Add DataPoint
    datachat dp sync                       # Rebuild vectors and graph
    datachat profile start                 # Start profiling via API
    datachat dp generate                   # Generate DataPoints via API
    datachat dev                           # Run backend + frontend dev servers
    datachat reset                         # Reset system state for testing
    datachat status                        # Show connection status
"""

import os

os.environ.setdefault("ABSL_LOGGING_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_LOGGING_STDERR_THRESHOLD", "3")
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_TRACE", "")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import asyncio
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import click
import httpx
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from backend.config import clear_settings_cache, get_settings
from backend.connectors.postgres import PostgresConnector
from backend.initialization.initializer import SystemInitializer
from backend.knowledge.datapoints import DataPointLoader
from backend.knowledge.graph import KnowledgeGraph
from backend.knowledge.retriever import Retriever
from backend.knowledge.vectors import VectorStore
from backend.pipeline.orchestrator import DataChatPipeline
from backend.settings_store import apply_config_defaults
from backend.tools import ToolExecutor, initialize_tools
from backend.tools.base import ToolContext

console = Console()
API_BASE_URL = os.getenv("DATA_CHAT_API_URL", "http://localhost:8000")


def configure_cli_logging() -> None:
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    logging.disable(logging.CRITICAL)
    logging.basicConfig(level=logging.CRITICAL)
    for logger_name in ("backend", "httpx", "openai", "asyncio", "google", "grpc"):
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)


# ============================================================================
# CLI State Management
# ============================================================================


class CLIState:
    """Manage CLI state (connection, pipeline, etc.)."""

    def __init__(self):
        self.refresh_paths()

    def refresh_paths(self) -> None:
        """Refresh config paths (useful when HOME changes in tests)."""
        self.config_dir = Path.home() / ".datachat"
        self.config_file = self.config_dir / "config.json"
        self.config_dir.mkdir(exist_ok=True, mode=0o700)

    def ensure_paths(self) -> None:
        """Ensure config directory exists and is writable."""
        try:
            self.config_dir.mkdir(exist_ok=True, mode=0o700)
            if not os.access(self.config_dir, os.W_OK):
                raise PermissionError("Config directory not writable")
        except OSError:
            self.refresh_paths()

    def load_config(self) -> dict[str, Any]:
        """Load CLI configuration."""
        if self.config_file.exists():
            with open(self.config_file) as f:
                return json.load(f)
        return {}

    def save_config(self, config: dict[str, Any]) -> None:
        """Save CLI configuration."""
        self.ensure_paths()
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)
        try:
            self.config_file.chmod(0o600)
        except OSError:
            pass

    def get_connection_string(self) -> str | None:
        """Get stored connection string."""
        config = self.load_config()
        return config.get("connection_string") or config.get("target_database_url")

    def set_target_database_url(self, connection_string: str) -> None:
        """Set target database URL."""
        config = self.load_config()
        config["target_database_url"] = connection_string
        config["connection_string"] = connection_string
        self.save_config(config)

    def set_connection_string(self, connection_string: str) -> None:
        """Set connection string."""
        self.set_target_database_url(connection_string)

    def get_system_database_url(self) -> str | None:
        """Get stored system database URL."""
        return self.load_config().get("system_database_url")

    def set_system_database_url(self, system_database_url: str) -> None:
        """Set system database URL."""
        config = self.load_config()
        config["system_database_url"] = system_database_url
        self.save_config(config)


state = CLIState()


# ============================================================================
# Helper Functions
# ============================================================================


async def create_pipeline_from_config() -> DataChatPipeline:
    """Create pipeline from configuration."""
    apply_config_defaults()
    settings = get_settings()

    # Initialize vector store
    vector_store = VectorStore()
    await vector_store.initialize()

    # Initialize knowledge graph
    knowledge_graph = KnowledgeGraph()

    # Initialize retriever
    retriever = Retriever(
        vector_store=vector_store,
        knowledge_graph=knowledge_graph,
    )

    # Initialize connector
    # Prefer .env / settings over persisted CLI state so local project config wins.
    connection_string = (
        str(settings.database.url)
        if settings.database.url
        else state.get_connection_string()
    )
    if not connection_string:
        console.print("[red]No target database configured.[/red]")
        console.print(
            "[yellow]Hint: Set DATABASE_URL in .env, use 'datachat connect', "
            "or run 'datachat setup'.[/yellow]"
        )
        raise click.ClickException("Missing target database")

    # Parse connection string
    from urllib.parse import urlparse

    parsed = urlparse(connection_string)
    connector = PostgresConnector(
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        database=parsed.path.lstrip("/") if parsed.path else "datachat",
        user=parsed.username or "postgres",
        password=parsed.password or "",
    )

    try:
        await connector.connect()
    except Exception as e:
        console.print(f"[red]Failed to connect to database: {e}[/red]")
        console.print("[yellow]Hint: Use 'datachat connect' to set connection string[/yellow]")
        raise

    initializer = SystemInitializer(
        {
            "connector": connector,
            "vector_store": vector_store,
        }
    )
    status_state = await initializer.status()
    if not status_state.is_initialized:
        if not status_state.has_databases:
            console.print("[red]DataChat requires setup before queries can run.[/red]")
            if not status_state.has_system_database:
                console.print(
                    "[yellow]Note: SYSTEM_DATABASE_URL is not set. Registry/profiling and "
                    "demo data are unavailable.[/yellow]"
                )
            for step in status_state.setup_required:
                console.print(f"[yellow]- {step.title}: {step.description}[/yellow]")
            console.print(
                "[cyan]Hint: Run 'datachat setup' or 'datachat demo' to continue.[/cyan]"
            )
            raise click.ClickException("System not initialized")

        if not status_state.has_datapoints:
            console.print(
                "[yellow]No DataPoints loaded. Continuing with live schema only.[/yellow]"
            )
            console.print(
                "[cyan]Hint: Run 'datachat dp sync' or enable profiling for richer answers.[/cyan]"
            )

    # Create pipeline
    pipeline = DataChatPipeline(
        retriever=retriever,
        connector=connector,
        max_retries=3,
    )

    return pipeline


def format_answer(answer: str, sql: str | None = None, data: dict | None = None) -> None:
    """Format and display answer."""
    # Display answer
    console.print(Panel(Markdown(answer), title="[bold green]Answer[/bold green]"))

    # Display SQL if available
    if sql:
        console.print("\n[bold cyan]Generated SQL:[/bold cyan]")
        console.print(
            Panel(
                sql,
                title="SQL",
                border_style="cyan",
                highlight=True,
            )
        )

    # Display data if available
    if data and isinstance(data, dict):
        console.print("\n[bold cyan]Results:[/bold cyan]")
        table = Table(show_header=True, header_style="bold cyan")

        # Add columns
        for col_name in data.keys():
            table.add_column(col_name)

        # Add rows
        if data:
            num_rows = len(next(iter(data.values())))
            for i in range(num_rows):
                row = [str(data[col][i]) if i < len(data[col]) else "" for col in data]
                table.add_row(*row)

        console.print(table)


def _build_columnar_data(query_result: dict[str, Any] | None) -> dict[str, list] | None:
    """Build columnar data from query results."""
    if not query_result:
        return None
    data = query_result.get("data")
    if data is not None:
        return data
    rows = query_result.get("rows")
    columns = query_result.get("columns")
    if isinstance(rows, list) and isinstance(columns, list):
        return {col: [row.get(col) for row in rows] for col in columns}
    return None


def _format_source_footer(result: dict[str, Any]) -> str | None:
    source = result.get("answer_source")
    confidence = result.get("answer_confidence")
    if not source:
        return None
    if isinstance(confidence, (int, float)):
        return f"Source: {source} ({confidence:.2f})"
    return f"Source: {source}"


def _print_evidence(result: dict[str, Any]) -> None:
    evidence = result.get("evidence") or []
    if not evidence:
        console.print("[dim]No evidence items available.[/dim]")
        return
    table = Table(title="Evidence", show_header=True, header_style="bold cyan")
    table.add_column("DataPoint")
    table.add_column("Type")
    table.add_column("Reason")
    for item in evidence:
        if isinstance(item, dict):
            table.add_row(
                str(item.get("name") or item.get("datapoint_id") or "unknown"),
                str(item.get("type") or "DataPoint"),
                str(item.get("reason") or ""),
            )
    console.print(table)


def _emit_query_output(
    answer: str,
    sql: str | None,
    data: dict | None,
    result: dict[str, Any],
    evidence: bool,
    show_metrics: bool,
) -> None:
    console.print()
    format_answer(answer, sql, data)
    console.print()

    footer = _format_source_footer(result)
    if footer:
        console.print(f"[dim]{footer}[/dim]")
        console.print()

    if show_metrics:
        metrics = Table(show_header=False, box=None)
        metrics.add_row("⏱️  Latency:", f"{result.get('total_latency_ms', 0):.0f}ms")
        metrics.add_row("🤖 LLM Calls:", str(result.get("llm_calls", 0)))
        metrics.add_row("🔄 Retries:", str(result.get("retry_count", 0)))
        console.print(metrics)
        console.print()

    clarifying_questions = result.get("clarifying_questions") or []
    if clarifying_questions:
        console.print("[bold]Clarifying questions:[/bold]")
        for question in clarifying_questions:
            console.print(f"- {question}")
        console.print("[dim]Reply with your answer, or type 'exit' to quit.[/dim]")
        console.print()

    if evidence:
        _print_evidence(result)
        console.print()


def _print_query_output(
    answer: str,
    sql: str | None,
    data: dict | None,
    result: dict[str, Any],
    evidence: bool,
    show_metrics: bool,
    pager: bool,
) -> None:
    if pager:
        with console.pager():
            _emit_query_output(answer, sql, data, result, evidence, show_metrics)
    else:
        _emit_query_output(answer, sql, data, result, evidence, show_metrics)


def _compose_clarification_answer(questions: list[str]) -> str:
    if not questions:
        return "I need a bit more detail to continue."
    bullets = "\n".join(f"- {q}" for q in questions)
    return f"I need a bit more detail to continue:\n{bullets}"


def _clarification_limit_message(limit: int) -> str:
    return (
        f"I reached the clarification limit ({limit}). "
        "Please ask a fully specified question with table and metric, or type `exit`."
    )


def _should_exit_chat(query: str) -> bool:
    text = query.strip().lower()
    if not text:
        return False
    exit_phrases = {
        "exit",
        "quit",
        "q",
        "end",
        "bye",
        "goodbye",
        "stop",
        "done",
        "done for now",
        "im done",
        "i'm done",
        "im done for now",
        "i'm done for now",
        "lets talk later",
        "let's talk later",
        "talk later",
        "see you later",
        "no further questions",
        "no more questions",
        "nothing else",
        "end chat",
        "end the chat",
        "close chat",
        "close the chat",
        "stop chat",
    }
    if text in exit_phrases:
        return True
    if re.search(r"\bnever\s*mind\b", text):
        return True
    if re.search(r"\b(i'?m|im|we'?re|were)\s+done\b", text):
        return True
    if re.search(r"\b(done for now|done here|that'?s all|all set)\b", text):
        return True
    if re.search(r"\b(let'?s\s+)?talk\s+later\b", text):
        return True
    if re.search(r"\b(talk|see)\s+you\s+later\b", text):
        return True
    if re.search(r"\b(no\s+more|no\s+further)\s+questions\b", text):
        return True
    if re.search(r"\b(end|stop|quit|exit)\b.*\b(chat|conversation)\b", text):
        return True
    return False


def _contains_data_keywords(text: str) -> bool:
    keywords = {
        "table",
        "tables",
        "column",
        "columns",
        "row",
        "rows",
        "schema",
        "database",
        "sql",
        "query",
        "count",
        "sum",
        "average",
        "avg",
        "min",
        "max",
        "join",
        "group",
        "order",
        "select",
        "from",
        "data",
        "dataset",
        "warehouse",
    }
    return any(word in text for word in keywords)


def _maybe_local_intent_response(query: str) -> tuple[str, str, float] | None:
    """Return a non-DB response for obvious non-query intents."""
    text = query.strip().lower()
    if not text:
        return None
    if _should_exit_chat(text):
        return (
            "Got it. Ending the session. If you need more, just start a new chat.",
            "system",
            0.9,
        )
    if text in {"help", "what can you do", "what can you do?"}:
        return (
            "I can help you explore your connected data. Try: list tables, show first 5 rows "
            "from a table, or ask for totals and trends.",
            "system",
            0.8,
        )
    if not _contains_data_keywords(text) and any(
        re.search(pattern, text)
        for pattern in (
            r"\bjoke\b",
            r"\bweather\b",
            r"\bnews\b",
            r"\bsports\b",
            r"\bmovie\b",
            r"\bmusic\b",
            r"\brecipe\b",
            r"\bpoem\b",
            r"\bstory\b",
        )
    ):
        return (
            "I can help with questions about your connected data. Try: list tables, "
            "show first 5 rows from a table, or total sales last month.",
            "system",
            0.8,
        )
    return None


# ============================================================================
# CLI Commands
# ============================================================================


@click.group()
@click.version_option(version="0.1.0", prog_name="DataChat")
def cli():
    """DataChat - Natural language interface for data warehouses."""
    configure_cli_logging()
    pass


@cli.command()
@click.option("--evidence", is_flag=True, help="Show DataPoint evidence details")
@click.option(
    "--pager/--no-pager",
    default=False,
    help="Show each response in a scrollable pager.",
)
@click.option(
    "--max-clarifications",
    default=3,
    show_default=True,
    type=int,
    help="Maximum clarification prompts before stopping.",
)
def chat(evidence: bool, pager: bool, max_clarifications: int):
    """Interactive REPL mode for conversations."""
    console.print(
        Panel.fit(
            "[bold green]DataChat Interactive Mode[/bold green]\n"
            "Ask questions in natural language. Type 'exit' or 'quit' to leave.",
            border_style="green",
        )
    )

    conversation_history = []
    conversation_id = None

    async def run_chat():
        nonlocal conversation_id
        clarification_attempts = 0
        max_clarifications_limit = max(0, max_clarifications)

        pipeline = None
        try:
            while True:
                try:
                    # Get user input
                    query = console.input("[bold cyan]You:[/bold cyan] ")

                    if not query.strip():
                        continue

                    if _should_exit_chat(query):
                        console.print("\n[yellow]Goodbye![/yellow]")
                        break

                    if pipeline is None:
                        pipeline = await create_pipeline_from_config()
                        pipeline.max_clarifications = max(0, max_clarifications)
                        console.print("[green]✓ Pipeline initialized[/green]\n")

                    # Show loading indicator
                    with console.status("[cyan]Processing...[/cyan]", spinner="dots"):
                        result = await pipeline.run(
                            query=query,
                            conversation_history=conversation_history,
                        )

                    # Extract results
                    clarifying_questions = result.get("clarifying_questions") or []
                    answer = result.get("natural_language_answer") or ""
                    if not answer and clarifying_questions:
                        answer = _compose_clarification_answer(clarifying_questions)
                    if not answer:
                        answer = "No answer generated"
                    sql = result.get("validated_sql") or result.get("generated_sql")
                    query_result = result.get("query_result")
                    data = _build_columnar_data(query_result)

                    if clarifying_questions and clarification_attempts >= max_clarifications_limit:
                        result = {
                            **result,
                            "answer_source": "system",
                            "answer_confidence": 0.5,
                            "clarifying_questions": [],
                        }
                        answer = _clarification_limit_message(max_clarifications_limit)
                        sql = None
                        data = None
                        clarifying_questions = []
                    elif clarifying_questions:
                        clarification_attempts += 1
                    else:
                        clarification_attempts = 0

                    # Display results
                    _print_query_output(
                        answer=answer,
                        sql=sql,
                        data=data,
                        result=result,
                        evidence=evidence,
                        show_metrics=True,
                        pager=pager or evidence,
                    )

                    # Update conversation history
                    conversation_history.append({"role": "user", "content": query})
                    conversation_history.append({"role": "assistant", "content": answer})

                except KeyboardInterrupt:
                    console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
                    continue
                except Exception as e:
                    console.print(f"\n[red]Error: {e}[/red]")
                    continue

        except Exception as e:
            console.print(f"[red]Failed to initialize pipeline: {e}[/red]")
            sys.exit(1)
        finally:
            # Cleanup
            if pipeline is not None:
                try:
                    await pipeline.connector.close()
                except Exception:
                    pass

    asyncio.run(run_chat())


@cli.command()
@click.argument("query")
@click.option("--evidence", is_flag=True, help="Show DataPoint evidence details")
@click.option(
    "--pager/--no-pager",
    default=False,
    help="Show the response in a scrollable pager.",
)
@click.option(
    "--max-clarifications",
    default=3,
    show_default=True,
    type=int,
    help="Maximum clarification prompts before stopping.",
)
def ask(query: str, evidence: bool, pager: bool, max_clarifications: int):
    """Ask a single question and exit."""

    async def run_query():
        local_response = _maybe_local_intent_response(query)
        if local_response:
            answer, source, confidence = local_response
            _print_query_output(
                answer=answer,
                sql=None,
                data=None,
                result={
                    "answer_source": source,
                    "answer_confidence": confidence,
                    "clarifying_questions": [],
                },
                evidence=evidence,
                show_metrics=False,
                pager=pager or evidence,
            )
            return

        try:
            pipeline = await create_pipeline_from_config()
            pipeline.max_clarifications = max(0, max_clarifications)
            conversation_history = []
            current_query = query
            clarification_attempts = 0
            max_clarifications_limit = max(0, max_clarifications)

            while True:
                # Show loading with progress
                with console.status("[cyan]Processing query...[/cyan]", spinner="dots"):
                    result = await pipeline.run(
                        query=current_query,
                        conversation_history=conversation_history,
                    )

                # Extract results
                clarifying_questions = result.get("clarifying_questions") or []
                answer = result.get("natural_language_answer") or ""
                if not answer and clarifying_questions:
                    answer = _compose_clarification_answer(clarifying_questions)
                if not answer:
                    answer = "No answer generated"
                sql = result.get("validated_sql") or result.get("generated_sql")
                query_result = result.get("query_result")
                data = _build_columnar_data(query_result)

                if clarifying_questions and clarification_attempts >= max_clarifications_limit:
                    result = {
                        **result,
                        "answer_source": "system",
                        "answer_confidence": 0.5,
                        "clarifying_questions": [],
                    }
                    answer = _clarification_limit_message(max_clarifications_limit)
                    sql = None
                    data = None
                    clarifying_questions = []

                # Display results
                _print_query_output(
                    answer=answer,
                    sql=sql,
                    data=data,
                    result=result,
                    evidence=evidence,
                    show_metrics=False,
                    pager=pager or evidence,
                )

                if not clarifying_questions or clarification_attempts >= max_clarifications_limit:
                    break

                followup = console.input("[bold cyan]Clarification:[/bold cyan] ").strip()
                if not followup:
                    break
                if _should_exit_chat(followup):
                    console.print("[yellow]Goodbye![/yellow]")
                    break

                conversation_history.extend(
                    [
                        {"role": "user", "content": current_query},
                        {"role": "assistant", "content": answer},
                    ]
                )
                current_query = followup
                clarification_attempts += 1

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        finally:
            if "pipeline" in locals():
                try:
                    await pipeline.connector.close()
                except Exception:
                    pass

    asyncio.run(run_query())


@cli.command()
@click.argument("connection_string")
def connect(connection_string: str):
    """Set database connection string.

    Example:
        datachat connect postgresql://user:pass@localhost:5432/dbname
    """
    try:
        # Validate connection string format
        from urllib.parse import urlparse

        parsed = urlparse(connection_string)
        if not parsed.scheme or not parsed.netloc:
            console.print(
                "[red]Invalid connection string format[/red]\n"
                "Expected: postgresql://user:pass@host:port/dbname"
            )
            sys.exit(1)

        # Save connection string
        state.set_connection_string(connection_string)
        console.print("[green]✓ Connection string saved[/green]")
        console.print(f"Host: {parsed.hostname}")
        console.print(f"Port: {parsed.port or 5432}")
        console.print(f"Database: {parsed.path.lstrip('/')}")
        console.print(f"User: {parsed.username}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
def status():
    """Show connection and system status."""

    async def check_status():
        apply_config_defaults()
        table = Table(title="DataChat Status", show_header=True, header_style="bold cyan")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")

        # Check configuration
        settings = get_settings()
        table.add_row("Configuration", "✓", f"Environment: {settings.environment}")
        if settings.system_database.url:
            table.add_row("System DB", "✓", "Configured")
        else:
            table.add_row("System DB", "⚠️", "SYSTEM_DATABASE_URL not set")

        # Check connection string
        conn_str = state.get_connection_string()
        if conn_str:
            from urllib.parse import urlparse

            parsed = urlparse(conn_str)
            table.add_row(
                "Connection",
                "✓",
                f"{parsed.hostname}:{parsed.port}/{parsed.path.lstrip('/')}",
            )
        elif settings.database.url:
            table.add_row("Connection", "⚠️", "Using default from config")
        else:
            table.add_row("Connection", "✗", "No target database configured")

        # Check database connection
        try:
            if conn_str:
                connection_string = conn_str
            elif settings.database.url:
                connection_string = str(settings.database.url)
            else:
                raise RuntimeError("No target database configured")
            from urllib.parse import urlparse

            parsed = urlparse(connection_string)
            connector = PostgresConnector(
                host=parsed.hostname or "localhost",
                port=parsed.port or 5432,
                database=parsed.path.lstrip("/") if parsed.path else "datachat",
                user=parsed.username or "postgres",
                password=parsed.password or "",
            )
            await connector.connect()
            await connector.execute("SELECT 1")
            table.add_row("Database", "✓", "Connected")
            await connector.close()
        except Exception as e:
            table.add_row("Database", "✗", f"Error: {str(e)[:50]}")

        # Check vector store
        try:
            vector_store = VectorStore()
            await vector_store.initialize()
            count = await vector_store.get_count()
            table.add_row("Vector Store", "✓", f"{count} datapoints")
        except Exception as e:
            table.add_row("Vector Store", "✗", f"Error: {str(e)[:50]}")

        # Check knowledge graph
        try:
            graph = KnowledgeGraph()
            stats = graph.get_stats()
            table.add_row(
                "Knowledge Graph",
                "✓",
                f"{stats['total_nodes']} nodes, {stats['total_edges']} edges",
            )
        except Exception as e:
            table.add_row("Knowledge Graph", "✗", f"Error: {str(e)[:50]}")

        console.print(table)

    asyncio.run(check_status())


@cli.command()
@click.option("--backend-port", default=8000, show_default=True, type=int)
@click.option("--frontend-port", default=3000, show_default=True, type=int)
@click.option("--backend-host", default="127.0.0.1", show_default=True)
@click.option("--frontend-host", default="127.0.0.1", show_default=True)
@click.option("--no-backend", is_flag=True, help="Skip starting the backend API server.")
@click.option("--no-frontend", is_flag=True, help="Skip starting the frontend dev server.")
def dev(
    backend_port: int,
    frontend_port: int,
    backend_host: str,
    frontend_host: str,
    no_backend: bool,
    no_frontend: bool,
):
    """Run backend and frontend dev servers in one command."""
    processes: list[subprocess.Popen] = []

    if no_backend and no_frontend:
        raise click.ClickException("Nothing to run. Remove --no-backend or --no-frontend.")

    if not no_backend:
        backend_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.api.main:app",
            "--reload",
            "--host",
            backend_host,
            "--port",
            str(backend_port),
        ]
        console.print(f"[cyan]Starting backend:[/cyan] {' '.join(backend_cmd)}")
        processes.append(subprocess.Popen(backend_cmd))

    if not no_frontend:
        frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
        if not frontend_dir.exists():
            raise click.ClickException("Frontend directory not found. Run from repo root.")
        frontend_cmd = [
            "npm",
            "run",
            "dev",
            "--",
            "-p",
            str(frontend_port),
            "-H",
            frontend_host,
        ]
        console.print(f"[cyan]Starting frontend:[/cyan] {' '.join(frontend_cmd)}")
        processes.append(subprocess.Popen(frontend_cmd, cwd=str(frontend_dir)))

    try:
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping dev servers...[/yellow]")
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


@cli.command()
@click.option("--target-db", "database_url", help="Target database URL.")
@click.option(
    "--system-db",
    "system_database_url",
    help="System database URL for registry/profiling.",
)
@click.option(
    "--auto-profile/--no-auto-profile",
    default=None,
    help="Auto-profile database (generate DataPoints draft).",
)
@click.option(
    "--max-tables",
    type=int,
    default=None,
    help="Max tables to auto-profile (0 = all).",
)
@click.option(
    "--non-interactive",
    is_flag=True,
    help="Fail instead of prompting for missing values.",
)
def setup(
    database_url: str | None,
    system_database_url: str | None,
    auto_profile: bool | None,
    max_tables: int | None,
    non_interactive: bool,
):
    """Guide system initialization for first-time setup."""

    async def run_setup():
        apply_config_defaults()
        settings = get_settings()
        default_url = state.get_connection_string() or (
            str(settings.database.url) if settings.database.url else ""
        )

        console.print(
            Panel.fit(
                "[bold green]DataChat Setup[/bold green]\n"
                "Initialize your database connection and load DataPoints.",
                border_style="green",
            )
        )

        resolved_database_url = (
            database_url
            or default_url
            or (str(settings.database.url) if settings.database.url else "")
        )
        if not resolved_database_url:
            if non_interactive:
                raise click.ClickException("Missing target database URL.")
            resolved_database_url = click.prompt(
                "Target Database URL", default=default_url, show_default=True
            )

        resolved_system_database_url = system_database_url or (
            str(settings.system_database.url)
            if settings.system_database.url
            else None
        )
        if not resolved_system_database_url and not non_interactive:
            resolved_system_database_url = click.prompt(
                "System Database URL (for demo/registry)",
                default="postgresql://datachat:datachat_password@localhost:5432/datachat",
                show_default=True,
            )

        if resolved_database_url:
            state.set_target_database_url(resolved_database_url)
        if resolved_system_database_url:
            state.set_system_database_url(resolved_system_database_url)

        resolved_auto_profile = auto_profile
        if resolved_auto_profile is None:
            if non_interactive:
                resolved_auto_profile = False
            else:
                resolved_auto_profile = click.confirm(
                    "Auto-profile database (generate DataPoints draft)",
                    default=False,
                    show_default=True,
                )

        resolved_max_tables = max_tables
        if resolved_auto_profile:
            if resolved_max_tables is None and not non_interactive:
                resolved_max_tables = click.prompt(
                    "Max tables to auto-profile (0 = all)",
                    default=10,
                    show_default=True,
                    type=int,
                )
            if resolved_max_tables is not None and resolved_max_tables <= 0:
                resolved_max_tables = None

        vector_store = VectorStore()
        await vector_store.initialize()

        initializer = SystemInitializer({"vector_store": vector_store})
        status_state, message = await initializer.initialize(
            database_url=resolved_database_url,
            auto_profile=bool(resolved_auto_profile),
            system_database_url=resolved_system_database_url,
        )

        console.print(f"[green]{message}[/green]")
        if resolved_auto_profile:
            clear_settings_cache()
            apply_config_defaults()
            refreshed_settings = get_settings()
            if not refreshed_settings.system_database.url:
                console.print(
                    "[yellow]Auto-profiling requires SYSTEM_DATABASE_URL. "
                    "Set it in your shell or .env and rerun setup.[/yellow]"
                )
            elif not refreshed_settings.database_credentials_key:
                console.print(
                    "[yellow]Auto-profiling requires DATABASE_CREDENTIALS_KEY. "
                    "Set it in your shell or .env and rerun setup.[/yellow]"
                )
            else:
                try:
                    initialize_tools(refreshed_settings.tools.policy_path)
                    executor = ToolExecutor()
                    ctx = ToolContext(user_id="cli", correlation_id="setup", approved=True)
                    args = {"depth": "metrics_basic", "batch_size": 10}
                    if resolved_max_tables:
                        args["max_tables"] = resolved_max_tables
                    result = await executor.execute(
                        "profile_and_generate_datapoints",
                        args,
                        ctx,
                    )
                    pending_count = result.get("result", {}).get("pending_count")
                    if pending_count is not None:
                        console.print(
                            f"[green]✓ Auto-profiling generated {pending_count} pending DataPoints.[/green]"
                        )
                        if resolved_max_tables and pending_count > resolved_max_tables:
                            console.print(
                                "[dim]Note: each table can generate multiple DataPoints (schema + metrics).[/dim]"
                            )
                    else:
                        console.print("[green]✓ Auto-profiling completed.[/green]")
                except Exception as exc:
                    console.print(
                        f"[yellow]Auto-profiling failed to start: {exc}[/yellow]"
                    )
        if status_state.setup_required:
            console.print("[yellow]Remaining setup steps:[/yellow]")
            for step in status_state.setup_required:
                console.print(f"- {step.title}: {step.description}")
            console.print("[cyan]Hint: Run 'datachat dp sync' after adding DataPoints.[/cyan]")
        else:
            console.print("[green]✓ System initialized. You're ready to query.[/green]")

    asyncio.run(run_setup())


@cli.command()
@click.option("--include-target", is_flag=True, help="Also clear target database tables.")
@click.option(
    "--drop-all-target",
    is_flag=True,
    help="Drop all tables in the target database (dangerous). Requires --include-target.",
)
@click.option("--keep-config", is_flag=True, help="Keep ~/.datachat/config.json.")
@click.option("--keep-vectors", is_flag=True, help="Keep local vector store on disk.")
@click.option("--yes", is_flag=True, help="Skip confirmation prompts (use with caution).")
def reset(
    include_target: bool,
    drop_all_target: bool,
    keep_config: bool,
    keep_vectors: bool,
    yes: bool,
):
    """Reset system state for testing or clean setup."""

    async def run_reset() -> None:
        apply_config_defaults()
        settings = get_settings()

        if drop_all_target and not include_target:
            raise click.ClickException("--drop-all-target requires --include-target.")

        if not yes:
            console.print(
                Panel.fit(
                    "[bold red]Reset DataChat State[/bold red]\n"
                    "This clears system registry/profiling, local vectors, and saved config.",
                    border_style="red",
                )
            )
            if not click.confirm("Continue?", default=False, show_default=True):
                console.print("[yellow]Reset cancelled.[/yellow]")
                return

        system_db_url = (
            str(settings.system_database.url) if settings.system_database.url else None
        )
        if system_db_url:
            from urllib.parse import urlparse

            parsed = urlparse(system_db_url)
            connector = PostgresConnector(
                host=parsed.hostname or "localhost",
                port=parsed.port or 5432,
                database=parsed.path.lstrip("/") if parsed.path else "datachat",
                user=parsed.username or "postgres",
                password=parsed.password or "",
            )
            await connector.connect()
            try:
                await connector.execute(
                    "TRUNCATE database_connections, profiling_jobs, "
                    "profiling_profiles, pending_datapoints"
                )
                console.print("[green]✓ System DB state cleared[/green]")
            finally:
                await connector.close()
        else:
            console.print("[yellow]System DB not configured; skipped registry reset.[/yellow]")

        if include_target:
            target_db_url = (
                state.get_connection_string()
                or (str(settings.database.url) if settings.database.url else None)
            )
            if not target_db_url:
                console.print("[yellow]Target DB not configured; skipped target reset.[/yellow]")
            else:
                from urllib.parse import urlparse

                parsed = urlparse(target_db_url)
                connector = PostgresConnector(
                    host=parsed.hostname or "localhost",
                    port=parsed.port or 5432,
                    database=parsed.path.lstrip("/") if parsed.path else "datachat",
                    user=parsed.username or "postgres",
                    password=parsed.password or "",
                )
                await connector.connect()
                try:
                    if drop_all_target:
                        await connector.execute(
                            """
                            DO $$
                            DECLARE r RECORD;
                            BEGIN
                              FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public')
                              LOOP
                                EXECUTE 'DROP TABLE IF EXISTS public.' || quote_ident(r.tablename)
                                || ' CASCADE';
                              END LOOP;
                            END $$;
                            """
                        )
                        console.print("[green]✓ Target DB tables dropped[/green]")
                    else:
                        await connector.execute("DROP TABLE IF EXISTS orders CASCADE")
                        await connector.execute("DROP TABLE IF EXISTS users CASCADE")
                        console.print("[green]✓ Target DB demo tables cleared[/green]")
                finally:
                    await connector.close()

        if not keep_vectors:
            vector_store = VectorStore()
            await vector_store.initialize()
            await vector_store.clear()
            shutil.rmtree(settings.chroma.persist_dir, ignore_errors=True)
            console.print("[green]✓ Local vector store cleared[/green]")

        managed_dir = Path("datapoints") / "managed"
        if managed_dir.exists():
            shutil.rmtree(managed_dir, ignore_errors=True)
            console.print("[green]✓ Managed DataPoints cleared[/green]")

        if not keep_config:
            state.refresh_paths()
            if state.config_file.exists():
                try:
                    state.config_file.unlink()
                    console.print("[green]✓ Saved config cleared[/green]")
                except OSError:
                    console.print("[yellow]Failed to remove saved config.[/yellow]")

        console.print("[green]Reset complete.[/green]")

    asyncio.run(run_reset())


@cli.command()
@click.option(
    "--persona",
    type=click.Choice(["base", "analyst", "engineer", "platform", "executive"], case_sensitive=False),
    default="base",
    show_default=True,
    help="Persona-specific demo setup to load.",
)
@click.option("--reset", is_flag=True, help="Drop and re-seed demo tables.")
@click.option("--no-workspace", is_flag=True, help="Skip workspace indexing (if available).")
def demo(persona: str, reset: bool, no_workspace: bool):
    """Seed demo tables and load demo DataPoints."""

    async def run_demo():
        apply_config_defaults()
        settings = get_settings()
        if not settings.system_database.url:
            raise click.ClickException("SYSTEM_DATABASE_URL must be set to run the demo.")
        database_url = str(settings.system_database.url)
        persona_name = persona.lower()

        from urllib.parse import urlparse

        parsed = urlparse(database_url)
        connector = PostgresConnector(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            database=parsed.path.lstrip("/") if parsed.path else "datachat",
            user=parsed.username or "postgres",
            password=parsed.password or "",
        )

        console.print(f"[cyan]Seeding demo tables (persona: {persona_name})...[/cyan]")
        await connector.connect()
        try:
            if reset:
                await connector.execute("DROP TABLE IF EXISTS orders")
                await connector.execute("DROP TABLE IF EXISTS users")

            await connector.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    email TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    is_active BOOLEAN NOT NULL DEFAULT TRUE
                );
                """
            )
            await connector.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    amount NUMERIC(12,2) NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    order_date DATE NOT NULL DEFAULT CURRENT_DATE
                );
                """
            )

            user_count = await connector.execute("SELECT COUNT(*) AS count FROM users")
            if user_count.rows and user_count.rows[0]["count"] == 0:
                await connector.execute(
                    """
                    INSERT INTO users (email, is_active)
                    VALUES
                        ('alice@example.com', TRUE),
                        ('bob@example.com', TRUE),
                        ('charlie@example.com', FALSE)
                    """
                )

            order_count = await connector.execute("SELECT COUNT(*) AS count FROM orders")
            if order_count.rows and order_count.rows[0]["count"] == 0:
                await connector.execute(
                    """
                    INSERT INTO orders (user_id, amount, status, order_date)
                    VALUES
                        (1, 120.50, 'completed', CURRENT_DATE - INTERVAL '12 days'),
                        (1, 75.00, 'completed', CURRENT_DATE - INTERVAL '6 days'),
                        (2, 200.00, 'completed', CURRENT_DATE - INTERVAL '2 days'),
                        (3, 15.00, 'refunded', CURRENT_DATE - INTERVAL '20 days')
                    """
                )
        finally:
            await connector.close()

        base_dir = Path("datapoints") / "demo"
        datapoints_dir = base_dir
        if persona_name != "base":
            persona_dir = base_dir / persona_name
            if persona_dir.exists():
                datapoints_dir = persona_dir
            else:
                console.print(
                    f"[yellow]Persona DataPoints not found at {persona_dir}. Falling back to base demo.[/yellow]"
                )

        if not datapoints_dir.exists():
            console.print(f"[red]Demo DataPoints not found at {datapoints_dir}[/red]")
            raise click.ClickException("Missing demo DataPoints.")

        console.print("[cyan]Loading demo DataPoints...[/cyan]")
        loader = DataPointLoader()
        datapoints = loader.load_directory(datapoints_dir)
        if not datapoints:
            raise click.ClickException("No demo DataPoints loaded.")

        vector_store = VectorStore()
        await vector_store.initialize()
        await vector_store.clear()
        await vector_store.add_datapoints(datapoints)

        graph = KnowledgeGraph()
        for datapoint in datapoints:
            graph.add_datapoint(datapoint)

        if not no_workspace:
            workspace_root = Path("workspace_demo") / persona_name
            if workspace_root.exists():
                console.print(
                    "[yellow]Workspace indexing is not implemented yet. "
                    "Found workspace demo content but skipped indexing.[/yellow]"
                )
            else:
                console.print(
                    f"[yellow]Workspace demo folder not found at {workspace_root} (skipping).[/yellow]"
                )

        console.print("[green]✓ Demo data loaded. Try: datachat ask \"How many users are active?\"[/green]")

    asyncio.run(run_demo())


# ============================================================================
# DataPoint Commands
# ============================================================================


@cli.group(name="profile")
def profile():
    """Manage profiling jobs via API."""
    pass


@profile.command(name="start")
@click.option("--connection-id", required=True, help="Database connection UUID.")
@click.option("--sample-size", default=100, show_default=True, type=int)
@click.option("--tables", multiple=True, help="Optional table names to profile.")
def start_profile(connection_id: str, sample_size: int, tables: tuple[str, ...]):
    """Start profiling for a registered database connection."""
    payload = {"sample_size": sample_size, "tables": list(tables) or None}
    try:
        response = httpx.post(
            f"{API_BASE_URL}/api/v1/databases/{connection_id}/profile",
            json=payload,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        console.print(
            f"[green]✓ Profiling started[/green] job_id={data['job_id']}"
        )
    except Exception as exc:
        console.print(f"[red]Failed to start profiling: {exc}[/red]")
        sys.exit(1)


@profile.command(name="status")
@click.argument("job_id")
def profile_status(job_id: str):
    """Check profiling job status."""
    try:
        response = httpx.get(
            f"{API_BASE_URL}/api/v1/profiling/jobs/{job_id}", timeout=15.0
        )
        response.raise_for_status()
        data = response.json()
        console.print(json.dumps(data, indent=2))
    except Exception as exc:
        console.print(f"[red]Failed to fetch status: {exc}[/red]")
        sys.exit(1)


@cli.group(name="dp")
def datapoint():
    """Manage DataPoints (knowledge base)."""
    pass


# ============================================================================
# Tool Commands
# ============================================================================


@cli.group(name="tools")
def tools():
    """Manage tool execution and reports."""
    pass


@tools.command(name="list")
def list_tools():
    """List available tools."""
    try:
        response = httpx.get(f"{API_BASE_URL}/api/v1/tools", timeout=15.0)
        response.raise_for_status()
        data = response.json()
        table = Table(title="Tools", show_header=True, header_style="bold cyan")
        table.add_column("Name")
        table.add_column("Category")
        table.add_column("Approval")
        table.add_column("Enabled")
        for tool in data:
            table.add_row(
                tool.get("name", ""),
                tool.get("category", ""),
                "yes" if tool.get("requires_approval") else "no",
                "yes" if tool.get("enabled") else "no",
            )
        console.print(table)
    except Exception as exc:
        console.print(f"[red]Failed to list tools: {exc}[/red]")
        sys.exit(1)


@tools.command(name="run")
@click.argument("name")
@click.option("--approve", is_flag=True, help="Approve tool execution.")
def run_tool(name: str, approve: bool):
    """Run a tool via the API."""
    if not approve:
        approve = click.confirm(
            "This tool may trigger profiling or other actions. Proceed?", default=False
        )
    payload = {"name": name, "arguments": {}, "approved": approve}
    try:
        response = httpx.post(
            f"{API_BASE_URL}/api/v1/tools/execute", json=payload, timeout=60.0
        )
        response.raise_for_status()
        console.print(json.dumps(response.json(), indent=2))
    except Exception as exc:
        console.print(f"[red]Tool execution failed: {exc}[/red]")
        sys.exit(1)


@tools.command(name="quality-report")
def quality_report():
    """Run DataPoint quality report."""
    payload = {"name": "datapoint_quality_report", "arguments": {"limit": 10}}
    try:
        response = httpx.post(
            f"{API_BASE_URL}/api/v1/tools/execute", json=payload, timeout=30.0
        )
        response.raise_for_status()
        console.print(json.dumps(response.json(), indent=2))
    except Exception as exc:
        console.print(f"[red]Failed to run quality report: {exc}[/red]")
        sys.exit(1)


@datapoint.command(name="list")
def list_datapoints():
    """List all DataPoints in the knowledge base."""

    async def run_list():
        try:
            vector_store = VectorStore()
            await vector_store.initialize()

            # Get all datapoints without embedding calls
            results = await vector_store.list_datapoints(limit=1000)

            if not results:
                console.print("[yellow]No DataPoints found[/yellow]")
                return

            # Create table
            table = Table(
                title=f"DataPoints ({len(results)} found)",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("ID", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Name")
            table.add_column("Score", justify="right")

            for result in results:
                metadata = result.get("metadata", {})
                score = result.get("distance")
                score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "-"
                table.add_row(
                    metadata.get("datapoint_id", "unknown"),
                    metadata.get("type", "unknown"),
                    metadata.get("name", "unknown"),
                    score_text,
                )

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    asyncio.run(run_list())


@datapoint.group(name="pending")
def pending_datapoints():
    """Review pending DataPoints (requires backend)."""


def _fetch_pending_datapoints() -> list[dict[str, Any]]:
    response = httpx.get(f"{API_BASE_URL}/api/v1/datapoints/pending", timeout=15.0)
    response.raise_for_status()
    data = response.json()
    return data.get("pending", [])


def _resolve_default_connection_id() -> str:
    response = httpx.get(f"{API_BASE_URL}/api/v1/databases", timeout=15.0)
    response.raise_for_status()
    connections = response.json()
    if not isinstance(connections, list) or not connections:
        raise click.ClickException(
            "No database connections found. Run 'datachat setup' or add one in the UI."
        )
    default = next((item for item in connections if item.get("is_default")), None)
    connection = default or connections[0]
    connection_id = connection.get("connection_id")
    if not connection_id:
        raise click.ClickException("Default connection is missing an ID.")
    return str(connection_id)


def _resolve_latest_profile_id(connection_id: str) -> str:
    response = httpx.get(
        f"{API_BASE_URL}/api/v1/profiling/jobs/connection/{connection_id}/latest",
        timeout=15.0,
    )
    response.raise_for_status()
    data = response.json()
    if not data:
        raise click.ClickException(
            "No profiling job found. Run 'datachat profile start' or 'datachat setup --auto-profile'."
        )
    profile_id = data.get("profile_id")
    if not profile_id:
        raise click.ClickException("Latest profiling job is missing a profile_id.")
    return str(profile_id)
    pass


@pending_datapoints.command(name="list")
def list_pending_datapoints():
    """List pending DataPoints awaiting approval."""
    try:
        pending = _fetch_pending_datapoints()
        if not pending:
            console.print("[yellow]No pending DataPoints found[/yellow]")
            return

        table = Table(title="Pending DataPoints", show_header=True, header_style="bold cyan")
        table.add_column("Pending ID")
        table.add_column("Type")
        table.add_column("Name")
        table.add_column("Confidence", justify="right")
        table.add_column("Status")
        for item in pending:
            datapoint = item.get("datapoint", {}) if isinstance(item, dict) else {}
            table.add_row(
                str(item.get("pending_id", "")),
                str(datapoint.get("type", "")),
                str(datapoint.get("name") or datapoint.get("datapoint_id") or ""),
                f"{item.get('confidence', 0):.2f}" if item.get("confidence") is not None else "-",
                str(item.get("status", "")),
            )
        console.print(table)
    except Exception as exc:
        console.print(f"[red]Failed to list pending DataPoints: {exc}[/red]")
        sys.exit(1)


@pending_datapoints.command(name="approve")
@click.argument("pending_id")
@click.option("--note", help="Optional review note.")
def approve_pending_datapoint(pending_id: str, note: str | None):
    """Approve a pending DataPoint."""
    payload = {"review_note": note} if note else None
    try:
        response = httpx.post(
            f"{API_BASE_URL}/api/v1/datapoints/pending/{pending_id}/approve",
            json=payload,
            timeout=15.0,
        )
        response.raise_for_status()
        console.print("[green]✓ Approved DataPoint[/green]")
    except Exception as exc:
        console.print(f"[red]Failed to approve DataPoint: {exc}[/red]")
        sys.exit(1)


@pending_datapoints.command(name="reject")
@click.argument("pending_id")
@click.option("--note", help="Optional review note.")
def reject_pending_datapoint(pending_id: str, note: str | None):
    """Reject a pending DataPoint."""
    payload = {"review_note": note} if note else None
    try:
        response = httpx.post(
            f"{API_BASE_URL}/api/v1/datapoints/pending/{pending_id}/reject",
            json=payload,
            timeout=15.0,
        )
        response.raise_for_status()
        console.print("[green]✓ Rejected DataPoint[/green]")
    except Exception as exc:
        console.print(f"[red]Failed to reject DataPoint: {exc}[/red]")
        sys.exit(1)


@pending_datapoints.command(name="approve-all")
@click.option("--profile-id", help="Approve pending items for a specific profile.")
@click.option(
    "--latest",
    is_flag=True,
    help="Approve pending items for the latest profile on the default connection.",
)
def approve_all_pending(profile_id: str | None, latest: bool):
    """Bulk-approve pending DataPoints."""
    if profile_id and latest:
        raise click.ClickException("Use either --profile-id or --latest, not both.")
    try:
        if latest and not profile_id:
            connection_id = _resolve_default_connection_id()
            profile_id = _resolve_latest_profile_id(connection_id)

        if profile_id:
            pending = [
                item
                for item in _fetch_pending_datapoints()
                if str(item.get("profile_id")) == profile_id
            ]
            if not pending:
                console.print(
                    "[yellow]No pending DataPoints found for that profile.[/yellow]"
                )
                return
            approved = 0
            for item in pending:
                pending_id = item.get("pending_id")
                if not pending_id:
                    continue
                response = httpx.post(
                    f"{API_BASE_URL}/api/v1/datapoints/pending/{pending_id}/approve",
                    timeout=15.0,
                )
                response.raise_for_status()
                approved += 1
            console.print(f"[green]✓ Approved {approved} DataPoints[/green]")
        else:
            response = httpx.post(
                f"{API_BASE_URL}/api/v1/datapoints/pending/bulk-approve", timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            pending = data.get("pending", [])
            console.print(f"[green]✓ Approved {len(pending)} DataPoints[/green]")
    except Exception as exc:
        console.print(f"[red]Failed to bulk-approve DataPoints: {exc}[/red]")
        sys.exit(1)


@datapoint.command(name="generate")
@click.option("--profile-id", help="Profiling profile UUID.")
@click.option("--connection-id", help="Connection UUID for latest profiling job lookup.")
@click.option(
    "--depth",
    type=click.Choice(["schema_only", "metrics_basic", "metrics_full"]),
    default="metrics_basic",
    show_default=True,
)
@click.option("--tables", multiple=True, help="Optional table names to include.")
@click.option("--batch-size", default=10, show_default=True, type=int)
@click.option("--max-tables", default=None, type=int)
@click.option("--max-metrics-per-table", default=3, show_default=True, type=int)
def generate_datapoints_cli(
    profile_id: str | None,
    connection_id: str | None,
    depth: str,
    tables: tuple[str, ...],
    batch_size: int,
    max_tables: int | None,
    max_metrics_per_table: int,
):
    """Start DataPoint generation for a profiling profile."""
    if max_tables is not None and max_tables <= 0:
        max_tables = None
    if not profile_id:
        try:
            resolved_connection_id = connection_id or _resolve_default_connection_id()
            profile_id = _resolve_latest_profile_id(resolved_connection_id)
            console.print(
                f"[dim]Using latest profile {profile_id} for connection {resolved_connection_id}.[/dim]"
            )
        except click.ClickException as exc:
            console.print(f"[red]{exc}[/red]")
            sys.exit(1)
    payload = {
        "profile_id": profile_id,
        "tables": list(tables) or None,
        "depth": depth,
        "batch_size": batch_size,
        "max_tables": max_tables,
        "max_metrics_per_table": max_metrics_per_table,
    }
    try:
        response = httpx.post(
            f"{API_BASE_URL}/api/v1/datapoints/generate",
            json=payload,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        console.print(
            f"[green]✓ Generation started[/green] job_id={data['job_id']}"
        )
    except Exception as exc:
        console.print(f"[red]Failed to start generation: {exc}[/red]")
        sys.exit(1)


@datapoint.command(name="generate-status")
@click.argument("job_id")
def generation_status(job_id: str):
    """Check DataPoint generation job status."""
    try:
        response = httpx.get(
            f"{API_BASE_URL}/api/v1/datapoints/generate/jobs/{job_id}",
            timeout=15.0,
        )
        response.raise_for_status()
        data = response.json()
        console.print(json.dumps(data, indent=2))
    except Exception as exc:
        console.print(f"[red]Failed to fetch generation status: {exc}[/red]")
        sys.exit(1)


@datapoint.command(name="add")
@click.argument("datapoint_type", type=click.Choice(["schema", "business", "process"]))
@click.argument("file", type=click.Path(exists=True))
def add_datapoint(datapoint_type: str, file: str):
    """Add a DataPoint from a JSON file.

    DATAPOINT_TYPE: schema, business, or process
    FILE: Path to JSON file
    """

    async def run_add():
        try:
            # Load DataPoint
            console.print(f"[cyan]Loading DataPoint from {file}...[/cyan]")
            loader = DataPointLoader()
            datapoint = loader.load_file(Path(file))

            # Validate type matches
            if datapoint.type.lower() != datapoint_type.lower():
                console.print(
                    f"[red]Error: DataPoint type '{datapoint.type}' "
                    f"doesn't match specified type '{datapoint_type}'[/red]"
                )
                sys.exit(1)

            # Add to vector store
            console.print("[cyan]Adding to vector store...[/cyan]")
            vector_store = VectorStore()
            await vector_store.initialize()
            await vector_store.add_datapoints([datapoint])

            # Add to knowledge graph
            console.print("[cyan]Adding to knowledge graph...[/cyan]")
            graph = KnowledgeGraph()
            graph.add_datapoint(datapoint)

            console.print(f"[green]✓ DataPoint '{datapoint.name}' added successfully[/green]")
            console.print(f"ID: {datapoint.datapoint_id}")
            console.print(f"Type: {datapoint.type}")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    asyncio.run(run_add())


@datapoint.command(name="sync")
@click.option(
    "--datapoints-dir",
    default="datapoints",
    help="Directory containing DataPoint JSON files",
)
def sync_datapoints(datapoints_dir: str):
    """Rebuild vector store and knowledge graph from DataPoints directory."""

    async def run_sync():
        try:
            datapoints_path = Path(datapoints_dir)
            if not datapoints_path.exists():
                console.print(f"[red]Directory not found: {datapoints_dir}[/red]")
                sys.exit(1)

            # Load all DataPoints
            console.print(f"[cyan]Loading DataPoints from {datapoints_dir}...[/cyan]")
            loader = DataPointLoader()
            datapoints = loader.load_directory(datapoints_path)
            stats = loader.get_stats()

            if stats["failed_count"] > 0:
                console.print(
                    f"[yellow]⚠ {stats['failed_count']} DataPoints failed to load[/yellow]"
                )
                for error in stats["failed_files"]:
                    console.print(
                        f"  [red]• {error['path']}: {error['error']}[/red]"
                    )

            if not datapoints:
                console.print("[yellow]No valid DataPoints found[/yellow]")
                return

            console.print(f"[green]✓ Loaded {len(datapoints)} DataPoints[/green]")

            # Rebuild vector store
            console.print("\n[cyan]Rebuilding vector store...[/cyan]")
            vector_store = VectorStore()
            await vector_store.initialize()

            # Clear existing
            await vector_store.clear()

            # Add all datapoints with progress
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            )

            with progress:
                task = progress.add_task("Adding to vector store...", total=len(datapoints))
                await vector_store.add_datapoints(datapoints)
                progress.update(task, completed=len(datapoints))

            # Rebuild knowledge graph
            console.print("[cyan]Rebuilding knowledge graph...[/cyan]")
            graph = KnowledgeGraph()

            with progress:
                task = progress.add_task(
                    "Adding to knowledge graph...", total=len(datapoints)
                )
                for datapoint in datapoints:
                    graph.add_datapoint(datapoint)
                    progress.update(task, advance=1)

            # Display summary
            console.print()
            summary = Table(show_header=False, box=None)
            summary.add_row("[green]✓ Sync complete[/green]")
            summary.add_row("DataPoints loaded:", f"[cyan]{len(datapoints)}[/cyan]")
            summary.add_row("Vector store:", f"[cyan]{await vector_store.get_count()}[/cyan]")

            stats = graph.get_stats()
            summary.add_row(
                "Knowledge graph:",
                f"[cyan]{stats['total_nodes']} nodes, {stats['total_edges']} edges[/cyan]",
            )

            console.print(summary)

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback

            console.print(f"[red]{traceback.format_exc()}[/red]")
            sys.exit(1)

    asyncio.run(run_sync())


# ============================================================================
# Entry Point
# ============================================================================


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
