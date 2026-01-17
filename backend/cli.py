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
    datachat status                        # Show connection status
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from backend.config import get_settings
from backend.connectors.postgres import PostgresConnector
from backend.knowledge.datapoints import DataPointLoader
from backend.knowledge.graph import KnowledgeGraph
from backend.knowledge.retriever import Retriever
from backend.knowledge.vectors import VectorStore
from backend.pipeline.orchestrator import DataChatPipeline

console = Console()


# ============================================================================
# CLI State Management
# ============================================================================


class CLIState:
    """Manage CLI state (connection, pipeline, etc.)."""

    def __init__(self):
        self.config_dir = Path.home() / ".datachat"
        self.config_file = self.config_dir / "config.json"
        self.config_dir.mkdir(exist_ok=True, mode=0o700)

    def load_config(self) -> dict[str, Any]:
        """Load CLI configuration."""
        if self.config_file.exists():
            with open(self.config_file) as f:
                return json.load(f)
        return {}

    def save_config(self, config: dict[str, Any]) -> None:
        """Save CLI configuration."""
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)
        try:
            self.config_file.chmod(0o600)
        except OSError:
            pass

    def get_connection_string(self) -> str | None:
        """Get stored connection string."""
        config = self.load_config()
        return config.get("connection_string")

    def set_connection_string(self, connection_string: str) -> None:
        """Set connection string."""
        config = self.load_config()
        config["connection_string"] = connection_string
        self.save_config(config)


state = CLIState()


# ============================================================================
# Helper Functions
# ============================================================================


async def create_pipeline_from_config() -> DataChatPipeline:
    """Create pipeline from configuration."""
    settings = get_settings()

    # Initialize vector store
    console.print("[cyan]Initializing vector store...[/cyan]")
    vector_store = VectorStore()
    await vector_store.initialize()

    # Initialize knowledge graph
    console.print("[cyan]Initializing knowledge graph...[/cyan]")
    knowledge_graph = KnowledgeGraph()

    # Initialize retriever
    retriever = Retriever(
        vector_store=vector_store,
        knowledge_graph=knowledge_graph,
    )

    # Initialize connector
    connection_string = state.get_connection_string()
    if not connection_string:
        connection_string = str(settings.database.url)

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


# ============================================================================
# CLI Commands
# ============================================================================


@click.group()
@click.version_option(version="0.1.0", prog_name="DataChat")
def cli():
    """DataChat - Natural language interface for data warehouses."""
    pass


@cli.command()
def chat():
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

        try:
            pipeline = await create_pipeline_from_config()
            console.print("[green]✓ Pipeline initialized[/green]\n")

            while True:
                try:
                    # Get user input
                    query = console.input("[bold cyan]You:[/bold cyan] ")

                    if not query.strip():
                        continue

                    if query.lower() in ["exit", "quit", "q"]:
                        console.print("\n[yellow]Goodbye![/yellow]")
                        break

                    # Show loading indicator
                    with console.status("[cyan]Processing...[/cyan]", spinner="dots"):
                        result = await pipeline.run(
                            query=query,
                            conversation_history=conversation_history,
                        )

                    # Extract results
                    answer = result.get("natural_language_answer", "No answer generated")
                    sql = result.get("validated_sql") or result.get("generated_sql")
                    query_result = result.get("query_result")
                    data = _build_columnar_data(query_result)

                    # Display results
                    console.print()
                    format_answer(answer, sql, data)

                    # Display metrics
                    metrics = Table(show_header=False, box=None)
                    metrics.add_row(
                        "⏱️  Latency:",
                        f"{result.get('total_latency_ms', 0):.0f}ms",
                    )
                    metrics.add_row("🤖 LLM Calls:", str(result.get("llm_calls", 0)))
                    metrics.add_row("🔄 Retries:", str(result.get("retry_count", 0)))
                    console.print(metrics)
                    console.print()

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
            if "pipeline" in locals():
                try:
                    await pipeline.connector.close()
                except Exception:
                    pass

    asyncio.run(run_chat())


@cli.command()
@click.argument("query")
def ask(query: str):
    """Ask a single question and exit."""

    async def run_query():
        try:
            pipeline = await create_pipeline_from_config()

            # Show loading with progress
            with console.status("[cyan]Processing query...[/cyan]", spinner="dots"):
                result = await pipeline.run(query=query)

            # Extract results
            answer = result.get("natural_language_answer", "No answer generated")
            sql = result.get("validated_sql") or result.get("generated_sql")
            query_result = result.get("query_result")
            data = _build_columnar_data(query_result)

            # Display results
            console.print()
            format_answer(answer, sql, data)
            console.print()

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
        table = Table(title="DataChat Status", show_header=True, header_style="bold cyan")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")

        # Check configuration
        settings = get_settings()
        table.add_row("Configuration", "✓", f"Environment: {settings.environment}")

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
        else:
            table.add_row("Connection", "⚠️", "Using default from config")

        # Check database connection
        try:
            connection_string = conn_str or str(settings.database.url)
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


# ============================================================================
# DataPoint Commands
# ============================================================================


@cli.group(name="dp")
def datapoint():
    """Manage DataPoints (knowledge base)."""
    pass


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
            result = loader.load_directory(datapoints_path)

            if result.failed > 0:
                console.print(f"[yellow]⚠ {result.failed} DataPoints failed to load[/yellow]")
                for error in result.errors:
                    console.print(f"  [red]• {error}[/red]")

            if not result.datapoints:
                console.print("[yellow]No valid DataPoints found[/yellow]")
                return

            console.print(f"[green]✓ Loaded {len(result.datapoints)} DataPoints[/green]")

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
                task = progress.add_task("Adding to vector store...", total=len(result.datapoints))
                await vector_store.add_datapoints(result.datapoints)
                progress.update(task, completed=len(result.datapoints))

            # Rebuild knowledge graph
            console.print("[cyan]Rebuilding knowledge graph...[/cyan]")
            graph = KnowledgeGraph()

            with progress:
                task = progress.add_task(
                    "Adding to knowledge graph...", total=len(result.datapoints)
                )
                for datapoint in result.datapoints:
                    graph.add_datapoint(datapoint)
                    progress.update(task, advance=1)

            # Display summary
            console.print()
            summary = Table(show_header=False, box=None)
            summary.add_row("[green]✓ Sync complete[/green]")
            summary.add_row("DataPoints loaded:", f"[cyan]{len(result.datapoints)}[/cyan]")
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
