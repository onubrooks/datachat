"""
Unit Tests for CLI

Tests the DataChat CLI commands.
"""

import json
from unittest.mock import AsyncMock, patch

import click
import pytest
from click.testing import CliRunner

from backend.cli import (
    ask,
    cli,
    connect,
    create_pipeline_from_config,
    datapoint,
    setup,
    status,
)
from backend.initialization.initializer import SystemStatus


class TestCLIBasics:
    """Test basic CLI functionality."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_cli_help(self, runner):
        """Test that CLI shows help text."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "DataChat" in result.output
        assert "Natural language interface" in result.output

    def test_cli_version(self, runner):
        """Test that CLI shows version."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_chat_command_exists(self, runner):
        """Test that chat command exists."""
        result = runner.invoke(cli, ["chat", "--help"])
        assert result.exit_code == 0
        assert "Interactive REPL mode" in result.output

    def test_ask_command_exists(self, runner):
        """Test that ask command exists."""
        result = runner.invoke(cli, ["ask", "--help"])
        assert result.exit_code == 0
        assert "Ask a single question" in result.output

    def test_connect_command_exists(self, runner):
        """Test that connect command exists."""
        result = runner.invoke(cli, ["connect", "--help"])
        assert result.exit_code == 0
        assert "Set database connection" in result.output

    def test_status_command_exists(self, runner):
        """Test that status command exists."""
        result = runner.invoke(cli, ["status", "--help"])
        assert result.exit_code == 0
        assert "Show connection and system status" in result.output

    def test_setup_command_exists(self, runner):
        """Test that setup command exists."""
        result = runner.invoke(cli, ["setup", "--help"])
        assert result.exit_code == 0
        assert "Guide system initialization" in result.output

    def test_datapoint_group_exists(self, runner):
        """Test that datapoint command group exists."""
        result = runner.invoke(cli, ["dp", "--help"])
        assert result.exit_code == 0
        assert "Manage DataPoints" in result.output


class TestConnectCommand:
    """Test connect command."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create temporary config directory."""
        return tmp_path / ".datachat"

    def test_connect_saves_connection_string(self, runner, temp_config_dir):
        """Test that connect command saves connection string."""
        # Patch the state object's config_dir and config_file directly
        from backend.cli import state

        with (
            patch.object(state, "config_dir", temp_config_dir),
            patch.object(state, "config_file", temp_config_dir / "config.json"),
        ):
            # Ensure directory exists
            temp_config_dir.mkdir(parents=True, exist_ok=True)

            result = runner.invoke(
                connect,
                ["postgresql://user:pass@localhost:5432/testdb"],
            )

            assert result.exit_code == 0
            assert "Connection string saved" in result.output
            assert "localhost" in result.output
            assert "5432" in result.output
            assert "testdb" in result.output
            assert "user" in result.output

            # Verify config file was created
            config_file = temp_config_dir / "config.json"
            assert config_file.exists()

            with open(config_file) as f:
                config = json.load(f)
                assert config["connection_string"] == "postgresql://user:pass@localhost:5432/testdb"

    def test_connect_validates_connection_string(self, runner):
        """Test that connect command validates connection string format."""
        result = runner.invoke(connect, ["invalid_connection_string"])

        assert result.exit_code == 1
        assert "Invalid connection string format" in result.output

    def test_connect_handles_missing_port(self, runner, temp_config_dir):
        """Test that connect command handles missing port."""
        with patch("pathlib.Path.home", return_value=temp_config_dir.parent):
            result = runner.invoke(
                connect,
                ["postgresql://user:pass@localhost/testdb"],
            )

            assert result.exit_code == 0
            assert "5432" in result.output  # Default port


class TestAskCommand:
    """Test ask command."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_pipeline(self):
        """Create mock pipeline."""
        pipeline = AsyncMock()
        pipeline.run = AsyncMock(
            return_value={
                "natural_language_answer": "The total revenue is $1,234,567.89",
                "validated_sql": "SELECT SUM(amount) FROM sales",
                "query_result": {"data": {"total": [1234567.89]}},
                "total_latency_ms": 1000.0,
                "llm_calls": 2,
                "retry_count": 0,
            }
        )
        pipeline.connector = AsyncMock()
        pipeline.connector.close = AsyncMock()
        return pipeline

    def test_ask_requires_query_argument(self, runner):
        """Test that ask command requires query argument."""
        result = runner.invoke(ask, [])
        assert result.exit_code != 0

    def test_ask_command_help(self, runner):
        """Test ask command help text."""
        result = runner.invoke(ask, ["--help"])
        assert result.exit_code == 0
        assert "Ask a single question" in result.output


class TestDataPointCommands:
    """Test DataPoint management commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_dp_list_command_exists(self, runner):
        """Test that dp list command exists."""
        result = runner.invoke(datapoint, ["list", "--help"])
        assert result.exit_code == 0
        assert "List all DataPoints" in result.output

    def test_dp_add_command_exists(self, runner):
        """Test that dp add command exists."""
        result = runner.invoke(datapoint, ["add", "--help"])
        assert result.exit_code == 0
        assert "Add a DataPoint from a JSON file" in result.output

    def test_dp_add_validates_type(self, runner, tmp_path):
        """Test that dp add validates DataPoint type."""
        # Create temporary JSON file
        test_file = tmp_path / "test.json"
        test_file.write_text(json.dumps({"type": "Schema", "name": "Test"}))

        result = runner.invoke(datapoint, ["add", "invalid_type", str(test_file)])
        assert result.exit_code != 0

    def test_dp_add_accepts_valid_types(self, runner):
        """Test that dp add accepts valid DataPoint types."""
        for dp_type in ["schema", "business", "process"]:
            result = runner.invoke(datapoint, ["add", dp_type, "--help"])
            assert result.exit_code == 0

    def test_dp_sync_command_exists(self, runner):
        """Test that dp sync command exists."""
        result = runner.invoke(datapoint, ["sync", "--help"])
        assert result.exit_code == 0
        assert "Rebuild vector store and knowledge graph" in result.output

    def test_dp_sync_has_datapoints_dir_option(self, runner):
        """Test that dp sync has --datapoints-dir option."""
        result = runner.invoke(datapoint, ["sync", "--help"])
        assert result.exit_code == 0
        assert "--datapoints-dir" in result.output


class TestStatusCommand:
    """Test status command."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_status_command_help(self, runner):
        """Test status command help text."""
        result = runner.invoke(status, ["--help"])
        assert result.exit_code == 0
        assert "Show connection and system status" in result.output


class TestCLIState:
    """Test CLI state management."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create temporary config directory."""
        return tmp_path / ".datachat"

    def test_cli_state_creates_config_dir(self, temp_config_dir):
        """Test that CLI state creates config directory."""
        from backend.cli import CLIState

        with patch("pathlib.Path.home", return_value=temp_config_dir.parent):
            _ = CLIState()  # State creation triggers directory creation
            assert temp_config_dir.exists()

    def test_cli_state_loads_empty_config(self, temp_config_dir):
        """Test that CLI state loads empty config when file doesn't exist."""
        from backend.cli import CLIState

        with patch("pathlib.Path.home", return_value=temp_config_dir.parent):
            state = CLIState()
            config = state.load_config()
            assert config == {}

    def test_cli_state_saves_and_loads_config(self, temp_config_dir):
        """Test that CLI state saves and loads configuration."""
        from backend.cli import CLIState

        with patch("pathlib.Path.home", return_value=temp_config_dir.parent):
            state = CLIState()

            # Save config
            test_config = {"key": "value", "number": 42}
            state.save_config(test_config)

            # Load config
            loaded_config = state.load_config()
            assert loaded_config == test_config

    def test_cli_state_get_connection_string(self, temp_config_dir):
        """Test getting connection string from state."""
        from backend.cli import CLIState

        with patch("pathlib.Path.home", return_value=temp_config_dir.parent):
            state = CLIState()

            # No connection string initially
            assert state.get_connection_string() is None

            # Set connection string
            conn_str = "postgresql://localhost/test"
            state.set_connection_string(conn_str)

            # Get connection string
            assert state.get_connection_string() == conn_str

    def test_cli_state_set_connection_string(self, temp_config_dir):
        """Test setting connection string in state."""
        from backend.cli import CLIState

        with patch("pathlib.Path.home", return_value=temp_config_dir.parent):
            state = CLIState()

            conn_str = "postgresql://user:pass@host:5432/db"
            state.set_connection_string(conn_str)

            # Verify it was saved
            config = state.load_config()
            assert config["connection_string"] == conn_str


class TestCLIErrorHandling:
    """Test CLI error handling."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_connect_handles_invalid_format(self, runner):
        """Test connect handles invalid connection string format."""
        result = runner.invoke(connect, ["not_a_valid_url"])
        assert result.exit_code == 1
        assert "Invalid connection string format" in result.output

    def test_dp_add_handles_missing_file(self, runner):
        """Test dp add handles missing file."""
        result = runner.invoke(datapoint, ["add", "schema", "nonexistent.json"])
        assert result.exit_code != 0

    def test_dp_add_validates_file_exists(self, runner, tmp_path):
        """Test that dp add validates file exists."""
        # Use Click's built-in path validation
        non_existent = tmp_path / "does_not_exist.json"
        result = runner.invoke(datapoint, ["add", "schema", str(non_existent)])
        assert result.exit_code != 0

    @pytest.mark.asyncio
    async def test_cli_allows_live_schema_when_no_datapoints(self):
        """Ensure CLI allows live schema queries when DataPoints are missing."""
        not_initialized = SystemStatus(
            is_initialized=False,
            has_databases=True,
            has_system_database=True,
            has_datapoints=False,
            setup_required=[],
        )
        with patch(
            "backend.cli.SystemInitializer.status",
            new=AsyncMock(return_value=not_initialized),
        ):
            with patch("backend.cli.VectorStore.initialize", new=AsyncMock()):
                with patch("backend.cli.PostgresConnector.connect", new=AsyncMock()):
                    with patch("backend.cli.DataChatPipeline") as pipeline_cls:
                        pipeline = await create_pipeline_from_config()
                        assert pipeline is pipeline_cls.return_value


class TestCLISetup:
    """Test setup command."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_setup_command_runs(self, runner):
        with (
            patch("backend.cli.VectorStore.initialize", new=AsyncMock()),
            patch(
                "backend.cli.SystemInitializer.initialize",
                new=AsyncMock(
                    return_value=(
                        SystemStatus(
                            is_initialized=True,
                            has_databases=True,
                            has_system_database=True,
                            has_datapoints=True,
                            setup_required=[],
                        ),
                        "Initialization completed.",
                    )
                ),
            ),
            patch("click.prompt", return_value="postgresql://user@localhost/db"),
            patch("click.confirm", return_value=False),
        ):
            result = runner.invoke(setup)
            assert result.exit_code == 0


class TestCLIIntegration:
    """Integration-style tests for CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    def test_full_cli_workflow_help_messages(self, runner):
        """Test that all commands have proper help messages."""
        commands = [
            ["--help"],
            ["chat", "--help"],
            ["ask", "--help"],
            ["connect", "--help"],
            ["setup", "--help"],
            ["status", "--help"],
            ["dp", "--help"],
            ["dp", "list", "--help"],
            ["dp", "add", "--help"],
            ["dp", "sync", "--help"],
        ]

        for cmd in commands:
            result = runner.invoke(cli, cmd)
            assert result.exit_code == 0, f"Command {cmd} failed"
            assert len(result.output) > 0, f"Command {cmd} has no output"

    def test_command_structure(self, runner):
        """Test that CLI has expected command structure."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

        # Check main commands exist
        assert "chat" in result.output
        assert "ask" in result.output
        assert "connect" in result.output
        assert "setup" in result.output
        assert "status" in result.output
        assert "dp" in result.output

    def test_datapoint_subcommands(self, runner):
        """Test that dp group has expected subcommands."""
        result = runner.invoke(datapoint, ["--help"])
        assert result.exit_code == 0

        # Check subcommands exist
        assert "list" in result.output
        assert "add" in result.output
        assert "sync" in result.output
