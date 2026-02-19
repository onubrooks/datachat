import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ChatInterface } from "@/components/chat/ChatInterface";
import { useChatStore } from "@/lib/stores/chat";

const { mockSystemStatus, mockListDatabases, mockGetDatabaseSchema, mockStreamChat } = vi.hoisted(() => ({
  mockSystemStatus: vi.fn(),
  mockListDatabases: vi.fn(),
  mockGetDatabaseSchema: vi.fn(),
  mockStreamChat: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    apiClient: {
      ...actual.apiClient,
      systemStatus: mockSystemStatus,
      listDatabases: mockListDatabases,
      getDatabaseSchema: mockGetDatabaseSchema,
    },
    wsClient: {
      ...actual.wsClient,
      streamChat: mockStreamChat,
    },
  };
});

describe("ChatInterface target database", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    Element.prototype.scrollIntoView = vi.fn();

    useChatStore.getState().clearMessages();
    useChatStore.getState().setLoading(false);
    useChatStore.getState().setConnected(false);
    useChatStore.getState().resetAgentStatus();

    mockSystemStatus.mockResolvedValue({
      is_initialized: true,
      has_databases: true,
      has_system_database: true,
      has_datapoints: false,
      setup_required: [],
    });

    mockListDatabases.mockResolvedValue([
      {
        connection_id: "db_pg",
        name: "Postgres Main",
        database_url: "postgresql://postgres@localhost:5432/app",
        database_type: "postgresql",
        is_active: true,
        is_default: false,
        tags: [],
        created_at: new Date().toISOString(),
        datapoint_count: 0,
      },
      {
        connection_id: "db_mysql",
        name: "MySQL Demo",
        database_url: "mysql://root:root@localhost:3306/demo",
        database_type: "mysql",
        is_active: true,
        is_default: true,
        tags: [],
        created_at: new Date().toISOString(),
        datapoint_count: 0,
      },
    ]);
    mockGetDatabaseSchema.mockResolvedValue({
      connection_id: "db_mysql",
      database_type: "mysql",
      fetched_at: new Date().toISOString(),
      tables: [],
    });
  });

  it("sends the default selected connection as target_database", async () => {
    render(<ChatInterface />);

    await waitFor(() => expect(mockListDatabases).toHaveBeenCalledTimes(1));

    const input = screen.getByPlaceholderText("Ask a question about your data...");
    fireEvent.change(input, { target: { value: "list all available tables" } });
    fireEvent.keyDown(input, { key: "Enter", code: "Enter" });

    await waitFor(() => expect(mockStreamChat).toHaveBeenCalledTimes(1));
    const request = mockStreamChat.mock.calls[0][0] as Record<string, unknown>;
    expect(request.target_database).toBe("db_mysql");
  });

  it("uses the user-selected connection id for chat", async () => {
    render(<ChatInterface />);

    await waitFor(() => expect(mockListDatabases).toHaveBeenCalledTimes(1));

    const select = await screen.findByLabelText("Target database");
    fireEvent.change(select, { target: { value: "db_pg" } });

    const input = screen.getByPlaceholderText("Ask a question about your data...");
    fireEvent.change(input, { target: { value: "show columns in customers" } });
    fireEvent.keyDown(input, { key: "Enter", code: "Enter" });

    await waitFor(() => expect(mockStreamChat).toHaveBeenCalledTimes(1));
    const request = mockStreamChat.mock.calls[0][0] as Record<string, unknown>;
    expect(request.target_database).toBe("db_pg");
  });

  it("restores input focus after response completes", async () => {
    let handlers:
      | {
          onComplete?: (response: { answer: string }) => void;
        }
      | undefined;

    mockStreamChat.mockImplementation((_request, callbacks) => {
      handlers = callbacks;
    });

    render(<ChatInterface />);

    await waitFor(() => expect(mockListDatabases).toHaveBeenCalledTimes(1));

    const input = screen.getByPlaceholderText(
      "Ask a question about your data..."
    ) as HTMLInputElement;

    fireEvent.change(input, { target: { value: "list all available tables" } });
    fireEvent.keyDown(input, { key: "Enter", code: "Enter" });

    await waitFor(() => expect(mockStreamChat).toHaveBeenCalledTimes(1));
    expect(handlers).toBeDefined();

    input.blur();
    expect(document.activeElement).not.toBe(input);

    await act(async () => {
      handlers?.onComplete?.({ answer: "Found tables." });
    });

    await waitFor(() => {
      expect(document.activeElement).toBe(input);
    });
  });

  it("applies quick query templates into the input", async () => {
    render(<ChatInterface />);
    await waitFor(() => expect(mockListDatabases).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(mockGetDatabaseSchema).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: "List Tables" }));
    expect(
      screen.getByDisplayValue("List all available tables.")
    ).toBeInTheDocument();
  });

  it("opens and closes keyboard shortcuts modal from keyboard shortcut", async () => {
    render(<ChatInterface />);
    await waitFor(() => expect(mockListDatabases).toHaveBeenCalledTimes(1));

    fireEvent.keyDown(window, { key: "/", ctrlKey: true });
    expect(
      screen.getByRole("dialog", { name: "Keyboard shortcuts" })
    ).toBeInTheDocument();

    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Keyboard shortcuts" })
      ).not.toBeInTheDocument()
    );
  });

  it("filters conversation history from sidebar search", async () => {
    window.localStorage.setItem(
      "datachat.conversation.history.v1",
      JSON.stringify([
        {
          frontendSessionId: "session-a",
          title: "Sales trend review",
          targetDatabaseId: "db_mysql",
          conversationId: "conv_a",
          sessionSummary: null,
          sessionState: null,
          updatedAt: new Date().toISOString(),
          messages: [],
        },
        {
          frontendSessionId: "session-b",
          title: "Inventory checks",
          targetDatabaseId: "db_pg",
          conversationId: "conv_b",
          sessionSummary: null,
          sessionState: null,
          updatedAt: new Date().toISOString(),
          messages: [],
        },
      ])
    );

    render(<ChatInterface />);
    await waitFor(() => expect(mockListDatabases).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(mockGetDatabaseSchema).toHaveBeenCalled());

    expect(screen.getByText("Sales trend review")).toBeInTheDocument();
    expect(screen.getByText("Inventory checks")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Search saved conversations"), {
      target: { value: "sales" },
    });

    expect(screen.getByText("Sales trend review")).toBeInTheDocument();
    expect(screen.queryByText("Inventory checks")).not.toBeInTheDocument();
  });

  it("sends SQL editor content as deterministic execution prompt", async () => {
    render(<ChatInterface />);
    await waitFor(() => expect(mockListDatabases).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole("button", { name: /SQL Editor/i }));
    fireEvent.change(screen.getByLabelText("SQL editor input"), {
      target: { value: "SELECT * FROM users LIMIT 10;" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Run SQL draft" }));

    await waitFor(() => expect(mockStreamChat).toHaveBeenCalledTimes(1));
    const request = mockStreamChat.mock.calls[0][0] as Record<string, unknown>;
    expect(String(request.message)).toContain("Execute this SQL query exactly as written");
    expect(String(request.message)).toContain("SELECT * FROM users LIMIT 10;");
  });
});
