import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ChatInterface } from "@/components/chat/ChatInterface";
import { useChatStore } from "@/lib/stores/chat";

const { mockSystemStatus, mockListDatabases, mockStreamChat } = vi.hoisted(() => ({
  mockSystemStatus: vi.fn(),
  mockListDatabases: vi.fn(),
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
});
