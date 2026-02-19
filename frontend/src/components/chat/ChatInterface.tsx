/**
 * Chat Interface Component
 *
 * Main chat UI with:
 * - Message list with auto-scroll
 * - Message input with send button
 * - Agent status display during processing
 * - WebSocket integration for real-time updates
 * - Error handling and loading states
 */

"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Send,
  Trash2,
  AlertCircle,
  Loader2,
  RefreshCw,
  Wifi,
  Clock,
  Database,
  AlertTriangle,
  PanelLeftOpen,
  PanelLeftClose,
  PanelRightOpen,
  PanelRightClose,
  ChevronDown,
  ChevronRight,
  History,
  Plus,
  Table2,
} from "lucide-react";
import { Message } from "./Message";
import { AgentStatus } from "../agents/AgentStatus";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Card } from "../ui/card";
import { useChatStore, type Message as ChatStoreMessage } from "@/lib/stores/chat";
import {
  apiClient,
  wsClient,
  type DatabaseConnection,
  type DatabaseSchemaTable,
  type SetupStep,
} from "@/lib/api";
import { SystemSetup } from "../system/SystemSetup";
import {
  getResultLayoutMode,
  getShowLiveReasoning,
  getShowAgentTimingBreakdown,
  getSynthesizeSimpleSql,
  getWaitingUxMode,
  type ResultLayoutMode,
  type WaitingUxMode,
} from "@/lib/settings";
import { formatWaitingChipLabel } from "./loadingUx";

const ACTIVE_DATABASE_STORAGE_KEY = "datachat.active_connection_id";
const CONVERSATION_HISTORY_STORAGE_KEY = "datachat.conversation.history.v1";
const MAX_CONVERSATION_HISTORY = 20;
const MAX_CONVERSATION_MESSAGES = 80;

type SerializedMessage = Omit<ChatStoreMessage, "timestamp"> & {
  timestamp: string;
};

interface ConversationSnapshot {
  frontendSessionId: string;
  title: string;
  targetDatabaseId: string | null;
  conversationId: string | null;
  sessionSummary: string | null;
  sessionState: Record<string, unknown> | null;
  updatedAt: string;
  messages: SerializedMessage[];
}

const QUERY_TEMPLATES: Array<{ id: string; label: string; build: (selectedTable?: string | null) => string }> = [
  {
    id: "list-tables",
    label: "List Tables",
    build: () => "List all available tables.",
  },
  {
    id: "show-columns",
    label: "Show Columns",
    build: (selectedTable) =>
      selectedTable
        ? `Show columns for ${selectedTable}.`
        : "Show columns for the table grocery_sales_transactions.",
  },
  {
    id: "sample-rows",
    label: "Sample 100 Rows",
    build: (selectedTable) =>
      selectedTable
        ? `Show first 100 rows from ${selectedTable}.`
        : "Show first 100 rows from grocery_sales_transactions.",
  },
  {
    id: "row-count",
    label: "Count Rows",
    build: (selectedTable) =>
      selectedTable
        ? `How many rows are in ${selectedTable}?`
        : "How many rows are in each table?",
  },
];

export function ChatInterface() {
  const router = useRouter();
  const {
    messages,
    conversationId,
    frontendSessionId,
    sessionSummary,
    sessionState,
    isLoading,
    isConnected,
    agentHistory,
    agentStatus,
    setLoading,
    setConnected,
    setAgentUpdate,
    resetAgentStatus,
    clearMessages,
    addMessage,
    updateLastMessage,
    setConversationId,
    setSessionMemory,
    loadSession,
    appendToLastMessage,
  } = useChatStore();

  const [input, setInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [errorCategory, setErrorCategory] = useState<"network" | "timeout" | "validation" | "database" | "unknown" | null>(null);
  const [lastFailedQuery, setLastFailedQuery] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const [setupSteps, setSetupSteps] = useState<SetupStep[]>([]);
  const [isInitialized, setIsInitialized] = useState(true);
  const [setupError, setSetupError] = useState<string | null>(null);
  const [setupNotice, setSetupNotice] = useState<string | null>(null);
  const [setupCompleted, setSetupCompleted] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);
  const [isBackendReachable, setIsBackendReachable] = useState(false);
  const [connections, setConnections] = useState<DatabaseConnection[]>([]);
  const [targetDatabaseId, setTargetDatabaseId] = useState<string | null>(null);
  const [conversationDatabaseId, setConversationDatabaseId] = useState<string | null>(null);
  const [waitingMode, setWaitingMode] = useState<WaitingUxMode>("animated");
  const [resultLayoutMode, setResultLayoutMode] =
    useState<ResultLayoutMode>("stacked");
  const [showAgentTimingBreakdown, setShowAgentTimingBreakdown] = useState(true);
  const [synthesizeSimpleSql, setSynthesizeSimpleSql] = useState(true);
  const [showLiveReasoning, setShowLiveReasoning] = useState(true);
  const [thinkingNotes, setThinkingNotes] = useState<string[]>([]);
  const [loadingElapsedSeconds, setLoadingElapsedSeconds] = useState(0);
  const [toolApprovalOpen, setToolApprovalOpen] = useState(false);
  const [toolApprovalCalls, setToolApprovalCalls] = useState<
    { name: string; arguments?: Record<string, unknown> }[]
  >([]);
  const [toolApprovalMessage, setToolApprovalMessage] = useState<string | null>(null);
  const [toolApprovalRunning, setToolApprovalRunning] = useState(false);
  const [toolApprovalError, setToolApprovalError] = useState<string | null>(null);
  const [isHistorySidebarOpen, setIsHistorySidebarOpen] = useState(true);
  const [isSchemaSidebarOpen, setIsSchemaSidebarOpen] = useState(true);
  const [conversationHistory, setConversationHistory] = useState<ConversationSnapshot[]>([]);
  const [schemaTables, setSchemaTables] = useState<DatabaseSchemaTable[]>([]);
  const [schemaLoading, setSchemaLoading] = useState(false);
  const [schemaError, setSchemaError] = useState<string | null>(null);
  const [schemaSearch, setSchemaSearch] = useState("");
  const [selectedSchemaTable, setSelectedSchemaTable] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const restoreInputFocus = () => {
    window.requestAnimationFrame(() => {
      inputRef.current?.focus();
    });
  };

  const serializeMessages = (items: ChatStoreMessage[]): SerializedMessage[] =>
    items.slice(-MAX_CONVERSATION_MESSAGES).map((message) => ({
      ...message,
      timestamp:
        message.timestamp instanceof Date
          ? message.timestamp.toISOString()
          : new Date(message.timestamp).toISOString(),
    }));

  const deserializeMessages = (items: SerializedMessage[]): ChatStoreMessage[] =>
    items.map((message) => ({
      ...message,
      timestamp: new Date(message.timestamp),
    }));

  const buildConversationTitle = (items: ChatStoreMessage[]): string => {
    const firstUserMessage = items.find((message) => message.role === "user")?.content?.trim();
    if (!firstUserMessage) {
      return "Untitled conversation";
    }
    const compact = firstUserMessage.replace(/\s+/g, " ");
    return compact.length > 70 ? `${compact.slice(0, 67)}...` : compact;
  };

  const persistConversationHistory = (items: ConversationSnapshot[]) => {
    window.localStorage.setItem(
      CONVERSATION_HISTORY_STORAGE_KEY,
      JSON.stringify(items)
    );
    setConversationHistory(items);
  };

  const upsertConversationSnapshot = (
    override: {
      frontendSessionId?: string;
      messages?: ChatStoreMessage[];
      conversationId?: string | null;
      sessionSummary?: string | null;
      sessionState?: Record<string, unknown> | null;
      targetDatabaseId?: string | null;
    } = {}
  ) => {
    const snapshotMessages = override.messages || messages;
    if (!snapshotMessages.some((message) => message.role === "user")) {
      return;
    }
    const nowIso = new Date().toISOString();
    const snapshot: ConversationSnapshot = {
      frontendSessionId: override.frontendSessionId || frontendSessionId,
      title: buildConversationTitle(snapshotMessages),
      targetDatabaseId:
        override.targetDatabaseId === undefined
          ? targetDatabaseId
          : override.targetDatabaseId,
      conversationId:
        override.conversationId === undefined
          ? conversationId
          : override.conversationId,
      sessionSummary:
        override.sessionSummary === undefined
          ? sessionSummary
          : override.sessionSummary,
      sessionState:
        override.sessionState === undefined ? sessionState : override.sessionState,
      updatedAt: nowIso,
      messages: serializeMessages(snapshotMessages),
    };

    setConversationHistory((previous) => {
      const merged = [
        snapshot,
        ...previous.filter(
          (entry) => entry.frontendSessionId !== snapshot.frontendSessionId
        ),
      ].slice(0, MAX_CONVERSATION_HISTORY);
      window.localStorage.setItem(
        CONVERSATION_HISTORY_STORAGE_KEY,
        JSON.stringify(merged)
      );
      return merged;
    });
  };

  const categorizeError = (errorMessage: string): "network" | "timeout" | "validation" | "database" | "unknown" => {
    const lower = errorMessage.toLowerCase();
    if (
      lower.includes("network") ||
      lower.includes("connection") ||
      lower.includes("econnrefused") ||
      lower.includes("enotfound") ||
      lower.includes("fetch failed") ||
      lower.includes("websocket")
    ) {
      return "network";
    }
    if (
      lower.includes("timeout") ||
      lower.includes("timed out") ||
      lower.includes("deadline exceeded")
    ) {
      return "timeout";
    }
    if (
      lower.includes("validation") ||
      lower.includes("invalid") ||
      lower.includes("syntax") ||
      lower.includes("required")
    ) {
      return "validation";
    }
    if (
      lower.includes("database") ||
      lower.includes("sql") ||
      lower.includes("table") ||
      lower.includes("column") ||
      lower.includes("schema") ||
      lower.includes("query")
    ) {
      return "database";
    }
    return "unknown";
  };

  const getErrorIcon = (category: "network" | "timeout" | "validation" | "database" | "unknown") => {
    switch (category) {
      case "network":
        return Wifi;
      case "timeout":
        return Clock;
      case "database":
        return Database;
      case "validation":
        return AlertTriangle;
      default:
        return AlertCircle;
    }
  };

  const getErrorSuggestion = (category: "network" | "timeout" | "validation" | "database" | "unknown"): string => {
    switch (category) {
      case "network":
        return "Check your internet connection and try again.";
      case "timeout":
        return "The request took too long. Try simplifying your query.";
      case "validation":
        return "Please check your input and try again.";
      case "database":
        return "There was an issue with the database. Try rephrasing your query.";
      default:
        return "An unexpected error occurred. Please try again.";
    }
  };

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    const raw = window.localStorage.getItem(CONVERSATION_HISTORY_STORAGE_KEY);
    if (!raw) {
      setConversationHistory([]);
      return;
    }
    try {
      const parsed = JSON.parse(raw) as ConversationSnapshot[];
      if (Array.isArray(parsed)) {
        setConversationHistory(
          parsed
            .filter((entry) => entry && Array.isArray(entry.messages))
            .slice(0, MAX_CONVERSATION_HISTORY)
        );
      }
    } catch {
      setConversationHistory([]);
    }
  }, []);

  useEffect(() => {
    let isMounted = true;
    Promise.all([apiClient.systemStatus(), apiClient.listDatabases().catch(() => [])])
      .then(([status, dbs]) => {
        if (!isMounted) return;
        setIsBackendReachable(true);
        setIsInitialized(status.is_initialized);
        setSetupSteps(status.setup_required || []);
        setConnections(dbs);

        const storedId = window.localStorage.getItem(ACTIVE_DATABASE_STORAGE_KEY);
        const selected =
          dbs.find((db) => db.connection_id === storedId) ||
          dbs.find((db) => db.is_default) ||
          dbs[0] ||
          null;
        setTargetDatabaseId(selected?.connection_id ?? null);
      })
      .catch((err) => {
        if (!isMounted) return;
        console.error("System status error:", err);
        setIsBackendReachable(false);
      });
    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    if (!targetDatabaseId) {
      window.localStorage.removeItem(ACTIVE_DATABASE_STORAGE_KEY);
      return;
    }
    window.localStorage.setItem(ACTIVE_DATABASE_STORAGE_KEY, targetDatabaseId);
  }, [targetDatabaseId]);

  useEffect(() => {
    if (!conversationId || !targetDatabaseId || conversationDatabaseId) {
      return;
    }
    setConversationDatabaseId(targetDatabaseId);
  }, [conversationId, targetDatabaseId, conversationDatabaseId]);

  useEffect(() => {
    let isMounted = true;
    if (!targetDatabaseId) {
      setSchemaTables([]);
      setSchemaError(null);
      setSelectedSchemaTable(null);
      return;
    }

    setSchemaLoading(true);
    setSchemaError(null);
    apiClient
      .getDatabaseSchema(targetDatabaseId)
      .then((response) => {
        if (!isMounted) return;
        setSchemaTables(response.tables || []);
        setSelectedSchemaTable((prev) => {
          if (!prev) return null;
          const stillExists = (response.tables || []).some(
            (table) => `${table.schema_name}.${table.table_name}` === prev
          );
          return stillExists ? prev : null;
        });
      })
      .catch((err) => {
        if (!isMounted) return;
        setSchemaTables([]);
        setSchemaError(err instanceof Error ? err.message : "Failed to load schema");
      })
      .finally(() => {
        if (!isMounted) return;
        setSchemaLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [targetDatabaseId]);

  useEffect(() => {
    setWaitingMode(getWaitingUxMode());
    setResultLayoutMode(getResultLayoutMode());
    setShowAgentTimingBreakdown(getShowAgentTimingBreakdown());
    setSynthesizeSimpleSql(getSynthesizeSimpleSql());
    setShowLiveReasoning(getShowLiveReasoning());
    const handleStorage = () => {
      setWaitingMode(getWaitingUxMode());
      setResultLayoutMode(getResultLayoutMode());
      setShowAgentTimingBreakdown(getShowAgentTimingBreakdown());
      setSynthesizeSimpleSql(getSynthesizeSimpleSql());
      setShowLiveReasoning(getShowLiveReasoning());
    };
    window.addEventListener("storage", handleStorage);
    return () => {
      window.removeEventListener("storage", handleStorage);
    };
  }, []);

  useEffect(() => {
    if (!isLoading) {
      setLoadingElapsedSeconds(0);
      return;
    }

    const startedAt = Date.now();
    setLoadingElapsedSeconds(0);
    const interval = window.setInterval(() => {
      setLoadingElapsedSeconds(Math.max(1, Math.floor((Date.now() - startedAt) / 1000)));
    }, 500);

    return () => {
      window.clearInterval(interval);
    };
  }, [isLoading]);

  useEffect(() => {
    if (!isLoading) return;
    if (agentStatus === "idle") return;
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [agentHistory.length, agentStatus, isLoading]);

  const filteredSchemaTables = useMemo(() => {
    const search = schemaSearch.trim().toLowerCase();
    if (!search) {
      return schemaTables;
    }
    return schemaTables.filter((table) => {
      const fullName = `${table.schema_name}.${table.table_name}`.toLowerCase();
      if (fullName.includes(search)) {
        return true;
      }
      return table.columns.some((column) => column.name.toLowerCase().includes(search));
    });
  }, [schemaSearch, schemaTables]);

  const sortedConversationHistory = useMemo(
    () =>
      [...conversationHistory].sort(
        (a, b) =>
          new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
      ),
    [conversationHistory]
  );

  const formatSnapshotTime = (value: string) => {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return "";
    }
    return date.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  };

  // Handle send message
  const handleSend = async () => {
    if (!input.trim() || isLoading || !isInitialized) return;

    const query = input.trim();
    const requestDatabaseId = targetDatabaseId || null;
    const canReuseConversation =
      !!conversationId &&
      !!conversationDatabaseId &&
      conversationDatabaseId === requestDatabaseId;
    const conversationHistory = canReuseConversation
      ? messages.map((m) => ({
          role: m.role,
          content: m.content,
        }))
      : [];

    setInput("");
    setError(null);
    setErrorCategory(null);
    setLastFailedQuery(null);
    setLoading(true);
    setThinkingNotes([]);
    resetAgentStatus();
    if (!canReuseConversation) {
      setConversationId(null);
      setSessionMemory(null, null);
    }

    addMessage({
      role: "user",
      content: query,
    });

    addMessage({
      role: "assistant",
      content: "",
    });

    try {
      wsClient.streamChat(
        {
          message: query,
          conversation_id: canReuseConversation ? conversationId || undefined : undefined,
          target_database: requestDatabaseId || undefined,
          conversation_history: conversationHistory,
          session_summary: canReuseConversation ? sessionSummary : undefined,
          session_state: canReuseConversation ? sessionState : undefined,
          synthesize_simple_sql: synthesizeSimpleSql,
        },
        {
          onOpen: () => {
            setConnected(true);
          },
          onClose: () => {
            setConnected(false);
            setLoading(false);
            setThinkingNotes([]);
            restoreInputFocus();
          },
          onAgentUpdate: (update) => {
            setAgentUpdate(update);
          },
          onThinking: (note) => {
            if (!showLiveReasoning) return;
            setThinkingNotes((prev) => {
              if (!note.trim()) return prev;
              if (prev[prev.length - 1] === note) return prev;
              return [...prev.slice(-7), note];
            });
          },
          onAnswerChunk: (chunk) => {
            appendToLastMessage(chunk);
          },
          onComplete: (response) => {
            const nextConversationId = response.conversation_id || conversationId || null;
            const nextSummary = response.session_summary || null;
            const nextState = response.session_state || null;
            updateLastMessage({
              content: response.answer,
              clarifying_questions: response.clarifying_questions,
              sql: response.sql,
              data: response.data,
              visualization_hint: response.visualization_hint,
              visualization_metadata: response.visualization_metadata,
              sources: response.sources,
              answer_source: response.answer_source,
              answer_confidence: response.answer_confidence,
              evidence: response.evidence,
              metrics: response.metrics,
              tool_approval_required: response.tool_approval_required,
              tool_approval_message: response.tool_approval_message,
              tool_approval_calls: response.tool_approval_calls,
            });
            if (response.conversation_id) {
              setConversationId(response.conversation_id);
            }
            setSessionMemory(nextSummary, nextState);
            setConversationDatabaseId(requestDatabaseId);
            const currentMessages = useChatStore.getState().messages;
            upsertConversationSnapshot({
              messages: currentMessages,
              conversationId: nextConversationId,
              sessionSummary: nextSummary,
              sessionState: nextState,
              targetDatabaseId: requestDatabaseId,
            });
            if (response.tool_approval_required && response.tool_approval_calls?.length) {
              setToolApprovalCalls(response.tool_approval_calls);
              setToolApprovalMessage(
                response.tool_approval_message ||
                  "Approval required to run the requested tool."
              );
              setToolApprovalOpen(true);
            }
            setLoading(false);
            setThinkingNotes([]);
            resetAgentStatus();
            restoreInputFocus();
          },
          onError: (message) => {
            setError(message);
            setErrorCategory(categorizeError(message));
            setLastFailedQuery(query);
            setRetryCount((c) => c + 1);
            setLoading(false);
            setThinkingNotes([]);
            resetAgentStatus();
            restoreInputFocus();
          },
          onSystemNotInitialized: (steps, message) => {
            setIsInitialized(false);
            setSetupSteps(steps);
            setSetupError(message);
            setLoading(false);
            setThinkingNotes([]);
            resetAgentStatus();
            restoreInputFocus();
          },
        }
      );
    } catch (err) {
      console.error("Chat error:", err);
      const errorMessage = err instanceof Error ? err.message : "Failed to send message";
      setError(errorMessage);
      setErrorCategory(categorizeError(errorMessage));
      setLastFailedQuery(query);
      setRetryCount((c) => c + 1);
      setLoading(false);
      resetAgentStatus();
      restoreInputFocus();
    }
  };

  const handleRetry = () => {
    if (!lastFailedQuery || isLoading) return;
    setInput(lastFailedQuery);
    setError(null);
    setErrorCategory(null);
    setLastFailedQuery(null);
    inputRef.current?.focus();
  };

  const handleApplyTemplate = (templateId: string) => {
    const template = QUERY_TEMPLATES.find((item) => item.id === templateId);
    if (!template) {
      return;
    }
    setInput(template.build(selectedSchemaTable));
    restoreInputFocus();
  };

  const handleStartNewConversation = () => {
    upsertConversationSnapshot();
    clearMessages();
    setConversationDatabaseId(null);
    setConversationId(null);
    setSessionMemory(null, null);
    setInput("");
    setError(null);
    setErrorCategory(null);
    setLastFailedQuery(null);
    setRetryCount(0);
    restoreInputFocus();
  };

  const handleLoadConversation = (snapshot: ConversationSnapshot) => {
    const restoredMessages = deserializeMessages(snapshot.messages);
    loadSession({
      frontendSessionId: snapshot.frontendSessionId,
      messages: restoredMessages,
      conversationId: snapshot.conversationId,
      sessionSummary: snapshot.sessionSummary,
      sessionState: snapshot.sessionState,
    });
    setConversationDatabaseId(snapshot.targetDatabaseId);
    setTargetDatabaseId(snapshot.targetDatabaseId);
    setInput("");
    setError(null);
    setErrorCategory(null);
    setLastFailedQuery(null);
    setRetryCount(0);
    restoreInputFocus();
  };

  const handleDeleteConversation = (sessionId: string) => {
    const remaining = conversationHistory.filter(
      (entry) => entry.frontendSessionId !== sessionId
    );
    persistConversationHistory(remaining);
  };

  const handleApproveTools = async () => {
    setToolApprovalError(null);
    setToolApprovalRunning(true);
    try {
      for (const call of toolApprovalCalls) {
        const result = await apiClient.executeTool({
          name: call.name,
          arguments: call.arguments || {},
          approved: true,
        });
        const payload = result.result || {};
        const summary =
          (payload as Record<string, unknown>).message ||
          (payload as Record<string, unknown>).answer ||
          `Tool ${call.name} completed.`;
        addMessage({
          role: "assistant",
          content: String(summary),
        });
      }
      setToolApprovalOpen(false);
      setToolApprovalCalls([]);
      setToolApprovalMessage(null);
    } catch (err) {
      setToolApprovalError((err as Error).message);
    } finally {
      setToolApprovalRunning(false);
    }
  };

  const toolCostEstimate = () => {
    if (!toolApprovalCalls.length) {
      return null;
    }
    const estimates = toolApprovalCalls.map((call) => {
      if (call.name === "profile_and_generate_datapoints") {
        const args = call.arguments || {};
        const batchSize = Number(args.batch_size || 10);
        const maxTables = args.max_tables ? Number(args.max_tables) : null;
        const tableCount = maxTables || "unknown";
        const batches =
          typeof tableCount === "number"
            ? Math.ceil(tableCount / batchSize)
            : "unknown";
        return {
          tool: call.name,
          tables: tableCount,
          batchSize,
          batches,
          llmCalls: typeof batches === "number" ? batches : "unknown",
        };
      }
      return { tool: call.name };
    });
    return estimates;
  };

  // Handle key press (Enter to send)
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Handle clear conversation
  const handleClear = () => {
    if (confirm("Clear all messages?")) {
      upsertConversationSnapshot();
      clearMessages();
      setConversationDatabaseId(null);
      setError(null);
    }
  };

  const handleInitialize = async (
    databaseUrl: string,
    autoProfile: boolean,
    systemDatabaseUrl?: string
  ) => {
    setSetupError(null);
    setSetupNotice(null);
    setIsInitializing(true);
    try {
      const response = await apiClient.systemInitialize({
        database_url: databaseUrl,
        system_database_url: systemDatabaseUrl,
        auto_profile: autoProfile,
      });
      setIsInitialized(response.is_initialized);
      setSetupSteps(response.setup_required || []);
      if (response.message) {
        setSetupNotice(response.message);
        if (response.message.toLowerCase().includes("initialization completed")) {
          setSetupCompleted(true);
          router.push("/databases");
        }
      }
    } catch (err) {
      console.error("Initialization error:", err);
      setSetupError(
        err instanceof Error ? err.message : "Initialization failed"
      );
    } finally {
      setIsInitializing(false);
    }
  };

  return (
    <div className="flex h-full min-h-0">
      <aside
        className={`hidden border-r bg-muted/20 transition-all duration-200 lg:flex lg:flex-col ${
          isHistorySidebarOpen ? "w-72" : "w-14"
        }`}
      >
        <div className="flex items-center justify-between border-b px-2 py-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsHistorySidebarOpen((prev) => !prev)}
            aria-label="Toggle conversation history sidebar"
          >
            {isHistorySidebarOpen ? <PanelLeftClose size={16} /> : <PanelLeftOpen size={16} />}
          </Button>
          {isHistorySidebarOpen && (
            <>
              <div className="text-xs font-medium">Conversations</div>
              <Button
                variant="ghost"
                size="icon"
                onClick={handleStartNewConversation}
                aria-label="Start new conversation"
              >
                <Plus size={15} />
              </Button>
            </>
          )}
        </div>
        {isHistorySidebarOpen ? (
          <div className="min-h-0 flex-1 overflow-y-auto p-2">
            {sortedConversationHistory.length === 0 ? (
              <div className="rounded border border-dashed p-3 text-xs text-muted-foreground">
                No saved conversations yet.
              </div>
            ) : (
              <div className="space-y-2">
                {sortedConversationHistory.map((snapshot) => {
                  const isActive = snapshot.frontendSessionId === frontendSessionId;
                  return (
                    <div
                      key={snapshot.frontendSessionId}
                      className={`rounded border ${
                        isActive ? "border-primary/50 bg-primary/5" : "border-border"
                      }`}
                    >
                      <button
                        type="button"
                        onClick={() => handleLoadConversation(snapshot)}
                        className="w-full px-2 py-2 text-left"
                      >
                        <p className="truncate text-xs font-medium">{snapshot.title}</p>
                        <p className="mt-1 text-[11px] text-muted-foreground">
                          {formatSnapshotTime(snapshot.updatedAt)}
                        </p>
                        {snapshot.targetDatabaseId && (
                          <p className="mt-1 text-[11px] text-muted-foreground">
                            DB: {snapshot.targetDatabaseId}
                          </p>
                        )}
                      </button>
                      <div className="flex justify-end border-t px-1 py-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 px-2 text-[11px]"
                          onClick={(event) => {
                            event.stopPropagation();
                            handleDeleteConversation(snapshot.frontendSessionId);
                          }}
                        >
                          <Trash2 size={12} />
                        </Button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-1 items-center justify-center text-muted-foreground">
            <History size={16} />
          </div>
        )}
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex-shrink-0 border-b p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="icon"
                className="hidden lg:inline-flex"
                onClick={() => setIsHistorySidebarOpen((prev) => !prev)}
                aria-label="Toggle conversation sidebar"
              >
                {isHistorySidebarOpen ? <PanelLeftClose size={16} /> : <PanelLeftOpen size={16} />}
              </Button>
              <div>
                <h1 className="text-2xl font-bold">DataChat</h1>
                <p className="text-sm text-muted-foreground">
                  Ask questions in natural language
                </p>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {connections.length > 0 && (
                <select
                  value={targetDatabaseId ?? ""}
                  onChange={(event) => {
                    const nextId = event.target.value || null;
                    if (nextId !== targetDatabaseId) {
                      setConversationId(null);
                      setConversationDatabaseId(null);
                      resetAgentStatus();
                    }
                    setTargetDatabaseId(nextId);
                  }}
                  className="h-8 rounded-md border border-input bg-background px-2 text-xs"
                  disabled={isLoading}
                  aria-label="Target database"
                >
                  {connections.map((connection) => (
                    <option key={connection.connection_id} value={connection.connection_id}>
                      {connection.name}
                      {` (${connection.database_type})`}
                      {connection.tags?.includes("env") ? " (env)" : ""}
                      {connection.is_default ? " (default)" : ""}
                    </option>
                  ))}
                </select>
              )}
              <Button
                variant="outline"
                size="sm"
                onClick={handleStartNewConversation}
                disabled={isLoading}
              >
                <Plus size={14} />
                New
              </Button>
              <Button asChild variant="ghost" size="sm">
                <Link href="/settings">Settings</Link>
              </Button>
              <Button asChild variant="secondary" size="sm">
                <Link href="/databases">Manage DataPoints</Link>
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="hidden xl:inline-flex"
                onClick={() => setIsSchemaSidebarOpen((prev) => !prev)}
                aria-label="Toggle schema sidebar"
              >
                {isSchemaSidebarOpen ? <PanelRightClose size={16} /> : <PanelRightOpen size={16} />}
              </Button>
              <div className="flex items-center gap-2 text-xs">
                <div
                  className={`h-2 w-2 rounded-full ${
                    isConnected || isBackendReachable ? "bg-green-500" : "bg-red-500"
                  }`}
                />
                {isLoading && <Loader2 className="h-3 w-3 animate-spin text-primary" />}
                <span className="text-muted-foreground">
                  {isLoading
                    ? formatWaitingChipLabel(loadingElapsedSeconds)
                    : isConnected
                      ? "Streaming"
                      : isBackendReachable
                        ? "Ready"
                        : "Disconnected"}
                </span>
              </div>
              {messages.length > 0 && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleClear}
                  disabled={isLoading}
                >
                  <Trash2 size={14} />
                  Clear
                </Button>
              )}
            </div>
          </div>
        </div>

        <div className="flex min-h-0 flex-1">
          <div className="flex min-w-0 flex-1 flex-col">
            <div className="flex-1 overflow-y-auto p-4">
              {!isInitialized && !setupCompleted && (
                <SystemSetup
                  steps={setupSteps}
                  onInitialize={handleInitialize}
                  isSubmitting={isInitializing}
                  error={setupError}
                  notice={setupNotice}
                />
              )}
              {!isInitialized && setupCompleted && (
                <div className="mb-4 rounded-md border border-muted bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
                  Setup saved. Add DataPoints from{" "}
                  <Link href="/databases" className="underline">
                    Database Manager
                  </Link>{" "}
                  (or run <strong>datachat demo</strong>) to enable chat.
                </div>
              )}
              {messages.length === 0 && (
                <div className="flex h-full items-center justify-center text-muted-foreground">
                  <div className="text-center">
                    <p className="mb-2 text-lg">Welcome to DataChat!</p>
                    <p className="text-sm">
                      Ask a question about your data to get started.
                    </p>
                    <p className="mt-2 text-xs text-muted-foreground">
                      New here? Run <strong>datachat demo</strong> to load sample data.
                    </p>
                  </div>
                </div>
              )}

              {messages.map((message) => (
                <Message
                  key={message.id}
                  message={message}
                  displayMode={resultLayoutMode}
                  showAgentTimingBreakdown={showAgentTimingBreakdown}
                  onClarifyingAnswer={(question) => {
                    setInput(`Regarding "${question}": `);
                    inputRef.current?.focus();
                  }}
                />
              ))}

              {isLoading && showLiveReasoning && (
                <Card className="mb-4 border-primary/20 bg-primary/5">
                  <div className="p-3">
                    <div className="mb-2 text-xs font-medium text-primary">Working...</div>
                    <ul className="space-y-1 text-xs text-muted-foreground">
                      {(thinkingNotes.length
                        ? thinkingNotes
                        : ["Understanding your request..."]).map((note, idx) => (
                        <li key={`${idx}-${note}`} className="flex items-start gap-2">
                          <span className="mt-1 inline-block h-1.5 w-1.5 rounded-full bg-primary/70" />
                          <span>{note}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </Card>
              )}

              <AgentStatus mode={waitingMode} />
              {isLoading && (
                <div className="mb-4 flex items-center justify-center">
                  <div className="inline-flex items-center gap-2 rounded-full border border-primary/25 bg-primary/5 px-3 py-1 text-xs text-primary">
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    <span>{formatWaitingChipLabel(loadingElapsedSeconds)}</span>
                  </div>
                </div>
              )}

              {error && errorCategory && (
                <Card className="mb-4 border-destructive bg-destructive/10">
                  <div className="p-4">
                    <div className="flex items-start gap-3">
                      {(() => {
                        const Icon = getErrorIcon(errorCategory);
                        return (
                          <Icon className="mt-0.5 h-5 w-5 flex-shrink-0 text-destructive" />
                        );
                      })()}
                      <div className="flex-1">
                        <div className="mb-1 flex items-center gap-2">
                          <p className="text-sm font-medium text-destructive">
                            {errorCategory.charAt(0).toUpperCase() + errorCategory.slice(1)} Error
                          </p>
                          {retryCount > 0 && (
                            <span className="text-xs text-muted-foreground">
                              (attempt {retryCount})
                            </span>
                          )}
                        </div>
                        <p className="text-sm text-muted-foreground">{error}</p>
                        <p className="mt-1 text-xs text-muted-foreground">
                          {getErrorSuggestion(errorCategory)}
                        </p>
                      </div>
                    </div>
                    <div className="mt-3 flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={handleRetry}
                        disabled={isLoading || !lastFailedQuery}
                        className="text-xs"
                      >
                        <RefreshCw size={14} className="mr-1" />
                        Retry Query
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          setError(null);
                          setErrorCategory(null);
                          setLastFailedQuery(null);
                          setRetryCount(0);
                        }}
                        className="text-xs text-muted-foreground"
                      >
                        Dismiss
                      </Button>
                    </div>
                  </div>
                </Card>
              )}
              <div ref={messagesEndRef} />
            </div>

            <div className="flex-shrink-0 border-t p-4">
              <div className="mb-2 flex flex-wrap gap-2">
                {QUERY_TEMPLATES.map((template) => (
                  <button
                    key={template.id}
                    type="button"
                    onClick={() => handleApplyTemplate(template.id)}
                    className="rounded-full border border-border bg-background px-3 py-1 text-xs hover:bg-muted"
                    disabled={isLoading || !isInitialized}
                  >
                    {template.label}
                  </button>
                ))}
                {selectedSchemaTable && (
                  <span className="inline-flex items-center rounded-full bg-primary/10 px-3 py-1 text-xs text-primary">
                    Table: {selectedSchemaTable}
                  </span>
                )}
              </div>
              <div className="flex gap-2">
                <Input
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyPress}
                  placeholder="Ask a question about your data..."
                  disabled={isLoading || !isInitialized}
                  className="flex-1"
                />
                <Button
                  onClick={handleSend}
                  disabled={!input.trim() || isLoading || !isInitialized}
                  size="icon"
                >
                  {isLoading ? (
                    <Loader2 size={18} className="animate-spin" />
                  ) : (
                    <Send size={18} />
                  )}
                </Button>
              </div>
              <p className="mt-2 text-xs text-muted-foreground">Press Enter to send</p>
              {conversationId &&
                conversationDatabaseId &&
                targetDatabaseId &&
                conversationDatabaseId !== targetDatabaseId && (
                  <p className="mt-1 text-xs text-amber-700">
                    Data source changed. Next query starts a fresh conversation context.
                  </p>
                )}
            </div>
          </div>

          <aside
            className={`hidden border-l bg-muted/20 transition-all duration-200 xl:flex xl:flex-col ${
              isSchemaSidebarOpen ? "w-80" : "w-14"
            }`}
          >
            <div className="flex items-center justify-between border-b px-2 py-2">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsSchemaSidebarOpen((prev) => !prev)}
                aria-label="Toggle schema sidebar"
              >
                {isSchemaSidebarOpen ? <PanelRightClose size={16} /> : <PanelRightOpen size={16} />}
              </Button>
              {isSchemaSidebarOpen && <div className="text-xs font-medium">Schema Explorer</div>}
            </div>
            {isSchemaSidebarOpen ? (
              <div className="flex min-h-0 flex-1 flex-col">
                <div className="border-b p-2">
                  <Input
                    value={schemaSearch}
                    onChange={(event) => setSchemaSearch(event.target.value)}
                    placeholder="Search table or column..."
                    className="h-8 text-xs"
                  />
                </div>
                <div className="min-h-0 flex-1 overflow-y-auto p-2">
                  {schemaLoading && (
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Loader2 className="h-3 w-3 animate-spin" />
                      Loading schema...
                    </div>
                  )}
                  {!schemaLoading && schemaError && (
                    <div className="rounded border border-destructive/30 bg-destructive/10 p-2 text-xs text-destructive">
                      {schemaError}
                    </div>
                  )}
                  {!schemaLoading && !schemaError && filteredSchemaTables.length === 0 && (
                    <div className="rounded border border-dashed p-3 text-xs text-muted-foreground">
                      No tables matched your search.
                    </div>
                  )}
                  <div className="space-y-2">
                    {filteredSchemaTables.map((table) => {
                      const fullName = `${table.schema_name}.${table.table_name}`;
                      const isSelected = selectedSchemaTable === fullName;
                      return (
                        <details key={fullName} className="rounded border border-border bg-background">
                          <summary
                            className={`cursor-pointer list-none px-2 py-2 text-xs ${
                              isSelected ? "bg-primary/5" : ""
                            }`}
                            onClick={() => setSelectedSchemaTable(fullName)}
                          >
                            <div className="flex items-center justify-between gap-2">
                              <div className="min-w-0">
                                <div className="truncate font-medium">{fullName}</div>
                                <div className="text-[11px] text-muted-foreground">
                                  {table.table_type}
                                  {typeof table.row_count === "number"
                                    ? ` · ~${table.row_count} rows`
                                    : ""}
                                </div>
                              </div>
                              <div className="flex items-center gap-1 text-muted-foreground">
                                <ChevronDown size={12} />
                                <ChevronRight size={12} />
                              </div>
                            </div>
                          </summary>
                          <div className="border-t px-2 py-2">
                            <button
                              type="button"
                              className="mb-2 inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-[11px] hover:bg-muted"
                              onClick={() => {
                                setSelectedSchemaTable(fullName);
                                setInput(`Show first 100 rows from ${fullName}.`);
                                restoreInputFocus();
                              }}
                            >
                              <Table2 size={12} />
                              Use In Query
                            </button>
                            <ul className="space-y-1 text-[11px]">
                              {table.columns.map((column) => (
                                <li key={`${fullName}.${column.name}`} className="flex flex-wrap gap-1">
                                  <span className="font-medium">{column.name}</span>
                                  <span className="text-muted-foreground">({column.data_type})</span>
                                  {column.is_primary_key && (
                                    <span className="rounded bg-blue-100 px-1 text-[10px] text-blue-800">
                                      PK
                                    </span>
                                  )}
                                  {column.is_foreign_key && (
                                    <span className="rounded bg-amber-100 px-1 text-[10px] text-amber-900">
                                      FK
                                    </span>
                                  )}
                                </li>
                              ))}
                            </ul>
                          </div>
                        </details>
                      );
                    })}
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-1 items-center justify-center text-muted-foreground">
                <Database size={16} />
              </div>
            )}
          </aside>
        </div>
      </div>
      {toolApprovalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="w-full max-w-lg rounded-lg bg-background p-6 shadow-lg">
            <h3 className="text-base font-semibold">Tool Approval Required</h3>
            <p className="mt-2 text-sm text-muted-foreground">
              {toolApprovalMessage}
            </p>
            <div className="mt-4 space-y-2 text-xs text-muted-foreground">
              {toolApprovalCalls.map((call) => (
                <div key={call.name} className="rounded-md border border-border p-2">
                  <div className="font-medium text-foreground">{call.name}</div>
                  <pre className="mt-1 whitespace-pre-wrap">
                    {JSON.stringify(call.arguments || {}, null, 2)}
                  </pre>
                </div>
              ))}
            </div>
            <div className="mt-4 rounded-md border border-muted bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
              Cost hint:
              {toolCostEstimate() ? (
                <div className="mt-2 space-y-1">
                  {toolCostEstimate()?.map((item, index) => (
                    <div key={`${item.tool}-${index}`}>
                      {item.tool}: tables={item.tables ?? "unknown"} · batch size=
                      {"batchSize" in item ? item.batchSize : "n/a"} · batches=
                      {"batches" in item ? item.batches : "unknown"} · LLM calls≈
                      {"llmCalls" in item ? item.llmCalls : "unknown"}
                    </div>
                  ))}
                </div>
              ) : (
                <span> this may run LLM calls or database profiling.</span>
              )}
            </div>
            {toolApprovalError && (
              <div className="mt-2 text-xs text-destructive">{toolApprovalError}</div>
            )}
            <div className="mt-4 flex gap-2">
              <button
                className="rounded-md bg-primary px-3 py-2 text-xs text-primary-foreground"
                onClick={handleApproveTools}
                disabled={toolApprovalRunning}
              >
                {toolApprovalRunning ? "Approving..." : "Approve"}
              </button>
              <button
                className="rounded-md border border-border px-3 py-2 text-xs"
                onClick={() => setToolApprovalOpen(false)}
                disabled={toolApprovalRunning}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
