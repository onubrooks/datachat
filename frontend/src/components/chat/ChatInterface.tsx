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

import React, { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Send, Trash2, AlertCircle, Loader2 } from "lucide-react";
import { Message } from "./Message";
import { AgentStatus } from "../agents/AgentStatus";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Card } from "../ui/card";
import { useChatStore } from "@/lib/stores/chat";
import {
  apiClient,
  wsClient,
  type DatabaseConnection,
  type SetupStep,
} from "@/lib/api";
import { SystemSetup } from "../system/SystemSetup";
import {
  getResultLayoutMode,
  getShowAgentTimingBreakdown,
  getSynthesizeSimpleSql,
  getWaitingUxMode,
  type ResultLayoutMode,
  type WaitingUxMode,
} from "@/lib/settings";
import { formatWaitingChipLabel } from "./loadingUx";

export function ChatInterface() {
  const ACTIVE_DATABASE_STORAGE_KEY = "datachat.active_connection_id";
  const router = useRouter();
  const {
    messages,
    conversationId,
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
    appendToLastMessage,
  } = useChatStore();

  const [input, setInput] = useState("");
  const [error, setError] = useState<string | null>(null);
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
  const [loadingElapsedSeconds, setLoadingElapsedSeconds] = useState(0);
  const [toolApprovalOpen, setToolApprovalOpen] = useState(false);
  const [toolApprovalCalls, setToolApprovalCalls] = useState<
    { name: string; arguments?: Record<string, unknown> }[]
  >([]);
  const [toolApprovalMessage, setToolApprovalMessage] = useState<string | null>(null);
  const [toolApprovalRunning, setToolApprovalRunning] = useState(false);
  const [toolApprovalError, setToolApprovalError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

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
    setWaitingMode(getWaitingUxMode());
    setResultLayoutMode(getResultLayoutMode());
    setShowAgentTimingBreakdown(getShowAgentTimingBreakdown());
    setSynthesizeSimpleSql(getSynthesizeSimpleSql());
    const handleStorage = () => {
      setWaitingMode(getWaitingUxMode());
      setResultLayoutMode(getResultLayoutMode());
      setShowAgentTimingBreakdown(getShowAgentTimingBreakdown());
      setSynthesizeSimpleSql(getSynthesizeSimpleSql());
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
    setLoading(true);
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
          },
          onAgentUpdate: (update) => {
            setAgentUpdate(update);
          },
          onAnswerChunk: (chunk) => {
            appendToLastMessage(chunk);
          },
          onComplete: (response) => {
            updateLastMessage({
              content: response.answer,
              clarifying_questions: response.clarifying_questions,
              sql: response.sql,
              data: response.data,
              visualization_hint: response.visualization_hint,
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
            setSessionMemory(response.session_summary || null, response.session_state || null);
            setConversationDatabaseId(requestDatabaseId);
            if (response.tool_approval_required && response.tool_approval_calls?.length) {
              setToolApprovalCalls(response.tool_approval_calls);
              setToolApprovalMessage(
                response.tool_approval_message ||
                  "Approval required to run the requested tool."
              );
              setToolApprovalOpen(true);
            }
            setLoading(false);
            resetAgentStatus();
          },
          onError: (message) => {
            setError(message);
            setLoading(false);
            resetAgentStatus();
          },
          onSystemNotInitialized: (steps, message) => {
            setIsInitialized(false);
            setSetupSteps(steps);
            setSetupError(message);
            setLoading(false);
            resetAgentStatus();
          },
        }
      );
    } catch (err) {
      console.error("Chat error:", err);
      setError(
        err instanceof Error ? err.message : "Failed to send message"
      );
      setLoading(false);
      resetAgentStatus();
    }
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
      clearMessages();
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
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex-shrink-0 border-b p-4 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">DataChat</h1>
          <p className="text-sm text-muted-foreground">
            Ask questions in natural language
          </p>
        </div>
        <div className="flex items-center gap-2">
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
          <Button asChild variant="ghost" size="sm">
            <Link href="/settings">Settings</Link>
          </Button>
          <Button asChild variant="secondary" size="sm">
            <Link href="/databases">Manage DataPoints</Link>
          </Button>
          {/* Connection status indicator */}
          <div className="flex items-center gap-2 text-xs">
            <div
              className={`w-2 h-2 rounded-full ${
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

          {/* Clear button */}
          {messages.length > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleClear}
              disabled={isLoading}
            >
              <Trash2 size={16} />
              Clear
            </Button>
          )}
        </div>
      </div>

      {/* Messages */}
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
          <div className="flex items-center justify-center h-full text-muted-foreground">
            <div className="text-center">
              <p className="text-lg mb-2">Welcome to DataChat!</p>
              <p className="text-sm">
                Ask a question about your data to get started.
              </p>
              <p className="text-xs text-muted-foreground mt-2">
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

        {/* Agent Status */}
        <AgentStatus mode={waitingMode} />
        {isLoading && (
          <div className="mb-4 flex items-center justify-center">
            <div className="inline-flex items-center gap-2 rounded-full border border-primary/25 bg-primary/5 px-3 py-1 text-xs text-primary">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              <span>{formatWaitingChipLabel(loadingElapsedSeconds)}</span>
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <Card className="mb-4 border-destructive bg-destructive/10">
            <div className="p-4 flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="text-sm font-medium text-destructive">
                  Error
                </p>
                <p className="text-sm text-muted-foreground">{error}</p>
              </div>
            </div>
          </Card>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="flex-shrink-0 border-t p-4">
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
            {isLoading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
          </Button>
        </div>
        <p className="text-xs text-muted-foreground mt-2">
          Press Enter to send
        </p>
        {conversationId &&
          conversationDatabaseId &&
          targetDatabaseId &&
          conversationDatabaseId !== targetDatabaseId && (
            <p className="text-xs text-amber-700 mt-1">
              Data source changed. Next query starts a fresh conversation context.
            </p>
          )}
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
