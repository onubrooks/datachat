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
import { Send, Trash2, AlertCircle } from "lucide-react";
import { Message } from "./Message";
import { AgentStatus } from "../agents/AgentStatus";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Card } from "../ui/card";
import { useChatStore } from "@/lib/stores/chat";
import { apiClient, wsClient, type SetupStep } from "@/lib/api";
import { SystemSetup } from "../system/SystemSetup";
import { getWaitingUxMode, type WaitingUxMode } from "@/lib/settings";

export function ChatInterface() {
  const router = useRouter();
  const {
    messages,
    conversationId,
    isLoading,
    isConnected,
    setLoading,
    setConnected,
    setAgentUpdate,
    resetAgentStatus,
    clearMessages,
    addMessage,
    updateLastMessage,
    setConversationId,
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
  const [waitingMode, setWaitingMode] = useState<WaitingUxMode>("animated");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    let isMounted = true;
    apiClient
      .systemStatus()
      .then((status) => {
        if (!isMounted) return;
        setIsBackendReachable(true);
        setIsInitialized(status.is_initialized);
        setSetupSteps(status.setup_required || []);
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
    setWaitingMode(getWaitingUxMode());
    const handleStorage = () => {
      setWaitingMode(getWaitingUxMode());
    };
    window.addEventListener("storage", handleStorage);
    return () => {
      window.removeEventListener("storage", handleStorage);
    };
  }, []);

  // Handle send message
  const handleSend = async () => {
    if (!input.trim() || isLoading || !isInitialized) return;

    const query = input.trim();
    const conversationHistory = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    setInput("");
    setError(null);
    setLoading(true);
    resetAgentStatus();

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
          conversation_id: conversationId || undefined,
          conversation_history: conversationHistory,
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
              sql: response.sql,
              data: response.data,
              visualization_hint: response.visualization_hint,
              sources: response.sources,
              answer_source: response.answer_source,
              answer_confidence: response.answer_confidence,
              evidence: response.evidence,
              metrics: response.metrics,
            });
            if (response.conversation_id) {
              setConversationId(response.conversation_id);
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
            <span className="text-muted-foreground">
              {isConnected
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
          <Message key={message.id} message={message} />
        ))}

        {/* Agent Status */}
        <AgentStatus mode={waitingMode} />

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
            <Send size={18} />
          </Button>
        </div>
        <p className="text-xs text-muted-foreground mt-2">
          Press Enter to send
        </p>
      </div>
    </div>
  );
}
