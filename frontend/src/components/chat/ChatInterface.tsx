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
import { Send, Trash2, AlertCircle } from "lucide-react";
import { Message } from "./Message";
import { AgentStatus } from "../agents/AgentStatus";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Card } from "../ui/card";
import { useChatStore } from "@/lib/stores/chat";
import { apiClient, wsClient } from "@/lib/api";

export function ChatInterface() {
  const {
    messages,
    conversationId,
    isLoading,
    isConnected,
    addChatResponse,
    setLoading,
    setConnected,
    setAgentUpdate,
    resetAgentStatus,
    clearMessages,
  } = useChatStore();

  const [input, setInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Initialize WebSocket connection
  useEffect(() => {
    // Connect to WebSocket
    wsClient.connect(
      (update) => {
        setAgentUpdate(update);
      },
      (error) => {
        console.error("WebSocket error:", error);
        setConnected(false);
      },
      () => {
        setConnected(false);
      }
    );

    // Update connection status
    const interval = setInterval(() => {
      setConnected(wsClient.isConnected());
    }, 1000);

    // Cleanup
    return () => {
      clearInterval(interval);
      wsClient.disconnect();
    };
  }, [setAgentUpdate, setConnected]);

  // Handle send message
  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const query = input.trim();
    setInput("");
    setError(null);
    setLoading(true);
    resetAgentStatus();

    try {
      const response = await apiClient.chat({
        message: query,
        conversation_id: conversationId || undefined,
        conversation_history: messages.map((m) => ({
          role: m.role,
          content: m.content,
        })),
      });

      addChatResponse(query, response);
    } catch (err) {
      console.error("Chat error:", err);
      setError(
        err instanceof Error ? err.message : "Failed to send message"
      );
    } finally {
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
          {/* Connection status indicator */}
          <div className="flex items-center gap-2 text-xs">
            <div
              className={`w-2 h-2 rounded-full ${
                isConnected ? "bg-green-500" : "bg-red-500"
              }`}
            />
            <span className="text-muted-foreground">
              {isConnected ? "Connected" : "Disconnected"}
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
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-muted-foreground">
            <div className="text-center">
              <p className="text-lg mb-2">Welcome to DataChat!</p>
              <p className="text-sm">
                Ask a question about your data to get started.
              </p>
            </div>
          </div>
        )}

        {messages.map((message) => (
          <Message key={message.id} message={message} />
        ))}

        {/* Agent Status */}
        <AgentStatus />

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
            disabled={isLoading}
            className="flex-1"
          />
          <Button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            size="icon"
          >
            <Send size={18} />
          </Button>
        </div>
        <p className="text-xs text-muted-foreground mt-2">
          Press Enter to send, Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}
