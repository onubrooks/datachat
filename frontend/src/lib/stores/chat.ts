/**
 * Chat Store - Zustand
 *
 * Manages chat state including messages, agent updates, and WebSocket connection.
 */

import { create } from "zustand";
import type { ChatMessage, ChatResponse, AgentUpdate } from "../api";

export interface Message extends ChatMessage {
  id: string;
  timestamp: Date;
  sql?: string | null;
  data?: Record<string, unknown>[] | null;
  visualization_hint?: string | null;
  sources?: Array<{
    datapoint_id: string;
    type: string;
    name: string;
    relevance_score: number;
  }>;
  metrics?: {
    total_latency_ms: number;
    agent_timings: Record<string, number>;
    llm_calls: number;
    retry_count: number;
  };
}

interface ChatState {
  // Messages
  messages: Message[];
  conversationId: string | null;

  // Agent status
  currentAgent: string | null;
  agentStatus: "idle" | "running" | "completed" | "error";
  agentMessage: string | null;
  agentError: string | null;
  agentHistory: AgentUpdate[];

  // Loading state
  isLoading: boolean;

  // WebSocket connection
  isConnected: boolean;

  // Actions
  addMessage: (message: Omit<Message, "id" | "timestamp">) => void;
  updateLastMessage: (
    updates: Partial<Omit<Message, "id" | "timestamp">>
  ) => void;
  setConversationId: (id: string) => void;
  setAgentUpdate: (update: AgentUpdate) => void;
  resetAgentStatus: () => void;
  setLoading: (loading: boolean) => void;
  setConnected: (connected: boolean) => void;
  clearMessages: () => void;
  addChatResponse: (query: string, response: ChatResponse) => void;
}

export const useChatStore = create<ChatState>((set) => ({
  // Initial state
  messages: [],
  conversationId: null,
  currentAgent: null,
  agentStatus: "idle",
  agentMessage: null,
  agentError: null,
  agentHistory: [],
  isLoading: false,
  isConnected: false,

  // Actions
  addMessage: (message) =>
    set((state) => ({
      messages: [
        ...state.messages,
        {
          ...message,
          id: crypto.randomUUID(),
          timestamp: new Date(),
        },
      ],
    })),

  updateLastMessage: (updates) =>
    set((state) => {
      const messages = [...state.messages];
      const lastMessage = messages[messages.length - 1];
      if (lastMessage) {
        messages[messages.length - 1] = {
          ...lastMessage,
          ...updates,
        };
      }
      return { messages };
    }),

  setConversationId: (id) => set({ conversationId: id }),

  setAgentUpdate: (update) =>
    set((state) => ({
      currentAgent: update.current_agent,
      agentStatus: update.status,
      agentMessage: update.message || null,
      agentError: update.error || null,
      agentHistory: [...state.agentHistory, update],
    })),

  resetAgentStatus: () =>
    set({
      currentAgent: null,
      agentStatus: "idle",
      agentMessage: null,
      agentError: null,
      agentHistory: [],
    }),

  setLoading: (loading) => set({ isLoading: loading }),

  setConnected: (connected) => set({ isConnected: connected }),

  clearMessages: () =>
    set({
      messages: [],
      conversationId: null,
      currentAgent: null,
      agentStatus: "idle",
      agentMessage: null,
      agentError: null,
      agentHistory: [],
    }),

  addChatResponse: (query, response) =>
    set((state) => {
      const userMessage: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content: query,
        timestamp: new Date(),
      };

      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: response.answer,
        timestamp: new Date(),
        sql: response.sql,
        data: response.data,
        visualization_hint: response.visualization_hint,
        sources: response.sources,
        metrics: response.metrics,
      };

      return {
        messages: [...state.messages, userMessage, assistantMessage],
        conversationId: response.conversation_id,
      };
    }),
}));
