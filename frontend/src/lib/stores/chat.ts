/**
 * Chat Store - Zustand
 *
 * Manages chat state including messages, agent updates, and WebSocket connection.
 */

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import type { ChatMessage, ChatResponse, AgentUpdate } from "../api";

export interface Message extends ChatMessage {
  id: string;
  timestamp: Date;
  sql?: string | null;
  data?: Record<string, unknown[]> | null;
  visualization_hint?: string | null;
  clarifying_questions?: string[];
  sources?: Array<{
    datapoint_id: string;
    type: string;
    name: string;
    relevance_score: number;
  }>;
  answer_source?: string | null;
  answer_confidence?: number | null;
  evidence?: Array<{
    datapoint_id: string;
    name?: string | null;
    type?: string | null;
    reason?: string | null;
  }>;
  tool_approval_required?: boolean;
  tool_approval_message?: string | null;
  tool_approval_calls?: Array<{
    name: string;
    arguments?: Record<string, unknown>;
  }>;
  metrics?: {
    total_latency_ms: number;
    agent_timings: Record<string, number>;
    llm_calls: number;
    retry_count: number;
  };
}

type PersistedMessage = Pick<
  Message,
  | "id"
  | "role"
  | "content"
  | "clarifying_questions"
  | "answer_source"
  | "answer_confidence"
  | "tool_approval_required"
  | "tool_approval_message"
> & {
  timestamp: string | Date;
};

const MAX_PERSISTED_MESSAGES = 60;
const MAX_PERSISTED_CONTENT_CHARS = 4000;

const createSessionId = () => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `session_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
};

const createMessageId = () => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `msg_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
};

const noopStorage = {
  getItem: () => null,
  setItem: () => undefined,
  removeItem: () => undefined,
};

const reviveMessage = (message: PersistedMessage): Message => ({
  ...message,
  timestamp: message.timestamp instanceof Date ? message.timestamp : new Date(message.timestamp),
});

const compactMessageForPersistence = (message: Message): PersistedMessage => ({
  id: message.id,
  role: message.role,
  content:
    message.content.length > MAX_PERSISTED_CONTENT_CHARS
      ? `${message.content.slice(0, MAX_PERSISTED_CONTENT_CHARS)}...`
      : message.content,
  clarifying_questions: message.clarifying_questions,
  answer_source: message.answer_source,
  answer_confidence: message.answer_confidence,
  tool_approval_required: message.tool_approval_required,
  tool_approval_message: message.tool_approval_message,
  timestamp: message.timestamp,
});

interface ChatState {
  // Messages
  messages: Message[];
  conversationId: string | null;
  frontendSessionId: string;

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
  appendToLastMessage: (content: string) => void;
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
  // Initial state
  messages: [],
  conversationId: null,
  frontendSessionId: createSessionId(),
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
          id: createMessageId(),
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
      frontendSessionId: createSessionId(),
      currentAgent: null,
      agentStatus: "idle",
      agentMessage: null,
      agentError: null,
      agentHistory: [],
    }),

  addChatResponse: (query, response) =>
    set((state) => {
      const userMessage: Message = {
        id: createMessageId(),
        role: "user",
        content: query,
        timestamp: new Date(),
      };

      const assistantMessage: Message = {
        id: createMessageId(),
        role: "assistant",
        content: response.answer,
        timestamp: new Date(),
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
      };

      return {
        messages: [...state.messages, userMessage, assistantMessage],
        conversationId: response.conversation_id,
      };
    }),

  appendToLastMessage: (content) =>
    set(() => {
      const messages = [...get().messages];
      const lastMessage = messages[messages.length - 1];
      if (!lastMessage) {
        return { messages };
      }
      messages[messages.length - 1] = {
        ...lastMessage,
        content: `${lastMessage.content}${content}`,
      };
      return { messages };
    }),
    }),
    {
      name: "datachat.chat.session.v1",
      storage: createJSONStorage(() =>
        typeof window === "undefined" ? noopStorage : window.localStorage
      ),
      partialize: (state) => ({
        messages: (
          state.isLoading &&
          state.messages.length > 0 &&
          state.messages[state.messages.length - 1]?.role === "assistant"
            ? state.messages.slice(0, -1)
            : state.messages
        )
          .slice(-MAX_PERSISTED_MESSAGES)
          .map((message) => compactMessageForPersistence(message)),
        conversationId: state.conversationId,
        frontendSessionId: state.frontendSessionId,
      }),
      onRehydrateStorage: () => (state) => {
        if (!state) return;
        state.messages = state.messages.map((message) => reviveMessage(message as PersistedMessage));
        if (!state.frontendSessionId) {
          state.frontendSessionId = createSessionId();
        }
      },
    }
  )
);
