/**
 * API Client for DataChat Backend
 *
 * Provides REST API client and WebSocket connection for real-time updates.
 */

// Type definitions matching backend API models
export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  target_database?: string;
  conversation_history?: ChatMessage[];
}

export interface DataSource {
  datapoint_id: string;
  type: string;
  name: string;
  relevance_score: number;
}

export interface ChatMetrics {
  total_latency_ms: number;
  agent_timings: Record<string, number>;
  llm_calls: number;
  retry_count: number;
}

export interface ChatResponse {
  answer: string;
  sql: string | null;
  data: Record<string, unknown[]> | null;
  visualization_hint: string | null;
  sources: DataSource[];
  metrics: ChatMetrics;
  conversation_id: string;
}

export interface AgentUpdate {
  current_agent: string;
  status: "running" | "completed" | "error";
  message?: string;
  error?: string;
}

export interface SetupStep {
  step: string;
  title: string;
  description: string;
  action: string;
}

export interface SystemStatusResponse {
  is_initialized: boolean;
  has_databases: boolean;
  has_datapoints: boolean;
  setup_required: SetupStep[];
}

export interface SystemInitializeRequest {
  database_url?: string;
  auto_profile: boolean;
}

export interface SystemInitializeResponse {
  message: string;
  is_initialized: boolean;
  has_databases: boolean;
  has_datapoints: boolean;
  setup_required: SetupStep[];
}

export interface StreamChatHandlers {
  onOpen?: () => void;
  onClose?: () => void;
  onAgentUpdate?: (update: AgentUpdate) => void;
  onAnswerChunk?: (chunk: string) => void;
  onComplete?: (response: ChatResponse) => void;
  onError?: (message: string) => void;
  onSystemNotInitialized?: (steps: SetupStep[], message: string) => void;
}

export interface HealthResponse {
  status: string;
  version: string;
  timestamp: string;
}

export interface DatabaseConnection {
  connection_id: string;
  name: string;
  database_url: string;
  database_type: string;
  is_active: boolean;
  is_default: boolean;
  tags: string[];
  description?: string | null;
  created_at: string;
  last_profiled?: string | null;
  datapoint_count: number;
}

export interface DatabaseConnectionCreate {
  name: string;
  database_url: string;
  database_type: string;
  tags: string[];
  description?: string;
  is_default?: boolean;
}

export interface ProfilingProgress {
  total_tables: number;
  tables_completed: number;
}

export interface ProfilingJob {
  job_id: string;
  connection_id: string;
  status: string;
  progress?: ProfilingProgress | null;
  error?: string | null;
  profile_id?: string | null;
}

export interface PendingDataPoint {
  pending_id: string;
  profile_id: string;
  datapoint: Record<string, unknown>;
  confidence: number;
  status: string;
  review_note?: string | null;
}

/**
 * API Client Configuration
 */
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_BASE_URL =
  process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

/**
 * REST API Client
 */
export class DataChatAPI {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Send a chat message and get response
   */
  async chat(request: ChatRequest): Promise<ChatResponse> {
    const response = await fetch(`${this.baseUrl}/api/v1/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        error: "Unknown error",
        message: response.statusText,
      }));
      throw new Error(error.message || `HTTP ${response.status}`);
    }

    return response.json();
  }

  /**
   * Check API health
   */
  async health(): Promise<HealthResponse> {
    const response = await fetch(`${this.baseUrl}/api/v1/health`);

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Check API readiness
   */
  async ready(): Promise<HealthResponse> {
    const response = await fetch(`${this.baseUrl}/api/v1/ready`);

    if (!response.ok) {
      throw new Error(`Readiness check failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Check system initialization status
   */
  async systemStatus(): Promise<SystemStatusResponse> {
    const response = await fetch(`${this.baseUrl}/api/v1/system/status`);

    if (!response.ok) {
      throw new Error(`System status failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Initialize system with database connection
   */
  async systemInitialize(
    payload: SystemInitializeRequest
  ): Promise<SystemInitializeResponse> {
    const response = await fetch(`${this.baseUrl}/api/v1/system/initialize`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        message: response.statusText,
      }));
      throw new Error(error.message || `HTTP ${response.status}`);
    }

    return response.json();
  }

  async listDatabases(): Promise<DatabaseConnection[]> {
    const response = await fetch(`${this.baseUrl}/api/v1/databases`);
    if (!response.ok) {
      throw new Error(`List databases failed: ${response.statusText}`);
    }
    return response.json();
  }

  async createDatabase(
    payload: DatabaseConnectionCreate
  ): Promise<DatabaseConnection> {
    const response = await fetch(`${this.baseUrl}/api/v1/databases`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: response.statusText }));
      throw new Error(error.message || `HTTP ${response.status}`);
    }
    return response.json();
  }

  async setDefaultDatabase(connectionId: string): Promise<void> {
    const response = await fetch(
      `${this.baseUrl}/api/v1/databases/${connectionId}/default`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ is_default: true }),
      }
    );
    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: response.statusText }));
      throw new Error(error.message || `HTTP ${response.status}`);
    }
  }

  async deleteDatabase(connectionId: string): Promise<void> {
    const response = await fetch(
      `${this.baseUrl}/api/v1/databases/${connectionId}`,
      { method: "DELETE" }
    );
    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: response.statusText }));
      throw new Error(error.message || `HTTP ${response.status}`);
    }
  }

  async startProfiling(
    connectionId: string,
    payload: { sample_size: number; tables?: string[] }
  ): Promise<ProfilingJob> {
    const response = await fetch(
      `${this.baseUrl}/api/v1/databases/${connectionId}/profile`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    );
    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: response.statusText }));
      throw new Error(error.message || `HTTP ${response.status}`);
    }
    return response.json();
  }

  async getProfilingJob(jobId: string): Promise<ProfilingJob> {
    const response = await fetch(`${this.baseUrl}/api/v1/profiling/jobs/${jobId}`);
    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: response.statusText }));
      throw new Error(error.message || `HTTP ${response.status}`);
    }
    return response.json();
  }

  async generateDatapoints(profileId: string): Promise<PendingDataPoint[]> {
    const response = await fetch(`${this.baseUrl}/api/v1/datapoints/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile_id: profileId }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: response.statusText }));
      throw new Error(error.message || `HTTP ${response.status}`);
    }
    const data = await response.json();
    return data.pending || [];
  }

  async listPendingDatapoints(): Promise<PendingDataPoint[]> {
    const response = await fetch(`${this.baseUrl}/api/v1/datapoints/pending`);
    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: response.statusText }));
      throw new Error(error.message || `HTTP ${response.status}`);
    }
    const data = await response.json();
    return data.pending || [];
  }

  async approvePendingDatapoint(pendingId: string): Promise<PendingDataPoint> {
    const response = await fetch(
      `${this.baseUrl}/api/v1/datapoints/pending/${pendingId}/approve`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      }
    );
    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: response.statusText }));
      throw new Error(error.message || `HTTP ${response.status}`);
    }
    return response.json();
  }

  async rejectPendingDatapoint(pendingId: string, reviewNote?: string): Promise<PendingDataPoint> {
    const response = await fetch(
      `${this.baseUrl}/api/v1/datapoints/pending/${pendingId}/reject`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ review_note: reviewNote || null }),
      }
    );
    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: response.statusText }));
      throw new Error(error.message || `HTTP ${response.status}`);
    }
    return response.json();
  }

  async bulkApproveDatapoints(): Promise<PendingDataPoint[]> {
    const response = await fetch(
      `${this.baseUrl}/api/v1/datapoints/pending/bulk-approve`,
      { method: "POST" }
    );
    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: response.statusText }));
      throw new Error(error.message || `HTTP ${response.status}`);
    }
    const data = await response.json();
    return data.pending || [];
  }
}

/**
 * WebSocket Client for Real-Time Agent Updates
 */
export class DataChatWebSocket {
  private ws: WebSocket | null = null;
  private baseUrl: string;

  constructor(baseUrl: string = WS_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Start a streaming chat session over WebSocket.
   */
  streamChat(request: ChatRequest, handlers: StreamChatHandlers): void {
    try {
      this.ws = new WebSocket(`${this.baseUrl}/ws/chat`);

      this.ws.onopen = () => {
        this.ws?.send(JSON.stringify(request));
        handlers.onOpen?.();
      };

      this.ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as {
            event?: string;
            agent?: string;
            message?: string;
            error?: string;
            chunk?: string;
          };

          if (payload.event === "agent_start" && payload.agent) {
            handlers.onAgentUpdate?.({
              current_agent: payload.agent,
              status: "running",
              message: payload.message,
            });
            return;
          }

          if (payload.event === "agent_complete" && payload.agent) {
            handlers.onAgentUpdate?.({
              current_agent: payload.agent,
              status: "completed",
              message: payload.message,
            });
            return;
          }

          if (payload.event === "answer_chunk" && payload.chunk) {
            handlers.onAnswerChunk?.(payload.chunk);
            return;
          }

          if (payload.event === "complete") {
            handlers.onComplete?.(payload as ChatResponse);
            this.disconnect();
            return;
          }

          if (payload.event === "error") {
            if (payload.error === "system_not_initialized") {
              handlers.onSystemNotInitialized?.(
                (payload as { setup_steps?: SetupStep[] }).setup_steps || [],
                payload.message || "DataChat requires setup."
              );
            } else {
              handlers.onError?.(payload.message || "WebSocket error");
            }
            this.disconnect();
          }
        } catch (error) {
          console.error("Failed to parse WebSocket message:", error);
        }
      };

      this.ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        handlers.onError?.("WebSocket connection error");
      };

      this.ws.onclose = () => {
        handlers.onClose?.();
      };
    } catch (error) {
      console.error("Failed to create WebSocket:", error);
      handlers.onError?.("Failed to create WebSocket");
    }
  }

  /**
   * Send a message through WebSocket
   */
  send(data: unknown): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.error("WebSocket is not connected");
    }
  }

  /**
   * Disconnect WebSocket
   */
  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  /**
   * Get connection status
   */
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}

// Export singleton instances
export const apiClient = new DataChatAPI();
export const wsClient = new DataChatWebSocket();
