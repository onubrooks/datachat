"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  DataChatAPI,
  DataPointSummary,
  DatabaseConnection,
  GenerationJob,
  PendingDataPoint,
  ProfilingJob,
  SyncStatusResponse,
} from "@/lib/api";

const api = new DataChatAPI();
const ENV_CONNECTION_ID = "00000000-0000-0000-0000-00000000dada";

const isEnvironmentConnection = (connection: DatabaseConnection): boolean =>
  String(connection.connection_id) === ENV_CONNECTION_ID ||
  (connection.tags || []).includes("env");

const inferDatabaseTypeFromUrl = (value: string): string | null => {
  const normalized = value.trim().toLowerCase();
  if (normalized.startsWith("postgresql://") || normalized.startsWith("postgres://")) {
    return "postgresql";
  }
  if (normalized.startsWith("mysql://")) {
    return "mysql";
  }
  if (normalized.startsWith("clickhouse://")) {
    return "clickhouse";
  }
  return null;
};

type QuickstartStatus = "done" | "ready" | "blocked";

type JobsState = {
  generating: boolean;
  bulkApproving: boolean;
  syncing: boolean;
  resetting: boolean;
  toolProfiling: boolean;
  qualityReport: boolean;
  editorLoading: boolean;
  editorSaving: boolean;
  editorDeleting: boolean;
  editSaving: boolean;
};

const DATAPOINT_EDITOR_TEMPLATE = `{
  "datapoint_id": "query_top_customers_001",
  "type": "Query",
  "name": "Top customers by revenue",
  "owner": "data-team@example.com",
  "tags": ["manual", "query-template"],
  "metadata": {},
  "description": "Top customers by completed revenue over a configurable date range.",
  "sql_template": "SELECT customer_id, SUM(amount) AS revenue FROM public.transactions WHERE status = 'completed' AND transaction_time >= {start_time} GROUP BY customer_id ORDER BY revenue DESC LIMIT {limit}",
  "parameters": {
    "start_time": {
      "type": "timestamp",
      "required": true,
      "description": "Start timestamp (inclusive)."
    },
    "limit": {
      "type": "integer",
      "required": false,
      "default": 20,
      "description": "Maximum rows to return."
    }
  },
  "related_tables": ["public.transactions"]
}`;

export function DatabaseManager() {
  const queryClient = useQueryClient();
  const [syncError, setSyncError] = useState<string | null>(null);
  const [syncScopeMode, setSyncScopeMode] = useState<"auto" | "global" | "database">("auto");
  const [syncScopeConnectionId, setSyncScopeConnectionId] = useState<string | null>(null);
  const [expandedPendingId, setExpandedPendingId] = useState<string | null>(null);
  const [pendingEdits, setPendingEdits] = useState<Record<string, string>>({});
  const [approvingId, setApprovingId] = useState<string | null>(null);
  const [jobs, setJobs] = useState<JobsState>({
    generating: false,
    bulkApproving: false,
    syncing: false,
    resetting: false,
    toolProfiling: false,
    qualityReport: false,
    editorLoading: false,
    editorSaving: false,
    editorDeleting: false,
    editSaving: false,
  });
  const [selectedConnectionId, setSelectedConnectionId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const noticeTimerRef = useRef<number | null>(null);
  const [toolProfileMessage, setToolProfileMessage] = useState<string | null>(null);
  const [toolProfileError, setToolProfileError] = useState<string | null>(null);
  const [toolApprovalOpen, setToolApprovalOpen] = useState(false);
  const [qualityReport, setQualityReport] = useState<Record<string, unknown> | null>(null);
  const [qualityError, setQualityError] = useState<string | null>(null);
  const [editorDatapointId, setEditorDatapointId] = useState("");
  const [editorPayload, setEditorPayload] = useState(DATAPOINT_EDITOR_TEMPLATE);
  const [editorNotice, setEditorNotice] = useState<string | null>(null);
  const [editorError, setEditorError] = useState<string | null>(null);

  const [name, setName] = useState("");
  const [databaseUrl, setDatabaseUrl] = useState("");
  const [databaseType, setDatabaseType] = useState("postgresql");
  const [description, setDescription] = useState("");
  const [isDefault, setIsDefault] = useState(false);
  const [editingConnectionId, setEditingConnectionId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState("");
  const [editingDatabaseUrl, setEditingDatabaseUrl] = useState("");
  const [editingDatabaseType, setEditingDatabaseType] = useState("postgresql");
  const [editingDescription, setEditingDescription] = useState("");
  const [selectedTables, setSelectedTables] = useState<string[]>([]);
  const [depth, setDepth] = useState("metrics_basic");
  const addConnectionRef = useRef<HTMLDivElement | null>(null);
  const previousGenerationStateRef = useRef<{ jobId: string; status: string } | null>(null);

  const dedupeApproved = useCallback((items: DataPointSummary[]) => {
    const seen = new Map<string, DataPointSummary>();
    for (const item of items) {
      const key = String(item.datapoint_id);
      if (!seen.has(key)) {
        seen.set(key, item);
      }
    }
    return Array.from(seen.values());
  }, []);

  const mapPendingToSummary = useCallback(
    (items: PendingDataPoint[]): DataPointSummary[] =>
      dedupeApproved(
        items.map((item) => ({
          datapoint_id: String(item.datapoint.datapoint_id || item.pending_id),
          type: String(item.datapoint.type || "Unknown"),
          name: item.datapoint.name ? String(item.datapoint.name) : null,
        }))
      ),
    [dedupeApproved]
  );

  const showNotice = (message: string) => {
    setNotice(message);
    if (noticeTimerRef.current) {
      window.clearTimeout(noticeTimerRef.current);
    }
    noticeTimerRef.current = window.setTimeout(() => {
      setNotice(null);
      noticeTimerRef.current = null;
    }, 5000);
  };

  const setJobState = useCallback((job: keyof JobsState, value: boolean) => {
    setJobs((current) => ({ ...current, [job]: value }));
  }, []);

  const emitEntryEvent = useCallback(
    async (
      step: string,
      status: "started" | "completed" | "failed" | "skipped",
      metadata?: Record<string, unknown>
    ) => {
      try {
        await api.emitEntryEvent({
          flow: "phase1_4_quickstart_ui",
          step,
          status,
          source: "ui",
          metadata,
        });
      } catch {
        // Telemetry is best-effort.
      }
    },
    []
  );

  useEffect(() => {
    return () => {
      if (noticeTimerRef.current) {
        window.clearTimeout(noticeTimerRef.current);
      }
    };
  }, []);

  const connectionsQuery = useQuery({
    queryKey: ["db-connections"],
    queryFn: () => api.listDatabases(),
  });
  const connections = useMemo(
    () => connectionsQuery.data ?? [],
    [connectionsQuery.data]
  );

  const selectedConnectionPreview = connections.find(
    (connection) => connection.connection_id === selectedConnectionId
  );
  const selectedManagedConnectionId =
    selectedConnectionPreview && !isEnvironmentConnection(selectedConnectionPreview)
      ? selectedConnectionPreview.connection_id
      : null;

  const profilingJobQuery = useQuery({
    queryKey: ["profiling-job-latest", selectedManagedConnectionId],
    queryFn: () => api.getLatestProfilingJob(selectedManagedConnectionId as string),
    enabled: Boolean(selectedManagedConnectionId),
    refetchInterval: (query) => {
      const job = query.state.data as ProfilingJob | null | undefined;
      if (!job || job.status === "completed" || job.status === "failed") {
        return false;
      }
      return 3000;
    },
  });
  const job = profilingJobQuery.data ?? null;

  const pendingQuery = useQuery({
    queryKey: ["pending-datapoints", selectedManagedConnectionId],
    queryFn: () =>
      api.listPendingDatapoints({
        statusFilter: "pending",
        connectionId: selectedManagedConnectionId,
      }),
    enabled: Boolean(selectedManagedConnectionId),
  });
  const pending = pendingQuery.data ?? [];

  const approvedPendingQuery = useQuery({
    queryKey: ["approved-datapoints", selectedManagedConnectionId],
    queryFn: () =>
      api.listPendingDatapoints({
        statusFilter: "approved",
        connectionId: selectedManagedConnectionId,
      }),
    enabled: Boolean(selectedManagedConnectionId),
  });
  const approved = useMemo(
    () => mapPendingToSummary(approvedPendingQuery.data ?? []),
    [approvedPendingQuery.data, mapPendingToSummary]
  );

  const syncStatusQuery = useQuery({
    queryKey: ["sync-status"],
    queryFn: () => api.getSyncStatus(),
    refetchInterval: (query) => {
      const status = query.state.data as SyncStatusResponse | null | undefined;
      return status?.status === "running" ? 3000 : false;
    },
  });
  const syncStatus = syncStatusQuery.data ?? null;

  const profileTablesQuery = useQuery({
    queryKey: ["profile-tables", job?.profile_id],
    queryFn: () => api.listProfileTables(job?.profile_id as string),
    enabled: Boolean(job?.profile_id),
  });
  const profileTables = useMemo(
    () => profileTablesQuery.data ?? [],
    [profileTablesQuery.data]
  );

  const generationJobQuery = useQuery({
    queryKey: ["generation-latest", job?.profile_id],
    queryFn: () => api.getLatestGenerationJob(job?.profile_id as string),
    enabled: Boolean(job?.profile_id),
    refetchInterval: (query) => {
      const generation = query.state.data as GenerationJob | null | undefined;
      if (!generation || generation.status === "completed" || generation.status === "failed") {
        return false;
      }
      return 5000;
    },
  });
  const generationJob = generationJobQuery.data ?? null;

  const isLoading =
    connectionsQuery.isLoading ||
    connectionsQuery.isFetching ||
    profilingJobQuery.isFetching ||
    pendingQuery.isFetching ||
    approvedPendingQuery.isFetching ||
    syncStatusQuery.isFetching;

  const invalidateManagerQueries = useCallback(async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["db-connections"] }),
      queryClient.invalidateQueries({ queryKey: ["profiling-job-latest"] }),
      queryClient.invalidateQueries({ queryKey: ["pending-datapoints"] }),
      queryClient.invalidateQueries({ queryKey: ["approved-datapoints"] }),
      queryClient.invalidateQueries({ queryKey: ["sync-status"] }),
      queryClient.invalidateQueries({ queryKey: ["profile-tables"] }),
      queryClient.invalidateQueries({ queryKey: ["generation-latest"] }),
    ]);
  }, [queryClient]);

  useEffect(() => {
    const current = generationJob
      ? { jobId: generationJob.job_id, status: generationJob.status }
      : null;
    const previous = previousGenerationStateRef.current;

    if (previous && current && previous.jobId === current.jobId) {
      const nowTerminal = current.status === "completed" || current.status === "failed";
      const wasTerminal = previous.status === "completed" || previous.status === "failed";
      if (nowTerminal && !wasTerminal) {
        void Promise.all([
          queryClient.invalidateQueries({ queryKey: ["pending-datapoints"] }),
          queryClient.invalidateQueries({ queryKey: ["approved-datapoints"] }),
        ]);
      }
    }

    previousGenerationStateRef.current = current;
  }, [generationJob, queryClient]);

  const handleToolProfile = async () => {
    setToolApprovalOpen(true);
  };

  const handleToolProfileApprove = async () => {
    setToolProfileError(null);
    setToolProfileMessage(null);
    setJobState("toolProfiling", true);
    try {
      const response = await api.executeTool({
        name: "profile_and_generate_datapoints",
        approved: true,
        arguments: {
          depth,
          batch_size: 10,
          max_tables: selectedTables.length ? selectedTables.length : null,
        },
      });
      const result = response.result || {};
      setToolProfileMessage(
        `Profiling complete. Pending DataPoints created: ${
          (result as Record<string, unknown>).pending_count ?? 0
        }.`
      );
      await invalidateManagerQueries();
    } catch (err) {
      setToolProfileError((err as Error).message);
    } finally {
      setJobState("toolProfiling", false);
      setToolApprovalOpen(false);
    }
  };

  const handleQualityReport = async () => {
    setQualityError(null);
    setJobState("qualityReport", true);
    try {
      const response = await api.executeTool({
        name: "datapoint_quality_report",
        arguments: { limit: 10 },
      });
      setQualityReport(response.result || {});
    } catch (err) {
      setQualityError((err as Error).message);
    } finally {
      setJobState("qualityReport", false);
    }
  };

  const selectedCount = selectedTables.length;
  const hasSelection = selectedCount > 0;
  const tableSelectionLabel = useMemo(() => {
    if (!profileTables.length) return "No tables found";
    if (!hasSelection) return "Select tables to generate metrics";
    return `${selectedCount} table(s) selected`;
  }, [profileTables.length, hasSelection, selectedCount]);

  useEffect(() => {
    if (!connections.length) {
      if (selectedConnectionId !== null) {
        setSelectedConnectionId(null);
      }
      return;
    }
    if (selectedConnectionId && connections.some((item) => item.connection_id === selectedConnectionId)) {
      return;
    }
    const defaultConnection = connections.find((item) => item.is_default) || connections[0];
    setSelectedConnectionId(defaultConnection.connection_id);
  }, [connections, selectedConnectionId]);

  useEffect(() => {
    if (!connections.length) {
      setSyncScopeConnectionId(null);
      return;
    }
    if (
      syncScopeConnectionId &&
      connections.some((connection) => connection.connection_id === syncScopeConnectionId)
    ) {
      return;
    }
    const defaultConnection =
      connections.find((connection) => connection.is_default) || connections[0];
    setSyncScopeConnectionId(defaultConnection.connection_id);
  }, [connections, syncScopeConnectionId]);

  useEffect(() => {
    if (!profileTables.length) {
      setSelectedTables([]);
      return;
    }
    setSelectedTables((current) => {
      const preserved = current.filter((table) => profileTables.includes(table));
      if (preserved.length > 0) {
        return preserved;
      }
      return profileTables.slice(0, Math.min(10, profileTables.length));
    });
  }, [profileTables]);

  useEffect(() => {
    const queryErrors: Array<unknown> = [
      connectionsQuery.error,
      profilingJobQuery.error,
      pendingQuery.error,
      approvedPendingQuery.error,
      profileTablesQuery.error,
      generationJobQuery.error,
    ];
    for (const candidate of queryErrors) {
      if (!candidate) continue;
      setError(candidate instanceof Error ? candidate.message : String(candidate));
      return;
    }
  }, [
    approvedPendingQuery.error,
    connectionsQuery.error,
    generationJobQuery.error,
    pendingQuery.error,
    profileTablesQuery.error,
    profilingJobQuery.error,
  ]);

  useEffect(() => {
    if (!syncStatusQuery.error) return;
    setSyncError(
      syncStatusQuery.error instanceof Error
        ? syncStatusQuery.error.message
        : String(syncStatusQuery.error)
    );
  }, [syncStatusQuery.error]);

  const handleCreate = async () => {
    setError(null);
    await emitEntryEvent("create_connection", "started", {
      database_type: databaseType,
      has_description: Boolean(description.trim()),
    });
    try {
      await api.createDatabase({
        name,
        database_url: databaseUrl,
        database_type: databaseType,
        tags: ["managed"],
        description: description || undefined,
        is_default: isDefault,
      });
      setName("");
      setDatabaseUrl("");
      setDatabaseType("postgresql");
      setDescription("");
      setIsDefault(false);
      await invalidateManagerQueries();
      await emitEntryEvent("create_connection", "completed", {
        database_type: databaseType,
      });
    } catch (err) {
      setError((err as Error).message);
      await emitEntryEvent("create_connection", "failed", {
        database_type: databaseType,
        error: (err as Error).message,
      });
    }
  };

  const handleProfile = async (connectionId: string): Promise<boolean> => {
    setError(null);
    if (connectionId === ENV_CONNECTION_ID) {
      setError(
        "Environment Database uses DATABASE_URL and cannot be profiled from Database Manager."
      );
      return false;
    }
    try {
      setSelectedConnectionId(connectionId);
      const started = await api.startProfiling(connectionId, {
        sample_size: 100,
      });
      queryClient.setQueryData(["profiling-job-latest", connectionId], started);
      await invalidateManagerQueries();
      return true;
    } catch (err) {
      setError((err as Error).message);
      return false;
    }
  };

  const handleStartEdit = (connection: DatabaseConnection) => {
    if (isEnvironmentConnection(connection)) {
      setError(
        "Environment Database uses DATABASE_URL and cannot be edited from Database Manager."
      );
      return;
    }
    setEditingConnectionId(connection.connection_id);
    setEditingName(connection.name);
    // Secret URLs are masked in API responses; leave empty unless user provides a new URL.
    setEditingDatabaseUrl("");
    setEditingDatabaseType(connection.database_type);
    setEditingDescription(connection.description || "");
  };

  const handleCancelEdit = () => {
    setEditingConnectionId(null);
    setEditingName("");
    setEditingDatabaseUrl("");
    setEditingDatabaseType("postgresql");
    setEditingDescription("");
  };

  const handleSaveEdit = async () => {
    if (!editingConnectionId) return;
    setError(null);
    setJobState("editSaving", true);
    try {
      const payload: {
        name?: string;
        database_url?: string;
        database_type?: string;
        description?: string | null;
      } = {};
      if (editingName.trim()) {
        payload.name = editingName.trim();
      }
      if (editingDescription.trim()) {
        payload.description = editingDescription.trim();
      } else {
        payload.description = null;
      }
      if (editingDatabaseUrl.trim()) {
        payload.database_url = editingDatabaseUrl.trim();
        payload.database_type = editingDatabaseType;
      }
      await api.updateDatabase(editingConnectionId, payload);
      showNotice("Connection updated successfully.");
      handleCancelEdit();
      await invalidateManagerQueries();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setJobState("editSaving", false);
    }
  };

  const handleGenerate = async (): Promise<boolean> => {
    if (!job?.profile_id) return false;
    setError(null);
    if (generationJob?.status === "completed") {
      const confirmReplace = confirm(
        "Regenerate DataPoints and replace pending drafts for this profile?"
      );
      if (!confirmReplace) {
        return false;
      }
    }
    setJobState("generating", true);
    try {
      const generation = await api.startDatapointGeneration({
        profile_id: job.profile_id,
        tables: selectedTables,
        depth,
        batch_size: 10,
        max_tables: selectedTables.length || null,
        max_metrics_per_table: 3,
        replace_existing: true,
      });
      queryClient.setQueryData(["generation-latest", job.profile_id], generation);
      await invalidateManagerQueries();
      return true;
    } catch (err) {
      setError((err as Error).message);
      return false;
    } finally {
      setJobState("generating", false);
    }
  };

  const parseEditedDatapoint = (pendingId: string) => {
    const raw = pendingEdits[pendingId];
    if (!raw) return null;
    try {
      return JSON.parse(raw) as Record<string, unknown>;
    } catch (err) {
      setError(`Invalid JSON for ${pendingId}: ${(err as Error).message}`);
      return null;
    }
  };

  const handleApprove = async (pendingId: string) => {
    setError(null);
    setApprovingId(pendingId);
    const editedDatapoint = parseEditedDatapoint(pendingId);
    if (pendingEdits[pendingId] && !editedDatapoint) {
      setApprovingId(null);
      return;
    }
    try {
      await api.approvePendingDatapoint(
        pendingId,
        editedDatapoint || undefined
      );
      showNotice(
        "Approved. Existing DataPoints for the same table were replaced."
      );
      await invalidateManagerQueries();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setApprovingId(null);
    }
  };

  const handleReject = async (pendingId: string) => {
    setError(null);
    try {
      await api.rejectPendingDatapoint(pendingId, "Rejected via UI");
      await invalidateManagerQueries();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const handleBulkApprove = async (): Promise<boolean> => {
    setError(null);
    setJobState("bulkApproving", true);
    try {
      const scopeConnectionId =
        selectedConnection && !isEnvironmentConnection(selectedConnection)
          ? selectedConnection.connection_id
          : null;
      const approved = await api.bulkApproveDatapoints(scopeConnectionId);
      if (approved.length) {
        showNotice(
          `Approved ${approved.length} DataPoints for ${selectedConnection?.name || "selected source"}. Existing DataPoints for the same tables were replaced.`
        );
      }
      await invalidateManagerQueries();
      return true;
    } catch (err) {
      setError((err as Error).message);
      return false;
    } finally {
      setJobState("bulkApproving", false);
    }
  };

  const handleSync = async (): Promise<boolean> => {
    setSyncError(null);
    setJobState("syncing", true);
    try {
      if (syncScopeMode === "database" && !syncScopeConnectionId) {
        throw new Error("Select a connection for database-scoped sync.");
      }
      await api.triggerSync({
        scope: syncScopeMode,
        connection_id: syncScopeMode === "database" ? syncScopeConnectionId : null,
      });
      await syncStatusQuery.refetch();
      await invalidateManagerQueries();
      return true;
    } catch (err) {
      setSyncError((err as Error).message);
      return false;
    } finally {
      setJobState("syncing", false);
    }
  };

  const handleSystemReset = async () => {
    if (
      !confirm(
        "Reset will clear system registry/profiling, local vectors, and saved setup config. Continue?"
      )
    ) {
      return;
    }
    setJobState("resetting", true);
    setError(null);
    try {
      await api.systemReset();
      setSelectedConnectionId(null);
      await invalidateManagerQueries();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setJobState("resetting", false);
    }
  };

  const formatTimestamp = (value: string | null) => {
    if (!value) return "—";
    return new Date(value).toLocaleString();
  };

  const togglePendingDetails = (item: PendingDataPoint) => {
    const nextId = expandedPendingId === item.pending_id ? null : item.pending_id;
    setExpandedPendingId(nextId);
    if (nextId && !pendingEdits[item.pending_id]) {
      setPendingEdits((current) => ({
        ...current,
        [item.pending_id]: JSON.stringify(item.datapoint, null, 2),
      }));
    }
  };

  const parseEditorPayload = () => {
    try {
      const parsed = JSON.parse(editorPayload) as Record<string, unknown>;
      const datapointId = String(parsed.datapoint_id || "").trim();
      if (!datapointId) {
        throw new Error("`datapoint_id` is required.");
      }
      return { parsed, datapointId };
    } catch (err) {
      throw new Error(
        err instanceof Error ? `Invalid JSON payload: ${err.message}` : "Invalid JSON payload."
      );
    }
  };

  const handleLoadDatapoint = async () => {
    const datapointId = editorDatapointId.trim();
    if (!datapointId) {
      setEditorError("Enter a DataPoint ID to load.");
      return;
    }
    setJobState("editorLoading", true);
    setEditorError(null);
    setEditorNotice(null);
    try {
      const payload = await api.getDatapoint(datapointId);
      setEditorPayload(JSON.stringify(payload, null, 2));
      setEditorNotice(`Loaded ${datapointId}`);
    } catch (err) {
      setEditorError(err instanceof Error ? err.message : "Failed to load DataPoint.");
    } finally {
      setJobState("editorLoading", false);
    }
  };

  const handleCreateDatapoint = async () => {
    setJobState("editorSaving", true);
    setEditorError(null);
    setEditorNotice(null);
    try {
      const { parsed, datapointId } = parseEditorPayload();
      await api.createDatapoint(parsed);
      setEditorDatapointId(datapointId);
      setEditorNotice(`Created ${datapointId}`);
      await invalidateManagerQueries();
    } catch (err) {
      setEditorError(err instanceof Error ? err.message : "Failed to create DataPoint.");
    } finally {
      setJobState("editorSaving", false);
    }
  };

  const handleUpdateDatapoint = async () => {
    setJobState("editorSaving", true);
    setEditorError(null);
    setEditorNotice(null);
    try {
      const { parsed, datapointId } = parseEditorPayload();
      await api.updateDatapoint(datapointId, parsed);
      setEditorDatapointId(datapointId);
      setEditorNotice(`Updated ${datapointId}`);
      await invalidateManagerQueries();
    } catch (err) {
      setEditorError(err instanceof Error ? err.message : "Failed to update DataPoint.");
    } finally {
      setJobState("editorSaving", false);
    }
  };

  const handleDeleteDatapoint = async () => {
    const datapointId = editorDatapointId.trim();
    if (!datapointId) {
      setEditorError("Enter a DataPoint ID to delete.");
      return;
    }
    if (!confirm(`Delete managed DataPoint ${datapointId}?`)) {
      return;
    }
    setJobState("editorDeleting", true);
    setEditorError(null);
    setEditorNotice(null);
    try {
      await api.deleteDatapoint(datapointId);
      setEditorNotice(`Deleted ${datapointId}`);
      await invalidateManagerQueries();
    } catch (err) {
      setEditorError(err instanceof Error ? err.message : "Failed to delete DataPoint.");
    } finally {
      setJobState("editorDeleting", false);
    }
  };

  const selectedConnection = connections.find(
    (connection) => connection.connection_id === selectedConnectionId
  );
  const managedConnections = connections.filter((connection) => !isEnvironmentConnection(connection));
  const quickstartConnection =
    (selectedConnection && !isEnvironmentConnection(selectedConnection)
      ? selectedConnection
      : managedConnections[0]) || null;
  const hasAnyConnection = connections.length > 0;
  const hasProfileCompleted = Boolean(job?.status === "completed" && job.profile_id);
  const hasPendingDatapoints = pending.length > 0;
  const hasApprovedDatapoints = approved.length > 0;
  const hasSynced = syncStatus?.status === "completed";

  const stepStatusIcon = (status: QuickstartStatus): string => {
    if (status === "done") return "✓";
    if (status === "ready") return "•";
    return "○";
  };

  const runQuickstartConnect = async () => {
    await emitEntryEvent("connect_database", "started", { has_connections: hasAnyConnection });
    if (!hasAnyConnection) {
      addConnectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      await emitEntryEvent("connect_database", "completed", { action: "scrolled_to_form" });
      return;
    }
    await emitEntryEvent("connect_database", "completed", { action: "already_connected" });
  };

  const runQuickstartProfile = async () => {
    await emitEntryEvent("profile_database", "started", {
      connection_id: quickstartConnection?.connection_id || null,
    });
    if (!quickstartConnection) {
      setError("Add a managed connection before profiling.");
      await emitEntryEvent("profile_database", "failed", { reason: "no_managed_connection" });
      return;
    }
    const ok = await handleProfile(quickstartConnection.connection_id);
    await emitEntryEvent("profile_database", ok ? "completed" : "failed", {
      connection_id: quickstartConnection.connection_id,
    });
  };

  const runQuickstartGenerate = async () => {
    await emitEntryEvent("generate_datapoints", "started", {
      profile_id: job?.profile_id || null,
    });
    const ok = await handleGenerate();
    await emitEntryEvent("generate_datapoints", ok ? "completed" : "failed", {
      profile_id: job?.profile_id || null,
    });
  };

  const runQuickstartApprove = async () => {
    await emitEntryEvent("approve_pending", "started", { pending_count: pending.length });
    const ok = await handleBulkApprove();
    await emitEntryEvent("approve_pending", ok ? "completed" : "failed", {
      pending_count: pending.length,
    });
  };

  const runQuickstartSync = async () => {
    await emitEntryEvent("sync_datapoints", "started", {
      scope_mode: syncScopeMode,
      scope_connection_id: syncScopeConnectionId,
    });
    const ok = await handleSync();
    await emitEntryEvent("sync_datapoints", ok ? "completed" : "failed", {
      scope_mode: syncScopeMode,
      scope_connection_id: syncScopeConnectionId,
    });
  };

  const quickstartSteps: Array<{
    key: string;
    title: string;
    description: string;
    status: QuickstartStatus;
  }> = [
    {
      key: "connect",
      title: "Connect a database",
      description: hasAnyConnection
        ? "Connection available."
        : "Add a target connection from the form below.",
      status: hasAnyConnection ? "done" : "ready",
    },
    {
      key: "profile",
      title: "Run profiling",
      description: hasProfileCompleted
        ? "Latest profiling job completed."
        : quickstartConnection
          ? `Use ${quickstartConnection.name} for profiling.`
          : "Requires a managed (non-env) connection.",
      status: hasProfileCompleted
        ? "done"
        : quickstartConnection
          ? "ready"
          : "blocked",
    },
    {
      key: "generate",
      title: "Generate DataPoints",
      description: hasPendingDatapoints
        ? "Pending DataPoints ready for review."
        : hasProfileCompleted
          ? "Generate pending DataPoints from the latest profile."
          : "Requires completed profiling first.",
      status: hasPendingDatapoints ? "done" : hasProfileCompleted ? "ready" : "blocked",
    },
    {
      key: "approve",
      title: "Approve pending DataPoints",
      description: hasApprovedDatapoints
        ? "Approved DataPoints available for this source."
        : hasPendingDatapoints
          ? "Bulk approve or review individual drafts."
          : "No pending DataPoints yet.",
      status: hasApprovedDatapoints ? "done" : hasPendingDatapoints ? "ready" : "blocked",
    },
    {
      key: "sync",
      title: "Sync retrieval index",
      description: hasSynced
        ? "Last sync completed."
        : "Sync updates vector and graph retrieval stores.",
      status: hasSynced ? "done" : hasApprovedDatapoints ? "ready" : "blocked",
    },
  ];

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Database Management</h1>
          <p className="text-sm text-muted-foreground">
            Add connections, run profiling, and review generated DataPoints.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button asChild variant="secondary">
            <Link href="/">Back to Chat</Link>
          </Button>
          <Button variant="outline" onClick={handleSystemReset} disabled={jobs.resetting}>
            {jobs.resetting ? "Resetting..." : "Reset System"}
          </Button>
          <Button onClick={invalidateManagerQueries} disabled={isLoading}>
            Refresh
          </Button>
        </div>
      </div>
      <div className="text-xs text-muted-foreground">
        Reset clears system registry/profiling, local vectors, and saved setup config.
        It does not delete target database tables.
      </div>

      {notice && <div className="text-sm text-emerald-600">{notice}</div>}
      {error && <div className="text-sm text-destructive">{error}</div>}

      <Card className="p-4 space-y-4">
        <div className="flex items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold">Quick Start (Phase 1.4)</h2>
            <p className="text-xs text-muted-foreground">
              Guided onboarding over existing setup/profile/sync actions.
            </p>
          </div>
          <Button asChild variant="secondary" size="sm">
            <Link href="/">Open Chat</Link>
          </Button>
        </div>

        <div className="space-y-2 text-xs">
          {quickstartSteps.map((step) => (
            <div
              key={step.key}
              className="flex items-start gap-2 rounded-md border border-border px-3 py-2"
            >
              <span className="mt-0.5 text-muted-foreground">{stepStatusIcon(step.status)}</span>
              <div>
                <div className="font-medium text-foreground">{step.title}</div>
                <div className="text-muted-foreground">{step.description}</div>
              </div>
            </div>
          ))}
        </div>

        <div className="flex flex-wrap gap-2">
          <Button variant="secondary" size="sm" onClick={runQuickstartConnect}>
            1) Connect
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={runQuickstartProfile}
            disabled={!quickstartConnection || hasProfileCompleted}
          >
            2) Profile
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={runQuickstartGenerate}
            disabled={!hasProfileCompleted || hasPendingDatapoints || jobs.generating}
          >
            3) Generate
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={runQuickstartApprove}
            disabled={!hasPendingDatapoints || jobs.bulkApproving}
          >
            4) Approve
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={runQuickstartSync}
            disabled={(!hasApprovedDatapoints && !hasPendingDatapoints) || jobs.syncing}
          >
            5) Sync
          </Button>
        </div>
        <div className="text-xs text-muted-foreground">
          Starter questions: &quot;list all available tables&quot;, &quot;what is total deposits?&quot;,
          &quot;show failed transaction rate by day&quot;.
        </div>
      </Card>

      <Card className="p-4 space-y-4" ref={addConnectionRef}>
        <h2 className="text-sm font-semibold">Add Connection</h2>
        <div className="grid gap-2 md:grid-cols-2">
          <Input
            placeholder="Name"
            value={name}
            onChange={(event) => setName(event.target.value)}
          />
          <Input
            placeholder="postgresql://user:pass@host:5432/db"
            value={databaseUrl}
            onChange={(event) => {
              const nextUrl = event.target.value;
              setDatabaseUrl(nextUrl);
              const inferredType = inferDatabaseTypeFromUrl(nextUrl);
              if (inferredType) {
                setDatabaseType(inferredType);
              }
            }}
          />
          <select
            className="h-10 rounded-md border border-input bg-background px-3 py-2 text-sm"
            value={databaseType}
            onChange={(event) => setDatabaseType(event.target.value)}
            aria-label="Database Type"
          >
            <option value="postgresql">postgresql</option>
            <option value="mysql">mysql</option>
            <option value="clickhouse">clickhouse</option>
          </select>
          <Input
            placeholder="Description (optional)"
            value={description}
            onChange={(event) => setDescription(event.target.value)}
          />
        </div>
        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          <input
            type="checkbox"
            checked={isDefault}
            onChange={(event) => setIsDefault(event.target.checked)}
          />
          Set as default connection
        </label>
        <Button onClick={handleCreate} disabled={isLoading || !name || !databaseUrl}>
          Add Connection
        </Button>
      </Card>

      <Card className="p-4 space-y-4">
        <h2 className="text-sm font-semibold">Connections</h2>
        {connections.length === 0 && (
          <p className="text-sm text-muted-foreground">No connections yet.</p>
        )}
        <div className="space-y-3">
          {connections.map((connection) => (
            <div
              key={connection.connection_id}
              className="flex flex-col gap-2 border-b border-border pb-3"
            >
              {String(connection.connection_id) === ENV_CONNECTION_ID && (
                <div className="text-xs text-muted-foreground">
                  Loaded from <code>DATABASE_URL</code>. Manage this value in your environment
                  file.
                </div>
              )}
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-medium">{connection.name}</div>
                  <div className="text-xs text-muted-foreground">
                    {connection.database_type} · {connection.connection_id}
                  </div>
                </div>
                {connection.is_default && (
                  <span className="text-xs font-semibold text-emerald-600">
                    Default
                  </span>
                )}
              </div>
              <div className="flex flex-wrap gap-2">
                <Button
                  variant="secondary"
                  onClick={() => handleStartEdit(connection)}
                  disabled={isEnvironmentConnection(connection)}
                >
                  Edit
                </Button>
                <Button
                  variant="secondary"
                  onClick={async () => {
                    if (isEnvironmentConnection(connection)) {
                      setError(
                        "Environment Database uses DATABASE_URL and cannot be set as a registry default."
                      );
                      return;
                    }
                    try {
                      await api.setDefaultDatabase(connection.connection_id);
                      await invalidateManagerQueries();
                    } catch (err) {
                      setError((err as Error).message);
                    }
                  }}
                  disabled={isEnvironmentConnection(connection)}
                >
                  Set Default
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => handleProfile(connection.connection_id)}
                  disabled={isEnvironmentConnection(connection)}
                >
                  Profile
                </Button>
                <Button
                  variant="destructive"
                  onClick={async () => {
                    if (isEnvironmentConnection(connection)) {
                      setError(
                        "Environment Database uses DATABASE_URL and cannot be deleted here."
                      );
                      return;
                    }
                    try {
                      await api.deleteDatabase(connection.connection_id);
                      await invalidateManagerQueries();
                    } catch (err) {
                      setError((err as Error).message);
                    }
                  }}
                  disabled={isEnvironmentConnection(connection)}
                >
                  Delete
                </Button>
              </div>
              {editingConnectionId === connection.connection_id && (
                <div className="mt-2 space-y-2 rounded-md border border-border p-3">
                  <div className="grid gap-2 md:grid-cols-2">
                    <Input
                      placeholder="Name"
                      value={editingName}
                      onChange={(event) => setEditingName(event.target.value)}
                    />
                    <Input
                      placeholder="Leave blank to keep current URL"
                      value={editingDatabaseUrl}
                      onChange={(event) => {
                        const nextUrl = event.target.value;
                        setEditingDatabaseUrl(nextUrl);
                        const inferredType = inferDatabaseTypeFromUrl(nextUrl);
                        if (inferredType) {
                          setEditingDatabaseType(inferredType);
                        }
                      }}
                    />
                    <select
                      className="h-10 rounded-md border border-input bg-background px-3 py-2 text-sm"
                      value={editingDatabaseType}
                      onChange={(event) => setEditingDatabaseType(event.target.value)}
                      aria-label="Edit Database Type"
                    >
                      <option value="postgresql">postgresql</option>
                      <option value="mysql">mysql</option>
                      <option value="clickhouse">clickhouse</option>
                    </select>
                    <Input
                      placeholder="Description (optional)"
                      value={editingDescription}
                      onChange={(event) => setEditingDescription(event.target.value)}
                    />
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Button
                      onClick={handleSaveEdit}
                      disabled={
                        jobs.editSaving ||
                        !editingName.trim()
                      }
                    >
                      {jobs.editSaving ? "Saving..." : "Save"}
                    </Button>
                    <Button
                      variant="outline"
                      onClick={handleCancelEdit}
                      disabled={jobs.editSaving}
                    >
                      Cancel
                    </Button>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </Card>

      <Card className="p-4 space-y-3">
        <h2 className="text-sm font-semibold">Active Data Source</h2>
        <p className="text-xs text-muted-foreground">
          Pending and approved profiling DataPoints are scoped to this connection.
        </p>
        <select
          className="h-10 rounded-md border border-input bg-background px-3 py-2 text-sm"
          value={selectedConnectionId || ""}
          onChange={(event) => setSelectedConnectionId(event.target.value || null)}
          aria-label="Active Data Source"
        >
          {connections.map((connection) => (
            <option key={connection.connection_id} value={connection.connection_id}>
              {connection.name} ({connection.database_type})
              {connection.is_default ? " · default" : ""}
              {isEnvironmentConnection(connection) ? " · env" : ""}
            </option>
          ))}
        </select>
        {!selectedConnection && (
          <p className="text-xs text-muted-foreground">
            No connection selected.
          </p>
        )}
        {selectedConnection && isEnvironmentConnection(selectedConnection) && (
          <p className="text-xs text-muted-foreground">
            Environment connection is chat-only here. Profiling draft DataPoints are available
            on managed connections.
          </p>
        )}
      </Card>

      <Card className="p-4 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Sync Status</h2>
          <Button
            variant="secondary"
            onClick={handleSync}
            disabled={syncStatus?.status === "running" || jobs.syncing}
          >
            {jobs.syncing
              ? "Syncing..."
              : syncStatus?.status === "completed"
                ? "Sync Again"
                : "Sync Now"}
          </Button>
        </div>
        <div className="grid gap-2 md:grid-cols-2">
          <select
            className="h-10 rounded-md border border-input bg-background px-3 py-2 text-sm"
            value={syncScopeMode}
            onChange={(event) =>
              setSyncScopeMode(event.target.value as "auto" | "global" | "database")
            }
            aria-label="Sync Scope"
          >
            <option value="auto">Scope: keep file metadata (auto)</option>
            <option value="database">Scope: selected database</option>
            <option value="global">Scope: global/shared</option>
          </select>
          <select
            className="h-10 rounded-md border border-input bg-background px-3 py-2 text-sm"
            value={syncScopeConnectionId || ""}
            onChange={(event) => setSyncScopeConnectionId(event.target.value || null)}
            aria-label="Sync Scope Connection"
            disabled={syncScopeMode !== "database"}
          >
            {connections.map((connection) => (
              <option key={connection.connection_id} value={connection.connection_id}>
                {connection.name} ({connection.database_type})
                {connection.is_default ? " · default" : ""}
              </option>
            ))}
          </select>
        </div>
        <div className="text-xs text-muted-foreground">
          Database scope stamps synced DataPoints with a connection id.
          Global scope marks them shared across all databases.
        </div>
        {job && (
          <div className="rounded-md border border-muted bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
            Auto-profiling {job.status}.{" "}
            {job.status === "completed"
              ? "Generate DataPoints to review and approve."
              : "This page will refresh as the job progresses."}
          </div>
        )}
        {syncError && <div className="text-xs text-destructive">{syncError}</div>}
        {!syncStatus && (
          <p className="text-sm text-muted-foreground">
            Sync status unavailable.
          </p>
        )}
        {syncStatus && (
          <div className="space-y-1 text-sm">
            <div>Status: {syncStatus.status}</div>
            {syncStatus.sync_type && (
              <div className="text-xs text-muted-foreground">
                Type: {syncStatus.sync_type}
              </div>
            )}
            {syncStatus.total_datapoints > 0 && (
              <div className="text-xs text-muted-foreground">
                Progress: {syncStatus.processed_datapoints}/
                {syncStatus.total_datapoints}
              </div>
            )}
            <div className="text-xs text-muted-foreground">
              Started: {formatTimestamp(syncStatus.started_at)} · Finished:{" "}
              {formatTimestamp(syncStatus.finished_at)}
            </div>
            {syncStatus.error && (
              <div className="text-xs text-destructive">{syncStatus.error}</div>
            )}
          </div>
        )}
      </Card>

      <Card className="p-4 space-y-3">
        <h2 className="text-sm font-semibold">Profiling Jobs</h2>
        {!job && (
          <p className="text-sm text-muted-foreground">No profiling job started.</p>
        )}
        {job && (
          <div className="space-y-2">
            <div className="text-sm">Job: {job.job_id}</div>
            <div className="text-xs text-muted-foreground">
              Status: {job.status}
              {job.progress && (
                <span>
                  {" "}
                  · {job.progress.tables_completed}/{job.progress.total_tables} tables
                </span>
              )}
            </div>
            {job.error && (
              <div className="text-xs text-destructive">{job.error}</div>
            )}
            {generationJob && (
              <div className="rounded-md border border-muted bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
                <div className="flex items-center gap-2">
                  {generationJob.status === "running" && (
                    <span className="h-3 w-3 animate-spin rounded-full border-2 border-muted-foreground border-t-transparent" />
                  )}
                  <span>DataPoint generation {generationJob.status}.</span>
                </div>
                {generationJob.progress && (
                  <span>
                    {" "}
                    {generationJob.progress.tables_completed}/
                    {generationJob.progress.total_tables} tables
                    {" "}
                    · batch size {generationJob.progress.batch_size}
                  </span>
                )}
                {generationJob.error && (
                  <div className="text-xs text-destructive">{generationJob.error}</div>
                )}
              </div>
            )}
            {job.status === "completed" && job.profile_id && (
              <div className="space-y-2">
                {profileTables.length > 0 && (
                  <div className="space-y-2">
                    <div className="text-xs font-medium text-muted-foreground">
                      Table Selection
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => setSelectedTables(profileTables)}
                      >
                        Select All
                      </Button>
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => setSelectedTables([])}
                      >
                        Clear
                      </Button>
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() =>
                          setSelectedTables(
                            profileTables.slice(0, Math.min(10, profileTables.length))
                          )
                        }
                      >
                        Top 10
                      </Button>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {tableSelectionLabel}
                    </div>
                    <div className="max-h-40 overflow-auto rounded-md border border-border p-2 text-xs">
                      {profileTables.map((table) => (
                        <label
                          key={table}
                          className="flex items-center gap-2 py-1"
                        >
                          <input
                            type="checkbox"
                            checked={selectedTables.includes(table)}
                            onChange={(event) => {
                              if (event.target.checked) {
                                setSelectedTables((current) => [...current, table]);
                              } else {
                                setSelectedTables((current) =>
                                  current.filter((item) => item !== table)
                                );
                              }
                            }}
                          />
                          <span>{table}</span>
                        </label>
                      ))}
                    </div>
                    <div className="text-xs font-medium text-muted-foreground">
                      Depth
                    </div>
                    <select
                      className="w-full rounded-md border border-border bg-background p-2 text-xs"
                      value={depth}
                      onChange={(event) => setDepth(event.target.value)}
                    >
                      <option value="schema_only">Schema only (no LLM)</option>
                      <option value="metrics_basic">Basic metrics (no LLM)</option>
                      <option value="metrics_full">Full metrics (LLM, batched)</option>
                    </select>
                  </div>
                )}
                <Button
                  onClick={handleGenerate}
                  disabled={
                    jobs.generating ||
                    !hasSelection ||
                    generationJob?.status === "running"
                  }
                >
                  {jobs.generating ? (
                    <span className="flex items-center gap-2">
                      <span className="h-3 w-3 animate-spin rounded-full border-2 border-muted-foreground border-t-transparent" />
                      Generating...
                    </span>
                  ) : (
                    "Generate DataPoints"
                  )}
                </Button>
                {jobs.generating && (
                  <div className="text-xs text-muted-foreground">
                    Hang tight while we draft DataPoints and evaluate metrics.
                  </div>
                )}
                <div className="text-xs text-muted-foreground">
                  Note: Auto-generated values are normalized to match DataPoint
                  schema. Invalid aggregations are skipped.
                </div>
                <div className="rounded-md border border-muted bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
                  Tool-based profiling runs the same workflow with explicit approval.
                </div>
                <Button
                  variant="secondary"
                  onClick={handleToolProfile}
                  disabled={jobs.toolProfiling}
                >
                  {jobs.toolProfiling ? "Running..." : "Profile + Generate (Tool)"}
                </Button>
                {toolProfileMessage && (
                  <div className="text-xs text-muted-foreground">
                    {toolProfileMessage}
                  </div>
                )}
                {toolProfileError && (
                  <div className="text-xs text-destructive">{toolProfileError}</div>
                )}
              </div>
            )}
          </div>
        )}
      </Card>

      <Card className="p-4 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">
            Approved DataPoints (Selected Source)
          </h2>
          <div className="text-xs text-muted-foreground">
            {approved.length} total
          </div>
        </div>
        <div className="text-xs text-muted-foreground">
          {selectedConnection
            ? `Showing approved profiling DataPoints for ${selectedConnection.name}.`
            : "Select a connection to view approved profiling DataPoints."}
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <Button
            variant="secondary"
            size="sm"
            onClick={handleQualityReport}
            disabled={jobs.qualityReport}
          >
            {jobs.qualityReport ? "Checking..." : "Run Quality Report"}
          </Button>
          {qualityError && <span className="text-destructive">{qualityError}</span>}
        </div>
        {qualityReport && (
          <div className="rounded-md border border-muted bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
            <div>Total DataPoints: {Number(qualityReport.total_datapoints ?? 0)}</div>
            <div>
              Weak Schema:{" "}
              {(qualityReport.weak_schema as unknown[] | undefined)?.length ?? 0}
            </div>
            <div>
              Weak Metrics:{" "}
              {(qualityReport.weak_business as unknown[] | undefined)?.length ?? 0}
            </div>
            <div>
              Duplicate Metrics:{" "}
              {(qualityReport.duplicate_metrics as unknown[] | undefined)?.length ?? 0}
            </div>
            <div className="mt-2 space-y-2">
              {(qualityReport.weak_schema as Array<Record<string, unknown>> | undefined)?.length ? (
                <div>
                  <div className="font-medium text-foreground">Weak Schema</div>
                  <ul className="mt-1 space-y-1">
                    {(qualityReport.weak_schema as Array<Record<string, unknown>>).map(
                      (item) => (
                        <li key={String(item.datapoint_id)}>
                          {String(item.table_name || item.datapoint_id)} ·{" "}
                          {String(item.reason)}
                        </li>
                      )
                    )}
                  </ul>
                </div>
              ) : null}
              {(qualityReport.weak_business as Array<Record<string, unknown>> | undefined)?.length ? (
                <div>
                  <div className="font-medium text-foreground">Weak Metrics</div>
                  <ul className="mt-1 space-y-1">
                    {(qualityReport.weak_business as Array<Record<string, unknown>>).map(
                      (item) => (
                        <li key={String(item.datapoint_id)}>
                          {String(item.name || item.datapoint_id)} ·{" "}
                          {String(item.reason)}
                        </li>
                      )
                    )}
                  </ul>
                </div>
              ) : null}
              {(qualityReport.duplicate_metrics as Array<Record<string, unknown>> | undefined)
                ?.length ? (
                <div>
                  <div className="font-medium text-foreground">Duplicate Metrics</div>
                  <ul className="mt-1 space-y-1">
                    {(qualityReport.duplicate_metrics as Array<Record<string, unknown>>).map(
                      (item, index) => (
                        <li key={`${item.table}-${index}`}>
                          {String(item.table)} · {String(item.calculation)} ·{" "}
                          {String((item.datapoint_ids as string[] | undefined)?.length || 0)} ids
                        </li>
                      )
                    )}
                  </ul>
                </div>
              ) : null}
              {(qualityReport.duplicate_ids as Array<Record<string, unknown>> | undefined)
                ?.length ? (
                <div>
                  <div className="font-medium text-foreground">Duplicate IDs</div>
                  <ul className="mt-1 space-y-1">
                    {(qualityReport.duplicate_ids as Array<Record<string, unknown>>).map(
                      (item) => (
                        <li key={String(item.datapoint_id)}>
                          {String(item.datapoint_id)} · {String(item.count)} copies
                        </li>
                      )
                    )}
                  </ul>
                </div>
              ) : null}
            </div>
          </div>
        )}
        {approved.length === 0 && (
          <p className="text-sm text-muted-foreground">
            No approved DataPoints found for the selected source.
          </p>
        )}
        {approved.length > 0 && (
          <div className="max-h-64 space-y-2 overflow-auto text-sm">
            {approved.map((item) => (
              <div key={item.datapoint_id} className="border-b border-border pb-2">
                <div className="font-medium">
                  {item.name || item.datapoint_id}
                </div>
                <div className="text-xs text-muted-foreground">
                  {item.type} · {item.datapoint_id}
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card className="p-4 space-y-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold">DataPoint Editor (Managed)</h2>
            <p className="text-xs text-muted-foreground">
              Create, load, update, or delete managed DataPoint JSON (including Query DataPoints).
            </p>
          </div>
        </div>
        <div className="grid gap-2 sm:grid-cols-[1fr_auto_auto]">
          <Input
            value={editorDatapointId}
            onChange={(event) => setEditorDatapointId(event.target.value)}
            placeholder="DataPoint ID (e.g. query_top_customers_001)"
          />
          <Button variant="secondary" onClick={handleLoadDatapoint} disabled={jobs.editorLoading}>
            {jobs.editorLoading ? "Loading..." : "Load"}
          </Button>
          <Button
            variant="destructive"
            onClick={handleDeleteDatapoint}
            disabled={jobs.editorDeleting}
          >
            {jobs.editorDeleting ? "Deleting..." : "Delete"}
          </Button>
        </div>
        <textarea
          className="min-h-[260px] w-full rounded-md border border-border bg-background p-3 text-xs font-mono"
          value={editorPayload}
          onChange={(event) => setEditorPayload(event.target.value)}
          aria-label="Managed DataPoint editor"
        />
        <div className="flex flex-wrap gap-2">
          <Button onClick={handleCreateDatapoint} disabled={jobs.editorSaving}>
            {jobs.editorSaving ? "Saving..." : "Create New"}
          </Button>
          <Button variant="secondary" onClick={handleUpdateDatapoint} disabled={jobs.editorSaving}>
            {jobs.editorSaving ? "Saving..." : "Update Existing"}
          </Button>
          <Button
            variant="outline"
            onClick={() => setEditorPayload(DATAPOINT_EDITOR_TEMPLATE)}
            disabled={jobs.editorSaving || jobs.editorLoading}
          >
            Reset Template
          </Button>
        </div>
        {editorNotice && <div className="text-xs text-emerald-600">{editorNotice}</div>}
        {editorError && <div className="text-xs text-destructive">{editorError}</div>}
      </Card>

      <Card className="p-4 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">
            Pending DataPoints (Selected Source)
          </h2>
          <Button
            onClick={handleBulkApprove}
            disabled={
              pending.length === 0 ||
              jobs.bulkApproving ||
              !selectedConnection ||
              isEnvironmentConnection(selectedConnection)
            }
          >
            {jobs.bulkApproving ? "Approving..." : "Bulk Approve"}
          </Button>
        </div>
        {selectedConnection && isEnvironmentConnection(selectedConnection) && (
          <p className="text-xs text-muted-foreground">
            Select a managed connection to review pending profiling DataPoints.
          </p>
        )}
        {pending.length === 0 && (
          <p className="text-sm text-muted-foreground">
            No pending DataPoints for the selected source.
          </p>
        )}
        <div className="space-y-3">
          {pending.map((item) => (
            <div key={item.pending_id} className="border-b border-border pb-3">
              <div className="text-sm font-medium">
                {String(item.datapoint.name || item.datapoint.datapoint_id)}
              </div>
              <div className="text-xs text-muted-foreground">
                Confidence: {Math.round(item.confidence * 100)}% · Status: {item.status}
              </div>
              <div className="flex gap-2 mt-2">
                <Button
                  variant="secondary"
                  onClick={() => togglePendingDetails(item)}
                >
                  {expandedPendingId === item.pending_id ? "Hide Details" : "Review"}
                </Button>
              </div>
              {expandedPendingId === item.pending_id && (
                <div className="mt-3 space-y-3">
                  <div className="text-xs text-muted-foreground">
                    Review and edit the JSON before approving.
                  </div>
                  <textarea
                    className="min-h-[160px] w-full rounded-md border border-border bg-background p-2 text-xs font-mono"
                    value={pendingEdits[item.pending_id] || ""}
                    onChange={(event) =>
                      setPendingEdits((current) => ({
                        ...current,
                        [item.pending_id]: event.target.value,
                      }))
                    }
                  />
                  <div className="flex gap-2">
                    <Button
                      variant="secondary"
                      onClick={() => handleApprove(item.pending_id)}
                      disabled={approvingId === item.pending_id}
                    >
                      {approvingId === item.pending_id ? "Approving..." : "Approve"}
                    </Button>
                    <Button
                      variant="destructive"
                      onClick={() => handleReject(item.pending_id)}
                    >
                      Reject
                    </Button>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </Card>
      {toolApprovalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="w-full max-w-lg rounded-lg bg-background p-6 shadow-lg">
            <h3 className="text-base font-semibold">Approve Tool Execution</h3>
            <p className="mt-2 text-sm text-muted-foreground">
              You are about to profile the default database and generate pending DataPoints.
            </p>
            <div className="mt-4 space-y-2 text-xs text-muted-foreground">
              <div>Depth: {depth}</div>
              <div>Tables selected: {selectedTables.length || "all"}</div>
              <div>Batch size: 10</div>
            </div>
            <div className="mt-4 rounded-md border border-muted bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
              Cost hint: this triggers LLM calls for metrics and can take several minutes
              on larger databases.
            </div>
            {toolProfileError && (
              <div className="mt-2 text-xs text-destructive">{toolProfileError}</div>
            )}
            <div className="mt-4 flex gap-2">
              <Button
                onClick={handleToolProfileApprove}
                disabled={jobs.toolProfiling}
              >
                {jobs.toolProfiling ? "Running..." : "Approve & Run"}
              </Button>
              <Button
                variant="secondary"
                onClick={() => setToolApprovalOpen(false)}
                disabled={jobs.toolProfiling}
              >
                Cancel
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
