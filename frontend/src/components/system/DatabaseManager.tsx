"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

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
const WS_BASE_URL =
  process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

export function DatabaseManager() {
  const [connections, setConnections] = useState<DatabaseConnection[]>([]);
  const [pending, setPending] = useState<PendingDataPoint[]>([]);
  const [approved, setApproved] = useState<DataPointSummary[]>([]);
  const [job, setJob] = useState<ProfilingJob | null>(null);
  const [generationJob, setGenerationJob] = useState<GenerationJob | null>(null);
  const [syncStatus, setSyncStatus] = useState<SyncStatusResponse | null>(null);
  const [syncError, setSyncError] = useState<string | null>(null);
  const [expandedPendingId, setExpandedPendingId] = useState<string | null>(null);
  const [pendingEdits, setPendingEdits] = useState<Record<string, string>>({});
  const [isGenerating, setIsGenerating] = useState(false);
  const [isBulkApproving, setIsBulkApproving] = useState(false);
  const [approvingId, setApprovingId] = useState<string | null>(null);
  const [isSyncing, setIsSyncing] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [name, setName] = useState("");
  const [databaseUrl, setDatabaseUrl] = useState("");
  const [databaseType, setDatabaseType] = useState("postgresql");
  const [description, setDescription] = useState("");
  const [isDefault, setIsDefault] = useState(false);
  const [profileTables, setProfileTables] = useState<string[]>([]);
  const [selectedTables, setSelectedTables] = useState<string[]>([]);
  const [depth, setDepth] = useState("metrics_basic");

  const refresh = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [dbs, pendingPoints, approvedPoints] = await Promise.all([
        api.listDatabases(),
        api.listPendingDatapoints(),
        api.listDatapoints(),
      ]);
      setConnections(dbs);
      setPending(pendingPoints);
      setApproved(approvedPoints);
      if (dbs.length > 0) {
        const defaultConnection = dbs.find((item) => item.is_default) || dbs[0];
        const latestJob = await api.getLatestProfilingJob(
          defaultConnection.connection_id
        );
        setJob(latestJob);
      } else {
        setJob(null);
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsLoading(false);
    }

    try {
      const status = await api.getSyncStatus();
      setSyncStatus(status);
      setSyncError(null);
    } catch (err) {
      setSyncError((err as Error).message);
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
    refresh();
  }, []);

  useEffect(() => {
    if (!job || job.status === "completed" || job.status === "failed") {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const updated = await api.getProfilingJob(job.job_id);
        setJob(updated);
        if (updated.status === "completed") {
          await refresh();
        }
      } catch (err) {
        setError((err as Error).message);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [job]);

  useEffect(() => {
    if (!job?.profile_id) return;
    api
      .listProfileTables(job.profile_id)
      .then((tables) => {
        setProfileTables(tables);
        if (tables.length && selectedTables.length === 0) {
          setSelectedTables(tables.slice(0, Math.min(10, tables.length)));
        }
      })
      .catch((err) => setError((err as Error).message));
  }, [job?.profile_id]);

  useEffect(() => {
    if (!job?.profile_id) return;
    api
      .getLatestGenerationJob(job.profile_id)
      .then((latest) => setGenerationJob(latest))
      .catch((err) => setError((err as Error).message));
  }, [job?.profile_id]);

  useEffect(() => {
    if (!generationJob?.job_id) return;
    const ws = new WebSocket(`${WS_BASE_URL}/ws/profiling`);
    ws.onopen = () => {
      ws.send(JSON.stringify({ job_id: generationJob.job_id }));
    };
    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.event === "generation_update") {
          setGenerationJob(payload.job);
        }
      } catch (err) {
        setError((err as Error).message);
      }
    };
    return () => ws.close();
  }, [generationJob?.job_id]);

  useEffect(() => {
    if (!generationJob || generationJob.status === "completed" || generationJob.status === "failed") {
      return;
    }
    const interval = setInterval(async () => {
      try {
        const updated = await api.getGenerationJob(generationJob.job_id);
        setGenerationJob(updated);
        if (updated.status === "completed" || updated.status === "failed") {
          await refresh();
        }
      } catch (err) {
        setError((err as Error).message);
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [generationJob?.job_id, generationJob?.status]);

  useEffect(() => {
    if (generationJob?.status === "completed") {
      refresh();
    }
  }, [generationJob?.status]);

  useEffect(() => {
    if (!syncStatus || syncStatus.status !== "running") {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const status = await api.getSyncStatus();
        setSyncStatus(status);
        if (status.status !== "running") {
          await refresh();
        }
      } catch (err) {
        setSyncError((err as Error).message);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [syncStatus]);

  const handleCreate = async () => {
    setIsLoading(true);
    setError(null);
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
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleProfile = async (connectionId: string) => {
    setError(null);
    try {
      const started = await api.startProfiling(connectionId, {
        sample_size: 100,
      });
      setJob(started);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const handleGenerate = async () => {
    if (!job?.profile_id) return;
    setError(null);
    setIsGenerating(true);
    try {
      const generation = await api.startDatapointGeneration({
        profile_id: job.profile_id,
        tables: selectedTables,
        depth,
        batch_size: 10,
        max_tables: selectedTables.length || null,
        max_metrics_per_table: 3,
      });
      setGenerationJob(generation);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsGenerating(false);
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
    const snapshot = pending.find((item) => item.pending_id === pendingId);
    const editedDatapoint = parseEditedDatapoint(pendingId);
    if (pendingEdits[pendingId] && !editedDatapoint) {
      setApprovingId(null);
      return;
    }
    try {
      const approved = await api.approvePendingDatapoint(
        pendingId,
        editedDatapoint || undefined
      );
      setPending((current) =>
        current.filter((item) => item.pending_id !== pendingId)
      );
      setApproved((current) => [
        {
          datapoint_id: String(approved.datapoint.datapoint_id || pendingId),
          type: String(approved.datapoint.type || "Unknown"),
          name: approved.datapoint.name ? String(approved.datapoint.name) : null,
        },
        ...current,
      ]);
      await refresh();
    } catch (err) {
      if (snapshot) {
        setPending((current) => [snapshot, ...current]);
      }
      setError((err as Error).message);
    } finally {
      setApprovingId(null);
    }
  };

  const handleReject = async (pendingId: string) => {
    setError(null);
    try {
      await api.rejectPendingDatapoint(pendingId, "Rejected via UI");
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const handleBulkApprove = async () => {
    setError(null);
    setIsBulkApproving(true);
    const snapshot = pending;
    try {
      const approved = await api.bulkApproveDatapoints();
      if (approved.length) {
        setPending([]);
        setApproved((current) => [
          ...approved.map((item) => ({
            datapoint_id: String(item.datapoint.datapoint_id || item.pending_id),
            type: String(item.datapoint.type || "Unknown"),
            name: item.datapoint.name ? String(item.datapoint.name) : null,
          })),
          ...current,
        ]);
      }
      await refresh();
    } catch (err) {
      setPending(snapshot);
      setError((err as Error).message);
    } finally {
      setIsBulkApproving(false);
    }
  };

  const handleSync = async () => {
    setSyncError(null);
    setIsSyncing(true);
    try {
      await api.triggerSync();
      const status = await api.getSyncStatus();
      setSyncStatus(status);
    } catch (err) {
      setSyncError((err as Error).message);
    } finally {
      setIsSyncing(false);
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
    setIsResetting(true);
    setError(null);
    try {
      await api.systemReset();
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsResetting(false);
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
          <Button variant="outline" onClick={handleSystemReset} disabled={isResetting}>
            {isResetting ? "Resetting..." : "Reset System"}
          </Button>
          <Button onClick={refresh} disabled={isLoading}>
            Refresh
          </Button>
        </div>
      </div>
      <div className="text-xs text-muted-foreground">
        Reset clears system registry/profiling, local vectors, and saved setup config.
        It does not delete target database tables.
      </div>

      {error && <div className="text-sm text-destructive">{error}</div>}

      <Card className="p-4 space-y-4">
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
            onChange={(event) => setDatabaseUrl(event.target.value)}
          />
          <Input
            placeholder="Database Type (postgresql, clickhouse)"
            value={databaseType}
            onChange={(event) => setDatabaseType(event.target.value)}
          />
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
                  onClick={async () => {
                    await api.setDefaultDatabase(connection.connection_id);
                    await refresh();
                  }}
                >
                  Set Default
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => handleProfile(connection.connection_id)}
                >
                  Profile
                </Button>
                <Button
                  variant="destructive"
                  onClick={async () => {
                    await api.deleteDatabase(connection.connection_id);
                    await refresh();
                  }}
                >
                  Delete
                </Button>
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card className="p-4 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Sync Status</h2>
          <Button
            variant="secondary"
            onClick={handleSync}
            disabled={syncStatus?.status === "running" || isSyncing}
          >
            {isSyncing
              ? "Syncing..."
              : syncStatus?.status === "completed"
                ? "Sync Again"
                : "Sync Now"}
          </Button>
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
                    isGenerating ||
                    !hasSelection ||
                    generationJob?.status === "running" ||
                    generationJob?.status === "completed"
                  }
                >
                  {isGenerating ? (
                    <span className="flex items-center gap-2">
                      <span className="h-3 w-3 animate-spin rounded-full border-2 border-muted-foreground border-t-transparent" />
                      Generating...
                    </span>
                  ) : (
                    "Generate DataPoints"
                  )}
                </Button>
                {isGenerating && (
                  <div className="text-xs text-muted-foreground">
                    Hang tight while we draft DataPoints and evaluate metrics.
                  </div>
                )}
                <div className="text-xs text-muted-foreground">
                  Note: Auto-generated values are normalized to match DataPoint
                  schema. Invalid aggregations are skipped.
                </div>
              </div>
            )}
          </div>
        )}
      </Card>

      <Card className="p-4 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Approved DataPoints</h2>
          <div className="text-xs text-muted-foreground">
            {approved.length} total
          </div>
        </div>
        {approved.length === 0 && (
          <p className="text-sm text-muted-foreground">
            No approved DataPoints yet.
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

      <Card className="p-4 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Pending DataPoints</h2>
          <Button
            onClick={handleBulkApprove}
            disabled={pending.length === 0 || isBulkApproving}
          >
            {isBulkApproving ? "Approving..." : "Bulk Approve"}
          </Button>
        </div>
        {pending.length === 0 && (
          <p className="text-sm text-muted-foreground">No pending DataPoints.</p>
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
    </div>
  );
}
