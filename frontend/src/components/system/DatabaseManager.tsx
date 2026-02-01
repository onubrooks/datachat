"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  DataChatAPI,
  DatabaseConnection,
  PendingDataPoint,
  ProfilingJob,
  SyncStatusResponse,
} from "@/lib/api";

const api = new DataChatAPI();

export function DatabaseManager() {
  const [connections, setConnections] = useState<DatabaseConnection[]>([]);
  const [pending, setPending] = useState<PendingDataPoint[]>([]);
  const [job, setJob] = useState<ProfilingJob | null>(null);
  const [syncStatus, setSyncStatus] = useState<SyncStatusResponse | null>(null);
  const [syncError, setSyncError] = useState<string | null>(null);
  const [expandedPendingId, setExpandedPendingId] = useState<string | null>(null);
  const [pendingEdits, setPendingEdits] = useState<Record<string, string>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [name, setName] = useState("");
  const [databaseUrl, setDatabaseUrl] = useState("");
  const [databaseType, setDatabaseType] = useState("postgresql");
  const [description, setDescription] = useState("");
  const [isDefault, setIsDefault] = useState(false);

  const refresh = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [dbs, pendingPoints] = await Promise.all([
        api.listDatabases(),
        api.listPendingDatapoints(),
      ]);
      setConnections(dbs);
      setPending(pendingPoints);
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
    try {
      const generated = await api.generateDatapoints(job.profile_id);
      setPending(generated);
    } catch (err) {
      setError((err as Error).message);
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
    const editedDatapoint = parseEditedDatapoint(pendingId);
    if (pendingEdits[pendingId] && !editedDatapoint) {
      return;
    }
    try {
      await api.approvePendingDatapoint(pendingId, editedDatapoint || undefined);
      await refresh();
    } catch (err) {
      setError((err as Error).message);
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
    try {
      await api.bulkApproveDatapoints();
      await refresh();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const handleSync = async () => {
    setSyncError(null);
    try {
      await api.triggerSync();
      const status = await api.getSyncStatus();
      setSyncStatus(status);
    } catch (err) {
      setSyncError((err as Error).message);
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
          <Button onClick={refresh} disabled={isLoading}>
            Refresh
          </Button>
        </div>
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
            disabled={syncStatus?.status === "running"}
          >
            Sync Now
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
            {job.status === "completed" && job.profile_id && (
              <Button onClick={handleGenerate}>Generate DataPoints</Button>
            )}
          </div>
        )}
      </Card>

      <Card className="p-4 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Pending DataPoints</h2>
          <Button onClick={handleBulkApprove} disabled={pending.length === 0}>
            Bulk Approve
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
                    >
                      Approve
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
