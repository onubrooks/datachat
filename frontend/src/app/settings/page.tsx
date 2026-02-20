/**
 * Settings Page
 *
 * UI preferences and behavior settings.
 */

"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  applyThemeMode,
  getSynthesizeSimpleSql,
  getShowLiveReasoning,
  getThemeMode,
  getResultLayoutMode,
  getShowAgentTimingBreakdown,
  setResultLayoutMode,
  setShowLiveReasoning,
  setShowAgentTimingBreakdown,
  setSynthesizeSimpleSql,
  setThemeMode,
  type ResultLayoutMode,
  type ThemeMode,
} from "@/lib/settings";
import { useChatStore } from "@/lib/stores/chat";

export default function SettingsPage() {
  const [resultLayout, setResultLayout] = useState<ResultLayoutMode>("stacked");
  const [showAgentTimings, setShowAgentTimings] = useState(true);
  const [synthesizeSimpleSql, setSynthesizeSimpleSqlState] = useState(true);
  const [showLiveReasoning, setShowLiveReasoningState] = useState(true);
  const [themeMode, setThemeModeState] = useState<ThemeMode>("system");
  const [clearedAt, setClearedAt] = useState<Date | null>(null);
  const clearMessages = useChatStore((state) => state.clearMessages);

  useEffect(() => {
    setResultLayout(getResultLayoutMode());
    setShowAgentTimings(getShowAgentTimingBreakdown());
    setSynthesizeSimpleSqlState(getSynthesizeSimpleSql());
    setShowLiveReasoningState(getShowLiveReasoning());
    setThemeModeState(getThemeMode());
  }, []);

  const handleLayoutChange = (value: ResultLayoutMode) => {
    setResultLayout(value);
    setResultLayoutMode(value);
  };

  const handleAgentTimingsChange = (value: boolean) => {
    setShowAgentTimings(value);
    setShowAgentTimingBreakdown(value);
  };

  const handleSynthesizeSimpleSqlChange = (value: boolean) => {
    setSynthesizeSimpleSqlState(value);
    setSynthesizeSimpleSql(value);
  };

  const handleShowLiveReasoningChange = (value: boolean) => {
    setShowLiveReasoningState(value);
    setShowLiveReasoning(value);
  };

  const handleThemeModeChange = (value: ThemeMode) => {
    setThemeModeState(value);
    setThemeMode(value);
    applyThemeMode(value);
  };

  const handleClearChatHistory = () => {
    clearMessages();
    setClearedAt(new Date());
  };

  return (
    <main className="h-screen flex flex-col p-6 gap-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Settings</h1>
          <p className="text-sm text-muted-foreground">
            Configure display and interaction preferences.
          </p>
        </div>
        <Button asChild variant="secondary" size="sm">
          <Link href="/">Back to Chat</Link>
        </Button>
      </div>

      <Card className="p-4 space-y-4">
        <div className="text-sm font-medium">Theme</div>
        <div className="grid gap-3 sm:grid-cols-3">
          <button
            type="button"
            onClick={() => handleThemeModeChange("light")}
            className={`rounded-md border px-4 py-3 text-left text-sm transition ${
              themeMode === "light"
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/40"
            }`}
          >
            <div className="font-medium">Light</div>
            <div className="text-xs text-muted-foreground mt-1">
              Always use light theme.
            </div>
          </button>
          <button
            type="button"
            onClick={() => handleThemeModeChange("dark")}
            className={`rounded-md border px-4 py-3 text-left text-sm transition ${
              themeMode === "dark"
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/40"
            }`}
          >
            <div className="font-medium">Dark</div>
            <div className="text-xs text-muted-foreground mt-1">
              Always use dark theme.
            </div>
          </button>
          <button
            type="button"
            onClick={() => handleThemeModeChange("system")}
            className={`rounded-md border px-4 py-3 text-left text-sm transition ${
              themeMode === "system"
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/40"
            }`}
          >
            <div className="font-medium">System</div>
            <div className="text-xs text-muted-foreground mt-1">
              Follow OS preference.
            </div>
          </button>
        </div>
      </Card>

      <Card className="p-4 space-y-4">
        <div className="text-sm font-medium">Result Layout</div>
        <div className="grid gap-3 sm:grid-cols-2">
          <button
            type="button"
            onClick={() => handleLayoutChange("stacked")}
            className={`rounded-md border px-4 py-3 text-left text-sm transition ${
              resultLayout === "stacked"
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/40"
            }`}
          >
            <div className="font-medium">Stacked</div>
            <div className="text-xs text-muted-foreground mt-1">
              Current style: answer, SQL, table, and sources in one flow.
            </div>
          </button>
          <button
            type="button"
            onClick={() => handleLayoutChange("tabbed")}
            className={`rounded-md border px-4 py-3 text-left text-sm transition ${
              resultLayout === "tabbed"
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/40"
            }`}
          >
            <div className="font-medium">Tabbed</div>
            <div className="text-xs text-muted-foreground mt-1">
              Answer, SQL, table, visualization, and sources in tabs.
            </div>
          </button>
        </div>
      </Card>

      <Card className="p-4 space-y-4">
        <div className="text-sm font-medium">Agent Timing Breakdown</div>
        <div className="grid gap-3 sm:grid-cols-2">
          <button
            type="button"
            onClick={() => handleAgentTimingsChange(true)}
            className={`rounded-md border px-4 py-3 text-left text-sm transition ${
              showAgentTimings
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/40"
            }`}
          >
            <div className="font-medium">Show</div>
            <div className="text-xs text-muted-foreground mt-1">
              Display per-agent execution times beside response metrics.
            </div>
          </button>
          <button
            type="button"
            onClick={() => handleAgentTimingsChange(false)}
            className={`rounded-md border px-4 py-3 text-left text-sm transition ${
              !showAgentTimings
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/40"
            }`}
          >
            <div className="font-medium">Hide</div>
            <div className="text-xs text-muted-foreground mt-1">
              Keep only summary metrics (latency, LLM calls, retries).
            </div>
          </button>
        </div>
      </Card>

      <Card className="p-4 space-y-4">
        <div className="text-sm font-medium">Live Reasoning Stream</div>
        <div className="grid gap-3 sm:grid-cols-2">
          <button
            type="button"
            onClick={() => handleShowLiveReasoningChange(true)}
            className={`rounded-md border px-4 py-3 text-left text-sm transition ${
              showLiveReasoning
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/40"
            }`}
          >
            <div className="font-medium">Show</div>
            <div className="text-xs text-muted-foreground mt-1">
              Show live, temporary step-by-step thinking notes while a response is running.
            </div>
          </button>
          <button
            type="button"
            onClick={() => handleShowLiveReasoningChange(false)}
            className={`rounded-md border px-4 py-3 text-left text-sm transition ${
              !showLiveReasoning
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/40"
            }`}
          >
            <div className="font-medium">Hide</div>
            <div className="text-xs text-muted-foreground mt-1">
              Keep only final output and agent timeline.
            </div>
          </button>
        </div>
      </Card>

      <Card className="p-4 space-y-4">
        <div className="text-sm font-medium">Simple SQL Response Synthesis</div>
        <div className="grid gap-3 sm:grid-cols-2">
          <button
            type="button"
            onClick={() => handleSynthesizeSimpleSqlChange(true)}
            className={`rounded-md border px-4 py-3 text-left text-sm transition ${
              synthesizeSimpleSql
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/40"
            }`}
          >
            <div className="font-medium">On</div>
            <div className="text-xs text-muted-foreground mt-1">
              Use the synthesis agent for simple SQL answers (best wording quality).
            </div>
          </button>
          <button
            type="button"
            onClick={() => handleSynthesizeSimpleSqlChange(false)}
            className={`rounded-md border px-4 py-3 text-left text-sm transition ${
              !synthesizeSimpleSql
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/40"
            }`}
          >
            <div className="font-medium">Off</div>
            <div className="text-xs text-muted-foreground mt-1">
              Skip synthesis on simple SQL answers to reduce latency.
            </div>
          </button>
        </div>
      </Card>

      <Card className="p-4 space-y-3">
        <div className="text-sm font-medium">Chat History</div>
        <p className="text-xs text-muted-foreground">
          Clears the locally saved chat session and starts a new frontend session id.
        </p>
        <div className="flex items-center gap-3">
          <Button type="button" variant="destructive" size="sm" onClick={handleClearChatHistory}>
            Clear Chat History
          </Button>
          {clearedAt && (
            <span className="text-xs text-muted-foreground">
              Cleared at {clearedAt.toLocaleTimeString()}
            </span>
          )}
        </div>
      </Card>
    </main>
  );
}
