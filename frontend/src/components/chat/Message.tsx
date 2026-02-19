/**
 * Message Component
 *
 * Displays a single chat message with support for:
 * - User and assistant messages
 * - SQL code blocks
 * - Data tables
 * - Source citations
 * - Performance metrics
 * - Optional tabbed result layout + visualization
 */

"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  User,
  Bot,
  Code,
  Table as TableIcon,
  BookOpen,
  Clock,
  BadgeCheck,
  Copy,
  Download,
  BarChart3,
  ChevronLeft,
  ChevronRight,
  Settings2,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { cn } from "@/lib/utils";
import type { Message as MessageType } from "@/lib/stores/chat";
import type { ResultLayoutMode } from "@/lib/settings";

interface MessageProps {
  message: MessageType;
  displayMode?: ResultLayoutMode;
  showAgentTimingBreakdown?: boolean;
  onClarifyingAnswer?: (question: string) => void;
  onEditSqlDraft?: (sql: string) => void;
}

type TabId = "answer" | "sql" | "table" | "visualization" | "sources" | "timing";
type VizHint = "table" | "bar_chart" | "line_chart" | "pie_chart" | "scatter" | "none";
type InteractiveChartType = "bar_chart" | "line_chart" | "pie_chart" | "scatter";

interface BarChartConfig {
  labelCol: string;
  valueCol: string;
  maxItems: number;
  zoom: number;
  showLegend: boolean;
  seriesVisible: boolean;
}

interface LineChartConfig {
  xCol: string;
  yCol: string;
  maxItems: number;
  zoom: number;
  showLegend: boolean;
  showGrid: boolean;
  seriesVisible: boolean;
}

interface ScatterChartConfig {
  xCol: string;
  yCol: string;
  maxItems: number;
  zoom: number;
  showLegend: boolean;
  showGrid: boolean;
  seriesVisible: boolean;
}

interface PieChartConfig {
  labelCol: string;
  valueCol: string;
  maxItems: number;
  zoom: number;
  showLegend: boolean;
}

interface ChartTooltipState {
  title: string;
  detail?: string;
}

const CHART_COLORS = [
  "#2563eb",
  "#16a34a",
  "#ea580c",
  "#9333ea",
  "#0891b2",
  "#dc2626",
  "#ca8a04",
  "#4f46e5",
];

function renderInlineMarkdown(text: string): React.ReactNode[] {
  const tokens = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean);
  return tokens.map((token, index) => {
    if (token.startsWith("**") && token.endsWith("**") && token.length >= 4) {
      return <strong key={`b-${index}`}>{token.slice(2, -2)}</strong>;
    }
    if (token.startsWith("`") && token.endsWith("`") && token.length >= 3) {
      return (
        <code
          key={`c-${index}`}
          className="rounded bg-secondary/70 px-1 py-0.5 text-xs"
        >
          {token.slice(1, -1)}
        </code>
      );
    }
    return <React.Fragment key={`t-${index}`}>{token}</React.Fragment>;
  });
}

function renderMarkdownish(text: string): React.ReactNode {
  if (!text) {
    return null;
  }

  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks: React.ReactNode[] = [];
  const listItems: string[] = [];
  let paragraphBuffer: string[] = [];

  const flushParagraph = () => {
    if (!paragraphBuffer.length) {
      return;
    }
    const paragraph = paragraphBuffer.join("\n").trim();
    paragraphBuffer = [];
    if (!paragraph) {
      return;
    }
    blocks.push(
      <p key={`p-${blocks.length}`} className="whitespace-pre-wrap leading-relaxed">
        {renderInlineMarkdown(paragraph)}
      </p>
    );
  };

  const flushList = () => {
    if (!listItems.length) {
      return;
    }
    blocks.push(
      <ul key={`l-${blocks.length}`} className="list-disc space-y-1 pl-5">
        {listItems.map((item, index) => (
          <li key={`li-${index}`}>{renderInlineMarkdown(item)}</li>
        ))}
      </ul>
    );
    listItems.length = 0;
  };

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    const bulletMatch = line.match(/^\s*[*-]\s+(.+)$/);

    if (bulletMatch) {
      flushParagraph();
      listItems.push(bulletMatch[1].trim());
      continue;
    }

    if (line.trim() === "") {
      flushParagraph();
      flushList();
      continue;
    }

    flushList();
    paragraphBuffer.push(line);
  }

  flushParagraph();
  flushList();

  return <div className="space-y-2">{blocks}</div>;
}

const formatMetricNumber = (value: number): string => {
  if (!Number.isFinite(value)) {
    return "0";
  }
  return new Intl.NumberFormat(undefined, {
    maximumFractionDigits: Math.abs(value) < 100 ? 2 : 0,
  }).format(value);
};

const formatAxisTick = (value: number): string => {
  if (!Number.isFinite(value)) {
    return "0";
  }
  const abs = Math.abs(value);
  const maxFractionDigits = abs >= 1000 ? 0 : abs >= 100 ? 1 : 2;
  return new Intl.NumberFormat(undefined, {
    maximumFractionDigits: maxFractionDigits,
  }).format(value);
};

const formatDurationSeconds = (milliseconds: number): string => {
  if (!Number.isFinite(milliseconds) || milliseconds <= 0) {
    return "0s";
  }
  const seconds = milliseconds / 1000;
  const maximumFractionDigits = seconds >= 10 ? 1 : 2;
  return `${new Intl.NumberFormat(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits,
  }).format(seconds)}s`;
};

const toNumber = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return null;
};

const isDateLikeColumn = (columnName: string): boolean => {
  const lowered = columnName.toLowerCase();
  const tokens = lowered.split(/[^a-z0-9]+/).filter(Boolean);
  const directMarkers = new Set(["date", "time", "timestamp", "datetime"]);
  const periodMarkers = new Set(["day", "week", "month", "quarter", "year"]);
  const periodDisqualifiers = new Set(["type", "category", "name", "code"]);
  if (tokens.some((token) => directMarkers.has(token))) {
    return true;
  }
  if (
    tokens.some((token) => periodMarkers.has(token)) &&
    !tokens.some((token) => periodDisqualifiers.has(token))
  ) {
    return true;
  }
  return false;
};

function clampPercent(value: number): number {
  return Math.max(0, Math.min(100, value));
}

export function Message({
  message,
  displayMode = "stacked",
  showAgentTimingBreakdown = true,
  onClarifyingAnswer,
  onEditSqlDraft,
}: MessageProps) {
  const isUser = message.role === "user";
  const [activeTab, setActiveTab] = useState<TabId>("answer");
  const [actionNotice, setActionNotice] = useState<string | null>(null);
  const [selectedSubAnswerIndex, setSelectedSubAnswerIndex] = useState<number>(0);
  const [tablePage, setTablePage] = useState<number>(0);
  const [rowsPerPage, setRowsPerPage] = useState<number>(10);
  const [showChartSettings, setShowChartSettings] = useState(false);
  const [chartTooltip, setChartTooltip] = useState<ChartTooltipState | null>(null);
  const [hiddenPieLabels, setHiddenPieLabels] = useState<string[]>([]);
  const [barConfig, setBarConfig] = useState<BarChartConfig>({
    labelCol: "",
    valueCol: "",
    maxItems: 12,
    zoom: 1,
    showLegend: true,
    seriesVisible: true,
  });
  const [lineConfig, setLineConfig] = useState<LineChartConfig>({
    xCol: "",
    yCol: "",
    maxItems: 30,
    zoom: 1,
    showLegend: true,
    showGrid: true,
    seriesVisible: true,
  });
  const [scatterConfig, setScatterConfig] = useState<ScatterChartConfig>({
    xCol: "",
    yCol: "",
    maxItems: 120,
    zoom: 1,
    showLegend: true,
    showGrid: true,
    seriesVisible: true,
  });
  const [pieConfig, setPieConfig] = useState<PieChartConfig>({
    labelCol: "",
    valueCol: "",
    maxItems: 8,
    zoom: 1,
    showLegend: true,
  });
  const tabButtonRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const tabListId = `message-tablist-${message.id}`;
  const activeTabPanelId = `${message.id}-panel-${activeTab}`;

  const subAnswers = message.sub_answers || [];
  const activeSubAnswer =
    subAnswers.length > 0
      ? subAnswers[Math.min(selectedSubAnswerIndex, subAnswers.length - 1)]
      : null;
  const activeContent = activeSubAnswer?.answer || message.content;
  const activeSql = activeSubAnswer?.sql ?? message.sql;
  const activeData = activeSubAnswer?.data ?? message.data;
  const activeVisualizationHint =
    activeSubAnswer?.visualization_hint ?? message.visualization_hint;
  const activeClarifyingQuestions =
    activeSubAnswer?.clarifying_questions || message.clarifying_questions;

  useEffect(() => {
    if (subAnswers.length === 0) {
      setSelectedSubAnswerIndex(0);
      return;
    }
    if (selectedSubAnswerIndex > subAnswers.length - 1) {
      setSelectedSubAnswerIndex(0);
    }
  }, [selectedSubAnswerIndex, subAnswers.length]);

  useEffect(() => {
    setTablePage(0);
    setRowsPerPage(10);
  }, [message.id, selectedSubAnswerIndex]);

  useEffect(() => {
    setShowChartSettings(false);
    setChartTooltip(null);
    setHiddenPieLabels([]);
  }, [message.id, selectedSubAnswerIndex]);

  const columnNames = useMemo(
    () => (activeData ? Object.keys(activeData) : []),
    [activeData]
  );
  const rowCount =
    columnNames.length > 0
      ? Math.max(...columnNames.map((column) => activeData?.[column]?.length ?? 0))
      : 0;

  useEffect(() => {
    const totalPages = Math.max(1, Math.ceil(rowCount / rowsPerPage));
    setTablePage((page) => Math.min(page, totalPages - 1));
  }, [rowCount, rowsPerPage]);

  const rows = useMemo(
    () =>
      Array.from({ length: rowCount }, (_, rowIndex) =>
        columnNames.map((column) => activeData?.[column]?.[rowIndex])
      ),
    [activeData, columnNames, rowCount]
  );

  const rowObjects = useMemo(
    () =>
      Array.from({ length: rowCount }, (_, rowIndex) => {
        const record: Record<string, unknown> = {};
        for (const column of columnNames) {
          record[column] = activeData?.[column]?.[rowIndex];
        }
        return record;
      }),
    [activeData, columnNames, rowCount]
  );

  const numericColumns = useMemo(
    () =>
      columnNames.filter((column) =>
        rowObjects.some((row) => toNumber(row[column]) !== null)
      ),
    [columnNames, rowObjects]
  );

  const nonNumericColumns = useMemo(
    () => columnNames.filter((column) => !numericColumns.includes(column)),
    [columnNames, numericColumns]
  );

  useEffect(() => {
    const defaultBarLabel = nonNumericColumns[0] || columnNames[0] || "";
    const defaultBarValue = numericColumns[0] || "";
    const defaultLineX =
      columnNames.find((column) => isDateLikeColumn(column)) ||
      nonNumericColumns[0] ||
      columnNames[0] ||
      "";
    const defaultLineY = numericColumns[0] || "";
    const defaultScatterX = numericColumns[0] || "";
    const defaultScatterY = numericColumns[1] || numericColumns[0] || "";
    const defaultPieLabel = nonNumericColumns[0] || columnNames[0] || "";
    const defaultPieValue = numericColumns[0] || "";

    setBarConfig((prev) => ({
      ...prev,
      labelCol:
        prev.labelCol && columnNames.includes(prev.labelCol)
          ? prev.labelCol
          : defaultBarLabel,
      valueCol:
        prev.valueCol && numericColumns.includes(prev.valueCol)
          ? prev.valueCol
          : defaultBarValue,
    }));

    setLineConfig((prev) => ({
      ...prev,
      xCol:
        prev.xCol && columnNames.includes(prev.xCol)
          ? prev.xCol
          : defaultLineX,
      yCol:
        prev.yCol && numericColumns.includes(prev.yCol)
          ? prev.yCol
          : defaultLineY,
    }));

    setScatterConfig((prev) => ({
      ...prev,
      xCol:
        prev.xCol && numericColumns.includes(prev.xCol)
          ? prev.xCol
          : defaultScatterX,
      yCol:
        prev.yCol &&
        numericColumns.includes(prev.yCol) &&
        (!prev.xCol || prev.yCol !== prev.xCol || numericColumns.length === 1)
          ? prev.yCol
          : defaultScatterY,
    }));

    setPieConfig((prev) => ({
      ...prev,
      labelCol:
        prev.labelCol && columnNames.includes(prev.labelCol)
          ? prev.labelCol
          : defaultPieLabel,
      valueCol:
        prev.valueCol && numericColumns.includes(prev.valueCol)
          ? prev.valueCol
          : defaultPieValue,
    }));
  }, [columnNames, nonNumericColumns, numericColumns]);

  useEffect(() => {
    if (!actionNotice) {
      return;
    }
    const timeout = window.setTimeout(() => setActionNotice(null), 2000);
    return () => window.clearTimeout(timeout);
  }, [actionNotice]);

  const formatCellValue = (value: unknown) => {
    if (value === null || value === undefined) {
      return { display: "", full: "", truncated: false };
    }
    const full = typeof value === "string" ? value : JSON.stringify(value);
    const maxLength = 160;
    if (full.length <= maxLength) {
      return { display: full, full, truncated: false };
    }
    return {
      display: `${full.slice(0, maxLength)}…`,
      full,
      truncated: true,
    };
  };

  const hasSources =
    Boolean(message.sources?.length) && message.answer_source !== "context";
  const hasEvidence = Boolean(message.evidence?.length);
  const hasTable = Boolean(activeData) && rowCount > 0;
  const hasAgentTimings = Boolean(
    message.metrics?.agent_timings &&
      Object.keys(message.metrics.agent_timings).length > 0 &&
      showAgentTimingBreakdown
  );

  const inferVisualizationType = (): VizHint => {
    const hint = (activeVisualizationHint || "").toLowerCase();
    if (
      hint === "bar_chart" ||
      hint === "line_chart" ||
      hint === "pie_chart" ||
      hint === "scatter" ||
      hint === "table" ||
      hint === "none"
    ) {
      return hint;
    }
    if (!hasTable) {
      return "none";
    }
    if (numericColumns.length >= 2 && rowCount <= 200) {
      return "scatter";
    }
    if (numericColumns.length >= 1 && columnNames.some((col) => isDateLikeColumn(col))) {
      return "line_chart";
    }
    if (numericColumns.length >= 1 && rowCount <= 20) {
      return "bar_chart";
    }
    return "table";
  };

  const resolvedVizHint = inferVisualizationType();

  const copySql = async () => {
    if (!activeSql) {
      return;
    }
    try {
      await navigator.clipboard.writeText(activeSql);
      setActionNotice("SQL copied");
    } catch {
      setActionNotice("Unable to copy SQL");
    }
  };

  const downloadCsv = () => {
    if (!hasTable) {
      return;
    }
    const escapeCsv = (value: unknown) => {
      if (value === null || value === undefined) {
        return "";
      }
      const text = String(value);
      if (text.includes(",") || text.includes('"') || text.includes("\n")) {
        return `"${text.replace(/"/g, '""')}"`;
      }
      return text;
    };

    const header = columnNames.join(",");
    const body = rowObjects
      .map((row) => columnNames.map((column) => escapeCsv(row[column])).join(","))
      .join("\n");
    const csv = `${header}\n${body}`;
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "datachat-results.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    setActionNotice("CSV downloaded");
  };

  const renderClarifyingQuestions = () => {
    if (!activeClarifyingQuestions || activeClarifyingQuestions.length === 0) {
      return null;
    }
    return (
      <Card className="mt-4">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Clarifying questions</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2 text-sm">
            {activeClarifyingQuestions.map((question, index) => (
              <li key={`${question}-${index}`} className="flex items-start gap-2">
                <span className="mt-0.5 flex-1">• {question}</span>
                {onClarifyingAnswer && (
                  <button
                    type="button"
                    className="text-xs text-primary underline"
                    onClick={() => onClarifyingAnswer(question)}
                    aria-label={`Answer clarifying question: ${question}`}
                  >
                    Answer
                  </button>
                )}
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>
    );
  };

  const renderSqlSection = () => {
    if (!activeSql) {
      return (
        <Card className="mt-4">
          <CardContent className="pt-6 text-sm text-muted-foreground">
            No SQL generated for this answer.
          </CardContent>
        </Card>
      );
    }
    return (
      <Card className="mt-4">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Code size={16} />
            Generated SQL
          </CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="bg-secondary p-3 rounded text-sm overflow-x-auto">
            <code>{activeSql}</code>
          </pre>
        </CardContent>
      </Card>
    );
  };

  const renderSubAnswerSelector = () => {
    if (!subAnswers.length) {
      return null;
    }
    return (
      <Card className="mt-4">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Sub-questions</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap gap-2">
            {subAnswers.map((item, idx) => (
              <button
                key={`sub-answer-${item.index}`}
                type="button"
                className={cn(
                  "rounded px-2.5 py-1 text-xs transition",
                  idx === selectedSubAnswerIndex
                    ? "bg-primary text-primary-foreground"
                    : "bg-secondary text-foreground hover:bg-secondary/80"
                )}
                onClick={() => setSelectedSubAnswerIndex(idx)}
              >
                Q{item.index}
              </button>
            ))}
          </div>
          {activeSubAnswer && (
            <div className="rounded border border-border/80 bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
              <div className="font-medium text-foreground">{activeSubAnswer.query}</div>
              {(activeSubAnswer.answer_source || activeSubAnswer.answer_confidence !== undefined) && (
                <div className="mt-1">
                  Source: {activeSubAnswer.answer_source || "unknown"}
                  {typeof activeSubAnswer.answer_confidence === "number" &&
                    ` · confidence ${activeSubAnswer.answer_confidence.toFixed(2)}`}
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  const renderTableSection = () => {
    if (!hasTable) {
      return (
        <Card className="mt-4">
          <CardContent className="pt-6 text-sm text-muted-foreground">
            No tabular data returned.
          </CardContent>
        </Card>
      );
    }

    const totalPages = Math.ceil(rowCount / rowsPerPage);
    const startRow = tablePage * rowsPerPage;
    const endRow = Math.min(startRow + rowsPerPage, rowCount);
    const pageRows = rows.slice(startRow, endRow);

    const handlePrevPage = () => {
      setTablePage((p) => Math.max(0, p - 1));
    };

    const handleNextPage = () => {
      setTablePage((p) => Math.min(totalPages - 1, p + 1));
    };

    return (
      <Card className="mt-4">
        <details>
          <summary className="cursor-pointer list-none px-6 py-4" aria-label="Toggle result table">
            <div className="flex items-center justify-between gap-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <TableIcon size={16} />
                Results ({rowCount} rows)
              </CardTitle>
              <span className="text-xs text-muted-foreground">Expand</span>
            </div>
          </summary>
          <CardContent className="pt-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm" aria-label="Query result table">
                <caption className="sr-only">Query results</caption>
                <thead>
                  <tr className="border-b">
                    {columnNames.map((key) => (
                      <th key={key} className="text-left p-2 font-medium">
                        {key}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {pageRows.map((row, idx) => (
                    <tr key={startRow + idx} className="border-b last:border-0">
                      {row.map((value, vidx) => {
                        const { display, full, truncated } = formatCellValue(value);
                        return (
                          <td key={vidx} className="p-2 align-top">
                            <span
                              className={truncated ? "block max-w-[320px] truncate" : "block"}
                              title={truncated ? full : undefined}
                            >
                              {display}
                            </span>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {totalPages > 1 && (
              <div className="flex items-center justify-between mt-3 pt-3 border-t">
                <div className="flex items-center gap-3">
                  <p className="text-xs text-muted-foreground">
                    Showing {startRow + 1}-{endRow} of {rowCount} rows
                  </p>
                  <label className="flex items-center gap-1 text-xs text-muted-foreground">
                    Rows per page
                    <input
                      type="number"
                      min={1}
                      step={1}
                      value={rowsPerPage}
                      onChange={(event) => {
                        const value = Number(event.target.value);
                        if (!Number.isFinite(value) || value < 1) {
                          return;
                        }
                        setRowsPerPage(Math.floor(value));
                      }}
                      className="h-7 w-20 rounded border border-input bg-background px-2 text-xs"
                      aria-label="Rows per page"
                    />
                  </label>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={handlePrevPage}
                    disabled={tablePage === 0}
                    className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-xs hover:bg-secondary disabled:opacity-50 disabled:cursor-not-allowed"
                    aria-label="Go to previous result page"
                  >
                    <ChevronLeft size={14} />
                    Previous
                  </button>
                  <span className="text-xs text-muted-foreground">
                    Page {tablePage + 1} of {totalPages}
                  </span>
                  <button
                    type="button"
                    onClick={handleNextPage}
                    disabled={tablePage >= totalPages - 1}
                    className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-xs hover:bg-secondary disabled:opacity-50 disabled:cursor-not-allowed"
                    aria-label="Go to next result page"
                  >
                    Next
                    <ChevronRight size={14} />
                  </button>
                </div>
              </div>
            )}
            {totalPages === 1 && rowCount > 0 && (
              <div className="mt-2 flex items-center gap-3">
                <p className="text-xs text-muted-foreground">
                  Showing {rowCount} rows
                </p>
                <label className="flex items-center gap-1 text-xs text-muted-foreground">
                  Rows per page
                  <input
                    type="number"
                    min={1}
                    step={1}
                    value={rowsPerPage}
                    onChange={(event) => {
                      const value = Number(event.target.value);
                      if (!Number.isFinite(value) || value < 1) {
                        return;
                      }
                      setRowsPerPage(Math.floor(value));
                    }}
                    className="h-7 w-20 rounded border border-input bg-background px-2 text-xs"
                    aria-label="Rows per page"
                  />
                </label>
              </div>
            )}
          </CardContent>
        </details>
      </Card>
    );
  };

  const renderSourcesSection = () => {
    if (!hasSources && !hasEvidence) {
      return (
        <Card className="mt-4">
          <CardContent className="pt-6 text-sm text-muted-foreground">
            No sources available for this response.
          </CardContent>
        </Card>
      );
    }
    return (
      <div className="space-y-4 mt-4">
        {hasSources && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <BookOpen size={16} />
                Sources
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm">
                {message.sources?.map((source) => (
                  <li key={source.datapoint_id} className="text-sm flex items-start gap-2">
                    <span className="text-xs px-2 py-0.5 rounded bg-secondary">
                      {source.type}
                    </span>
                    <span className="flex-1">
                      {source.name}
                      <span className="text-xs text-muted-foreground ml-2">
                        (score: {source.relevance_score.toFixed(2)})
                      </span>
                    </span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        )}

        {hasEvidence && (
          <Card>
            <details>
              <summary className="cursor-pointer list-none px-6 py-4">
                <div className="flex items-center justify-between gap-3">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <BookOpen size={16} />
                    Evidence ({message.evidence?.length || 0})
                  </CardTitle>
                  <span className="text-xs text-muted-foreground">Expand</span>
                </div>
              </summary>
              <CardContent className="pt-0">
                {message.sources && message.sources.length > 0 && (
                  <div className="mb-3 text-xs text-muted-foreground">
                    Context summary: {message.sources.length} sources · Top:{" "}
                    {message.sources
                      .slice(0, 3)
                      .map((source) => source.name)
                      .join(", ")}
                  </div>
                )}
                {activeSql && (
                  <div className="mb-3">
                    <div className="text-xs font-medium text-muted-foreground">Raw SQL</div>
                    <pre className="mt-1 rounded bg-secondary p-2 text-xs overflow-x-auto">
                      <code>{activeSql}</code>
                    </pre>
                  </div>
                )}
                <ul className="space-y-2">
                  {message.evidence?.map((item) => (
                    <li
                      key={item.datapoint_id}
                      className="text-sm flex items-start gap-2"
                    >
                      <span className="text-xs px-2 py-0.5 rounded bg-secondary">
                        {item.type || "DataPoint"}
                      </span>
                      <span className="flex-1">
                        {item.name || item.datapoint_id}
                        {item.reason && (
                          <span className="text-xs text-muted-foreground ml-2">
                            ({item.reason})
                          </span>
                        )}
                      </span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </details>
          </Card>
        )}
      </div>
    );
  };

  const renderVisualizationSection = () => {
    if (!hasTable) {
      return (
        <Card className="mt-4">
          <CardContent className="pt-6 text-sm text-muted-foreground">
            No data returned for visualization.
          </CardContent>
        </Card>
      );
    }

    const interactiveChartType: InteractiveChartType | null =
      resolvedVizHint === "bar_chart" ||
      resolvedVizHint === "line_chart" ||
      resolvedVizHint === "scatter" ||
      resolvedVizHint === "pie_chart"
        ? resolvedVizHint
        : null;

    const chartTooltipDescription = chartTooltip?.detail
      ? `${chartTooltip.title} · ${chartTooltip.detail}`
      : chartTooltip?.title || null;

    const renderChartSettings = (chartType: InteractiveChartType) => {
      if (!showChartSettings) {
        return null;
      }

      const commonClassName = "h-8 rounded-md border border-input bg-background px-2 text-xs";

      if (chartType === "bar_chart") {
        return (
          <Card className="mt-2">
            <CardContent className="space-y-3 pt-4">
              <div className="grid gap-3 sm:grid-cols-2">
                <label className="space-y-1 text-xs">
                  <span className="font-medium">X axis column</span>
                  <select
                    className={commonClassName}
                    value={barConfig.labelCol}
                    aria-label="Bar chart X axis column"
                    onChange={(event) =>
                      setBarConfig((prev) => ({ ...prev, labelCol: event.target.value }))
                    }
                  >
                    {columnNames.map((column) => (
                      <option key={`bar-label-${column}`} value={column}>
                        {column}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="space-y-1 text-xs">
                  <span className="font-medium">Y axis column</span>
                  <select
                    className={commonClassName}
                    value={barConfig.valueCol}
                    aria-label="Bar chart Y axis column"
                    onChange={(event) =>
                      setBarConfig((prev) => ({ ...prev, valueCol: event.target.value }))
                    }
                  >
                    {numericColumns.map((column) => (
                      <option key={`bar-value-${column}`} value={column}>
                        {column}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                <label className="space-y-1 text-xs">
                  <span className="font-medium">Max bars: {barConfig.maxItems}</span>
                  <input
                    type="range"
                    min={2}
                    max={50}
                    value={barConfig.maxItems}
                    aria-label="Bar chart max bars"
                    onChange={(event) =>
                      setBarConfig((prev) => ({ ...prev, maxItems: Number(event.target.value) }))
                    }
                  />
                </label>
                <label className="space-y-1 text-xs">
                  <span className="font-medium">Zoom: {barConfig.zoom.toFixed(1)}x</span>
                  <input
                    type="range"
                    min={1}
                    max={2}
                    step={0.1}
                    value={barConfig.zoom}
                    aria-label="Bar chart zoom"
                    onChange={(event) =>
                      setBarConfig((prev) => ({ ...prev, zoom: Number(event.target.value) }))
                    }
                  />
                </label>
              </div>
              <div className="flex flex-wrap items-center gap-3 text-xs">
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={barConfig.showLegend}
                    onChange={(event) =>
                      setBarConfig((prev) => ({ ...prev, showLegend: event.target.checked }))
                    }
                  />
                  Show legend
                </label>
              </div>
            </CardContent>
          </Card>
        );
      }

      if (chartType === "line_chart") {
        return (
          <Card className="mt-2">
            <CardContent className="space-y-3 pt-4">
              <div className="grid gap-3 sm:grid-cols-2">
                <label className="space-y-1 text-xs">
                  <span className="font-medium">X axis column</span>
                  <select
                    className={commonClassName}
                    value={lineConfig.xCol}
                    aria-label="Line chart X axis column"
                    onChange={(event) =>
                      setLineConfig((prev) => ({ ...prev, xCol: event.target.value }))
                    }
                  >
                    {columnNames.map((column) => (
                      <option key={`line-x-${column}`} value={column}>
                        {column}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="space-y-1 text-xs">
                  <span className="font-medium">Y axis column</span>
                  <select
                    className={commonClassName}
                    value={lineConfig.yCol}
                    aria-label="Line chart Y axis column"
                    onChange={(event) =>
                      setLineConfig((prev) => ({ ...prev, yCol: event.target.value }))
                    }
                  >
                    {numericColumns.map((column) => (
                      <option key={`line-y-${column}`} value={column}>
                        {column}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                <label className="space-y-1 text-xs">
                  <span className="font-medium">Max points: {lineConfig.maxItems}</span>
                  <input
                    type="range"
                    min={2}
                    max={120}
                    value={lineConfig.maxItems}
                    aria-label="Line chart max points"
                    onChange={(event) =>
                      setLineConfig((prev) => ({ ...prev, maxItems: Number(event.target.value) }))
                    }
                  />
                </label>
                <label className="space-y-1 text-xs">
                  <span className="font-medium">Zoom: {lineConfig.zoom.toFixed(1)}x</span>
                  <input
                    type="range"
                    min={1}
                    max={3}
                    step={0.1}
                    value={lineConfig.zoom}
                    aria-label="Line chart zoom"
                    onChange={(event) =>
                      setLineConfig((prev) => ({ ...prev, zoom: Number(event.target.value) }))
                    }
                  />
                </label>
              </div>
              <div className="flex flex-wrap items-center gap-3 text-xs">
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={lineConfig.showLegend}
                    onChange={(event) =>
                      setLineConfig((prev) => ({ ...prev, showLegend: event.target.checked }))
                    }
                  />
                  Show legend
                </label>
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={lineConfig.showGrid}
                    onChange={(event) =>
                      setLineConfig((prev) => ({ ...prev, showGrid: event.target.checked }))
                    }
                  />
                  Show grid
                </label>
              </div>
            </CardContent>
          </Card>
        );
      }

      if (chartType === "scatter") {
        return (
          <Card className="mt-2">
            <CardContent className="space-y-3 pt-4">
              <div className="grid gap-3 sm:grid-cols-2">
                <label className="space-y-1 text-xs">
                  <span className="font-medium">X axis column</span>
                  <select
                    className={commonClassName}
                    value={scatterConfig.xCol}
                    aria-label="Scatter chart X axis column"
                    onChange={(event) =>
                      setScatterConfig((prev) => ({ ...prev, xCol: event.target.value }))
                    }
                  >
                    {numericColumns.map((column) => (
                      <option key={`scatter-x-${column}`} value={column}>
                        {column}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="space-y-1 text-xs">
                  <span className="font-medium">Y axis column</span>
                  <select
                    className={commonClassName}
                    value={scatterConfig.yCol}
                    aria-label="Scatter chart Y axis column"
                    onChange={(event) =>
                      setScatterConfig((prev) => ({ ...prev, yCol: event.target.value }))
                    }
                  >
                    {numericColumns.map((column) => (
                      <option key={`scatter-y-${column}`} value={column}>
                        {column}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                <label className="space-y-1 text-xs">
                  <span className="font-medium">Max points: {scatterConfig.maxItems}</span>
                  <input
                    type="range"
                    min={2}
                    max={300}
                    value={scatterConfig.maxItems}
                    aria-label="Scatter chart max points"
                    onChange={(event) =>
                      setScatterConfig((prev) => ({
                        ...prev,
                        maxItems: Number(event.target.value),
                      }))
                    }
                  />
                </label>
                <label className="space-y-1 text-xs">
                  <span className="font-medium">Zoom: {scatterConfig.zoom.toFixed(1)}x</span>
                  <input
                    type="range"
                    min={1}
                    max={3}
                    step={0.1}
                    value={scatterConfig.zoom}
                    aria-label="Scatter chart zoom"
                    onChange={(event) =>
                      setScatterConfig((prev) => ({ ...prev, zoom: Number(event.target.value) }))
                    }
                  />
                </label>
              </div>
              <div className="flex flex-wrap items-center gap-3 text-xs">
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={scatterConfig.showLegend}
                    onChange={(event) =>
                      setScatterConfig((prev) => ({ ...prev, showLegend: event.target.checked }))
                    }
                  />
                  Show legend
                </label>
                <label className="inline-flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={scatterConfig.showGrid}
                    onChange={(event) =>
                      setScatterConfig((prev) => ({ ...prev, showGrid: event.target.checked }))
                    }
                  />
                  Show grid
                </label>
              </div>
            </CardContent>
          </Card>
        );
      }

      return (
        <Card className="mt-2">
          <CardContent className="space-y-3 pt-4">
            <div className="grid gap-3 sm:grid-cols-2">
              <label className="space-y-1 text-xs">
                <span className="font-medium">Category column</span>
                <select
                  className={commonClassName}
                  value={pieConfig.labelCol}
                  aria-label="Pie chart category column"
                  onChange={(event) =>
                    setPieConfig((prev) => ({ ...prev, labelCol: event.target.value }))
                  }
                >
                  {columnNames.map((column) => (
                    <option key={`pie-label-${column}`} value={column}>
                      {column}
                    </option>
                  ))}
                </select>
              </label>
              <label className="space-y-1 text-xs">
                <span className="font-medium">Value column</span>
                <select
                  className={commonClassName}
                  value={pieConfig.valueCol}
                  aria-label="Pie chart value column"
                  onChange={(event) =>
                    setPieConfig((prev) => ({ ...prev, valueCol: event.target.value }))
                  }
                >
                  {numericColumns.map((column) => (
                    <option key={`pie-value-${column}`} value={column}>
                      {column}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <label className="space-y-1 text-xs">
                <span className="font-medium">Max slices: {pieConfig.maxItems}</span>
                <input
                  type="range"
                  min={2}
                  max={20}
                  value={pieConfig.maxItems}
                  aria-label="Pie chart max slices"
                  onChange={(event) =>
                    setPieConfig((prev) => ({ ...prev, maxItems: Number(event.target.value) }))
                  }
                />
              </label>
              <label className="space-y-1 text-xs">
                <span className="font-medium">Zoom: {pieConfig.zoom.toFixed(1)}x</span>
                <input
                  type="range"
                  min={1}
                  max={2.4}
                  step={0.1}
                  value={pieConfig.zoom}
                  aria-label="Pie chart zoom"
                  onChange={(event) =>
                    setPieConfig((prev) => ({ ...prev, zoom: Number(event.target.value) }))
                  }
                />
              </label>
            </div>
            <div className="flex flex-wrap items-center gap-3 text-xs">
              <label className="inline-flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={pieConfig.showLegend}
                  onChange={(event) =>
                    setPieConfig((prev) => ({ ...prev, showLegend: event.target.checked }))
                  }
                />
                Show legend
              </label>
            </div>
          </CardContent>
        </Card>
      );
    };

    const renderFallback = (messageText: string) => (
      <Card className="mt-4">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <BarChart3 size={16} />
            Visualization
          </CardTitle>
        </CardHeader>
        <CardContent>
          {interactiveChartType && (
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <button
                type="button"
                className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-xs hover:bg-secondary"
                onClick={() => setShowChartSettings((prev) => !prev)}
                aria-pressed={showChartSettings}
                aria-label="Toggle chart settings panel"
              >
                <Settings2 size={12} />
                {showChartSettings ? "Hide settings" : "Chart settings"}
              </button>
            </div>
          )}
          {interactiveChartType && renderChartSettings(interactiveChartType)}
          <p className="text-sm text-muted-foreground">{messageText}</p>
          <p className="mt-2 text-xs text-muted-foreground">
            Tip: switch to the Table tab for raw results.
          </p>
        </CardContent>
      </Card>
    );

    const renderBarChart = () => {
      const valueCol = barConfig.valueCol || numericColumns[0];
      const labelCol = barConfig.labelCol || nonNumericColumns[0] || columnNames[0];
      if (!valueCol || !labelCol) {
        return renderFallback("This result shape is not suitable for a bar chart.");
      }

      const points = rowObjects
        .map((row) => ({
          label: String(row[labelCol] ?? ""),
          value: toNumber(row[valueCol]),
        }))
        .filter((row) => row.value !== null)
        .slice(0, Math.max(2, barConfig.maxItems)) as Array<{ label: string; value: number }>;

      if (points.length < 2) {
        return renderFallback("Not enough points to draw a bar chart.");
      }

      const maxValue = Math.max(...points.map((row) => Math.abs(row.value)));
      if (maxValue <= 0) {
        return renderFallback("Bar values are all zero.");
      }

      return (
        <Card className="mt-4">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <BarChart3 size={16} />
              Bar Chart
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-xs hover:bg-secondary"
                onClick={() => setShowChartSettings((prev) => !prev)}
                aria-pressed={showChartSettings}
                aria-label="Toggle bar chart settings"
              >
                <Settings2 size={12} />
                {showChartSettings ? "Hide settings" : "Chart settings"}
              </button>
            </div>
            {renderChartSettings("bar_chart")}
            {!barConfig.seriesVisible ? (
              <div className="rounded border border-dashed p-3 text-xs text-muted-foreground">
                The <code>{valueCol}</code> series is hidden. Re-enable it from the legend toggle.
              </div>
            ) : (
              points.map((point, index) => (
                <div key={`${point.label}-${index}`} className="flex items-center gap-2">
                  <div className="w-28 truncate text-xs">{point.label}</div>
                  <div className="h-4 flex-1 rounded bg-secondary overflow-hidden">
                    <div
                      className="h-full origin-left bg-primary transition-transform"
                      style={{
                        width: `${clampPercent((Math.abs(point.value) / maxValue) * 100)}%`,
                        transform: `scaleX(${barConfig.zoom})`,
                      }}
                      role="img"
                      tabIndex={0}
                      aria-label={`${labelCol}: ${point.label}, ${valueCol}: ${formatMetricNumber(point.value)}`}
                      onMouseEnter={() =>
                        setChartTooltip({
                          title: point.label,
                          detail: `${valueCol}: ${formatMetricNumber(point.value)}`,
                        })
                      }
                      onMouseLeave={() => setChartTooltip(null)}
                      onFocus={() =>
                        setChartTooltip({
                          title: point.label,
                          detail: `${valueCol}: ${formatMetricNumber(point.value)}`,
                        })
                      }
                      onBlur={() => setChartTooltip(null)}
                    />
                  </div>
                  <div className="w-16 text-right text-xs">{formatMetricNumber(point.value)}</div>
                </div>
              ))
            )}
            <div className="rounded border border-border/80 bg-muted/30 px-2 py-2 text-xs text-muted-foreground">
              <div>
                <span className="font-medium text-foreground">X axis:</span> {labelCol}
              </div>
              <div>
                <span className="font-medium text-foreground">Y axis:</span> {valueCol}
              </div>
              {barConfig.showLegend && (
                <div className="mt-1 flex items-center gap-2">
                  <button
                    type="button"
                    className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-[11px] hover:bg-secondary"
                    aria-pressed={!barConfig.seriesVisible}
                    onClick={() =>
                      setBarConfig((prev) => ({
                        ...prev,
                        seriesVisible: !prev.seriesVisible,
                      }))
                    }
                  >
                    <span
                      className="inline-block h-2.5 w-2.5 rounded-full"
                      style={{ backgroundColor: CHART_COLORS[0] }}
                    />
                    {barConfig.seriesVisible ? "Hide" : "Show"} {valueCol}
                  </button>
                </div>
              )}
            </div>
            {chartTooltipDescription && (
              <p className="text-xs text-muted-foreground" aria-live="polite">
                {chartTooltipDescription}
              </p>
            )}
          </CardContent>
        </Card>
      );
    };

    const renderLineChart = () => {
      const valueCol = lineConfig.yCol || numericColumns[0];
      const xCol = lineConfig.xCol || columnNames.find((column) => isDateLikeColumn(column)) || nonNumericColumns[0] || columnNames[0];
      if (!valueCol || !xCol) {
        return renderFallback("This result shape is not suitable for a line chart.");
      }

      const points = rowObjects
        .map((row, index) => ({
          xLabel: String(row[xCol] ?? index + 1),
          y: toNumber(row[valueCol]),
        }))
        .filter((row) => row.y !== null)
        .slice(0, Math.max(2, lineConfig.maxItems)) as Array<{ xLabel: string; y: number }>;

      if (points.length < 2) {
        return renderFallback("Not enough points to draw a line chart.");
      }

      const width = 640;
      const height = 280;
      const padLeft = 56;
      const padRight = 24;
      const padTop = 20;
      const padBottom = 58;
      const plotWidth = width - padLeft - padRight;
      const plotHeight = height - padTop - padBottom;
      const ys = points.map((point) => point.y);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const yRange = maxY - minY || 1;
      const yTicks = [minY, minY + yRange / 2, maxY];
      const xTickIndexes = Array.from(
        new Set([0, Math.floor((points.length - 1) / 2), points.length - 1])
      );

      const linePoints = points
        .map((point, index) => {
          const x = padLeft + (index / (points.length - 1)) * plotWidth;
          const y = padTop + (1 - (point.y - minY) / yRange) * plotHeight;
          return `${x},${y}`;
        })
        .join(" ");

      return (
        <Card className="mt-4">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <BarChart3 size={16} />
              Line Chart
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-xs hover:bg-secondary"
                onClick={() => setShowChartSettings((prev) => !prev)}
                aria-pressed={showChartSettings}
                aria-label="Toggle line chart settings"
              >
                <Settings2 size={12} />
                {showChartSettings ? "Hide settings" : "Chart settings"}
              </button>
            </div>
            {renderChartSettings("line_chart")}
            <div className="overflow-x-auto">
              <svg
                viewBox={`0 0 ${width} ${height}`}
                className="min-w-[360px] h-[280px]"
                role="img"
                aria-label={`Line chart of ${valueCol} by ${xCol}`}
                style={{ width: `${Math.max(100, lineConfig.zoom * 100)}%` }}
              >
                {yTicks.map((tick, index) => {
                  const y = padTop + (1 - (tick - minY) / yRange) * plotHeight;
                  return (
                    <g key={`line-y-tick-${index}`}>
                      {lineConfig.showGrid && (
                        <line
                          x1={padLeft}
                          y1={y}
                          x2={width - padRight}
                          y2={y}
                          stroke="#e2e8f0"
                          strokeDasharray="3 3"
                        />
                      )}
                      <text
                        x={padLeft - 8}
                        y={y + 4}
                        textAnchor="end"
                        fontSize="10"
                        fill="#64748b"
                      >
                        {formatAxisTick(tick)}
                      </text>
                    </g>
                  );
                })}
                <line
                  x1={padLeft}
                  y1={height - padBottom}
                  x2={width - padRight}
                  y2={height - padBottom}
                  stroke="#94a3b8"
                />
                <line
                  x1={padLeft}
                  y1={padTop}
                  x2={padLeft}
                  y2={height - padBottom}
                  stroke="#94a3b8"
                />
                <polyline
                  fill="none"
                  stroke="#2563eb"
                  strokeWidth="2.5"
                  points={linePoints}
                  opacity={lineConfig.seriesVisible ? 1 : 0.2}
                />
                {lineConfig.seriesVisible &&
                  points.map((point, index) => {
                  const x = padLeft + (index / (points.length - 1)) * plotWidth;
                  const y = padTop + (1 - (point.y - minY) / yRange) * plotHeight;
                  return (
                    <circle
                      key={`${point.xLabel}-${index}`}
                      cx={x}
                      cy={y}
                      r="3"
                      fill="#2563eb"
                      tabIndex={0}
                      aria-label={`${xCol}: ${point.xLabel}, ${valueCol}: ${formatMetricNumber(point.y)}`}
                      onMouseEnter={() =>
                        setChartTooltip({
                          title: point.xLabel,
                          detail: `${valueCol}: ${formatMetricNumber(point.y)}`,
                        })
                      }
                      onMouseLeave={() => setChartTooltip(null)}
                      onFocus={() =>
                        setChartTooltip({
                          title: point.xLabel,
                          detail: `${valueCol}: ${formatMetricNumber(point.y)}`,
                        })
                      }
                      onBlur={() => setChartTooltip(null)}
                    />
                  );
                })}
                {xTickIndexes.map((index) => {
                  const x = padLeft + (index / Math.max(points.length - 1, 1)) * plotWidth;
                  const raw = points[index]?.xLabel ?? "";
                  const label = raw.length > 16 ? `${raw.slice(0, 15)}…` : raw;
                  return (
                    <text
                      key={`line-x-tick-${index}`}
                      x={x}
                      y={height - padBottom + 16}
                      textAnchor="middle"
                      fontSize="10"
                      fill="#64748b"
                    >
                      {label}
                    </text>
                  );
                })}
                <text
                  x={padLeft + plotWidth / 2}
                  y={height - 8}
                  textAnchor="middle"
                  fontSize="11"
                  fill="#334155"
                >
                  {xCol}
                </text>
                <text
                  x={14}
                  y={padTop + plotHeight / 2}
                  textAnchor="middle"
                  fontSize="11"
                  fill="#334155"
                  transform={`rotate(-90 14 ${padTop + plotHeight / 2})`}
                >
                  {valueCol}
                </text>
                <g transform={`translate(${width - padRight - 140}, ${padTop + 6})`}>
                  {lineConfig.showLegend && (
                    <>
                      <rect x="0" y="0" width="134" height="24" fill="#f8fafc" stroke="#e2e8f0" rx="4" />
                      <rect x="8" y="9" width="14" height="2.5" fill="#2563eb" />
                      <text x="28" y="13" fontSize="10" fill="#334155">
                        {valueCol}
                      </text>
                    </>
                  )}
                </g>
              </svg>
            </div>
            {lineConfig.showLegend && (
              <button
                type="button"
                className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-xs hover:bg-secondary"
                onClick={() =>
                  setLineConfig((prev) => ({ ...prev, seriesVisible: !prev.seriesVisible }))
                }
                aria-pressed={!lineConfig.seriesVisible}
              >
                {lineConfig.seriesVisible ? "Hide" : "Show"} {valueCol}
              </button>
            )}
            <p className="text-xs text-muted-foreground">
              X axis: {xCol} · Y axis: {valueCol} · Legend: {valueCol} · {points.length} points
            </p>
            {chartTooltipDescription && (
              <p className="text-xs text-muted-foreground" aria-live="polite">
                {chartTooltipDescription}
              </p>
            )}
          </CardContent>
        </Card>
      );
    };

    const renderScatter = () => {
      const xCol = scatterConfig.xCol || numericColumns[0];
      const yCol = scatterConfig.yCol || numericColumns[1] || numericColumns[0];
      if (!xCol || !yCol) {
        return renderFallback("Scatter plot needs at least two numeric columns.");
      }

      const points = rowObjects
        .map((row) => ({
          x: toNumber(row[xCol]),
          y: toNumber(row[yCol]),
        }))
        .filter((row) => row.x !== null && row.y !== null)
        .slice(0, Math.max(2, scatterConfig.maxItems)) as Array<{ x: number; y: number }>;

      if (points.length < 2) {
        return renderFallback("Not enough numeric points for scatter plot.");
      }

      const width = 640;
      const height = 280;
      const padLeft = 56;
      const padRight = 24;
      const padTop = 20;
      const padBottom = 58;
      const plotWidth = width - padLeft - padRight;
      const plotHeight = height - padTop - padBottom;
      const xs = points.map((point) => point.x);
      const ys = points.map((point) => point.y);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const xRange = maxX - minX || 1;
      const yRange = maxY - minY || 1;
      const xTicks = [minX, minX + xRange / 2, maxX];
      const yTicks = [minY, minY + yRange / 2, maxY];

      return (
        <Card className="mt-4">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <BarChart3 size={16} />
              Scatter Plot
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-xs hover:bg-secondary"
                onClick={() => setShowChartSettings((prev) => !prev)}
                aria-pressed={showChartSettings}
                aria-label="Toggle scatter chart settings"
              >
                <Settings2 size={12} />
                {showChartSettings ? "Hide settings" : "Chart settings"}
              </button>
            </div>
            {renderChartSettings("scatter")}
            <div className="overflow-x-auto">
              <svg
                viewBox={`0 0 ${width} ${height}`}
                className="min-w-[360px] h-[280px]"
                role="img"
                aria-label={`Scatter plot of ${yCol} by ${xCol}`}
                style={{ width: `${Math.max(100, scatterConfig.zoom * 100)}%` }}
              >
                {yTicks.map((tick, index) => {
                  const y = padTop + (1 - (tick - minY) / yRange) * plotHeight;
                  return (
                    <g key={`scatter-y-tick-${index}`}>
                      {scatterConfig.showGrid && (
                        <line
                          x1={padLeft}
                          y1={y}
                          x2={width - padRight}
                          y2={y}
                          stroke="#e2e8f0"
                          strokeDasharray="3 3"
                        />
                      )}
                      <text
                        x={padLeft - 8}
                        y={y + 4}
                        textAnchor="end"
                        fontSize="10"
                        fill="#64748b"
                      >
                        {formatAxisTick(tick)}
                      </text>
                    </g>
                  );
                })}
                {xTicks.map((tick, index) => {
                  const x = padLeft + ((tick - minX) / xRange) * plotWidth;
                  return (
                    <g key={`scatter-x-tick-${index}`}>
                      {scatterConfig.showGrid && (
                        <line
                          x1={x}
                          y1={padTop}
                          x2={x}
                          y2={height - padBottom}
                          stroke="#e2e8f0"
                          strokeDasharray="3 3"
                        />
                      )}
                      <text
                        x={x}
                        y={height - padBottom + 16}
                        textAnchor="middle"
                        fontSize="10"
                        fill="#64748b"
                      >
                        {formatAxisTick(tick)}
                      </text>
                    </g>
                  );
                })}
                <line
                  x1={padLeft}
                  y1={height - padBottom}
                  x2={width - padRight}
                  y2={height - padBottom}
                  stroke="#94a3b8"
                />
                <line
                  x1={padLeft}
                  y1={padTop}
                  x2={padLeft}
                  y2={height - padBottom}
                  stroke="#94a3b8"
                />
                {points.map((point, index) => {
                  const x = padLeft + ((point.x - minX) / xRange) * plotWidth;
                  const y = padTop + (1 - (point.y - minY) / yRange) * plotHeight;
                  return (
                    <circle
                      key={`scatter-${index}`}
                      cx={x}
                      cy={y}
                      r="3.2"
                      fill="#2563eb"
                      opacity={scatterConfig.seriesVisible ? 0.85 : 0.2}
                      tabIndex={0}
                      aria-label={`${xCol}: ${formatMetricNumber(point.x)}, ${yCol}: ${formatMetricNumber(point.y)}`}
                      onMouseEnter={() =>
                        setChartTooltip({
                          title: `${xCol}: ${formatMetricNumber(point.x)}`,
                          detail: `${yCol}: ${formatMetricNumber(point.y)}`,
                        })
                      }
                      onMouseLeave={() => setChartTooltip(null)}
                      onFocus={() =>
                        setChartTooltip({
                          title: `${xCol}: ${formatMetricNumber(point.x)}`,
                          detail: `${yCol}: ${formatMetricNumber(point.y)}`,
                        })
                      }
                      onBlur={() => setChartTooltip(null)}
                    />
                  );
                })}
                <text
                  x={padLeft + plotWidth / 2}
                  y={height - 8}
                  textAnchor="middle"
                  fontSize="11"
                  fill="#334155"
                >
                  {xCol}
                </text>
                <text
                  x={14}
                  y={padTop + plotHeight / 2}
                  textAnchor="middle"
                  fontSize="11"
                  fill="#334155"
                  transform={`rotate(-90 14 ${padTop + plotHeight / 2})`}
                >
                  {yCol}
                </text>
                <g transform={`translate(${width - padRight - 188}, ${padTop + 6})`}>
                  {scatterConfig.showLegend && (
                    <>
                      <rect x="0" y="0" width="182" height="24" fill="#f8fafc" stroke="#e2e8f0" rx="4" />
                      <circle cx="12" cy="12" r="3.2" fill="#2563eb" />
                      <text x="24" y="15" fontSize="10" fill="#334155">
                        {`${xCol} vs ${yCol}`}
                      </text>
                    </>
                  )}
                </g>
              </svg>
            </div>
            {scatterConfig.showLegend && (
              <button
                type="button"
                className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-xs hover:bg-secondary"
                onClick={() =>
                  setScatterConfig((prev) => ({
                    ...prev,
                    seriesVisible: !prev.seriesVisible,
                  }))
                }
                aria-pressed={!scatterConfig.seriesVisible}
              >
                {scatterConfig.seriesVisible ? "Hide" : "Show"} points
              </button>
            )}
            <p className="text-xs text-muted-foreground">
              X axis: {xCol} · Y axis: {yCol} · Legend: points · {points.length} points
            </p>
            {chartTooltipDescription && (
              <p className="text-xs text-muted-foreground" aria-live="polite">
                {chartTooltipDescription}
              </p>
            )}
          </CardContent>
        </Card>
      );
    };

    const renderPie = () => {
      const valueCol = pieConfig.valueCol || numericColumns[0];
      const labelCol = pieConfig.labelCol || nonNumericColumns[0] || columnNames[0];
      if (!valueCol || !labelCol) {
        return renderFallback("This result shape is not suitable for a pie chart.");
      }

      const allSlices = rowObjects
        .map((row) => ({
          label: String(row[labelCol] ?? ""),
          value: toNumber(row[valueCol]),
        }))
        .filter((row) => row.value !== null && row.value > 0)
        .slice(0, Math.max(2, pieConfig.maxItems)) as Array<{ label: string; value: number }>;
      const slices = allSlices.filter((slice) => !hiddenPieLabels.includes(slice.label));

      if (allSlices.length < 2) {
        return renderFallback("Not enough positive categories for a pie chart.");
      }
      if (slices.length < 1) {
        return renderFallback("All slices are hidden. Re-enable legend categories.");
      }

      const total = slices.reduce((sum, slice) => sum + slice.value, 0);
      if (total <= 0) {
        return renderFallback("Pie values are not positive.");
      }
      let cumulative = 0;
      const stops = slices.map((slice, index) => {
        const start = (cumulative / total) * 360;
        cumulative += slice.value;
        const end = (cumulative / total) * 360;
        const color = CHART_COLORS[index % CHART_COLORS.length];
        return `${color} ${start}deg ${end}deg`;
      });
      const pieSize = Math.max(120, Math.round(160 * pieConfig.zoom));

      return (
        <Card className="mt-4">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <BarChart3 size={16} />
              Pie Chart
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-xs hover:bg-secondary"
                onClick={() => setShowChartSettings((prev) => !prev)}
                aria-pressed={showChartSettings}
                aria-label="Toggle pie chart settings"
              >
                <Settings2 size={12} />
                {showChartSettings ? "Hide settings" : "Chart settings"}
              </button>
              {hiddenPieLabels.length > 0 && (
                <button
                  type="button"
                  className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-xs hover:bg-secondary"
                  onClick={() => setHiddenPieLabels([])}
                >
                  Reset hidden slices
                </button>
              )}
            </div>
            {renderChartSettings("pie_chart")}
            <div className="flex flex-wrap items-start gap-4">
              <div
                className="rounded-full border border-border"
                style={{ background: `conic-gradient(${stops.join(", ")})` }}
                role="img"
                aria-label={`Pie chart with ${slices.length} visible categories`}
                aria-describedby={chartTooltipDescription ? `${message.id}-viz-tooltip` : undefined}
                onMouseLeave={() => setChartTooltip(null)}
                onBlur={() => setChartTooltip(null)}
              >
                <div style={{ height: pieSize, width: pieSize }} />
              </div>
              {pieConfig.showLegend && (
                <ul className="flex-1 space-y-1 text-xs" aria-label="Pie chart legend">
                  {allSlices.map((slice, index) => {
                    const isHidden = hiddenPieLabels.includes(slice.label);
                    return (
                      <li key={`${slice.label}-${index}`} className="flex items-center gap-2">
                        <button
                          type="button"
                          className="inline-flex w-full items-center gap-2 rounded border border-border px-2 py-1 text-left hover:bg-secondary"
                          aria-pressed={!isHidden}
                          onClick={() =>
                            setHiddenPieLabels((prev) =>
                              prev.includes(slice.label)
                                ? prev.filter((label) => label !== slice.label)
                                : [...prev, slice.label]
                            )
                          }
                          onMouseEnter={() =>
                            setChartTooltip({
                              title: slice.label,
                              detail: `${formatMetricNumber(slice.value)} (${((slice.value / (allSlices.reduce((sum, item) => sum + item.value, 0) || 1)) * 100).toFixed(1)}%)`,
                            })
                          }
                          onFocus={() =>
                            setChartTooltip({
                              title: slice.label,
                              detail: `${formatMetricNumber(slice.value)} (${((slice.value / (allSlices.reduce((sum, item) => sum + item.value, 0) || 1)) * 100).toFixed(1)}%)`,
                            })
                          }
                          onMouseLeave={() => setChartTooltip(null)}
                          onBlur={() => setChartTooltip(null)}
                        >
                          <span
                            className="inline-block h-2.5 w-2.5 rounded-full"
                            style={{ backgroundColor: CHART_COLORS[index % CHART_COLORS.length] }}
                          />
                          <span className={`flex-1 truncate ${isHidden ? "line-through opacity-60" : ""}`}>
                            {slice.label}
                          </span>
                          <span className={isHidden ? "opacity-60" : ""}>
                            {formatMetricNumber(slice.value)}
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              Legend: {labelCol} categories · Slice value: {valueCol}
            </p>
            {chartTooltipDescription && (
              <p
                id={`${message.id}-viz-tooltip`}
                className="text-xs text-muted-foreground"
                aria-live="polite"
              >
                {chartTooltipDescription}
              </p>
            )}
          </CardContent>
        </Card>
      );
    };

    if (resolvedVizHint === "none" || resolvedVizHint === "table") {
      return renderFallback("This query is better represented as a table.");
    }
    if (resolvedVizHint === "bar_chart") {
      return renderBarChart();
    }
    if (resolvedVizHint === "line_chart") {
      return renderLineChart();
    }
    if (resolvedVizHint === "scatter") {
      return renderScatter();
    }
    if (resolvedVizHint === "pie_chart") {
      return renderPie();
    }
    return renderFallback("No suitable visualization is available.");
  };

  const renderTimingSection = () => {
    if (!hasAgentTimings || !message.metrics?.agent_timings) {
      return (
        <Card className="mt-4">
          <CardContent className="pt-6 text-sm text-muted-foreground">
            No agent timing breakdown available.
          </CardContent>
        </Card>
      );
    }

    return (
      <Card className="mt-4">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Clock size={16} />
            Agent Timing Breakdown
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {Object.entries(message.metrics.agent_timings)
              .sort((a, b) => b[1] - a[1])
              .map(([agent, ms]) => (
                <div key={agent} className="flex items-center justify-between gap-3 text-sm">
                  <span>{formatAgentTimingLabel(agent)}</span>
                  <span className="text-muted-foreground">{formatDurationSeconds(ms)}</span>
                </div>
              ))}
          </div>
        </CardContent>
      </Card>
    );
  };

  const tabs: Array<{ id: TabId; label: string }> = useMemo(() => {
    const items: Array<{ id: TabId; label: string }> = [{ id: "answer", label: "Answer" }];
    if (activeSql) {
      items.push({ id: "sql", label: "SQL" });
    }
    if (hasTable) {
      items.push({ id: "table", label: "Table" });
      items.push({ id: "visualization", label: "Visualization" });
    }
    if (hasSources || hasEvidence) {
      items.push({ id: "sources", label: "Sources" });
    }
    if (hasAgentTimings) {
      items.push({ id: "timing", label: "Timing" });
    }
    return items;
  }, [activeSql, hasAgentTimings, hasEvidence, hasSources, hasTable]);

  useEffect(() => {
    if (!tabs.some((tab) => tab.id === activeTab)) {
      setActiveTab("answer");
    }
  }, [activeTab, tabs]);

  const focusTabByIndex = (index: number) => {
    const nextIndex = (index + tabs.length) % tabs.length;
    tabButtonRefs.current[nextIndex]?.focus();
    setActiveTab(tabs[nextIndex].id);
  };

  const handleTabKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>, index: number) => {
    if (event.key === "ArrowRight") {
      event.preventDefault();
      focusTabByIndex(index + 1);
      return;
    }
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      focusTabByIndex(index - 1);
      return;
    }
    if (event.key === "Home") {
      event.preventDefault();
      focusTabByIndex(0);
      return;
    }
    if (event.key === "End") {
      event.preventDefault();
      focusTabByIndex(tabs.length - 1);
    }
  };

  const assistantMeta = !isUser && (message.answer_source || message.tool_approval_required);
  const showActions = !isUser && (Boolean(activeSql) || hasTable);

  const renderAnswerOnly = () => (
    <>
      {renderMarkdownish(activeContent)}
      {renderClarifyingQuestions()}
    </>
  );

  const renderTabContent = () => {
    if (activeTab === "answer") {
      return renderAnswerOnly();
    }
    if (activeTab === "sql") {
      return renderSqlSection();
    }
    if (activeTab === "table") {
      return renderTableSection();
    }
    if (activeTab === "visualization") {
      return renderVisualizationSection();
    }
    if (activeTab === "sources") {
      return renderSourcesSection();
    }
    if (activeTab === "timing") {
      return renderTimingSection();
    }
    return null;
  };

  const formatAgentTimingLabel = (agent: string) => {
    const labels: Record<string, string> = {
      tool_planner: "Tool Planner",
      classifier: "Classifier",
      context: "Context",
      sql: "SQL",
      validator: "Validator",
      executor: "Executor",
      context_answer: "Context Answer",
      response_synthesis: "Response Synthesis",
    };
    if (labels[agent]) {
      return labels[agent];
    }
    return agent
      .split("_")
      .map((part) => (part ? `${part[0].toUpperCase()}${part.slice(1)}` : part))
      .join(" ");
  };

  return (
    <div
      className={cn("flex gap-3 mb-4", isUser ? "justify-end" : "justify-start")}
      role="article"
      aria-label={isUser ? "User message" : "Assistant message"}
    >
      {!isUser && (
        <div
          className="flex-shrink-0 w-8 h-8 rounded-full bg-primary flex items-center justify-center text-primary-foreground"
          aria-hidden="true"
        >
          <Bot size={18} />
        </div>
      )}

      <div className={cn("flex-1 max-w-4xl", isUser && "flex justify-end")}>
        <div
          className={cn(
            "rounded-xl px-4 py-3",
            isUser
              ? "bg-primary/90 text-primary-foreground shadow-sm"
              : "border border-border/70 bg-card text-foreground shadow-sm"
          )}
        >
          {assistantMeta && (
            <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
              {message.answer_source && (
                <span className="inline-flex items-center gap-1 rounded-full bg-secondary px-2 py-1 text-foreground">
                  <BadgeCheck size={12} />
                  {message.answer_source}
                  {typeof message.answer_confidence === "number" &&
                    ` · ${message.answer_confidence.toFixed(2)}`}
                </span>
              )}
              {message.tool_approval_required && (
                <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2 py-1 text-amber-900">
                  Approval required
                </span>
              )}
            </div>
          )}

          {showActions && (
            <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
              {activeSql && (
                <button
                  type="button"
                  className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 hover:bg-secondary"
                  onClick={copySql}
                  aria-label="Copy generated SQL"
                >
                  <Copy size={12} />
                  Copy SQL
                </button>
              )}
              {activeSql && onEditSqlDraft && (
                <button
                  type="button"
                  className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 hover:bg-secondary"
                  onClick={() => onEditSqlDraft(activeSql)}
                  aria-label="Edit SQL draft"
                >
                  <Code size={12} />
                  Edit SQL
                </button>
              )}
              {hasTable && (
                <button
                  type="button"
                  className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 hover:bg-secondary"
                  onClick={downloadCsv}
                  aria-label="Download query results as CSV"
                >
                  <Download size={12} />
                  Download CSV
                </button>
              )}
              {actionNotice && (
                <span className="text-muted-foreground" role="status" aria-live="polite">
                  {actionNotice}
                </span>
              )}
            </div>
          )}

          {!isUser && renderSubAnswerSelector()}

          {!isUser && displayMode === "tabbed" ? (
            <>
              <div
                className="mb-2 flex flex-wrap gap-2 border-b border-border pb-2"
                role="tablist"
                aria-label="Assistant response sections"
                id={tabListId}
              >
                {tabs.map((tab, index) => (
                  <button
                    key={tab.id}
                    type="button"
                    ref={(element) => {
                      tabButtonRefs.current[index] = element;
                    }}
                    role="tab"
                    id={`${message.id}-tab-${tab.id}`}
                    aria-selected={activeTab === tab.id}
                    aria-controls={`${message.id}-panel-${tab.id}`}
                    className={cn(
                      "rounded px-2.5 py-1 text-xs transition",
                      activeTab === tab.id
                        ? "bg-primary text-primary-foreground"
                        : "bg-secondary text-foreground hover:bg-secondary/80"
                    )}
                    onClick={() => setActiveTab(tab.id)}
                    onKeyDown={(event) => handleTabKeyDown(event, index)}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
              <div
                role="tabpanel"
                id={activeTabPanelId}
                aria-labelledby={`${message.id}-tab-${activeTab}`}
              >
                {renderTabContent()}
              </div>
            </>
          ) : (
            <>
              {renderAnswerOnly()}
              {!isUser && activeSql && renderSqlSection()}
              {!isUser && hasTable && renderTableSection()}
              {!isUser && (hasSources || hasEvidence) && renderSourcesSection()}
            </>
          )}

          {message.metrics && (
            <div className="mt-3 text-xs text-muted-foreground">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-1">
                  <Clock size={12} />
                  {formatDurationSeconds(message.metrics.total_latency_ms)}
                </div>
                {message.metrics.llm_calls > 0 && <div>LLM calls: {message.metrics.llm_calls}</div>}
                {message.metrics.retry_count > 0 && <div>Retries: {message.metrics.retry_count}</div>}
                <div>
                  Formatter: {message.metrics.sql_formatter_fallback_calls ?? 0} (
                  {message.metrics.sql_formatter_fallback_successes ?? 0} recovered)
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {isUser && (
        <div
          className="flex-shrink-0 w-8 h-8 rounded-full bg-secondary flex items-center justify-center text-secondary-foreground"
          aria-hidden="true"
        >
          <User size={18} />
        </div>
      )}
    </div>
  );
}
