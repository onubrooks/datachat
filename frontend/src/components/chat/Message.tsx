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

import React, { useEffect, useMemo, useState } from "react";
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
}

type TabId = "answer" | "sql" | "table" | "visualization" | "sources" | "timing";
type VizHint = "table" | "bar_chart" | "line_chart" | "pie_chart" | "scatter" | "none";

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
  return (
    lowered.includes("date") ||
    lowered.includes("time") ||
    lowered.includes("month") ||
    lowered.includes("year") ||
    lowered.includes("day")
  );
};

function clampPercent(value: number): number {
  return Math.max(0, Math.min(100, value));
}

export function Message({
  message,
  displayMode = "stacked",
  showAgentTimingBreakdown = true,
  onClarifyingAnswer,
}: MessageProps) {
  const isUser = message.role === "user";
  const [activeTab, setActiveTab] = useState<TabId>("answer");
  const [actionNotice, setActionNotice] = useState<string | null>(null);

  const columnNames = useMemo(
    () => (message.data ? Object.keys(message.data) : []),
    [message.data]
  );
  const rowCount =
    columnNames.length > 0
      ? Math.max(...columnNames.map((column) => message.data?.[column]?.length ?? 0))
      : 0;

  const rows = useMemo(
    () =>
      Array.from({ length: rowCount }, (_, rowIndex) =>
        columnNames.map((column) => message.data?.[column]?.[rowIndex])
      ),
    [columnNames, message.data, rowCount]
  );

  const rowObjects = useMemo(
    () =>
      Array.from({ length: rowCount }, (_, rowIndex) => {
        const record: Record<string, unknown> = {};
        for (const column of columnNames) {
          record[column] = message.data?.[column]?.[rowIndex];
        }
        return record;
      }),
    [columnNames, message.data, rowCount]
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
  const hasTable = Boolean(message.data) && rowCount > 0;
  const hasAgentTimings = Boolean(
    message.metrics?.agent_timings &&
      Object.keys(message.metrics.agent_timings).length > 0 &&
      showAgentTimingBreakdown
  );

  const inferVisualizationType = (): VizHint => {
    const hint = (message.visualization_hint || "").toLowerCase();
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
    if (numericColumns.length >= 1 && columnNames.some((col) => isDateLikeColumn(col))) {
      return "line_chart";
    }
    if (numericColumns.length >= 1 && nonNumericColumns.length >= 1) {
      if (rowCount <= 30) {
        return "bar_chart";
      }
      return "table";
    }
    if (numericColumns.length >= 1 && rowCount <= 25) {
      return "bar_chart";
    }
    return "table";
  };

  const resolvedVizHint = inferVisualizationType();

  const copySql = async () => {
    if (!message.sql) {
      return;
    }
    try {
      await navigator.clipboard.writeText(message.sql);
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
    if (!message.clarifying_questions || message.clarifying_questions.length === 0) {
      return null;
    }
    return (
      <Card className="mt-4">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Clarifying questions</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2 text-sm">
            {message.clarifying_questions.map((question, index) => (
              <li key={`${question}-${index}`} className="flex items-start gap-2">
                <span className="mt-0.5 flex-1">• {question}</span>
                {onClarifyingAnswer && (
                  <button
                    type="button"
                    className="text-xs text-primary underline"
                    onClick={() => onClarifyingAnswer(question)}
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
    if (!message.sql) {
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
            <code>{message.sql}</code>
          </pre>
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
    return (
      <Card className="mt-4">
        <details>
          <summary className="cursor-pointer list-none px-6 py-4">
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
              <table className="w-full text-sm">
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
                  {rows.slice(0, 10).map((row, idx) => (
                    <tr key={idx} className="border-b last:border-0">
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
              {rowCount > 10 && (
                <p className="text-xs text-muted-foreground mt-2">
                  Showing 10 of {rowCount} rows
                </p>
              )}
            </div>
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
                {message.sql && (
                  <div className="mb-3">
                    <div className="text-xs font-medium text-muted-foreground">Raw SQL</div>
                    <pre className="mt-1 rounded bg-secondary p-2 text-xs overflow-x-auto">
                      <code>{message.sql}</code>
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

    const renderFallback = (messageText: string) => (
      <Card className="mt-4">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <BarChart3 size={16} />
            Visualization
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">{messageText}</p>
          <p className="mt-2 text-xs text-muted-foreground">
            Tip: switch to the Table tab for raw results.
          </p>
        </CardContent>
      </Card>
    );

    const renderBarChart = () => {
      const valueCol = numericColumns[0];
      const labelCol = nonNumericColumns[0] || columnNames[0];
      if (!valueCol || !labelCol) {
        return renderFallback("This result shape is not suitable for a bar chart.");
      }

      const points = rowObjects
        .map((row) => ({
          label: String(row[labelCol] ?? ""),
          value: toNumber(row[valueCol]),
        }))
        .filter((row) => row.value !== null)
        .slice(0, 12) as Array<{ label: string; value: number }>;

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
          <CardContent className="space-y-2">
            {points.map((point, index) => (
              <div key={`${point.label}-${index}`} className="flex items-center gap-2">
                <div className="w-28 truncate text-xs">{point.label}</div>
                <div className="h-4 flex-1 rounded bg-secondary overflow-hidden">
                  <div
                    className="h-full bg-primary"
                    style={{ width: `${clampPercent((Math.abs(point.value) / maxValue) * 100)}%` }}
                  />
                </div>
                <div className="w-16 text-right text-xs">{formatMetricNumber(point.value)}</div>
              </div>
            ))}
          </CardContent>
        </Card>
      );
    };

    const renderLineChart = () => {
      const valueCol = numericColumns[0];
      const xCol =
        columnNames.find((column) => isDateLikeColumn(column)) ||
        nonNumericColumns[0] ||
        columnNames[0];
      if (!valueCol || !xCol) {
        return renderFallback("This result shape is not suitable for a line chart.");
      }

      const points = rowObjects
        .map((row, index) => ({
          xLabel: String(row[xCol] ?? index + 1),
          y: toNumber(row[valueCol]),
        }))
        .filter((row) => row.y !== null)
        .slice(0, 30) as Array<{ xLabel: string; y: number }>;

      if (points.length < 2) {
        return renderFallback("Not enough points to draw a line chart.");
      }

      const width = 520;
      const height = 220;
      const pad = 28;
      const ys = points.map((point) => point.y);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const yRange = maxY - minY || 1;

      const linePoints = points
        .map((point, index) => {
          const x = pad + (index / (points.length - 1)) * (width - pad * 2);
          const y = height - pad - ((point.y - minY) / yRange) * (height - pad * 2);
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
          <CardContent>
            <div className="overflow-x-auto">
              <svg viewBox={`0 0 ${width} ${height}`} className="min-w-[320px] w-full h-[220px]">
                <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#94a3b8" />
                <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#94a3b8" />
                <polyline
                  fill="none"
                  stroke="#2563eb"
                  strokeWidth="2.5"
                  points={linePoints}
                />
                {points.map((point, index) => {
                  const x = pad + (index / (points.length - 1)) * (width - pad * 2);
                  const y = height - pad - ((point.y - minY) / yRange) * (height - pad * 2);
                  return <circle key={`${point.xLabel}-${index}`} cx={x} cy={y} r="3" fill="#2563eb" />;
                })}
              </svg>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              {xCol} vs {valueCol} ({points.length} points)
            </p>
          </CardContent>
        </Card>
      );
    };

    const renderScatter = () => {
      const [xCol, yCol] = numericColumns;
      if (!xCol || !yCol) {
        return renderFallback("Scatter plot needs at least two numeric columns.");
      }

      const points = rowObjects
        .map((row) => ({
          x: toNumber(row[xCol]),
          y: toNumber(row[yCol]),
        }))
        .filter((row) => row.x !== null && row.y !== null)
        .slice(0, 120) as Array<{ x: number; y: number }>;

      if (points.length < 2) {
        return renderFallback("Not enough numeric points for scatter plot.");
      }

      const width = 520;
      const height = 220;
      const pad = 28;
      const xs = points.map((point) => point.x);
      const ys = points.map((point) => point.y);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const xRange = maxX - minX || 1;
      const yRange = maxY - minY || 1;

      return (
        <Card className="mt-4">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <BarChart3 size={16} />
              Scatter Plot
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <svg viewBox={`0 0 ${width} ${height}`} className="min-w-[320px] w-full h-[220px]">
                <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#94a3b8" />
                <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#94a3b8" />
                {points.map((point, index) => {
                  const x = pad + ((point.x - minX) / xRange) * (width - pad * 2);
                  const y = height - pad - ((point.y - minY) / yRange) * (height - pad * 2);
                  return <circle key={`scatter-${index}`} cx={x} cy={y} r="3.2" fill="#2563eb" opacity="0.85" />;
                })}
              </svg>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              {xCol} vs {yCol} ({points.length} points)
            </p>
          </CardContent>
        </Card>
      );
    };

    const renderPie = () => {
      const valueCol = numericColumns[0];
      const labelCol = nonNumericColumns[0] || columnNames[0];
      if (!valueCol || !labelCol) {
        return renderFallback("This result shape is not suitable for a pie chart.");
      }
      const slices = rowObjects
        .map((row) => ({
          label: String(row[labelCol] ?? ""),
          value: toNumber(row[valueCol]),
        }))
        .filter((row) => row.value !== null && row.value > 0)
        .slice(0, 8) as Array<{ label: string; value: number }>;
      if (slices.length < 2) {
        return renderFallback("Not enough positive categories for a pie chart.");
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

      return (
        <Card className="mt-4">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center gap-2">
              <BarChart3 size={16} />
              Pie Chart
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-start gap-4">
              <div
                className="h-40 w-40 rounded-full border border-border"
                style={{ background: `conic-gradient(${stops.join(", ")})` }}
              />
              <ul className="flex-1 space-y-1 text-xs">
                {slices.map((slice, index) => (
                  <li key={`${slice.label}-${index}`} className="flex items-center gap-2">
                    <span
                      className="inline-block h-2.5 w-2.5 rounded-full"
                      style={{ backgroundColor: CHART_COLORS[index % CHART_COLORS.length] }}
                    />
                    <span className="flex-1 truncate">{slice.label}</span>
                    <span>{formatMetricNumber(slice.value)}</span>
                  </li>
                ))}
              </ul>
            </div>
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
    if (message.sql) {
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
  }, [hasAgentTimings, hasEvidence, hasSources, hasTable, message.sql]);

  useEffect(() => {
    if (!tabs.some((tab) => tab.id === activeTab)) {
      setActiveTab("answer");
    }
  }, [activeTab, tabs]);

  const assistantMeta = !isUser && (message.answer_source || message.tool_approval_required);
  const showActions = !isUser && (Boolean(message.sql) || hasTable);

  const renderAnswerOnly = () => (
    <>
      {renderMarkdownish(message.content)}
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
    <div className={cn("flex gap-3 mb-4", isUser ? "justify-end" : "justify-start")}>
      {!isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary flex items-center justify-center text-primary-foreground">
          <Bot size={18} />
        </div>
      )}

      <div className={cn("flex-1 max-w-3xl", isUser && "flex justify-end")}>
        <div
          className={cn(
            "rounded-lg px-4 py-3",
            isUser ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"
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
              {message.sql && (
                <button
                  type="button"
                  className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 hover:bg-secondary"
                  onClick={copySql}
                >
                  <Copy size={12} />
                  Copy SQL
                </button>
              )}
              {hasTable && (
                <button
                  type="button"
                  className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 hover:bg-secondary"
                  onClick={downloadCsv}
                >
                  <Download size={12} />
                  Download CSV
                </button>
              )}
              {actionNotice && <span className="text-muted-foreground">{actionNotice}</span>}
            </div>
          )}

          {!isUser && displayMode === "tabbed" ? (
            <>
              <div className="mb-2 flex flex-wrap gap-2 border-b border-border pb-2">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    className={cn(
                      "rounded px-2.5 py-1 text-xs transition",
                      activeTab === tab.id
                        ? "bg-primary text-primary-foreground"
                        : "bg-secondary text-foreground hover:bg-secondary/80"
                    )}
                    onClick={() => setActiveTab(tab.id)}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
              {renderTabContent()}
            </>
          ) : (
            <>
              {renderAnswerOnly()}
              {!isUser && message.sql && renderSqlSection()}
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
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-secondary flex items-center justify-center text-secondary-foreground">
          <User size={18} />
        </div>
      )}
    </div>
  );
}
