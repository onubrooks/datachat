/**
 * Message Component
 *
 * Displays a single chat message with support for:
 * - User and assistant messages
 * - SQL code blocks
 * - Data tables
 * - Source citations
 * - Performance metrics
 */

"use client";

import React from "react";
import {
  User,
  Bot,
  Code,
  Table as TableIcon,
  BookOpen,
  Clock,
  BadgeCheck,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { cn } from "@/lib/utils";
import type { Message as MessageType } from "@/lib/stores/chat";

interface MessageProps {
  message: MessageType;
}

export function Message({ message }: MessageProps) {
  const isUser = message.role === "user";
  const columnNames = message.data ? Object.keys(message.data) : [];
  const rowCount =
    columnNames.length > 0
      ? Math.max(
          ...columnNames.map((column) => message.data?.[column]?.length ?? 0)
        )
      : 0;
  const rows = Array.from({ length: rowCount }, (_, rowIndex) =>
    columnNames.map((column) => message.data?.[column]?.[rowIndex])
  );
  const formatCellValue = (value: unknown) => {
    if (value === null || value === undefined) {
      return { display: "", full: "", truncated: false };
    }
    const full =
      typeof value === "string" ? value : JSON.stringify(value);
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

  return (
    <div
      className={cn(
        "flex gap-3 mb-4",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      {/* Avatar */}
      {!isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary flex items-center justify-center text-primary-foreground">
          <Bot size={18} />
        </div>
      )}

      {/* Message Content */}
      <div className={cn("flex-1 max-w-3xl", isUser && "flex justify-end")}>
        <div
          className={cn(
            "rounded-lg px-4 py-3",
            isUser
              ? "bg-primary text-primary-foreground"
              : "bg-muted text-muted-foreground"
          )}
        >
          {!isUser && (message.answer_source || message.tool_approval_required) && (
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
          {/* Main message text */}
          <div className="whitespace-pre-wrap">{message.content}</div>

          {/* SQL Query */}
          {message.sql && (
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
          )}

          {/* Data Table */}
          {message.data && rowCount > 0 && (
            <Card className="mt-4">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <TableIcon size={16} />
                  Results ({rowCount} rows)
                </CardTitle>
              </CardHeader>
              <CardContent>
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
                            const { display, full, truncated } =
                              formatCellValue(value);
                            return (
                              <td key={vidx} className="p-2 align-top">
                                <span
                                  className={
                                    truncated
                                      ? "block max-w-[320px] truncate"
                                      : "block"
                                  }
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
            </Card>
          )}

          {/* Sources */}
          {message.sources &&
            message.sources.length > 0 &&
            message.answer_source !== "context" && (
              <Card className="mt-4">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <BookOpen size={16} />
                    Sources
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <details className="text-sm">
                    <summary className="cursor-pointer text-xs text-muted-foreground">
                      Show sources ({message.sources.length})
                    </summary>
                    <ul className="mt-2 space-y-2">
                      {message.sources.map((source) => (
                        <li
                          key={source.datapoint_id}
                          className="text-sm flex items-start gap-2"
                        >
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
                  </details>
                </CardContent>
              </Card>
            )}

          {message.evidence && message.evidence.length > 0 && (
            <Card className="mt-4">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <BookOpen size={16} />
                  Evidence ({message.evidence.length})
                </CardTitle>
              </CardHeader>
              <CardContent>
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
                    <div className="text-xs font-medium text-muted-foreground">
                      Raw SQL
                    </div>
                    <pre className="mt-1 rounded bg-secondary p-2 text-xs overflow-x-auto">
                      <code>{message.sql}</code>
                    </pre>
                  </div>
                )}
                <details className="text-sm">
                  <summary className="cursor-pointer text-xs text-muted-foreground">
                    Show evidence details
                  </summary>
                  <ul className="mt-2 space-y-2">
                    {message.evidence.map((item) => (
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
                </details>
              </CardContent>
            </Card>
          )}

          {/* Metrics */}
          {message.metrics && (
            <div className="mt-3 flex items-center gap-4 text-xs text-muted-foreground">
              <div className="flex items-center gap-1">
                <Clock size={12} />
                {message.metrics.total_latency_ms}ms
              </div>
              {message.metrics.llm_calls > 0 && (
                <div>LLM calls: {message.metrics.llm_calls}</div>
              )}
              {message.metrics.retry_count > 0 && (
                <div>Retries: {message.metrics.retry_count}</div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* User Avatar */}
      {isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-secondary flex items-center justify-center text-secondary-foreground">
          <User size={18} />
        </div>
      )}
    </div>
  );
}
