import { fireEvent, render, screen, within } from "@testing-library/react";

import { Message } from "@/components/chat/Message";

describe("Message", () => {
  it("renders simple markdown bullets and bold text for assistant messages", () => {
    render(
      <Message
        message={{
          id: "msg-1",
          role: "assistant",
          content:
            "Here are the grocery stores:\n\n* **Downtown Fresh** in Austin\n* **Midtown Market** in Austin",
          timestamp: new Date(),
        }}
      />
    );

    expect(screen.getByText("Here are the grocery stores:")).toBeInTheDocument();
    const list = screen.getByRole("list");
    const items = within(list).getAllByRole("listitem");
    expect(items).toHaveLength(2);
    expect(within(items[0]).getByText("Downtown Fresh", { selector: "strong" })).toBeInTheDocument();
    expect(within(items[1]).getByText("Midtown Market", { selector: "strong" })).toBeInTheDocument();
  });

  it("renders tabbed layout with visualization tab for assistant results", () => {
    render(
      <Message
        displayMode="tabbed"
        message={{
          id: "msg-2",
          role: "assistant",
          content: "Sales by region",
          sql: "SELECT region, total FROM sales_by_region",
          data: {
            region: ["South", "North", "East"],
            total: [120, 90, 45],
          },
          sources: [
            {
              datapoint_id: "dp_1",
              type: "Schema",
              name: "sales_by_region",
              relevance_score: 0.9,
            },
          ],
          visualization_hint: "bar_chart",
          timestamp: new Date(),
        }}
      />
    );

    expect(screen.getByRole("button", { name: "Answer" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "SQL" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Table" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Visualization" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Sources" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Visualization" }));
    expect(screen.getByText("Bar Chart")).toBeInTheDocument();
  });

  it("renders axis and legend metadata for line visualization", () => {
    render(
      <Message
        displayMode="tabbed"
        message={{
          id: "msg-2b",
          role: "assistant",
          content: "Revenue trend",
          data: {
            business_date: ["2026-01-01", "2026-01-02", "2026-01-03"],
            revenue: [100, 130, 120],
          },
          visualization_hint: "line_chart",
          timestamp: new Date(),
        }}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Visualization" }));
    expect(screen.getByText("Line Chart")).toBeInTheDocument();
    expect(screen.getByText(/X axis:/)).toBeInTheDocument();
    expect(screen.getByText(/Y axis:/)).toBeInTheDocument();
    expect(screen.getByText(/Legend:/)).toBeInTheDocument();
  });

  it("collapses results and evidence sections by default in stacked mode", () => {
    const { container } = render(
      <Message
        message={{
          id: "msg-3",
          role: "assistant",
          content: "Summary",
          sql: "SELECT name, total FROM revenue",
          data: {
            name: ["A", "B"],
            total: [10, 20],
          },
          sources: [
            {
              datapoint_id: "dp_2",
              type: "Business",
              name: "Revenue Metric",
              relevance_score: 0.95,
            },
          ],
          evidence: [
            {
              datapoint_id: "dp_2",
              type: "Business",
              name: "Revenue Metric",
              reason: "Used to answer the query",
            },
          ],
          timestamp: new Date(),
        }}
      />
    );

    const details = Array.from(container.querySelectorAll("details"));
    expect(details.length).toBeGreaterThanOrEqual(2);
    for (const section of details) {
      expect(section.open).toBe(false);
    }
  });

  it("can hide agent timing breakdown while keeping summary metrics", () => {
    render(
      <Message
        showAgentTimingBreakdown={false}
        message={{
          id: "msg-4",
          role: "assistant",
          content: "Done",
          metrics: {
            total_latency_ms: 1250,
            agent_timings: {
              classifier: 120,
              context: 330,
            },
            llm_calls: 1,
            retry_count: 0,
          },
          timestamp: new Date(),
        }}
      />
    );

    expect(screen.queryByText("Classifier")).not.toBeInTheDocument();
    expect(screen.getByText("1.25s")).toBeInTheDocument();
    expect(screen.getByText("LLM calls: 1")).toBeInTheDocument();
  });

  it("shows agent timing breakdown in a timing tab sorted longest to shortest", () => {
    render(
      <Message
        displayMode="tabbed"
        message={{
          id: "msg-5",
          role: "assistant",
          content: "Done",
          metrics: {
            total_latency_ms: 3250,
            agent_timings: {
              classifier: 1000,
              context: 2100,
              sql: 150,
            },
            llm_calls: 2,
            retry_count: 0,
          },
          timestamp: new Date(),
        }}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "Timing" }));

    const rows = screen.getAllByText(/s$/);
    expect(rows[0]).toHaveTextContent("2.1s");
    expect(rows[1]).toHaveTextContent("1s");
    expect(rows[2]).toHaveTextContent("0.15s");
  });

  it("supports toggling between multi-question sub-answers for SQL and table views", () => {
    render(
      <Message
        displayMode="tabbed"
        message={{
          id: "msg-6",
          role: "assistant",
          content: "I handled your request as multiple questions.",
          answer_source: "multi",
          sub_answers: [
            {
              index: 1,
              query: "Which suppliers have highest late-delivery rate?",
              answer: "Supplier A leads.",
              sql: "SELECT supplier_id, late_rate FROM supplier_late_rates LIMIT 10",
              data: {
                supplier_id: ["SUP1"],
                late_rate: [0.31],
              },
              visualization_hint: "bar_chart",
            },
            {
              index: 2,
              query: "What is the average delay in days?",
              answer: "Average delay is 2.4 days.",
              sql: "SELECT AVG(delay_days) AS avg_delay_days FROM supplier_delays",
              data: {
                avg_delay_days: [2.4],
              },
              visualization_hint: "none",
            },
          ],
          timestamp: new Date(),
        }}
      />
    );

    expect(screen.getByText("Sub-questions")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Q1" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Q2" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "SQL" }));
    expect(screen.getByText(/supplier_late_rates/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Q2" }));
    expect(screen.getByText(/average delay in days/i)).toBeInTheDocument();
    expect(screen.getByText(/avg_delay_days/)).toBeInTheDocument();
  });
});
