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
});
