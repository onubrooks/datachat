import { render, screen, within } from "@testing-library/react";

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
});
