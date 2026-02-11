import { describe, expect, it } from "vitest";

import {
  WAITING_LABEL_THRESHOLD_SECONDS,
  formatWaitingChipLabel,
} from "@/components/chat/loadingUx";

describe("formatWaitingChipLabel", () => {
  it("shows only elapsed seconds before threshold", () => {
    expect(formatWaitingChipLabel(0)).toBe("0s");
    expect(formatWaitingChipLabel(3.8)).toBe("3s");
    expect(formatWaitingChipLabel(WAITING_LABEL_THRESHOLD_SECONDS - 1)).toBe(
      "9s"
    );
  });

  it("shows still waiting message at threshold and after", () => {
    expect(formatWaitingChipLabel(WAITING_LABEL_THRESHOLD_SECONDS)).toBe(
      "Still waiting (10s)"
    );
    expect(formatWaitingChipLabel(17)).toBe("Still waiting (17s)");
  });
});

