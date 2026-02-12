export type WaitingUxMode = "basic" | "animated" | "progress";
export type ResultLayoutMode = "stacked" | "tabbed";

const WAITING_UX_KEY = "datachat.waitingUxMode";
const RESULT_LAYOUT_KEY = "datachat.resultLayoutMode";
const SHOW_AGENT_TIMINGS_KEY = "datachat.showAgentTimingBreakdown";
const SYNTHESIZE_SIMPLE_SQL_KEY = "datachat.synthesizeSimpleSql";

export const getWaitingUxMode = (): WaitingUxMode => {
  if (typeof window === "undefined") {
    return "animated";
  }
  const value = window.localStorage.getItem(WAITING_UX_KEY);
  if (value === "basic" || value === "progress" || value === "animated") {
    return value;
  }
  return "animated";
};

export const setWaitingUxMode = (mode: WaitingUxMode) => {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(WAITING_UX_KEY, mode);
};

export const getResultLayoutMode = (): ResultLayoutMode => {
  if (typeof window === "undefined") {
    return "stacked";
  }
  const value = window.localStorage.getItem(RESULT_LAYOUT_KEY);
  if (value === "stacked" || value === "tabbed") {
    return value;
  }
  return "stacked";
};

export const setResultLayoutMode = (mode: ResultLayoutMode) => {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(RESULT_LAYOUT_KEY, mode);
};

export const getShowAgentTimingBreakdown = (): boolean => {
  if (typeof window === "undefined") {
    return true;
  }
  const value = window.localStorage.getItem(SHOW_AGENT_TIMINGS_KEY);
  if (value === null) {
    return true;
  }
  return value === "true";
};

export const setShowAgentTimingBreakdown = (enabled: boolean) => {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(SHOW_AGENT_TIMINGS_KEY, String(enabled));
};

export const getSynthesizeSimpleSql = (): boolean => {
  if (typeof window === "undefined") {
    return true;
  }
  const value = window.localStorage.getItem(SYNTHESIZE_SIMPLE_SQL_KEY);
  if (value === null) {
    return true;
  }
  return value === "true";
};

export const setSynthesizeSimpleSql = (enabled: boolean) => {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(SYNTHESIZE_SIMPLE_SQL_KEY, String(enabled));
};
